"""Overlap audit and preprocessing cache.

One pass per image computes:
  - SHA-256 of the file bytes   (byte-identical duplicates)
  - 64-bit perceptual hash      (near duplicates across versions / splits)
  - a cached 256x256 JPEG: fundus cropped to its bounding box, padded square

Policies (fixed before looking at results, as in RD1):
  - within RFMiD 2.0: byte-identical copies keep the copy in the earliest
    split (train < val < test), lowest ID; the others are removed;
  - cross-version: an RFMiD 2.0 image within Hamming distance <= 5 of any
    RFMiD 1.0 TRAINING image is removed;
  - within RFMiD 2.0 across splits (optional): a val/test image within
    Hamming <= 5 of a 2.0 training image is removed from val/test.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}
_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def fundus_crop(im: Image.Image, thr: int = 15) -> Image.Image:
    """Crop to the bounding box of the illuminated fundus (removes black borders)."""
    g = np.asarray(im.convert("L"))
    mask = g > thr
    rows = np.where(mask.mean(1) > 0.01)[0]
    cols = np.where(mask.mean(0) > 0.01)[0]
    if len(rows) == 0 or len(cols) == 0:
        return im
    return im.crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))


def pad_square(im: Image.Image) -> Image.Image:
    w, h = im.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), (0, 0, 0))
    canvas.paste(im, ((s - w) // 2, (s - h) // 2))
    return canvas


def fingerprint_and_cache(man: pd.DataFrame, cache_dir: Path, size: int = 256,
                          fp_file: Path | None = None) -> pd.DataFrame:
    """SHA-256 + pHash for every image, and write the preprocessed cache.

    Results are stored in fp_file; a rerun reuses them and only processes new rows.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(fp_file, dtype={"ID": str}) if fp_file and fp_file.exists() else None
    done_keys = set(zip(done["split"], done["ID"])) if done is not None else set()

    rows = []
    todo = [r for r in man.itertuples(index=False) if (r.split, r.ID) not in done_keys]
    for k, r in enumerate(todo):
        data = Path(r.path).read_bytes()
        out = cache_dir / f"{r.split}_{r.ID}.jpg"
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB")
            ph = int(str(imagehash.phash(im)), 16)
            if not out.exists():
                pad_square(fundus_crop(im)).resize((size, size), Image.LANCZOS).save(out, quality=95)
        rows.append({"split": r.split, "ID": r.ID, "sha256": hashlib.sha256(data).hexdigest(),
                     "phash": f"{ph:016x}", "cache": str(out)})
        if (k + 1) % 200 == 0:
            print(f"    {k + 1}/{len(todo)}")
    new = pd.DataFrame(rows)
    fp = pd.concat([done, new], ignore_index=True) if done is not None else new
    if fp_file:
        fp.to_csv(fp_file, index=False)
    keys = man[["split", "ID"]]
    return keys.merge(fp, on=["split", "ID"], how="left")


# ---------------------------------------------------------------------------
# Hamming distances between 64-bit hashes
# ---------------------------------------------------------------------------
def _to_u64(hex_list) -> np.ndarray:
    return np.array([int(h, 16) for h in hex_list], dtype=np.uint64)


def hamming_min(a_hex, b_hex) -> tuple[np.ndarray, np.ndarray]:
    """For each hash in a: (min Hamming distance to b, index of the closest b)."""
    a, b = _to_u64(a_hex), _to_u64(b_hex)
    dmin = np.full(len(a), 64, dtype=np.int64)
    arg = np.zeros(len(a), dtype=np.int64)
    for s in range(0, len(a), 256):
        x = np.bitwise_xor(a[s:s + 256, None], b[None, :])
        d = _POP8[x.view(np.uint8)].reshape(x.shape + (8,)).sum(-1)
        dmin[s:s + 256] = d.min(1)
        arg[s:s + 256] = d.argmin(1)
    return dmin, arg


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
def within_version_duplicates(fp: pd.DataFrame) -> pd.DataFrame:
    """Byte-identical files: keep earliest split, lowest ID; return rows to drop."""
    f = fp.copy()
    f["_o"] = f["split"].map(SPLIT_ORDER)
    f["_n"] = pd.to_numeric(f["ID"], errors="coerce")
    f = f.sort_values(["_o", "_n", "ID"])
    drop = f[f.duplicated("sha256", keep="first")].copy()
    keep = f.drop_duplicates("sha256", keep="first").set_index("sha256")
    drop["duplicate_of"] = drop["sha256"].map(lambda s: f"{keep.loc[s, 'split']}_{keep.loc[s, 'ID']}")
    return drop[["split", "ID", "duplicate_of"]]


def run_audit(fp1: pd.DataFrame, fp2: pd.DataFrame, cross_thr: int = 5,
              remove_v2_internal: bool = True) -> tuple[pd.DataFrame, dict]:
    """Return (rows of RFMiD 2.0 to drop with reason, audit summary)."""
    drops = []

    dup = within_version_duplicates(fp2)
    drops.append(dup.assign(reason="byte-identical duplicate within 2.0"))
    fp2 = fp2[~fp2.set_index(["split", "ID"]).index.isin(dup.set_index(["split", "ID"]).index)]

    v1_train = fp1[fp1["split"] == "train"]
    dmin, arg = hamming_min(fp2["phash"], v1_train["phash"])
    summary = {"within_2.0_byte_duplicates": len(dup)}
    for t in (0, 3, 5, 8):
        sel = fp2[dmin <= t]
        summary[f"cross_version_H<={t}"] = {s: int((sel["split"] == s).sum()) for s in SPLIT_ORDER}
    cross = fp2[dmin <= cross_thr].copy()
    cross["duplicate_of"] = ["rfmid1_train_" + v1_train["ID"].iat[i] for i in arg[dmin <= cross_thr]]
    drops.append(cross[["split", "ID", "duplicate_of"]].assign(reason=f"near-duplicate of RFMiD 1.0 train (H<={cross_thr})"))
    fp2 = fp2[dmin > cross_thr]

    tr = fp2[fp2["split"] == "train"]
    ev = fp2[fp2["split"] != "train"]
    d_int, a_int = hamming_min(ev["phash"], tr["phash"])
    internal = ev[d_int <= cross_thr].copy()
    internal["duplicate_of"] = ["rfmid2_train_" + tr["ID"].iat[i] for i in a_int[d_int <= cross_thr]]
    summary["within_2.0_train_vs_eval_H<=5"] = {s: int((internal["split"] == s).sum()) for s in ("val", "test")}
    summary["within_2.0_internal_removed"] = bool(remove_v2_internal)
    if remove_v2_internal:
        drops.append(internal[["split", "ID", "duplicate_of"]].assign(reason="near-duplicate of a 2.0 training image"))

    drop_df = pd.concat(drops, ignore_index=True)
    return drop_df, summary
