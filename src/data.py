"""Reading RFMiD 1.0 and RFMiD 2.0: label tables, image files, manifests.

The loader is deliberately defensive, because the released tables have known
quirks (header whitespace, empty trailing columns, IDs stored as floats,
annotation rows without images and images without annotation rows).
Nothing is silently dropped: every removal is counted in the split report.
"""
from __future__ import annotations

import difflib
from pathlib import Path

import pandas as pd
from PIL import Image

from . import config


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def normalize_id(value) -> str:
    """'12', 12, 12.0, '12.png', ' 12 ' -> '12'."""
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    s = str(value).strip()
    stem = Path(s).stem if Path(s).suffix.lower() in config.IMAGE_EXTENSIONS else s
    if stem.endswith(".0") and stem[:-2].isdigit():
        stem = stem[:-2]
    return stem


def normalize_label_name(name: str) -> str:
    """Canonical class name used to match label spaces across versions."""
    return str(name).strip().upper()


# ---------------------------------------------------------------------------
# Label files
# ---------------------------------------------------------------------------
def resolve_label_file(path: Path) -> Path:
    """Find the label table for a configured path.

    Accepts the exact file, the name without extension, a folder holding a
    single table, or a close spelling match in the same folder.
    """
    path = Path(path)
    if path.is_file():
        return path
    for ext in config.LABEL_EXTENSIONS:
        cand = path.with_name(path.name + ext)
        if cand.is_file():
            return cand
    if path.is_dir():
        tables = [p for p in path.iterdir() if p.suffix.lower() in config.LABEL_EXTENSIONS]
        if len(tables) == 1:
            return tables[0]
        raise FileNotFoundError(f"{path} is a folder with {len(tables)} label tables; expected 1: {tables}")

    parent = path.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"Label folder does not exist: {parent}")
    tables = [p for p in parent.iterdir() if p.suffix.lower() in config.LABEL_EXTENSIONS]
    best = difflib.get_close_matches(path.name.lower(), [p.stem.lower() for p in tables], n=1, cutoff=0.8)
    if best:
        match = next(p for p in tables if p.stem.lower() == best[0])
        print(f"  note: '{path.name}' not found; using closest match '{match.name}'")
        return match
    raise FileNotFoundError(
        f"No label table for '{path.name}' in {parent}. Tables present: {[p.name for p in tables]}"
    )


CSV_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def read_csv_any_encoding(path: Path) -> tuple[pd.DataFrame, str]:
    """Read a CSV saved as UTF-8 or as a Windows/Excel export.

    Some RFMiD tables were saved by Excel in cp1252, where byte 0xA0 is a
    non-breaking space; pandas' default UTF-8 decoding fails on it.
    utf-8-sig is tried first (also handles an Excel byte-order mark);
    latin-1 decodes any byte, so it is the last resort.
    """
    for enc in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc), enc
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"{path.name}: none of {CSV_ENCODINGS} worked")


def clean_header(name) -> str:
    """Strip, turn non-breaking spaces into spaces, collapse repeated spaces."""
    return " ".join(str(name).replace("\xa0", " ").split())


def read_label_table(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read and clean one label table. Returns (table, repair notes)."""
    path = resolve_label_file(path)
    df, encoding = read_csv_any_encoding(path)

    notes = {"file": path.name, "encoding": encoding}
    original_cols = list(df.columns)
    df.columns = [clean_header(c) for c in df.columns]
    notes["headers_stripped"] = sum(a != b for a, b in zip(map(str, original_cols), df.columns))

    empty_cols = [c for c in df.columns if c.startswith("Unnamed") or df[c].isna().all()]
    df = df.drop(columns=empty_cols)
    notes["empty_columns_dropped"] = empty_cols

    n_before = len(df)
    df = df.dropna(how="all")
    notes["empty_rows_dropped"] = n_before - len(df)

    id_col = next((c for c in df.columns if c.strip().lower() in {"id", "image id", "image_id", "imageid"}), df.columns[0])
    df = df.rename(columns={id_col: "ID"})
    df["ID"] = df["ID"].map(normalize_id)
    return df.reset_index(drop=True), notes


def split_label_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(disease label columns, summary-flag columns) in table order."""
    cols = [c for c in df.columns if c != "ID"]
    flags = [c for c in cols if normalize_label_name(c) in config.NON_DISEASE_COLUMNS]
    diseases = [c for c in cols if c not in flags]
    return diseases, flags


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def index_images(folder: Path) -> tuple[dict[str, Path], list[str]]:
    """Map image ID (file stem) -> path, searching subfolders.

    Returns (index, duplicated IDs). A duplicated ID means two files share a
    stem (e.g. 5.png and 5.jpg); the first one found is kept.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")
    index, dups = {}, []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS:
            key = normalize_id(p.stem)
            if key in index:
                dups.append(key)
                continue
            index[key] = p
    return index, dups


def image_size(path: Path) -> tuple[int, int]:
    """(width, height) from the file header, without decoding pixels."""
    with Image.open(path) as im:
        return im.size


# ---------------------------------------------------------------------------
# One split -> manifest
# ---------------------------------------------------------------------------
def load_split(ds: dict, split: str) -> tuple[pd.DataFrame, dict]:
    """Join a split's labels and images into a manifest.

    Policy (conservative): keep only records with BOTH an image file and an
    annotation row; duplicated annotation IDs keep the first row.
    """
    labels, notes = read_label_table(ds["labels"][split])
    images, dup_files = index_images(ds["images"][split])
    diseases, flags = split_label_columns(labels)

    dup_mask = labels["ID"].duplicated(keep="first")
    dup_ids = sorted(labels.loc[dup_mask, "ID"].unique().tolist())
    labels = labels.loc[~dup_mask]

    label_ids, image_ids = set(labels["ID"]), set(images)
    no_image = sorted(label_ids - image_ids, key=_sort_key)
    no_label = sorted(image_ids - label_ids, key=_sort_key)

    man = labels[labels["ID"].isin(image_ids)].copy()
    lab_cols = diseases + flags
    # Values typed as text (e.g. "1 " with a stray space) -> numbers.
    n_unparsable = 0
    for c in lab_cols:
        if not pd.api.types.is_numeric_dtype(man[c]):   # text column (pandas 2: object, pandas 3: str)
            parsed = pd.to_numeric(man[c].map(lambda v: clean_header(v) if pd.notna(v) else v), errors="coerce")
            n_unparsable += int(parsed.isna().sum() - man[c].isna().sum())
            man[c] = parsed
    n_missing_values = int(man[lab_cols].isna().sum().sum())
    man[lab_cols] = man[lab_cols].fillna(0)
    non_binary = [c for c in lab_cols if not set(pd.unique(man[c])).issubset({0, 1})]
    man[lab_cols] = man[lab_cols].astype(int)

    man.insert(1, "path", man["ID"].map(lambda i: str(images[i])))
    man.insert(1, "split", split)
    man.insert(1, "dataset", ds["name"])

    report = {
        "dataset": ds["name"], "split": split, **notes,
        "label_rows": len(labels) + len(dup_ids), "image_files": len(images) + len(dup_files),
        "duplicate_label_ids": dup_ids, "duplicate_image_stems": sorted(set(dup_files)),
        "labels_without_image": no_image, "images_without_label": no_label,
        "missing_label_values_filled_0": n_missing_values, "unparsable_label_values": n_unparsable, "non_binary_columns": non_binary,
        "kept": len(man), "n_disease_classes": len(diseases), "flag_columns": flags,
    }
    return man.reset_index(drop=True), report


def _sort_key(x: str):
    return (0, int(x), "") if x.isdigit() else (1, 0, x)


def load_dataset(key: str) -> tuple[dict[str, pd.DataFrame], list[dict], list[str]]:
    """All splits of one dataset: (manifests by split, reports, disease classes)."""
    ds = config.DATASETS[key]
    manifests, reports, classes = {}, [], None
    for split in config.SPLITS:
        man, rep = load_split(ds, split)
        manifests[split] = man
        reports.append(rep)
        split_classes, _ = split_label_columns(man.drop(columns=["dataset", "split", "path"]))
        if classes is None:
            classes = split_classes
        elif split_classes != classes:
            raise ValueError(f"{ds['name']} {split}: class columns differ from the first split.\n"
                             f"only here: {set(split_classes) - set(classes)}; "
                             f"missing here: {set(classes) - set(split_classes)}")
    return manifests, reports, classes


# ---------------------------------------------------------------------------
# Label spaces across versions
# ---------------------------------------------------------------------------
def compare_label_spaces(old: list[str], new: list[str]) -> dict:
    """Match classes across releases by name, after two corrections from config:

    - LABEL_ALIASES: the same disease spelt differently is matched (OPDM -> ODPM);
    - LABEL_COLLISIONS: the same abbreviation for different diseases is NOT
      matched (MCA), and the class counts as retired in 1.0 and new in 2.0.
    """
    alias = {normalize_label_name(k): normalize_label_name(v) for k, v in config.LABEL_ALIASES.items()}
    collide = {normalize_label_name(c) for c in config.LABEL_COLLISIONS}
    o = {normalize_label_name(c): c for c in old}
    n = {alias.get(normalize_label_name(c), normalize_label_name(c)): c for c in new}
    shared_keys = [k for k in o if k in n and k not in collide]
    return {
        "shared": [o[k] for k in shared_keys],
        "shared_map": {o[k]: n[k] for k in shared_keys},   # RFMiD 1.0 name -> RFMiD 2.0 column
        "retired": [o[k] for k in o if k not in shared_keys],
        "new": [n[k] for k in n if k not in shared_keys],
        "aliases_applied": {n[k]: o[k] for k in shared_keys if n[k] != o[k]},
        "collisions_split": sorted(o[k] for k in o if k in collide and k in n),
    }


def prevalence(man: pd.DataFrame, classes: list[str]) -> pd.Series:
    return man[classes].sum().astype(int)
