"""Cross-validated evaluation of model updates on the pooled, cleaned RFMiD 2.0 set.

- Patient-safe folds: images within perceptual-hash distance <= 5 are linked
  (union-find) into groups; a group never spans folds.
- Multi-label stratification at group level (iterative stratification,
  Sechidis et al., ECML-PKDD 2011, adapted to groups).
- Per fold k: test = fold k, validation = fold k+1, training = the other three.
  Every image is tested exactly once, so metrics are computed on the pooled
  out-of-fold predictions of all images.
- Operating thresholds: plain per-class F1 on the validation fold, or
  congruence-constrained (CT): the F1-best threshold among those that keep
  every validation positive the legacy model detected.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from .audit import _POP8, _to_u64


# ===========================================================================
# Folds
# ===========================================================================
def near_duplicate_groups(phash_hex, thr: int = 5) -> np.ndarray:
    """Connected components of the 'Hamming distance <= thr' graph."""
    h = _to_u64(phash_hex)
    n = len(h)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for s in range(0, n, 256):
        x = np.bitwise_xor(h[s:s + 256, None], h[None, :])
        d = _POP8[x.view(np.uint8)].reshape(x.shape + (8,)).sum(-1)
        for a, b in zip(*np.where(d <= thr)):
            i, j = find(s + a), find(b)
            if i != j:
                parent[max(i, j)] = min(i, j)
    roots = np.array([find(i) for i in range(n)])
    _, groups = np.unique(roots, return_inverse=True)
    return groups


def group_stratified_folds(Y: np.ndarray, groups: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """Fold index per image; groups kept whole, label counts balanced across folds."""
    rng = np.random.default_rng(seed)
    G = groups.max() + 1
    gY = np.zeros((G, Y.shape[1]))
    np.add.at(gY, groups, Y)
    gsize = np.bincount(groups, minlength=G).astype(float)

    desired_lab = np.tile(gY.sum(0) / k, (k, 1))          # k x C
    desired_size = np.full(k, gsize.sum() / k)
    fold_of_group = np.full(G, -1)
    remaining = set(range(G))

    while remaining:
        rem = np.array(sorted(remaining))
        lab_tot = gY[rem].sum(0)
        live = np.where(lab_tot > 0)[0]
        if len(live) == 0:                                  # groups with no positive labels
            for g in rng.permutation(rem):
                f = int(np.argmax(desired_size))
                fold_of_group[g] = f
                desired_size[f] -= gsize[g]
            break
        c = live[np.argmin(lab_tot[live])]                  # rarest remaining label first
        cand = rem[gY[rem, c] > 0]
        for g in cand[np.argsort(-gY[cand, c], kind="stable")]:
            score = desired_lab[:, c]
            best = np.flatnonzero(score == score.max())
            if len(best) > 1:
                ds = desired_size[best]
                best = best[ds == ds.max()]
            f = int(rng.choice(best))
            fold_of_group[g] = f
            desired_lab[f] -= gY[g]
            desired_size[f] -= gsize[g]
            remaining.discard(g)
    return fold_of_group[groups]


# ===========================================================================
# Thresholds
# ===========================================================================
def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def best_f1_threshold(y, p, cap=None, default=0.5):
    """F1-optimal threshold on (y, p); only thresholds <= cap are allowed."""
    if y.sum() == 0 or y.sum() == len(y):
        return default if cap is None else min(default, cap)
    cand = np.unique(p)
    if cap is not None:
        cand = cand[cand <= cap]
        if len(cand) == 0:
            return cap
    order = np.argsort(-p)
    ps, ys = p[order], y[order]
    tp_cum = np.cumsum(ys)
    best, bt = -1.0, cand[-1]
    for th in cand:
        npred = np.searchsorted(-ps, -th, side="right")      # count of p >= th
        tp = tp_cum[npred - 1] if npred > 0 else 0
        f1 = 2 * tp / (npred + ys.sum())
        if f1 > best:
            best, bt = f1, th
    return float(bt)


def plain_thresholds(l_val, Y, M, fallback):
    p = _sigmoid(l_val)
    t = np.array(fallback, dtype=float).copy()
    for c in range(Y.shape[1]):
        k = M[:, c] > 0
        y = Y[k, c]
        if 0 < y.sum() < len(y):
            t[c] = best_f1_threshold(y, p[k, c])
    return t


def constrained_thresholds(l_val, Y, M, l_old_val, thr_old, shared_idx, fallback, eps=0.0):
    """CT: per shared class, the F1-best threshold that keeps >= (1-eps) of the
    validation positives the legacy model detected at its deployed threshold."""
    t = plain_thresholds(l_val, Y, M, fallback)
    p, q = _sigmoid(l_val), _sigmoid(l_old_val)
    for c in shared_idx:
        k = M[:, c] > 0
        caught = k & (Y[:, c] == 1) & (q[:, c] >= thr_old[c])
        n = int(caught.sum())
        if n == 0:
            continue
        need = int(np.ceil((1 - eps) * n))
        cap = np.sort(p[caught, c])[::-1][need - 1]
        y = Y[k, c]
        t[c] = best_f1_threshold(y, p[k, c], cap=cap) if 0 < y.sum() < len(y) else min(t[c], cap)
    return t


# ===========================================================================
# Pooled metrics
# ===========================================================================
def class_ap(L, Y, M, cols, idx=None):
    out = {}
    for c in cols:
        k = M[:, c] > 0
        if idx is not None:
            sel = idx[k[idx]]
            y, s = Y[sel, c], L[sel, c]
        else:
            y, s = Y[k, c], L[k, c]
        out[c] = float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else np.nan
    return out


def flip_rates(P_new, P_old, Y, M, shared_idx, idx=None):
    """NFR, NFR+ (missed-disease flips among positives), PFR in %, on (image, shared class) pairs."""
    sh = shared_idx
    if idx is None:
        idx = np.arange(len(Y))
    known = M[idx][:, sh] > 0
    y = Y[idx][:, sh] == 1
    new_ok = P_new[idx][:, sh] == y
    old_ok = P_old[idx][:, sh] == y
    nf = old_ok & ~new_ok & known
    pf = ~old_ok & new_ok & known
    return (100 * nf.sum() / max(1, known.sum()),
            100 * (nf & y).sum() / max(1, (known & y).sum()),
            100 * pf.sum() / max(1, known.sum()))


def pooled_metrics(L, THR, L_old, thr_old, Y, M, groups_cols, shared_idx, min_pos=5, regress_delta=0.05):
    """All metrics from pooled out-of-fold logits L (N x K) and per-row thresholds THR (N x K)."""
    P_new = _sigmoid(L) >= THR
    P_old = _sigmoid(L_old) >= thr_old[None, :]
    P_scoreonly = _sigmoid(L) >= thr_old[None, :]
    res = {}
    for g, cols in groups_cols.items():
        ap = class_ap(L, Y, M, cols)
        vals = [v for v in ap.values() if not np.isnan(v)]
        res[f"mAP_{g}"] = float(np.mean(vals)) if vals else np.nan
    npos = (Y * M).sum(0)
    ap_all = class_ap(L, Y, M, groups_cols["all"])
    big = [c for c in groups_cols["all"] if npos[c] >= min_pos and not np.isnan(ap_all[c])]
    res[f"mAP_all_min{min_pos}"] = float(np.mean([ap_all[c] for c in big])) if big else np.nan
    res["NFR_%"], res["NFR+_%"], res["PFR_%"] = flip_rates(P_new, P_old, Y, M, shared_idx)
    _, res["NFR+_scoreonly_%"], _ = flip_rates(P_scoreonly, P_old, Y, M, shared_idx)
    ap_new = class_ap(L, Y, M, shared_idx)
    ap_old = class_ap(L_old, Y, M, shared_idx)
    sh_big = [c for c in shared_idx if npos[c] >= min_pos and not np.isnan(ap_old[c])]
    res[f"regressed_classes_min{min_pos}"] = int(sum(ap_new[c] < ap_old[c] - regress_delta for c in sh_big))
    res[f"n_shared_classes_min{min_pos}"] = len(sh_big)
    res["per_class_ap"] = dict(ap_all)
    return res


# ===========================================================================
# Paired bootstrap over images (same resamples for both methods and all seeds)
# ===========================================================================
def paired_bootstrap(runs_a, runs_b, Y, M, cols, shared_idx, L_olds, thr_olds, n_boot=300, seed=0):
    """Paired image-level bootstrap of the seed-averaged difference A - B.

    runs_a, runs_b: lists over seeds of (L, THR) pooled out-of-fold logits / per-row thresholds;
    L_olds, thr_olds: the legacy logits and deployed thresholds of each seed.
    Every replicate uses the same resampled images for both methods and all seeds.
    Returns {"d_mAP": [mean, 2.5%, 97.5%], "d_NFR+": [...]} (mAP over `cols`, NFR+ in %).
    """
    rng = np.random.default_rng(seed)
    n = len(Y)
    P_olds = [_sigmoid(Lo) >= to[None, :] for Lo, to in zip(L_olds, thr_olds)]
    Pa = [_sigmoid(L) >= T for L, T in runs_a]
    Pb = [_sigmoid(L) >= T for L, T in runs_b]

    def stats(idx):
        dm, dn = [], []
        for (La, _), (Lb, _), pa, pb, po in zip(runs_a, runs_b, Pa, Pb, P_olds):
            ap_a = class_ap(La, Y, M, cols, idx)
            ap_b = class_ap(Lb, Y, M, cols, idx)
            both = [c for c in cols if not (np.isnan(ap_a[c]) or np.isnan(ap_b[c]))]
            dm.append(np.mean([ap_a[c] - ap_b[c] for c in both]) if both else np.nan)
            dn.append(flip_rates(pa, po, Y, M, shared_idx, idx)[1] - flip_rates(pb, po, Y, M, shared_idx, idx)[1])
        return np.nanmean(dm), np.mean(dn)

    point = stats(np.arange(n))
    reps = np.array([stats(rng.integers(0, n, n)) for _ in range(n_boot)])
    out = {}
    for j, key in enumerate(("d_mAP", "d_NFR+")):
        lo, hi = np.nanpercentile(reps[:, j], [2.5, 97.5])
        out[key] = [round(float(point[j]), 4), round(float(lo), 4), round(float(hi), 4)]
    return out
