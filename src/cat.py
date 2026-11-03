"""CAT baseline: class-distribution-aware thresholds (Xie et al., TPAMI 2026; NeurIPS 2023).

CAP's "CAT for label decision" labels as positive, per class k, the top gamma_k share of the
instances ranked by predicted score, where gamma_k is the class proportion estimated on the
labelled data. It needs no retraining, uses one threshold per class and gives no guarantee.

Adaptation to the CURE protocol (Section III-F of the paper):
- gamma_k is estimated on the labelled RFMiD 2.0 training folds of run k (the labelled set of
  the update). Only images whose label of class k is known count.
- Inductive variant (primary, "CAT"): the threshold of class k is placed so that a share gamma_k
  of the validation fold V is predicted positive. V is the set every other baseline uses to set
  thresholds, and the test fold is never looked at.
- Transductive variant ("CAT-T", sensitivity analysis, closest to the original paper): the same
  rule applied to the scores of the test fold itself. It reads test scores but not test labels.
- Classes: shared and new classes get CAT thresholds. Discarded classes have no RFMiD 2.0 labels,
  so no proportion can be estimated, and they keep the fallback (the legacy threshold), exactly
  as for every other method in the paper.
- A class with no positive (or no negative) training label has an undefined proportion for this
  rule. It keeps the method's own F1-maximizing threshold on V and is reported in the log.
- The threshold is set in logit space halfway between the scores ranked k and k + 1, so ties do
  not change the predicted share and the rule is the same for every update method.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def class_prevalence(Y: np.ndarray, M: np.ndarray, cols) -> dict:
    """gamma_c = positives / known labels of class c, over the rows given (the training folds)."""
    out = {}
    for c in cols:
        k = M[:, c] > 0
        n = int(k.sum())
        out[c] = (float(Y[k, c].sum()) / n) if n else float("nan")
    return out


def _quantile_threshold(scores: np.ndarray, gamma: float) -> float:
    """Probability threshold that predicts the top round(gamma * n) of `scores` (logits) positive."""
    s = np.sort(np.asarray(scores, dtype=np.float64))[::-1]      # descending logits
    n = len(s)
    k = int(np.clip(np.round(gamma * n), 0, n))
    if k == 0:
        mid = s[0] + 1.0                                         # above every score: none predicted
    elif k == n:
        mid = s[-1] - 1.0                                        # below every score: all predicted
    else:
        mid = 0.5 * (s[k - 1] + s[k])
    return float(_sigmoid(mid))


def cat_thresholds(l_ref: np.ndarray, gamma: dict, fallback: np.ndarray, cols, M_ref: np.ndarray | None = None):
    """Per-class CAT thresholds (probability scale) from reference logits l_ref (n x K).

    gamma: class -> estimated proportion; fallback: thresholds kept for classes not in `cols`
    and for classes whose proportion is 0, 1 or undefined. M_ref (optional): known-label mask of
    the reference rows; only rows with a known label of class c are ranked for c.
    Returns (thresholds, list of classes that kept the fallback).
    """
    t = np.array(fallback, dtype=float).copy()
    kept = []
    for c in cols:
        g = gamma.get(c, float("nan"))
        if not np.isfinite(g) or g <= 0.0 or g >= 1.0:
            kept.append(c)
            continue
        rows = l_ref[:, c] if M_ref is None else l_ref[M_ref[:, c] > 0, c]
        if len(rows) == 0:
            kept.append(c)
            continue
        t[c] = _quantile_threshold(rows, g)
    return t, kept


def multilabel_f1(L: np.ndarray, T: np.ndarray, Y: np.ndarray, M: np.ndarray, cols, tag: str = ""):
    """Threshold-dependent multi-label measures over known labels of `cols`, in %.

    CF1: mean per-class F1 over classes with at least one known positive.
    OF1: F1 of all (image, class) pairs pooled.
    HL: Hamming loss, share of wrong known (image, class) decisions.
    tag: suffix for the keys, e.g. "sh" gives "CF1_sh_%".
    Only call it on classes the model actually predicts. The legacy model has no outputs for
    the new classes (its padded logits are 0), so for Legacy use the shared classes only.
    """
    P = _sigmoid(L[:, cols]) >= T[:, cols]
    y = Y[:, cols] == 1
    k = M[:, cols] > 0
    tp = (P & y & k).sum(0).astype(float)
    fp = (P & ~y & k).sum(0).astype(float)
    fn = (~P & y & k).sum(0).astype(float)
    has = (y & k).sum(0) > 0
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12), 0.0)
    cf1 = float(f1[has].mean()) if has.any() else float("nan")
    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    of1 = float(2 * TP / (2 * TP + FP + FN)) if (2 * TP + FP + FN) > 0 else float("nan")
    hl = float(((P != y) & k).sum() / max(1, k.sum()))
    sfx = f"_{tag}" if tag else ""
    return {f"CF1{sfx}_%": 100 * cf1, f"OF1{sfx}_%": 100 * of1, f"HL{sfx}_%": 100 * hl}
