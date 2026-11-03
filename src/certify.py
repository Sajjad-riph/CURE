"""Affordable certified regression control (notebook 07).

Notebook 06 certified CT with conformal risk control (CRC, Angelopoulos et al., ICLR 2024) but
paid for it with many false alarms: the per-class threshold *shape* was fitted on half of the
small RFMiD 2.0 validation fold, so only the other half (about 22 images with a legacy-detected
finding) was left to calibrate the certificate, and the 1/(n+1) correction alone used most of α.

Three changes, each testable on its own:

1. Legacy-anchored shape. The per-class shape is fitted on the legacy release's validation set
   (RFMiD 1.0 val), which is always available when a model is updated and is independent of the
   new release's held-out images. Every new-release validation image then calibrates the single
   certified scalar. Together with fixed-epoch training (no early stopping on the validation
   fold), the calibration images are never seen by training or shape fitting, as CRC requires.

2. Cost-aware shape. Instead of CT's "keep every legacy-detected positive, then best F1", the
   shape is chosen greedily to reach a target miss risk with the fewest new false alarms: at each
   step, lower the threshold of the class whose next rescued legacy-detected positive costs the
   fewest legacy-correct negatives per unit of risk removed.

3. High-probability certificate (Learn then Test, Angelopoulos, Bates, Candes, Jordan and Lei).
   CRC bounds the risk in expectation over calibration sets; LTT bounds P(risk > α) ≤ δ using
   Hoeffding-Bentkus p-values (Bates et al., J. ACM 2021) with fixed-sequence testing over the
   monotone family of shifted thresholds.

Threshold family as in crc.py: t_c(lam) = sigmoid(logit(base_c) - lam) on shared classes.
"""
from __future__ import annotations

import math

import numpy as np

from . import crc, cv, frontier


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ===========================================================================
# Shapes
# ===========================================================================
def legacy_correct_negatives(L_old, thr_old, Y, M, shared_idx):
    """(n x |shared|) mask of known negatives the legacy model calls negative (NFR- denominator pairs)."""
    sh = shared_idx
    return (M[:, sh] > 0) & (Y[:, sh] == 0) & (L_old[:, sh] < _logit(thr_old[sh])[None, :])


def cost_aware_shape(L, base, caught, okneg, shared_idx, target):
    """Greedy per-class threshold lowering until the image-level miss risk on this set is <= target.

    risk = mean over images with >= 1 legacy-detected positive of (missed / detected).
    Each step lowers one class's threshold to just include its highest-scoring missed
    legacy-detected positive; the class chosen maximises (risk removed) / (1 + new false alarms
    on legacy-correct negatives). Returns (thresholds, risk reached on this set, steps).
    """
    sh = np.asarray(shared_idx)
    t = np.array(base, dtype=float).copy()
    d = L[:, sh] - _logit(t[sh])[None, :]
    cnt = caught.sum(1)
    sel = cnt > 0
    if not sel.any():
        return t, 0.0, 0
    w = np.zeros(len(L))
    w[sel] = 1.0 / cnt[sel] / sel.sum()                    # risk carried by one missed pair of image i
    delta = np.zeros(len(sh))
    risk = float((w[:, None] * (caught & (d < 0))).sum())
    steps = 0
    while risk > target + 1e-12:
        best = None
        for j in range(len(sh)):
            missed = caught[:, j] & (d[:, j] < -delta[j])
            if not missed.any():
                continue
            new_delta = -float(d[missed, j].max()) + 1e-4   # include the closest missed positive (small float margin)
            band = (d[:, j] >= -new_delta) & (d[:, j] < -delta[j])
            gain = float(w[band & caught[:, j]].sum())
            cost = int((band & okneg[:, j]).sum())
            score = gain / (cost + 1.0)
            if best is None or score > best[0] + 1e-15 or (abs(score - best[0]) <= 1e-15 and cost < best[3]):
                best = (score, j, new_delta, cost, gain)
        if best is None:
            break
        _, j, nd, _, g = best
        delta[j] = nd
        risk -= g
        steps += 1
    t[sh] = _sigmoid(_logit(t[sh]) - delta)
    return t, max(risk, 0.0), steps


def fit_shape(kind, L, Y, M, L_old, thr_old, shared_idx, fallback, target=None, ct_eps=0.0):
    """Per-class thresholds fitted on one labelled set.

    kind: "uniform" (per-class F1), "ct" (congruence-constrained, cv.constrained_thresholds),
          "cost" (cost-aware greedy from the F1 thresholds to `target` risk).
    """
    if kind == "uniform":
        return cv.plain_thresholds(L, Y, M, fallback)
    if kind == "ct":
        return cv.constrained_thresholds(L, Y, M, L_old, thr_old, shared_idx, fallback, ct_eps)
    if kind == "cost":
        base = cv.plain_thresholds(L, Y, M, fallback)
        caught = crc.legacy_caught(L_old, thr_old, Y, M, shared_idx)
        okneg = legacy_correct_negatives(L_old, thr_old, Y, M, shared_idx)
        return cost_aware_shape(L, base, caught, okneg, shared_idx, target)[0]
    raise ValueError(kind)


# ===========================================================================
# High-probability certificate: Learn-then-Test with Hoeffding-Bentkus p-values
# ===========================================================================
def hb_pvalue(rhat, n, alpha):
    """p-value of H0: risk > alpha, for the mean rhat of n i.i.d. losses in [0, 1]."""
    from scipy.stats import binom
    if n == 0:
        return 1.0
    a = min(float(rhat), alpha)
    h1 = (a * math.log(a / alpha) if a > 0 else 0.0) + (1 - a) * math.log((1 - a) / (1 - alpha))
    p_h = math.exp(-n * h1)
    p_b = math.e * float(binom.cdf(math.ceil(n * rhat - 1e-9), n, alpha))
    return min(p_h, p_b, 1.0)


def ltt_calibrate(L_cal, base, caught_cal, shared_idx, alpha, delta, grid):
    """Smallest lam whose miss risk is certified <= alpha with probability >= 1 - delta.

    Fixed-sequence testing from the most conservative shift (largest lam) downwards; stops at the
    first lam that cannot be certified. Returns (lam, n, feasible); infeasible -> largest lam.
    """
    sel = caught_cal.sum(1) > 0
    n = int(sel.sum())
    if n == 0:
        return float(np.max(grid)), n, False
    d = L_cal[:, shared_idx] - _logit(np.asarray(base)[shared_idx])[None, :]
    cnt = caught_cal.sum(1)
    lam_hat, ok = float(np.max(grid)), False
    for lam in np.sort(grid)[::-1]:
        missed = (caught_cal & (d < -lam)).sum(1)
        risk = float(np.mean(missed[sel] / cnt[sel]))
        if hb_pvalue(risk, n, alpha) <= delta:
            lam_hat, ok = float(lam), True
        else:
            break
    return lam_hat, n, ok


# ===========================================================================
# Evaluation helpers
# ===========================================================================
def pooled_image_risk(L, T, caught, shared_idx, blocks):
    """Mean image loss over all evaluated images; T is constant within each block of row indices."""
    parts = [crc.image_losses(L[b], T[b][0], caught[b], shared_idx) for b in blocks if len(b)]
    parts = [p for p in parts if len(p)]
    return float(np.concatenate(parts).mean()) if parts else float("nan")


def envelope_stats(FS, curve_models, points, seeds, grid, w=None):
    """Seed-averaged (gap vs envelope, NFR-, NFR+) of every point; gap < 0 = below every model's curve."""
    curves = {(m, s): frontier.curve(FS[(m, s)], grid, w) for m in curve_models for s in seeds}
    gap, xs, ys = {}, {}, {}
    for p in points:
        g, xx, yy = [], [], []
        for s in seeds:
            x, y = frontier.point(FS[(p, s)], w)
            g.append(y - min(frontier.frontier_at(*curves[(m, s)], x) for m in curve_models))
            xx.append(x)
            yy.append(y)
        gap[p], xs[p], ys[p] = float(np.mean(g)), float(np.mean(xx)), float(np.mean(yy))
    return gap, xs, ys


def ci(est, reps):
    """[estimate, 2.5%, 97.5%] rounded."""
    reps = np.asarray(reps, dtype=float)
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    return [round(float(est), 3), round(float(lo), 3), round(float(hi), 3)]
