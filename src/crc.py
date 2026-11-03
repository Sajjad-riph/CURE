"""Certified regression control for model updates (conformal risk control).

Risk controlled. For an image whose findings the legacy model detected (at its deployed
thresholds), the loss is the fraction of those legacy-detected positives that the
updated model misses:

    loss_i(lam) = #(legacy-detected positives of image i missed at thresholds t(lam))
                  / #(legacy-detected positives of image i)          in [0, 1]

Only images with at least one legacy-detected positive enter; this subset is a
deterministic function of each image's own data, so the images in it stay
exchangeable.

Threshold family. t_c(lam) = sigmoid(logit(base_c) - lam) on shared classes: larger
lam lowers every shared threshold by the same logit amount, so loss_i is
non-increasing in lam. `base` is the CT thresholds (per-class shape) or the plain F1
thresholds ("uniform" shape, for the ablation).

Calibration (Angelopoulos et al., Conformal Risk Control, ICLR 2024), with B = 1:

    lam_hat = min{ lam : n/(n+1) * mean_i loss_i(lam) + 1/(n+1) <= alpha }

Guarantee: E[loss of a new exchangeable image] <= alpha, in expectation over the
calibration set. To keep calibration data exchangeable with test data, the base
thresholds are fitted on one half of the validation fold and lam_hat on the other half.
"""
from __future__ import annotations

import numpy as np


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def shift_thresholds(base, shared_idx, lam):
    t = np.array(base, dtype=float).copy()
    t[shared_idx] = _sigmoid(_logit(t[shared_idx]) - lam)
    return t


def legacy_caught(L_old, thr_old, Y, M, shared_idx):
    """(n x |shared|) mask of positives the legacy model detects at its deployed thresholds."""
    sh = shared_idx
    return (M[:, sh] > 0) & (Y[:, sh] == 1) & (L_old[:, sh] >= _logit(thr_old[sh])[None, :])


def image_losses(L, thr, caught, shared_idx):
    """Per-image fraction of legacy-detected positives missed; only images with >= 1 caught."""
    sel = caught.sum(1) > 0
    pred = L[:, shared_idx] >= _logit(thr[shared_idx])[None, :]
    missed = (caught & ~pred).sum(1)
    return missed[sel] / caught.sum(1)[sel]


def calibrate(L_cal, base, caught_cal, shared_idx, alpha, grid):
    """Smallest lam (fewest false alarms) whose corrected calibration risk is <= alpha.

    Returns (lam, n, feasible). With n calibration images the bound needs alpha > 1/(n+1);
    if no lam in the grid satisfies it, the most conservative shift (largest lam) is
    returned with feasible=False, which keeps the guarantee but costs many false alarms.
    """
    sel = caught_cal.sum(1) > 0
    n = int(sel.sum())
    if n == 0:
        return float(np.max(grid)), n, False
    d = L_cal[:, shared_idx] - _logit(np.asarray(base)[shared_idx])[None, :]   # margins at lam = 0
    cnt = caught_cal.sum(1)
    for lam in np.sort(grid):
        missed = (caught_cal & (d < -lam)).sum(1)          # pred positive iff d >= -lam
        risk = float(np.mean(missed[sel] / cnt[sel]))
        if n / (n + 1) * risk + 1 / (n + 1) <= alpha:
            return float(lam), n, True
    return float(np.max(grid)), n, False


def split_halves(groups, seed):
    """Random 50/50 split of a set of images, keeping near-duplicate groups together."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    a = set(rng.permutation(ug)[: len(ug) // 2].tolist())
    in_a = np.array([g in a for g in groups])
    return np.where(in_a)[0], np.where(~in_a)[0]
