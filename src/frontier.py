"""Regression trade-off frontiers: is CT better than simply lowering thresholds?

For one updated model (pooled out-of-fold logits L, per-row thresholds THR),
shift every shared-class threshold by the same amount delta in logit space.
Each delta gives one operating point:

  x = new false alarms, NFR- (%): legacy-correct negatives the update now calls positive,
      over all known negative (image, shared class) pairs
  y = missed-disease flips, NFR+ (%): legacy-detected positives the update now misses,
      over all known positive pairs (same definition as in notebooks 02-04)

Lowering thresholds moves along the curve (fewer misses, more false alarms).
CT is a real improvement only if its point lies BELOW the curve of the same
model, i.e. fewer misses at the same false-alarm rate.

Counting uses margins d = logit - logit(threshold): an entry is predicted
positive at shift delta iff d >= delta, so every curve is computed exactly
from sorted margins, with bootstrap weights (image multiplicities).
"""
from __future__ import annotations

import numpy as np


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def flip_sets(L, THR, L_old, thr_old, Y, M, shared_idx):
    """Margins and masks on (image, shared class) pairs."""
    sh = shared_idx
    d = L[:, sh] - _logit(THR[:, sh])
    old_pos = L_old[:, sh] >= _logit(thr_old[sh])[None, :]
    known = M[:, sh] > 0
    pos = known & (Y[:, sh] == 1)
    neg = known & (Y[:, sh] == 0)
    return {"d": d, "pos": pos, "neg": neg, "caught": pos & old_pos, "old_correct_neg": neg & ~old_pos}


def _weighted_curve(values, wts, grid, side):
    """sum of wts where value < grid (side='below') or value >= grid (side='atleast'), per grid point."""
    o = np.argsort(values)
    v, w = values[o], wts[o]
    cw = np.concatenate([[0.0], np.cumsum(w)])
    k = np.searchsorted(v, grid, side="left")            # number of values < grid
    below = cw[k]
    return below if side == "below" else cw[-1] - below


def curve(fs, grid, w_img=None):
    """(NFR-, NFR+) in % for every threshold shift in grid; w_img = bootstrap image weights."""
    n = fs["d"].shape[0]
    w = np.ones(n) if w_img is None else w_img
    W = np.broadcast_to(w[:, None], fs["d"].shape)
    pos_tot = W[fs["pos"]].sum()
    neg_tot = W[fs["neg"]].sum()
    miss = _weighted_curve(fs["d"][fs["caught"]], W[fs["caught"]], grid, "below")
    fa = _weighted_curve(fs["d"][fs["old_correct_neg"]], W[fs["old_correct_neg"]], grid, "atleast")
    return 100 * fa / max(neg_tot, 1e-12), 100 * miss / max(pos_tot, 1e-12)


def point(fs, w_img=None):
    """Operating point at the model's own thresholds (shift 0)."""
    x, y = curve(fs, np.array([0.0]), w_img)
    return float(x[0]), float(y[0])


def frontier_at(xc, yc, x):
    """NFR+ of a curve at false-alarm rate x (linear interpolation; curve sorted by x)."""
    o = np.lexsort((yc, xc))                              # by x, then y: duplicates keep their lowest NFR+
    xs, ys = xc[o], yc[o]                                 # (the most favourable to the baseline)
    xs, idx = np.unique(xs, return_index=True)
    return float(np.interp(x, xs, ys[idx]))


def area(xc, yc, lo, hi, n=200):
    """Mean NFR+ over false-alarm rates in [lo, hi] (lower = better trade-off)."""
    xs = np.linspace(lo, hi, n)
    return float(np.mean([frontier_at(xc, yc, v) for v in xs]))
