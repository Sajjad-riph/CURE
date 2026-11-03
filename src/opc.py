"""OPC: Operating-Point-Congruent updating.

Motivation (notebooks 02-03): most missed-disease flips after an update come
from re-choosing per-class operating thresholds, not from the new scores; a
legacy model with only re-tuned thresholds already loses 8.4% of the positives
it used to detect. A loss that protects margins at the *legacy* threshold
cannot control flips measured at the *new* threshold.

OPC couples the two:
  1. Congruence-constrained operating points (CT): for each shared class, the
     deployed threshold is the F1-best one among thresholds that keep every
     validation positive the legacy model detected.
  2. Operating-point-aware training: the hinge that protects legacy-detected
     positives is measured at the CURRENT CT threshold, re-estimated on the
     validation fold after every epoch, so training and deployment share
     one operating point:
         L = ASL + lambda * sum_{legacy-detected positives} relu(delta - (logit - logit(t_c^CT)))
               + lambda_r * KD(retired classes -> legacy)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import cure, cv


def loss_opc(method, out, aux, y, m, ctx, cfg):
    sh, rt = ctx["shared_idx"], ctx["retired_idx"]
    loss = cure.asl(out, y, m)
    l_old = aux["l_old"][:, sh].float()
    caught = ((l_old >= ctx["thr_logit"][sh]) & (y[:, sh] > 0)).float() * m[:, sh]
    margin = out[:, sh].float() - ctx["thr_cur_logit"][sh]
    loss = loss + cfg["op_lambda"] * (F.relu(cfg["op_delta"] - margin) * caught).sum() / out.size(0)
    if len(rt):
        loss = loss + cfg["op_ret_lambda"] * cure.kd_loss(out[:, rt], aux["l_old"][:, rt], torch.ones_like(out[:, rt]))
    return loss


def make_threshold_updater(ctx, Yv, Mv, l_old_val, thr_old, shared_idx, fallback, eps, device):
    """Epoch callback: re-estimate the CT thresholds on the validation fold."""
    def update(val_logits):
        t = cv.constrained_thresholds(val_logits, Yv, Mv, l_old_val, thr_old, shared_idx, fallback, eps)
        ctx["thr_cur_logit"] = torch.tensor(cure.logit(t), dtype=torch.float32, device=device)
        ctx["thr_cur"] = t
    return update
