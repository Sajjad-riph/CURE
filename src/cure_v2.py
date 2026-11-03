"""CURE v2: trust-weighted, flip-focused congruent updating (single model).

Revision after the first go/no-go run, where the v1 gate + margin-matching
loss cost 0.08 mAP: its constraint covered almost every (image, class) pair,
because most pairs are legacy-correct negatives, and it tied the update to a
legacy model that is weak on RFMiD 2.0 (shared mAP 0.23).

CURE v2 keeps one model and one inference pass. Loss on RFMiD 2.0:

    L = ASL(all labelled classes)
      + lambda_c * sum_{c in shared} trust_c * sum_i  w(y_ic) * 1[legacy correct_ic]
                     * relu(delta - m_new_ic)                           (flip-focused congruence)
      + lambda_r * KD(retired classes -> legacy)                        (label-space retention)

    m_new_ic = (2 y_ic - 1) (logit_ic - logit(t_c)),  t_c = legacy operating threshold
    w(1) = w_pos, w(0) = w_neg   (a lost detection costs more than a new false alarm)

trust_c in [0, 1] comes from the legacy model's separability retention on the
2.0 training labels, r_c = delta_T / delta_S: classes the legacy model still
separates are protected; classes where it collapsed are left free to change.

Ablations: noTrust (trust = 1), noRet (no retired KD), matchMargin (v1-style
target min(m_old, kappa) instead of delta), sym (w_pos = w_neg = 1).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import cure

V2_METHODS = ["CURE2", "CURE2-noTrust", "CURE2-noRet", "CURE2-matchMargin", "CURE2-sym"]


def separability_trust(l_src_val, Y_src_val, l_tgt_train, Y_tgt_train, M_tgt_train, shared_idx, n_union,
                       tau=0.5, lo=0.3, hi=0.7, undefined=0.5):
    """trust_c = clip((r_c - lo) / (hi - lo), 0, 1); undefined r_c -> `undefined`.

    r_c = (mean legacy logit on positives - on negatives) in RFMiD 2.0 train
          divided by the same quantity on RFMiD 1.0 val.
    """
    trust = np.zeros(n_union, dtype=np.float32)
    ratio = {}
    for c in shared_idx:
        ys, ls = Y_src_val[:, c], l_src_val[:, c]
        k = M_tgt_train[:, c] > 0
        yt, lt = Y_tgt_train[k, c], l_tgt_train[k, c]
        ok = 0 < ys.sum() < len(ys) and 0 < yt.sum() < len(yt)
        dS = ls[ys == 1].mean() - ls[ys == 0].mean() if ok else 0.0
        if not ok or dS <= tau:
            trust[c], ratio[c] = undefined, float("nan")
            continue
        r = (lt[yt == 1].mean() - lt[yt == 0].mean()) / dS
        trust[c], ratio[c] = float(np.clip((r - lo) / (hi - lo), 0, 1)), float(r)
    return trust, ratio


def forward_v2(net, legacy, x, method, ctx):
    """Single updated model; the frozen legacy model is used only inside the loss."""
    logits, _ = net(x)
    with torch.no_grad():
        lo, _ = legacy(x)
    pad = torch.zeros(lo.size(0), logits.size(1) - lo.size(1), device=lo.device, dtype=lo.dtype)
    return logits, {"l_old": torch.cat([lo, pad], 1)}


def congruence_loss(new, old, y, known, thr_logit, trust, mode="flip", delta=0.5, kappa=2.0,
                    w_pos=4.0, w_neg=0.25):
    """Penalise the update where the legacy model was right and the update moves toward a flip.

    mode="flip":  target margin = delta (only near-flips are penalised)
    mode="match": target margin = min(legacy margin, kappa) (v1 behaviour, for the ablation)
    """
    s = 2 * y - 1
    m_old = s * (old.float() - thr_logit)
    m_new = s * (new.float() - thr_logit)
    correct = (m_old > 0).float() * known
    target = torch.full_like(m_old, delta) if mode == "flip" else m_old.clamp(max=kappa)
    w = torch.where(y > 0, torch.full_like(y, w_pos), torch.full_like(y, w_neg)) * trust.unsqueeze(0)
    return (F.relu(target - m_new) * correct * w).sum() / new.size(0)


def loss_v2(method, out, aux, y, m, ctx, cfg):
    sh, rt = ctx["shared_idx"], ctx["retired_idx"]
    loss = cure.asl(out, y, m)
    trust = torch.ones_like(ctx["trust"]) if method == "CURE2-noTrust" else ctx["trust"]
    w_pos, w_neg = (1.0, 1.0) if method == "CURE2-sym" else (cfg["c2_wpos"], cfg["c2_wneg"])
    mode = "match" if method == "CURE2-matchMargin" else "flip"
    loss = loss + cfg["c2_lambda"] * congruence_loss(
        out[:, sh], aux["l_old"][:, sh], y[:, sh], m[:, sh], ctx["thr_logit"][sh], trust[sh],
        mode=mode, delta=cfg["c2_delta"], kappa=cfg["c2_kappa"], w_pos=w_pos, w_neg=w_neg)
    if method != "CURE2-noRet" and len(rt):
        loss = loss + cfg["c2_ret_lambda"] * cure.kd_loss(out[:, rt], aux["l_old"][:, rt], torch.ones_like(out[:, rt]))
    return loss
