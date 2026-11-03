"""2026 competitors, adapted to multi-label updating across RFMiD versions (notebook 09).

MPT, margin preserving training (Ricci, Biondi, Pernici, Del Bimbo, AAAI 2026).
  Original (softmax, single label): min L_Delta + lambda * L_FD(phi, phi_old) + lambda * L_FD(phi, phi_new), where
  L_Delta is cross-entropy with a positive logit bias k on the NEW classes during training only (removed at
  inference), and L_FD is focal distillation (Yan et al., CVPR 2021) from the old model and from a reference
  model phi_new trained on all classes with the standard loss.
  Multi-label adaptation used here (stated in the paper's methods):
    - L_Delta: asymmetric loss on (logits + Delta), Delta_c = k for the 12 RFMiD 2.0-only classes, 0 otherwise.
    - L_FD(phi, phi_old): sigmoid distillation on the 45 legacy outputs, per-(image, class) weight
      alpha + beta * [legacy correct at its deployed threshold] on shared classes, alpha elsewhere.
    - L_FD(phi, phi_new): the same on all 57 union outputs, with phi_new = the FT model of the same fold and its
      correctness judged at its own F1 thresholds fitted on the training fold (the validation fold stays unseen).
    - lambda = 1.0 (the paper's value for the KL distance), k = 1.5 (the paper's ImageNet value), alpha = 1, beta = 5.

RA-LDL, random anchors with low-rank decorrelated learning (Wu, Xu, Tong, ICLR 2026).
  Frozen pre-trained features phi(x) (CLS token); h(x) = ReLU(B^T phi) + ReLU(GELU(phi W1) W2), with a frozen
  Gaussian random anchor B (d1 = 5 d0) and a low-rank projection (r = 64) trained in the first session only;
  analytic ridge classifier W = (G + beta I)^-1 C from accumulated statistics G = sum h h^T, C = sum h y^T.
  Adaptation used here:
    - Feature extractor = the legacy model's own adapted backbone (DINOv2 + LoRA trained on RFMiD 1.0), i.e. the
      "adapted PTM" of the first session; session 1 = RFMiD 1.0 training set, session 2 = RFMiD 2.0 training folds.
    - Multi-hot targets; a label unknown in a session (new classes in session 1, retired classes in session 2)
      contributes 0 to C.
    - LRP trained in session 1 with a temporary linear head and the asymmetric loss; beta chosen on RFMiD 1.0 val.
    - B ~ N(0, 1/d0) (the paper's standard Gaussian up to a global scale, to which the ridge solution with a
      relative beta grid is invariant).
    - Ridge scores z are mapped to logits as L = 10 (z - 0.5) (monotone; only for threshold-grid resolution).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cure


# ===========================================================================
# MPT
# ===========================================================================
def mpt_forward(net, legacy, x, method, ctx):
    logits, _ = net(x)
    with torch.no_grad():
        lo, _ = legacy(x)
        ln, _ = ctx["ft_ref"](x)
    l_old = torch.cat([lo, torch.zeros(lo.size(0), logits.size(1) - lo.size(1), device=lo.device, dtype=lo.dtype)], 1)
    return logits, {"l_old": l_old, "l_ref": ln}


def mpt_loss(method, out, aux, y, m, ctx, cfg):
    out = out.float()
    loss = cure.asl(out + ctx["mpt_bias"][None, :], y, m)                 # margin-calibrated loss (bias in training only)
    s = 2 * y - 1
    sh, old = ctx["shared_idx"], ctx["old_idx"]
    a, b = cfg["fd_alpha"], cfg["fd_beta"]
    w_old = torch.zeros_like(y)
    w_old[:, old] = a
    correct_old = ((s * (aux["l_old"].float() - ctx["thr_logit"])) > 0).float() * m
    w_old[:, sh] = a + b * correct_old[:, sh]
    loss = loss + cfg["mpt_lambda"] * cure.kd_loss(out[:, old], aux["l_old"][:, old], w_old[:, old])
    correct_new = ((s * (aux["l_ref"].float() - ctx["thr_ref_logit"])) > 0).float() * m
    w_new = a + b * correct_new
    loss = loss + cfg["mpt_lambda"] * cure.kd_loss(out, aux["l_ref"], w_new)
    return loss


# ===========================================================================
# RA-LDL
# ===========================================================================
class RALDLFeatures(nn.Module):
    def __init__(self, d0: int, expand: int = 5, rank: int = 64, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        d1 = expand * d0
        self.register_buffer("B", torch.randn(d0, d1, generator=g) / np.sqrt(d0))
        self.W1 = nn.Linear(d0, rank, bias=False)
        self.W2 = nn.Linear(rank, d1, bias=False)
        nn.init.zeros_(self.W2.weight)                                     # residual starts at 0
        self.d1 = d1

    def forward(self, phi):
        return F.relu(phi @ self.B) + F.relu(self.W2(F.gelu(self.W1(phi))))


@torch.no_grad()
def cls_features(backbone, X, cfg, device):
    """CLS-token features of a (frozen) timm ViT for a uint8 image tensor."""
    backbone.eval()
    out = []
    for s in range(0, len(X), cfg["eval_bs"]):
        xb = cure.augment(X[s:s + cfg["eval_bs"]].to(device), cfg["crop"], train=False)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            tok = backbone.forward_features(xb)
        out.append(tok[:, 0].float().cpu())
    return torch.cat(out)


def train_lrp(feat: RALDLFeatures, phi, Y, M, device, epochs=20, bs=64, lr=1e-3, seed=0):
    """Session 1: train the low-rank projection with a temporary linear head (asymmetric loss)."""
    torch.manual_seed(seed)
    feat = feat.to(device)
    head = nn.Linear(feat.d1, Y.shape[1]).to(device)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    opt = torch.optim.AdamW(list(feat.W1.parameters()) + list(feat.W2.parameters()) + list(head.parameters()),
                            lr=lr, weight_decay=1e-4)
    phi, Y, M = phi.to(device), Y.to(device), M.to(device)
    for _ in range(epochs):
        perm = torch.randperm(len(phi), device=device)
        for s in range(0, len(perm), bs):
            i = perm[s:s + bs]
            loss = cure.asl(head(feat(phi[i])), Y[i], M[i])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    feat.eval()
    return feat


@torch.no_grad()
def project(feat, phi, device, bs=512):
    return torch.cat([feat(phi[s:s + bs].to(device)).cpu() for s in range(0, len(phi), bs)]).double().numpy()


def accumulate(H, Y, M):
    """Gram and class-accumulated matrices; unknown labels contribute 0."""
    return H.T @ H, H.T @ (np.asarray(Y, float) * np.asarray(M, float))


def ridge(G, C, beta_rel):
    d = G.shape[0]
    scale = np.trace(G) / d
    return np.linalg.solve(G + beta_rel * scale * np.eye(d), C)


def to_logits(H, W):
    return (10.0 * (H @ W - 0.5)).astype(np.float32)


def select_beta(G1, C1, Hval, Yval, Mval, cols, grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    best, best_b = -1.0, grid[0]
    for b in grid:
        L = to_logits(Hval, ridge(G1, C1, b))
        v = cure.macro_ap(L, Yval, Mval, cols)
        if v > best:
            best, best_b = v, b
    return best_b, best
