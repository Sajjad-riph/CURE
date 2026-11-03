"""2025 competitors adapted to multi-label updating across RFMiD versions (notebook 10).

All three use the shared architecture (DINOv2 ViT-B/14 + LoRA + class-query decoder), start from the legacy
weights, train for the same fixed number of epochs and never see the validation fold.

HCP (Zhang, Yang, Liu, Hong, Zhou, AAAI 2025), multi-label class-incremental learning without replay.
  - Stability / plasticity: the per-class parameters (query, weight, bias) of the 45 legacy classes are frozen;
    those of the 12 new classes, the shared decoder layers and the LoRA adapters are trained. The class-query
    decoder plays the role of HCP's feature purification module (one learnable embedding per class over the
    patch tokens).
  - Recall enhancement: labels unknown in the new release (the 7 retired classes) are pseudo-labelled from the
    legacy model with class-specific thresholds equal to the legacy mean confidence on positives of that class
    (RFMiD 1.0 validation split).
  - Probing unknown knowledge: a synthetic "unknown" feature is a random convex combination (Beta(1,1) weights)
    of the decoder features of the classes absent from the image; an extra classifier scores it as unknown (1)
    and scores the features of the present classes as known (0). The official repository contains no code, so
    this is our reading of the paper.
  - Weighted asymmetric loss with weight sqrt(|all classes| / |new classes|) on the new classes.

RCLP (Ceccon, Pezze, Fabris, Susto, WACV 2025), replay consolidation with label propagation.
  - Replay buffer: 3% of the RFMiD 1.0 training images, drawn at random; mixed 50/50 with new images.
  - Forward label propagation: labels unknown in the new release (retired classes) are pseudo-labelled by the
    legacy model at its deployed thresholds (F1-optimal on the old validation split, as in the paper).
  - Masking loss: replayed images contribute only through the 45 legacy outputs (their mask covers old labels).
  - Feature distillation: mean squared distance between the outputs of block 9 of 12 of the new and legacy
    backbones (the paper uses block 12 of 16 of a DenseNet), weight gamma = 1.
  - The backward step only matters for later updates and is not needed for a single update.

CRCL (Wu et al., MICCAI 2025), conservative-radical complementary learning (official code: CUHK-BMEAI/CRCL).
  - Conservative learner = the legacy model; radical learner = a copy trained on the new release with
    asymmetric loss + lambda * sum over the batch of (1 - cos) between the class tokens of both learners
    (lambda = 50, the official value) + cross-classification: the conservative head applied to the radical
    features, on the known legacy labels.
  - After training, the conservative adapters are consolidated by EMA (0.99 conservative + 0.01 radical).
  - Inference adds the logits of both learners on the 45 legacy outputs (task-agnostic fusion); the 12 new
    outputs come from the radical learner. The trained decoder replaces CRCL's random-projection ridge head,
    because the decoder is shared by all compared methods.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cure


# ===========================================================================
# Shared helpers
# ===========================================================================
def asl_weighted(logits, y, m, w, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
    """cure.asl with a per-class weight vector w."""
    p = torch.sigmoid(logits.float())
    p_neg = (p - clip).clamp(min=0)
    lp = y * torch.log(p.clamp(min=eps)) * (1 - p) ** gamma_pos
    ln = (1 - y) * torch.log((1 - p_neg).clamp(min=eps)) * p_neg ** gamma_neg
    return -((lp + ln) * m * w[None, :]).sum() / logits.size(0)


def pad_old(lo, k):
    return torch.cat([lo, torch.zeros(lo.size(0), k - lo.size(1), device=lo.device, dtype=lo.dtype)], 1)


def with_pseudo_labels(y, m, l_old, cols, thr):
    """Fill unknown labels of `cols` from the legacy model (threshold per class, probability scale)."""
    y, m = y.clone(), m.clone()
    unk = m[:, cols] == 0
    pseudo = (torch.sigmoid(l_old[:, cols].float()) >= thr[None, :]).float()
    y[:, cols] = torch.where(unk, pseudo, y[:, cols])
    m[:, cols] = torch.where(unk, torch.ones_like(m[:, cols]), m[:, cols])
    return y, m


def tokens_with_mid(bb, x, mid_block):
    """Backbone tokens plus the output of block `mid_block` (0-based), captured with a forward hook so the
    timm forward_features path (positional embedding, register tokens, checkpointing) is left untouched."""
    store = {}
    hook = bb.blocks[mid_block].register_forward_hook(lambda mod, inp, out: store.__setitem__("mid", out))
    try:
        tok = bb.forward_features(x)
    finally:
        hook.remove()
    return tok, store["mid"]


# ===========================================================================
# HCP
# ===========================================================================
class HCPDecoder(nn.Module):
    """Class-query decoder with frozen legacy-class parameters and an extra 'unknown' classifier."""

    def __init__(self, dec: cure.Decoder, n_old: int):
        super().__init__()
        self.n_old = n_old
        for name in ("queries", "W", "b"):
            v = getattr(dec, name).detach().clone()
            self.register_buffer(f"{name}_old", v[:n_old].clone())
            setattr(self, f"{name}_new", nn.Parameter(v[n_old:].clone()))
        self.attn, self.ln1, self.ln2, self.ffn = dec.attn, dec.ln1, dec.ln2, dec.ffn
        d = dec.queries.size(1)
        self.w_unk = nn.Parameter(torch.zeros(d))
        self.b_unk = nn.Parameter(torch.zeros(1))

    def forward(self, tokens):
        queries = torch.cat([self.queries_old, self.queries_new])
        W = torch.cat([self.W_old, self.W_new])
        b = torch.cat([self.b_old, self.b_new])
        q = queries.unsqueeze(0).expand(tokens.size(0), -1, -1)
        a, _ = self.attn(q, tokens, tokens, need_weights=False)
        h = self.ln1(q + a)
        h = self.ln2(h + self.ffn(h))
        return (h * W).sum(-1) + b, h                      # second slot: class features (used by the loss)


def hcp_forward(net, legacy, x, method, ctx):
    logits, h = net(x)
    with torch.no_grad():
        lo, _ = legacy(x)
    return logits, {"h": h, "l_old": pad_old(lo, logits.size(1))}


def hcp_loss(method, out, aux, y, m, ctx, cfg):
    y, m = with_pseudo_labels(y, m, aux["l_old"], ctx["retired_idx"], ctx["hcp_thr"])
    loss = asl_weighted(out, y, m, ctx["hcp_w"])
    h = aux["h"].float()                                   # B x K x d
    absent = ((m > 0) & (y == 0)).float()
    present = ((m > 0) & (y == 1)).float()
    lam = torch.rand(absent.shape, device=h.device) * absent       # Beta(1, 1) = uniform weights
    ok = lam.sum(1) > 0
    if ok.any():
        lam = lam[ok] / lam[ok].sum(1, keepdim=True)
        o_v = (lam.unsqueeze(-1) * h[ok]).sum(1)            # synthetic unknown features
        z_v = o_v @ net_param(ctx, "w_unk") + net_param(ctx, "b_unk")
        z_p = h[ok] @ net_param(ctx, "w_unk") + net_param(ctx, "b_unk")
        pm = present[ok]
        l_v = F.binary_cross_entropy_with_logits(z_v, torch.ones_like(z_v))
        l_p = (F.binary_cross_entropy_with_logits(z_p, torch.zeros_like(z_p), reduction="none") * pm).sum() / pm.sum().clamp(min=1)
        loss = loss + cfg["hcp_pu"] * (l_v + l_p)
    return loss


def net_param(ctx, name):
    return getattr(ctx["hcp_decoder"], name)


# ===========================================================================
# RCLP
# ===========================================================================
def rclp_forward(net, legacy, x, method, ctx):
    tok, mid = tokens_with_mid(net.backbone, x, ctx["mid_block"])
    logits, _ = net.decoder(tok)
    with torch.no_grad():
        tok_o, mid_o = tokens_with_mid(legacy.backbone, x, ctx["mid_block"])
        lo, _ = legacy.decoder(tok_o)
    return logits, {"mid": mid, "mid_old": mid_o, "l_old": pad_old(lo, logits.size(1))}


def rclp_loss(method, out, aux, y, m, ctx, cfg):
    y, m = with_pseudo_labels(y, m, aux["l_old"], ctx["retired_idx"], ctx["thr_prob"][ctx["retired_idx"]])
    loss = cure.asl(out, y, m)
    return loss + cfg["rclp_gamma"] * F.mse_loss(aux["mid"].float(), aux["mid_old"].float())


# ===========================================================================
# CRCL
# ===========================================================================
def crcl_forward(net, legacy, x, method, ctx):
    tok = net.backbone.forward_features(x)
    logits, _ = net.decoder(tok)
    cross, _ = legacy.decoder(tok)                         # conservative head on radical features
    with torch.no_grad():
        tok_c = legacy.backbone.forward_features(x)
    return logits, {"cls": tok[:, 0], "cls_c": tok_c[:, 0], "cross": cross}


def crcl_loss(method, out, aux, y, m, ctx, cfg):
    old = ctx["old_idx"]
    loss = cure.asl(out, y, m)
    loss = loss + cfg["crcl_lambda"] * (1 - F.cosine_similarity(aux["cls"].float(), aux["cls_c"].float(), dim=-1)).sum()
    return loss + cure.asl(aux["cross"], y[:, old], m[:, old])


def ema_conservative(legacy, radical, alpha=0.99):
    """Conservative learner after consolidation: legacy adapters moved towards the radical adapters by EMA."""
    cons = copy.deepcopy(legacy)
    rad = dict(radical.backbone.named_parameters())
    with torch.no_grad():
        for n_, p in cons.backbone.named_parameters():
            if n_.endswith(".A") or n_.endswith(".B"):
                p.mul_(alpha).add_(rad[n_].detach().to(p.device), alpha=1 - alpha)
    return cons


class CRCLNet(nn.Module):
    """Task-agnostic fusion: radical logits, plus conservative logits on the legacy outputs."""

    def __init__(self, radical, conservative, n_old):
        super().__init__()
        self.radical, self.conservative, self.n_old = radical, conservative, n_old

    def forward(self, x):
        lr, g = self.radical(x)
        lc, _ = self.conservative(x)
        out = lr.clone()
        out[:, :self.n_old] = out[:, :self.n_old] + lc
        return out, g
