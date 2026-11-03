"""CURE: Congruent Update for Retinal-label Evolution, plus the update baselines.

Setting. A legacy multi-label model was trained on RFMiD 1.0 (45 classes).
RFMiD 2.0 brings new images and a new label space: 38 shared classes,
12 new classes, 7 retired classes (unlabelled in 2.0). The old training
images are NOT used by any update method (only the legacy model is).
The update must learn the new classes, adapt the shared ones, and not
regress cases the legacy model already got right.

Output space of every updated model ("union" space, K = 45 + 12 = 57):
    union[:45]  = RFMiD 1.0 classes in their original order (shared + retired)
    union[45:]  = RFMiD 2.0-only classes

CURE (single novelty, three coupled parts):
  1. Per-class, per-image gated fusion of a frozen legacy path and an update
     path for the shared classes; the gate bias is initialised from each
     class's measured separability retention on the 2.0 labelled data.
  2. Multi-label positive-congruent margin loss (ML-PC): wherever the legacy
     model was correct at its operating threshold, the updated model's signed
     margin may not fall below the legacy margin (capped), with a higher
     weight on positives (a lost detection costs more than a lost negative).
  3. Per-class non-inferiority fallback on the 2.0 validation set.
  Retired classes keep the legacy output (they have no 2.0 labels).

Baselines implemented here (same backbone, LoRA budget, splits, loss):
  FT      naive update from the legacy weights
  KD      FT + distillation to the legacy outputs (LwF-style)
  FD      focal distillation (Yan et al., CVPR 2021), multi-label adaptation
  WiSE    weight interpolation of FT with the legacy weights (alpha = 0.5)
  OE      output ensemble of legacy and FT (same inference cost as CURE)
  Select  per-class choice of legacy or FT by 2.0 validation AP (same cost)
  Joint   upper bound trained on RFMiD 1.0 + 2.0 training images together
"""
from __future__ import annotations

import copy
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ===========================================================================
# Utilities
# ===========================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cache(paths) -> torch.Tensor:
    """Cached JPEGs -> uint8 tensor N x 3 x S x S (kept on CPU)."""
    arr = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
    return torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).contiguous()


def augment(xb: torch.Tensor, crop: int, train: bool) -> torch.Tensor:
    """uint8 batch on device -> normalised float batch. Random crop/flip/jitter when training."""
    x = xb.float() / 255.0
    n, _, s, _ = x.shape
    if train:
        out = []
        for k in range(n):
            i, j = np.random.randint(0, s - crop + 1, size=2)
            out.append(x[k, :, i:i + crop, j:j + crop])
        x = torch.stack(out)
        flip = torch.rand(n, device=x.device) < 0.5
        x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
        b = (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.2
        c = 1 + (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.2
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        x = ((x - m) * c + m + b).clamp(0, 1)
    else:
        o = (s - crop) // 2
        x = x[:, :, o:o + crop, o:o + crop]
    return (x - MEAN.to(x.device)) / STD.to(x.device)


# ===========================================================================
# Model: LoRA ViT backbone + class-query decoder (+ per-class gate)
# ===========================================================================
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def build_backbone(name: str, pretrained: bool, img_size: int, lora_r: int, weights_file: str | None = None):
    """timm ViT with LoRA on every attention qkv projection; everything else frozen.

    weights_file: optional local checkpoint (.pth/.safetensors) used instead of a download.
    """
    import timm
    kw = {"pretrained_cfg_overlay": {"file": str(weights_file)}} if weights_file else {}
    m = timm.create_model(name, pretrained=pretrained, num_classes=0, img_size=img_size, **kw)
    for p in m.parameters():
        p.requires_grad = False
    for blk in m.blocks:
        blk.attn.qkv = LoRALinear(blk.attn.qkv, r=lora_r, alpha=2 * lora_r)
    return m


class Decoder(nn.Module):
    """One learnable query per class attends to the patch tokens.

    logits_c = <h_c, W_c> + b_c ;  gate logit_c = <h_c, U_c> + beta_c
    """

    def __init__(self, dim: int, n_classes: int, heads: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_classes, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.W = nn.Parameter(torch.randn(n_classes, dim) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_classes))
        self.U = nn.Parameter(torch.zeros(n_classes, dim))
        self.beta = nn.Parameter(torch.zeros(n_classes))

    PER_CLASS = ("queries", "W", "b", "U", "beta")

    def forward(self, tokens):
        q = self.queries.unsqueeze(0).expand(tokens.size(0), -1, -1)
        a, _ = self.attn(q, tokens, tokens, need_weights=False)
        h = self.ln1(q + a)
        h = self.ln2(h + self.ffn(h))
        return (h * self.W).sum(-1) + self.b, (h * self.U).sum(-1) + self.beta


class Net(nn.Module):
    def __init__(self, backbone, n_classes: int):
        super().__init__()
        self.backbone = backbone
        self.decoder = Decoder(backbone.num_features, n_classes)

    def forward(self, x):
        return self.decoder(self.backbone.forward_features(x))

    def trainable_state(self):
        return {k: v.detach().clone() for k, v in self.named_parameters() if v.requires_grad}

    def load_trainable(self, state):
        own = dict(self.named_parameters())
        with torch.no_grad():
            for k, v in state.items():
                own[k].copy_(v)


def expand_from_legacy(legacy: Net, n_union: int, n_old: int) -> Net:
    """Copy the legacy model; rows 0..n_old-1 keep the legacy class parameters."""
    net = Net(copy.deepcopy(legacy.backbone), n_union)
    d_new, d_old = net.decoder, legacy.decoder
    for mod in ("attn", "ln1", "ln2", "ffn"):
        getattr(d_new, mod).load_state_dict(getattr(d_old, mod).state_dict())
    with torch.no_grad():
        for name in Decoder.PER_CLASS:
            getattr(d_new, name)[:n_old].copy_(getattr(d_old, name))
    for p in net.backbone.parameters():
        p.requires_grad = False
    for n_, p in net.backbone.named_parameters():
        if n_.endswith(".A") or n_.endswith(".B"):
            p.requires_grad = True
    return net


# ===========================================================================
# Losses
# ===========================================================================
def asl(logits, y, m, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
    """Asymmetric loss (Ridnik et al., ICCV 2021), masked to known labels."""
    p = torch.sigmoid(logits.float())
    p_neg = (p - clip).clamp(min=0)
    lp = y * torch.log(p.clamp(min=eps)) * (1 - p) ** gamma_pos
    ln = (1 - y) * torch.log((1 - p_neg).clamp(min=eps)) * p_neg ** gamma_neg
    return -((lp + ln) * m).sum() / logits.size(0)


def kd_loss(student, teacher, w, T=2.0):
    """Multi-label distillation: BCE to the teacher's tempered sigmoid.

    Weighted mean over (image, class) entries, so its scale matches the
    classification loss whatever the number of distilled classes.
    """
    pt = torch.sigmoid(teacher.float() / T)
    l = F.binary_cross_entropy_with_logits(student.float() / T, pt, reduction="none")
    return (l * w).sum() / w.sum().clamp(min=1e-6) * T * T


def mlpc_loss(new, old, y, known, thr_logit, kappa=2.0, w_pos=2.0):
    """Multi-label positive-congruent margin loss (CURE).

    Signed margin at the legacy operating point: m = (2y-1)(logit - logit(t_c)).
    Where the legacy margin is positive (legacy correct), penalise the update
    for a margin below min(legacy margin, kappa). Positives weigh w_pos.
    """
    s = 2 * y - 1
    m_old = s * (old.float() - thr_logit)
    m_new = s * (new.float() - thr_logit)
    correct = (m_old > 0).float() * known
    w = torch.where(y > 0, torch.full_like(y, w_pos), torch.ones_like(y))
    return (F.relu(m_old.clamp(max=kappa) - m_new) * correct * w).sum() / new.size(0)


# ===========================================================================
# Forward per method
# ===========================================================================
GATED = {"CURE", "CURE-noPC", "CURE-noPrior"}
USES_LEGACY_IN_TRAINING = GATED | {"KD", "FD", "CURE-noGate"}


def forward_method(net, legacy, x, method, ctx):
    """Final union-space logits for a method, plus legacy logits when used."""
    logits, gate_logit = net(x)
    l_old = None
    if method in USES_LEGACY_IN_TRAINING:
        with torch.no_grad():
            lo, _ = legacy(x)
        l_old = torch.cat([lo, torch.zeros(lo.size(0), logits.size(1) - lo.size(1), device=lo.device, dtype=lo.dtype)], 1)
    if method in GATED:
        sh, rt = ctx["shared_idx"], ctx["retired_idx"]
        g = torch.sigmoid(gate_logit[:, sh].float())
        final = logits.float().clone()
        final[:, sh] = g * logits[:, sh].float() + (1 - g) * l_old[:, sh].float()
        final[:, rt] = l_old[:, rt].float()
        return final, {"l_old": l_old, "l_new": logits, "gate": g}
    return logits, {"l_old": l_old, "l_new": logits}


def method_loss(method, out, aux, y, m, ctx, cfg):
    loss = asl(out, y, m)
    if method in ("legacy", "FT"):
        return loss
    sh, old = ctx["shared_idx"], ctx["old_idx"]
    if method in ("KD", "FD"):
        w = torch.zeros_like(y)
        w[:, old] = 1.0
        if method == "FD":                       # focal: extra weight where legacy was right
            s = 2 * y - 1
            correct = ((s * (aux["l_old"].float() - ctx["thr_logit"])) > 0).float() * m
            w[:, sh] = cfg["fd_alpha"] + cfg["fd_beta"] * correct[:, sh]
        loss = loss + cfg["kd_lambda"] * kd_loss(aux["l_new"][:, old], aux["l_old"][:, old], w[:, old])
    if method in GATED:
        loss = loss + cfg["aux_lambda"] * asl(aux["l_new"], y, m)    # keep the update path learning
    if method in GATED or method == "CURE-noGate":
        if method != "CURE-noPC":
            loss = loss + cfg["pc_lambda"] * mlpc_loss(
                out[:, sh], aux["l_old"][:, sh], y[:, sh], m[:, sh], ctx["thr_logit"][sh],
                kappa=cfg["pc_kappa"], w_pos=cfg["pc_wpos"])
    return loss


# ===========================================================================
# Training / prediction
# ===========================================================================
@torch.no_grad()
def predict(net, legacy, X, method, ctx, cfg, device):
    net.eval()
    outs = []
    for s in range(0, len(X), cfg["eval_bs"]):
        xb = augment(X[s:s + cfg["eval_bs"]].to(device), cfg["crop"], train=False)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out, _ = forward_method(net, legacy, xb, method, ctx)
        outs.append(out.float().cpu())
    return torch.cat(outs).numpy()


def macro_ap(logits, Y, M, cols):
    aps = per_class_ap(logits, Y, M, cols)
    vals = [v for v in aps.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def per_class_ap(logits, Y, M, cols):
    out = {}
    for c in cols:
        k = M[:, c] > 0
        y = Y[k, c]
        out[c] = float(average_precision_score(y, logits[k, c])) if 0 < y.sum() < len(y) else float("nan")
    return out


def train(net, legacy, method, tr, va, ctx, cfg, device, eval_cols, log=print, forward_fn=None, loss_fn=None,
          epoch_callback=None):
    """Train the trainable parameters; early stopping on validation macro mAP over eval_cols.

    va=None: no validation data is touched; trains exactly cfg["epochs"] epochs and keeps the
    final weights (used when the validation images must stay unseen for certification).
    forward_fn / loss_fn: optional replacements for forward_method / method_loss (CURE v2, OPC).
    epoch_callback(val_logits): called after each epoch's validation pass (OPC updates its
    operating thresholds here).
    """
    forward_fn = forward_fn or forward_method
    loss_fn = loss_fn or method_loss
    net.to(device)
    lora = [p for n, p in net.named_parameters() if p.requires_grad and (n.endswith(".A") or n.endswith(".B"))]
    head = [p for n, p in net.named_parameters() if p.requires_grad and not (n.endswith(".A") or n.endswith(".B"))]
    opt = torch.optim.AdamW([{"params": lora, "lr": cfg["lr_lora"]}, {"params": head, "lr": cfg["lr_head"]}],
                            weight_decay=cfg["wd"])
    epochs, bs = cfg["epochs"], cfg["bs"]
    steps_per_epoch = math.ceil(len(tr["X"]) / bs)
    total, warm = epochs * steps_per_epoch, steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    amp_scaler = getattr(torch.amp, "GradScaler", None)
    scaler = (amp_scaler("cuda", enabled=device.type == "cuda") if amp_scaler
              else torch.cuda.amp.GradScaler(enabled=device.type == "cuda"))

    best, best_state, bad = -1.0, net.trainable_state(), 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(tr["X"]))
        t0, run = time.time(), 0.0
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]
            xb = augment(tr["X"][idx].to(device, non_blocking=True), cfg["crop"], train=True)
            yb, mb = tr["Y"][idx].to(device), tr["M"][idx].to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out, aux = forward_fn(net, legacy, xb, method, ctx)
                loss = loss_fn(method, out, aux, yb, mb, ctx, cfg)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(lora + head, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
        if va is None:
            log(f"    ep {ep + 1:02d}  loss {run / steps_per_epoch:.3f}  ({time.time() - t0:.0f}s)")
            continue
        val_logits = predict(net, legacy, va["X"], method, ctx, cfg, device)
        val = macro_ap(val_logits, va["Y"].numpy(), va["M"].numpy(), eval_cols)
        if epoch_callback is not None:
            epoch_callback(val_logits)
        log(f"    ep {ep + 1:02d}  loss {run / steps_per_epoch:.3f}  val mAP {val:.4f}  ({time.time() - t0:.0f}s)")
        if val > best + 1e-4:
            best, best_state, bad = val, net.trainable_state(), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    if va is None:
        return float("nan")
    net.load_trainable(best_state)
    return best


# ===========================================================================
# Thresholds, gate prior, post-hoc variants
# ===========================================================================
def f1_thresholds(logits, Y, M, fallback):
    """Per-class F1-optimal probability threshold on a labelled split; fallback where undefined."""
    p = 1 / (1 + np.exp(-logits))
    t = np.array(fallback, dtype=float).copy()
    for c in range(Y.shape[1]):
        k = M[:, c] > 0
        y, pc = Y[k, c], p[k, c]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        cand = np.unique(np.concatenate([pc, [0.5]]))
        best, bt = -1, t[c]
        for th in cand:
            pred = pc >= th
            tp = (pred & (y == 1)).sum()
            f1 = 2 * tp / max(1, pred.sum() + y.sum())
            if f1 > best:
                best, bt = f1, th
        t[c] = bt
    return t


def gate_prior(legacy_logits_src_val, Ysv, legacy_logits_tgt_tr, Ytt, Mtt, shared_idx, tau=0.5, beta=0.5, mag=1.5):
    """Gate bias per shared class from separability retention r = delta_T / delta_S.

    r >= beta (intact)  -> -mag (lean on the legacy path)
    r <  beta (collapsed)-> +mag (lean on the update path)
    undefined            ->  0
    """
    prior, info = {}, {}
    for c in shared_idx:
        ys, ls = Ysv[:, c], legacy_logits_src_val[:, c]
        k = Mtt[:, c] > 0
        yt, lt = Ytt[k, c], legacy_logits_tgt_tr[k, c]
        if ys.sum() == 0 or ys.sum() == len(ys) or yt.sum() == 0 or yt.sum() == len(yt):
            prior[c], info[c] = 0.0, float("nan")
            continue
        dS = ls[ys == 1].mean() - ls[ys == 0].mean()
        dT = lt[yt == 1].mean() - lt[yt == 0].mean()
        if dS <= tau:
            prior[c], info[c] = 0.0, float("nan")
            continue
        r = dT / dS
        prior[c], info[c] = (-mag if r >= beta else mag), float(r)
    return prior, info


def wise_interpolate(ft: Net, init_state: dict, alpha: float, n_old: int) -> Net:
    """theta = (1-alpha) theta_legacy + alpha theta_FT; new-class rows taken from FT."""
    net = copy.deepcopy(ft)
    st = ft.trainable_state()
    mixed = {}
    for k, v in st.items():
        v0 = init_state[k].to(v.device)
        mix = (1 - alpha) * v0 + alpha * v
        if k.split(".")[-1] in Decoder.PER_CLASS and k.startswith("decoder."):
            mix[n_old:] = v[n_old:]
        mixed[k] = mix
    net.load_trainable(mixed)
    return net


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def output_ensemble(l_ft, l_leg_union, n_old):
    p = 1 / (1 + np.exp(-l_ft))
    q = 1 / (1 + np.exp(-l_leg_union))
    out = l_ft.copy()
    out[:, :n_old] = logit(0.5 * (p[:, :n_old] + q[:, :n_old]))
    return out


def per_class_select(l_ft_val, l_leg_val, Yv, Mv, l_ft, l_leg, shared_idx, retired_idx):
    """Per shared class pick legacy or FT by 2.0 validation AP; retired -> legacy."""
    out = l_ft.copy()
    ap_ft = per_class_ap(l_ft_val, Yv, Mv, shared_idx)
    ap_lg = per_class_ap(l_leg_val, Yv, Mv, shared_idx)
    chose = {}
    for c in shared_idx:
        if not np.isnan(ap_lg[c]) and ap_lg[c] > ap_ft[c]:
            out[:, c] = l_leg[:, c]
            chose[c] = "legacy"
        else:
            chose[c] = "FT"
    out[:, retired_idx] = l_leg[:, retired_idx]
    return out, chose


def noninferiority_fallback(l_val, l_leg_val, Yv, Mv, shared_idx, n_boot=500, p_worse=0.8, seed=0):
    """CURE part 3: classes confidently worse than legacy on 2.0 val fall back to legacy."""
    rng = np.random.default_rng(seed)
    fall = []
    for c in shared_idx:
        k = np.where(Mv[:, c] > 0)[0]
        y = Yv[k, c]
        if y.sum() < 2 or y.sum() == len(y):
            continue
        worse = 0
        for _ in range(n_boot):
            b = rng.choice(len(k), len(k), replace=True)
            yb = y[b]
            if yb.sum() == 0 or yb.sum() == len(yb):
                continue
            worse += average_precision_score(yb, l_val[k[b], c]) < average_precision_score(yb, l_leg_val[k[b], c])
        if worse / n_boot > p_worse:
            fall.append(c)
    return fall


# ===========================================================================
# Evaluation
# ===========================================================================
def evaluate(logits, thr, leg_logits, leg_thr, Y, M, groups, shared_idx, regress_delta=0.05):
    """Metrics on the 2.0 test set.

    NFR  = negative flips / known (image, shared class) pairs
    NFR+ = negative flips among positives / known positive pairs (missed-disease flips)
    PFR  = positive flips / known pairs
    regressed = shared classes with AP below legacy AP by more than regress_delta
    """
    res = {f"mAP_{g}": macro_ap(logits, Y, M, cols) for g, cols in groups.items()}
    p, q = 1 / (1 + np.exp(-logits)), 1 / (1 + np.exp(-leg_logits))
    sh = shared_idx
    known = M[:, sh] > 0
    y = Y[:, sh] == 1
    new_ok = (p[:, sh] >= thr[sh]) == y
    old_ok = (q[:, sh] >= leg_thr[sh]) == y
    nf = old_ok & ~new_ok & known
    pf = ~old_ok & new_ok & known
    res["NFR_%"] = 100 * nf.sum() / max(1, known.sum())
    res["NFR+_%"] = 100 * (nf & y).sum() / max(1, (known & y).sum())
    res["PFR_%"] = 100 * pf.sum() / max(1, known.sum())
    ap_new, ap_old = per_class_ap(logits, Y, M, sh), per_class_ap(leg_logits, Y, M, sh)
    res["regressed_classes"] = int(sum(1 for c in sh if not np.isnan(ap_old[c]) and ap_new[c] < ap_old[c] - regress_delta))
    return res
