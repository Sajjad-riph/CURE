"""Backbone for the model-update scenario (notebook 07, scenario 2).

The legacy model stays the DINOv2 ViT-B/14 model of notebook 02. The update is a different
architecture: RETFound (MAE ViT-L/16 pretrained on colour fundus photographs), with the same
LoRA budget on every attention qkv projection and the same class-query decoder, trained from
scratch on RFMiD 1.0. This is the classic backward-compatibility setting (new architecture,
same label space; Yan et al., CVPR 2021).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .cure import LoRALinear


class CheckpointedBlock(nn.Module):
    """Recompute a transformer block in the backward pass (saves activation memory).

    Non-reentrant checkpointing, so gradients reach the LoRA parameters even though the
    block input does not require grad (the reentrant variant would silently drop them).
    """

    def __init__(self, blk: nn.Module):
        super().__init__()
        self.blk = blk

    def forward(self, x, *args, **kwargs):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.blk, x, *args, use_reentrant=False, **kwargs)
        return self.blk(x, *args, **kwargs)


def _load_state(weights_file):
    try:
        ck = torch.load(weights_file, map_location="cpu", weights_only=False)
    except TypeError:                                       # very old torch without weights_only
        ck = torch.load(weights_file, map_location="cpu")
    if isinstance(ck, dict):
        for key in ("model", "state_dict", "model_state"):
            if key in ck and isinstance(ck[key], dict):
                ck = ck[key]
                break
    out = {}
    for k, v in ck.items():
        for pre in ("module.", "backbone.", "encoder."):
            if k.startswith(pre):
                k = k[len(pre):]
        out[k] = v
    return out


def build_retfound(name: str, weights_file, img_size: int, lora_r: int, grad_ckpt: bool = True, log=print):
    """timm ViT loaded from a RETFound checkpoint, frozen, LoRA on qkv, optional checkpointing."""
    import timm
    m = timm.create_model(name, pretrained=False, num_classes=0, img_size=img_size)
    if weights_file:
        sd = _load_state(weights_file)
        sd = {k: v for k, v in sd.items() if not k.startswith(("decoder", "mask_token", "head", "fc_norm"))}
        if "pos_embed" in sd and sd["pos_embed"].shape != m.pos_embed.shape:
            from timm.layers import resample_abs_pos_embed
            n_prefix = 0 if getattr(m, "no_embed_class", False) else getattr(m, "num_prefix_tokens", 1)
            sd["pos_embed"] = resample_abs_pos_embed(sd["pos_embed"], new_size=list(m.patch_embed.grid_size),
                                                     num_prefix_tokens=n_prefix)
        try:
            res = m.load_state_dict(sd, strict=False)
        except RuntimeError as e:
            raise RuntimeError(f"{weights_file} does not fit {name} (tensor shapes differ). Check s2_backbone and "
                               f"s2_weights in the configuration. Details: {str(e)[:300]}") from None
        core_missing = [k for k in res.missing_keys if k.startswith(("blocks.", "patch_embed.", "pos_embed", "cls_token"))]
        if core_missing:
            raise RuntimeError(f"{weights_file} does not match {name}: {len(core_missing)} encoder weights missing, "
                               f"e.g. {core_missing[:3]}. Check s2_backbone and s2_weights in the configuration.")
        log(f"  RETFound weights loaded ({len(sd)} tensors; not in checkpoint: {res.missing_keys or 'none'}; "
            f"ignored: {len(res.unexpected_keys)})")
    else:
        log("  WARNING: no weights file, the update backbone starts from random weights (smoke test only)")
    for p in m.parameters():
        p.requires_grad = False
    for blk in m.blocks:
        blk.attn.qkv = LoRALinear(blk.attn.qkv, r=lora_r, alpha=2 * lora_r)
    if grad_ckpt:
        m.blocks = nn.Sequential(*[CheckpointedBlock(b) for b in m.blocks])
    return m
