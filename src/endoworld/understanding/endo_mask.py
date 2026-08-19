"""Endoscopic appearance weights for JEPA: down-weight specular / overexposed tubelets.

High-luminance, low-saturation pixels (glare, fluid highlights) are weakly
predictable and should not dominate the latent loss.
"""

from __future__ import annotations

import torch


def specular_map(
    clip: torch.Tensor, lum_thr: float = 0.85, sat_thr: float = 0.18
) -> torch.Tensor:
    """clip (B,T,C,H,W) in [0,1] -> (B,T,H,W) in {0,1}, 1 = keep (not specular)."""
    r, g, b = clip[:, :, 0], clip[:, :, 1], clip[:, :, 2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    mx = clip.max(dim=2).values
    mn = clip.min(dim=2).values
    sat = torch.where(mx > 1e-6, (mx - mn) / mx.clamp_min(1e-6), torch.zeros_like(mx))
    glare = (lum > lum_thr) & (sat < sat_thr)
    return (~glare).float()


def tubelet_keep(keep: torch.Tensor, tubelet: int, patch: int) -> torch.Tensor:
    """Pool a (B,T,H,W) keep map onto JEPA tubelets (B, n_tokens)."""
    b, t, h, w = keep.shape
    t2 = (t // tubelet) * tubelet
    h2 = (h // patch) * patch
    w2 = (w // patch) * patch
    x = keep[:, :t2, :h2, :w2]
    x = x.reshape(b, t2 // tubelet, tubelet, h2 // patch, patch, w2 // patch, patch)
    # mean over tubelet, patch_h, patch_w -> (B, t', h', w')
    pooled = x.mean(dim=(2, 4, 6))
    return pooled.reshape(b, -1)


def token_loss_weights(
    clip: torch.Tensor,
    tubelet: int,
    patch: int,
    floor: float = 0.25,
    instrument_mask: torch.Tensor | None = None,
    instrument_boost: float = 1.0,
) -> torch.Tensor:
    keep = specular_map(clip)
    w = tubelet_keep(keep, tubelet, patch)
    if instrument_mask is not None:
        # instrument_mask: (B,T,H,W) in {0,1}; up-weight tool tubelets when available
        inst = tubelet_keep(instrument_mask.float(), tubelet, patch)
        w = w * (1.0 + instrument_boost * inst)
    return w.clamp(min=floor)
