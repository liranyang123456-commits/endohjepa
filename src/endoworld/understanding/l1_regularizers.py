"""Optional L1 regularizers: temporal smoothness (STIR-like) and depth consistency.

These constrain the encoder latents only. The world-model loss stays in JEPA space
and never regresses pixels.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_smoothness(z_dense: torch.Tensor) -> torch.Tensor:
    """Penalise abrupt token jumps between adjacent timesteps (B,T,N,D)."""
    if z_dense.size(1) < 2:
        return z_dense.new_zeros(())
    return F.smooth_l1_loss(z_dense[:, 1:], z_dense[:, :-1])


def stir_endpoint_consistency(z_dense: torch.Tensor, pts_start: torch.Tensor,
                              pts_end: torch.Tensor, image_size: int) -> torch.Tensor:
    """Chamfer between tokens at start-point sites (t=0) and end-point sites (t=-1).

    pts_*: (B, N, 2) pixel xy. Missing points are ignored.
    """
    if z_dense.size(1) < 2 or pts_start.numel() == 0:
        return z_dense.new_zeros(())
    b, t, n, d = z_dense.shape
    hw = int(n ** 0.5)
    if hw * hw != n:
        return temporal_smoothness(z_dense)
    z0 = z_dense[:, 0].reshape(b, hw, hw, d)
    z1 = z_dense[:, -1].reshape(b, hw, hw, d)
    scale = hw / float(image_size)
    def _gather(z, pts):
        x = (pts[..., 0] * scale).long().clamp(0, hw - 1)
        y = (pts[..., 1] * scale).long().clamp(0, hw - 1)
        out = []
        for i in range(b):
            out.append(z[i, y[i], x[i]])  # (N, D)
        return torch.stack(out, 0)
    s = _gather(z0, pts_start)
    e = _gather(z1, pts_end)
    # chamfer
    dist = torch.cdist(s, e)
    return dist.min(-1).values.mean() + dist.min(-2).values.mean()


def depth_consistency(z_dense: torch.Tensor, depth: torch.Tensor,
                      patch: int) -> torch.Tensor:
    """Align spatial-token change with pooled depth change (B,T,H,W depth)."""
    b, t, n, d = z_dense.shape
    if t < 2:
        return z_dense.new_zeros(())
    h = w = int(n ** 0.5)
    if h * w != n:
        return temporal_smoothness(z_dense)
    dep = F.interpolate(depth.reshape(b * t, 1, *depth.shape[-2:]),
                        size=(h, w), mode="bilinear", align_corners=False)
    dep = dep.view(b, t, n)
    dz = (z_dense[:, 1:] - z_dense[:, :-1]).abs().mean(dim=-1)
    dd = (dep[:, 1:] - dep[:, :-1]).abs()
    dd = dd / dd.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
    return F.smooth_l1_loss(dz, dd)
