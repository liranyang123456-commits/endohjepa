"""A compact, self-contained differentiable 3D Gaussian Splatting renderer (PyTorch).

Real 3DGS optimises a set of 3D Gaussians (mean, scale, opacity, colour) so that
their differentiable rasterisation matches reference images; the scene can then be
rendered from novel views. This implementation uses per-Gaussian screen-space
footprints splatted with a fixed kernel and normalised alpha accumulation - the same
core idea, kept dependency-free (no CUDA compilation) so it runs anywhere torch runs.

For production-scale quality use `gsplat` / `nerfstudio` (or EndoGaussian for tissue);
this module shares the same interface concept (init from a point cloud, optimise, render).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    H: int
    W: int
    R: torch.Tensor  # (3,3) world->cam
    t: torch.Tensor  # (3,)


class GaussianModel(nn.Module):
    """Learnable 3D Gaussians initialised from a colored point cloud."""

    def __init__(self, points: torch.Tensor, colors: torch.Tensor,
                 init_scale: float = 1.5):
        super().__init__()
        n = points.shape[0]
        self.means = nn.Parameter(points.clone())
        self.log_scale = nn.Parameter(torch.full((n, 1), float(torch.log(torch.tensor(init_scale)))))
        self.color_logit = nn.Parameter(torch.logit(colors.clamp(1e-3, 1 - 1e-3)))
        self.opacity_logit = nn.Parameter(torch.full((n, 1), 2.0))  # ~0.88 opacity

    @property
    def n(self):
        return self.means.shape[0]

    def render(self, cam: Camera, kernel: int = 5, near: float = 1.0):
        """Differentiable render -> (H, W, 3) in [0,1]."""
        device = self.means.device
        p = self.means @ cam.R.T.to(device) + cam.t.to(device)     # (G,3) camera frame
        z = p[:, 2]
        front = z > near
        p, z = p[front], z[front]
        color = torch.sigmoid(self.color_logit[front])            # (G,3)
        opacity = torch.sigmoid(self.opacity_logit[front])        # (G,1)
        scale = torch.exp(self.log_scale[front]).squeeze(1)       # (G,)

        u = cam.fx * p[:, 0] / z + cam.cx
        v = cam.fy * p[:, 1] / z + cam.cy
        radius = (cam.fx * scale / z).clamp(0.6, kernel)          # screen-space sigma (px)

        h = kernel // 2
        offs = torch.arange(-h, h + 1, device=device)
        du, dv = torch.meshgrid(offs, offs, indexing="ij")
        du = du.reshape(-1).float(); dv = dv.reshape(-1).float()  # (O,)

        u0 = u.round().long(); v0 = v.round().long()
        pix_u = u0[:, None] + du.long()[None, :]                  # (G,O)
        pix_v = v0[:, None] + dv.long()[None, :]
        valid = (pix_u >= 0) & (pix_u < cam.W) & (pix_v >= 0) & (pix_v < cam.H)

        dist2 = (pix_u.float() - u[:, None]) ** 2 + (pix_v.float() - v[:, None]) ** 2
        sig2 = (radius[:, None] ** 2) + 1e-6
        gw = torch.exp(-0.5 * dist2 / sig2)                       # (G,O)
        w = (opacity * gw) * valid.float()                       # (G,O)

        # depth-aware weighting: nearer Gaussians dominate (soft occlusion)
        depth_w = torch.softmax(-z, dim=0)[:, None] * self.n     # relative, mild
        w = w * depth_w.clamp(0.2, 5.0)

        flat = (pix_v.clamp(0, cam.H - 1) * cam.W + pix_u.clamp(0, cam.W - 1)).reshape(-1)
        wf = w.reshape(-1)
        col_contrib = (w[:, :, None] * color[:, None, :]).reshape(-1, 3)

        col_acc = torch.zeros(cam.H * cam.W, 3, device=device)
        w_acc = torch.zeros(cam.H * cam.W, device=device)
        col_acc.index_add_(0, flat, col_contrib)
        w_acc.index_add_(0, flat, wf)
        img = col_acc / (w_acc[:, None] + 1e-6)
        return img.view(cam.H, cam.W, 3)


def fit(model: GaussianModel, cam: Camera, target: torch.Tensor,
        iters: int = 200, lr: float = 0.02, kernel: int = 5, log_every: int = 50):
    """Optimise Gaussians to match a reference image `target` (H,W,3) in [0,1]."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(iters):
        img = model.render(cam, kernel=kernel)
        loss = (img - target).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (it + 1) % log_every == 0 or it == 0:
            with torch.no_grad():
                psnr = -10 * torch.log10(((img - target) ** 2).mean() + 1e-8)
            print(f"  [gs] iter {it+1:4d} L1={loss.item():.4f} PSNR={psnr.item():.2f}dB")
    return model
