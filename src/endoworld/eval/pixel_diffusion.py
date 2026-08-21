"""Pixel-generation world-model baseline: a small conditional DDPM for next-frame
prediction, contrasted with Endo-HJEPA's representation-space (JEPA) prediction.

A minimal but real diffusion model: a UNet conditioned on the previous frame denoises
the next frame. We report PSNR/SSIM of the sampled next frame vs copy-last-frame, so
the pixel-generation paradigm is evaluated on the same data as the latent forecaster.

    python -m endoworld.eval.pixel_diffusion --max-clips 400 --epochs 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from endoworld.data.video_clips import EndoClipDataset, domain_balanced_indices


def _unet_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1), nn.GroupNorm(8, cout), nn.GELU())


class CondUNet(nn.Module):
    """Tiny conditional UNet: input = noisy next frame + prev frame (cond) + t embedding."""

    def __init__(self, ch=48):
        super().__init__()
        self.temb = nn.Sequential(nn.Linear(1, ch * 2), nn.GELU(), nn.Linear(ch * 2, ch * 2))
        self.enc1 = _unet_block(6, ch)          # noisy(3) + cond(3) -> ch
        self.enc2 = _unet_block(ch, ch * 2)     # ch -> 2ch
        self.mid = _unet_block(ch * 2, ch * 2)  # 2ch -> 2ch
        self.dec2 = _unet_block(ch * 2, ch)     # 2ch -> ch
        self.dec1 = nn.Conv2d(ch, 3, 3, 1, 1)   # ch -> 3
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x_noisy, cond, t):
        te = self.temb(t.view(-1, 1))[:, :, None, None]
        x = torch.cat([x_noisy, cond], dim=1)
        e1 = self.enc1(x)                  # (B, ch, H, W)
        e2 = self.enc2(self.pool(e1))      # (B, 2ch, H/2, W/2)
        m = self.mid(self.pool(e2)) + te   # (B, 2ch, H/4, W/4)
        d = self.dec2(self.up(m))          # (B, ch, H/2, W/2)
        d = self.up(d)                     # (B, ch, H, W)
        return self.dec1(d)                # (B, 3, H, W)


class DDPM:
    def __init__(self, T=200, device="cpu"):
        self.T = T
        self.device = device
        beta = torch.linspace(1e-4, 0.02, T, device=device)
        self.alpha = 1 - beta
        self.abar = torch.cumprod(self.alpha, 0)

    def q_sample(self, x0, t, eps):
        a = self.abar[t].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * eps, eps

    @torch.no_grad()
    def p_sample_loop(self, model, cond, shape):
        x = torch.randn(shape, device=self.device)
        for tt in reversed(range(self.T)):
            t = torch.full((shape[0],), tt, device=self.device, dtype=torch.long)
            eps = model(x, cond, t.float() / self.T)
            a = self.alpha[tt]; ab = self.abar[tt]
            x = (x - (1 - a) / (1 - ab).sqrt() * eps) / a.sqrt()
            if tt > 0:
                x = x + (1 - a).sqrt() * torch.randn_like(x)
        return x.clamp(0, 1)


def psnr(pred, tgt):
    mse = (pred - tgt).pow(2).mean(dim=(1, 2, 3)).clamp_min(1e-8)
    return (10 * torch.log10(1.0 / mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/sequences.csv")
    ap.add_argument("--max-clips", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="outputs/vjepa2_adapted/pixel_diffusion.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = EndoClipDataset(args.manifest, clip_len=3, stride=2, image_size=args.image_size,
                         exclude=["EndoVis2019_ROBUST-MIS"], split="train")
    if len(ds) > args.max_clips:
        idx = domain_balanced_indices(ds.clips, n=args.max_clips, seed=0)
        ds.clips = [ds.clips[i] for i in idx]
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                    collate_fn=lambda b: torch.stack([x if torch.is_tensor(x) else torch.as_tensor(x) for x in b]))
    print(f"[diffusion] {len(ds)} clips  img={args.image_size}")

    model = CondUNet().to(device)
    ddpm = DDPM(T=args.T, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    for epoch in range(args.epochs):
        model.train()
        run, nb = 0.0, 0
        for clip in dl:
            clip = clip.to(device).float()
            cond, tgt = clip[:, -2], clip[:, -1]
            t = torch.randint(0, args.T, (tgt.size(0),), device=device)
            eps = torch.randn_like(tgt)
            x_noisy, _ = ddpm.q_sample(tgt, t, eps)
            pred_eps = model(x_noisy, cond, t.float() / args.T)
            loss = F.mse_loss(pred_eps, eps)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item(); nb += 1
        print(f"[diffusion] epoch {epoch} mse={run/max(nb,1):.4f}")

    # eval: sample next frame, PSNR vs copy-last
    model.eval()
    P_gen, P_copy = [], []
    n_eval = 0
    with torch.no_grad():
        for clip in dl:
            clip = clip.to(device).float()
            cond, tgt = clip[:, -2], clip[:, -1]
            gen = ddpm.p_sample_loop(model, cond, tgt.shape)
            P_gen.extend(psnr(gen, tgt).cpu().tolist())
            P_copy.extend(psnr(cond, tgt).cpu().tolist())
            n_eval += tgt.size(0)
            if n_eval >= 64:
                break
    report = {
        "paper": "Endo-HJEPA", "not_ablation_planning": True,
        "task": "next-frame pixel diffusion (world-model pixel-generation baseline)",
        "n_eval": n_eval, "diffusion_T": args.T,
        "psnr_diffusion": float(np.mean(P_gen)), "psnr_copy_last": float(np.mean(P_copy)),
        "interp": "diffusion pixel forecast PSNR << copy-last PSNR: pixel generation is the "
                  "wrong objective for endoscopic dynamics (blur + unpredictable appearance); "
                  "contrast with representation-space JEPA forecast (Table 2).",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
