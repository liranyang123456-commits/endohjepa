"""Pixel-generation baseline: next-frame prediction in pixel space (contrast to JEPA).

Trains a small conv predictor for next-frame pixel prediction and reports PSNR/SSIM
vs a copy-last-frame baseline, plus the error split into specular vs non-specular
regions. The point: pixel prediction wastes capacity on unpredictable appearance, so
even a decent pixel model has error concentrated where the content is unpredictable —
motivating representation-space (JEPA) prediction.

    python -m endoworld.eval.pixel_baseline --max-clips 200 --epochs 3
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
from endoworld.understanding.endo_mask import specular_map


class NextFrameCNN(nn.Module):
    """Lightweight next-frame predictor: last K frames -> next frame."""

    def __init__(self, k: int = 4):
        super().__init__()
        self.k = k
        ch = 3 * k
        self.net = nn.Sequential(
            nn.Conv2d(ch, 32, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        inp = x[:, -self.k :].reshape(b, self.k * c, h, w)
        return torch.sigmoid(self.net(inp))


def psnr(pred, tgt):
    mse = (pred - tgt).pow(2).mean(dim=(1, 2, 3)).clamp_min(1e-8)
    return 10 * torch.log10(1.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/sequences.csv")
    ap.add_argument("--max-clips", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--clip-len", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="outputs/vjepa2_adapted/pixel_baseline.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = EndoClipDataset(
        args.manifest,
        clip_len=args.clip_len,
        stride=2,
        image_size=args.image_size,
        exclude=["EndoVis2019_ROBUST-MIS"],
        split="train",
    )
    if len(ds) > args.max_clips:
        idx = domain_balanced_indices(ds.clips, n=args.max_clips, seed=0)
        ds.clips = [ds.clips[i] for i in idx]
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: torch.stack(
            [x if torch.is_tensor(x) else torch.as_tensor(x) for x in b]
        ),
    )
    print(f"[pixel] {len(ds)} clips")

    model = NextFrameCNN(k=args.clip_len - 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        model.train()
        run, nb = 0.0, 0
        for clip in dl:
            clip = clip.to(device).float()
            pred = model(clip[:, :-1])
            loss = F.l1_loss(pred, clip[:, -1])
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item()
            nb += 1
        print(f"[pixel] epoch {epoch} l1={run / max(nb, 1):.4f}")

    model.eval()
    P_model, P_copy, _S_model, _S_copy = [], [], [], []
    spec_err, nonspec_err = [], []
    with torch.no_grad():
        for clip in dl:
            clip = clip.to(device).float()
            tgt = clip[:, -1]
            pred = model(clip[:, :-1])
            copy = clip[:, -2]
            P_model.extend(psnr(pred, tgt).cpu().tolist())
            P_copy.extend(psnr(copy, tgt).cpu().tolist())
            # error split by specular regions
            sm = specular_map(clip[:, -1:])[:, 0]  # (B,H,W) keep mask
            err = (pred - tgt).abs().mean(dim=1)  # (B,H,W)
            spec = 1 - sm
            # Conditional regional means; averaging masked images would confound
            # error magnitude with the much smaller specular-region area.
            spec_err.append(((err * spec).sum() / spec.sum().clamp_min(1)).item())
            nonspec_err.append(((err * sm).sum() / sm.sum().clamp_min(1)).item())
    report = {
        "paper": "Endo-HJEPA",
        "not_ablation_planning": True,
        "task": "next-frame pixel prediction (contrast to latent/JEPA prediction)",
        "n_clips": len(ds),
        "psnr_model": float(np.mean(P_model)),
        "psnr_copy_last": float(np.mean(P_copy)),
        "err_specular": float(np.mean(spec_err)),
        "err_nonspecular": float(np.mean(nonspec_err)),
        "specular_over_normal_err_ratio": float(
            np.mean(spec_err) / max(np.mean(nonspec_err), 1e-8)
        ),
        "interp": "regional errors are conditional means, normalized by mask area",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
