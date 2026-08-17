"""V-JEPA self-supervised pretraining loop.

Examples:
    # quick smoke test (tiny model, few steps, CPU/GPU)
    python -m endoworld.understanding.pretrain --smoke

    # real run
    python -m endoworld.understanding.pretrain \
        --manifest manifests/sequences.csv --epochs 100 --batch-size 16
"""
from __future__ import annotations

import argparse
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from endoworld.data.video_clips import EndoClipDataset
from endoworld.understanding.vjepa import VJEPA, VJEPAConfig


def cosine_lr(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * p))


def collate(batch):
    return torch.stack([b if torch.is_tensor(b) else torch.as_tensor(b) for b in batch])


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    cfg = VJEPAConfig(
        image_size=args.image_size, clip_len=args.clip_len,
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.heads,
        mask_ratio=args.mask_ratio,
    )
    if args.smoke:
        cfg = VJEPAConfig(image_size=64, clip_len=8, embed_dim=128, depth=2,
                          num_heads=4, predictor_dim=96, predictor_depth=2)

    ds = EndoClipDataset(args.manifest, clip_len=cfg.clip_len,
                         stride=args.stride, image_size=cfg.image_size,
                         exclude=["EndoVis2019_ROBUST-MIS"],
                         split=None if args.smoke else args.split)
    print(f"[data] {len(ds)} clips  split={getattr(args, 'split', None)}")
    if args.smoke:
        ds.clips = ds.clips[: args.smoke_clips]
        print(f"[smoke] using {len(ds.clips)} clips")
    elif args.max_clips and len(ds.clips) > args.max_clips:
        import random as _r
        _r.seed(0)
        ds.clips = _r.sample(ds.clips, args.max_clips)
        print(f"[data] subsampled to {len(ds.clips)} clips")

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, collate_fn=collate, drop_last=True)

    model = VJEPA(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] VJEPA {n_params:.1f}M params, tokens/clip={cfg.n_tokens}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.05)
    total_steps = args.epochs * max(len(dl), 1)
    warmup = max(int(0.05 * total_steps), 1)
    os.makedirs(args.out, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        run = 0.0
        for clip in dl:
            clip = clip.to(device, non_blocking=True).float()
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total_steps, args.lr, warmup)
            weights = None
            if args.endo_mask:
                from endoworld.understanding.endo_mask import token_loss_weights
                weights = token_loss_weights(clip, cfg.tubelet_size, cfg.patch_size).to(clip.device)
            loss = model(clip, token_weights=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.update_target()
            run += loss.item()
            step += 1
            if step % args.log_every == 0:
                print(f"  epoch {epoch} step {step} loss {loss.item():.4f} "
                      f"lr {opt.param_groups[0]['lr']:.2e}")
        avg = run / max(len(dl), 1)
        print(f"[epoch {epoch}] avg_loss={avg:.4f} time={time.time()-t0:.1f}s")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt = os.path.join(args.out, f"vjepa_epoch{epoch+1}.pt")
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ckpt)
            print(f"[ckpt] {ckpt}")

    if args.smoke:
        print("[smoke] OK - forward/backward/EMA/save all ran")


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/sequences.csv")
    ap.add_argument("--out", default="outputs/vjepa")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--embed-dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--max-clips", type=int, default=None,
                    help="subsample the clip index to at most this many clips")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-clips", type=int, default=8)
    ap.add_argument("--split", default="train")
    ap.add_argument("--endo-mask", action="store_true")
    return ap


if __name__ == "__main__":
    train(build_argparser().parse_args())
