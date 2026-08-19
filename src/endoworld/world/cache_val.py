"""Augment an existing latents_cache.pt with video-level val latents.

Encodes only the val split (cheap) and adds Z_val / D_val so that cache-reusing
ablation runs evaluate on the same video-level val split as a fresh run.

    python -m endoworld.world.cache_val --cache outputs/endohjepa_v2_full/latents_cache.pt
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from endoworld.data.video_clips import EndoClipDataset, domain_balanced_indices
from endoworld.world.train import cache_latents, collate_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--manifest", default="manifests/sequences.csv")
    ap.add_argument(
        "--max-clips",
        type=int,
        default=320,
        help="must match the training run that produced the cache",
    )
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--vjepa2-id", default="facebook/vjepa2-vitl-fpc64-256")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pack = torch.load(args.cache, map_location="cpu", weights_only=False)

    from endoworld.understanding.vjepa2_hf import VJEPA2Encoder

    enc = VJEPA2Encoder(args.vjepa2_id, device=device)
    # match the temporal length of the cached train latents (T = clip_len / tubelet)
    t_cached = int(pack["Z"].size(1))
    clip_len = t_cached * int(getattr(enc, "tubelet", 2))
    image_size = enc.image_size
    print(f"[cache-val] train T={t_cached} -> clip_len={clip_len}")

    ds_va = EndoClipDataset(
        args.manifest,
        clip_len=clip_len,
        stride=4,
        image_size=image_size,
        exclude=["EndoVis2019_ROBUST-MIS"],
        return_meta=True,
        split="val",
    )
    n_va = max(8, args.max_clips // 8)
    idx = domain_balanced_indices(ds_va.clips, n=min(n_va, len(ds_va.clips)), seed=1)
    ds_va.clips = [ds_va.clips[i] for i in idx]
    dl_va = DataLoader(
        ds_va,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_meta,
    )
    Z_val, D_val = cache_latents(enc, dl_va, device, pack.get("dense", True))
    pack["Z_val"], pack["D_val"] = Z_val, D_val
    torch.save(pack, args.cache)
    print(f"[cache-val] added Z_val {tuple(Z_val.shape)} to {args.cache}")


if __name__ == "__main__":
    main()
