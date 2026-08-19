"""Encoder-level action supervision for latent-action grounding (the real fix).

The self-supervised negative result (eval/action_triplet_align.py) showed residual
dynamics don't encode surgical-verb semantics, and codebook-level supervision can't
fix that (world/grounded_finetune.py). Here we go one level up: fine-tune the V-JEPA 2
encoder's last block with a per-frame verb-classification loss on CholecT50, so the
latents *themselves* encode action semantics, then re-measure residual->verb
alignment. This tests whether encoder-level action supervision grounds latent actions.

    python -m endoworld.world.encoder_action_grounding
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from endoworld.data.cholect50 import (
    N_VERB,
    list_videos,
    load_video_labels,
    video_frames_dir,
)
from endoworld.world.pose_align import action_pose_nmi, quantise_deltas


def _load_frame(path, image_size):
    from PIL import Image

    return (
        np.asarray(
            Image.open(path).convert("RGB").resize((image_size, image_size)), np.float32
        )
        / 255.0
    )


def build_frame_data(root, max_videos, max_per_video):
    """Per-frame (image, verb_label, vid)."""
    videos_dir = Path(root) / "videos"
    imgs, verbs, vids = [], [], []
    for vid in list_videos()[:max_videos]:
        vdir = video_frames_dir(vid, videos_dir)
        if not vdir.is_dir():
            continue
        labels = load_video_labels(vid)
        frames = sorted(vdir.glob("*.png"))
        step = max(1, len(frames) // max_per_video)
        for i in range(0, len(frames), step):
            lab = labels.get(i)
            if lab is None or lab["verb"].sum() == 0:
                continue
            imgs.append(frames[i])
            verbs.append(int(lab["verb"].argmax()))
            vids.append(vid)
    return imgs, np.array(verbs), vids


def residual_verb_alignment(
    enc, device, root, clip_len, max_videos, max_per_video, n_actions=10
):
    """Residual->verb NMI + probe using the encoder's latent residuals."""
    videos_dir = Path(root) / "videos"
    res_list, verb_list, vid_list = [], [], []
    for vid in list_videos()[:max_videos]:
        vdir = video_frames_dir(vid, videos_dir)
        if not vdir.is_dir():
            continue
        labels = load_video_labels(vid)
        frames = sorted(vdir.glob("*.png"))
        n = len(frames)
        if n < clip_len + 1:
            continue
        starts = np.linspace(
            0, n - clip_len - 1, min(max_per_video, max(1, n // (clip_len * 4)))
        ).astype(int)
        for s in starts:
            idxs = [min(s + i, n - 1) for i in range(clip_len + 1)]
            try:
                imgs = [_load_frame(frames[i], enc.image_size) for i in idxs]
            except Exception:
                continue
            clip = torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2)).unsqueeze(0)
            with torch.no_grad():
                z = enc.encode_temporal(clip.to(device).float())[0].cpu().numpy()
            for t in range(len(z) - 1):
                fi = idxs[min(t + 1, len(idxs) - 1)]
                lab = labels.get(fi)
                if lab is None or lab["verb"].sum() == 0:
                    continue
                res_list.append(z[t + 1] - z[t])
                verb_list.append(int(lab["verb"].argmax()))
                vid_list.append(vid)
    if len(res_list) < 20:
        return None
    R = np.stack(res_list)
    V = np.array(verb_list)
    ids = quantise_deltas(R, n_actions)
    nmi = action_pose_nmi(ids, V)
    rng = np.random.default_rng(0)
    nmi_rand = action_pose_nmi(rng.integers(0, n_actions, size=len(ids)), V)
    return {
        "nmi_residual_verb": float(nmi),
        "nmi_random": float(nmi_rand),
        "n_transitions": len(R),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/CholecT50/CholecT50")
    ap.add_argument("--vjepa2-id", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--unfreeze-last", type=int, default=1)
    ap.add_argument("--max-videos", type=int, default=32)
    ap.add_argument("--max-per-video", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument(
        "--out", default="outputs/vjepa2_adapted/encoder_action_grounding.json"
    )
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from endoworld.understanding.vjepa2_hf import VJEPA2Encoder

    enc = VJEPA2Encoder(args.vjepa2_id, device=device)

    # BEFORE: frozen-encoder residual->verb alignment
    before = residual_verb_alignment(enc, device, args.root, 8, 16, 6)
    print(f"[action-ground] before (frozen): {before}")

    # Fine-tune encoder last block with per-frame verb supervision
    imgs, verbs, vids = build_frame_data(args.root, args.max_videos, args.max_per_video)
    print(f"[action-ground] {len(imgs)} frames for verb supervision")
    if not imgs:
        return
    enc2 = VJEPA2Encoder(
        args.vjepa2_id, device=device, unfreeze_last=args.unfreeze_last
    )
    head = nn.Linear(enc2.embed_dim, N_VERB).to(device)
    params = enc2.trainable_parameters() + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    enc2.train()
    V = torch.from_numpy(verbs).to(device)
    for epoch in range(args.epochs):
        perm = torch.randperm(len(imgs))
        run, nb = 0.0, 0
        for i in range(0, len(imgs), 8):
            sl = perm[i : i + 8]
            xb = np.stack([_load_frame(imgs[j], enc2.image_size) for j in sl])
            # (B,1,C,H,W) single-frame "clips"
            xb = (
                torch.from_numpy(xb.transpose(0, 3, 1, 2))
                .unsqueeze(1)
                .to(device)
                .float()
            )
            z = enc2.encode(xb)  # (B, D)
            loss = F.cross_entropy(head(z), V[sl])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item()
            nb += 1
        print(f"[action-ground] epoch {epoch} verb loss={run / max(nb, 1):.4f}")
    enc2.eval()

    after = residual_verb_alignment(enc2, device, args.root, 8, 16, 6)
    print(f"[action-ground] after (encoder-supervised): {after}")
    report = {
        "paper": "Endo-HJEPA",
        "not_ablation_planning": True,
        "task": "encoder-level action supervision for latent-action grounding",
        "before_frozen": before,
        "after_encoder_supervised": after,
        "improvement_nmi": (after["nmi_residual_verb"] - before["nmi_residual_verb"])
        if before and after
        else None,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
