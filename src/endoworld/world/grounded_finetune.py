"""Supervised grounding of the world model's latent actions (CholecT50 verbs).

The emergent VQ latent actions are not semantically grounded without supervision
(shown in eval/action_triplet_align.py: below-chance verb NMI). Here we add a light
classification head on the VQ action embedding and fine-tune the *codebook + head*
on CholecT50 verb labels, so discrete actions become semantically meaningful —
without touching the forecast/planning objective.

    python -m endoworld.world.grounded_finetune --ckpt outputs/p2000_full_causal/endohjepa.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from endoworld.data.cholect50 import N_VERB, list_videos, load_video_labels, video_frames_dir
from endoworld.data.splits import assign_split
from endoworld.world.pose_align import action_pose_nmi


def build_transitions(enc, device, root, clip_len, max_per_video, max_videos):
    """Encode CholecT50 clips -> per-transition (z_t, z_{t+1}, verb_label, vid)."""
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    videos_dir = Path(root) / "videos"
    Z0, Z1, V, VIDS = [], [], [], []
    for vid in list_videos()[:max_videos]:
        vdir = video_frames_dir(vid, videos_dir)
        if not vdir.is_dir():
            continue
        labels = load_video_labels(vid)
        frames = sorted(vdir.glob("*.png"))
        n = len(frames)
        if n < clip_len + 1:
            continue
        starts = np.linspace(0, n - clip_len - 1, min(max_per_video, max(1, n // (clip_len * 4)))).astype(int)
        for s in starts:
            idxs = [min(s + i, n - 1) for i in range(clip_len + 1)]
            imgs = []
            ok = True
            for i in idxs:
                try:
                    im = Image.open(frames[i]).convert("RGB").resize((enc.image_size, enc.image_size))
                    imgs.append(np.asarray(im, np.float32) / 255.0)
                except Exception:
                    ok = False
                    break
            if not ok:
                continue
            clip = torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2)).unsqueeze(0)
            with torch.no_grad():
                z = enc.encode_temporal(clip.to(device).float())[0].cpu()  # (T, D)
            for t in range(len(z) - 1):
                frame_idx = idxs[min(t + 1, len(idxs) - 1)]
                lab = labels.get(frame_idx)
                if lab is None or lab["verb"].sum() == 0:
                    continue
                Z0.append(z[t]); Z1.append(z[t + 1])
                V.append(int(lab["verb"].argmax())); VIDS.append(vid)
    if not Z0:
        return None
    return torch.stack(Z0), torch.stack(Z1), torch.tensor(V), VIDS


def grounding_metric(model, Z0, Z1, V, vids, device):
    """NMI + video-level verb probe from latent action ids."""
    with torch.no_grad():
        ids, _, _ = model.actions(Z0.to(device), Z1.to(device))
    ids = ids.cpu().numpy()
    v = V.numpy()
    nmi = action_pose_nmi(ids, v)
    rng = np.random.default_rng(0)
    nmi_rand = action_pose_nmi(rng.integers(0, int(ids.max()) + 1, size=len(ids)), v)
    tr = [i for i, vv in enumerate(vids) if assign_split(f"t50::{vv}", train=0.8, val=0.0) == "train"]
    te = [i for i, vv in enumerate(vids) if assign_split(f"t50::{vv}", train=0.8, val=0.0) != "train"]
    if not te:
        te = tr[-max(1, len(tr) // 5):]
    n_act = int(ids.max()) + 1
    A = np.zeros((len(ids), n_act), dtype=np.float32)
    A[np.arange(len(ids)), ids] = 1.0
    W = np.stack([np.log(A[tr].T @ (v[tr] == c) + 1) for c in range(N_VERB)], 1)
    acc = float(((A[te] @ W).argmax(1) == v[te]).mean()) if te else float("nan")
    chance = float(np.bincount(v[tr]).max() / len(tr))
    return {"nmi": float(nmi), "nmi_random": float(nmi_rand),
            "verb_probe_acc": acc, "verb_chance": chance}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/p2000_full_causal/endohjepa.pt")
    ap.add_argument("--root", default="datasets/CholecT50/CholecT50")
    ap.add_argument("--vjepa2-id", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--clip-len", type=int, default=8)
    ap.add_argument("--max-per-video", type=int, default=8)
    ap.add_argument("--max-videos", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="outputs/p2000_full_causal/grounded_actions.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from endoworld.understanding.vjepa2_hf import VJEPA2Encoder
    from endoworld.eval.world_benchmark import load_predictor
    enc = VJEPA2Encoder(args.vjepa2_id, device=device)
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    model, kind, _, _, _ = load_predictor(blob, device)
    if kind != "hjepa":
        raise SystemExit("need H-JEPA ckpt")

    data = build_transitions(enc, device, args.root, args.clip_len, args.max_per_video, args.max_videos)
    if data is None:
        print("[ground] no transitions")
        return
    Z0, Z1, V, VIDS = data
    print(f"[ground] {len(Z0)} transitions, {len(set(VIDS))} videos")

    before = grounding_metric(model, Z0, Z1, V, VIDS, device)
    print(f"[ground] before: {before}")

    # grounding head on the quantised action embedding
    dim = model.actions.codebook.weight.size(1)
    verb_head = nn.Linear(dim, N_VERB).to(device)
    params = list(model.actions.parameters()) + list(verb_head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    Z0, Z1, V = Z0.to(device), Z1.to(device), V.to(device)
    n = len(Z0)
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        run, nb = 0.0, 0
        for i in range(0, n, 256):
            sl = perm[i:i + 256]
            idx, quant, commit = model.actions(Z0[sl], Z1[sl])
            logits = verb_head(quant)
            loss = F.cross_entropy(logits, V[sl]) + 0.1 * commit
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item(); nb += 1
        print(f"[ground] epoch {epoch} loss={run/max(nb,1):.4f}")

    after = grounding_metric(model, Z0.cpu(), Z1.cpu(), V.cpu(), VIDS, device)
    print(f"[ground] after: {after}")
    report = {"paper": "Endo-HJEPA", "not_ablation_planning": True,
              "task": "supervised latent-action grounding (CholecT50 verbs)",
              "before": before, "after": after,
              "improvement_nmi": after["nmi"] - before["nmi"],
              "improvement_probe": after["verb_probe_acc"] - before["verb_probe_acc"]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    # save the grounded codebook
    torch.save({"actions": model.actions.state_dict(), "verb_head": verb_head.state_dict()},
               Path(args.out).parent / "grounded_actions.pt")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
