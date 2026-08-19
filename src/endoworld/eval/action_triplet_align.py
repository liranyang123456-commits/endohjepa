"""Do the world model's discrete latent actions carry *semantic* action content?

Aligns the trained H-JEPA VQ action codebook with CholecT50 action-triplet labels
(<instrument, verb, target>). For each clip we take the latent action id at each
transition and the verb/triplet label at that frame, then measure NMI and a
classification probe. This tests *semantic* action grounding (distinct from the
physical camera-pose grounding in pose_latent_align).

    python -m endoworld.eval.action_triplet_align --ckpt outputs/p2000_full_causal/endohjepa.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from endoworld.data.cholect50 import (
    N_VERB,
    list_videos,
    load_video_labels,
    video_frames_dir,
)
from endoworld.data.splits import assign_split
from endoworld.world.pose_align import action_pose_nmi


@torch.no_grad()
def encode_clip(enc, frames, device):
    import numpy as np

    clip = (
        torch.from_numpy(np.stack(frames).transpose(0, 3, 1, 2))
        .unsqueeze(0)
        .to(device)
        .float()
    )
    return enc.encode_temporal(clip)[0].cpu().numpy()  # (T, D)


def build_data(enc, model, device, root, clip_len, max_per_video, max_videos):
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    videos_dir = Path(root) / "videos"
    act_ids, verb_labels, vid_list = [], [], []
    vids = list_videos()[:max_videos]
    for vid in vids:
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
            imgs = []
            ok = True
            for i in idxs:
                try:
                    im = (
                        Image.open(frames[i])
                        .convert("RGB")
                        .resize((enc.image_size, enc.image_size))
                    )
                    imgs.append(np.asarray(im, np.float32) / 255.0)
                except Exception:
                    ok = False
                    break
            if not ok:
                continue
            z = encode_clip(enc, imgs, device)  # (T, D)
            if len(z) < 2:
                continue
            zt = torch.from_numpy(z).to(device).float()
            ids, _, _ = model.actions(zt[:-1], zt[1:])  # (T-1,) latent action ids
            ids = ids.cpu().numpy()
            # verb label per transition (use the target frame's verb)
            for t in range(len(ids)):
                frame_idx = idxs[min(t + 1, len(idxs) - 1)]
                lab = labels.get(frame_idx)
                if lab is None:
                    continue
                v = lab["verb"]
                if v.sum() > 0:
                    act_ids.append(int(ids[t]))
                    verb_labels.append(int(v.argmax()))
                    vid_list.append(vid)
    return np.array(act_ids), np.array(verb_labels), vid_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/p2000_full_causal/endohjepa.pt")
    ap.add_argument("--root", default="datasets/CholecT50/CholecT50")
    ap.add_argument("--vjepa2-id", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--clip-len", type=int, default=8)
    ap.add_argument("--max-per-video", type=int, default=8)
    ap.add_argument("--max-videos", type=int, default=16)
    ap.add_argument(
        "--out", default="outputs/p2000_full_causal/action_triplet_align.json"
    )
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from endoworld.understanding.vjepa2_hf import VJEPA2Encoder
    from endoworld.eval.world_benchmark import load_predictor

    enc = VJEPA2Encoder(args.vjepa2_id, device=device)
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    model, kind, _, _, _ = load_predictor(blob, device)
    if kind != "hjepa":
        raise SystemExit("need an H-JEPA checkpoint (has action codebook)")

    act_ids, verb_labels, vids = build_data(
        enc,
        model,
        device,
        args.root,
        args.clip_len,
        args.max_per_video,
        args.max_videos,
    )
    print(
        f"[triplet-align] {len(act_ids)} transitions, {len(set(vids))} videos, "
        f"{len(np.unique(act_ids))} active actions, {len(np.unique(verb_labels))} verbs"
    )
    if len(act_ids) < 20:
        print("[triplet-align] too few transitions")
        return
    n_act = max(int(act_ids.max()) + 1, 2)
    nmi = action_pose_nmi(act_ids, verb_labels)
    rng = np.random.default_rng(0)
    nmi_rand = action_pose_nmi(rng.integers(0, n_act, size=len(act_ids)), verb_labels)
    # video-level classification probe: latent action id -> verb
    tr = [
        i
        for i, v in enumerate(vids)
        if assign_split(f"t50::{v}", train=0.8, val=0.0) == "train"
    ]
    te = [
        i
        for i, v in enumerate(vids)
        if assign_split(f"t50::{v}", train=0.8, val=0.0) != "train"
    ]
    if not te:
        te = tr[-max(1, len(tr) // 5) :]
    # one-hot action id -> verb logistic
    A = F_onehot(act_ids, n_act)
    W = np.zeros((n_act, N_VERB))
    for c in range(N_VERB):
        W[:, c] = np.log((A[tr].T @ (verb_labels[tr] == c)) + 1)
    pred = (A[te] @ W).argmax(1)
    acc = float((pred == verb_labels[te]).mean())
    # chance = most common verb
    chance = float(np.bincount(verb_labels[tr]).max() / len(tr))

    report = {
        "paper": "Endo-HJEPA",
        "not_ablation_planning": True,
        "task": "latent-action -> semantic verb grounding (CholecT50)",
        "n_transitions": int(len(act_ids)),
        "n_videos": len(set(vids)),
        "n_actions_active": int(n_act),
        "nmi_action_verb": float(nmi),
        "nmi_random": float(nmi_rand),
        "verb_probe_acc": acc,
        "verb_chance": chance,
        "note": "semantic action grounding (distinct from physical pose grounding)",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def F_onehot(x, k):
    out = np.zeros((len(x), k), dtype=np.float32)
    out[np.arange(len(x)), x] = 1.0
    return out


if __name__ == "__main__":
    main()
