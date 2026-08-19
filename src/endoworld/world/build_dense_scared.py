"""Dense past-only SCARED cache for spatially supervised risk training.

The pooled v2 cache discards the spatial axis, but near-wall risk is a local
spatial signal. This builder keeps the last-tubelet dense tokens (256 x 1024)
per step, in float16, aligned to the same window-end frames as the v2 cache.

    python -m endoworld.world.build_dense_scared \
        --out outputs/physical_actions_v2/dense_scared.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from endoworld.world.scared_actions import (
    find_scared_keyframes,
    find_scared_rgb,
    load_scared_poses,
    read_video_frames,
)


@torch.no_grad()
def encode_dense_past_only(
    enc,
    frames: torch.Tensor,
    device: str,
    lookback_frames: int,
    tubelet: int,
    batch: int = 4,
):
    """Latent step t = last-tubelet dense tokens of frames ending at 2t+2."""
    n_steps = frames.size(0) // tubelet
    warm = lookback_frames // tubelet - 1
    outputs = []
    for start in range(warm, n_steps, batch):
        steps = range(start, min(start + batch, n_steps))
        part = (
            torch.stack(
                [
                    frames[(s + 1) * tubelet - lookback_frames : (s + 1) * tubelet]
                    for s in steps
                ]
            )
            .to(device)
            .float()
        )
        dense = enc.encode_dense(part)[:, -1]  # (B, N, D)
        outputs.append(dense.half().cpu())
    return torch.cat(outputs) if outputs else torch.zeros(0, 1, 1, dtype=torch.float16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scared", default="datasets/SCARED")
    parser.add_argument("--out", default="outputs/physical_actions_v2/dense_scared.pt")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=4096)
    parser.add_argument("--lookback-tubelets", type=int, default=8)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from endoworld.understanding.encoders import load_any_encoder

    enc, _, _, _ = load_any_encoder(
        "vjepa2", device, "facebook/vjepa2-vitl-fpc64-256", ""
    )
    tubelet = int(getattr(enc, "tubelet", 2))
    lookback_frames = args.lookback_tubelets * tubelet

    rows = []
    for keyframe in find_scared_keyframes(args.scared):
        video, _ = find_scared_rgb(keyframe)
        if video is None:
            continue
        poses = load_scared_poses(keyframe)
        import cv2

        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        indices = np.arange(0, min(total, len(poses)), args.stride, dtype=np.int64)[
            : args.max_frames
        ]
        frames = read_video_frames(video, indices, enc.image_size, stereo_eye="top")
        dense = encode_dense_past_only(enc, frames, device, lookback_frames, tubelet)
        case_id = next((p for p in keyframe.parts if p.startswith("dataset_")), None)
        rows.append(
            {
                "sequence_id": f"scared:{keyframe.as_posix()}",
                "case_id": case_id,
                "dense": dense,
                "n_frames": int(len(indices)),
            }
        )
        print(f"[dense-scared] {keyframe.name} -> {tuple(dense.shape)}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "rows": rows,
            "lookback_tubelets": args.lookback_tubelets,
            "stride": args.stride,
            "tubelet": tubelet,
        },
        out,
    )
    meta = {
        "n_sequences": len(rows),
        "steps": int(sum(r["dense"].size(0) for r in rows)),
        "encoding": "past-only dense last-tubelet, mono top eye, float16",
    }
    out.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"[dense-scared] saved {len(rows)} sequences -> {out}")


if __name__ == "__main__":
    main()
