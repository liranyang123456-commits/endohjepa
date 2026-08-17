"""Attach SCARED near-wall proximity labels to a physical-action cache.

For each SCARED sequence, per-frame 5th-percentile scene depth (mm) is read
once from scene_points.tar.gz and mapped to latent steps. In the v2 past-only
cache, latent step k ends at original video frame 4k+30 (stride 2 sampling,
8-tubelet look-back, tubelet 2). C3VD sequences are left unlabelled (the
reprojection gate covers pose; depth there uses a different convention).

    python -m endoworld.world.attach_scared_risk \
        --cache outputs/physical_actions_v2/sequences.pt \
        --out outputs/physical_actions_v2/sequences_risk.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from endoworld.eval.scared_collision import _depth_proximities
from endoworld.world.physical_actions import (
    PhysicalSequence,
    load_sequences,
    save_sequences,
)


def attach(cache: str, out: str, scared_root: str, stride: int,
           lookback_tubelets: int, tubelet: int) -> dict:
    sequences = load_sequences(cache)
    warm = lookback_tubelets - 1
    labelled, missing = 0, []
    for seq in sequences:
        if seq.dataset != "SCARED":
            continue
        keyframe = Path(seq.sequence_id.split("scared:", 1)[1])
        if not keyframe.is_absolute():
            keyframe = Path(scared_root) / keyframe
        proximities = _depth_proximities(keyframe)
        if not proximities:
            missing.append(seq.sequence_id)
            continue
        keys = sorted(proximities)
        values = []
        for k in range(seq.latents.size(0)):
            frame = (warm + k + 1) * tubelet * stride - stride + tubelet - 1
            nearest = min(keys, key=lambda x: abs(x - frame))
            values.append(proximities[nearest])
        seq.depth_or_risk = torch.tensor(values, dtype=torch.float32).unsqueeze(-1)
        labelled += 1
    save_sequences(sequences, out)
    report = {
        "cache": cache, "out": out,
        "label": "5th-percentile scene depth (mm) per latent step",
        "frame_mapping": f"latent k -> video frame {(warm + 1) * tubelet * stride - stride + tubelet - 1} + {tubelet * stride}k",
        "n_scared_labelled": labelled,
        "n_scared_missing": len(missing),
        "missing": missing,
    }
    Path(out).with_suffix(".risk.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="outputs/physical_actions_v2/sequences.pt")
    parser.add_argument("--out", default="outputs/physical_actions_v2/sequences_risk.pt")
    parser.add_argument("--scared", default="E:/MIS_Datasets/SCARED")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--lookback-tubelets", type=int, default=8)
    parser.add_argument("--tubelet", type=int, default=2)
    args = parser.parse_args()
    attach(args.cache, args.out, args.scared, args.stride,
           args.lookback_tubelets, args.tubelet)


if __name__ == "__main__":
    main()
