"""Re-evaluate the v2 continuous model on C3VD with gate-corrected poses.

C3VD is test-only (cecum_t1_a was never in training), so correcting the pose
convention changes evaluation labels, not training. Latents are unchanged;
only the SE(3) action labels are recomputed from the corrected c2w poses with
the same window-end alignment used by the v2 cache.

    python -m endoworld.eval.c3vd_reval \
        --cache outputs/physical_actions_v2/sequences.pt \
        --checkpoint outputs/continuous_actions_v2/continuous_dynamics.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from endoworld.world.c3vd_actions import (
    find_c3vd_color_frames,
    find_c3vd_pose_files,
    load_pose_txt,
)
from endoworld.world.continuous_dynamics import (
    ContinuousActionDynamics,
    ContinuousDynamicsConfig,
)
from endoworld.world.physical_actions import (
    PhysicalActionDataset,
    PhysicalSequence,
    interpolate_pose_rows,
    load_sequences,
    pose_deltas,
)
from endoworld.world.train_continuous_actions import evaluate, evaluate_fixed_bank


def corrected_c3vd_sequence(seq: PhysicalSequence, c3vd_root: Path,
                            stride: int, lookback_tubelets: int,
                            tubelet: int) -> PhysicalSequence:
    pose_files = find_c3vd_pose_files(c3vd_root)
    if len(pose_files) != 1:
        raise RuntimeError(f"expected one C3VD pose.txt, found {len(pose_files)}")
    poses = load_pose_txt(pose_files[0])  # gate-corrected convention
    paths = find_c3vd_color_frames(pose_files[0].parent)[::stride]
    digits = []
    import re
    for p in paths:
        numbers = re.findall(r"\d+", p.stem)
        digits.append(int(numbers[-1]) if numbers else -1)
    frame_indices = np.asarray(digits, dtype=np.int64)
    n_latents = seq.latents.size(0)
    warm = lookback_tubelets - 1
    end_steps = (warm + np.arange(n_latents) + 1) * tubelet - 1
    positions = frame_indices[end_steps].astype(np.float64)
    aligned = interpolate_pose_rows(poses, positions)
    actions = pose_deltas(aligned)
    return PhysicalSequence(
        sequence_id=seq.sequence_id, dataset=seq.dataset,
        latents=seq.latents, actions=torch.from_numpy(actions).float(),
        case_id=seq.case_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="outputs/physical_actions_v2/sequences.pt")
    parser.add_argument("--checkpoint",
                        default="outputs/continuous_actions_v2/continuous_dynamics.pt")
    parser.add_argument("--c3vd", default="datasets/C3VD")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--lookback-tubelets", type=int, default=8)
    parser.add_argument("--out", default="outputs/continuous_actions_v2/c3vd_corrected_eval.json")
    args = parser.parse_args()

    sequences = load_sequences(args.cache)
    c3vd = [s for s in sequences if s.dataset == "C3VD"]
    if len(c3vd) != 1:
        raise RuntimeError(f"expected one C3VD sequence, found {len(c3vd)}")
    corrected = corrected_c3vd_sequence(
        c3vd[0], Path(args.c3vd), args.stride, args.lookback_tubelets, tubelet=2)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ContinuousActionDynamics(ContinuousDynamicsConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    test = PhysicalActionDataset([corrected], 4, 4, "test")
    report = {
        "protocol": "v2 model evaluated on C3VD with reprojection-gated pose convention "
                    "(transpose + GL->CV flip); C3VD was never in training",
        "pose_gate": "docs/endohjepa/c3vd_pose_gate_gap5.json",
        "n": len(test),
        "canonical": evaluate(model, DataLoader(test, batch_size=32), device),
        "fixed_bank": evaluate_fixed_bank(model, test, device, n_negatives=10, seed=0),
    }
    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
