"""Qualitative real-frame retrieval for the continuous SE(3) latent predictor.

The model has no pixel decoder. Therefore the visual result is the real frame
whose cached V-JEPA2 latent is nearest to the predicted terminal latent within
a local temporal neighbourhood. It must be labelled retrieval, not generation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from endoworld.world.continuous_dynamics import (  # noqa: E402
    ContinuousActionDynamics,
    ContinuousDynamicsConfig,
)
from endoworld.world.physical_actions import (  # noqa: E402
    PhysicalActionDataset,
    load_sequences,
)
from endoworld.world.scared_actions import find_scared_rgb  # noqa: E402


def _read_frame(sequence_id: str, latent_index: int) -> np.ndarray:
    keyframe = Path(sequence_id.split("scared:", 1)[1])
    video, _ = find_scared_rgb(keyframe)
    if video is None:
        raise FileNotFoundError(f"no rgb.mp4 for {keyframe}")
    # v2 past-only cache: latent k ends at sampled frame 2k+15 (stride 2,
    # 8-tubelet look-back), i.e. original video frame 4k+30.
    frame_index = 4 * latent_index + 30
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, bgr = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_index} from {video}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # SCARED rgb.mp4 stores left/right views vertically; use the left/top view.
    if rgb.shape[0] >= 1.5 * rgb.shape[1]:
        rgb = rgb[:rgb.shape[0] // 2]
    return rgb


def _nearest_local(sequence, prediction, current, radius=16):
    stop = min(current + radius + 1, len(sequence.latents))
    candidates = sequence.latents[current:stop]
    return current + int(torch.cdist(
        prediction[None].cpu(), candidates).argmin())


def main():
    data_path = ROOT / "outputs" / "physical_actions_v2" / "sequences.pt"
    checkpoint_path = (
        ROOT / "outputs" / "continuous_actions_v2"
        / "continuous_dynamics.pt"
    )
    output = Path(__file__).resolve().parent / "figures" / "figure8_qualitative.pdf"
    sequences = load_sequences(data_path)
    dataset = PhysicalActionDataset(sequences, history=4, horizon=4, split="test")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ContinuousActionDynamics(
        ContinuousDynamicsConfig(**checkpoint["config"]))
    # The counterfactual checkpoint predates the zero-initialised
    # action_delta_head; strict=False leaves it at zero, reproducing the
    # reported model exactly.
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    candidates = []
    generator = torch.Generator().manual_seed(7)
    scared_indices = [
        i for i, (sequence_index, _) in enumerate(dataset.windows)
        if dataset.sequences[sequence_index].dataset == "SCARED"
    ]
    with torch.no_grad():
        for offset in range(0, len(scared_indices), 64):
            indices = scared_indices[offset:offset + 64]
            history = torch.stack([dataset[i]["history"] for i in indices])
            actions = torch.stack([dataset[i]["actions"] for i in indices])
            future = torch.stack([dataset[i]["future"] for i in indices])
            real_prediction = model(history, actions)
            permutation = torch.randperm(len(indices), generator=generator)
            shuffled_prediction = model(history, actions[permutation])
            real_error = (real_prediction - future).square().mean(dim=(1, 2))
            shuffled_error = (
                shuffled_prediction - future).square().mean(dim=(1, 2))
            for j, dataset_index in enumerate(indices):
                candidates.append({
                    "index": dataset_index,
                    "gain": float(shuffled_error[j] - real_error[j]),
                    "real_error": float(real_error[j]),
                    "shuffled_error": float(shuffled_error[j]),
                    "real_prediction": real_prediction[j, -1].cpu(),
                    "shuffled_prediction": shuffled_prediction[j, -1].cpu(),
                })

    eligible = []
    for result in candidates:
        sequence_index, start = dataset.windows[result["index"]]
        sequence = dataset.sequences[sequence_index]
        current = start + dataset.history - 1
        goal = current + dataset.horizon
        result["real_nearest"] = _nearest_local(
            sequence, result["real_prediction"], current)
        result["shuffled_nearest"] = _nearest_local(
            sequence, result["shuffled_prediction"], current)
        if (
            result["gain"] > 0
            and abs(result["real_nearest"] - goal)
            < abs(result["shuffled_nearest"] - goal)
        ):
            eligible.append(result)
    eligible.sort(key=lambda row: row["gain"])
    if len(eligible) < 3:
        raise RuntimeError("fewer than three positive real-vs-shuffled examples")
    # Deterministic representative examples, not the three maxima.
    selected = [eligible[int(q * (len(eligible) - 1))] for q in (0.25, 0.5, 0.75)]

    fig, axes = plt.subplots(3, 5, figsize=(13.2, 7.4))
    titles = [
        "Last observed frame",
        "Ground-truth future",
        "SE(3)-conditioned retrieval",
        "Shuffled-action retrieval",
        "Shuffled retrieval error",
    ]
    for axis, title in zip(axes[0], titles):
        axis.set_title(title, fontsize=10, fontweight="bold")

    for row_index, result in enumerate(selected):
        sequence_index, start = dataset.windows[result["index"]]
        sequence = dataset.sequences[sequence_index]
        current = start + dataset.history - 1
        goal = current + dataset.horizon
        real_nearest = result["real_nearest"]
        shuffled_nearest = result["shuffled_nearest"]
        observed_image = _read_frame(sequence.sequence_id, current)
        goal_image = _read_frame(sequence.sequence_id, goal)
        real_image = _read_frame(sequence.sequence_id, real_nearest)
        shuffled_image = _read_frame(sequence.sequence_id, shuffled_nearest)
        shuffled_resized = cv2.resize(
            shuffled_image, (goal_image.shape[1], goal_image.shape[0]))
        error = np.abs(
            shuffled_resized.astype(np.float32) - goal_image.astype(np.float32)
        ).mean(axis=-1)
        panels = [observed_image, goal_image, real_image, shuffled_image, error]
        for column, (axis, panel) in enumerate(zip(axes[row_index], panels)):
            if column == 4:
                axis.imshow(panel, cmap="magma", vmin=0, vmax=96)
            else:
                axis.imshow(panel)
            axis.axis("off")
        axes[row_index, 0].set_ylabel(
            f"Case {row_index + 1}\nreal/shuf MSE\n"
            f"{result['real_error']:.3f}/{result['shuffled_error']:.3f}",
            fontsize=9,
        )
        axes[row_index, 2].text(
            0.02, 0.03, f"retrieved t+{real_nearest-current}",
            transform=axes[row_index, 2].transAxes, color="white", fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
        )
        axes[row_index, 3].text(
            0.02, 0.03, f"retrieved t+{shuffled_nearest-current}",
            transform=axes[row_index, 3].transAxes, color="white", fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
        )
    fig.suptitle(
        "Real SCARED qualitative retrieval (representative positive-action cases)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    print(f"[qualitative] wrote {output}")


if __name__ == "__main__":
    main()
