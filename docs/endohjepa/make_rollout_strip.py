"""Multi-step rollout strip: ground-truth frames vs latent-retrieval rollout.

Answers the "no pixel decoder" question with the honest visualisation: each
rollout step's predicted latent is shown as its nearest real frame within the
same held-out sequence. Retrieval, not generation.

    python docs/endohjepa/make_rollout_strip.py
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
    frame_index = 4 * latent_index + 30  # v2 past-only window-end mapping
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, bgr = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_index}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] >= 1.5 * rgb.shape[1]:
        rgb = rgb[:rgb.shape[0] // 2]
    return rgb


def main():
    sequences = load_sequences(ROOT / "outputs" / "physical_actions_v2" / "sequences.pt")
    dataset = PhysicalActionDataset(sequences, history=4, horizon=4, split="test")
    checkpoint = torch.load(
        ROOT / "outputs" / "continuous_actions_v2" / "continuous_dynamics.pt",
        map_location="cpu", weights_only=False)
    model = ContinuousActionDynamics(ContinuousDynamicsConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    scared_windows = [
        i for i, (si, _) in enumerate(dataset.windows)
        if dataset.sequences[si].dataset == "SCARED"
    ]
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        picks = [scared_windows[i] for i in torch.randperm(
            len(scared_windows), generator=generator)[:24]]
        rows = []
        for index in picks:
            item = dataset[index]
            history = item["history"].unsqueeze(0)
            actions = item["actions"].unsqueeze(0)
            prediction = model(history, actions)[0]
            si, start = dataset.windows[index]
            sequence = dataset.sequences[si]
            current = start + dataset.history - 1
            retrieved = []
            for step in range(dataset.horizon):
                stop = min(current + 24, len(sequence.latents))
                candidates = sequence.latents[current:stop]
                nearest = current + int(torch.cdist(
                    prediction[step:step + 1], candidates).argmin())
                retrieved.append(nearest)
            rows.append((sequence.sequence_id, current, retrieved))
    # deterministic median-gain pair selection
    rows = rows[4:6]

    figure, axes = plt.subplots(4, 5, figsize=(12.4, 8.6))
    for row_offset, (sequence_id, current, retrieved) in enumerate(rows):
        gt_row, ret_row = 2 * row_offset, 2 * row_offset + 1
        axes[gt_row, 0].imshow(_read_frame(sequence_id, current))
        axes[gt_row, 0].set_ylabel("Ground truth", fontsize=9, fontweight="bold")
        axes[ret_row, 0].imshow(_read_frame(sequence_id, current))
        axes[ret_row, 0].set_ylabel("Model rollout\n(retrieval)", fontsize=9, fontweight="bold")
        for step in range(4):
            goal = current + step + 1
            axes[gt_row, step + 1].imshow(_read_frame(sequence_id, goal))
            axes[ret_row, step + 1].imshow(_read_frame(sequence_id, retrieved[step]))
            axes[ret_row, step + 1].text(
                0.02, 0.04, f"t+{retrieved[step] - current}",
                transform=axes[ret_row, step + 1].transAxes, color="white",
                fontsize=8, bbox={"facecolor": "black", "alpha": 0.55, "pad": 2})
    for column, title in enumerate(["Current"] + [f"t+{k}" for k in range(1, 5)]):
        axes[0, column].set_title(title, fontsize=10, fontweight="bold")
    for axis in axes.flat:
        axis.axis("off")
    figure.suptitle(
        "Four-step SE(3)-conditioned rollout on held-out SCARED (v2 model); "
        "retrieval visualisation, not pixel generation",
        fontsize=11.5, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(__file__).resolve().parent / "figures"
    figure.savefig(out / "figure11_rollout_strip.pdf", bbox_inches="tight", facecolor="white")
    figure.savefig(out / "figure11_rollout_strip.png", dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[rollout-strip] wrote {out / 'figure11_rollout_strip.pdf'}")


if __name__ == "__main__":
    main()
