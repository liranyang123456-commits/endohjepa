"""Multi-step rollout on held-out SCARED: qualitative strip plus aggregate drift.

Endo-HJEPA has no pixel decoder, so a rollout step is visualised by the nearest
real frame of the same held-out sequence (retrieval, not generation). Two
consequences follow and both are shown here.

First, consecutive SCARED frames are visually almost identical, so a strip of
RGB panels alone cannot demonstrate anything. Each retrieved panel is therefore
annotated with the temporal index it recovered, and the ground-truth row is
annotated with the index it should have recovered, which is the quantity the
reader can actually compare.

Second, a single pair of examples cannot support a claim about rollout
behaviour. The right-hand panel therefore reports the mean recovered index per
horizon over every held-out SCARED window, against the ground truth and against
the persistence retrieval, which makes the systematic lag of the rollout
visible instead of leaving it to be inferred from two strips.

Cases are selected by the mean absolute index error of their rollout, at the
25th and 50th percentile of the held-out distribution, so they are
representative rather than favourable.

    python docs/endohjepa/make_rollout_strip.py
"""

from __future__ import annotations

import hashlib
import json
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

SEARCH_RADIUS = 24
INK = "#2D3741"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        rgb = rgb[: rgb.shape[0] // 2]
    return rgb


def rollout_indices(model, dataset, index: int) -> tuple[str, int, list[int], int]:
    """Recovered temporal index per horizon, plus the persistence retrieval."""
    item = dataset[index]
    with torch.no_grad():
        prediction = model(item["history"].unsqueeze(0), item["actions"].unsqueeze(0))[
            0
        ]
    sequence_index, start = dataset.windows[index]
    sequence = dataset.sequences[sequence_index]
    current = start + dataset.history - 1
    stop = min(current + SEARCH_RADIUS, len(sequence.latents))
    candidates = sequence.latents[current:stop]
    retrieved = [
        int(torch.cdist(prediction[step : step + 1], candidates).argmin())
        for step in range(dataset.horizon)
    ]
    persistence = int(
        torch.cdist(sequence.latents[current : current + 1], candidates).argmin()
    )
    return sequence.sequence_id, current, retrieved, persistence


def main():
    sequences = load_sequences(
        ROOT / "outputs" / "physical_actions_v2" / "sequences.pt"
    )
    dataset = PhysicalActionDataset(sequences, history=4, horizon=4, split="test")
    checkpoint = torch.load(
        ROOT / "outputs" / "continuous_actions_v2" / "continuous_dynamics.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = ContinuousActionDynamics(ContinuousDynamicsConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    horizon = dataset.horizon
    truth = np.arange(1, horizon + 1)
    windows = [
        i
        for i, (sequence_index, _) in enumerate(dataset.windows)
        if dataset.sequences[sequence_index].dataset == "SCARED"
    ]

    records = []
    for index in windows:
        sequence_id, current, retrieved, persistence = rollout_indices(
            model, dataset, index
        )
        records.append(
            {
                "sequence_id": sequence_id,
                "current": current,
                "retrieved": np.array(retrieved),
                "persistence": persistence,
                "error": float(np.abs(np.array(retrieved) - truth).mean()),
            }
        )

    errors = np.array([r["error"] for r in records])
    model_steps = np.stack([r["retrieved"] for r in records])
    persistence_error = np.array(
        [np.abs(np.full(horizon, r["persistence"]) - truth).mean() for r in records]
    )
    order = np.argsort(errors)
    selected = [records[order[int(q * (len(order) - 1))]] for q in (0.25, 0.50)]

    figure = plt.figure(figsize=(15.6, 7.2))
    grid = figure.add_gridspec(
        4, 7, width_ratios=[1, 1, 1, 1, 1, 0.28, 2.05], wspace=0.06, hspace=0.10
    )

    for case, record in enumerate(selected):
        sequence_id, current = record["sequence_id"], record["current"]
        for offset, kind in enumerate(("Ground truth", "Model rollout")):
            row = 2 * case + offset
            axis = figure.add_subplot(grid[row, 0])
            axis.imshow(_read_frame(sequence_id, current))
            axis.axis("off")
            axis.text(
                -0.04,
                0.5,
                f"Case {case + 1}\n{kind}",
                transform=axis.transAxes,
                ha="right",
                va="center",
                fontsize=8.6,
                fontweight="bold",
                color=INK,
            )
            if row == 0:
                axis.set_title("Current frame", fontsize=9.5, fontweight="bold")
            for step in range(horizon):
                panel = figure.add_subplot(grid[row, step + 1])
                if kind == "Ground truth":
                    recovered = step + 1
                else:
                    recovered = int(record["retrieved"][step])
                panel.imshow(_read_frame(sequence_id, current + recovered))
                panel.axis("off")
                if row == 0:
                    panel.set_title(f"Step {step + 1}", fontsize=9.5, fontweight="bold")
                colour = (
                    "#3F7A5A"
                    if kind == "Ground truth"
                    else ("#B84A4A" if recovered != step + 1 else "#3F7A5A")
                )
                panel.text(
                    0.03,
                    0.05,
                    f"$t{{+}}{recovered}$",
                    transform=panel.transAxes,
                    color="white",
                    fontsize=8.4,
                    bbox={"facecolor": colour, "alpha": 0.85, "pad": 2.2},
                )

    summary = figure.add_subplot(grid[:, 6])
    steps = np.arange(1, horizon + 1)
    mean_model = model_steps.mean(axis=0)
    standard_error = model_steps.std(axis=0) / np.sqrt(len(model_steps))
    summary.plot(
        steps, steps, "-o", color="#3F7A5A", lw=1.8, ms=5, label="ground truth"
    )
    summary.errorbar(
        steps,
        mean_model,
        yerr=standard_error,
        fmt="-o",
        color="#2F5D8A",
        lw=1.8,
        ms=5,
        capsize=3,
        label="model rollout",
    )
    summary.plot(
        steps,
        np.zeros_like(steps),
        "--s",
        color="#9AA3AD",
        lw=1.5,
        ms=4,
        label="persistence",
    )
    summary.set_xticks(steps)
    summary.set_ylim(-0.55, horizon + 0.35)
    summary.set_xlabel("Rollout step", fontsize=9.5)
    summary.set_ylabel("Recovered temporal index $t{+}k$", fontsize=9.5)
    summary.set_title(
        f"Aggregate over {len(records)} audit-partition windows",
        fontsize=9.8,
        fontweight="bold",
    )
    summary.legend(fontsize=8.4, loc="upper left", frameon=False)
    summary.grid(alpha=0.25, lw=0.6)
    summary.text(
        0.98,
        0.04,
        f"mean index error {errors.mean():.2f} vs {persistence_error.mean():.2f}\n"
        f"better than persistence in "
        f"{100 * (errors < persistence_error).mean():.1f}% of windows",
        transform=summary.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.2,
        color=INK,
    )

    figure.suptitle(
        "Four-step SE(3)-conditioned rollout on SCARED audit sequences: retrieval "
        "visualisation, not pixel generation",
        fontsize=11.5,
        fontweight="bold",
    )
    out = Path(__file__).resolve().parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        out / "figure11_rollout_strip.pdf", bbox_inches="tight", facecolor="white"
    )
    figure.savefig(
        out / "figure11_rollout_strip.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )
    data_path = ROOT / "outputs" / "physical_actions_v2" / "sequences.pt"
    checkpoint_path = (
        ROOT / "outputs" / "continuous_actions_v2" / "continuous_dynamics.pt"
    )
    provenance = {
        "figure": "figure11_rollout_strip.pdf",
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "cache": str(data_path.relative_to(ROOT)),
        "cache_sha256": _sha256(data_path),
        "split": "SCARED test windows",
        "selection": "25th/50th percentile rollout index error",
        "n_windows": len(records),
        "mean_index_error": float(errors.mean()),
        "persistence_index_error": float(persistence_error.mean()),
        "win_fraction": float((errors < persistence_error).mean()),
        "selected": [
            {
                "sequence_id": record["sequence_id"],
                "current_index": int(record["current"]),
                "retrieved_indices": record["retrieved"].tolist(),
                "persistence_index": int(record["persistence"]),
                "mean_index_error": float(record["error"]),
            }
            for record in selected
        ],
    }
    (Path(__file__).resolve().parent / "figure11_rollout_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(
        f"[rollout] windows={len(records)} "
        f"mean index error model={errors.mean():.3f} "
        f"persistence={persistence_error.mean():.3f} "
        f"win={100 * (errors < persistence_error).mean():.1f}%"
    )
    print("[rollout] mean recovered index per step:", mean_model.round(2))
    print(
        "[rollout] selected case index errors:",
        [round(record["error"], 2) for record in selected],
    )


if __name__ == "__main__":
    main()
