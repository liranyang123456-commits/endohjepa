"""Qualitative real-frame retrieval for the continuous SE(3) latent predictor.

The model has no pixel decoder. Therefore the visual result is the real frame
whose cached V-JEPA2 latent is nearest to the predicted terminal latent within
a local temporal neighbourhood. It must be labelled retrieval, not generation.
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
from endoworld.world.train_continuous_actions import _derangement  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        rgb = rgb[: rgb.shape[0] // 2]
    return rgb


def _nearest_local(sequence, prediction, current, radius=16):
    stop = min(current + radius + 1, len(sequence.latents))
    candidates = sequence.latents[current:stop]
    return current + int(torch.cdist(prediction[None].cpu(), candidates).argmin())


def main():
    data_path = ROOT / "outputs" / "physical_actions_v2" / "sequences.pt"
    checkpoint_path = (
        ROOT / "outputs" / "continuous_actions_v2_seeded" / "continuous_dynamics.pt"
    )
    output = Path(__file__).resolve().parent / "figures" / "figure8_qualitative.pdf"
    sequences = load_sequences(data_path)
    dataset = PhysicalActionDataset(sequences, history=4, horizon=4, split="test")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ContinuousActionDynamics(ContinuousDynamicsConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    candidates = []
    generator = torch.Generator().manual_seed(0)
    scared_indices = [
        i
        for i, (sequence_index, _) in enumerate(dataset.windows)
        if dataset.sequences[sequence_index].dataset == "SCARED"
    ]
    with torch.no_grad():
        for offset in range(0, len(scared_indices), 32):
            indices = scared_indices[offset : offset + 32]
            history = torch.stack([dataset[i]["history"] for i in indices])
            actions = torch.stack([dataset[i]["actions"] for i in indices])
            future = torch.stack([dataset[i]["future"] for i in indices])
            real_prediction = model(history, actions)
            permutation = _derangement(len(indices), generator)
            shuffled_prediction = model(history, actions[permutation])
            real_error = (real_prediction - future).square().mean(dim=(1, 2))
            shuffled_error = (shuffled_prediction - future).square().mean(dim=(1, 2))
            for j, dataset_index in enumerate(indices):
                candidates.append(
                    {
                        "index": dataset_index,
                        "gain": float(shuffled_error[j] - real_error[j]),
                        "real_error": float(real_error[j]),
                        "shuffled_error": float(shuffled_error[j]),
                        "real_prediction": real_prediction[j, -1].cpu(),
                        "shuffled_prediction": shuffled_prediction[j, -1].cpu(),
                    }
                )

    eligible = []
    for result in candidates:
        sequence_index, start = dataset.windows[result["index"]]
        sequence = dataset.sequences[sequence_index]
        current = start + dataset.history - 1
        goal = current + dataset.horizon
        result["real_nearest"] = _nearest_local(
            sequence, result["real_prediction"], current
        )
        result["shuffled_nearest"] = _nearest_local(
            sequence, result["shuffled_prediction"], current
        )
        if result["gain"] > 0 and abs(result["real_nearest"] - goal) < abs(
            result["shuffled_nearest"] - goal
        ):
            eligible.append(result)
    eligible.sort(key=lambda row: row["gain"])
    if len(eligible) < 3:
        raise RuntimeError("fewer than three positive real-vs-shuffled examples")
    # Deterministic representative examples, not the three maxima.
    selected = [eligible[int(q * (len(eligible) - 1))] for q in (0.25, 0.5, 0.75)]

    # Symmetric layout: both retrievals get an error map on a shared scale, so
    # the real-versus-shuffled comparison is legible rather than asserted.
    fig, axes = plt.subplots(3, 6, figsize=(15.4, 7.6), layout="constrained")
    titles = [
        "Last observed frame",
        f"Ground truth $t{{+}}{dataset.horizon}$",
        "Real-action retrieval",
        "Deranged-action retrieval",
        "|real $-$ truth|",
        "|deranged $-$ truth|",
    ]
    for axis, title in zip(axes[0], titles):
        axis.set_title(title, fontsize=9.5, fontweight="bold")

    summary = []
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

        def _error(image):
            resized = cv2.resize(image, (goal_image.shape[1], goal_image.shape[0]))
            return np.abs(
                resized.astype(np.float32) - goal_image.astype(np.float32)
            ).mean(axis=-1)

        real_error_map, shuffled_error_map = _error(real_image), _error(shuffled_image)
        panels = [
            observed_image,
            goal_image,
            real_image,
            shuffled_image,
            real_error_map,
            shuffled_error_map,
        ]
        for column, (axis, panel) in enumerate(zip(axes[row_index], panels)):
            if column >= 4:
                axis.imshow(panel, cmap="magma", vmin=0, vmax=96)
            else:
                axis.imshow(panel)
            axis.axis("off")
        # Row labels use axes-relative text because the panels have no frame.
        axes[row_index, 0].text(
            -0.04,
            0.5,
            f"Case {row_index + 1}\nlatent MSE\nreal {result['real_error']:.3f}\n"
            f"deranged {result['shuffled_error']:.3f}",
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=8.5,
        )
        for column, step in (
            (2, real_nearest - current),
            (3, shuffled_nearest - current),
        ):
            axes[row_index, column].text(
                0.02,
                0.03,
                f"retrieved $t{{+}}{step}$",
                transform=axes[row_index, column].transAxes,
                color="white",
                fontsize=8,
                bbox={"facecolor": "black", "alpha": 0.6, "pad": 2},
            )
        for column, error_map in ((4, real_error_map), (5, shuffled_error_map)):
            axes[row_index, column].text(
                0.02,
                0.03,
                f"MAE {error_map.mean():.1f}",
                transform=axes[row_index, column].transAxes,
                color="white",
                fontsize=8,
                bbox={"facecolor": "black", "alpha": 0.6, "pad": 2},
            )
            if row_index == 0 and column == 5:
                colorbar = fig.colorbar(
                    axes[row_index, column].images[0],
                    ax=axes[:, 4:],
                    fraction=0.035,
                    pad=0.02,
                )
                colorbar.set_label("RGB absolute error (0--96)")
        summary.append(
            {
                "case": row_index + 1,
                "real_step": real_nearest - current,
                "deranged_step": shuffled_nearest - current,
                "real_mae": float(real_error_map.mean()),
                "deranged_mae": float(shuffled_error_map.mean()),
                "real_latent_mse": result["real_error"],
                "deranged_latent_mse": result["shuffled_error"],
            }
        )

    fig.suptitle(
        "Real SCARED qualitative retrieval "
        "(25th/50th/75th percentile positive-gain cases)",
        fontsize=12,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    fig.savefig(
        output.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white"
    )
    provenance = {
        "figure": output.name,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "cache": str(data_path.relative_to(ROOT)),
        "cache_sha256": _sha256(data_path),
        "split": "previously contacted SCARED audit windows",
        "negative_protocol": (
            "deterministic no-fixed-point batch derangement, batch size 32, seed 0"
        ),
        "retrieval_candidates": (
            "same-sequence latent offsets 0 through 16 relative to the current "
            "latent, truncated at sequence end; offset 0 is allowed"
        ),
        "selection": "25th/50th/75th percentile latent-MSE gain among eligible positive cases",
        "n_candidates": len(candidates),
        "n_eligible": len(eligible),
        "selected": [
            {
                "dataset_index": int(result["index"]),
                "sequence_id": dataset.sequences[
                    dataset.windows[result["index"]][0]
                ].sequence_id,
                "real_latent_mse": float(result["real_error"]),
                "deranged_latent_mse": float(result["shuffled_error"]),
                "real_retrieved_step": int(
                    result["real_nearest"]
                    - (dataset.windows[result["index"]][1] + dataset.history - 1)
                ),
                "deranged_retrieved_step": int(
                    result["shuffled_nearest"]
                    - (dataset.windows[result["index"]][1] + dataset.history - 1)
                ),
                "real_mae": next(
                    row["real_mae"]
                    for row in summary
                    if row["real_latent_mse"] == result["real_error"]
                ),
                "deranged_mae": next(
                    row["deranged_mae"]
                    for row in summary
                    if row["deranged_latent_mse"] == result["shuffled_error"]
                ),
            }
            for result in selected
        ],
    }
    (
        Path(__file__).resolve().parent / "figure8_qualitative_provenance.json"
    ).write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    for row in summary:
        print(
            "[qualitative] case {case}: real t+{real_step} (MAE {real_mae:.1f}) "
            "vs deranged t+{deranged_step} (MAE {deranged_mae:.1f}); "
            "latent MSE {real_latent_mse:.3f} vs {deranged_latent_mse:.3f}".format(
                **row
            )
        )
    print(f"[qualitative] wrote {output}")


if __name__ == "__main__":
    main()
