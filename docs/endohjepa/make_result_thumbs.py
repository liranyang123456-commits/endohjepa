"""Clean input/output thumbnails for Figure 1.

The earlier version cropped panels out of the composite qualitative figures,
which baked panel titles and neighbouring rows into the thumbnails. This
version reads the *raw* frames instead, so every thumbnail is a complete,
un-annotated image and Figure 1 never shows a stretched or truncated panel.

Forecast lane. For each of the three domains we take the clips of one
validation dataset, keep the clips where the L1 forecast beats persistence,
and among those pick the frame with the least device-overlay content (ION
bronchoscopy stores navigation-console composites and some GI videos carry
processor text; the score below prefers a full endoscopic view). The output
thumbnail is then the nearest-terminal-latent retrieval from a *different*
clip of the same dataset, i.e. the identical procedure used for Figure 10a.

Physical lane. The SCARED example is re-selected with the same deterministic
procedure as ``make_qualitative_results.py`` (Figure 8), so the pair shown in
Figure 1 is one of the cases audited there.

Selected indices and per-pair statistics are written to
``figure1_thumb_selection.json`` so the figure is reproducible.

    python docs/endohjepa/make_result_thumbs.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"

CACHE = ROOT / "outputs" / "cache_15000_pool" / "latents_cache.pt"
CHECKPOINT = ROOT / "outputs" / "scale_15000_causal" / "endohjepa.pt"

# One validation dataset per domain for the three forecast rows of Figure 1.
FORECAST_ROWS = [
    ("laparo", "CholecT50"),
    ("gi", "Kvasir-Capsule"),
    ("bronch", "ION_bronch"),
]


def _sha256(path: Path | str) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _public_path(path: Path | str) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _save(array: np.ndarray, name: str, long_side: int = 384) -> None:
    """Write a thumbnail without any text, preserving the aspect ratio."""
    image = Image.fromarray(np.asarray(array, dtype=np.uint8))
    scale = long_side / max(image.size)
    if scale < 1.0:
        image = image.resize(
            (
                max(1, round(image.size[0] * scale)),
                max(1, round(image.size[1] * scale)),
            ),
            Image.LANCZOS,
        )
    FIG.mkdir(parents=True, exist_ok=True)
    image.save(FIG / name)
    print(f"[thumbs] {name} {image.size[0]}x{image.size[1]}")


def tissue_score(rgb: np.ndarray) -> float:
    """Fraction of pixels that look like illuminated tissue rather than UI.

    Device overlays are either near-neutral light panels or saturated blue
    graphics; endoscopic tissue is red-dominant with moderate saturation.
    """
    x = rgb.astype(np.float32) / 255.0
    high, low = x.max(axis=-1), x.min(axis=-1)
    saturation = np.where(high > 1e-6, (high - low) / np.maximum(high, 1e-6), 0.0)
    red_dominant = (x[..., 0] >= x[..., 1]) & (x[..., 0] >= x[..., 2])
    lit = high > 0.12
    return float(np.mean(red_dominant & lit & (saturation > 0.12)))


def forecast_thumbs() -> list[dict]:
    import torch
    import torch.nn.functional as F

    from endoworld.data.video_clips import EndoClipDataset, domain_balanced_indices
    from endoworld.world.h_jepa import EndoHJEPA, HJEPAConfig

    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    z_val = cache["Z_val"].float()
    d_val = cache["D_val"].long()
    history, horizon = int(checkpoint["history"]), int(checkpoint["horizon"])

    dataset = EndoClipDataset(
        str(ROOT / "manifests" / "sequences.csv"),
        clip_len=16,
        stride=4,
        image_size=256,
        exclude=["EndoVis2019_ROBUST-MIS"],
        return_meta=True,
        split="val",
    )
    selected = domain_balanced_indices(
        dataset.clips, n=min(len(z_val), len(dataset.clips)), seed=1
    )
    clips = [dataset.clips[index] for index in selected]
    if len(clips) != len(z_val):
        raise RuntimeError(f"clip/cache mismatch: {len(clips)} != {len(z_val)}")

    model = EndoHJEPA(HJEPAConfig(**checkpoint["wcfg"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    predictions = []
    with torch.no_grad():
        for start in range(0, len(z_val), 64):
            predictions.append(
                model.forward_l1(
                    z_val[start : start + 64, :history], d_val[start : start + 64]
                )
            )
    prediction = torch.cat(predictions)
    future = z_val[:, history : history + horizon]
    persistence = z_val[:, history - 1 : history].expand_as(future)
    model_error = (prediction - future).square().mean(dim=(1, 2))
    persistence_error = (persistence - future).square().mean(dim=(1, 2))
    model_cos = F.cosine_similarity(prediction, future, dim=-1).mean(dim=1)

    records = []
    for tag, name in FORECAST_ROWS:
        dataset_pool = torch.tensor(
            [i for i, clip in enumerate(clips) if clip.dataset == name],
            dtype=torch.long,
        )
        better = dataset_pool[
            model_error[dataset_pool] < persistence_error[dataset_pool]
        ]
        # Prefer a full endoscopic view over device-console composites.
        eligible = better.tolist() if len(better) else dataset_pool.tolist()
        ranked = sorted(
            eligible,
            key=lambda i: tissue_score(_read(clips[i].frame_paths()[2 * history - 1])),
            reverse=True,
        )
        chosen = ranked[0]
        others = dataset_pool[dataset_pool != chosen]
        if len(others):
            nearest = int(
                torch.cdist(prediction[chosen, -1][None], future[others, -1]).argmin()
            )
            retrieved = int(others[nearest])
            scope = "nearest terminal latent from another clip of the same dataset"
        else:
            retrieved = chosen
            scope = "within-clip terminal future (single validation clip)"
        observed = clips[chosen].frame_paths()[2 * history - 1]
        target = clips[retrieved].frame_paths()[2 * (history + horizon) - 1]
        _save(_read(observed), f"_fig1_in_{tag}.png")
        _save(_read(target), f"_fig1_out_{tag}.png")
        records.append(
            {
                "row": tag,
                "dataset": name,
                "n_dataset_clips": int(len(dataset_pool)),
                "n_forecast_beats_persistence": int(len(better)),
                "selected_cache_index": int(chosen),
                "retrieved_cache_index": int(retrieved),
                "observed_source": _public_path(observed),
                "observed_source_sha256": _sha256(observed),
                "retrieved_source": _public_path(target),
                "retrieved_source_sha256": _sha256(target),
                "cos_model": float(model_cos[chosen]),
                "mse_model": float(model_error[chosen]),
                "mse_persistence": float(persistence_error[chosen]),
                "retrieval_scope": scope,
                "selection": "highest tissue-visibility score among clips where "
                "the forecast beats persistence (or all dataset clips if none)",
            }
        )
    return records


def physical_thumbs() -> dict:
    import torch

    from endoworld.world.continuous_dynamics import (
        ContinuousActionDynamics,
        ContinuousDynamicsConfig,
    )
    from endoworld.world.physical_actions import PhysicalActionDataset, load_sequences
    from endoworld.world.scared_actions import find_scared_rgb
    from endoworld.world.train_continuous_actions import _derangement

    sys.path.insert(0, str(HERE))
    from make_qualitative_results import _nearest_local, _read_frame

    sequences = load_sequences(
        ROOT / "outputs" / "physical_actions_v2" / "sequences.pt"
    )
    dataset = PhysicalActionDataset(sequences, history=4, horizon=4, split="test")
    checkpoint = torch.load(
        ROOT / "outputs" / "continuous_actions_v2_seeded" / "continuous_dynamics.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = ContinuousActionDynamics(ContinuousDynamicsConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    generator = torch.Generator().manual_seed(0)
    scared = [
        i
        for i, (sequence_index, _) in enumerate(dataset.windows)
        if dataset.sequences[sequence_index].dataset == "SCARED"
    ]
    candidates = []
    with torch.no_grad():
        for offset in range(0, len(scared), 32):
            indices = scared[offset : offset + 32]
            history = torch.stack([dataset[i]["history"] for i in indices])
            actions = torch.stack([dataset[i]["actions"] for i in indices])
            future = torch.stack([dataset[i]["future"] for i in indices])
            real = model(history, actions)
            permutation = _derangement(len(indices), generator)
            shuffled = model(history, actions[permutation])
            real_error = (real - future).square().mean(dim=(1, 2))
            shuffled_error = (shuffled - future).square().mean(dim=(1, 2))
            for j, dataset_index in enumerate(indices):
                candidates.append(
                    {
                        "index": dataset_index,
                        "gain": float(shuffled_error[j] - real_error[j]),
                        "real_error": float(real_error[j]),
                        "shuffled_error": float(shuffled_error[j]),
                        "real_prediction": real[j, -1].cpu(),
                        "shuffled_prediction": shuffled[j, -1].cpu(),
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
    if not eligible:
        raise RuntimeError("no positive real-vs-shuffled SCARED example")
    chosen = eligible[int(0.5 * (len(eligible) - 1))]  # median case, as in Figure 8

    sequence_index, start = dataset.windows[chosen["index"]]
    sequence = dataset.sequences[sequence_index]
    current = start + dataset.history - 1
    keyframe = Path(sequence.sequence_id.split("scared:", 1)[1])
    video, _ = find_scared_rgb(keyframe)
    if video is None:
        raise FileNotFoundError(f"no rgb.mp4 for {keyframe}")
    _save(_read_frame(sequence.sequence_id, current), "_fig1_in_scared.png")
    _save(_read_frame(sequence.sequence_id, chosen["real_nearest"]), "_fig1_se3.png")
    return {
        "row": "scared",
        "dataset": "SCARED",
        "sequence_id": sequence.sequence_id,
        "observed_latent_index": int(current),
        "retrieved_latent_index": int(chosen["real_nearest"]),
        "observed_source": _public_path(video),
        "observed_source_frame": 4 * int(current) + 30,
        "observed_source_sha256": _sha256(video),
        "retrieved_source": _public_path(video),
        "retrieved_source_frame": 4 * int(chosen["real_nearest"]) + 30,
        "retrieved_source_sha256": _sha256(video),
        "real_mse": chosen["real_error"],
        "deranged_mse": chosen["shuffled_error"],
        "selection": ("median real-vs-deranged gain among eligible SCARED windows"),
    }


def main() -> None:
    records = forecast_thumbs()
    records.append(physical_thumbs())
    (HERE / "figure1_thumb_selection.json").write_text(
        json.dumps({"thumbnails": records}, indent=2), encoding="utf-8"
    )
    physical_cache = ROOT / "outputs" / "physical_actions_v2" / "sequences.pt"
    physical_checkpoint = (
        ROOT / "outputs" / "continuous_actions_v2_seeded" / "continuous_dynamics.pt"
    )
    risk_checkpoint = (
        ROOT / "outputs" / "probabilistic_risk_v2_corrected" / "probabilistic_risk.pt"
    )
    thumbnail_names = [
        "_fig1_in_laparo.png",
        "_fig1_out_laparo.png",
        "_fig1_in_gi.png",
        "_fig1_out_gi.png",
        "_fig1_in_bronch.png",
        "_fig1_out_bronch.png",
        "_fig1_in_scared.png",
        "_fig1_se3.png",
    ]
    provenance = {
        "figure": [
            "figures/figure1_pipeline.pdf",
            "figures/figure1_pipeline.png",
        ],
        "render_dpi": 220,
        "physical_assets": {
            "cache": _public_path(physical_cache),
            "cache_sha256": _sha256(physical_cache),
            "deterministic_action_cem_checkpoint": _public_path(physical_checkpoint),
            "deterministic_action_cem_checkpoint_sha256": _sha256(physical_checkpoint),
            "separate_probabilistic_risk_checkpoint": _public_path(risk_checkpoint),
            "separate_probabilistic_risk_checkpoint_sha256": _sha256(risk_checkpoint),
            "risk_status": "optional/inactive; failed documented AUC gate",
        },
        "thumbnail_selection": {
            "protocol": {
                "forecast_cache": _public_path(CACHE),
                "forecast_cache_sha256": _sha256(CACHE),
                "forecast_checkpoint": _public_path(CHECKPOINT),
                "forecast_checkpoint_sha256": _sha256(CHECKPOINT),
                "physical_cache": _public_path(physical_cache),
                "physical_cache_sha256": _sha256(physical_cache),
                "physical_checkpoint": _public_path(physical_checkpoint),
                "physical_checkpoint_sha256": _sha256(physical_checkpoint),
                "physical_negative_protocol": (
                    "deterministic no-fixed-point batch derangement, "
                    "batch size 32, seed 0"
                ),
            },
            "thumbnails": records,
        },
        "rendered_thumbnails": [
            {
                "file": f"figures/{name}",
                "sha256": _sha256(FIG / name),
            }
            for name in thumbnail_names
        ],
    }
    (HERE / "figure1_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print("[thumbs] wrote figure1_thumb_selection.json")
    print("[thumbs] wrote figure1_provenance.json")


if __name__ == "__main__":
    main()
