"""Visualise latent forecasts for every dataset represented in cached validation.

No pixel decoder is trained. A predicted terminal latent is visualised by
retrieving the nearest of the four real future tubelets from the same clip.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from endoworld.data.video_clips import EndoClipDataset, domain_balanced_indices  # noqa: E402
from endoworld.world.h_jepa import EndoHJEPA, HJEPAConfig  # noqa: E402


def _read(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def main():
    cache_path = ROOT / "outputs" / "cache_15000_pool" / "latents_cache.pt"
    checkpoint_path = ROOT / "outputs" / "scale_15000_causal" / "endohjepa.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    z_val = cache["Z_val"].float()
    d_val = cache["D_val"].long()
    history = int(checkpoint["history"])
    horizon = int(checkpoint["horizon"])

    dataset = EndoClipDataset(
        str(ROOT / "manifests" / "sequences.csv"),
        clip_len=16, stride=4, image_size=256,
        exclude=["EndoVis2019_ROBUST-MIS"], return_meta=True, split="val",
    )
    selected_indices = domain_balanced_indices(
        dataset.clips, n=min(len(z_val), len(dataset.clips)), seed=1)
    clips = [dataset.clips[index] for index in selected_indices]
    if len(clips) != len(z_val):
        raise RuntimeError(f"clip/cache mismatch: {len(clips)} != {len(z_val)}")

    model = EndoHJEPA(HJEPAConfig(**checkpoint["wcfg"]))
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    predictions = []
    with torch.no_grad():
        for start in range(0, len(z_val), 64):
            z = z_val[start:start + 64].to(device)
            domain = d_val[start:start + 64].to(device)
            predictions.append(model.forward_l1(z[:, :history], domain).cpu())
    prediction = torch.cat(predictions)
    future = z_val[:, history:history + horizon]
    persistence = z_val[:, history - 1:history].expand_as(future)
    model_error = (prediction - future).square().mean(dim=(1, 2))
    persistence_error = (persistence - future).square().mean(dim=(1, 2))
    model_cos = F.cosine_similarity(prediction, future, dim=-1).mean(dim=1)
    persistence_cos = F.cosine_similarity(persistence, future, dim=-1).mean(dim=1)

    by_dataset: dict[str, list[int]] = defaultdict(list)
    for index, clip in enumerate(clips):
        by_dataset[clip.dataset].append(index)

    rows = []
    examples = []
    for name in sorted(by_dataset):
        indices = torch.tensor(by_dataset[name], dtype=torch.long)
        gains = persistence_error[indices] - model_error[indices]
        positive = indices[gains > 0]
        pool = positive if len(positive) else indices
        ordered = pool[torch.argsort(
            (persistence_error[pool] - model_error[pool]))]
        chosen = int(ordered[len(ordered) // 2])
        pred_terminal = prediction[chosen, -1]
        cross_clip = indices[indices != chosen]
        if len(cross_clip):
            nearest_position = int(torch.cdist(
                pred_terminal[None], future[cross_clip, -1]).argmin())
            retrieved_index = int(cross_clip[nearest_position])
            retrieval_scope = "nearest terminal latent from another validation clip"
        else:
            candidate_future = future[chosen]
            retrieved_horizon = int(torch.cdist(
                pred_terminal[None], candidate_future).argmin())
            retrieved_index = chosen
            retrieval_scope = f"within-clip future h={retrieved_horizon + 1}; n=1"
        rows.append({
            "dataset": name,
            "domain": clips[chosen].domain,
            "n": len(indices),
            "cos_model": float(model_cos[indices].mean()),
            "cos_persistence": float(persistence_cos[indices].mean()),
            "mse_model": float(model_error[indices].mean()),
            "mse_persistence": float(persistence_error[indices].mean()),
            "win_fraction": float((model_error[indices] < persistence_error[indices]).float().mean()),
            "selected_cache_index": chosen,
            "retrieved_cache_index": retrieved_index,
            "retrieval_scope": retrieval_scope,
            "selection": "median model-vs-persistence MSE gain among positive cases",
        })
        examples.append((name, chosen, retrieved_index))

    out = Path(__file__).resolve().parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    report_by_name = {row["dataset"]: row for row in rows}
    titles = [
        "Last observed frame", "Ground-truth terminal future",
        "Predicted-latent retrieval", "Retrieval error",
    ]
    for part, chunk in enumerate((examples[:6], examples[6:]), start=1):
        figure, axes = plt.subplots(
            len(chunk), 4, figsize=(11.8, 2.15 * len(chunk)))
        if len(chunk) == 1:
            axes = axes[None, :]
        for axis, title in zip(axes[0], titles):
            axis.set_title(title, fontsize=10, fontweight="bold")
        for row_index, (name, index, retrieved_index) in enumerate(chunk):
            paths = clips[index].frame_paths()
            observed = _read(paths[2 * history - 1])
            target = _read(paths[2 * (history + horizon) - 1])
            retrieved_paths = clips[retrieved_index].frame_paths()
            retrieved_image = _read(retrieved_paths[2 * (history + horizon) - 1])
            resized = np.asarray(
                Image.fromarray(retrieved_image).resize(
                    (target.shape[1], target.shape[0])))
            error = np.abs(
                resized.astype(np.float32) - target.astype(np.float32)
            ).mean(axis=-1)
            for column, panel in enumerate(
                    (observed, target, retrieved_image, error)):
                axis = axes[row_index, column]
                if column == 3:
                    axis.imshow(panel, cmap="magma", vmin=0, vmax=96)
                else:
                    axis.imshow(panel)
                axis.axis("off")
            report = report_by_name[name]
            axes[row_index, 0].text(
                -0.06, 0.5,
                f"{name}\ncos {report['cos_model']:.3f} vs "
                f"{report['cos_persistence']:.3f}\n"
                f"win {100 * report['win_fraction']:.1f}%",
                transform=axes[row_index, 0].transAxes, ha="right", va="center",
                fontsize=8.5,
            )
            axes[row_index, 2].text(
                0.02, 0.03,
                "cross-clip" if retrieved_index != index else "within-clip; n=1",
                transform=axes[row_index, 2].transAxes, color="white", fontsize=8,
                bbox={"facecolor": "black", "alpha": 0.58, "pad": 2},
            )
        figure.suptitle(
            f"Cross-dataset latent forecast retrieval "
            f"({part}/2, video-level validation)",
            fontsize=13, fontweight="bold",
        )
        figure.tight_layout(rect=(0, 0, 1, 0.975))
        stem = f"figure10{'a' if part == 1 else 'b'}_forecast_qualitative"
        figure.savefig(out / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
        figure.savefig(out / f"{stem}.png", dpi=180,
                       bbox_inches="tight", facecolor="white")
        plt.close(figure)
    report_path = Path(__file__).resolve().parent / "qualitative_forecast_15000.json"
    report_path.write_text(json.dumps({
        "protocol": {
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "cache": str(cache_path.relative_to(ROOT)),
            "split": "video-level val",
            "n_val": len(z_val),
            "history": history,
            "horizon": horizon,
            "visualisation": "nearest real future tubelet; not pixel generation",
        },
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"[forecast qualitative] wrote {len(rows)} dataset rows")


if __name__ == "__main__":
    main()
