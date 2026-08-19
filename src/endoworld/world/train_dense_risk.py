"""Dense-token near-wall risk training with per-patch depth supervision.

The pooled-latent risk head stays at chance (AUC 0.54) because near-wall
geometry is spatially local. This script trains a per-token risk head on dense
past-only tokens with per-patch depth supervision from SCARED scene_points,
then evaluates frame-level AUC/ECE on the frozen test case (dataset_7).

    python -m endoworld.world.train_dense_risk \
        --dense outputs/physical_actions_v2/dense_scared.pt
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from endoworld.world.physical_actions import video_split
from endoworld.world.probabilistic_dynamics import (
    RiskCalibrator,
    binary_auc,
    binary_ece,
)


def _depth_maps(keyframe_dir: Path) -> dict[int, np.ndarray]:
    tar_path = keyframe_dir / "data" / "scene_points.tar.gz"
    if not tar_path.is_file():
        return {}
    out = {}
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".tiff"):
                continue
            index = int("".join(c for c in member.name if c.isdigit()) or -1)
            handle = tar.extractfile(member)
            if handle is None:
                continue
            img = cv2.imdecode(
                np.frombuffer(handle.read(), np.uint8), cv2.IMREAD_UNCHANGED
            )
            if img is None or img.ndim != 3:
                continue
            depth = img[..., 0].astype(np.float32)
            depth = depth[: depth.shape[0] // 2]  # mono top eye, matching v2 crop
            out[index] = depth
    return out


def _patch_targets(depth: np.ndarray, grid: int, threshold: float) -> np.ndarray:
    """(H, W) depth -> (grid*grid,) near-wall flags from patch minima."""
    small = cv2.resize(depth, (grid * 16, grid * 16), interpolation=cv2.INTER_AREA)
    patches = small.reshape(grid, 16, grid, 16)
    patch_min = patches.min(axis=(1, 3))
    return (patch_min < threshold).astype(np.float32).reshape(-1)


class DenseRiskHead(torch.nn.Module):
    """Per-token risk scores with learned attention pooling to a frame logit."""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, 1)
        )
        self.attention = torch.nn.Linear(dim, 1)

    def forward(self, tokens):  # (B, N, D) -> (B,)
        scores = self.network(tokens).squeeze(-1)
        weights = torch.softmax(self.attention(tokens).squeeze(-1), dim=1)
        return (scores * weights).sum(dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dense", default="outputs/physical_actions_v2/dense_scared.pt"
    )
    parser.add_argument(
        "--risk-cache", default="outputs/physical_actions_v2/sequences_risk.pt"
    )
    parser.add_argument("--scared", default="datasets/SCARED")
    parser.add_argument("--threshold", type=float, default=34.4)
    parser.add_argument(
        "--label-mode", choices=["absolute", "relative"], default="absolute"
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="outputs/dense_risk_v2/metrics.json")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pack = torch.load(args.dense, map_location="cpu", weights_only=False)
    pack["stride"]
    pack["tubelet"]
    pack["lookback_tubelets"] - 1

    # Frame-level near-wall labels come from the v2 risk cache (5th-percentile
    # scene depth per latent step); dense steps align 1:1 with pooled latents.
    risk_pack = torch.load(args.risk_cache, map_location="cpu", weights_only=False)
    label_by_seq = {}
    for row in risk_pack["sequences"]:
        if row.get("depth_or_risk") is None:
            continue
        depth = row["depth_or_risk"].flatten().float()
        if args.label_mode == "absolute":
            label_by_seq[row["sequence_id"]] = (depth < args.threshold).float()
        else:
            # Relative near-wall: depth below the sequence's own causal running
            # quartile. Removes the cross-case base-rate shift (9.4% val vs 43%
            # test) that made absolute labels untransferable.
            values = []
            for k in range(len(depth)):
                if k < 8:
                    values.append(0.0)
                    continue
                q = torch.quantile(depth[:k], 0.25)
                values.append(float(depth[k] < q))
            label_by_seq[row["sequence_id"]] = torch.tensor(values)
    data = {"train": [], "val": [], "test": []}
    for row in pack["rows"]:
        labels = label_by_seq.get(row["sequence_id"])
        if labels is None:
            continue
        dense = row["dense"]
        n = min(dense.size(0), labels.size(0))
        split = video_split(
            row["sequence_id"], case_id=row["case_id"], dataset="SCARED"
        )
        data[split].append(
            {
                "dense": dense[:n].float(),
                "targets": labels[:n],
            }
        )
    print(
        {k: sum(int(v["dense"].size(0)) for v in vv) for k, vv in data.items()},
        flush=True,
    )

    dim = pack["rows"][0]["dense"].size(-1)
    model = DenseRiskHead(dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(args.epochs):
        model.train()
        for seq in data["train"]:
            tokens = seq["dense"].to(device)
            target = seq["targets"].to(device)
            logits = model(tokens)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(f"[epoch {epoch + 1}] loss={loss.item():.4f}", flush=True)

    model.eval()
    report = {
        "threshold": args.threshold,
        "label_mode": args.label_mode,
        "supervision": "frame-level 5th-pct depth labels on dense tokens",
    }
    split_logits = {}
    for split in ("val", "test"):
        logits_all, target_all = [], []
        with torch.no_grad():
            for seq in data[split]:
                tokens = seq["dense"].to(device)
                logits_all.append(model(tokens).cpu())
                target_all.append(seq["targets"].cpu())
        logits = torch.cat(logits_all)
        target = torch.cat(target_all)
        split_logits[split] = (logits, target)
        report[split] = {
            "n_frames": int(len(target)),
            "positive_fraction": float(target.mean()),
            "auc": binary_auc(torch.sigmoid(logits).numpy(), target.numpy()),
        }
    # calibrate on val, report test ECE
    calibrator = RiskCalibrator(alpha=0.1)
    val_logits, val_target = split_logits["val"]
    test_logits, test_target = split_logits["test"]
    calibrator.fit(val_logits, val_target)
    probability, _ = calibrator.predict(test_logits)
    report["test"]["ece_calibrated"] = binary_ece(
        probability.numpy(), test_target.numpy()
    )
    report["gate"] = {"auc": 0.75, "min_transitions": 500}
    report["passed"] = bool(
        report["test"]["auc"] >= 0.75 and report["test"]["n_frames"] >= 500
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
