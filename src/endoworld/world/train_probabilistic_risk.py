"""Train probabilistic SE(3) dynamics and calibrate near-wall risk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from endoworld.world.continuous_dynamics import ContinuousDynamicsConfig
from endoworld.world.physical_actions import PhysicalActionDataset, load_sequences
from endoworld.world.probabilistic_dynamics import (
    DynamicsEnsemble,
    NearWallRiskHead,
    RiskCalibrator,
    near_wall_labels,
)
from endoworld.world.train_continuous_actions import _action_stats, _synthetic_sequences


def _risk_target(values: torch.Tensor, threshold: float) -> torch.Tensor:
    if values.ndim >= 4:
        return near_wall_labels(values, threshold)
    if values.ndim == 3 and values.size(-1) > 4:
        return near_wall_labels(values, threshold)
    while values.ndim > 2:
        values = values.mean(dim=-1)
    return (values < threshold).float()


def _make_smoke_risk(sequences):
    all_values = torch.cat([sequence.latents[:, :1] for sequence in sequences])
    median = all_values.median()
    for sequence in sequences:
        # Synthetic wall distance: small when the first latent coordinate is high.
        sequence.depth_or_risk = (median - sequence.latents[:, :1]).sigmoid()
    return 0.5


@torch.no_grad()
def _risk_logits(ensemble, head, loader, device, threshold):
    logits, targets = [], []
    for batch in loader:
        if "depth_or_risk" not in batch:
            continue
        history = batch["history"].to(device)
        actions = batch["actions"].to(device)
        result = ensemble.predict(history, actions)
        logits.append(
            head(
                result["mean"],
                result["aleatoric_variance"],
                result["epistemic_variance"],
            )
            .flatten()
            .cpu()
        )
        targets.append(_risk_target(batch["depth_or_risk"], threshold).flatten().cpu())
    if not logits:
        return torch.zeros(0), torch.zeros(0)
    return torch.cat(logits), torch.cat(targets)


def train(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    sequences = _synthetic_sequences(n=84) if args.smoke else load_sequences(args.data)
    if not args.smoke:
        labelled = [s for s in sequences if s.depth_or_risk is not None]
        dropped = len(sequences) - len(labelled)
        if dropped:
            print(
                f"[risk] dropping {dropped} unlabelled sequences (risk training is SCARED-only)"
            )
        sequences = labelled
    threshold = _make_smoke_risk(sequences) if args.smoke else args.near_wall_threshold
    datasets = {
        split: PhysicalActionDataset(sequences, args.history, args.horizon, split)
        for split in ("train", "val", "test")
    }
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise RuntimeError("need non-empty video-level train/val/test splits")
    if not all(sequence.depth_or_risk is not None for sequence in sequences):
        raise RuntimeError(
            "all sequences need depth_or_risk for calibrated risk training"
        )
    loaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train")
        for split, dataset in datasets.items()
    }
    cfg = ContinuousDynamicsConfig(
        latent_dim=sequences[0].latents.size(-1),
        hidden_dim=args.hidden,
        n_heads=args.heads,
        n_layers=args.layers,
        history=args.history,
        horizon=args.horizon,
        dropout=args.dropout,
    )
    ensemble = DynamicsEnsemble(cfg, args.members).to(device)
    mean, std = _action_stats(sequences)
    ensemble.set_action_stats(mean.to(device), std.to(device))
    dynamics_optimizer = torch.optim.AdamW(
        ensemble.parameters(), lr=args.lr, weight_decay=0.01
    )
    for epoch in range(args.epochs):
        ensemble.train()
        for batch in loaders["train"]:
            history = batch["history"].to(device)
            actions = batch["actions"].to(device)
            future = batch["future"].to(device)
            losses = [
                member.probabilistic_losses(history, actions, future)["total"]
                for member in ensemble.members
            ]
            loss = torch.stack(losses).mean()
            dynamics_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ensemble.parameters(), 1.0)
            dynamics_optimizer.step()

    risk_head = NearWallRiskHead(cfg.latent_dim, args.hidden).to(device)
    risk_optimizer = torch.optim.AdamW(risk_head.parameters(), lr=args.lr)
    for _ in range(args.risk_epochs):
        risk_head.train()
        ensemble.eval()
        for batch in loaders["train"]:
            history = batch["history"].to(device)
            actions = batch["actions"].to(device)
            target = _risk_target(batch["depth_or_risk"].to(device), threshold)
            with torch.no_grad():
                prediction = ensemble.predict(history, actions)
            logits = risk_head(
                prediction["mean"],
                prediction["aleatoric_variance"],
                prediction["epistemic_variance"],
            )
            loss = F.binary_cross_entropy_with_logits(logits, target)
            risk_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            risk_optimizer.step()

    ensemble.eval()
    risk_head.eval()
    val_logits, val_target = _risk_logits(
        ensemble, risk_head, loaders["val"], device, threshold
    )
    test_logits, test_target = _risk_logits(
        ensemble, risk_head, loaders["test"], device, threshold
    )
    calibrator = RiskCalibrator(alpha=args.alpha).fit(val_logits, val_target)
    report = {
        "n_calibration": len(val_target),
        "n_test": len(test_target),
        "near_wall_threshold": threshold,
        "test": calibrator.metrics(test_logits, test_target),
        "thresholds": {"auc": 0.75, "min_transitions": 500},
    }
    report["passed"] = bool(report["test"]["auc"] >= 0.75 and len(test_target) >= 500)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ensemble": ensemble.state_dict(),
            "risk_head": risk_head.state_dict(),
            "config": cfg.__dict__,
            "calibration": {
                "temperature": calibrator.temperature,
                "radius": calibrator.radius,
                "alpha": calibrator.alpha,
            },
            "report": report,
        },
        output / "probabilistic_risk.pt",
    )
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/physical_actions/sequences.pt")
    parser.add_argument("--out", default="outputs/probabilistic_risk")
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--risk-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--near-wall-threshold", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
