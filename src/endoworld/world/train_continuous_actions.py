"""Train/evaluate the continuous SE(3) action-conditioned baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from endoworld.world.continuous_dynamics import (
    ContinuousActionDynamics,
    ContinuousDynamicsConfig,
)
from endoworld.world.physical_actions import (
    PhysicalActionDataset,
    PhysicalSequence,
    load_sequences,
    video_split,
)


def _derangement(n: int, generator: torch.Generator) -> torch.Tensor:
    """Return a deterministic random permutation with no fixed points."""
    if n < 2:
        raise ValueError("a derangement requires at least two elements")
    order = torch.randperm(n, generator=generator)
    permutation = torch.empty(n, dtype=torch.long)
    permutation[order] = order.roll(1)
    return permutation


def _synthetic_sequences(n: int = 30, length: int = 16, dim: int = 24):
    generator = torch.Generator().manual_seed(0)
    mapping = torch.randn(6, dim, generator=generator) * 0.2
    sequences = []
    for i in range(n):
        actions = torch.randn(length - 1, 6, generator=generator)
        z = [torch.randn(dim, generator=generator)]
        for action in actions:
            z.append(
                z[-1] + action @ mapping + torch.randn(dim, generator=generator) * 0.01
            )
        sequences.append(
            PhysicalSequence(
                sequence_id=f"synthetic-{i}",
                dataset="synthetic",
                latents=torch.stack(z),
                actions=actions,
            )
        )
    return sequences


def _action_stats(sequences):
    actions = torch.cat(
        [
            sequence.actions
            for sequence in sequences
            if video_split(
                sequence.sequence_id,
                case_id=sequence.case_id,
                dataset=sequence.dataset,
            )
            == "train"
        ]
    )
    return actions.mean(0), actions.std(0).clamp_min(1e-6)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    real_error, shuffled_error, wins = [], [], []
    inverse_prediction, inverse_target = [], []
    generator = torch.Generator().manual_seed(0)
    for batch in loader:
        history = batch["history"].to(device)
        actions = batch["actions"].to(device)
        future = batch["future"].to(device)
        if actions.size(0) < 2:
            continue
        prediction = model(history, actions)
        perm = _derangement(actions.size(0), generator)
        shuffled = model(history, actions[perm.to(actions.device)])
        real = (prediction - future).square().mean(dim=(1, 2))
        random = (shuffled - future).square().mean(dim=(1, 2))
        real_error.extend(real.cpu().tolist())
        shuffled_error.extend(random.cpu().tolist())
        wins.extend((real < random).cpu().tolist())
        current = torch.cat([history[:, -1:], future[:, :-1]], dim=1)
        inverse_prediction.append(model.inverse(current, future).cpu())
        inverse_target.append(model.normalise_actions(actions).cpu())
    if not real_error:
        return {"n": 0}
    pred = torch.cat(inverse_prediction).flatten(0, 1)
    target = torch.cat(inverse_target).flatten(0, 1)
    residual = (pred - target).square().sum()
    total = (target - target.mean(0)).square().sum().clamp_min(1e-8)
    return {
        "n": len(real_error),
        "mse_real_actions": float(np.mean(real_error)),
        "mse_shuffled_actions": float(np.mean(shuffled_error)),
        "real_action_win_fraction": float(np.mean(wins)),
        "inverse_action_r2": float(1.0 - residual / total),
    }


def _local_negative_actions(
    batch,
    dataset: PhysicalActionDataset,
    radius: int,
    device,
    rng: np.random.Generator,
):
    """Same-sequence, temporally adjacent counterfactual actions (hard negatives)."""
    indices = batch["index"].tolist()
    negative_indices = [
        dataset.hard_negative_index(int(i), radius=radius, rng=rng) for i in indices
    ]
    negative_actions = torch.stack(
        [dataset[int(j)]["actions"] for j in negative_indices]
    )
    return negative_actions.to(device)


@torch.no_grad()
def evaluate_fixed_bank(
    model,
    dataset: PhysicalActionDataset,
    device: str,
    n_negatives: int = 10,
    radius: int = 64,
    seed: int = 0,
    batch_size: int = 64,
):
    """Deterministic hard-negative evaluation independent of batch order.

    Each window is compared against a fixed bank of same-sequence negatives.
    Reports the pair-level win fraction (mean over all window/negative pairs),
    matching the local-negative distribution that evaluation batches induce,
    plus the stricter all-negatives win rate per window.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    pair_wins, window_all = [], []
    real_errors, negative_errors = [], []
    actual_counts = []
    windows_with_reduced_bank = 0
    windows_without_candidates = 0
    for start in range(0, len(dataset), batch_size):
        batch_indices = list(range(start, min(start + batch_size, len(dataset))))
        history = torch.stack([dataset[i]["history"] for i in batch_indices]).to(device)
        actions = torch.stack([dataset[i]["actions"] for i in batch_indices]).to(device)
        future = torch.stack([dataset[i]["future"] for i in batch_indices]).to(device)
        real_error = (model(history, actions) - future).square().mean(dim=(1, 2)).cpu()
        for local_index, dataset_index in enumerate(batch_indices):
            sequence_index, window_start = dataset.windows[dataset_index]
            same_sequence = dataset.sequence_windows()[sequence_index]
            candidates = [
                candidate
                for candidate in same_sequence
                if candidate != dataset_index
                and abs(dataset.windows[candidate][1] - window_start) <= radius
            ]
            if not candidates:
                candidates = [
                    candidate
                    for candidate in same_sequence
                    if candidate != dataset_index
                ]
            if not candidates:
                actual_counts.append(0)
                windows_with_reduced_bank += 1
                windows_without_candidates += 1
                continue
            count = min(n_negatives, len(candidates))
            chosen = rng.choice(candidates, size=count, replace=False).tolist()
            actual_counts.append(count)
            if count < n_negatives:
                windows_with_reduced_bank += 1
            negative_actions = torch.stack(
                [dataset[int(candidate)]["actions"] for candidate in chosen]
            ).to(device)
            repeated_history = history[local_index : local_index + 1].expand(
                count, -1, -1
            )
            repeated_future = future[local_index : local_index + 1].expand(
                count, -1, -1
            )
            errors = (
                (model(repeated_history, negative_actions) - repeated_future)
                .square()
                .mean(dim=(1, 2))
                .cpu()
            )
            wins = real_error[local_index] < errors
            pair_wins.extend(wins.tolist())
            window_all.append(float(bool(wins.all())))
            real_errors.append(float(real_error[local_index]))
            negative_errors.extend(errors.tolist())
    count_summary = {
        "minimum": min(actual_counts) if actual_counts else 0,
        "maximum": max(actual_counts) if actual_counts else 0,
        "mean": float(np.mean(actual_counts)) if actual_counts else 0.0,
        "total_pairs": int(sum(actual_counts)),
    }
    return {
        "n": len(dataset),
        "n_negatives": n_negatives,
        "requested_negatives_per_window": n_negatives,
        "actual_unique_negative_counts": actual_counts,
        "actual_unique_negatives_per_window": count_summary,
        "windows_with_reduced_bank": windows_with_reduced_bank,
        "windows_without_candidates": windows_without_candidates,
        "negative_sampling_strategy": (
            "sample eligible same-sequence local candidates without replacement; "
            "when fewer than requested exist, use every eligible distinct candidate; "
            "when the local radius is empty, fall back to all other same-sequence windows"
        ),
        "radius": radius,
        "batch_size": batch_size,
        "mse_real_actions": float(np.mean(real_errors)) if real_errors else None,
        "mse_negative_actions": (
            float(np.mean(negative_errors)) if negative_errors else None
        ),
        "pair_win_fraction": float(np.mean(pair_wins)) if pair_wins else None,
        "all_negative_win_fraction": (
            float(np.mean(window_all)) if window_all else None
        ),
    }


def _scared_cases(sequences) -> list[str]:
    cases = sorted(
        {s.case_id for s in sequences if s.dataset == "SCARED" and s.case_id}
    )
    return cases


def train(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    sequences = _synthetic_sequences() if args.smoke else load_sequences(args.data)
    train_data = PhysicalActionDataset(sequences, args.history, args.horizon, "train")
    val_data = PhysicalActionDataset(sequences, args.history, args.horizon, "val")
    test_data = PhysicalActionDataset(sequences, args.history, args.horizon, "test")
    if not train_data or not val_data:
        raise RuntimeError(
            "physical cache needs non-empty video-level train and val splits"
        )
    cfg = ContinuousDynamicsConfig(
        latent_dim=sequences[0].latents.size(-1),
        hidden_dim=args.hidden,
        n_heads=args.heads,
        n_layers=args.layers,
        history=args.history,
        horizon=args.horizon,
        dropout=args.dropout,
    )
    model = ContinuousActionDynamics(cfg).to(device)
    mean, std = _action_stats(sequences)
    model.set_action_stats(mean.to(device), std.to(device))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    negative_rng = np.random.default_rng(args.seed)
    best = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        running, seen = 0.0, 0
        for batch in train_loader:
            history = batch["history"].to(device)
            actions = batch["actions"].to(device)
            future = batch["future"].to(device)
            negative_actions = None
            if args.negatives == "local" and args.counterfactual_weight > 0:
                negative_actions = _local_negative_actions(
                    batch,
                    train_data,
                    radius=args.negative_radius,
                    device=device,
                    rng=negative_rng,
                )
            losses = model.losses(
                history,
                actions,
                future,
                inverse_weight=args.inverse_weight,
                cycle_weight=args.cycle_weight,
                counterfactual_weight=args.counterfactual_weight,
                counterfactual_margin=args.counterfactual_margin,
                negative_actions=negative_actions,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += losses["total"].detach().item() * history.size(0)
            seen += history.size(0)
        metrics = evaluate(model, val_loader, device)
        if metrics["mse_real_actions"] < best:
            best = metrics["mse_real_actions"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(
            f"[epoch {epoch + 1}] loss={running / max(seen, 1):.5f} "
            f"val_real={metrics['mse_real_actions']:.5f} "
            f"val_shuffle={metrics['mse_shuffled_actions']:.5f}"
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    report = {
        "task": "continuous camera-frame SE(3) action-conditioned prediction",
        "pose_convention": "c2w_local_log_se3_v_then_w",
        "data": str(args.data),
        "negatives": args.negatives,
        "seed": args.seed,
        "training_negative_sampling": (
            "same-sequence local candidates drawn from one NumPy generator "
            "initialised by seed and advanced deterministically across batches"
            if args.negatives == "local"
            else "batch permutation generated by PyTorch"
        ),
        "validation": evaluate(model, val_loader, device),
        "test": (
            {"skipped": "contacted audit partition withheld during development"}
            if args.skip_test
            else evaluate(model, test_loader, device)
        ),
        "validation_fixed_bank": evaluate_fixed_bank(
            model, val_data, device, n_negatives=args.bank_negatives, seed=args.seed
        ),
        "test_fixed_bank": (
            {"skipped": "contacted audit partition withheld during development"}
            if args.skip_test
            else evaluate_fixed_bank(
                model,
                test_data,
                device,
                n_negatives=args.bank_negatives,
                seed=args.seed,
            )
        ),
        "success_thresholds": {
            "real_action_win_fraction": 0.8,
            "inverse_action_r2": 0.3,
        },
        "loss_weights": {
            "inverse": args.inverse_weight,
            "cycle": args.cycle_weight,
            "counterfactual": args.counterfactual_weight,
            "counterfactual_margin": args.counterfactual_margin,
        },
    }
    if args.skip_test:
        val_bank = report["validation_fixed_bank"]
        report["development_gate_passed"] = bool(
            val_bank.get("pair_win_fraction", 0) >= 0.8
            and report["validation"].get("inverse_action_r2", -1) > 0.3
        )
        report["audit_gate_evaluated"] = False
        report["passed"] = None
    else:
        report["audit_gate_evaluated"] = True
        report["passed"] = bool(
            report["test"].get("real_action_win_fraction", 0) > 0.8
            and report["test"].get("inverse_action_r2", -1) > 0.3
        )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.__dict__,
            "report": report,
        },
        output / "continuous_dynamics.pt",
    )
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/physical_actions/sequences.pt")
    parser.add_argument("--out", default="outputs/continuous_actions")
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--inverse-weight", type=float, default=0.25)
    parser.add_argument("--cycle-weight", type=float, default=0.1)
    parser.add_argument("--counterfactual-weight", type=float, default=0.5)
    parser.add_argument("--counterfactual-margin", type=float, default=0.02)
    parser.add_argument(
        "--negatives",
        choices=["global", "local"],
        default="global",
        help="counterfactual distribution; 'local' matches evaluation batches "
        "(same-sequence, temporally adjacent hard negatives)",
    )
    parser.add_argument("--negative-radius", type=int, default=64)
    parser.add_argument("--bank-negatives", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="withhold the contacted audit split (development runs only)",
    )
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
