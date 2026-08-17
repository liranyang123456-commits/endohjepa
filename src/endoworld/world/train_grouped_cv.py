"""Leave-one-case-out grouped CV for the continuous SE(3) dynamics.

The canonical test set (dataset_7 + the held-out C3VD trajectory) was examined
during the audit and is frozen for model development. All architecture/loss
choices are made on grouped cross-validation over the SCARED *training* cases;
the frozen test set is evaluated at most once after a variant is locked.

Example:
    python -m endoworld.world.train_grouped_cv \
        --data outputs/physical_actions_v2/sequences.pt \
        --negatives local --epochs 30
"""
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
    load_sequences,
    video_split,
)
from endoworld.world.train_continuous_actions import (
    _local_negative_actions,
    evaluate,
    evaluate_fixed_bank,
)


def _train_fold(
    args, train_data, val_data, device, cfg, mean, std, seed: int,
) -> dict:
    torch.manual_seed(seed)
    model = ContinuousActionDynamics(cfg).to(device)
    model.set_action_stats(mean.to(device), std.to(device))
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            history = batch["history"].to(device)
            actions = batch["actions"].to(device)
            future = batch["future"].to(device)
            negative_actions = None
            if args.negatives == "local" and args.counterfactual_weight > 0:
                negative_actions = _local_negative_actions(
                    batch, train_data, radius=args.negative_radius, device=device)
            losses = model.losses(
                history, actions, future,
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
        metrics = evaluate(model, val_loader, device)
        if metrics["mse_real_actions"] < best:
            best = metrics["mse_real_actions"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"  [fold-epoch {epoch + 1}] val_real={metrics['mse_real_actions']:.5f}",
              flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "canonical": evaluate(model, val_loader, device),
        "fixed_bank": evaluate_fixed_bank(
            model, val_data, device, n_negatives=args.bank_negatives, seed=seed),
    }


def main() -> dict:
    args = build_parser().parse_args()
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    sequences = load_sequences(args.data)
    train_sequences = [
        s for s in sequences
        if video_split(s.sequence_id, case_id=s.case_id, dataset=s.dataset) == "train"
    ]
    cases = sorted({
        s.case_id for s in train_sequences
        if s.dataset == "SCARED" and s.case_id
    })
    if len(cases) < 2:
        raise RuntimeError("grouped CV needs at least two SCARED training cases")
    action_mean, action_std = None, None
    train_actions = torch.cat([s.actions for s in train_sequences])
    action_mean, action_std = train_actions.mean(0), train_actions.std(0).clamp_min(1e-6)

    cfg = ContinuousDynamicsConfig(
        latent_dim=sequences[0].latents.size(-1),
        hidden_dim=args.hidden, n_heads=args.heads, n_layers=args.layers,
        history=args.history, horizon=args.horizon, dropout=args.dropout,
    )
    folds = {}
    for fold_index, held_case in enumerate(cases):
        fold_train = [s for s in train_sequences if s.case_id != held_case]
        fold_val = [s for s in train_sequences if s.case_id == held_case]
        train_data = PhysicalActionDataset(
            fold_train, args.history, args.horizon, "train")
        # Held-out case is itself a "train" case under video_split, so this
        # constructor keeps its windows.
        val_data = PhysicalActionDataset(
            fold_val, args.history, args.horizon, "train")
        if len(train_data) < 32 or len(val_data) < 8:
            print(f"[cv] skip {held_case}: {len(train_data)}/{len(val_data)} windows")
            continue
        print(f"[cv] fold {held_case}: train={len(train_data)} val={len(val_data)}",
              flush=True)
        folds[held_case] = _train_fold(
            args, train_data, val_data, device, cfg,
            action_mean, action_std, seed=args.seed + fold_index)

    wins = [f["fixed_bank"]["pair_win_fraction"] for f in folds.values()]
    r2 = [f["canonical"]["inverse_action_r2"] for f in folds.values()]
    report = {
        "protocol": "leave-one-case-out grouped CV over SCARED training cases; "
                    "frozen test set untouched",
        "data": str(args.data),
        "negatives": args.negatives,
        "cases": cases,
        "folds": folds,
        "macro_pair_win_fraction": float(np.mean(wins)),
        "min_fold_pair_win_fraction": float(np.min(wins)),
        "macro_inverse_action_r2": float(np.mean(r2)),
        "thresholds": {"pair_win_fraction": 0.8, "inverse_action_r2": 0.3},
        "passed": bool(np.mean(wins) >= 0.8 and np.min(wins) >= 0.5
                       and np.mean(r2) > 0.3),
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "grouped_cv_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(
        {k: v for k, v in report.items() if k != "folds"}, indent=2))
    return report


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/physical_actions_v2/sequences.pt")
    parser.add_argument("--out", default="outputs/continuous_actions_v2_cv")
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
    parser.add_argument("--negatives", choices=["global", "local"], default="local")
    parser.add_argument("--negative-radius", type=int, default=64)
    parser.add_argument("--bank-negatives", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser


if __name__ == "__main__":
    main()
