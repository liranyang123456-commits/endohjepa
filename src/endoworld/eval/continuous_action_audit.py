"""Evaluate continuous-action evidence with explicitly named negative banks.

The canonical batch-shuffled score and same-sequence fixed-bank scores answer
different questions.  This entry point reports both without relabelling either
as the other, and records the number of correlated windows and sequences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from endoworld.world.continuous_dynamics import (
    ContinuousActionDynamics,
    ContinuousDynamicsConfig,
)
from endoworld.world.physical_actions import PhysicalActionDataset, load_sequences
from endoworld.world.train_continuous_actions import (
    evaluate,
    evaluate_fixed_bank,
)


def _load_model(path: str, device: str) -> ContinuousActionDynamics:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ContinuousActionDynamics(ContinuousDynamicsConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def audit(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    sequences = load_sequences(args.data)
    if args.dataset:
        sequences = [
            sequence for sequence in sequences if sequence.dataset == args.dataset
        ]
    dataset = PhysicalActionDataset(sequences, args.history, args.horizon, "test")
    if not len(dataset):
        raise RuntimeError("no test windows for the requested dataset")
    model = _load_model(args.checkpoint, device)
    batch_shuffled = evaluate(
        model, DataLoader(dataset, batch_size=args.batch_size, shuffle=False), device
    )
    fixed_bank = evaluate_fixed_bank(
        model, dataset, device, n_negatives=args.negatives, seed=args.seed
    )
    report = {
        "task": "continuous SE(3) action-conditioned sensitivity audit",
        "checkpoint": args.checkpoint,
        "dataset_filter": args.dataset,
        "history": args.history,
        "horizon": args.horizon,
        "n_windows": len(dataset),
        "n_test_sequences": len(
            {
                dataset.sequences[sequence_index].sequence_id
                for sequence_index, _ in dataset.windows
            }
        ),
        "batch_shuffled": {
            **batch_shuffled,
            "negative_protocol": (
                "one deterministic random permutation within each evaluation "
                "batch; negatives can come from another sequence"
            ),
        },
        "fixed_same_sequence_bank": {
            **fixed_bank,
            "negative_protocol": (
                f"{args.negatives} deterministic same-sequence actions sampled "
                "within the evaluator's local hard-negative radius"
            ),
        },
        "dependence_note": (
            "Windows overlap within each sequence. Window-level fractions are "
            "descriptive; they are not independent case-level inference."
        ),
        "test_contact_note": (
            "This audit set was previously contacted during the action-path "
            "audit. Results are audit-selected evidence, not an independent "
            "confirmation test."
        ),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/physical_actions_v2/sequences.pt")
    parser.add_argument(
        "--checkpoint",
        default="outputs/continuous_actions_v2/continuous_dynamics.pt",
    )
    parser.add_argument(
        "--out",
        default="outputs/continuous_actions_v2/action_audit.json",
    )
    parser.add_argument("--dataset", default="SCARED")
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--negatives", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser


if __name__ == "__main__":
    audit(build_parser().parse_args())
