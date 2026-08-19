"""Single-shot oracle-goal latent-retrieval proxy for continuous actions.

This evaluator is not a closed-loop navigation benchmark. It conditions CEM on
the held-out terminal latent, performs one open-loop optimisation, and maps its
terminal prediction to the nearest latent later in that same recorded
trajectory. Pose values therefore quantify a retrieval proxy, not an endpoint
produced by executing the planned action.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from endoworld.world.continuous_dynamics import ContinuousDynamicsConfig
from endoworld.world.physical_actions import (
    PhysicalActionDataset,
    PhysicalSequence,
    integrate_actions,
    load_sequences,
)
from endoworld.world.plan_mpc import ContinuousMPCConfig, continuous_cem
from endoworld.world.probabilistic_dynamics import (
    DynamicsEnsemble,
    NearWallRiskHead,
    RiskCalibrator,
)


class _Integrator(torch.nn.Module):
    def __init__(self, history: int, horizon: int):
        super().__init__()
        self.cfg = ContinuousDynamicsConfig(
            latent_dim=6, history=history, horizon=horizon, action_dim=6
        )

    def forward(self, history, actions):
        return history[:, -1:] + actions.cumsum(dim=1)


def _smoke_sequences(n=100, length=20):
    generator = torch.Generator().manual_seed(2)
    sequences = []
    for i in range(n):
        actions = torch.randn(length - 1, 6, generator=generator) * 0.2
        latents = torch.cat(
            [
                torch.zeros(1, 6),
                actions.cumsum(dim=0),
            ]
        )
        sequences.append(
            PhysicalSequence(
                sequence_id=f"navigation-smoke-{i}",
                dataset="synthetic",
                latents=latents,
                actions=actions,
            )
        )
    return sequences


class _NormalisedActionDynamics(torch.nn.Module):
    """Wrap a ContinuousActionDynamics so the planner samples in normalised
    action space (zero mean / unit std), i.e. within the training support.

    Raw CEM used initial_std=1 on physical twists whose rotation axes have
    std ~0.003-0.01 rad, putting candidates 100-380 sigma out of distribution.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.cfg = model.cfg

    def forward(self, history, actions_norm):
        actions = actions_norm * self.model.action_std + self.model.action_mean
        return self.model(history, actions)

    def denormalise(self, actions_norm):
        return actions_norm * self.model.action_std + self.model.action_mean


def _load_model(path: str, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = ContinuousDynamicsConfig(**checkpoint["config"])
    if "model" in checkpoint:
        from endoworld.world.continuous_dynamics import ContinuousActionDynamics

        model = ContinuousActionDynamics(cfg)
        model.load_state_dict(checkpoint["model"], strict=False)
        return model.to(device).eval(), None, None
    member_ids = {
        int(key.split(".")[1])
        for key in checkpoint["ensemble"]
        if key.startswith("members.")
    }
    ensemble = DynamicsEnsemble(cfg, len(member_ids)).to(device)
    ensemble.load_state_dict(checkpoint["ensemble"])
    risk_head = NearWallRiskHead(cfg.latent_dim, cfg.hidden_dim).to(device)
    risk_head.load_state_dict(checkpoint["risk_head"])
    calibration = checkpoint["calibration"]
    calibrator = RiskCalibrator(calibration["alpha"])
    calibrator.temperature = calibration["temperature"]
    calibrator.radius = calibration["radius"]
    return ensemble.eval(), risk_head.eval(), calibrator


def _pose_error(source: np.ndarray, target: np.ndarray):
    relative = np.linalg.inv(source) @ target
    translation = float(np.linalg.norm(relative[:3, 3]))
    cosine = np.clip((np.trace(relative[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    rotation = float(np.degrees(np.arccos(cosine)))
    return translation, rotation


def _strict_future_nearest_index(
    query: torch.Tensor,
    latents: torch.Tensor,
    current_index: int,
) -> int:
    """Retrieve only indices strictly after the current state."""
    first = current_index + 1
    if first >= len(latents):
        raise ValueError("strict-future retrieval requires a later latent")
    candidates = latents[first:]
    return first + int(torch.cdist(query.reshape(1, -1), candidates).argmin())


def _public_sequence_id(sequence: PhysicalSequence) -> str:
    """Return a dataset-relative identifier without local filesystem paths."""
    normalised = str(sequence.sequence_id).replace("\\", "/")
    leaf = normalised.rstrip("/").split("/")[-1]
    parts = [sequence.dataset]
    if sequence.case_id:
        parts.append(sequence.case_id)
    if leaf and leaf.lower() not in {part.lower() for part in parts}:
        parts.append(leaf)
    return "/".join(parts)


def _bootstrap(values, statistic, seed=0, samples=2000):
    values = np.asarray(values)
    generator = np.random.default_rng(seed)
    estimates = [
        statistic(values[generator.integers(0, len(values), len(values))])
        for _ in range(samples)
    ]
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def _bootstrap_relative_reduction(
    model_errors, persistence_errors, seed=0, samples=2000
):
    """CI for 1 - mean(model error) / mean(persistence error)."""
    model_errors = np.asarray(model_errors, dtype=np.float64)
    persistence_errors = np.asarray(persistence_errors, dtype=np.float64)
    if model_errors.shape != persistence_errors.shape:
        raise ValueError("model and persistence errors must have the same shape")
    generator = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        indices = generator.integers(0, len(model_errors), len(model_errors))
        baseline = persistence_errors[indices].mean()
        estimates.append(1.0 - model_errors[indices].mean() / max(baseline, 1e-8))
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


@torch.no_grad()
def evaluate(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    if args.smoke:
        sequences = _smoke_sequences()
        dynamics, risk_head, calibrator = (
            _Integrator(args.history, args.horizon).to(device),
            None,
            None,
        )
    else:
        sequences = load_sequences(args.data)
        dynamics, risk_head, calibrator = _load_model(args.checkpoint, device)
        if args.normalised_actions:
            if not hasattr(dynamics, "action_mean"):
                raise RuntimeError(
                    "--normalised-actions requires a ContinuousActionDynamics checkpoint"
                )
            dynamics = _NormalisedActionDynamics(dynamics).to(device)
    dataset = PhysicalActionDataset(sequences, args.history, args.horizon, "test")
    if args.dataset:
        selected = [
            index
            for index, (sequence_index, _) in enumerate(dataset.windows)
            if dataset.sequences[sequence_index].dataset == args.dataset
        ]
        dataset = torch.utils.data.Subset(dataset, selected)
    if len(dataset) < args.trials:
        raise RuntimeError(
            f"need at least {args.trials} test windows for "
            f"{args.dataset or 'all datasets'}, found {len(dataset)}"
        )
    generator = torch.Generator().manual_seed(args.seed)
    chosen = torch.randperm(len(dataset), generator=generator)[: args.trials]
    subset = torch.utils.data.Subset(dataset, chosen.tolist())
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)
    planner_cfg = ContinuousMPCConfig(
        horizon=args.horizon,
        samples=args.samples,
        iterations=args.iterations,
        elites=args.elites,
        action_limit=args.action_limit,
        max_collision_probability=args.max_collision_probability,
        max_uncertainty=args.max_uncertainty,
    )
    sequence_by_id = {sequence.sequence_id: sequence for sequence in sequences}
    trajectories = {
        sequence.sequence_id: integrate_actions(sequence.actions.numpy())
        for sequence in sequences
    }
    rows = []
    for batch in loader:
        history = batch["history"].to(device)
        # Oracle held-out target for an offline proxy only; it is not a
        # deployable navigation goal.
        goal = batch["future"][:, -1].to(device)
        result = continuous_cem(
            dynamics,
            history,
            goal,
            planner_cfg,
            risk_head=risk_head,
            calibrator=calibrator,
        )
        predicted = result["prediction"][:, -1].cpu()
        for i in range(history.size(0)):
            sequence_id = batch["sequence_id"][i]
            sequence = sequence_by_id[sequence_id]
            start = int(batch["start_index"][i])
            current_index = start + args.history - 1
            goal_index = current_index + args.horizon
            # The pose below belongs to a retrieved real future state, not to
            # the endpoint induced by ``planned_actions``.
            nearest = _strict_future_nearest_index(
                predicted[i], sequence.latents, current_index
            )
            poses = trajectories[sequence_id]
            model_trans, model_rot = _pose_error(poses[nearest], poses[goal_index])
            persist_trans, persist_rot = _pose_error(
                poses[current_index], poses[goal_index]
            )
            planned_actions = result["actions"][i]
            if isinstance(dynamics, _NormalisedActionDynamics):
                planned_actions = dynamics.denormalise(
                    planned_actions.to(dynamics.model.action_mean.device)
                )
            plan_pose = integrate_actions(planned_actions.cpu().numpy())[-1]
            true_pose = integrate_actions(batch["actions"][i].numpy())[-1]
            command_trans, command_rot = _pose_error(plan_pose, true_pose)
            rows.append(
                {
                    "sequence_id": _public_sequence_id(sequence),
                    "current_index": current_index,
                    "goal_index": goal_index,
                    "retrieved_index": nearest,
                    "accepted": bool(result["accepted"][i]),
                    "model_translation_error": model_trans,
                    "persistence_translation_error": persist_trans,
                    "model_rotation_error_deg": model_rot,
                    "persistence_rotation_error_deg": persist_rot,
                    "command_translation_error": command_trans,
                    "command_rotation_error_deg": command_rot,
                    "retrieval_beats_persistence": model_trans < persist_trans,
                }
            )
    retrieval_wins = [float(row["retrieval_beats_persistence"]) for row in rows]
    model_trans = np.asarray(
        [row["model_translation_error"] for row in rows], dtype=np.float64
    )
    persistence_trans = np.asarray(
        [row["persistence_translation_error"] for row in rows], dtype=np.float64
    )
    persist = float(persistence_trans.mean())
    command_trans = [row["command_translation_error"] for row in rows]
    report = {
        "task": "offline single-shot oracle-goal latent-retrieval proxy",
        "planning_mode": "one open-loop CEM optimisation per window",
        "goal_source": "held-out terminal latent from the recorded trajectory",
        "pose_metric": (
            "translation/rotation error of the nearest future recorded latent; "
            "not the endpoint of the planned command"
        ),
        "retrieval_candidates": (
            "strictly future same-sequence latent indices "
            "[current_index + 1, sequence_length); current_index is excluded"
        ),
        "n_trials": len(rows),
        "n_sequences": len({row["sequence_id"] for row in rows}),
        "dataset_filter": args.dataset,
        "synthetic_smoke": args.smoke,
        "normalised_action_space": bool(args.normalised_actions),
        "accepted_rate": float(np.mean([row["accepted"] for row in rows])),
        "proxy_win_fraction": float(np.mean(retrieval_wins)),
        "proxy_win_fraction_window_bootstrap95": _bootstrap(retrieval_wins, np.mean),
        "retrieval_translation_error_mean": float(model_trans.mean()),
        "persistence_translation_error_mean": float(persist),
        "retrieval_translation_error_reduction_fraction": float(
            1.0 - model_trans.mean() / max(persist, 1e-8)
        ),
        "retrieval_translation_reduction_window_bootstrap95": _bootstrap_relative_reduction(
            model_trans, persistence_trans
        ),
        "rotation_error_deg_mean": float(
            np.mean([row["model_rotation_error_deg"] for row in rows])
        ),
        "command_translation_error_mean": float(np.mean(command_trans)),
        "command_translation_error_ci95": _bootstrap(command_trans, np.mean),
        "window_dependence_note": (
            "Windows from the same sequence/keyframe overlap; bootstrap intervals "
            "are descriptive window-resampling summaries, not case-level inference."
        ),
        "rows": rows,
    }
    report["proxy_screening"] = {
        "n_trials": 100,
        "proxy_win_fraction": 0.5,
        "retrieval_translation_error_reduction_fraction": 0.2,
    }
    report["navigation_validated"] = False
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "rows"}, indent=2
        )
    )
    return report


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/physical_actions/sequences.pt")
    parser.add_argument(
        "--checkpoint", default="outputs/probabilistic_risk/probabilistic_risk.pt"
    )
    parser.add_argument("--out", default="outputs/physical_navigation/report.json")
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--elites", type=int, default=32)
    parser.add_argument("--action-limit", type=float, default=2.5)
    parser.add_argument("--max-collision-probability", type=float, default=0.5)
    parser.add_argument("--max-uncertainty", type=float, default=2.0)
    parser.add_argument(
        "--normalised-actions",
        action="store_true",
        help="sample CEM candidates in the model's normalised "
        "action space (scaled to training coordinates)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="optional exact PhysicalSequence dataset name; "
        "use this to keep an evaluation corpus pure",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
