"""Offline continuous-action navigation evaluation with physical pose metrics."""
from __future__ import annotations

import argparse
import json
import math
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
    se3_log,
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
            latent_dim=6, history=history, horizon=horizon, action_dim=6)

    def forward(self, history, actions):
        return history[:, -1:] + actions.cumsum(dim=1)


def _smoke_sequences(n=100, length=20):
    generator = torch.Generator().manual_seed(2)
    sequences = []
    for i in range(n):
        actions = torch.randn(length - 1, 6, generator=generator) * 0.2
        latents = torch.cat([
            torch.zeros(1, 6), actions.cumsum(dim=0),
        ])
        sequences.append(PhysicalSequence(
            sequence_id=f"navigation-smoke-{i}",
            dataset="synthetic",
            latents=latents,
            actions=actions,
        ))
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
        int(key.split(".")[1]) for key in checkpoint["ensemble"]
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
    twist = se3_log(np.linalg.inv(source) @ target)
    return float(np.linalg.norm(twist[:3])), float(np.linalg.norm(twist[3:]) * 180 / math.pi)


def _bootstrap(values, statistic, seed=0, samples=2000):
    values = np.asarray(values)
    generator = np.random.default_rng(seed)
    estimates = [
        statistic(values[generator.integers(0, len(values), len(values))])
        for _ in range(samples)
    ]
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


@torch.no_grad()
def evaluate(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    if args.smoke:
        sequences = _smoke_sequences()
        dynamics, risk_head, calibrator = (
            _Integrator(args.history, args.horizon).to(device), None, None)
    else:
        sequences = load_sequences(args.data)
        dynamics, risk_head, calibrator = _load_model(args.checkpoint, device)
        if args.normalised_actions:
            if not hasattr(dynamics, "action_mean"):
                raise RuntimeError(
                    "--normalised-actions requires a ContinuousActionDynamics checkpoint")
            dynamics = _NormalisedActionDynamics(dynamics).to(device)
    dataset = PhysicalActionDataset(
        sequences, args.history, args.horizon, "test")
    if len(dataset) < args.trials:
        raise RuntimeError(
            f"need at least {args.trials} test windows, found {len(dataset)}")
    generator = torch.Generator().manual_seed(args.seed)
    chosen = torch.randperm(len(dataset), generator=generator)[:args.trials]
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
        goal = batch["future"][:, -1].to(device)
        result = continuous_cem(
            dynamics, history, goal, planner_cfg,
            risk_head=risk_head, calibrator=calibrator)
        predicted = result["prediction"][:, -1].cpu()
        for i in range(history.size(0)):
            sequence_id = batch["sequence_id"][i]
            sequence = sequence_by_id[sequence_id]
            start = int(batch["start_index"][i])
            current_index = start + args.history - 1
            goal_index = current_index + args.horizon
            candidates = sequence.latents[current_index:]
            nearest = current_index + int(torch.cdist(
                predicted[i:i + 1], candidates).argmin())
            poses = trajectories[sequence_id]
            model_trans, model_rot = _pose_error(
                poses[nearest], poses[goal_index])
            persist_trans, persist_rot = _pose_error(
                poses[current_index], poses[goal_index])
            planned_actions = result["actions"][i]
            if isinstance(dynamics, _NormalisedActionDynamics):
                planned_actions = dynamics.denormalise(
                    planned_actions.to(dynamics.model.action_mean.device))
            plan_pose = integrate_actions(planned_actions.cpu().numpy())[-1]
            true_pose = integrate_actions(
                batch["actions"][i].numpy())[-1]
            command_trans, command_rot = _pose_error(plan_pose, true_pose)
            rows.append({
                "sequence_id": sequence_id,
                "accepted": bool(result["accepted"][i]),
                "model_translation_error": model_trans,
                "persistence_translation_error": persist_trans,
                "model_rotation_error_deg": model_rot,
                "persistence_rotation_error_deg": persist_rot,
                "command_translation_error": command_trans,
                "command_rotation_error_deg": command_rot,
                "reach_success": model_trans < persist_trans,
            })
    reach = [float(row["reach_success"]) for row in rows]
    reduction = [
        row["persistence_translation_error"] - row["model_translation_error"]
        for row in rows
    ]
    persist = np.mean([row["persistence_translation_error"] for row in rows])
    command_trans = [row["command_translation_error"] for row in rows]
    report = {
        "task": "offline receding-horizon continuous-action navigation",
        "n_trials": len(rows),
        "synthetic_smoke": args.smoke,
        "normalised_action_space": bool(args.normalised_actions),
        "accepted_rate": float(np.mean([row["accepted"] for row in rows])),
        "reach_rate": float(np.mean(reach)),
        "reach_rate_ci95": _bootstrap(reach, np.mean),
        "translation_error_mean": float(np.mean([
            row["model_translation_error"] for row in rows])),
        "persistence_translation_error_mean": float(persist),
        "translation_error_reduction_fraction": float(
            np.mean(reduction) / max(persist, 1e-8)),
        "translation_reduction_ci95": _bootstrap(reduction, np.mean),
        "rotation_error_deg_mean": float(np.mean([
            row["model_rotation_error_deg"] for row in rows])),
        "command_translation_error_mean": float(np.mean(command_trans)),
        "command_translation_error_ci95": _bootstrap(command_trans, np.mean),
        "rows": rows,
    }
    report["passed"] = bool(
        not args.smoke
        and report["reach_rate"] >= 0.5
        and report["translation_error_reduction_fraction"] >= 0.2
        and report["n_trials"] >= 100
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return report


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/physical_actions/sequences.pt")
    parser.add_argument("--checkpoint", default="outputs/probabilistic_risk/probabilistic_risk.pt")
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
    parser.add_argument("--normalised-actions", action="store_true",
                        help="sample CEM candidates in the model's normalised "
                             "action space (behaviour-support constrained)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
