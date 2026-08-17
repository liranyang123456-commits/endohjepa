"""Latent-space sampling MPC: minimise energy to a goal representation."""
from __future__ import annotations

from dataclasses import dataclass

import torch


def rollout_path_energy(model, z_start, actions, pred):
    """Return per-step and summed energies for a predicted latent path.

    Args:
        z_start: (B, D) latent immediately before the rollout.
        actions: (B, H) discrete actions aligned with ``pred``.
        pred: (B, H, D) predicted future latents.
    """
    h = min(actions.size(1), pred.size(1))
    pred = pred[:, :h]
    actions = actions[:, :h]
    prev = torch.cat([z_start.unsqueeze(1), pred[:, :-1]], dim=1)
    step_energy = model.energy(prev, actions, pred)
    return step_energy, step_energy.sum(dim=1)


@torch.no_grad()
def latent_mpc(model, z_hist, z_goal, domain_id, n_samples: int = 32, steps: int = 4):
    """Sample discrete action sequences; pick the lowest-energy path toward z_goal.

    Returns (best_actions (B, steps), best_energy (B,)).
    """
    cfg = model.cfg
    b = z_hist.size(0)
    device = z_hist.device
    best_e = torch.full((b,), 1e9, device=device)
    best_a = torch.zeros(b, steps, dtype=torch.long, device=device)
    horizon = cfg.horizon
    for _ in range(n_samples):
        a = torch.randint(0, cfg.n_actions, (b, max(steps, horizon)), device=device)
        pred = model.forward_l3(z_hist, a[:, :horizon], domain_id)
        rollout_h = min(steps, pred.size(1), horizon)
        z_last = pred[:, rollout_h - 1]
        goal_cost = (z_last - z_goal).pow(2).mean(dim=-1)
        _, path_energy = rollout_path_energy(
            model, z_hist[:, -1], a[:, :rollout_h], pred[:, :rollout_h])
        e = goal_cost + path_energy
        better = e < best_e
        best_e = torch.where(better, e, best_e)
        best_a = torch.where(better.unsqueeze(-1), a[:, :steps], best_a)
    return best_a, best_e


@dataclass
class ContinuousMPCConfig:
    horizon: int = 4
    samples: int = 256
    iterations: int = 5
    elites: int = 32
    action_limit: float = 2.5
    initial_std: float = 1.0
    goal_weight: float = 1.0
    risk_weight: float = 5.0
    uncertainty_weight: float = 1.0
    smoothness_weight: float = 0.05
    energy_weight: float = 0.1
    max_collision_probability: float = 0.5
    max_uncertainty: float = 2.0
    mppi_temperature: float = 0.2


def _continuous_distribution(dynamics, history, actions):
    if hasattr(dynamics, "predict"):
        return dynamics.predict(history, actions)
    if hasattr(dynamics, "distribution"):
        mean, log_variance = dynamics.distribution(history, actions)
        return {
            "mean": mean,
            "aleatoric_variance": log_variance.exp(),
            "epistemic_variance": torch.zeros_like(mean),
        }
    mean = dynamics(history, actions)
    return {
        "mean": mean,
        "aleatoric_variance": torch.zeros_like(mean),
        "epistemic_variance": torch.zeros_like(mean),
    }


def _continuous_config(dynamics):
    if hasattr(dynamics, "cfg"):
        return dynamics.cfg
    if hasattr(dynamics, "members") and len(dynamics.members):
        return dynamics.members[0].cfg
    raise TypeError("continuous dynamics must expose cfg or ensemble members")


def continuous_rollout_cost(
    dynamics,
    history: torch.Tensor,
    actions: torch.Tensor,
    goal: torch.Tensor,
    cfg: ContinuousMPCConfig,
    risk_head=None,
    calibrator=None,
    energy_fn=None,
) -> dict[str, torch.Tensor]:
    """Evaluate continuous candidates and apply a non-negotiable safety gate."""
    result = _continuous_distribution(dynamics, history, actions)
    prediction = result["mean"]
    aleatoric = result["aleatoric_variance"]
    epistemic = result["epistemic_variance"]
    uncertainty_step = (aleatoric + epistemic).mean(dim=-1)
    if risk_head is None:
        risk = torch.zeros_like(uncertainty_step)
    else:
        logits = risk_head(prediction, aleatoric, epistemic)
        if calibrator is None:
            risk = torch.sigmoid(logits)
        else:
            risk = calibrator.predict(logits)[0]
    terminal_goal = (prediction[:, -1] - goal).square().mean(dim=-1)
    if actions.size(1) > 1:
        smoothness = (actions[:, 1:] - actions[:, :-1]).square().mean(dim=(1, 2))
    else:
        smoothness = torch.zeros_like(terminal_goal)
    if energy_fn is None:
        previous = torch.cat([history[:, -1:], prediction[:, :-1]], dim=1)
        path_energy = (prediction - previous).square().mean(dim=(1, 2))
    else:
        path_energy = energy_fn(history[:, -1], actions, prediction)
    cost = (
        cfg.goal_weight * terminal_goal
        + cfg.risk_weight * risk.mean(dim=-1)
        + cfg.uncertainty_weight * uncertainty_step.mean(dim=-1)
        + cfg.smoothness_weight * smoothness
        + cfg.energy_weight * path_energy
    )
    unsafe = (
        (risk.max(dim=-1).values > cfg.max_collision_probability)
        | (uncertainty_step.max(dim=-1).values > cfg.max_uncertainty)
        | (actions.abs().amax(dim=(1, 2)) > cfg.action_limit)
    )
    safe_cost = cost.masked_fill(unsafe, float("inf"))
    return {
        "cost": safe_cost,
        "raw_cost": cost,
        "safe": ~unsafe,
        "prediction": prediction,
        "risk": risk,
        "uncertainty": uncertainty_step,
    }


@torch.no_grad()
def continuous_cem(
    dynamics,
    history: torch.Tensor,
    goal: torch.Tensor,
    cfg: ContinuousMPCConfig | None = None,
    risk_head=None,
    calibrator=None,
    energy_fn=None,
) -> dict[str, torch.Tensor]:
    """Cross-entropy continuous MPC with hard reject/zero-motion fallback."""
    cfg = cfg or ContinuousMPCConfig()
    batch = history.size(0)
    dynamics_cfg = _continuous_config(dynamics)
    if cfg.horizon != dynamics_cfg.horizon:
        raise ValueError("planner and dynamics horizon must match")
    action_dim = int(dynamics_cfg.action_dim)
    mean = torch.zeros(batch, cfg.horizon, action_dim, device=history.device)
    std = torch.full_like(mean, cfg.initial_std)
    elite_count = min(cfg.elites, cfg.samples)
    for _ in range(cfg.iterations):
        noise = torch.randn(
            batch, cfg.samples, cfg.horizon, action_dim, device=history.device)
        candidates = (mean[:, None] + std[:, None] * noise).clamp(
            -cfg.action_limit, cfg.action_limit)
        flat_actions = candidates.flatten(0, 1)
        flat_history = history[:, None].expand(
            -1, cfg.samples, -1, -1).flatten(0, 1)
        flat_goal = goal[:, None].expand(
            -1, cfg.samples, -1).flatten(0, 1)
        evaluated = continuous_rollout_cost(
            dynamics, flat_history, flat_actions, flat_goal, cfg,
            risk_head, calibrator, energy_fn)
        costs = evaluated["cost"].view(batch, cfg.samples)
        elite_index = costs.topk(elite_count, largest=False).indices
        gather = elite_index[..., None, None].expand(
            -1, -1, cfg.horizon, action_dim)
        elites = candidates.gather(1, gather)
        finite = torch.isfinite(
            costs.gather(1, elite_index)).float()[..., None, None]
        count = finite.sum(dim=1).clamp_min(1.0)
        mean = (elites * finite).sum(dim=1) / count
        variance = ((elites - mean[:, None]).square() * finite).sum(dim=1) / count
        std = variance.sqrt().clamp_min(0.05)
    final = continuous_rollout_cost(
        dynamics, history, mean, goal, cfg, risk_head, calibrator, energy_fn)
    accepted = final["safe"]
    actions = torch.where(
        accepted[:, None, None], mean, torch.zeros_like(mean))
    return {**final, "actions": actions, "accepted": accepted}


@torch.no_grad()
def continuous_mppi(
    dynamics,
    history: torch.Tensor,
    goal: torch.Tensor,
    cfg: ContinuousMPCConfig | None = None,
    risk_head=None,
    calibrator=None,
    energy_fn=None,
) -> dict[str, torch.Tensor]:
    """MPPI update with the same hard safety gate as CEM."""
    cfg = cfg or ContinuousMPCConfig()
    batch = history.size(0)
    dynamics_cfg = _continuous_config(dynamics)
    if cfg.horizon != dynamics_cfg.horizon:
        raise ValueError("planner and dynamics horizon must match")
    action_dim = int(dynamics_cfg.action_dim)
    nominal = torch.zeros(batch, cfg.horizon, action_dim, device=history.device)
    for _ in range(cfg.iterations):
        noise = torch.randn(
            batch, cfg.samples, cfg.horizon, action_dim, device=history.device)
        candidates = (nominal[:, None] + cfg.initial_std * noise).clamp(
            -cfg.action_limit, cfg.action_limit)
        flat_actions = candidates.flatten(0, 1)
        flat_history = history[:, None].expand(
            -1, cfg.samples, -1, -1).flatten(0, 1)
        flat_goal = goal[:, None].expand(
            -1, cfg.samples, -1).flatten(0, 1)
        evaluated = continuous_rollout_cost(
            dynamics, flat_history, flat_actions, flat_goal, cfg,
            risk_head, calibrator, energy_fn)
        costs = evaluated["cost"].view(batch, cfg.samples)
        finite_cost = torch.where(
            torch.isfinite(costs), costs, torch.full_like(costs, 1e6))
        shifted = finite_cost - finite_cost.min(dim=1, keepdim=True).values
        weight = torch.softmax(-shifted / cfg.mppi_temperature, dim=1)
        nominal = (candidates * weight[..., None, None]).sum(dim=1)
    final = continuous_rollout_cost(
        dynamics, history, nominal, goal, cfg, risk_head, calibrator, energy_fn)
    accepted = final["safe"]
    actions = torch.where(
        accepted[:, None, None], nominal, torch.zeros_like(nominal))
    return {**final, "actions": actions, "accepted": accepted}
