"""Safety gate: policy proposes, submodular optimiser verifies / repairs.

Deployment contract
-------------------
1. A learned policy (or human) proposes a burn sequence.
2. The gate evaluates coverage / over-treatment / forbidden-region violations.
3. If coverage < γ  →  repair with classical greedy cover (``plan_ablation``)
   or continue stepping with ``env.greedy_action()`` until γ is met.
4. Only plans that pass the gate are returned for clinical use.

This keeps the $(1-1/e)$ coverage guarantee as a hard safety backbone while
still benefiting from a fast learned proposer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from endoworld.ablation.planner import plan_ablation
from endoworld.ablation.sim_env import (
    AblationAction,
    AblationSimEnv,
    make_env_from_axes,
    rollout,
)
from endoworld.ablation.trajectory_schema import (
    AblationTrajectory,
    LesionGeometry,
    plan_to_trajectory,
)


@dataclass
class GateConfig:
    coverage_target: float = 0.99
    max_overtreat_mL: float | None = None   # None = no hard cap
    max_burns: int = 40
    repair: str = "cascade"                 # cascade | greedy_continue | replan | reject
    force_zone_mm: float | None = None
    spacing_mm: float = 1.5


@dataclass
class GateResult:
    accepted: bool
    repaired: bool
    trajectory: AblationTrajectory
    metrics: dict
    violations: list = field(default_factory=list)
    note: str = ""


def evaluate_trajectory(traj: AblationTrajectory,
                        coverage_target: float = 0.99,
                        max_overtreat_mL: float | None = None) -> list[str]:
    """Return list of violation strings (empty ⇒ pass)."""
    v = []
    cov = float(traj.metrics.get("target_coverage_incl_margin") or 0.0)
    if cov < coverage_target - 1e-6:
        v.append(f"coverage {cov:.3f} < {coverage_target}")
    over = float(traj.metrics.get("healthy_overtreated_mL") or 0.0)
    if max_overtreat_mL is not None and over > max_overtreat_mL:
        v.append(f"overtreat {over:.2f} mL > {max_overtreat_mL}")
    if traj.n_burns() == 0:
        v.append("empty plan")
    return v


def _continue_with_greedy(env: AblationSimEnv) -> AblationTrajectory:
    """From current env state, keep applying greedy until done."""
    assert env._state is not None
    while True:
        m = env._metrics_snapshot()
        if m["target_coverage"] >= env.cfg.coverage_target:
            break
        if env._state.t >= env.cfg.max_burns:
            break
        action = env.greedy_action()
        _, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    return env.to_trajectory()


def gate_rollout(
    geometry: LesionGeometry,
    policy: Callable | None = None,
    device: str = "MWA",
    cfg: GateConfig | None = None,
    seed: int | None = 0,
) -> GateResult:
    """Run policy (or greedy) then verify / repair.

    ``policy``: callable(obs, env) -> AblationAction, or None for pure greedy.
    """
    cfg = cfg or GateConfig()
    env = make_env_from_axes(
        geometry.tumor_axes_mm,
        margin_mm=geometry.margin_mm,
        device=device,
        spacing_mm=cfg.spacing_mm,
        coverage_target=cfg.coverage_target,
        max_burns=cfg.max_burns,
        force_zone_mm=cfg.force_zone_mm,
        seed=seed,
        case_id=geometry.case_id,
    )
    # copy anatomy
    for k in ("lobe", "airway_generation", "dist_pleura_mm",
              "dist_chestwall_mm", "dist_vessel_mm", "solidity"):
        if getattr(geometry, k, None) is not None:
            setattr(env.geometry, k, getattr(geometry, k))

    pol = policy if policy is not None else "greedy"
    traj, _ = rollout(env, policy=pol, seed=seed)
    violations = evaluate_trajectory(
        traj, cfg.coverage_target, cfg.max_overtreat_mL)
    repaired = False
    note = "policy plan accepted"

    if violations:
        if cfg.repair == "reject":
            return GateResult(
                accepted=False, repaired=False, trajectory=traj,
                metrics=dict(traj.metrics), violations=violations,
                note="rejected: " + "; ".join(violations),
            )

        # Cascade repair: try greedy continuation from a fresh greedy
        # episode first; if still failing, fall back to full replan.
        if cfg.repair in ("greedy_continue", "cascade"):
            env_g = make_env_from_axes(
                geometry.tumor_axes_mm,
                margin_mm=geometry.margin_mm,
                device=device,
                spacing_mm=cfg.spacing_mm,
                coverage_target=cfg.coverage_target,
                max_burns=max(cfg.max_burns, 40),
                force_zone_mm=cfg.force_zone_mm,
                seed=seed,
                case_id=geometry.case_id,
            )
            # Prefer restarting with pure greedy (more reliable than
            # continuing a drifted BC trajectory on a capped applicator).
            traj_g, _ = rollout(env_g, policy="greedy", seed=seed)
            v_g = evaluate_trajectory(
                traj_g, cfg.coverage_target, cfg.max_overtreat_mL)
            if not v_g:
                traj = traj_g
                repaired = True
                note = "repaired by greedy restart: " + "; ".join(violations)
                violations = []
            else:
                # last resort: classical optimiser (uncapped analytic zone)
                plan = plan_ablation(
                    geometry.tumor_axes_mm,
                    margin_mm=geometry.margin_mm,
                    device=device,
                    spacing_mm=cfg.spacing_mm,
                    coverage_target=cfg.coverage_target,
                    max_burns=cfg.max_burns,
                )
                traj = plan_to_trajectory(
                    plan, case_id=geometry.case_id, geometry=geometry)
                repaired = True
                note = ("repaired by full replan after greedy failed: "
                        + "; ".join(violations + v_g))
                violations = evaluate_trajectory(
                    traj, cfg.coverage_target, cfg.max_overtreat_mL)
        elif cfg.repair == "replan":
            plan = plan_ablation(
                geometry.tumor_axes_mm,
                margin_mm=geometry.margin_mm,
                device=device,
                spacing_mm=cfg.spacing_mm,
                coverage_target=cfg.coverage_target,
                max_burns=cfg.max_burns,
            )
            traj = plan_to_trajectory(plan, case_id=geometry.case_id,
                                      geometry=geometry)
            repaired = True
            note = "repaired by full replan: " + "; ".join(violations)
            violations = evaluate_trajectory(
                traj, cfg.coverage_target, cfg.max_overtreat_mL)
        else:
            note = f"unknown repair mode {cfg.repair}"

        # legacy path removed — cascade handles greedy_continue

    accepted = len(violations) == 0
    if accepted and repaired:
        note = note  # keep repair note
    elif accepted:
        note = "policy plan accepted"
    else:
        note = "still failing after repair: " + "; ".join(violations)

    traj.meta["gate"] = {
        "accepted": accepted, "repaired": repaired,
        "violations": violations, "note": note,
        "repair_mode": cfg.repair,
    }
    return GateResult(
        accepted=accepted, repaired=repaired, trajectory=traj,
        metrics=dict(traj.metrics), violations=violations, note=note,
    )


def gated_policy_callable(policy, gate_cfg: GateConfig | None = None):
    """Wrap a BC policy so each *episode* is gated (for evaluation loops).

    For per-step use inside the env, prefer ``gate_rollout`` which owns the
    full episode.  This helper returns a step-level policy that falls back to
    greedy when the BC action would be far from the uncovered region.
    """
    cfg = gate_cfg or GateConfig()

    def _act(obs, env: AblationSimEnv) -> AblationAction:
        a = policy.act(obs, env) if hasattr(policy, "act") else policy(obs, env)
        # Soft safety: if proposed position is far from uncovered tissue,
        # snap toward greedy centre
        uc = np.asarray(obs.get("uncovered_centroid_mm") or (0, 0, 0), float)
        pos = np.asarray(a.position_mm, float)
        if np.linalg.norm(pos - uc) > 15.0 and obs.get("uncovered_fraction", 1) > 0.05:
            g = env.greedy_action()
            return AblationAction(g.position_mm, a.power_W, a.time_s)
        return a

    _act.gate_cfg = cfg  # type: ignore[attr-defined]
    return _act
