"""Gym-style ablation simulation environment for trajectory learning.

Wraps the existing analytic zone model (+ optional Pennes FDM) and the
coverage planner into a sequential decision process:

    state  s_t  = uncovered target + ablated region + budget used
    action a_t  = (position_mm, power_W, time_s)   [optionally temperature]
    reward r_t  = Δcoverage − λ_over·Δovertreat − λ_time·Δtime
    done        when coverage ≥ γ  or  max burns reached  or  budget exhausted

A rollout produces an ``AblationTrajectory`` (see ``trajectory_schema``) that
can be mixed with clinical logs for imitation / offline RL / preference learning.

Usage
-----
    from endoworld.ablation.sim_env import AblationSimEnv, make_env_from_axes

    env = make_env_from_axes((8, 7, 9), margin_mm=5.0, device="MWA")
    obs, info = env.reset()
    done = False
    while not done:
        action = env.greedy_action()          # or a learned policy
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    traj = env.to_trajectory(case_id="rollout_001")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from endoworld.ablation.bioheat import analytic_zone_axes_mm
from endoworld.ablation.planner import (
    DEVICES,
    _voxelize_ellipsoid,
    evaluate_plan,
    plan_ablation,
)
from endoworld.ablation.trajectory_schema import (
    AblationTrajectory,
    BurnStep,
    DeviceParams,
    LesionGeometry,
    OutcomeLabel,
    plan_to_trajectory,
    save_mask,
)


# --------------------------------------------------------------------------- #
# Action / observation helpers
# --------------------------------------------------------------------------- #
@dataclass
class AblationAction:
    """One burn decision."""

    position_mm: tuple          # (x, y, z) tumour-local
    power_W: float
    time_s: float
    temperature_C: float | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "AblationAction":
        pos = d.get("position_mm") or d.get("position") or (0.0, 0.0, 0.0)
        return cls(
            position_mm=tuple(float(x) for x in pos),
            power_W=float(d["power_W"] if "power_W" in d else d.get("power", 45)),
            time_s=float(d["time_s"] if "time_s" in d else d.get("time", 300)),
            temperature_C=d.get("temperature_C"),
        )

    def as_dict(self) -> dict:
        return {
            "position_mm": list(self.position_mm),
            "power_W": float(self.power_W),
            "time_s": float(self.time_s),
            "temperature_C": self.temperature_C,
        }


@dataclass
class EnvConfig:
    margin_mm: float = 5.0
    spacing_mm: float = 1.5
    coverage_target: float = 0.99
    max_burns: int = 12
    max_total_time_s: float = 3600.0
    w_coverage: float = 1.0
    w_overtreat: float = 0.15
    w_time: float = 0.02          # per second
    w_step_penalty: float = 0.01  # small cost per burn
    use_fdm: bool = False         # if True, refine zone with Pennes (slow)
    fdm_spacing_mm: float = 2.0
    fdm_grid_mm: float = 70.0
    # Cap transverse zone radius (mm). None = use device analytic zone.
    # Set e.g. 10.0 to force multi-burn coverage (matches clinical single-zone limits).
    force_zone_mm: float | None = None
    seed: int | None = None


@dataclass
class EnvState:
    """Internal mutable state (not the observation vector)."""

    covered: np.ndarray
    tumor: np.ndarray
    target: np.ndarray
    steps: list = field(default_factory=list)       # BurnStep
    total_time_s: float = 0.0
    total_energy_J: float = 0.0
    t: int = 0


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
class AblationSimEnv:
    """Sequential multi-burn ablation environment.

    Observation (dict):
        coverage          float in [0,1]   — target (tumour+margin) coverage
        tumor_coverage    float
        overtreat_mL      float
        n_burns           int
        remaining_budget  float in [0,1]   — time budget left
        uncovered_centroid_mm  (3,)        — centroid of uncovered target
        uncovered_fraction     float
        last_zone_axes_mm      (3,) | None

    Action (dict or AblationAction):
        position_mm, power_W, time_s [, temperature_C]
    """

    metadata = {"render_modes": ["human", "ansi"], "name": "AblationSim-v0"}

    def __init__(
        self,
        geometry: LesionGeometry,
        device: DeviceParams | None = None,
        config: EnvConfig | None = None,
    ):
        self.geometry = geometry
        self.device = device or DeviceParams.from_device_name("MWA")
        self.cfg = config or EnvConfig(
            margin_mm=geometry.margin_mm,
            spacing_mm=geometry.spacing_mm,
        )
        self.rng = np.random.default_rng(self.cfg.seed)

        self._shape: tuple | None = None
        self._center: tuple | None = None
        self._grid: np.ndarray | None = None
        self._tumor: np.ndarray | None = None
        self._target: np.ndarray | None = None
        self._state: EnvState | None = None
        self._build_grid()

    # ---- grid construction ------------------------------------------------ #
    def _build_grid(self) -> None:
        a, b, c = self.geometry.tumor_axes_mm
        m = self.cfg.margin_mm
        sp = self.cfg.spacing_mm
        # pad with a large zone so multi-burn centres stay inside the grid
        zt = analytic_zone_axes_mm(
            self.device.max_power_W, max(self.device.time_presets_s))
        half = np.array([a + m, b + m, c + m]) + np.array(zt) + 6.0
        shape = tuple(int(2 * h / sp) + 1 for h in half)
        center = tuple(s // 2 for s in shape)
        tumor = _voxelize_ellipsoid(shape, sp, center, self.geometry.tumor_axes_mm)
        target = _voxelize_ellipsoid(
            shape, sp, center, (a + m, b + m, c + m))
        grid = np.stack(np.meshgrid(
            (np.arange(shape[0]) - center[0]) * sp,
            (np.arange(shape[1]) - center[1]) * sp,
            (np.arange(shape[2]) - center[2]) * sp,
            indexing="ij"), -1)
        self._shape, self._center, self._grid = shape, center, grid
        self._tumor, self._target = tumor, target

    # ---- Gym-like API ----------------------------------------------------- #
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.cfg.seed = seed
        assert self._tumor is not None and self._target is not None
        self._state = EnvState(
            covered=np.zeros(self._shape, dtype=bool),
            tumor=self._tumor.copy(),
            target=self._target.copy(),
            steps=[],
            total_time_s=0.0,
            total_energy_J=0.0,
            t=0,
        )
        obs = self._observe()
        info = self._info(reward_terms={})
        return obs, info

    def step(self, action: AblationAction | dict):
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if not isinstance(action, AblationAction):
            action = AblationAction.from_dict(action)

        power = self.device.clamp_power(action.power_W)
        time_s = float(max(1.0, action.time_s))
        zone = self._zone_for(power, time_s)

        prev = self._metrics_snapshot()
        self._apply_burn(action.position_mm, zone)
        step = BurnStep(
            step_index=len(self._state.steps) + 1,
            position_mm=tuple(round(float(x), 2) for x in action.position_mm),
            power_W=power,
            time_s=time_s,
            temperature_C=action.temperature_C,
            zone_axes_mm=tuple(round(float(x), 2) for x in zone),
            approach="unknown",
        )
        self._state.steps.append(step)
        self._state.total_time_s += time_s
        self._state.total_energy_J += power * time_s
        self._state.t += 1

        cur = self._metrics_snapshot()
        reward, terms = self._reward(prev, cur, time_s)
        terminated = cur["target_coverage"] >= self.cfg.coverage_target
        truncated = (
            self._state.t >= self.cfg.max_burns
            or self._state.total_time_s >= self.cfg.max_total_time_s
        )
        obs = self._observe()
        info = self._info(reward_terms=terms, zone=zone)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, mode: str = "ansi") -> str:
        m = self._metrics_snapshot()
        lines = [
            f"[AblationSim] t={self._state.t if self._state else 0}  "
            f"cov={m['target_coverage']*100:.1f}%  "
            f"tumor={m['tumor_coverage']*100:.1f}%  "
            f"over={m['overtreat_mL']:.2f}mL  "
            f"time={m['total_time_s']:.0f}s  "
            f"burns={m['n_burns']}",
        ]
        text = "\n".join(lines)
        if mode == "human":
            print(text)
        return text

    # ---- zone model ------------------------------------------------------- #
    def _zone_for(self, power_W: float, time_s: float) -> tuple:
        """Return (rt, rt, ra) mm; optionally refine with FDM / force-cap."""
        from endoworld.ablation.bioheat import AXIAL_RATIO
        if self.cfg.force_zone_mm is not None:
            rt = float(self.cfg.force_zone_mm)
            return (rt, rt, rt * AXIAL_RATIO)
        zone = analytic_zone_axes_mm(power_W, time_s)
        if not self.cfg.use_fdm:
            return zone
        # Optional slow path: run Pennes and replace transverse radius
        from endoworld.ablation.bioheat import (
            Applicator, Tissue, simulate_zone, zone_radii_mm,
        )
        mask, _, _ = simulate_zone(
            Tissue(),
            Applicator(
                power_W=power_W, time_s=time_s,
                tip_len_mm=self.device.tip_length_mm,
                lam_mm=self.device.lam_mm,
            ),
            spacing_mm=self.cfg.fdm_spacing_mm,
            grid_mm=self.cfg.fdm_grid_mm,
        )
        rx, ry, rz = zone_radii_mm(mask, self.cfg.fdm_spacing_mm)
        if rx > 0:
            return (rx, ry, max(rz, rx))
        return zone

    def _apply_burn(self, position_mm: tuple, zone: tuple) -> None:
        assert self._state and self._center and self._shape
        sp = self.cfg.spacing_mm
        idx = tuple(
            int(round(self._center[d] + position_mm[d] / sp))
            for d in range(3)
        )
        # clamp index into grid
        idx = tuple(
            int(np.clip(idx[d], 0, self._shape[d] - 1)) for d in range(3)
        )
        self._state.covered |= _voxelize_ellipsoid(
            self._shape, sp, idx, zone)

    # ---- metrics / reward / obs ------------------------------------------- #
    def _metrics_snapshot(self) -> dict:
        assert self._state is not None
        vox = self.cfg.spacing_mm ** 3 / 1000.0
        cov = self._state.covered
        tumor, target = self._state.tumor, self._state.target
        return {
            "n_burns": len(self._state.steps),
            "tumor_coverage": float((cov & tumor).sum() / max(tumor.sum(), 1)),
            "target_coverage": float((cov & target).sum() / max(target.sum(), 1)),
            "overtreat_mL": float((cov & ~target).sum() * vox),
            "ablated_mL": float(cov.sum() * vox),
            "tumor_mL": float(tumor.sum() * vox),
            "total_time_s": float(self._state.total_time_s),
            "total_energy_kJ": float(self._state.total_energy_J / 1000.0),
        }

    def _reward(self, prev: dict, cur: dict, dt: float) -> tuple[float, dict]:
        dcov = cur["target_coverage"] - prev["target_coverage"]
        dover = cur["overtreat_mL"] - prev["overtreat_mL"]
        terms = {
            "coverage_gain": self.cfg.w_coverage * dcov,
            "overtreat_cost": -self.cfg.w_overtreat * dover,
            "time_cost": -self.cfg.w_time * dt,
            "step_penalty": -self.cfg.w_step_penalty,
        }
        # terminal bonus
        if cur["target_coverage"] >= self.cfg.coverage_target:
            terms["success_bonus"] = 0.5
        else:
            terms["success_bonus"] = 0.0
        return float(sum(terms.values())), terms

    def _uncovered_centroid(self) -> tuple:
        assert self._state and self._grid is not None
        unc = self._state.target & ~self._state.covered
        if not unc.any():
            return (0.0, 0.0, 0.0)
        pts = self._grid[unc]
        c = pts.mean(axis=0)
        return tuple(float(x) for x in c)

    def _observe(self) -> dict:
        m = self._metrics_snapshot()
        budget = 1.0 - min(1.0, m["total_time_s"] / self.cfg.max_total_time_s)
        unc = self._state.target & ~self._state.covered if self._state else None
        unc_frac = float(unc.sum() / max(self._state.target.sum(), 1)) if unc is not None else 1.0
        last_zone = None
        if self._state and self._state.steps:
            last_zone = self._state.steps[-1].zone_axes_mm
        return {
            "coverage": m["target_coverage"],
            "tumor_coverage": m["tumor_coverage"],
            "overtreat_mL": m["overtreat_mL"],
            "n_burns": m["n_burns"],
            "remaining_budget": budget,
            "uncovered_centroid_mm": self._uncovered_centroid(),
            "uncovered_fraction": unc_frac,
            "last_zone_axes_mm": last_zone,
            "tumor_axes_mm": tuple(self.geometry.tumor_axes_mm),
            "margin_mm": self.cfg.margin_mm,
        }

    def _info(self, reward_terms: dict, zone: tuple | None = None) -> dict:
        info = self._metrics_snapshot()
        info["reward_terms"] = reward_terms
        info["zone_axes_mm"] = zone
        info["coverage_target"] = self.cfg.coverage_target
        return info

    # ---- built-in policies (for demos / baselines) ------------------------ #
    def greedy_action(self) -> AblationAction:
        """Max-marginal-gain style: place next burn at farthest uncovered voxel.

        Uses the device default power/time (largest preset) so a single call
        sequence reproduces the planner's greedy cover behaviour.
        """
        assert self._state and self._grid is not None
        from scipy import ndimage
        unc = self._state.target & ~self._state.covered
        if not unc.any():
            return AblationAction((0.0, 0.0, 0.0),
                                  self.device.power_presets_W[-1],
                                  self.device.time_presets_s[-1])
        dist = ndimage.distance_transform_edt(unc)
        idx = np.unravel_index(np.argmax(dist), dist.shape)
        pos = tuple(float(x) for x in self._grid[idx])
        # choose mid-high power/time for efficiency
        pw = self.device.power_presets_W[len(self.device.power_presets_W) // 2]
        ts = self.device.time_presets_s[len(self.device.time_presets_s) // 2]
        return AblationAction(pos, float(pw), float(ts))

    def random_action(self) -> AblationAction:
        """Uniform random burn centre inside the target bounding box."""
        assert self._state and self._grid is not None
        unc = self._state.target & ~self._state.covered
        if unc.any():
            idxs = np.argwhere(unc)
            pick = idxs[self.rng.integers(0, len(idxs))]
            pos = tuple(float(x) for x in self._grid[tuple(pick)])
        else:
            pos = (0.0, 0.0, 0.0)
        pw = float(self.rng.choice(self.device.power_presets_W))
        ts = float(self.rng.choice(self.device.time_presets_s))
        return AblationAction(pos, pw, ts)

    # ---- export ----------------------------------------------------------- #
    def to_trajectory(self, case_id: str | None = None,
                      save_masks_dir: str | None = None) -> AblationTrajectory:
        if self._state is None:
            raise RuntimeError("No episode to export; call reset/step first.")
        cid = case_id or self.geometry.case_id or "rollout"
        geom = LesionGeometry(**{
            **{f: getattr(self.geometry, f)
               for f in LesionGeometry.__dataclass_fields__},
            "case_id": cid,
            "margin_mm": self.cfg.margin_mm,
            "spacing_mm": self.cfg.spacing_mm,
        })
        m = self._metrics_snapshot()
        if save_masks_dir:
            os_makedirs = __import__("os").makedirs
            os_makedirs(save_masks_dir, exist_ok=True)
            pre = __import__("os").path.join(save_masks_dir, f"{cid}_pre.npz")
            post = __import__("os").path.join(save_masks_dir, f"{cid}_post.npz")
            save_mask(pre, self._state.target, self.cfg.spacing_mm, label="target_pre")
            save_mask(post, self._state.covered, self.cfg.spacing_mm, label="ablated_post")
            geom.pre_mask_file = pre
            geom.post_mask_file = post

        # preference: high coverage, low overtreat, fewer burns
        pref = (
            m["target_coverage"]
            - 0.05 * m["overtreat_mL"]
            - 0.02 * m["n_burns"]
        )
        return AblationTrajectory(
            case_id=cid,
            device=self.device,
            geometry=geom,
            steps=list(self._state.steps),
            outcome=OutcomeLabel(
                verdict="simulated",
                pre_volume_mL=m["tumor_mL"],
                peak_volume_mL=m["ablated_mL"],
                late_volume_mL=None,
                preference_score=round(float(pref), 4),
                note="simulated rollout",
            ),
            source="simulated",
            metrics={
                "tumor_coverage": round(m["tumor_coverage"], 4),
                "target_coverage_incl_margin": round(m["target_coverage"], 4),
                "healthy_overtreated_mL": round(m["overtreat_mL"], 3),
                "ablated_volume_mL": round(m["ablated_mL"], 3),
                "n_burns": m["n_burns"],
                "total_ablation_time_min": round(m["total_time_s"] / 60.0, 2),
                "total_energy_kJ": round(m["total_energy_kJ"], 2),
            },
            meta={"env": "AblationSim-v0", "use_fdm": self.cfg.use_fdm},
        )

    def verify_with_planner(self) -> dict:
        """Compare greedy-env rollout metrics against ``plan_ablation``."""
        plan = plan_ablation(
            self.geometry.tumor_axes_mm,
            margin_mm=self.cfg.margin_mm,
            device=self.device.device_type,
            spacing_mm=self.cfg.spacing_mm,
            coverage_target=self.cfg.coverage_target,
            max_burns=self.cfg.max_burns,
        )
        return {
            "planner_metrics": dict(plan.metrics),
            "planner_n_burns": len(plan.burns),
            "env_metrics": self._metrics_snapshot() if self._state else {},
        }


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def make_env_from_axes(
    tumor_axes_mm: tuple,
    margin_mm: float = 5.0,
    device: str = "MWA",
    spacing_mm: float = 1.5,
    coverage_target: float = 0.99,
    max_burns: int = 12,
    use_fdm: bool = False,
    force_zone_mm: float | None = None,
    seed: int | None = None,
    case_id: str = "synthetic",
) -> AblationSimEnv:
    geom = LesionGeometry(
        case_id=case_id,
        tumor_axes_mm=tuple(float(x) for x in tumor_axes_mm),
        margin_mm=margin_mm,
        spacing_mm=spacing_mm,
    )
    cfg = EnvConfig(
        margin_mm=margin_mm,
        spacing_mm=spacing_mm,
        coverage_target=coverage_target,
        max_burns=max_burns,
        use_fdm=use_fdm,
        force_zone_mm=force_zone_mm,
        seed=seed,
    )
    return AblationSimEnv(geom, DeviceParams.from_device_name(device), cfg)


def make_env_from_record(
    row: dict,
    device: str = "MWA",
    margin_mm: float = 5.0,
    spacing_mm: float = 1.5,
    **kw,
) -> AblationSimEnv | None:
    from endoworld.ablation.trajectory_schema import geometry_from_record_row
    geom = geometry_from_record_row(row, margin_mm=margin_mm, spacing_mm=spacing_mm)
    if geom is None:
        return None
    cfg = EnvConfig(margin_mm=margin_mm, spacing_mm=spacing_mm, **{
        k: v for k, v in kw.items() if k in EnvConfig.__dataclass_fields__
    })
    return AblationSimEnv(geom, DeviceParams.from_device_name(device), cfg)


def rollout(
    env: AblationSimEnv,
    policy: str = "greedy",
    max_steps: int | None = None,
    seed: int | None = None,
) -> tuple[AblationTrajectory, list[dict]]:
    """Run one episode with a built-in or callable policy.

    ``policy``: ``\"greedy\"`` | ``\"random\"`` | callable(obs, env) -> action
    """
    obs, info = env.reset(seed=seed)
    history = [{"obs": obs, "info": info, "reward": 0.0, "action": None}]
    limit = max_steps or env.cfg.max_burns
    for _ in range(limit):
        if callable(policy):
            action = policy(obs, env)
        elif policy == "greedy":
            action = env.greedy_action()
        elif policy == "random":
            action = env.random_action()
        else:
            raise ValueError(f"Unknown policy: {policy}")
        obs, reward, term, trunc, info = env.step(action)
        history.append({
            "obs": obs, "info": info, "reward": reward,
            "action": action.as_dict() if isinstance(action, AblationAction)
            else dict(action),
        })
        if term or trunc:
            break
    traj = env.to_trajectory()
    return traj, history


def optimiser_demo_trajectory(
    tumor_axes_mm: tuple,
    margin_mm: float = 5.0,
    device: str = "MWA",
    case_id: str = "optimiser_demo",
) -> AblationTrajectory:
    """Produce a trajectory by running the classical planner (demonstration data)."""
    plan = plan_ablation(tumor_axes_mm, margin_mm=margin_mm, device=device)
    geom = LesionGeometry(
        case_id=case_id,
        tumor_axes_mm=tuple(plan.tumor_axes_mm),
        margin_mm=margin_mm,
    )
    return plan_to_trajectory(plan, case_id=case_id, geometry=geom)
