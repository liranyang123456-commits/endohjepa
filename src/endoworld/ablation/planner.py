"""Ablation treatment planner.

Given a nodule's geometry (semi-axes) and anatomical constraints, compute a plan that
covers the tumour PLUS a safety margin with one or more overlapping ablation zones,
choosing per-burn power/time, an insertion trajectory, and a burn sequence, while
limiting damage to healthy tissue.

Coverage is optimised on a local voxel grid via greedy farthest-point placement of
ablation ellipsoids (each sized by the literature-calibrated device model). This is the
classic "cover the target with overlapping thermal spheres" formulation used in the
ablation-planning literature (multi-antenna / multi-burn coverage).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from endoworld.ablation.bioheat import analytic_zone_axes_mm, AXIAL_RATIO

# Device presets: (min_power, max_power, default_power_W, typical times s)
DEVICES = {
    "MWA": {"power_W": 50.0, "times_s": (300, 420, 600), "pmin": 30, "pmax": 100},
    "RFA": {"power_W": 40.0, "times_s": (600, 720, 900), "pmin": 20, "pmax": 60},
    "CRYO": {"power_W": 0.0, "times_s": (600,), "pmin": 0, "pmax": 0},
}


@dataclass
class Burn:
    center_mm: tuple           # (x,y,z) in tumour-local frame (origin=tumour centroid)
    power_W: float
    time_s: float
    zone_axes_mm: tuple        # (rt, rt, ra) ablation ellipsoid semi-axes
    order: int = 0             # ablation sequence index


@dataclass
class Trajectory:
    approach: str              # "percutaneous" | "transbronchial"
    entry_mm: tuple            # entry point in local frame
    target_mm: tuple          # target (tumour centroid)
    insertion_depth_mm: float
    note: str = ""


@dataclass
class AblationPlan:
    device: str
    margin_mm: float
    tumor_axes_mm: tuple
    burns: list = field(default_factory=list)
    trajectory: Trajectory | None = None
    metrics: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
def _voxelize_ellipsoid(shape, spacing, center_vox, semi_axes_mm):
    zc = np.stack(np.meshgrid(
        (np.arange(shape[0]) - center_vox[0]) * spacing,
        (np.arange(shape[1]) - center_vox[1]) * spacing,
        (np.arange(shape[2]) - center_vox[2]) * spacing, indexing="ij"), -1)
    a, b, c = [max(s, 1e-3) for s in semi_axes_mm]
    return (zc[..., 0]**2 / a**2 + zc[..., 1]**2 / b**2 + zc[..., 2]**2 / c**2) <= 1.0


def _greedy_cover(target_mask, grid, shape, center, spacing_mm, zone,
                  power, time_s, coverage_target, max_burns):
    from scipy import ndimage
    covered = np.zeros(shape, bool)
    burns: list[Burn] = []
    for _ in range(max_burns):
        if (covered & target_mask).sum() / max(target_mask.sum(), 1) >= coverage_target:
            break
        uncovered = target_mask & ~covered
        if not uncovered.any():
            break
        dist = ndimage.distance_transform_edt(uncovered)
        idx = np.unravel_index(np.argmax(dist), dist.shape)
        covered |= _voxelize_ellipsoid(shape, spacing_mm, idx, zone)
        burns.append(Burn(center_mm=tuple(round(float(x), 1) for x in grid[idx]),
                          power_W=power, time_s=time_s,
                          zone_axes_mm=tuple(round(float(x), 1) for x in zone)))
    return burns, covered


def plan_ablation(tumor_axes_mm, margin_mm=5.0, device="MWA",
                  spacing_mm=1.0, coverage_target=0.99, max_burns=8,
                  constraints: dict | None = None,
                  w_overtreat=1.0, w_time=0.15) -> AblationPlan:
    """Plan burns to cover tumour+margin, optimising power/time.

    Searches candidate (power, time) settings; for each, greedily covers the target
    with that zone size, then selects the plan minimising
        cost = w_overtreat * healthy_overtreated_mL + w_time * total_time_min
    subject to target coverage >= coverage_target. This trades off sparing healthy
    tissue (small zones) against procedure duration / number of burns (large zones).
    """
    constraints = constraints or {}
    dev = DEVICES[device]
    a, b, c = tumor_axes_mm
    target = (a + margin_mm, b + margin_mm, c + margin_mm)

    zt = analytic_zone_axes_mm(dev["pmax"], dev["times_s"][-1])
    half = np.array(target) + np.array(zt) + 4
    shape = tuple(int(2 * h / spacing_mm) + 1 for h in half)
    center = tuple(s // 2 for s in shape)
    tumor = _voxelize_ellipsoid(shape, spacing_mm, center, tumor_axes_mm)
    target_mask = _voxelize_ellipsoid(shape, spacing_mm, center, target)
    grid = np.stack(np.meshgrid(
        (np.arange(shape[0]) - center[0]) * spacing_mm,
        (np.arange(shape[1]) - center[1]) * spacing_mm,
        (np.arange(shape[2]) - center[2]) * spacing_mm, indexing="ij"), -1)

    powers = sorted({dev["pmin"], int(dev["power_W"]), dev["pmax"],
                     (dev["pmin"] + int(dev["power_W"])) // 2})
    best = None
    for power in powers:
        for time_s in dev["times_s"]:
            zone = analytic_zone_axes_mm(power, time_s)
            if min(zone) < 3:
                continue
            burns, _ = _greedy_cover(target_mask, grid, shape, center, spacing_mm,
                                     zone, power, time_s, coverage_target, max_burns)
            if not burns:
                continue
            m = evaluate_plan(tumor, target_mask, burns, shape, center, spacing_mm)
            if m["target_coverage_incl_margin"] < coverage_target - 1e-6:
                continue
            cost = w_overtreat * m["healthy_overtreated_mL"] + w_time * m["total_ablation_time_min"]
            if best is None or cost < best[0]:
                best = (cost, burns, m)

    if best is None:  # fallback: max settings
        zone = analytic_zone_axes_mm(dev["pmax"], dev["times_s"][-1])
        burns, _ = _greedy_cover(target_mask, grid, shape, center, spacing_mm,
                                 zone, dev["pmax"], dev["times_s"][-1], 0.0, max_burns)
        m = evaluate_plan(tumor, target_mask, burns, shape, center, spacing_mm)
        best = (0, burns, m)

    _, burns, metrics = best
    burns.sort(key=lambda bn: bn.center_mm[2])
    for k, bn in enumerate(burns):
        bn.order = k + 1

    plan = AblationPlan(device=device, margin_mm=margin_mm,
                        tumor_axes_mm=tuple(round(x, 1) for x in tumor_axes_mm),
                        burns=burns)
    plan.trajectory = _plan_trajectory(tumor_axes_mm, constraints, burns)
    plan.metrics = metrics
    return plan


def _plan_trajectory(tumor_axes_mm, constraints, burns) -> Trajectory:
    """Heuristic insertion path from anatomical constraints in the record."""
    gen = constraints.get("airway_generation")
    dist_pleura = constraints.get("dist_pleura_mm")
    dist_wall = constraints.get("dist_chestwall_mm")
    # transbronchial if reachable via a not-too-peripheral airway generation
    transbronchial = gen is not None and gen <= 8 and (dist_pleura or 99) > 3
    a, b, c = tumor_axes_mm
    if transbronchial:
        entry = (0.0, 0.0, -(c + 40.0))          # approach along airway (-z)
        note = f"经支气管路径 (气道第{gen}级, 节段{constraints.get('bronchial_segment')})"
        depth = float(c + 40.0)
        approach = "transbronchial"
    else:
        # percutaneous: enter from nearest chest wall side (+x), avoid vessels
        entry = (a + (dist_wall or 20.0) + 15.0, 0.0, 0.0)
        note = "经皮穿刺路径 (经胸壁, 避开血管/叶间裂)"
        depth = float(entry[0])
        approach = "percutaneous"
    return Trajectory(approach=approach, entry_mm=tuple(round(x, 1) for x in entry),
                      target_mm=(0.0, 0.0, 0.0), insertion_depth_mm=round(depth, 1),
                      note=note)


def evaluate_plan(tumor, target_mask, burns, shape, center, spacing_mm) -> dict:
    covered = np.zeros(shape, bool)
    for bn in burns:
        idx = tuple(int(round(center[d] + bn.center_mm[d] / spacing_mm)) for d in range(3))
        covered |= _voxelize_ellipsoid(shape, spacing_mm, idx, bn.zone_axes_mm)
    vox = spacing_mm**3 / 1000.0                 # mL per voxel
    tumor_cov = (covered & tumor).sum() / max(tumor.sum(), 1)
    target_cov = (covered & target_mask).sum() / max(target_mask.sum(), 1)
    overtreat = (covered & ~target_mask).sum() * vox
    ablated_vol = covered.sum() * vox
    tumor_vol = tumor.sum() * vox
    return {
        "n_burns": len(burns),
        "tumor_coverage": round(float(tumor_cov), 4),
        "target_coverage_incl_margin": round(float(target_cov), 4),
        "tumor_volume_mL": round(float(tumor_vol), 2),
        "ablated_volume_mL": round(float(ablated_vol), 2),
        "healthy_overtreated_mL": round(float(overtreat), 2),
        "total_ablation_time_min": round(sum(bn.time_s for bn in burns) / 60.0, 1),
    }


def plan_to_dict(plan: AblationPlan) -> dict:
    d = asdict(plan)
    return d
