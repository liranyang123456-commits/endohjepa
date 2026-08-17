"""Build auditable, in-silico executable ablation-plan records.

This module is deliberately a thin layer over :mod:`planner`.  It does not
claim image-space collision checking when only structured record distances are
available.  Instead, every constraint check is labelled as pass, fail, or
indeterminate so a clinician or downstream navigation system can distinguish
computed evidence from information that still requires verification.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from endoworld.ablation.planner import AblationPlan, DEVICES, plan_ablation
from endoworld.ablation.trajectory_schema import DeviceParams, LesionGeometry, plan_to_trajectory


DEFAULT_VESSEL_CLEARANCE_MM = 5.0


def _check(
    check_id: str,
    status: str,
    message: str,
    *,
    value: float | str | None = None,
    threshold: float | str | None = None,
    evidence: str,
) -> dict[str, Any]:
    """Create a JSON-friendly check with explicit evidence provenance."""
    return {
        "id": check_id,
        "status": status,  # pass | fail | indeterminate
        "message": message,
        "value": value,
        "threshold": threshold,
        "evidence": evidence,
    }


def _waypoints(entry_mm: tuple, target_mm: tuple, count: int = 8) -> list[list[float]]:
    """Linearly sample the heuristic insertion path for downstream visualisation."""
    start = np.asarray(entry_mm, dtype=float)
    end = np.asarray(target_mm, dtype=float)
    return [
        [round(float(x), 2) for x in point]
        for point in np.linspace(start, end, max(2, int(count)))
    ]


def _trajectory_dict(plan: AblationPlan) -> dict[str, Any] | None:
    if plan.trajectory is None:
        return None
    trajectory = asdict(plan.trajectory)
    trajectory["waypoints_mm"] = _waypoints(
        plan.trajectory.entry_mm,
        plan.trajectory.target_mm,
    )
    trajectory["heuristic"] = True
    return trajectory


def _constraint_checks(
    plan: AblationPlan,
    geometry: LesionGeometry,
    *,
    coverage_target: float,
    max_burns: int,
    min_vessel_clearance_mm: float,
) -> list[dict[str, Any]]:
    metrics = plan.metrics
    checks = [
        _check(
            "target_coverage",
            "pass" if metrics.get("target_coverage_incl_margin", 0.0) >= coverage_target else "fail",
            "Voxelised target coverage meets the requested threshold."
            if metrics.get("target_coverage_incl_margin", 0.0) >= coverage_target
            else "Voxelised target coverage is below the requested threshold.",
            value=float(metrics.get("target_coverage_incl_margin", 0.0)),
            threshold=coverage_target,
            evidence="local geometric simulation",
        ),
        _check(
            "burn_count",
            "pass" if len(plan.burns) <= max_burns else "fail",
            "Number of planned activations is within the configured limit."
            if len(plan.burns) <= max_burns
            else "Number of planned activations exceeds the configured limit.",
            value=len(plan.burns),
            threshold=max_burns,
            evidence="planner output",
        ),
    ]
    vessel_distance = geometry.dist_vessel_mm
    if vessel_distance is None:
        checks.append(
            _check(
                "vessel_clearance",
                "indeterminate",
                "No vessel-distance field or vessel mask was supplied; collision and heat-sink risk are unverified.",
                threshold=min_vessel_clearance_mm,
                evidence="structured record unavailable",
            )
        )
    else:
        status = "pass" if vessel_distance >= min_vessel_clearance_mm else "fail"
        checks.append(
            _check(
                "vessel_clearance",
                status,
                "Recorded minimum vessel distance meets the screening threshold."
                if status == "pass"
                else "Recorded minimum vessel distance is below the screening threshold.",
                value=float(vessel_distance),
                threshold=min_vessel_clearance_mm,
                evidence="structured record distance; not a segment-wise collision check",
            )
        )
    checks.append(
        _check(
            "path_collision",
            "indeterminate",
            "The insertion path is a geometric heuristic. Spatial collision checking requires patient-specific masks.",
            evidence="no ray--mask intersection calculation",
        )
    )
    return checks


def build_executable_plan(
    geometry: LesionGeometry,
    *,
    device: str = "MWA",
    coverage_target: float = 0.99,
    max_burns: int = 8,
    w_overtreat: float = 1.0,
    w_time: float = 0.15,
    min_vessel_clearance_mm: float = DEFAULT_VESSEL_CLEARANCE_MM,
) -> dict[str, Any]:
    """Return a complete in-silico plan plus auditable feasibility checks.

    ``executable_in_simulation`` only means that all device presets, burn
    fields and geometric coverage metrics are serialised. It is never a claim
    that the plan is clinically executable without image-space verification.
    """
    if device not in DEVICES:
        raise ValueError(f"Unsupported device {device!r}; choose one of {sorted(DEVICES)}")
    constraints = {
        "airway_generation": geometry.airway_generation,
        "bronchial_segment": geometry.bronchial_segment,
        "dist_pleura_mm": geometry.dist_pleura_mm,
        "dist_chestwall_mm": geometry.dist_chestwall_mm,
        "dist_vessel_mm": geometry.dist_vessel_mm,
    }
    plan = plan_ablation(
        geometry.tumor_axes_mm,
        margin_mm=geometry.margin_mm,
        device=device,
        spacing_mm=geometry.spacing_mm,
        coverage_target=coverage_target,
        max_burns=max_burns,
        constraints=constraints,
        w_overtreat=w_overtreat,
        w_time=w_time,
    )
    checks = _constraint_checks(
        plan,
        geometry,
        coverage_target=coverage_target,
        max_burns=max_burns,
        min_vessel_clearance_mm=min_vessel_clearance_mm,
    )
    steps = []
    device_params = DeviceParams.from_device_name(device)
    for burn in plan.burns:
        preset_power, preset_time = device_params.nearest_preset(burn.power_W, burn.time_s)
        steps.append(
            {
                "order": burn.order,
                "center_mm": list(burn.center_mm),
                "power_W": burn.power_W,
                "time_s": burn.time_s,
                "zone_axes_mm": list(burn.zone_axes_mm),
                "preset_matched": preset_power == burn.power_W and preset_time == burn.time_s,
                "nearest_preset": {"power_W": preset_power, "time_s": preset_time},
            }
        )
    trajectory = plan_to_trajectory(plan, case_id=geometry.case_id, geometry=geometry)
    hard_failures = [check["id"] for check in checks if check["status"] == "fail"]
    return {
        "schema_version": 2,
        "case_id": geometry.case_id,
        "device": asdict(device_params),
        "geometry": asdict(geometry),
        "trajectory": _trajectory_dict(plan),
        "burns": steps,
        "metrics": dict(plan.metrics),
        "constraint_checks": checks,
        "executable_in_simulation": not hard_failures,
        "clinical_verification_required": any(
            check["status"] == "indeterminate" for check in checks
        ),
        "provenance": {
            "planner": "farthest-interior geometric coverage planner",
            "trajectory_schema": trajectory.schema_version,
            "coverage_target": coverage_target,
            "w_overtreat": w_overtreat,
            "w_time": w_time,
            "hard_failures": hard_failures,
        },
        "scope_note": (
            "This is an in-silico plan record. It is not a clinical prescription "
            "and requires patient-specific image-space trajectory verification."
        ),
    }
