"""Transparent risk-aware ranking of alternative in-silico ablation plans.

The uncertainty estimate is a bounded, scenario-based analysis: lesion
coverage is recomputed after perturbing delivered power, dwell time, perfusion,
and residual zone-model size. It is not a calibrated probability of clinical
success and is labelled accordingly in the returned record.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from endoworld.ablation.executable_plan import build_executable_plan
from endoworld.ablation.bioheat import analytic_zone_axes_mm
from endoworld.ablation.trajectory_schema import LesionGeometry


def _target_points(geometry: LesionGeometry) -> np.ndarray:
    """Return voxel-centre points inside the margin-included target ellipsoid."""
    axes = np.asarray(geometry.tumor_axes_mm, dtype=float) + geometry.margin_mm
    spacing = max(1.0, float(geometry.spacing_mm))
    coordinates = [
        np.arange(-axis, axis + spacing * 0.5, spacing) for axis in axes
    ]
    mesh = np.meshgrid(*coordinates, indexing="ij")
    points = np.column_stack([axis.ravel() for axis in mesh])
    inside = ((points / axes) ** 2).sum(axis=1) <= 1.0
    return points[inside]


def _coverage_for_perturbed_burns(
    plan: dict[str, Any],
    points: np.ndarray,
    *,
    power_scales: np.ndarray,
    time_scales: np.ndarray,
    perfusions: np.ndarray,
    zone_scales: np.ndarray,
) -> float:
    covered = np.zeros(len(points), dtype=bool)
    for index, burn in enumerate(plan["burns"]):
        center = np.asarray(burn["center_mm"], dtype=float)
        axes = np.asarray(
            analytic_zone_axes_mm(
                float(burn["power_W"]) * float(power_scales[index]),
                float(burn["time_s"]) * float(time_scales[index]),
                perfusion=float(perfusions[index]),
            ),
            dtype=float,
        ) * float(zone_scales[index])
        covered |= (((points - center) / axes) ** 2).sum(axis=1) <= 1.0
    return float(covered.mean()) if len(points) else 0.0


def model_form_uncertainty(
    plan: dict[str, Any],
    geometry: LesionGeometry,
    *,
    samples: int = 24,
    power_scale_range: tuple[float, float] = (0.90, 1.10),
    time_scale_range: tuple[float, float] = (0.90, 1.10),
    perfusion_range: tuple[float, float] = (0.004, 0.006),
    zone_scale_range: tuple[float, float] = (0.95, 1.05),
    seed: int = 0,
) -> dict[str, Any]:
    """Recompute coverage under bounded delivery and zone-model perturbations."""
    if samples < 3:
        raise ValueError("samples must be at least 3")
    for name, (low, high) in {
        "power_scale_range": power_scale_range,
        "time_scale_range": time_scale_range,
        "perfusion_range": perfusion_range,
        "zone_scale_range": zone_scale_range,
    }.items():
        if not 0 < low <= high:
            raise ValueError(f"{name} must contain positive increasing values")
    points = _target_points(geometry)
    rng = np.random.default_rng(seed)
    burn_count = len(plan["burns"])
    coverages = []
    for _ in range(samples):
        coverages.append(
            _coverage_for_perturbed_burns(
                plan,
                points,
                power_scales=rng.uniform(*power_scale_range, size=burn_count),
                time_scales=rng.uniform(*time_scale_range, size=burn_count),
                perfusions=rng.uniform(*perfusion_range, size=burn_count),
                zone_scales=rng.uniform(*zone_scale_range, size=burn_count),
            )
        )
    coverages = np.asarray(coverages)
    return {
        "method": "bounded Monte Carlo delivery and zone-model perturbation",
        "scope": (
            "Sensitivity to bounded power, dwell-time, perfusion and residual "
            "zone-model variation; not a calibrated clinical outcome probability."
        ),
        "samples": samples,
        "power_scale_range": list(power_scale_range),
        "time_scale_range": list(time_scale_range),
        "perfusion_range_per_s": list(perfusion_range),
        "zone_scale_range": list(zone_scale_range),
        "robust_coverage_rate": round(
            float(
                np.mean(
                    coverages >= float(plan["provenance"].get("coverage_target", 0.99))
                )
            ),
            4,
        ),
        "coverage_p05": round(float(np.quantile(coverages, 0.05)), 4),
        "coverage_p50": round(float(np.quantile(coverages, 0.50)), 4),
        "coverage_p95": round(float(np.quantile(coverages, 0.95)), 4),
        "coverage_min": round(float(coverages.min()), 4),
        "coverage_max": round(float(coverages.max()), 4),
    }


def _vessel_risk(plan: dict[str, Any]) -> float:
    check = next(
        (item for item in plan["constraint_checks"] if item["id"] == "vessel_clearance"),
        None,
    )
    if not check or check["status"] == "indeterminate":
        return 0.5
    if check["status"] == "fail":
        return 1.0
    value = float(check["value"])
    threshold = float(check["threshold"])
    return round(max(0.0, min(1.0, (threshold - value) / threshold + 0.15)), 4)


def _risk_components(
    plan: dict[str, Any],
    uncertainty: dict[str, Any],
    geometry: LesionGeometry,
) -> dict[str, float]:
    metrics = plan["metrics"]
    coverage_risk = max(0.0, 0.99 - float(uncertainty["coverage_p05"])) / 0.99
    healthy = float(metrics.get("healthy_overtreated_mL", 0.0))
    target_axes = np.asarray(geometry.tumor_axes_mm, dtype=float) + geometry.margin_mm
    target_volume = max(1.0, 4.0 / 3.0 * np.pi * float(np.prod(target_axes)) / 1000.0)
    overtreat_risk = min(1.0, healthy / target_volume)
    time_risk = min(1.0, float(metrics.get("total_ablation_time_min", 0.0)) / 30.0)
    return {
        "coverage_shortfall": round(coverage_risk, 4),
        "healthy_overtreatment": round(overtreat_risk, 4),
        "vessel_screening": _vessel_risk(plan),
        "procedure_time": round(time_risk, 4),
    }


def _score(components: dict[str, float]) -> float:
    weights = {
        "coverage_shortfall": 0.45,
        "healthy_overtreatment": 0.20,
        "vessel_screening": 0.25,
        "procedure_time": 0.10,
    }
    return round(sum(weights[key] * value for key, value in components.items()), 4)


def generate_candidates(
    geometry: LesionGeometry,
    *,
    device: str = "MWA",
    time_weights: Iterable[float] = (0.05, 0.15, 0.40),
) -> list[dict[str, Any]]:
    """Generate explainable alternatives by varying the time trade-off only."""
    candidates = []
    for index, time_weight in enumerate(time_weights, start=1):
        plan = build_executable_plan(geometry, device=device, w_time=float(time_weight))
        candidates.append(
            {
                "candidate_id": f"{device.lower()}_time_{index}",
                "configuration": {
                    "device": device,
                    "w_time": float(time_weight),
                    "w_overtreat": 1.0,
                },
                "plan": plan,
            }
        )
    return candidates


def rank_candidates(
    geometry: LesionGeometry,
    *,
    device: str = "MWA",
    samples: int = 24,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Rank candidates from lower to higher transparent composite risk."""
    ranked = []
    for index, candidate in enumerate(generate_candidates(geometry, device=device)):
        uncertainty = model_form_uncertainty(
            candidate["plan"], geometry, samples=samples, seed=seed + index
        )
        components = _risk_components(candidate["plan"], uncertainty, geometry)
        ranked.append(
            {
                **candidate,
                "uncertainty": uncertainty,
                "risk_components": components,
                "composite_risk": _score(components),
            }
        )
    ranked.sort(key=lambda item: item["composite_risk"])
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked
