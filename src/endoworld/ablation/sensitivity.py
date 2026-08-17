"""One-at-a-time planning-input sensitivity analysis.

Only variables accepted by the current geometric planner are perturbed here.
The report deliberately separates this deterministic input analysis from the
zone-model uncertainty analysis in :mod:`risk_ranking`.
"""
from __future__ import annotations

from typing import Any

from endoworld.ablation.executable_plan import build_executable_plan
from endoworld.ablation.trajectory_schema import LesionGeometry


def _summary(plan: dict[str, Any]) -> dict[str, float]:
    metrics = plan["metrics"]
    return {
        "target_coverage": float(metrics.get("target_coverage_incl_margin", 0.0)),
        "healthy_overtreated_mL": float(metrics.get("healthy_overtreated_mL", 0.0)),
        "total_ablation_time_min": float(metrics.get("total_ablation_time_min", 0.0)),
        "burn_count": float(len(plan["burns"])),
    }


def _delta(scenario: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {
        key: round(scenario[key] - baseline[key], 4)
        for key in baseline
    }


def local_sensitivity_report(
    geometry: LesionGeometry,
    *,
    device: str = "MWA",
    margin_delta_mm: float = 2.0,
    coverage_delta: float = 0.02,
    time_weight_multiplier: float = 2.0,
) -> dict[str, Any]:
    """Evaluate each configurable planner input while holding others fixed."""
    if margin_delta_mm <= 0 or coverage_delta <= 0 or time_weight_multiplier <= 1:
        raise ValueError("Sensitivity deltas must be positive and multiplier must exceed one.")
    baseline_plan = build_executable_plan(geometry, device=device)
    baseline = _summary(baseline_plan)
    scenarios: list[dict[str, Any]] = []

    for label, parameter, value, adjusted_geometry, kwargs in [
        (
            "margin_minus",
            "margin_mm",
            geometry.margin_mm - margin_delta_mm,
            LesionGeometry(
                **{**geometry.__dict__, "margin_mm": max(0.0, geometry.margin_mm - margin_delta_mm)}
            ),
            {},
        ),
        (
            "margin_plus",
            "margin_mm",
            geometry.margin_mm + margin_delta_mm,
            LesionGeometry(**{**geometry.__dict__, "margin_mm": geometry.margin_mm + margin_delta_mm}),
            {},
        ),
        (
            "coverage_target_lower",
            "coverage_target",
            0.99 - coverage_delta,
            geometry,
            {"coverage_target": 0.99 - coverage_delta},
        ),
        (
            "coverage_target_higher",
            "coverage_target",
            min(0.999, 0.99 + coverage_delta),
            geometry,
            {"coverage_target": min(0.999, 0.99 + coverage_delta)},
        ),
        (
            "time_weight_lower",
            "w_time",
            0.15 / time_weight_multiplier,
            geometry,
            {"w_time": 0.15 / time_weight_multiplier},
        ),
        (
            "time_weight_higher",
            "w_time",
            0.15 * time_weight_multiplier,
            geometry,
            {"w_time": 0.15 * time_weight_multiplier},
        ),
    ]:
        plan = build_executable_plan(adjusted_geometry, device=device, **kwargs)
        summary = _summary(plan)
        scenarios.append(
            {
                "scenario_id": label,
                "parameter": parameter,
                "value": value,
                "metrics": summary,
                "delta_from_baseline": _delta(summary, baseline),
                "hard_failures": plan["provenance"]["hard_failures"],
            }
        )

    tornado = []
    for metric in baseline:
        largest = max(scenarios, key=lambda item: abs(item["delta_from_baseline"][metric]))
        tornado.append(
            {
                "metric": metric,
                "max_abs_delta": largest["delta_from_baseline"][metric],
                "driver": largest["scenario_id"],
            }
        )
    return {
        "case_id": geometry.case_id,
        "method": "one-at-a-time deterministic planning-input analysis",
        "scope_note": (
            "This analysis perturbs planner inputs only. Zone-model uncertainty "
            "and anatomy-space collision checks are reported separately."
        ),
        "baseline": baseline,
        "scenarios": scenarios,
        "tornado": tornado,
    }
