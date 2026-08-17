"""Evaluate executable planning and risk-aware ranking on a geometry cohort.

Example:
    PYTHONPATH=src python -m endoworld.ablation.evaluate_intelligent_planning \
        --params manifests/nodule_params.csv --out outputs/intelligent_planning
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from endoworld.ablation.risk_ranking import rank_candidates
from endoworld.ablation.sensitivity import local_sensitivity_report
from endoworld.ablation.trajectory_schema import geometry_from_record_row


def _mean(rows: list[dict], key: str) -> float:
    return round(mean(float(row[key]) for row in rows), 4) if rows else 0.0


def evaluate(
    params_path: str | Path,
    out_dir: str | Path,
    *,
    samples: int = 24,
    include_sensitivity: bool = True,
) -> dict:
    """Run every valid structured geometry and write reproducible JSON outputs."""
    with Path(params_path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    geometries = [
        geometry_from_record_row(row, case_id=f"case_{index + 1:03d}")
        for index, row in enumerate(rows)
    ]
    geometries = [geometry for geometry in geometries if geometry is not None]
    cases = []
    for index, geometry in enumerate(geometries):
        ranked = rank_candidates(geometry, samples=samples, seed=index)
        selected = ranked[0]
        plan = selected["plan"]
        vessel_check = next(
            check for check in plan["constraint_checks"] if check["id"] == "vessel_clearance"
        )
        case = {
            "case_id": geometry.case_id,
            "selected_candidate_id": selected["candidate_id"],
            "ranked_candidate_count": len(ranked),
            "executable_in_simulation": plan["executable_in_simulation"],
            "clinical_verification_required": plan["clinical_verification_required"],
            "target_coverage": plan["metrics"]["target_coverage_incl_margin"],
            "healthy_overtreated_mL": plan["metrics"]["healthy_overtreated_mL"],
            "total_ablation_time_min": plan["metrics"]["total_ablation_time_min"],
            "burn_count": len(plan["burns"]),
            "robust_coverage_rate": selected["uncertainty"]["robust_coverage_rate"],
            "coverage_p05": selected["uncertainty"]["coverage_p05"],
            "composite_risk": selected["composite_risk"],
            "vessel_check_status": vessel_check["status"],
            "vessel_distance_mm": vessel_check["value"],
        }
        if include_sensitivity:
            case["sensitivity"] = local_sensitivity_report(geometry)
        cases.append(case)

    summary = {
        "evaluation_scope": (
            "Structured-geometry in-silico planning evaluation. This output does "
            "not validate clinical safety, treatment delivery, or patient outcomes."
        ),
        "uncertainty_model": {
            "power_scale_range": [0.90, 1.10],
            "time_scale_range": [0.90, 1.10],
            "perfusion_range_per_s": [0.004, 0.006],
            "residual_zone_scale_range": [0.95, 1.05],
            "samples_per_candidate": samples,
        },
        "n_cases": len(cases),
        "n_candidates_per_case": 3,
        "mean_target_coverage": _mean(cases, "target_coverage"),
        "mean_robust_coverage_rate": _mean(cases, "robust_coverage_rate"),
        "mean_coverage_p05": _mean(cases, "coverage_p05"),
        "mean_healthy_overtreated_mL": _mean(cases, "healthy_overtreated_mL"),
        "mean_total_ablation_time_min": _mean(cases, "total_ablation_time_min"),
        "mean_burn_count": _mean(cases, "burn_count"),
        "mean_composite_risk": _mean(cases, "composite_risk"),
        "simulation_executable_rate": round(
            mean(float(case["executable_in_simulation"]) for case in cases), 4
        )
        if cases
        else 0.0,
        "requires_clinical_verification_rate": round(
            mean(float(case["clinical_verification_required"]) for case in cases), 4
        )
        if cases
        else 0.0,
        "vessel_check_status_counts": {
            status: sum(case["vessel_check_status"] == status for case in cases)
            for status in ("pass", "fail", "indeterminate")
        },
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "case_results.json").write_text(
        json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="manifests/nodule_params.csv")
    parser.add_argument("--out", default="outputs/intelligent_planning")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--without-sensitivity", action="store_true")
    args = parser.parse_args()
    summary = evaluate(
        args.params,
        args.out,
        samples=args.samples,
        include_sensitivity=not args.without_sensitivity,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
