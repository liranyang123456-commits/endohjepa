"""Tests for auditable in-silico executable-plan records.

Run with:
    PYTHONPATH=src python -m endoworld.ablation.tests_executable
"""
from __future__ import annotations

from tempfile import TemporaryDirectory

def test_executable_plan_contains_serialised_actions_and_checks() -> None:
    from endoworld.ablation.executable_plan import build_executable_plan
    from endoworld.ablation.trajectory_schema import LesionGeometry

    geometry = LesionGeometry(
        case_id="test_plan",
        tumor_axes_mm=(7.0, 6.0, 8.0),
        margin_mm=5.0,
        spacing_mm=2.0,
        dist_vessel_mm=12.0,
        dist_chestwall_mm=18.0,
    )
    result = build_executable_plan(geometry, max_burns=6)
    assert result["schema_version"] == 2
    assert result["burns"]
    assert result["trajectory"]["waypoints_mm"][0] == list(
        result["trajectory"]["entry_mm"]
    )
    assert result["trajectory"]["waypoints_mm"][-1] == list(
        result["trajectory"]["target_mm"]
    )
    checks = {item["id"]: item for item in result["constraint_checks"]}
    assert checks["target_coverage"]["status"] == "pass"
    assert checks["burn_count"]["status"] == "pass"
    assert checks["vessel_clearance"]["status"] == "pass"
    assert checks["path_collision"]["status"] == "indeterminate"
    assert result["clinical_verification_required"] is True
    print("[test] executable plan serialisation OK")


def test_missing_vessel_information_is_not_reported_as_safe() -> None:
    from endoworld.ablation.executable_plan import build_executable_plan
    from endoworld.ablation.trajectory_schema import LesionGeometry

    result = build_executable_plan(
        LesionGeometry(
            case_id="missing_vessel",
            tumor_axes_mm=(6.0, 5.0, 6.0),
            spacing_mm=2.0,
        )
    )
    checks = {item["id"]: item for item in result["constraint_checks"]}
    assert checks["vessel_clearance"]["status"] == "indeterminate"
    assert result["clinical_verification_required"] is True
    print("[test] missing vessel information is indeterminate OK")


def test_risk_ranking_returns_ordered_uncertainty_aware_candidates() -> None:
    from endoworld.ablation.risk_ranking import rank_candidates
    from endoworld.ablation.trajectory_schema import LesionGeometry

    ranked = rank_candidates(
        LesionGeometry(
            case_id="ranking_case",
            tumor_axes_mm=(6.0, 5.0, 7.0),
            margin_mm=5.0,
            spacing_mm=2.0,
            dist_vessel_mm=10.0,
        ),
        samples=8,
        seed=7,
    )
    assert len(ranked) == 3
    assert [item["rank"] for item in ranked] == [1, 2, 3]
    assert [item["composite_risk"] for item in ranked] == sorted(
        item["composite_risk"] for item in ranked
    )
    for item in ranked:
        uncertainty = item["uncertainty"]
        assert "coverage_p05" in uncertainty
        assert uncertainty["power_scale_range"] == [0.9, 1.1]
        assert uncertainty["time_scale_range"] == [0.9, 1.1]
        assert uncertainty["perfusion_range_per_s"] == [0.004, 0.006]
        assert 0.0 <= uncertainty["robust_coverage_rate"] <= 1.0
    print("[test] uncertainty-aware risk ranking OK")


def test_local_sensitivity_returns_auditable_scenarios() -> None:
    from endoworld.ablation.sensitivity import local_sensitivity_report
    from endoworld.ablation.trajectory_schema import LesionGeometry

    report = local_sensitivity_report(
        LesionGeometry(
            case_id="sensitivity_case",
            tumor_axes_mm=(6.0, 5.0, 7.0),
            margin_mm=5.0,
            spacing_mm=2.0,
        )
    )
    assert len(report["scenarios"]) == 6
    assert {item["parameter"] for item in report["scenarios"]} == {
        "margin_mm",
        "coverage_target",
        "w_time",
    }
    assert len(report["tornado"]) == 4
    print("[test] local sensitivity report OK")


def test_collaboration_log_exports_deidentified_summary() -> None:
    from endoworld.ablation.collab_log import append_event, export_study_bundle

    with TemporaryDirectory() as temp_dir:
        append_event(temp_dir, session_id="session_1", event_type="task_started")
        append_event(temp_dir, session_id="session_1", event_type="plan_generated")
        append_event(temp_dir, session_id="session_1", event_type="plan_edited")
        append_event(temp_dir, session_id="session_1", event_type="plan_accepted")
        append_event(temp_dir, session_id="session_1", event_type="task_completed")
        bundle = export_study_bundle(temp_dir, "session_1")
    assert bundle["manifest"]["contains_identifiers"] is False
    assert bundle["summary"]["plan_override_count"] == 1
    assert bundle["summary"]["plan_acceptance_count"] == 1
    assert len(bundle["events"]) == 5
    print("[test] collaboration-log export OK")


def main() -> int:
    test_executable_plan_contains_serialised_actions_and_checks()
    test_missing_vessel_information_is_not_reported_as_safe()
    test_risk_ranking_returns_ordered_uncertainty_aware_candidates()
    test_local_sensitivity_returns_auditable_scenarios()
    test_collaboration_log_exports_deidentified_summary()
    print("[ALL EXECUTABLE-PLAN TESTS PASSED]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
