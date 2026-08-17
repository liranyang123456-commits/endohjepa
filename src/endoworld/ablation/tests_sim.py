"""Smoke tests for trajectory schema + AblationSimEnv.

Run:
    PYTHONPATH=src python -m endoworld.ablation.run_sim --smoke
    PYTHONPATH=src python -m endoworld.ablation.tests_sim
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np


def test_schema_roundtrip():
    from endoworld.ablation.trajectory_schema import (
        AblationTrajectory, BurnStep, DeviceParams, LesionGeometry,
        OutcomeLabel, dict_to_trajectory, trajectory_to_dict,
        save_trajectory, load_trajectory, plan_to_trajectory,
        geometry_from_record_row, save_mask, load_mask, SCHEMA_VERSION,
    )
    from endoworld.ablation.planner import plan_ablation

    traj = AblationTrajectory(
        case_id="t1",
        device=DeviceParams.from_device_name("MWA"),
        geometry=LesionGeometry(case_id="t1", tumor_axes_mm=(8, 7, 9)),
        steps=[BurnStep(1, (0, 0, 0), 45, 300, zone_axes_mm=(10, 10, 13))],
        outcome=OutcomeLabel(verdict="simulated", preference_score=0.8),
        source="simulated",
    )
    d = trajectory_to_dict(traj)
    assert d["schema_version"] == SCHEMA_VERSION
    t2 = dict_to_trajectory(d)
    assert t2.n_burns() == 1
    assert t2.geometry.tumor_axes_mm == (8, 7, 9)
    assert abs(t2.total_energy_kJ() - 45 * 300 / 1000) < 1e-9

    with tempfile.TemporaryDirectory() as td:
        p = save_trajectory(traj, os.path.join(td, "t.json"))
        t3 = load_trajectory(p)
        assert t3.case_id == "t1"
        mp = os.path.join(td, "m.npz")
        mask = np.zeros((5, 5, 5), bool)
        mask[2, 2, 2] = True
        save_mask(mp, mask, 1.5)
        m2, sp, _ = load_mask(mp)
        assert m2.sum() == 1 and sp == 1.5

    plan = plan_ablation((7, 6, 7), margin_mm=5.0, spacing_mm=2.0, max_burns=6)
    pt = plan_to_trajectory(plan, case_id="opt")
    assert pt.source == "optimiser"
    assert pt.n_burns() == len(plan.burns)
    assert pt.metrics.get("tumor_coverage") is not None

    row = {"case_id": "r1", "diam_axial_mm": 16, "diam_coronal_mm": 14,
           "diam_sagittal_mm": 18, "lobe": "左上肺叶", "airway_generation": "6"}
    g = geometry_from_record_row(row)
    assert g is not None
    assert abs(g.tumor_axes_mm[0] - 8) < 1e-6
    print("[test] schema_roundtrip OK")


def test_env_greedy_reaches_coverage():
    from endoworld.ablation.sim_env import make_env_from_axes, rollout

    env = make_env_from_axes(
        (6, 5, 6), margin_mm=5.0, spacing_mm=2.0,
        coverage_target=0.99, max_burns=10, seed=0)
    traj, hist = rollout(env, policy="greedy", seed=0)
    cov = traj.metrics["target_coverage_incl_margin"]
    assert cov >= 0.95, cov
    assert traj.n_burns() >= 1
    assert traj.outcome.preference_score is not None
    assert hist[-1]["info"]["target_coverage"] >= 0.95
    # rewards finite
    assert all(np.isfinite(h["reward"]) for h in hist)
    print(f"[test] greedy coverage={cov:.4f} burns={traj.n_burns()} OK")


def test_env_random_and_export_masks():
    from endoworld.ablation.sim_env import make_env_from_axes, rollout
    from endoworld.ablation.trajectory_schema import load_mask

    env = make_env_from_axes(
        (5, 5, 5), margin_mm=5.0, spacing_mm=2.0, max_burns=3, seed=2)
    traj, _ = rollout(env, policy="random", seed=2)
    assert traj.n_burns() >= 1
    with tempfile.TemporaryDirectory() as td:
        traj2 = env.to_trajectory(case_id="rnd", save_masks_dir=td)
        assert traj2.geometry.pre_mask_file and os.path.isfile(traj2.geometry.pre_mask_file)
        assert traj2.geometry.post_mask_file and os.path.isfile(traj2.geometry.post_mask_file)
        pre, _, _ = load_mask(traj2.geometry.pre_mask_file)
        post, _, _ = load_mask(traj2.geometry.post_mask_file)
        assert pre.any() and post.any()
    print("[test] random + mask export OK")


def test_optimiser_demo():
    from endoworld.ablation.sim_env import optimiser_demo_trajectory
    traj = optimiser_demo_trajectory((8, 7, 8), margin_mm=5.0, device="MWA")
    assert traj.source == "optimiser"
    assert traj.n_burns() >= 1
    assert traj.metrics.get("target_coverage_incl_margin", 0) >= 0.9
    print(f"[test] optimiser demo burns={traj.n_burns()} OK")


def test_action_clamp_and_step_api():
    from endoworld.ablation.sim_env import AblationAction, make_env_from_axes

    env = make_env_from_axes((5, 5, 5), spacing_mm=2.0, max_burns=2, seed=0)
    obs, info = env.reset()
    assert 0.0 <= obs["coverage"] <= 1.0
    # over-power should be clamped
    a = AblationAction((0, 0, 0), power_W=9999, time_s=300)
    obs2, r, term, trunc, info2 = env.step(a)
    assert env.device.min_power_W <= env._state.steps[0].power_W <= env.device.max_power_W
    assert isinstance(r, float)
    print("[test] action clamp + step API OK")


def main():
    test_schema_roundtrip()
    test_env_greedy_reaches_coverage()
    test_env_random_and_export_masks()
    test_optimiser_demo()
    test_action_clamp_and_step_api()
    print("[ALL TESTS PASSED]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
