"""Gate optional geometry teachers against SCARED ground truth.

Each input NPZ may contain ``pred_depth``, ``gt_depth``, ``pred_pose``,
``gt_pose``, ``confidence`` and ``reprojection_error``. Pose arrays use the
canonical local [v,w] twist convention. A teacher is accepted only when it
beats the supplied simple baseline on available depth and pose metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from endoworld.world.geometry import confidence_ece, depth_metrics, pose_metrics


def evaluate_pack(path: str | Path) -> dict:
    pack = np.load(path)
    report: dict = {"source": str(path)}
    if {"pred_depth", "gt_depth"} <= set(pack.files):
        report["depth"] = depth_metrics(pack["pred_depth"], pack["gt_depth"])
    if {"pred_pose", "gt_pose"} <= set(pack.files):
        report["pose"] = pose_metrics(pack["pred_pose"], pack["gt_pose"])
    if "reprojection_error" in pack.files:
        values = pack["reprojection_error"]
        report["reprojection_error_mean"] = float(np.nanmean(values))
        report["reprojection_error_median"] = float(np.nanmedian(values))
    if "confidence" in pack.files:
        if "correct" in pack.files:
            correct = pack["correct"]
        elif {"pred_depth", "gt_depth"} <= set(pack.files):
            p, t = pack["pred_depth"], pack["gt_depth"]
            correct = (np.maximum(p / np.maximum(t, 1e-8), t / np.maximum(p, 1e-8)) < 1.25)
        else:
            correct = np.zeros_like(pack["confidence"])
        report["confidence_ece"] = confidence_ece(pack["confidence"], correct)
    return report


def gate(candidate: dict, baseline: dict) -> dict:
    checks = {}
    if "depth" in candidate and "depth" in baseline:
        checks["depth_abs_rel"] = (
            candidate["depth"]["abs_rel"] < baseline["depth"]["abs_rel"])
        checks["depth_delta1"] = (
            candidate["depth"]["delta1"] > baseline["depth"]["delta1"])
    if "pose" in candidate and "pose" in baseline:
        checks["pose_translation"] = (
            candidate["pose"]["translation_rmse"]
            < baseline["pose"]["translation_rmse"])
        checks["pose_rotation"] = (
            candidate["pose"]["rotation_deg"] < baseline["pose"]["rotation_deg"])
    if "reprojection_error_mean" in candidate and "reprojection_error_mean" in baseline:
        checks["reprojection"] = (
            candidate["reprojection_error_mean"] < baseline["reprojection_error_mean"])
    return {
        "accepted": bool(checks) and all(checks.values()),
        "checks": checks,
        "rule": "all available geometry metrics must beat the simple baseline",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--name", action="append", default=[])
    parser.add_argument("--out", default="outputs/geometry_gate/report.json")
    args = parser.parse_args()
    baseline = evaluate_pack(args.baseline)
    candidates = {}
    for i, path in enumerate(args.candidate):
        name = args.name[i] if i < len(args.name) else Path(path).stem
        result = evaluate_pack(path)
        result["gate"] = gate(result, baseline)
        candidates[name] = result
    report = {
        "dataset": "SCARED",
        "baseline": baseline,
        "candidates": candidates,
        "selected": [
            name for name, result in candidates.items()
            if result["gate"]["accepted"]
        ],
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
