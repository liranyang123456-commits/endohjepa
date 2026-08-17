"""Patient-specific simulation: 术前3D → burn steps → 合成术后区.

Uses Cohort-A masks from ``segment3d`` (lung / nodule / target / vessel) when
available; otherwise falls back to tabular ellipsoid geometry.

For each case:
  1. Load pre-op nodule (+margin) mask as planning target.
  2. Roll out greedy (or BC) burns in ``AblationSimEnv`` (optionally
     force-capped applicator for multi-burn).
  3. Rasterise the union of burn ellipsoids onto the patient grid → synthetic
     ``M_post``.
  4. Export trajectory JSON + pre/post NPZ (schema-compatible).

Vessel voxels overlapping the ablation zone are reported as heat-sink risk
(not yet a hard constraint in the analytic zone model).

    PYTHONPATH=src python -m endoworld.ablation.patient_sim --limit 3 \\
        --seg outputs/ablation_seg3d --out outputs/ablation_patient_sim
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from endoworld.ablation.sim_env import make_env_from_axes, rollout
from endoworld.ablation.trajectory_schema import (
    AblationTrajectory,
    DeviceParams,
    LesionGeometry,
    OutcomeLabel,
    load_mask,
    save_mask,
    save_trajectory,
)


def _axes_from_mask(mask: np.ndarray, spacing_zyx: tuple) -> tuple:
    """Approximate semi-axes (x,y,z) mm from a binary mask bbox."""
    if not mask.any():
        return (8.0, 7.0, 8.0)
    idx = np.argwhere(mask)
    ext = (idx.max(0) - idx.min(0) + 1).astype(float)
    # ext in voxels along Z,Y,X → mm; planner wants (x,y,z) ≈ (LR, AP, SI)
    z_mm = float(ext[0] * spacing_zyx[0] / 2)
    y_mm = float(ext[1] * spacing_zyx[1] / 2)
    x_mm = float(ext[2] * spacing_zyx[2] / 2)
    return (max(x_mm, 2.0), max(y_mm, 2.0), max(z_mm, 2.0))


def _rasterize_burns_on_grid(
    shape, spacing_zyx, center_zyx, burns,
) -> np.ndarray:
    """Paint analytic burn ellipsoids onto the patient (Z,Y,X) grid.

    Burn positions are in tumour-local millimetres (x,y,z) ↔ (X,Y,Z).
    Only a local bounding box around each burn is updated (memory-safe).
    """
    from endoworld.ablation.bioheat import analytic_zone_axes_mm
    covered = np.zeros(shape, dtype=bool)
    dz, py, px = spacing_zyx
    cz0, cy0, cx0 = [float(c) for c in center_zyx]
    for bn in burns:
        pos = bn.position_mm if hasattr(bn, "position_mm") else bn.center_mm
        bx, by, bz = [float(v) for v in pos]   # local mm → (X,Y,Z)
        if bn.zone_axes_mm:
            rt, _, ra = [float(v) for v in bn.zone_axes_mm]
        else:
            rt, _, ra = analytic_zone_axes_mm(bn.power_W, bn.time_s)
        # burn centre in voxel coordinates
        vc_z = cz0 + bz / max(dz, 1e-6)
        vc_y = cy0 + by / max(py, 1e-6)
        vc_x = cx0 + bx / max(px, 1e-6)
        # local bbox in voxels (±1.2× semi-axis)
        hz = int(np.ceil(1.2 * ra / max(dz, 1e-6))) + 1
        hy = int(np.ceil(1.2 * rt / max(py, 1e-6))) + 1
        hx = int(np.ceil(1.2 * rt / max(px, 1e-6))) + 1
        z0 = max(0, int(vc_z) - hz); z1 = min(shape[0], int(vc_z) + hz + 1)
        y0 = max(0, int(vc_y) - hy); y1 = min(shape[1], int(vc_y) + hy + 1)
        x0 = max(0, int(vc_x) - hx); x1 = min(shape[2], int(vc_x) + hx + 1)
        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            continue
        zz = (np.arange(z0, z1) - vc_z) * dz
        yy = (np.arange(y0, y1) - vc_y) * py
        xx = (np.arange(x0, x1) - vc_x) * px
        Z, Y, X = np.meshgrid(zz, yy, xx, indexing="ij")
        ell = (X ** 2 + Y ** 2) / max(rt, 1e-3) ** 2 + Z ** 2 / max(ra, 1e-3) ** 2
        covered[z0:z1, y0:y1, x0:x1] |= ell <= 1.0
    return covered


def simulate_case(
    meta: dict,
    out_dir: str,
    force_zone_mm: float | None = 10.0,
    device: str = "MWA",
    policy: str = "greedy",
    seed: int = 0,
) -> dict:
    cid = meta["case_id"]
    sp = tuple(meta["spacing_zyx_mm"])
    nodule, _, _ = load_mask(meta["masks"]["nodule"])
    target, _, _ = load_mask(meta["masks"]["target"])
    vessel, _, _ = load_mask(meta["masks"]["vessel"])
    lung, _, _ = load_mask(meta["masks"]["lung"])

    axes = _axes_from_mask(nodule, sp)
    # tumour-local env (fast planning)
    env = make_env_from_axes(
        axes, margin_mm=5.0, device=device, spacing_mm=float(np.mean(sp)),
        force_zone_mm=force_zone_mm, max_burns=40, seed=seed, case_id=cid,
        coverage_target=0.99,
    )
    # copy anatomy hints from record
    rec = meta.get("record") or {}
    for k_src, k_dst in (
        ("airway_generation", "airway_generation"),
        ("dist_pleura_mm", "dist_pleura_mm"),
        ("dist_chestwall_mm", "dist_chestwall_mm"),
        ("dist_vessel_mm", "dist_vessel_mm"),
        ("lobe", "lobe"),
    ):
        v = rec.get(k_src)
        if v not in (None, ""):
            try:
                setattr(env.geometry, k_dst, float(v) if k_src != "lobe" else v)
            except (TypeError, ValueError):
                setattr(env.geometry, k_dst, v)

    traj, _ = rollout(env, policy=policy, seed=seed)

    # rasterise onto patient grid (do not require lung intersection for coverage;
    # still report lung-clipped overtreat / heat-sink)
    cent = np.array(meta["centroid_zyx_vox"], dtype=float)
    post_raw = _rasterize_burns_on_grid(nodule.shape, sp, cent, traj.steps)
    post = post_raw & lung if lung.any() else post_raw
    # if lung clip wiped the zone (seed near lung edge / downsample), keep raw
    if post_raw.any() and post.sum() < 0.05 * post_raw.sum():
        post = post_raw

    vox_ml = float(np.prod(sp) / 1000.0)
    heat_sink_overlap_mL = float((post_raw & vessel).sum() * vox_ml)
    target_cov = float((post_raw & target).sum() / max(target.sum(), 1))
    nodule_cov = float((post_raw & nodule).sum() / max(nodule.sum(), 1))
    over_mL = float((post_raw & ~target).sum() * vox_ml)

    cdir = os.path.join(out_dir, cid)
    os.makedirs(cdir, exist_ok=True)
    pre_path = os.path.join(cdir, "M_pre.npz")
    post_path = os.path.join(cdir, "M_post_synth.npz")
    save_mask(pre_path, target, spacing_mm=float(np.mean(sp)), label="M_pre_target")
    save_mask(post_path, post, spacing_mm=float(np.mean(sp)), label="M_post_synthetic")

    geom = LesionGeometry(
        case_id=cid,
        tumor_axes_mm=axes,
        margin_mm=5.0,
        spacing_mm=float(np.mean(sp)),
        lobe=str(rec.get("lobe") or ""),
        airway_generation=_float_or_none(rec.get("airway_generation")),
        dist_pleura_mm=_float_or_none(rec.get("dist_pleura_mm")),
        dist_chestwall_mm=_float_or_none(rec.get("dist_chestwall_mm")),
        dist_vessel_mm=_float_or_none(rec.get("dist_vessel_mm")),
        pre_mask_file=pre_path,
        post_mask_file=post_path,
        lung_mask_file=meta["masks"]["lung"],
        vessel_mask_file=meta["masks"]["vessel"],
    )
    traj.geometry = geom
    traj.case_id = cid
    traj.source = "simulated"
    traj.device = DeviceParams.from_device_name(device)
    traj.outcome = OutcomeLabel(
        verdict="simulated",
        pre_volume_mL=round(float(nodule.sum()) * vox_ml, 3),
        peak_volume_mL=round(float(post.sum()) * vox_ml, 3),
        preference_score=round(target_cov - 0.04 * over_mL, 4),
        note="synthetic post from patient-grid burn rasterisation",
    )
    traj.metrics.update({
        "patient_target_coverage": round(target_cov, 4),
        "patient_nodule_coverage": round(nodule_cov, 4),
        "patient_overtreat_mL": round(over_mL, 3),
        "heat_sink_overlap_mL": round(heat_sink_overlap_mL, 3),
        "synth_post_mL": round(float(post.sum()) * vox_ml, 3),
    })
    traj.meta["patient_sim"] = {
        "force_zone_mm": force_zone_mm,
        "policy": policy,
        "seg_meta": meta.get("masks"),
    }
    tpath = save_trajectory(traj, os.path.join(cdir, "trajectory.json"))

    # preview
    _preview(os.path.join(cdir, "preview.png"), nodule, target, post, vessel, cent)

    summary = {
        "case_id": cid,
        "axes_mm": axes,
        "n_burns": traj.n_burns(),
        "patient_target_coverage": traj.metrics["patient_target_coverage"],
        "patient_overtreat_mL": traj.metrics["patient_overtreat_mL"],
        "heat_sink_overlap_mL": traj.metrics["heat_sink_overlap_mL"],
        "trajectory": tpath,
        "M_pre": pre_path,
        "M_post": post_path,
    }
    print(f"[{cid}] burns={summary['n_burns']}  "
          f"cov={summary['patient_target_coverage']}  "
          f"over={summary['patient_overtreat_mL']}mL  "
          f"heatsink={summary['heat_sink_overlap_mL']}mL")
    return summary


def _float_or_none(v):
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _preview(path, nodule, target, post, vessel, cent):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = int(round(cent[0]))
    z = int(np.clip(z, 0, nodule.shape[0] - 1))
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for a, m, title, color in (
        (ax[0], nodule, "M_pre nodule", "red"),
        (ax[1], target, "target (+margin)", "lime"),
        (ax[2], post, "M_post synth", "cyan"),
    ):
        a.imshow(nodule[z] * 0 + target[z] * 0.3 + post[z] * 0.5, cmap="gray")
        # show binary overlays on blank-ish
        canvas = np.zeros(nodule.shape[1:], dtype=float)
        canvas = nodule[z].astype(float) * 0.4 + target[z].astype(float) * 0.3
        a.imshow(canvas, cmap="gray", vmin=0, vmax=1)
        a.contour(m[z], levels=[0.5], colors=color, linewidths=1.2)
        if vessel.any():
            a.contour(vessel[z], levels=[0.5], colors="yellow", linewidths=0.4,
                      alpha=0.5)
        a.set_title(title)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", default="outputs/ablation_seg3d")
    ap.add_argument("--out", default="outputs/ablation_patient_sim")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--force-zone", type=float, default=10.0)
    ap.add_argument("--device", default="MWA")
    ap.add_argument("--policy", default="greedy", choices=["greedy", "random"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    idx_path = os.path.join(args.seg, "index.json")
    if not os.path.isfile(idx_path):
        raise SystemExit(f"Missing {idx_path}; run segment3d first.")
    index = json.load(open(idx_path, encoding="utf-8"))
    cases = index.get("cases") or []
    if args.cases:
        want = set(c.zfill(3) for c in args.cases)
        cases = [c for c in cases if c["case_id"] in want]
    if args.limit > 0:
        cases = cases[: args.limit]

    os.makedirs(args.out, exist_ok=True)
    fz = args.force_zone if args.force_zone > 0 else None
    results = []
    for meta in cases:
        try:
            results.append(simulate_case(
                meta, args.out, force_zone_mm=fz, device=args.device,
                policy=args.policy, seed=args.seed,
            ))
        except Exception as e:
            print(f"[{meta.get('case_id')}] FAIL: {e}")

    summary = {
        "n": len(results),
        "mean_coverage": round(float(np.mean([
            r["patient_target_coverage"] for r in results])), 4) if results else None,
        "mean_overtreat_mL": round(float(np.mean([
            r["patient_overtreat_mL"] for r in results])), 3) if results else None,
        "cases": results,
    }
    json.dump(summary, open(os.path.join(args.out, "index.json"), "w",
                            encoding="utf-8"), indent=2)
    print(f"[done] {len(results)} patient sims → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
