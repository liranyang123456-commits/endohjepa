"""Run ablation planning over the parsed nodule table and visualise the plans.

    python -m endoworld.ablation.run_planning --params manifests/nodule_params.csv \
        --device MWA --margin 5 --out outputs/ablation
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from endoworld.ablation.planner import plan_ablation, plan_to_dict


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def dims_from_row(row):
    """Semi-axes (mm) from planning sizes AP/SI/LR, else axial/coronal/sagittal diam."""
    ap, si, lr = _f(row.get("size_AP_mm")), _f(row.get("size_SI_mm")), _f(row.get("size_LR_mm"))
    if ap and si and lr:
        return (lr / 2, ap / 2, si / 2)  # (x=LR, y=AP, z=SI)
    da, dc, ds = _f(row.get("diam_axial_mm")), _f(row.get("diam_coronal_mm")), _f(row.get("diam_sagittal_mm"))
    if da and dc and ds:
        return (da / 2, dc / 2, ds / 2)
    return None


def visualize(plan, dims, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    a, b, c = dims
    m = plan.margin_mm
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    planes = [("Axial (x-y)", 0, 1, a, b), ("Coronal (x-z)", 0, 2, a, c),
              ("Sagittal (y-z)", 1, 2, b, c)]
    for ax, (title, i, j, ai, aj) in zip(axes, planes):
        # tumour + margin ellipses
        ax.add_patch(Ellipse((0, 0), 2*(ai+m), 2*(aj+m), fill=False, ls="--",
                             ec="tab:green", lw=1.6, label="tumor+margin"))
        ax.add_patch(Ellipse((0, 0), 2*ai, 2*aj, fc="tab:red", alpha=0.35,
                             ec="tab:red", label="tumor"))
        for bn in plan.burns:
            za = bn.zone_axes_mm
            ax.add_patch(Ellipse((bn.center_mm[i], bn.center_mm[j]),
                                 2*za[i], 2*za[j], fc="tab:blue", alpha=0.18,
                                 ec="tab:blue", lw=0.8))
            ax.plot(bn.center_mm[i], bn.center_mm[j], "x", color="navy", ms=6)
        lim = (max(ai, aj) + m + max(z for z in plan.burns[0].zone_axes_mm)) * 1.1 if plan.burns else (max(ai, aj)+m)*1.3
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal"); ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    mtr = plan.metrics
    fig.suptitle(f"{plan.device}  burns={mtr['n_burns']}  "
                 f"tumor_cov={mtr['tumor_coverage']*100:.1f}%  "
                 f"target_cov={mtr['target_coverage_incl_margin']*100:.1f}%  "
                 f"overtreat={mtr['healthy_overtreated_mL']:.1f}mL  "
                 f"time={mtr['total_ablation_time_min']}min", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="manifests/nodule_params.csv")
    ap.add_argument("--device", default="MWA", choices=["MWA", "RFA"])
    ap.add_argument("--margin", type=float, default=5.0)
    ap.add_argument("--out", default="outputs/ablation")
    ap.add_argument("--viz-n", type=int, default=6, help="how many cases to visualise")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "plans"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "figs"), exist_ok=True)

    rows = list(csv.DictReader(open(args.params, encoding="utf-8-sig")))
    summary = []
    n_viz = 0
    for row in rows:
        dims = dims_from_row(row)
        if not dims:
            continue
        cons = {k: row.get(k) for k in ("airway_generation", "bronchial_segment",
                                        "dist_pleura_mm", "dist_chestwall_mm", "dist_vessel_mm")}
        cons["airway_generation"] = int(_f(cons["airway_generation"])) if _f(cons["airway_generation"]) else None
        cons["dist_pleura_mm"] = _f(cons["dist_pleura_mm"])
        cons["dist_chestwall_mm"] = _f(cons["dist_chestwall_mm"])
        plan = plan_ablation(dims, margin_mm=args.margin, device=args.device, constraints=cons)
        cid = row["note"].split(".")[0]
        json.dump(plan_to_dict(plan), open(os.path.join(args.out, "plans", f"{cid}.json"), "w",
                                           encoding="utf-8"), ensure_ascii=False, indent=2)
        m = plan.metrics
        summary.append({
            "case": cid, "lobe": row.get("lobe"), "segment": row.get("bronchial_segment"),
            "tumor_mm": f"{dims[0]*2:.0f}x{dims[1]*2:.0f}x{dims[2]*2:.0f}",
            "approach": plan.trajectory.approach, "device": args.device,
            **m,
        })
        if n_viz < args.viz_n:
            visualize(plan, dims, os.path.join(args.out, "figs", f"{cid}.png"))
            n_viz += 1

    cols = ["case", "lobe", "segment", "tumor_mm", "approach", "device", "n_burns",
            "tumor_coverage", "target_coverage_incl_margin", "tumor_volume_mL",
            "ablated_volume_mL", "healthy_overtreated_mL", "total_ablation_time_min"]
    with open(os.path.join(args.out, "plans_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for s in summary:
            w.writerow({k: s.get(k) for k in cols})

    cov = np.mean([s["tumor_coverage"] for s in summary])
    tc = np.mean([s["target_coverage_incl_margin"] for s in summary])
    nb = np.mean([s["n_burns"] for s in summary])
    print(f"[plan] {len(summary)} cases | mean tumor_cov={cov*100:.1f}% "
          f"target_cov={tc*100:.1f}% mean_burns={nb:.1f}")
    print(f"[plan] summary -> {args.out}/plans_summary.csv ; plans/*.json ; figs/*.png ({n_viz})")


if __name__ == "__main__":
    main()
