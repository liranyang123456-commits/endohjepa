"""Quantitative experiments for the manuscript: margin sensitivity, device comparison,
coverage-target sweep, cost-weight Pareto front, and bioheat-vs-analytic agreement.

    python -m endoworld.ablation.experiments --params manifests/nodule_params.csv \
        --out outputs/ablation_exp
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from endoworld.ablation.planner import plan_ablation
from endoworld.ablation.bioheat import (simulate_zone, zone_radii_mm, Tissue, Applicator,
                                        analytic_zone_radius_mm)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_dims(params):
    rows = list(csv.DictReader(open(params, encoding="utf-8-sig")))
    dims = []
    for r in rows:
        ap = _f(r.get("size_AP_mm")) or _f(r.get("diam_coronal_mm")) or 0
        si = _f(r.get("size_SI_mm")) or _f(r.get("diam_sagittal_mm")) or 0
        lr = _f(r.get("size_LR_mm")) or _f(r.get("diam_axial_mm")) or 0
        if min(ap, si, lr) > 0:
            dims.append((lr / 2, ap / 2, si / 2))
    return dims


def agg(dimlist, spacing=1.5, **kw):
    cov, tcov, over, burns, tmin = [], [], [], [], []
    for d in dimlist:
        p = plan_ablation(d, spacing_mm=spacing, **kw)
        m = p.metrics
        cov.append(m["tumor_coverage"]); tcov.append(m["target_coverage_incl_margin"])
        over.append(m["healthy_overtreated_mL"]); burns.append(m["n_burns"])
        tmin.append(m["total_ablation_time_min"])
    f = lambda x: (float(np.mean(x)), float(np.std(x)))
    return {"tumor_cov": f(cov), "target_cov": f(tcov), "overtreat": f(over),
            "burns": f(burns), "time_min": f(tmin)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="manifests/nodule_params.csv")
    ap.add_argument("--out", default="outputs/ablation_exp")
    ap.add_argument("--spacing", type=float, default=1.5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dims = load_dims(args.params)
    print(f"[exp] {len(dims)} cases, spacing={args.spacing}mm")
    R = {}

    # E1: margin sensitivity (MWA)
    R["margin"] = {}
    for mgn in [0, 3, 5, 10]:
        R["margin"][mgn] = agg(dims, margin_mm=mgn, device="MWA", spacing=args.spacing)
        print(f"[E1 margin={mgn}] cov={R['margin'][mgn]['tumor_cov'][0]*100:.1f}% "
              f"over={R['margin'][mgn]['overtreat'][0]:.1f}mL burns={R['margin'][mgn]['burns'][0]:.2f}")

    # E2: device comparison (5 mm margin)
    R["device"] = {}
    for dev in ["MWA", "RFA"]:
        R["device"][dev] = agg(dims, margin_mm=5, device=dev, spacing=args.spacing)
        print(f"[E2 {dev}] cov={R['device'][dev]['tumor_cov'][0]*100:.1f}% "
              f"over={R['device'][dev]['overtreat'][0]:.1f}mL time={R['device'][dev]['time_min'][0]:.1f}min")

    # E3: coverage-target sweep
    R["coverage"] = {}
    for ct in [0.90, 0.95, 0.99, 0.999]:
        R["coverage"][ct] = agg(dims, margin_mm=5, device="MWA", coverage_target=ct, spacing=args.spacing)
        print(f"[E3 cov_target={ct}] achieved={R['coverage'][ct]['target_cov'][0]*100:.1f}% "
              f"burns={R['coverage'][ct]['burns'][0]:.2f} over={R['coverage'][ct]['overtreat'][0]:.1f}mL")

    # E4: cost-weight Pareto (overtreat vs time), vary w_time
    R["pareto"] = {}
    for wt in [0.0, 0.05, 0.15, 0.5, 1.5]:
        a = agg(dims, margin_mm=5, device="MWA", w_overtreat=1.0, w_time=wt, spacing=args.spacing)
        R["pareto"][wt] = a
        print(f"[E4 w_time={wt}] over={a['overtreat'][0]:.1f}mL time={a['time_min'][0]:.1f}min burns={a['burns'][0]:.2f}")

    # E5: bioheat vs analytic agreement
    R["bioheat"] = []
    for p in [30, 40, 50, 60]:
        for t in [300, 420, 600]:
            m, T, _ = simulate_zone(Tissue(), Applicator(power_W=p, time_s=t),
                                    spacing_mm=1.5, grid_mm=90)
            rx, ry, rz = zone_radii_mm(m, 1.5)
            R["bioheat"].append({"P": p, "t": t, "sim_transverse": round((rx+ry)/2, 1),
                                 "sim_axial": round(rz, 1),
                                 "analytic": round(analytic_zone_radius_mm(p, t, Tissue().w_b), 1),
                                 "Tmax": round(float(T.max()), 1)})
    sim = np.array([b["sim_transverse"] for b in R["bioheat"]])
    ana = np.array([b["analytic"] for b in R["bioheat"]])
    rmse = float(np.sqrt(np.mean((sim - ana) ** 2)))
    ccc = float(np.corrcoef(sim, ana)[0, 1])
    R["bioheat_agreement"] = {"rmse_mm": round(rmse, 2), "pearson_r": round(ccc, 3)}
    print(f"[E5 bioheat vs analytic] transverse RMSE={rmse:.2f}mm r={ccc:.3f}")

    json.dump(R, open(os.path.join(args.out, "experiments.json"), "w"), indent=2)

    _figures(R, args.out)
    print(f"[done] -> {args.out}/experiments.json + experiments.png")


def _figures(R, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))

    # margin
    mg = sorted(R["margin"]); ax0 = ax[0, 0]
    ax0.plot(mg, [R["margin"][m]["overtreat"][0] for m in mg], "-o", label="healthy over-treated (mL)")
    ax0.plot(mg, [R["margin"][m]["burns"][0] for m in mg], "-s", label="burns")
    ax0.set_xlabel("safety margin (mm)"); ax0.set_title("(a) Margin sensitivity"); ax0.legend(); ax0.grid(alpha=.3)

    # device
    devs = list(R["device"]); ax1 = ax[0, 1]
    x = np.arange(len(devs))
    ax1.bar(x - 0.2, [R["device"][d]["overtreat"][0] for d in devs], 0.4, label="over-treat (mL)")
    ax1.bar(x + 0.2, [R["device"][d]["time_min"][0] for d in devs], 0.4, label="time (min)")
    ax1.set_xticks(x); ax1.set_xticklabels(devs); ax1.set_title("(b) Device comparison"); ax1.legend(); ax1.grid(alpha=.3)

    # coverage sweep
    cts = sorted(R["coverage"]); ax2 = ax[0, 2]
    ax2.plot([c*100 for c in cts], [R["coverage"][c]["burns"][0] for c in cts], "-o", label="burns")
    ax2.plot([c*100 for c in cts], [R["coverage"][c]["overtreat"][0] for c in cts], "-s", label="over-treat (mL)")
    ax2.set_xlabel("required coverage (%)"); ax2.set_title("(c) Coverage-target sweep"); ax2.legend(); ax2.grid(alpha=.3)

    # pareto
    wts = sorted(R["pareto"]); ax3 = ax[1, 0]
    ox = [R["pareto"][w]["overtreat"][0] for w in wts]
    ty = [R["pareto"][w]["time_min"][0] for w in wts]
    ax3.plot(ox, ty, "-o")
    for w, x0, y0 in zip(wts, ox, ty):
        ax3.annotate(f"w_t={w}", (x0, y0), fontsize=7)
    ax3.set_xlabel("healthy over-treated (mL)"); ax3.set_ylabel("total time (min)")
    ax3.set_title("(d) Cost-weight Pareto front"); ax3.grid(alpha=.3)

    # bioheat vs analytic
    ax4 = ax[1, 1]
    sim = [b["sim_transverse"] for b in R["bioheat"]]; ana = [b["analytic"] for b in R["bioheat"]]
    ax4.scatter(ana, sim, c="tab:purple")
    lim = [min(min(sim), min(ana))-1, max(max(sim), max(ana))+1]
    ax4.plot(lim, lim, "k--")
    ax4.set_xlabel("analytic radius (mm)"); ax4.set_ylabel("bioheat radius (mm)")
    ax4.set_title(f"(e) Bioheat vs analytic (r={R['bioheat_agreement']['pearson_r']}, "
                  f"RMSE={R['bioheat_agreement']['rmse_mm']}mm)"); ax4.grid(alpha=.3)

    # tumor vs ablated volume trend (from margin=5 device MWA already; recompute quick proxy)
    ax5 = ax[1, 2]
    mg5 = R["margin"][5]
    ax5.bar(["tumor cov", "target cov"], [mg5["tumor_cov"][0]*100, mg5["target_cov"][0]*100],
            color=["tab:red", "tab:green"])
    ax5.set_ylim(90, 101); ax5.set_ylabel("%"); ax5.set_title("(f) Coverage @ 5mm margin (MWA)")
    ax5.grid(alpha=.3)

    fig.suptitle("Fig. 3  Planning experiments: sensitivity, device comparison, coverage, Pareto, model agreement", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "experiments.png"), dpi=150)
    fig.savefig(os.path.join("docs/paper/figures", "fig3_experiments.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
