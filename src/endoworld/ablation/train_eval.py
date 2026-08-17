"""End-to-end training + evaluation of the hybrid ablation planner.

Pipeline
--------
1. Build mixed dataset  (optimiser + sim + clinical follow-up)
2. Train BC policy
3. Evaluate: random / greedy / BC / BC+gate  on held-out geometries
4. Write a summary JSON + optional figure

    python -m endoworld.ablation.train_eval --all \\
        --params manifests/nodule_params.csv \\
        --followup outputs/ablation_followup/followup_summary.csv \\
        --out outputs/ablation_hybrid
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Callable

import numpy as np

from endoworld.ablation.dataset import build_dataset, preference_from_metrics
from endoworld.ablation.policy import (
    AblationPolicy, PolicyConfig, load_policy, save_policy, train_policy,
)
from endoworld.ablation.safety_gate import GateConfig, gate_rollout
from endoworld.ablation.sim_env import make_env_from_axes, rollout
from endoworld.ablation.trajectory_schema import (
    LesionGeometry, geometry_from_record_row, save_trajectory,
)


def _load_geometries(params_csv: str, limit: int = 0) -> list[LesionGeometry]:
    rows = list(csv.DictReader(open(params_csv, encoding="utf-8-sig")))
    if limit > 0:
        rows = rows[:limit]
    geoms = []
    for row in rows:
        g = geometry_from_record_row(row)
        if g is not None:
            geoms.append(g)
    return geoms


def evaluate_methods(
    geoms: list[LesionGeometry],
    policy: AblationPolicy | None,
    device: str = "MWA",
    force_zone_mm: float | None = 10.0,
    coverage_target: float = 0.99,
    max_burns: int = 20,
    spacing_mm: float = 1.5,
    seed: int = 0,
    out_dir: str | None = None,
) -> dict:
    """Compare random / greedy / BC / BC+gate on the same geometries."""
    methods = ["random", "greedy"]
    if policy is not None:
        methods += ["bc", "bc_gated"]

    gate_cfg = GateConfig(
        coverage_target=coverage_target,
        max_burns=max_burns,
        force_zone_mm=force_zone_mm,
        spacing_mm=spacing_mm,
        repair="cascade",
    )
    results = {m: [] for m in methods}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for i, g in enumerate(geoms):
        for method in methods:
            t0 = time.time()
            if method == "bc_gated":
                assert policy is not None
                gr = gate_rollout(
                    g, policy=policy.act, device=device,
                    cfg=gate_cfg, seed=seed + i)
                traj = gr.trajectory
                extra = {
                    "accepted": gr.accepted, "repaired": gr.repaired,
                    "gate_note": gr.note,
                }
            else:
                env = make_env_from_axes(
                    g.tumor_axes_mm, margin_mm=g.margin_mm, device=device,
                    spacing_mm=spacing_mm, coverage_target=coverage_target,
                    max_burns=max_burns, force_zone_mm=force_zone_mm,
                    seed=seed + i, case_id=g.case_id)
                for k in ("lobe", "airway_generation", "dist_pleura_mm",
                          "dist_chestwall_mm", "dist_vessel_mm"):
                    setattr(env.geometry, k, getattr(g, k))
                if method == "bc":
                    assert policy is not None
                    pol: str | Callable = policy.act
                else:
                    pol = method
                traj, _ = rollout(env, policy=pol, seed=seed + i)
                extra = {"accepted": None, "repaired": False, "gate_note": ""}

            elapsed = time.time() - t0
            pref = preference_from_metrics(traj.metrics)
            row = {
                "case_id": g.case_id,
                "method": method,
                "n_burns": traj.n_burns(),
                "tumor_coverage": traj.metrics.get("tumor_coverage"),
                "target_coverage": traj.metrics.get("target_coverage_incl_margin"),
                "overtreat_mL": traj.metrics.get("healthy_overtreated_mL"),
                "time_min": traj.metrics.get("total_ablation_time_min"),
                "preference": pref,
                "elapsed_s": round(elapsed, 3),
                **extra,
            }
            results[method].append(row)
            if out_dir:
                save_trajectory(
                    traj,
                    os.path.join(out_dir, f"{g.case_id}_{method}.json"))
        print(f"  [{i+1}/{len(geoms)}] {g.case_id}: "
              + "  ".join(
                  f"{m}={results[m][-1]['target_coverage']}"
                  for m in methods))

    # aggregate
    summary = {"n_cases": len(geoms), "methods": {}}
    for m, rows in results.items():
        cov = [r["target_coverage"] for r in rows if r["target_coverage"] is not None]
        over = [r["overtreat_mL"] for r in rows if r["overtreat_mL"] is not None]
        burns = [r["n_burns"] for r in rows]
        prefs = [r["preference"] for r in rows if r["preference"] is not None]
        accepted = [r["accepted"] for r in rows if r["accepted"] is not None]
        repaired = [r["repaired"] for r in rows]
        summary["methods"][m] = {
            "mean_coverage": round(float(np.mean(cov)), 4) if cov else None,
            "mean_overtreat_mL": round(float(np.mean(over)), 3) if over else None,
            "mean_burns": round(float(np.mean(burns)), 2) if burns else None,
            "mean_preference": round(float(np.mean(prefs)), 4) if prefs else None,
            "frac_cov_ge_99": round(float(np.mean([c >= 0.99 for c in cov])), 3) if cov else None,
            "frac_accepted": round(float(np.mean(accepted)), 3) if accepted else None,
            "frac_repaired": round(float(np.mean(repaired)), 3) if repaired else None,
            "mean_elapsed_s": round(float(np.mean([r["elapsed_s"] for r in rows])), 3),
            "rows": rows,
        }
    return summary


def _plot_summary(summary: dict, out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skip: {e}")
        return

    methods = list(summary["methods"].keys())
    cov = [summary["methods"][m]["mean_coverage"] or 0 for m in methods]
    over = [summary["methods"][m]["mean_overtreat_mL"] or 0 for m in methods]
    burns = [summary["methods"][m]["mean_burns"] or 0 for m in methods]
    frac = [summary["methods"][m]["frac_cov_ge_99"] or 0 for m in methods]

    fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.0))
    colors = {"random": "#E66101", "greedy": "#1A9641",
              "bc": "#5E3C99", "bc_gated": "#2C7BB6"}
    cols = [colors.get(m, "#666") for m in methods]

    ax[0].bar(methods, [c * 100 for c in cov], color=cols, edgecolor="white")
    ax[0].axhline(99, ls=":", color="gray")
    ax[0].set_ylabel("Mean target coverage (%)")
    ax[0].set_title("(a) Coverage")
    ax[0].set_ylim(0, 105)

    ax[1].bar(methods, over, color=cols, edgecolor="white")
    ax[1].set_ylabel("Mean over-treatment (mL)")
    ax[1].set_title("(b) Healthy over-treatment")

    ax[2].bar(methods, burns, color=cols, edgecolor="white")
    for i, (b, f) in enumerate(zip(burns, frac)):
        ax[2].text(i, b + 0.15, f"{f*100:.0f}%≥99", ha="center", fontsize=8)
    ax[2].set_ylabel("Mean #burns")
    ax[2].set_title("(c) Burns (label: frac ≥99% cov)")

    for a in ax:
        a.grid(axis="y", alpha=0.3)
        a.set_axisbelow(True)
    fig.suptitle(
        "Hybrid planner evaluation: random / greedy / BC / BC+safety-gate",
        fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_path}")


def run_all(args) -> dict:
    os.makedirs(args.out, exist_ok=True)
    data_dir = os.path.join(args.out, "dataset")
    policy_dir = os.path.join(args.out, "policy")
    eval_dir = os.path.join(args.out, "eval")
    fz = args.force_zone if args.force_zone > 0 else None

    # 1. dataset
    print("=" * 60)
    print("[1/3] Building mixed trajectory dataset")
    summary_ds = build_dataset(
        args.params, data_dir, followup_csv=args.followup,
        device=args.device, margin_mm=args.margin, spacing_mm=args.spacing,
        force_zone_mm=fz, limit=args.limit, seed=args.seed,
        include_noisy=not args.no_noisy,
    )

    # 2. train policy
    print("=" * 60)
    print("[2/3] Training BC policy")
    pack = np.load(os.path.join(data_dir, "steps.npz"))
    X, y = pack["X"], pack["y"]
    print(f"  steps={len(X)}")
    all_metrics = {}
    best = None
    for name in ("ridge", "rf", "gbrt"):
        p = train_policy(X, y, PolicyConfig(
            model=name, seed=args.seed, test_frac=0.2))
        all_metrics[name] = p.metrics
        print(f"  {name}: mae_pos={p.metrics['mae_pos_mm']}mm  "
              f"R2x={p.metrics['r2']['x']:.2f}")
        if name == args.model:
            best = p
    assert best is not None
    save_policy(best, policy_dir)
    json.dump(all_metrics, open(os.path.join(policy_dir, "all_models.json"), "w"),
              indent=2)

    # 3. evaluate
    print("=" * 60)
    print("[3/3] Evaluating methods")
    geoms = _load_geometries(args.params, limit=args.limit or 0)
    # hold out last 30% for eval if enough cases
    if len(geoms) >= 8:
        n_te = max(3, len(geoms) // 3)
        geoms_te = geoms[-n_te:]
    else:
        geoms_te = geoms
    print(f"  eval cases: {len(geoms_te)}")
    summary_ev = evaluate_methods(
        geoms_te, best, device=args.device, force_zone_mm=fz,
        coverage_target=args.coverage, max_burns=args.max_burns,
        spacing_mm=args.spacing, seed=args.seed,
        out_dir=os.path.join(eval_dir, "trajs"),
    )
    json.dump(summary_ev, open(os.path.join(eval_dir, "summary.json"), "w"),
              indent=2)
    _plot_summary(summary_ev, os.path.join(eval_dir, "comparison.png"))

    # print table
    print("\n=== Evaluation summary ===")
    print(f"{'method':<12} {'cov':>8} {'over(mL)':>10} {'burns':>7} "
          f"{'≥99%':>7} {'repaired':>9}")
    for m, s in summary_ev["methods"].items():
        print(f"{m:<12} {100*(s['mean_coverage'] or 0):7.1f}% "
              f"{s['mean_overtreat_mL'] or 0:10.2f} "
              f"{s['mean_burns'] or 0:7.1f} "
              f"{100*(s['frac_cov_ge_99'] or 0):6.0f}% "
              f"{100*(s['frac_repaired'] or 0):8.0f}%")

    report = {
        "dataset": {k: summary_ds[k] for k in (
            "n_trajectories", "n_steps", "by_source", "force_zone_mm")},
        "policy": best.metrics,
        "all_policies": all_metrics,
        "evaluation": {
            "n_cases": summary_ev["n_cases"],
            "methods": {m: {k: v for k, v in s.items() if k != "rows"}
                        for m, s in summary_ev["methods"].items()},
        },
    }
    report_path = os.path.join(args.out, "hybrid_report.json")
    json.dump(report, open(report_path, "w"), indent=2)
    print(f"\n[done] report → {report_path}")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Hybrid ablation planner train+eval")
    ap.add_argument("--all", action="store_true",
                    help="run dataset → train → eval end-to-end")
    ap.add_argument("--params", default="manifests/nodule_params.csv")
    ap.add_argument("--followup",
                    default="outputs/ablation_followup/followup_summary.csv")
    ap.add_argument("--out", default="outputs/ablation_hybrid")
    ap.add_argument("--device", default="MWA")
    ap.add_argument("--margin", type=float, default=5.0)
    ap.add_argument("--spacing", type=float, default=1.5)
    ap.add_argument("--force-zone", type=float, default=10.0)
    ap.add_argument("--coverage", type=float, default=0.99)
    ap.add_argument("--max-burns", type=int, default=40)
    ap.add_argument("--model", choices=["ridge", "rf", "gbrt"], default="gbrt")
    ap.add_argument("--limit", type=int, default=0,
                    help="limit planning cases (0=all); use e.g. 10 for a quick run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-noisy", action="store_true")
    # partial modes
    ap.add_argument("--eval-only", action="store_true",
                    help="evaluate an existing policy (needs --policy)")
    ap.add_argument("--policy", default=None,
                    help="path to policy_*.joblib for --eval-only")
    args = ap.parse_args(argv)

    if args.eval_only:
        if not args.policy:
            raise SystemExit("--eval-only requires --policy")
        pol = load_policy(args.policy)
        geoms = _load_geometries(args.params, limit=args.limit or 0)
        fz = args.force_zone if args.force_zone > 0 else None
        summary = evaluate_methods(
            geoms, pol, device=args.device, force_zone_mm=fz,
            coverage_target=args.coverage, max_burns=args.max_burns,
            spacing_mm=args.spacing, seed=args.seed,
            out_dir=os.path.join(args.out, "eval", "trajs"),
        )
        os.makedirs(os.path.join(args.out, "eval"), exist_ok=True)
        json.dump(summary, open(os.path.join(args.out, "eval", "summary.json"), "w"),
                  indent=2)
        _plot_summary(summary, os.path.join(args.out, "eval", "comparison.png"))
        return 0

    if args.all:
        run_all(args)
        return 0

    ap.print_help()
    print("\nTip: start with  python -m endoworld.ablation.train_eval --all --limit 10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
