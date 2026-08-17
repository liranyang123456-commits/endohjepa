"""Learn an ablation-planning policy from the real clinical cases.

Two supervised tasks on the 29 structured cases:
  (1) Imitation of real clinical decisions: predict the number of planned navigation
      paths the clinicians actually used, from nodule geometry + anatomy.
  (2) Learned optimiser surrogate: predict the physics-planner's optimised plan
      (n_burns, total ablation time) from the same features -> an instant "learned
      planner" that reproduces the optimiser without re-solving coverage each time.

Both are evaluated with leave-one-out cross-validation and compared to a mean baseline.
The fitted models are saved so a new nodule can be turned into a plan immediately.

    python -m endoworld.ablation.learn_plan --params manifests/nodule_params.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np


LOBES = ["右上肺叶", "右中肺叶", "右下肺叶", "左上肺叶", "左下肺叶"]


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def featurize(row):
    ap = _f(row.get("size_AP_mm")) or _f(row.get("diam_coronal_mm")) or 0
    si = _f(row.get("size_SI_mm")) or _f(row.get("diam_sagittal_mm")) or 0
    lr = _f(row.get("size_LR_mm")) or _f(row.get("diam_axial_mm")) or 0
    vol = 4 / 3 * np.pi * (ap / 2) * (si / 2) * (lr / 2) / 1000.0  # mL
    maxd = max(ap, si, lr)
    sol = {"实性": 1.0, "部分实性": 0.5, "磨玻璃": 0.0, "亚实性": 0.5}.get(row.get("solidity"), 0.75)
    reg = {"outer": 2.0, "middle": 1.0, "inner": 0.0}.get(row.get("region_third"), 1.0)
    lobe_oh = [1.0 if row.get("lobe") == L else 0.0 for L in LOBES]
    feat = [ap, si, lr, vol, maxd,
            _f(row.get("dist_pleura_mm")) or 0.0,
            _f(row.get("dist_chestwall_mm")) or 0.0,
            _f(row.get("dist_vessel_mm")) or 5.0,
            _f(row.get("airway_generation")) or 6.0,
            sol, reg, _f(row.get("malignancy_pct")) or 90.0] + lobe_oh
    return feat


FEATURE_NAMES = (["ap", "si", "lr", "vol_mL", "maxd", "dist_pleura", "dist_wall",
                  "dist_vessel", "airway_gen", "solidity", "region", "malignancy"]
                 + [f"lobe_{L}" for L in LOBES])


def physics_targets(row, margin=5.0, device="MWA"):
    from endoworld.ablation.planner import plan_ablation
    ap = _f(row.get("size_AP_mm")) or _f(row.get("diam_coronal_mm")) or 0
    si = _f(row.get("size_SI_mm")) or _f(row.get("diam_sagittal_mm")) or 0
    lr = _f(row.get("size_LR_mm")) or _f(row.get("diam_axial_mm")) or 0
    if min(ap, si, lr) <= 0:
        return None
    plan = plan_ablation((lr / 2, ap / 2, si / 2), margin_mm=margin, device=device)
    m = plan.metrics
    energy = sum(b.power_W * b.time_s for b in plan.burns) / 1000.0  # kJ
    return {"n_burns": m["n_burns"], "total_time_min": m["total_ablation_time_min"],
            "ablated_volume_mL": m["ablated_volume_mL"], "energy_kJ": round(energy, 1),
            "overtreat": m["healthy_overtreated_mL"], "approach": plan.trajectory.approach}


def loo_cv(X, y, model_ctor):
    from sklearn.base import clone
    n = len(X)
    preds = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        m = clone(model_ctor)
        m.fit(X[tr], y[tr])
        preds[i] = m.predict(X[i:i + 1])[0]
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="manifests/nodule_params.csv")
    ap.add_argument("--out", default="outputs/ablation_learn")
    ap.add_argument("--device", default="MWA")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from sklearn.ensemble import RandomForestRegressor
    import joblib

    rows = list(csv.DictReader(open(args.params, encoding="utf-8-sig")))
    cache_path = os.path.join(args.out, "physics_targets_cache.json")
    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
    X, real_paths, phys = [], [], []
    kept = []
    for r in rows:
        feat = featurize(r)
        cid = r["note"].split(".")[0]
        if cid in cache:
            pt = cache[cid]
        else:
            pt = physics_targets(r, device=args.device)
            if pt is not None:
                cache[cid] = pt
        if pt is None:
            continue
        X.append(feat)
        real_paths.append(_f(r.get("n_planned_paths")) or np.nan)
        phys.append(pt)
        kept.append(cid)
    json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
    X = np.array(X, float)
    real_paths = np.array(real_paths, float)
    print(f"[data] {len(X)} cases, {X.shape[1]} features")

    report = {}

    # ---- Task 1: imitate real clinical decision (n planned paths) ----
    mask = ~np.isnan(real_paths)
    Xr, yr = X[mask], real_paths[mask]
    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=0)
    preds = loo_cv(Xr, yr, rf)
    mae = float(np.mean(np.abs(preds - yr)))
    base = float(np.mean(np.abs(yr - yr.mean())))
    rf.fit(Xr, yr)
    imp = sorted(zip(FEATURE_NAMES, rf.feature_importances_), key=lambda t: -t[1])[:6]
    report["task1_real_paths"] = {"n": int(mask.sum()), "LOO_MAE": round(mae, 3),
                                  "baseline_MAE": round(base, 3),
                                  "top_features": [(f, round(float(w), 3)) for f, w in imp]}
    joblib.dump(rf, os.path.join(args.out, "model_real_paths.joblib"))
    print(f"[task1] imitate real #paths: LOO MAE={mae:.3f} (baseline {base:.3f}) "
          f"top={[f for f,_ in imp[:3]]}")

    # ---- Task 2: learn the physics-optimiser plan (report MAE and R^2) ----
    def r2(pred, y):
        ss_res = np.sum((y - pred) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    for tgt in ["n_burns", "total_time_min", "ablated_volume_mL", "energy_kJ"]:
        y = np.array([p[tgt] for p in phys], float)
        rf2 = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=0)
        preds = loo_cv(X, y, rf2)
        mae = float(np.mean(np.abs(preds - y)))
        base = float(np.mean(np.abs(y - y.mean())))
        rf2.fit(X, y)
        joblib.dump(rf2, os.path.join(args.out, f"model_{tgt}.joblib"))
        report[f"task2_{tgt}"] = {"LOO_MAE": round(mae, 3), "baseline_MAE": round(base, 3),
                                  "LOO_R2": round(r2(preds, y), 3)}
        print(f"[task2] '{tgt}': LOO MAE={mae:.3f} (base {base:.3f}) R2={r2(preds,y):.3f}")

    # ---- Task 3: model comparison on the required-ablated-volume target ----
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    y = np.array([p["ablated_volume_mL"] for p in phys], float)
    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=6, random_state=0),
        "GradientBoosting": GradientBoostingRegressor(random_state=0),
    }
    report["task3_model_comparison_ablated_volume"] = {}
    for name, mdl in models.items():
        preds = loo_cv(X, y, mdl)
        mae = float(np.mean(np.abs(preds - y)))
        report["task3_model_comparison_ablated_volume"][name] = {
            "LOO_MAE": round(mae, 3), "LOO_R2": round(r2(preds, y), 3)}
        print(f"[task3] {name:16s} MAE={mae:.3f} R2={r2(preds,y):.3f}")

    # approach classification agreement (transbronchial vs percutaneous)
    appr = [p["approach"] for p in phys]
    report["approach_distribution"] = {a: appr.count(a) for a in set(appr)}

    json.dump(report, open(os.path.join(args.out, "learn_report.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=2)

    # scatter fig: predicted vs actual (task1) + feature importance
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    preds1 = loo_cv(Xr, yr, RandomForestRegressor(n_estimators=300, max_depth=6, random_state=0))
    axes[0].scatter(yr, preds1, c="tab:blue", alpha=0.7)
    lim = [yr.min() - 0.5, yr.max() + 0.5]
    axes[0].plot(lim, lim, "k--", lw=1)
    axes[0].set_xlabel("real #planned paths"); axes[0].set_ylabel("LOO predicted")
    axes[0].set_title(f"Imitate clinical decision (MAE={report['task1_real_paths']['LOO_MAE']})")
    names = [f for f, _ in imp][::-1]; vals = [w for _, w in imp][::-1]
    axes[1].barh(names, vals, color="tab:green")
    axes[1].set_title("Top feature importances (real #paths)")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "learn_plan.png"), dpi=120)
    plt.close(fig)

    print(f"[done] report -> {args.out}/learn_report.json ; models + learn_plan.png")


if __name__ == "__main__":
    main()
