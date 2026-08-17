"""Preference / outcome-calibrated ranking of ablation trajectories.

Given a set of trajectories with ``outcome.preference_score`` (from follow-up
or from simulated metrics), learn a linear / tree ranker

    score(traj) ≈ w · φ(traj)

that can re-rank candidate plans at deployment time, and provide a calibrated
reward bonus for future RL.

    python -m endoworld.ablation.preference --data outputs/ablation_hybrid/dataset
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np

from endoworld.ablation.trajectory_schema import load_trajectory


def trajectory_features(traj) -> np.ndarray:
    m = traj.metrics or {}
    g = traj.geometry
    return np.asarray([
        float(g.tumor_volume_mL),
        float(g.margin_mm),
        float(max(g.tumor_axes_mm)),
        float(m.get("n_burns") or traj.n_burns()),
        float(m.get("tumor_coverage") or 0.0),
        float(m.get("target_coverage_incl_margin") or 0.0),
        float(m.get("healthy_overtreated_mL") or 0.0),
        float(m.get("total_ablation_time_min") or traj.total_time_s() / 60.0),
        float(m.get("total_energy_kJ") or traj.total_energy_kJ()),
        float(g.airway_generation or 0.0),
        float(g.dist_pleura_mm or 0.0),
        1.0 if traj.source == "clinical" else 0.0,
        1.0 if traj.outcome.verdict == "complete_ablation" else 0.0,
    ], dtype=np.float32)


FEATURE_NAMES = [
    "tumor_vol", "margin", "max_axis", "n_burns", "tumor_cov", "target_cov",
    "overtreat", "time_min", "energy_kJ", "airway_gen", "dist_pleura",
    "is_clinical", "is_complete",
]


def load_labelled(index_path: str):
    idx = json.load(open(index_path, encoding="utf-8"))
    X, y, meta = [], [], []
    for e in idx.get("entries", idx if isinstance(idx, list) else []):
        path = e.get("path")
        if not path or not os.path.isfile(path):
            continue
        traj = load_trajectory(path)
        pref = traj.outcome.preference_score
        if pref is None:
            pref = e.get("preference")
        if pref is None:
            continue
        X.append(trajectory_features(traj))
        y.append(float(pref))
        meta.append({"case_id": e.get("case_id"), "source": e.get("source"),
                     "path": path})
    return np.stack(X), np.asarray(y, dtype=np.float64), meta


def train_ranker(X, y, seed: int = 0):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_predict

    models = {
        "ridge": Ridge(alpha=1.0),
        "gbrt": GradientBoostingRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.08, random_state=seed),
    }
    out = {}
    best_name, best_r2, best_model = None, -1e9, None
    for name, m in models.items():
        if len(X) < 5:
            m.fit(X, y)
            pred = m.predict(X)
        else:
            pred = cross_val_predict(m, X, y, cv=min(5, len(X)))
            m.fit(X, y)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = float(1 - ss_res / max(ss_tot, 1e-9))
        mae = float(np.mean(np.abs(y - pred)))
        out[name] = {"r2": round(r2, 3), "mae": round(mae, 4)}
        if r2 > best_r2:
            best_r2, best_name, best_model = r2, name, m
    return best_model, best_name, out


def pairwise_accuracy(scores: np.ndarray, y: np.ndarray) -> float:
    """Fraction of pairs whose preference order is preserved."""
    n = len(y)
    if n < 2:
        return 1.0
    correct = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(y[i] - y[j]) < 1e-6:
                continue
            total += 1
            if (scores[i] - scores[j]) * (y[i] - y[j]) > 0:
                correct += 1
    return float(correct / max(total, 1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/ablation_hybrid/dataset")
    ap.add_argument("--out", default="outputs/ablation_hybrid/preference")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    index_path = os.path.join(args.data, "index.json")
    if not os.path.isfile(index_path):
        raise SystemExit(f"Missing {index_path}")
    X, y, meta = load_labelled(index_path)
    print(f"[preference] {len(X)} labelled trajectories")
    if len(X) < 3:
        raise SystemExit("Need ≥3 labelled trajectories")

    model, name, metrics = train_ranker(X, y, seed=args.seed)
    pred = model.predict(X)
    pair_acc = pairwise_accuracy(pred, y)
    metrics[name]["pairwise_acc"] = round(pair_acc, 3)
    print(f"  best={name}  R2={metrics[name]['r2']}  "
          f"MAE={metrics[name]['mae']}  pairwise={pair_acc:.3f}")

    os.makedirs(args.out, exist_ok=True)
    joblib.dump({"model": model, "name": name, "features": FEATURE_NAMES,
                 "metrics": metrics},
                os.path.join(args.out, "ranker.joblib"))
    json.dump({"metrics": metrics, "n": len(X),
               "by_source": {
                   s: sum(1 for m in meta if m["source"] == s)
                   for s in ("optimiser", "simulated", "clinical")
               }},
              open(os.path.join(args.out, "metrics.json"), "w"), indent=2)

    # Rank clinical cases if present
    clin = [(m, yi, pi) for m, yi, pi in zip(meta, y, pred)
            if m["source"] == "clinical"]
    if clin:
        print("  clinical ranking (true pref → predicted):")
        for m, yi, pi in sorted(clin, key=lambda t: -t[2]):
            print(f"    {m['case_id']}: true={yi:.2f}  pred={pi:.2f}")
    print(f"[preference] saved → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
