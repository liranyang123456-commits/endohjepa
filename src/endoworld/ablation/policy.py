"""Behaviour-cloning policy for sequential ablation planning.

Learns  a_t = π_θ(s_t)  where
    s_t  = state_features(obs, geometry)     (16-D)
    a_t  = (x, y, z, power_W, time_s)        (5-D)

Models: Ridge / RandomForest / GradientBoosting multi-output regressors
(scikit-learn).  The policy is wrapped so it can be dropped into
``AblationSimEnv`` as ``policy(obs, env) -> AblationAction``.

After inference the **safety gate** (``safety_gate.py``) verifies coverage
and may repair the plan with the classical submodular optimiser.

    python -m endoworld.ablation.policy --data outputs/ablation_learn_traj \\
        --out outputs/ablation_policy --model gbrt
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import joblib
import numpy as np

from endoworld.ablation.dataset import ACTION_DIM, STATE_DIM, action_features, state_features
from endoworld.ablation.sim_env import AblationAction, AblationSimEnv


@dataclass
class PolicyConfig:
    model: str = "gbrt"          # ridge | rf | gbrt
    test_frac: float = 0.2
    seed: int = 0
    # Normalise targets for stabler regression
    normalize_y: bool = True


@dataclass
class AblationPolicy:
    """Trained multi-output regressor + feature scalers."""

    model: Any
    model_name: str
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    metrics: dict = field(default_factory=dict)
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM

    def _scale_x(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean) / np.maximum(self.x_std, 1e-6)

    def _unscale_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.y_std + self.y_mean

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        pred = self.model.predict(self._scale_x(X))
        return self._unscale_y(np.asarray(pred, dtype=np.float64))

    def act(self, obs: dict, env: AblationSimEnv) -> AblationAction:
        """Env-compatible policy callable.

        Strategy: predict power/time from BC; place the burn with the
        environment's greedy farthest-point rule.  Pure end-to-end position
        regression is unreliable at small n, while power/time are smoother
        targets.  This keeps the policy useful as a *parameter proposer*
        under the safety-gate architecture.
        """
        s = state_features(obs, env.geometry)
        a = self.predict(s)[0]
        pw = float(min(env.device.power_presets_W,
                       key=lambda p: abs(p - a[3])))
        ts = float(min(env.device.time_presets_s,
                       key=lambda t: abs(t - a[4])))
        # Placement: trust greedy geometry (submodular backbone)
        g = env.greedy_action()
        # Optional: mild blend of BC position toward greedy when coverage low
        if obs.get("coverage", 0) < 0.5:
            blend = 0.2
            pos = tuple(
                (1 - blend) * gp + blend * bp
                for gp, bp in zip(g.position_mm, (a[0], a[1], a[2]))
            )
        else:
            pos = g.position_mm
        return AblationAction(pos, pw, ts)

    def as_callable(self) -> Callable:
        return self.act


def _make_model(name: str, seed: int = 0):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor

    if name == "ridge":
        return MultiOutputRegressor(Ridge(alpha=1.0))
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=seed, n_jobs=-1)
    if name == "gbrt":
        return MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=150, max_depth=3, learning_rate=0.08,
                random_state=seed))
    raise ValueError(f"Unknown model: {name}")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    # R² per dim
    ss_res = np.sum(err ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-9)
    names = ["x", "y", "z", "power_W", "time_s"]
    return {
        "mae": {n: round(float(v), 3) for n, v in zip(names, mae)},
        "rmse": {n: round(float(v), 3) for n, v in zip(names, rmse)},
        "r2": {n: round(float(v), 3) for n, v in zip(names, r2)},
        "mae_pos_mm": round(float(np.mean(mae[:3])), 3),
        "mae_power_W": round(float(mae[3]), 3),
        "mae_time_s": round(float(mae[4]), 3),
    }


def train_policy(
    X: np.ndarray,
    y: np.ndarray,
    cfg: PolicyConfig | None = None,
) -> AblationPolicy:
    cfg = cfg or PolicyConfig()
    assert X.shape[1] == STATE_DIM and y.shape[1] == ACTION_DIM
    rng = np.random.default_rng(cfg.seed)
    n = len(X)
    idx = rng.permutation(n)
    n_te = max(1, int(n * cfg.test_frac))
    te, tr = idx[:n_te], idx[n_te:]
    if len(tr) < 5:
        tr, te = idx, idx[: max(1, n // 5)]

    x_mean, x_std = X[tr].mean(0), X[tr].std(0)
    y_mean, y_std = y[tr].mean(0), y[tr].std(0)
    y_std = np.where(y_std < 1e-6, 1.0, y_std)

    Xs = (X - x_mean) / np.maximum(x_std, 1e-6)
    if cfg.normalize_y:
        ys = (y - y_mean) / y_std
    else:
        ys = y
        y_mean = np.zeros(ACTION_DIM)
        y_std = np.ones(ACTION_DIM)

    model = _make_model(cfg.model, cfg.seed)
    model.fit(Xs[tr], ys[tr])
    pred = model.predict(Xs[te])
    if cfg.normalize_y:
        pred = pred * y_std + y_mean
    metrics = _metrics(y[te], pred)
    metrics["n_train"] = int(len(tr))
    metrics["n_test"] = int(len(te))
    metrics["model"] = cfg.model

    return AblationPolicy(
        model=model, model_name=cfg.model,
        x_mean=x_mean.astype(np.float64),
        x_std=x_std.astype(np.float64),
        y_mean=y_mean.astype(np.float64),
        y_std=y_std.astype(np.float64),
        metrics=metrics,
    )


def save_policy(policy: AblationPolicy, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"policy_{policy.model_name}.joblib")
    joblib.dump({
        "model": policy.model,
        "model_name": policy.model_name,
        "x_mean": policy.x_mean, "x_std": policy.x_std,
        "y_mean": policy.y_mean, "y_std": policy.y_std,
        "metrics": policy.metrics,
        "state_dim": policy.state_dim,
        "action_dim": policy.action_dim,
    }, path)
    json.dump(policy.metrics, open(os.path.join(out_dir, "metrics.json"), "w"),
              indent=2)
    return path


def load_policy(path: str) -> AblationPolicy:
    blob = joblib.load(path)
    return AblationPolicy(
        model=blob["model"], model_name=blob["model_name"],
        x_mean=np.asarray(blob["x_mean"]), x_std=np.asarray(blob["x_std"]),
        y_mean=np.asarray(blob["y_mean"]), y_std=np.asarray(blob["y_std"]),
        metrics=dict(blob.get("metrics") or {}),
        state_dim=int(blob.get("state_dim", STATE_DIM)),
        action_dim=int(blob.get("action_dim", ACTION_DIM)),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train BC ablation policy")
    ap.add_argument("--data", default="outputs/ablation_learn_traj",
                    help="dataset folder with steps.npz")
    ap.add_argument("--out", default="outputs/ablation_policy")
    ap.add_argument("--model", choices=["ridge", "rf", "gbrt"], default="gbrt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args(argv)

    npz = os.path.join(args.data, "steps.npz")
    if not os.path.isfile(npz):
        raise SystemExit(f"Missing {npz}; run dataset builder first.")
    pack = np.load(npz)
    X, y = pack["X"], pack["y"]
    print(f"[policy] loaded {len(X)} steps from {npz}")
    if len(X) < 10:
        raise SystemExit("Too few steps to train; rebuild dataset with more cases.")

    cfg = PolicyConfig(model=args.model, seed=args.seed, test_frac=args.test_frac)
    # Train all three for comparison
    results = {}
    best = None
    for name in ("ridge", "rf", "gbrt"):
        p = train_policy(X, y, PolicyConfig(model=name, seed=args.seed,
                                            test_frac=args.test_frac))
        results[name] = p.metrics
        print(f"  {name}: mae_pos={p.metrics['mae_pos_mm']}mm  "
              f"mae_P={p.metrics['mae_power_W']}W  "
              f"mae_t={p.metrics['mae_time_s']}s  "
              f"R2_xyz={p.metrics['r2']['x']:.2f}/{p.metrics['r2']['y']:.2f}/"
              f"{p.metrics['r2']['z']:.2f}")
        if name == args.model:
            best = p

    assert best is not None
    path = save_policy(best, args.out)
    json.dump(results, open(os.path.join(args.out, "all_models.json"), "w"),
              indent=2)
    print(f"[policy] saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
