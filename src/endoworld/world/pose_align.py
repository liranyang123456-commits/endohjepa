"""Align discrete latent actions with physical SE(3) camera deltas (SCARED / C3VD)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def quantise_deltas(deltas: np.ndarray, n_actions: int) -> np.ndarray:
    """K-means-lite on 6D deltas → integer action ids (numpy, no sklearn required)."""
    x = np.asarray(deltas, dtype=np.float64)
    if len(x) == 0:
        return np.zeros(0, dtype=np.int64)
    k = min(n_actions, len(x))
    rng = np.random.default_rng(0)
    centres = x[rng.choice(len(x), k, replace=False)]
    for _ in range(12):
        d = ((x[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for i in range(k):
            m = lab == i
            if m.any():
                centres[i] = x[m].mean(0)
    return lab.astype(np.int64)


def action_pose_nmi(action_ids: np.ndarray, pose_ids: np.ndarray) -> float:
    """Normalised mutual information between latent actions and pose clusters."""
    a = np.asarray(action_ids).reshape(-1)
    p = np.asarray(pose_ids).reshape(-1)
    n = min(len(a), len(p))
    if n < 4:
        return float("nan")
    a, p = a[:n], p[:n]
    ka, kp = int(a.max()) + 1, int(p.max()) + 1
    joint = np.zeros((ka, kp), dtype=np.float64)
    for i in range(n):
        joint[a[i], p[i]] += 1
    joint = joint / max(float(joint.sum()), 1.0)
    pa, pp = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    mi = float((joint[nz] * np.log((joint[nz] + 1e-12) / (pa * pp)[nz])).sum())
    ha = float(-(pa[pa > 0] * np.log(pa[pa > 0] + 1e-12)).sum())
    hp = float(-(pp[pp > 0] * np.log(pp[pp > 0] + 1e-12)).sum())
    return mi / max(0.5 * (ha + hp), 1e-8)


def residual_to_delta_loss(residual: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Supervise z_{t+1}-z_t to predict physical 6D delta (linear probe inside eval)."""
    # residual (B, D), delta (B, 6) — closed-form is done outside; here SmoothL1 if a head is passed
    return (
        F.smooth_l1_loss(residual[:, :6], delta)
        if residual.size(-1) >= 6
        else residual.new_zeros(())
    )


def residual_delta_probe(residuals: np.ndarray, deltas: np.ndarray) -> dict:
    """Closed-form linear map residual → 6D pose delta. Video-agnostic 80/20 split."""
    x = np.asarray(residuals, dtype=np.float64)
    y = np.asarray(deltas, dtype=np.float64)
    n = min(len(x), len(y))
    if n < 8:
        return {"n": int(n), "r2": float("nan"), "mae": float("nan")}
    x, y = x[:n], y[:n]
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    xb = np.concatenate([x, np.ones((n, 1))], 1)
    n_tr = max(4, int(0.8 * n))
    w, *_ = np.linalg.lstsq(xb[:n_tr], y[:n_tr], rcond=None)
    pred = xb @ w
    err = pred[n_tr:] - y[n_tr:]
    mse = float((err**2).mean())
    var = float(y[n_tr:].var() + 1e-8)
    return {
        "n": int(n),
        "n_test": int(n - n_tr),
        "r2": 1.0 - mse / var,
        "mae": float(np.abs(err).mean()),
        "trans_mae": float(np.abs(err[:, :3]).mean()),
        "rot_mae": float(np.abs(err[:, 3:]).mean()),
    }
