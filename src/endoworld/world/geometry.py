"""Small, dependency-light geometry losses and evaluation metrics."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def scale_invariant_depth_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    variance_weight: float = 0.5,
) -> torch.Tensor:
    """Scale-invariant log-depth loss with an explicit validity mask."""
    if valid is None:
        valid = torch.isfinite(target) & (target > 0)
    valid = valid & torch.isfinite(prediction) & (prediction > 0)
    if not bool(valid.any()):
        return prediction.new_zeros(())
    residual = torch.log(prediction[valid].clamp_min(1e-6)) - torch.log(
        target[valid].clamp_min(1e-6))
    return residual.square().mean() - variance_weight * residual.mean().square()


def depth_edge_aware_smoothness(
    inverse_depth: torch.Tensor, image: torch.Tensor,
) -> torch.Tensor:
    """Free-form/Laplacian-style smoothness relaxed across image edges."""
    if inverse_depth.ndim == 3:
        inverse_depth = inverse_depth.unsqueeze(1)
    dx_d = inverse_depth[..., :, 1:] - inverse_depth[..., :, :-1]
    dy_d = inverse_depth[..., 1:, :] - inverse_depth[..., :-1, :]
    dx_i = image[..., :, 1:] - image[..., :, :-1]
    dy_i = image[..., 1:, :] - image[..., :-1, :]
    return (
        dx_d.abs() * torch.exp(-dx_i.abs().mean(1, keepdim=True))
    ).mean() + (
        dy_d.abs() * torch.exp(-dy_i.abs().mean(1, keepdim=True))
    ).mean()


def pose_twist_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    translation_scale: float = 1.0,
    rotation_scale: float = 1.0,
) -> torch.Tensor:
    """Robust loss for canonical [v,w] camera-frame SE(3) twists."""
    trans = F.smooth_l1_loss(prediction[..., :3], target[..., :3])
    rot = F.smooth_l1_loss(prediction[..., 3:], target[..., 3:])
    return translation_scale * trans + rotation_scale * rot


def temporal_geometry_consistency(
    geometry: torch.Tensor, confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Second-order temporal regularizer for depth/shape slot trajectories."""
    if geometry.size(1) < 3:
        return geometry.new_zeros(())
    acceleration = geometry[:, 2:] - 2 * geometry[:, 1:-1] + geometry[:, :-2]
    error = acceleration.square().mean(dim=-1)
    if confidence is not None:
        weight = confidence[:, 1:-1].detach().clamp(0, 1)
        error = error * weight
    return error.mean()


def observability_penalty(jacobian: torch.Tensor, threshold: float = 1e-3) -> torch.Tensor:
    """Penalise poorly observable directions using the smallest singular value."""
    singular = torch.linalg.svdvals(jacobian.float())
    return F.relu(threshold - singular[..., -1]).mean()


def depth_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(prediction) & np.isfinite(target) & (prediction > 0) & (target > 0)
    if not valid.any():
        return {"n": 0, "abs_rel": float("nan"), "rmse": float("nan"), "delta1": float("nan")}
    p, t = prediction[valid], target[valid]
    ratio = np.maximum(p / t, t / p)
    return {
        "n": int(valid.sum()),
        "abs_rel": float(np.mean(np.abs(p - t) / t)),
        "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
    }


def pose_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.size == 0:
        return {"n": 0, "translation_rmse": float("nan"), "rotation_deg": float("nan")}
    n = min(len(prediction), len(target))
    trans = prediction[:n, :3] - target[:n, :3]
    rot = prediction[:n, 3:] - target[:n, 3:]
    return {
        "n": int(n),
        "translation_rmse": float(np.sqrt(np.mean(np.sum(trans**2, axis=-1)))),
        "rotation_deg": float(np.mean(np.linalg.norm(rot, axis=-1)) * 180.0 / math.pi),
    }


def confidence_ece(
    confidence: np.ndarray, correct: np.ndarray, bins: int = 10,
) -> float:
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)
    n = min(len(confidence), len(correct))
    if n == 0:
        return float("nan")
    confidence, correct = confidence[:n], correct[:n]
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = (confidence >= lo) & (
            confidence <= hi if hi == 1.0 else confidence < hi)
        if selected.any():
            ece += selected.mean() * abs(confidence[selected].mean() - correct[selected].mean())
    return float(ece)
