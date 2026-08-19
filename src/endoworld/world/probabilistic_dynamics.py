"""Probabilistic continuous dynamics, ensemble uncertainty and risk calibration."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from endoworld.world.continuous_dynamics import (
    ContinuousActionDynamics,
    ContinuousDynamicsConfig,
)


class ProbabilisticContinuousDynamics(ContinuousActionDynamics):
    def __init__(self, cfg: ContinuousDynamicsConfig):
        super().__init__(cfg)
        self.log_variance_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

    def distribution(
        self,
        history: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._future_features(history, actions)
        mean = self.head(features) + self.action_residual(actions)
        if self.cfg.residual:
            mean = mean + history[:, -1:].detach()
        log_variance = self.log_variance_head(features).clamp(-8.0, 4.0)
        return mean, log_variance

    def forward(self, history: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution(history, actions)[0]

    def probabilistic_losses(
        self,
        history: torch.Tensor,
        actions: torch.Tensor,
        future: torch.Tensor,
        inverse_weight: float = 0.25,
    ) -> dict[str, torch.Tensor]:
        mean, log_variance = self.distribution(history, actions)
        nll = (
            0.5
            * (
                log_variance + (future - mean).square() * torch.exp(-log_variance)
            ).mean()
        )
        current = torch.cat([history[:, -1:], future[:, :-1]], dim=1)
        inverse = F.smooth_l1_loss(
            self.inverse(current, future), self.normalise_actions(actions)
        )
        return {
            "total": nll + inverse_weight * inverse,
            "nll": nll,
            "inverse": inverse,
            "mean": mean,
            "log_variance": log_variance,
        }


class DynamicsEnsemble(nn.Module):
    def __init__(self, cfg: ContinuousDynamicsConfig, members: int = 3):
        super().__init__()
        if members < 2:
            raise ValueError("ensemble needs at least two members")
        self.members = nn.ModuleList(
            [ProbabilisticContinuousDynamics(cfg) for _ in range(members)]
        )

    def set_action_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        for member in self.members:
            member.set_action_stats(mean, std)

    def predict(
        self,
        history: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        distributions = [
            member.distribution(history, actions) for member in self.members
        ]
        means = torch.stack([item[0] for item in distributions])
        variances = torch.stack([item[1].exp() for item in distributions])
        return {
            "mean": means.mean(0),
            "aleatoric_variance": variances.mean(0),
            "epistemic_variance": means.var(0, unbiased=False),
            "member_means": means,
        }


class NearWallRiskHead(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        aleatoric_variance: torch.Tensor,
        epistemic_variance: torch.Tensor,
    ) -> torch.Tensor:
        aleatoric = aleatoric_variance.mean(dim=-1, keepdim=True)
        epistemic = epistemic_variance.mean(dim=-1, keepdim=True)
        return self.network(torch.cat([state, aleatoric, epistemic], dim=-1)).squeeze(
            -1
        )


def near_wall_labels(
    depth: torch.Tensor,
    threshold: float,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary near-wall labels from lower-tail valid depth per frame."""
    if valid is None:
        valid = torch.isfinite(depth) & (depth > 0)
    flattened = depth.flatten(start_dim=-2)
    mask = valid.flatten(start_dim=-2)
    filled = torch.where(mask, flattened, torch.full_like(flattened, float("inf")))
    min_depth = filled.quantile(0.05, dim=-1)
    return (min_depth < threshold).float()


def binary_auc(probability: np.ndarray, target: np.ndarray) -> float:
    probability = np.asarray(probability).reshape(-1)
    target = np.asarray(target).astype(bool).reshape(-1)
    positive, negative = probability[target], probability[~target]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    comparison = (positive[:, None] > negative[None, :]).mean() + 0.5 * (
        positive[:, None] == negative[None, :]
    ).mean()
    return float(comparison)


def binary_ece(probability: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    probability = np.asarray(probability).reshape(-1)
    target = np.asarray(target).reshape(-1)
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = (probability >= lo) & (
            probability <= hi if hi == 1 else probability < hi
        )
        if selected.any():
            result += selected.mean() * abs(
                probability[selected].mean() - target[selected].mean()
            )
    return float(result)


class RiskCalibrator:
    """Temperature and split-conformal calibration fitted on held-out samples."""

    def __init__(self, alpha: float = 0.1):
        self.temperature = 1.0
        self.radius = 1.0
        self.alpha = alpha

    def fit(self, logits: torch.Tensor, target: torch.Tensor) -> "RiskCalibrator":
        log_temperature = torch.zeros((), device=logits.device, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_temperature], max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                logits.detach() / log_temperature.exp(), target.float()
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(log_temperature.detach().exp().clamp(0.05, 20))
        probability = torch.sigmoid(logits.detach() / self.temperature)
        scores = (target.float() - probability).abs().cpu().numpy()
        n = len(scores)
        quantile = min(math.ceil((n + 1) * (1 - self.alpha)) / max(n, 1), 1.0)
        self.radius = float(np.quantile(scores, quantile, method="higher"))
        return self

    def predict(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probability = torch.sigmoid(logits / self.temperature)
        interval = torch.stack(
            [
                (probability - self.radius).clamp(0, 1),
                (probability + self.radius).clamp(0, 1),
            ],
            dim=-1,
        )
        return probability, interval

    def metrics(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        probability, interval = self.predict(logits)
        p = probability.detach().cpu().numpy()
        y = target.detach().cpu().numpy()
        coverage = (
            ((target >= interval[..., 0]) & (target <= interval[..., 1])).float().mean()
        )
        return {
            "auc": binary_auc(p, y),
            "brier": float(np.mean((p - y) ** 2)),
            "ece": binary_ece(p, y),
            "coverage": float(coverage),
            "temperature": self.temperature,
            "conformal_radius": self.radius,
        }
