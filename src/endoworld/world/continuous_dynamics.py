"""Continuous SE(3)-conditioned latent dynamics baseline."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from endoworld.world.h_jepa import TransformerPredictor


@dataclass
class ContinuousDynamicsConfig:
    latent_dim: int
    hidden_dim: int = 256
    n_heads: int = 4
    n_layers: int = 3
    history: int = 4
    horizon: int = 4
    action_dim: int = 6
    dropout: float = 0.1
    residual: bool = True


class ContinuousActionDynamics(nn.Module):
    """Block-causal predictor driven by metric camera-frame SE(3) twists."""

    def __init__(self, cfg: ContinuousDynamicsConfig):
        super().__init__()
        self.cfg = cfg
        self.history_proj = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        self.action_proj = nn.Sequential(
            nn.Linear(cfg.action_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.action_delta_head = nn.Sequential(
            nn.Linear(cfg.action_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )
        nn.init.zeros_(self.action_delta_head[-1].weight)
        nn.init.zeros_(self.action_delta_head[-1].bias)
        self.query = nn.Parameter(torch.randn(1, cfg.horizon, cfg.hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            cfg.hidden_dim,
            cfg.n_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )
        self.inverse_head = nn.Sequential(
            nn.Linear(cfg.latent_dim * 3, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
        )
        self.register_buffer("action_mean", torch.zeros(cfg.action_dim))
        self.register_buffer("action_std", torch.ones(cfg.action_dim))

    def set_action_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.action_mean.copy_(mean)
        self.action_std.copy_(std.clamp_min(1e-6))

    def normalise_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self.action_mean) / self.action_std

    def action_residual(self, actions: torch.Tensor) -> torch.Tensor:
        """Metric-action shortcut accumulated over the rollout horizon."""
        step_delta = self.action_delta_head(
            self.normalise_actions(actions[:, :self.cfg.horizon]))
        return step_delta.cumsum(dim=1)

    def _future_features(
        self, history: torch.Tensor, actions: torch.Tensor,
    ) -> torch.Tensor:
        if actions.size(1) < self.cfg.horizon:
            raise ValueError(
                f"need {self.cfg.horizon} actions, got {actions.size(1)}")
        history_token = self.history_proj(history)
        query = self.query.expand(history.size(0), -1, -1)
        query = query + self.action_proj(
            self.normalise_actions(actions[:, :self.cfg.horizon]))
        tokens = torch.cat([history_token, query], dim=1)
        tokens = tokens + TransformerPredictor._positions(
            tokens.size(1), tokens.size(2), tokens.device, tokens.dtype)
        tokens = self.encoder(
            tokens,
            mask=TransformerPredictor._block_causal_mask(
                history.size(1), self.cfg.horizon, tokens.device),
        )
        return tokens[:, -self.cfg.horizon:]

    def forward(self, history: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        prediction = (
            self.head(self._future_features(history, actions))
            + self.action_residual(actions)
        )
        if self.cfg.residual:
            prediction = prediction + history[:, -1:].detach()
        return prediction

    def inverse(self, current: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        return self.inverse_head(
            torch.cat([current, future, future - current], dim=-1))

    def losses(
        self,
        history: torch.Tensor,
        actions: torch.Tensor,
        future: torch.Tensor,
        inverse_weight: float = 0.25,
        cycle_weight: float = 0.1,
        counterfactual_weight: float = 0.5,
        counterfactual_margin: float = 0.02,
        negative_actions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        prediction = self(history, actions)
        forward = F.smooth_l1_loss(prediction, future)
        current = torch.cat([history[:, -1:], future[:, :-1]], dim=1)
        inverse_prediction = self.inverse(current, future)
        inverse = F.smooth_l1_loss(
            inverse_prediction, self.normalise_actions(actions))
        predicted_current = torch.cat([history[:, -1:], prediction[:, :-1]], dim=1)
        cycle_action = self.inverse(predicted_current, prediction)
        cycle = F.smooth_l1_loss(
            cycle_action, self.normalise_actions(actions))
        if history.size(0) > 1 and counterfactual_weight > 0:
            if negative_actions is None:
                permutation = torch.randperm(history.size(0), device=history.device)
                negative_actions = actions[permutation]
            shuffled_prediction = self(history, negative_actions)
            real_error = (prediction - future).square().mean(dim=(1, 2))
            shuffled_error = (
                shuffled_prediction - future).square().mean(dim=(1, 2))
            counterfactual = F.relu(
                real_error - shuffled_error + counterfactual_margin).mean()
        else:
            counterfactual = forward.new_zeros(())
        return {
            "total": (
                forward
                + inverse_weight * inverse
                + cycle_weight * cycle
                + counterfactual_weight * counterfactual
            ),
            "forward": forward,
            "inverse": inverse,
            "cycle": cycle,
            "counterfactual": counterfactual,
            "prediction": prediction,
        }
