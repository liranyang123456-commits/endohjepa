"""Dynamics baselines for paper ablations (not the main world model): GRU + Mamba/SSM."""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUDynamics(nn.Module):
    def __init__(
        self, latent_dim: int, hidden_dim: int, horizon: int, residual: bool = True
    ):
        super().__init__()
        self.horizon = horizon
        self.residual = residual
        self.proj = nn.Linear(latent_dim, hidden_dim)
        self.core = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_hist, domain_id=None):
        h = self.proj(z_hist)
        _, state = self.core(h)
        last = h[:, -1:]
        preds = []
        for _ in range(self.horizon):
            out, state = self.core(last, state)
            preds.append(self.head(out))
            last = out
        pred = torch.cat(preds, dim=1)
        if self.residual:
            pred = pred + z_hist[:, -1:].expand(-1, self.horizon, -1)
        return pred


class MambaDynamics(nn.Module):
    """Selective state-space (Mamba-style) dynamics baseline for latent forecast.

    Implements a per-channel input-dependent (selective) linear recurrence:
      h_t = a(x_t) * h_{t-1} + b(x_t),   y_t = C h_t
    with a(x) = sigmoid gate, b(x) = input map. Autoregressive over the horizon.
    """

    def __init__(
        self, latent_dim: int, hidden_dim: int, horizon: int, residual: bool = True
    ):
        super().__init__()
        self.horizon = horizon
        self.residual = residual
        self.in_proj = nn.Linear(latent_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)  # a(x): input-dependent decay
        self.in_map = nn.Linear(hidden_dim, hidden_dim)  # b(x)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)
        self.act = nn.SiLU()

    def forward(self, z_hist, domain_id=None):
        h = self.act(self.in_proj(z_hist))  # (B, T, H)
        state = h[:, -1]  # (B, H)
        last = h[:, -1]
        preds = []
        for _ in range(self.horizon):
            a = torch.sigmoid(self.gate(last))
            b = self.in_map(last)
            state = a * state + b * last
            preds.append(self.out_proj(self.act(state)))
            last = state
        pred = torch.stack(preds, dim=1)
        if self.residual:
            pred = pred + z_hist[:, -1:].expand(-1, self.horizon, -1)
        return pred
