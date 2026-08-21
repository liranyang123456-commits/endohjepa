"""Legacy GRU latent dynamics (kept for paper ablations).

The primary world model is Endo-HJEPA (`endoworld.world.h_jepa`): transformer
predictors on dense/pooled tokens, hierarchy, latent actions, and energy MPC.
This module still exposes a GRU predictor so published GRU numbers remain
reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorldModelConfig:
    latent_dim: int = 768
    action_dim: int = 0
    hidden_dim: int = 1024
    history: int = 4
    horizon: int = 8
    kind: str = "transformer"  # transformer | gru


def build_predictor(cfg: WorldModelConfig):
    import torch.nn as nn

    if cfg.kind == "gru":
        from endoworld.world.baselines import GRUDynamics
        return GRUDynamics(cfg.latent_dim, cfg.hidden_dim, cfg.horizon)

    from endoworld.world.h_jepa import HJEPAConfig, TransformerPredictor

    wcfg = HJEPAConfig(
        latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim,
        history=cfg.history, horizon=cfg.horizon,
    )
    pred = TransformerPredictor(wcfg, action_cond=False)

    class _Wrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.cfg = cfg

        def forward(self, z_hist, actions=None):
            return self.inner(z_hist, domain_tok=None, actions=None)

    return _Wrap(pred)


def rollout(predictor, z_hist, actions=None):
    import torch
    with torch.no_grad():
        return predictor(z_hist, actions)
