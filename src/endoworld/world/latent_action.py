"""Discrete latent actions: VQ of z_{t+1}-z_t (Genie/SurgWorld style, JEPA space)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentActionTokenizer(nn.Module):
    """Vector-quantise residual latents into a discrete action codebook."""

    def __init__(self, dim: int, n_actions: int):
        super().__init__()
        self.codebook = nn.Embedding(n_actions, dim)
        nn.init.normal_(self.codebook.weight, std=0.02)
        self.n_actions = n_actions

    def forward(self, z_t, z_tp1):
        residual = z_tp1 - z_t
        dist = torch.cdist(residual.unsqueeze(1), self.codebook.weight.unsqueeze(0))
        idx = dist.squeeze(1).argmin(dim=-1)
        quantized = self.codebook(idx)
        quantized = residual + (quantized - residual).detach()
        commit = F.mse_loss(residual.detach(), self.codebook(idx)) + F.mse_loss(
            residual, quantized.detach())
        return idx, quantized, commit
