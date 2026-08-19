"""Energy head E(z, a, z_next): low energy = compatible / in-distribution transition."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyHead(nn.Module):
    def __init__(self, dim: int, n_actions: int, hidden: int):
        super().__init__()
        self.action_embed = nn.Embedding(n_actions, dim)
        self.net = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z, action, z_next):
        a = self.action_embed(action)
        x = torch.cat([z, a, z_next], dim=-1)
        return self.net(x).squeeze(-1)


def contrastive_energy_loss(head: EnergyHead, z_t, action, z_pos, z_neg):
    e_pos = head(z_t, action, z_pos)
    e_neg = head(z_t, action, z_neg)
    loss = F.softplus(e_pos - e_neg).mean()
    return loss, e_pos.mean().detach(), e_neg.mean().detach()
