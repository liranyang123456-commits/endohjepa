"""Factorized state adapter with frozen-teacher fidelity constraints."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FactorizedStateConfig:
    teacher_dim: int
    slot_dim: int = 128
    adapter_rank: int = 16
    geometry_dim: int = 16
    tool_dim: int = 16
    semantic_dim: int = 16
    nuisance_dim: int = 16
    adapter_scale: float = 0.1


class LowRankResidualAdapter(nn.Module):
    def __init__(self, dim: int, rank: int, scale: float):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.scale = scale
        nn.init.zeros_(self.up.weight)

    def forward(self, teacher: torch.Tensor) -> torch.Tensor:
        return teacher + self.scale * self.up(F.silu(self.down(teacher)))


class FactorizedStateAdapter(nn.Module):
    """Split V-JEPA2 features into planner-safe and nuisance state slots."""

    slot_names = ("geometry", "tool", "semantic", "nuisance")

    def __init__(self, cfg: FactorizedStateConfig):
        super().__init__()
        self.cfg = cfg
        self.adapter = LowRankResidualAdapter(
            cfg.teacher_dim, cfg.adapter_rank, cfg.adapter_scale)
        self.slot_norm = nn.LayerNorm(cfg.teacher_dim)
        self.slot_projectors = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(cfg.teacher_dim, cfg.slot_dim),
                nn.GELU(),
                nn.LayerNorm(cfg.slot_dim),
            )
            for name in self.slot_names
        })
        self.teacher_decoder = nn.Linear(cfg.slot_dim * 4, cfg.teacher_dim)
        self.geometry_head = nn.Linear(cfg.slot_dim, cfg.geometry_dim)
        self.tool_head = nn.Linear(cfg.slot_dim, cfg.tool_dim)
        self.semantic_head = nn.Linear(cfg.slot_dim, cfg.semantic_dim)
        self.nuisance_head = nn.Linear(cfg.slot_dim, cfg.nuisance_dim)

    def forward(self, teacher_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        teacher_latent = teacher_latent.detach()
        adapted = self.adapter(teacher_latent)
        normalised = self.slot_norm(adapted)
        slots = {
            name: projector(normalised)
            for name, projector in self.slot_projectors.items()
        }
        all_slots = torch.cat([slots[name] for name in self.slot_names], dim=-1)
        return {
            **slots,
            "adapted": adapted,
            "reconstructed_teacher": self.teacher_decoder(all_slots),
            # Nuisance is deliberately excluded from the state exposed to planner.
            "planner_state": torch.cat([
                slots["geometry"], slots["tool"], slots["semantic"],
            ], dim=-1),
        }

    @staticmethod
    def _slot_covariance(slots: list[torch.Tensor]) -> torch.Tensor:
        flattened = [
            (slot.flatten(0, -2) - slot.flatten(0, -2).mean(0, keepdim=True))
            for slot in slots
        ]
        penalty = flattened[0].new_zeros(())
        count = 0
        for i in range(len(flattened)):
            for j in range(i + 1, len(flattened)):
                covariance = flattened[i].T @ flattened[j] / max(flattened[i].size(0) - 1, 1)
                penalty = penalty + covariance.square().mean()
                count += 1
        return penalty / max(count, 1)

    def losses(
        self,
        teacher_latent: torch.Tensor,
        geometry_target: torch.Tensor | None = None,
        tool_target: torch.Tensor | None = None,
        semantic_target: torch.Tensor | None = None,
        nuisance_target: torch.Tensor | None = None,
        fidelity_weight: float = 1.0,
        separation_weight: float = 0.05,
    ) -> dict[str, torch.Tensor]:
        output = self(teacher_latent)
        fidelity = F.smooth_l1_loss(
            output["reconstructed_teacher"], teacher_latent.detach())
        adapter_drift = F.mse_loss(output["adapted"], teacher_latent.detach())
        separation = self._slot_covariance(
            [output[name] for name in self.slot_names])
        total = fidelity_weight * fidelity + 0.1 * adapter_drift
        total = total + separation_weight * separation
        losses = {
            "fidelity": fidelity,
            "adapter_drift": adapter_drift,
            "slot_separation": separation,
        }
        targets = {
            "geometry": (geometry_target, self.geometry_head),
            "tool": (tool_target, self.tool_head),
            "semantic": (semantic_target, self.semantic_head),
            "nuisance": (nuisance_target, self.nuisance_head),
        }
        for name, (target, head) in targets.items():
            if target is None:
                continue
            prediction = head(output[name])
            auxiliary = F.smooth_l1_loss(prediction, target)
            losses[f"{name}_supervision"] = auxiliary
            total = total + auxiliary
        losses["total"] = total
        losses["planner_state"] = output["planner_state"]
        return losses
