"""Shared training helpers."""
from __future__ import annotations

import torch


def clone_state_dict(model: torch.nn.Module) -> dict:
    """Storage-independent CPU snapshot of a state dict.

    ``v.detach().cpu()`` aliases the live storage for CPU models, so an
    in-place optimizer step would silently corrupt a "best" snapshot taken
    that way. Cloning is required for correctness on every device.
    """
    return {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
