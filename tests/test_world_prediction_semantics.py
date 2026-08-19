"""Fast regression tests for published Endo-HJEPA forecast semantics."""
from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch
from torch import nn

from endoworld.eval.world_benchmark import (
    horizon_table,
)
from endoworld.world.h_jepa import (
    HJEPAConfig,
    TransformerPredictor,
)
from endoworld.world.train import build_argparser, train


class _MaskRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask = "not-called"

    def forward(self, inputs, mask=None):
        self.mask = mask
        return inputs


def _small_config(**kwargs) -> HJEPAConfig:
    base = HJEPAConfig(
        latent_dim=8,
        hidden_dim=16,
        n_heads=4,
        n_layers=1,
        history=2,
        horizon=4,
        dropout=0.0,
        l1_causal=False,
    )
    return replace(base, **kwargs)


def test_query_predictor_always_applies_block_causal_mask():
    cfg = _small_config()
    predictor = TransformerPredictor(cfg).eval()
    recorder = _MaskRecorder()
    predictor.encoder = recorder
    history = torch.randn(2, cfg.history, cfg.latent_dim)
    prediction = predictor(history)

    mask = recorder.mask
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (cfg.history + cfg.horizon,) * 2
    # Observed tokens must not read predicted future queries.
    assert mask[:cfg.history, cfg.history:].all()
    # Query k can read all history and queries <= k, never later queries.
    assert torch.equal(
        mask[cfg.history:, cfg.history:],
        torch.triu(torch.ones(cfg.horizon, cfg.horizon, dtype=torch.bool), diagonal=1),
    )
    assert prediction.shape == (2, cfg.horizon, cfg.latent_dim)


def test_horizon_table_uses_cumulative_steps_one_through_h():
    target = torch.zeros(1, 4, 2)
    pred = torch.arange(1, 5, dtype=torch.float32).view(1, 4, 1).expand(-1, -1, 2)
    persist = torch.zeros_like(pred)

    rows = {row["horizon"]: row for row in horizon_table(pred, persist, target)}
    assert rows[4]["mse_model"] == pytest.approx((1 + 4 + 9 + 16) / 4)
    assert rows[4]["mse_model"] != pytest.approx(16.0)


def test_training_saves_loadable_checkpoint_and_metrics(tmp_path):
    source_cache = tmp_path / "source_cache.pt"
    output_dir = tmp_path / "run"
    torch.save({
        "Z": torch.randn(12, 6, 8),
        "D": torch.zeros(12, dtype=torch.long),
        "dense": False,
    }, source_cache)
    args = build_argparser().parse_args([
        "--latents", str(source_cache),
        "--out", str(output_dir),
        "--manifest", str(tmp_path / "missing.csv"),
        "--ablation", "l1",
        "--epochs", "1",
        "--batch-size", "2",
        "--hidden", "16",
        "--heads", "4",
        "--layers", "1",
        "--history", "2",
        "--horizon", "4",
    ])
    train(args)

    checkpoint = torch.load(
        output_dir / "endohjepa.pt", map_location="cpu", weights_only=False)
    with (output_dir / "val_metrics.json").open(encoding="utf-8") as handle:
        report = json.load(handle)
    assert "model" in checkpoint and "wcfg" in checkpoint
    assert checkpoint["history"] == 2 and checkpoint["horizon"] == 4
    assert "cos_model" in report and "cos_persist" in report
