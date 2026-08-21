"""Fast regression tests for published Endo-HJEPA forecast semantics."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch
from torch import nn

from endoworld.eval.world_benchmark import (
    _cfg_from_blob,
    horizon_table,
)
from endoworld.world.h_jepa import (
    EndoHJEPA,
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
    cfg = _small_config(l1_causal=False)
    predictor = TransformerPredictor(cfg).eval()
    recorder = _MaskRecorder()
    predictor.encoder = recorder
    predictor(torch.randn(2, cfg.history, cfg.latent_dim))

    mask = recorder.mask
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (cfg.history + cfg.horizon,) * 2
    # observed history cannot read future query slots
    assert mask[: cfg.history, cfg.history :].all()
    # query k reads history and queries <= k only
    assert torch.equal(
        mask[cfg.history :, cfg.history :],
        torch.triu(torch.ones(cfg.horizon, cfg.horizon, dtype=torch.bool), diagonal=1),
    )


def test_causal_l1_is_autoregressive_over_future_steps():
    cfg = _small_config(l1_causal=True)
    model = EndoHJEPA(cfg).eval()
    history = torch.randn(2, cfg.history, cfg.latent_dim)
    domains = torch.zeros(2, dtype=torch.long)
    pred = model.forward_l1(history, domains)
    assert pred.shape == (2, cfg.horizon, cfg.latent_dim)
    # perturbing the history must change every predicted step (causal coupling)
    history2 = history.clone()
    history2[:, 0] += 1.0
    pred2 = model.forward_l1(history2, domains)
    assert not torch.allclose(pred, pred2)


def test_l2_predicts_horizon_over_stride_steps():
    cfg = _small_config(ablation="l1l2")
    model = EndoHJEPA(cfg).eval()
    history = torch.randn(2, cfg.history, cfg.latent_dim)
    domains = torch.zeros(2, dtype=torch.long)
    pred2 = model.forward_l2(history, domains)
    assert pred2.size(1) == max(cfg.horizon // cfg.l2_stride, 1)


def test_horizon_table_uses_cumulative_steps_one_through_h():
    target = torch.zeros(1, 4, 2)
    pred = torch.arange(1, 5, dtype=torch.float32).view(1, 4, 1).expand(-1, -1, 2)
    persist = torch.zeros_like(pred)

    rows = {row["horizon"]: row for row in horizon_table(pred, persist, target)}
    assert rows[4]["mse_model"] == pytest.approx((1 + 4 + 9 + 16) / 4)
    assert rows[4]["mse_model"] != pytest.approx(16.0)


def test_training_writes_seed_provenance(tmp_path):
    source_cache = tmp_path / "source_cache.pt"
    output_dir = tmp_path / "run"
    torch.save(
        {
            "Z": torch.randn(12, 6, 8),
            "D": torch.zeros(12, dtype=torch.long),
            "dense": False,
        },
        source_cache,
    )
    args = build_argparser().parse_args(
        [
            "--latents",
            str(source_cache),
            "--out",
            str(output_dir),
            "--manifest",
            str(tmp_path / "missing.csv"),
            "--ablation",
            "l1",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--hidden",
            "16",
            "--heads",
            "4",
            "--layers",
            "1",
            "--history",
            "2",
            "--horizon",
            "4",
            "--seed",
            "17",
        ]
    )
    train(args)

    checkpoint = torch.load(
        output_dir / "endohjepa.pt", map_location="cpu", weights_only=False
    )
    with (output_dir / "val_metrics.json").open(encoding="utf-8") as handle:
        report = json.load(handle)
    for payload in (checkpoint, report):
        assert payload["seed"] == 17
    assert checkpoint["wcfg"]["horizon"] == 4
    assert checkpoint["history"] == 2
