"""Fast regression tests for published Endo-HJEPA forecast semantics."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch
from torch import nn

from endoworld.eval.world_benchmark import (
    CUMULATIVE_AGGREGATION,
    _cfg_from_blob,
    horizon_table,
)
from endoworld.world.h_jepa import (
    EndoHJEPA,
    HJEPAConfig,
    TransformerPredictor,
    l2_future_targets,
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


def test_legacy_query_checkpoint_defaults_to_unmasked_parallel_queries():
    saved = _small_config().__dict__.copy()
    saved.pop("query_mask")
    cfg = _cfg_from_blob({"wcfg": saved})
    assert cfg.query_mask == "parallel"

    predictor = TransformerPredictor(cfg).eval()
    recorder = _MaskRecorder()
    predictor.encoder = recorder
    history = torch.randn(2, cfg.history, cfg.latent_dim)
    prediction = predictor(history)
    expected = predictor.head(predictor.query.expand(history.size(0), -1, -1))
    expected = expected + history[:, -1:].expand_as(expected)
    assert recorder.mask == "not-called"
    assert torch.allclose(prediction, expected)


def test_block_causal_query_mask_remains_explicitly_available():
    cfg = _small_config(query_mask="block_causal")
    predictor = TransformerPredictor(cfg).eval()
    recorder = _MaskRecorder()
    predictor.encoder = recorder
    predictor(torch.randn(2, cfg.history, cfg.latent_dim))

    mask = recorder.mask
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (cfg.history + cfg.horizon,) * 2
    assert mask[: cfg.history, cfg.history :].all()
    assert torch.equal(
        mask[cfg.history :, cfg.history :],
        torch.triu(torch.ones(cfg.horizon, cfg.horizon, dtype=torch.bool), diagonal=1),
    )


def test_query_mask_does_not_change_causal_l1_predictor():
    parallel_cfg = _small_config(l1_causal=True, query_mask="parallel")
    block_cfg = replace(parallel_cfg, query_mask="block_causal")
    parallel = EndoHJEPA(parallel_cfg).eval()
    block = EndoHJEPA(block_cfg).eval()
    block.load_state_dict(parallel.state_dict())

    history = torch.randn(2, parallel_cfg.history, parallel_cfg.latent_dim)
    domains = torch.zeros(2, dtype=torch.long)
    assert torch.allclose(
        parallel.forward_l1(history, domains),
        block.forward_l1(history, domains),
        atol=1e-7,
    )


def test_l2_targets_are_second_and_fourth_future_steps():
    future = torch.arange(1, 5, dtype=torch.float32).view(1, 4, 1)
    targets = l2_future_targets(future, stride=2)
    assert targets.flatten().tolist() == [2.0, 4.0]


def test_horizon_table_uses_cumulative_steps_one_through_h():
    target = torch.zeros(1, 4, 2)
    pred = torch.arange(1, 5, dtype=torch.float32).view(1, 4, 1).expand(-1, -1, 2)
    persist = torch.zeros_like(pred)

    rows = {row["horizon"]: row for row in horizon_table(pred, persist, target)}
    assert rows[4]["aggregation"] == CUMULATIVE_AGGREGATION
    assert rows[4]["mse_model"] == pytest.approx((1 + 4 + 9 + 16) / 4)
    assert rows[4]["mse_model"] != pytest.approx(16.0)


def test_training_writes_seed_provenance_and_metric_definition(tmp_path):
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
        assert payload["training_parameters"]["batch_size"] == 2
        assert payload["training_parameters"]["query_mask"] == "parallel"
        assert len(payload["cache_sha256"]) == 64
        assert payload["manifest_sha256"] is None
        assert payload["metric_definition"]["aggregation"].startswith("mean_over_")
