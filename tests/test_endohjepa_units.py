"""Offline unit checks for Endo-HJEPA (no GPU, no large data)."""

from __future__ import annotations

import torch
import numpy as np

from endoworld.data.splits import assign_split, video_key
from endoworld.simulation.latent_world_model import WorldModelConfig, build_predictor
from endoworld.understanding.endo_mask import specular_map, token_loss_weights
from endoworld.world.baselines import GRUDynamics
from endoworld.world.h_jepa import EndoHJEPA, HJEPAConfig, subsample_spatial
from endoworld.world.geometry import (
    confidence_ece,
    depth_metrics,
    observability_penalty,
    scale_invariant_depth_loss,
)
from endoworld.world.factorized_state import (
    FactorizedStateAdapter,
    FactorizedStateConfig,
)
from endoworld.world.continuous_dynamics import ContinuousDynamicsConfig
from endoworld.world.probabilistic_dynamics import (
    DynamicsEnsemble,
    RiskCalibrator,
)
from endoworld.world.plan_mpc import (
    ContinuousMPCConfig,
    continuous_cem,
    continuous_mppi,
)
from endoworld.world.physical_actions import (
    PhysicalActionDataset,
    PhysicalSequence,
    pose_deltas,
    tubelet_pose_indices,
    tubelet_pose_positions,
    video_split,
)


def test_split_stable():
    k = video_key("HyperKvasir", r"foo\bar")
    assert assign_split(k) == assign_split(k)


def test_specular_token_weights():
    clip = torch.rand(2, 8, 3, 64, 64)
    clip[:, :, :, :8, :8] = 1.0
    w = token_loss_weights(clip, tubelet=2, patch=16)
    assert w.shape == (2, 4 * 16)
    assert specular_map(clip).shape == (2, 8, 64, 64)


def test_dense_hjepa_and_ablation():
    cfg = HJEPAConfig(
        latent_dim=32,
        hidden_dim=64,
        n_heads=4,
        n_layers=1,
        history=2,
        horizon=2,
        spatial_keep=4,
        ablation="full",
    )
    m = EndoHJEPA(cfg)
    z = torch.randn(3, 6, 8, 32)
    d = torch.zeros(3, dtype=torch.long)
    out = m.losses_dense(z, d, 2, 2)
    assert out["pred"].shape[:2] == (3, 2)
    assert torch.isfinite(out["total"])
    pred = m.forward_l1_dense(subsample_spatial(z[:, :2], 4), d)
    assert pred.shape == (3, 2, 4, 32)
    a, e = m.plan(z.mean(2)[:, :2], z.mean(2)[:, -1], d, n_samples=4, steps=2)
    assert a.shape == (3, 2)
    cfg.ablation = "l1"
    m1 = EndoHJEPA(cfg)
    o1 = m1.losses(z.mean(2), d, 2, 2)
    assert "l3" not in o1 and "l1" in o1


def test_hierarchical_predictors_use_context_and_actions():
    cfg = HJEPAConfig(
        latent_dim=16,
        hidden_dim=32,
        n_heads=4,
        n_layers=2,
        history=3,
        horizon=3,
        n_actions=8,
        dropout=0.0,
        ablation="full",
    )
    model = EndoHJEPA(cfg).eval()
    z = torch.randn(2, 3, 16)
    d0 = torch.zeros(2, dtype=torch.long)
    d1 = torch.ones(2, dtype=torch.long)
    a0 = torch.zeros(2, 3, dtype=torch.long)
    a1 = a0.clone()
    a1[:, 0] = 1

    l2 = model.forward_l2(z, d0)
    l2_changed_history = model.forward_l2(z + 0.5, d0)
    assert not torch.allclose(l2, l2_changed_history)
    assert not torch.allclose(l2, model.forward_l2(z, d1))

    l3 = model.forward_l3(z, a0, d0)
    l3_changed_action = model.forward_l3(z, a1, d0)
    assert not torch.allclose(l3, l3_changed_action)

    # Block causality: changing only the final action cannot alter earlier futures.
    a_future = a0.clone()
    a_future[:, -1] = 2
    l3_future = model.forward_l3(z, a_future, d0)
    assert torch.allclose(l3[:, :-1], l3_future[:, :-1], atol=1e-6)
    assert not torch.allclose(l3[:, -1], l3_future[:, -1])

    model.zero_grad(set_to_none=True)
    model.forward_l3(z, a1, d0).square().mean().backward()
    assert model.l3.action_embed.weight.grad is not None
    assert float(model.l3.action_embed.weight.grad.abs().sum()) > 0
    enc_grad = sum(
        float(p.grad.abs().sum())
        for p in model.l3.encoder.parameters()
        if p.grad is not None
    )
    assert enc_grad > 0


def test_physical_action_alignment_and_video_split():
    poses = np.repeat(np.eye(4)[None], 6, axis=0)
    poses[:, 0, 3] = np.arange(6) * 0.01
    delta = pose_deltas(poses)
    assert delta.shape == (5, 6)
    assert np.allclose(delta[:, 0], 0.01)
    assert np.allclose(delta[:, 1:], 0.0)
    assert np.array_equal(
        tubelet_pose_indices(np.arange(6), tubelet=2), np.array([0, 2, 4])
    )
    assert np.allclose(
        tubelet_pose_positions(np.arange(6), tubelet=2), np.array([0.5, 2.5, 4.5])
    )
    assert (
        video_split("scared-sequence-a", case_id="dataset_1", dataset="SCARED")
        == "train"
    )
    assert (
        video_split("scared-sequence-b", case_id="dataset_6", dataset="SCARED") == "val"
    )
    assert (
        video_split("scared-sequence-c", case_id="dataset_7", dataset="SCARED")
        == "test"
    )

    # Find deterministic ids for two splits, then verify every transition from
    # one video stays in only that split.
    ids = {}
    i = 0
    while len(ids) < 2 and i < 1000:
        split = video_split(f"video-{i}")
        ids.setdefault(split, f"video-{i}")
        i += 1
    sequences = [
        PhysicalSequence(
            sequence_id=sequence_id,
            dataset="synthetic",
            latents=torch.randn(8, 12),
            actions=torch.randn(7, 6),
        )
        for sequence_id in ids.values()
    ]
    for split, sequence_id in ids.items():
        dataset = PhysicalActionDataset(sequences, history=2, horizon=2, split=split)
        assert len(dataset) == 5
        assert {dataset[j]["sequence_id"] for j in range(len(dataset))} == {sequence_id}
        item = dataset[0]
        assert item["history"].shape == (2, 12)
        assert item["actions"].shape == (2, 6)
        assert item["future"].shape == (2, 12)


def test_geometry_losses_and_metrics():
    target = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    assert scale_invariant_depth_loss(target, target).item() < 1e-8
    metrics = depth_metrics(target.numpy(), target.numpy())
    assert metrics["abs_rel"] == 0.0 and metrics["delta1"] == 1.0
    jacobian = torch.eye(6).unsqueeze(0)
    assert observability_penalty(jacobian).item() == 0.0
    assert confidence_ece(np.ones(4), np.ones(4)) == 0.0


def test_factorized_state_preserves_teacher_and_hides_nuisance():
    adapter = FactorizedStateAdapter(
        FactorizedStateConfig(
            teacher_dim=24,
            slot_dim=8,
            adapter_rank=4,
            geometry_dim=3,
            tool_dim=2,
            semantic_dim=4,
            nuisance_dim=2,
        )
    )
    teacher = torch.randn(2, 5, 24, requires_grad=True)
    targets = {
        "geometry_target": torch.randn(2, 5, 3),
        "tool_target": torch.randn(2, 5, 2),
        "semantic_target": torch.randn(2, 5, 4),
        "nuisance_target": torch.randn(2, 5, 2),
    }
    losses = adapter.losses(teacher, **targets)
    assert losses["planner_state"].shape == (2, 5, 24)
    losses["total"].backward()
    assert teacher.grad is None
    assert adapter.adapter.down.weight.grad is not None
    assert adapter.slot_projectors["nuisance"][0].weight.grad is not None


def test_probabilistic_ensemble_and_risk_calibration():
    cfg = ContinuousDynamicsConfig(
        latent_dim=12,
        hidden_dim=24,
        n_heads=4,
        n_layers=1,
        history=2,
        horizon=2,
        dropout=0.0,
    )
    ensemble = DynamicsEnsemble(cfg, members=2).eval()
    prediction = ensemble.predict(torch.randn(3, 2, 12), torch.randn(3, 2, 6))
    assert prediction["mean"].shape == (3, 2, 12)
    assert torch.all(prediction["aleatoric_variance"] > 0)
    assert torch.all(prediction["epistemic_variance"] >= 0)

    logits = torch.tensor([-3.0, -2.0, 2.0, 3.0])
    target = torch.tensor([0.0, 0.0, 1.0, 1.0])
    calibrator = RiskCalibrator(alpha=0.25).fit(logits, target)
    metrics = calibrator.metrics(logits, target)
    assert metrics["auc"] == 1.0
    assert 0.0 <= metrics["ece"] <= 1.0
    assert metrics["coverage"] >= 0.75


def test_continuous_cem_and_hard_safety_gate():
    class Integrator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = ContinuousDynamicsConfig(
                latent_dim=6, history=2, horizon=2, action_dim=6
            )

        def forward(self, history, actions):
            return history[:, -1:] + actions.cumsum(dim=1)

    class UnsafeRisk(torch.nn.Module):
        def forward(self, state, aleatoric, epistemic):
            return torch.full(state.shape[:-1], 20.0, device=state.device)

    dynamics = Integrator()
    history = torch.zeros(2, 2, 6)
    goal = torch.ones(2, 6)
    cfg = ContinuousMPCConfig(
        horizon=2,
        samples=128,
        iterations=4,
        elites=16,
        max_uncertainty=10.0,
    )
    plan = continuous_cem(dynamics, history, goal, cfg)
    assert plan["accepted"].all()
    assert (
        torch.linalg.vector_norm(plan["prediction"][:, -1] - goal, dim=-1).mean() < 1.0
    )

    rejected = continuous_cem(dynamics, history, goal, cfg, risk_head=UnsafeRisk())
    assert not rejected["accepted"].any()
    assert torch.count_nonzero(rejected["actions"]) == 0
    assert torch.count_nonzero(rejected["prediction"]) == 0

    rejected_mppi = continuous_mppi(
        dynamics, history, goal, cfg, risk_head=UnsafeRisk()
    )
    assert not rejected_mppi["accepted"].any()
    assert torch.count_nonzero(rejected_mppi["actions"]) == 0
    assert torch.count_nonzero(rejected_mppi["prediction"]) == 0


def test_gru_and_transformer_baseline():
    z = torch.randn(3, 6, 32)
    gru = GRUDynamics(32, 64, 2)
    assert gru(z[:, :2]).shape == (3, 2, 32)
    pred = build_predictor(
        WorldModelConfig(
            latent_dim=32, hidden_dim=64, history=2, horizon=2, kind="transformer"
        )
    )
    assert pred(z[:, :2]).shape == (3, 2, 32)


def test_c3vd_pose_last_row_translation():
    from endoworld.world.c3vd_actions import load_pose_txt, pose_deltas
    from pathlib import Path

    p = Path("datasets/C3VD/cecum_t1_a/cecum_t1_a/pose.txt")
    if not p.is_file():
        return
    poses = load_pose_txt(p)
    assert poses.shape[1:] == (4, 4)
    assert np.linalg.norm(poses[0, :3, 3]) > 1.0
    d = pose_deltas(poses)
    assert d.shape[1] == 6
    assert float((d[:, :3] ** 2).mean() ** 0.5) > 1e-4


def test_cholect50_official_cv_partition():
    from endoworld.data.cholect50 import CHOLECT50_CV_FOLDS

    vids = [v for fold in CHOLECT50_CV_FOLDS.values() for v in fold]
    assert len(CHOLECT50_CV_FOLDS) == 5
    assert all(len(fold) == 10 for fold in CHOLECT50_CV_FOLDS.values())
    assert len(vids) == len(set(vids)) == 50


if __name__ == "__main__":
    test_split_stable()
    test_specular_token_weights()
    test_dense_hjepa_and_ablation()
    test_hierarchical_predictors_use_context_and_actions()
    test_physical_action_alignment_and_video_split()
    test_geometry_losses_and_metrics()
    test_factorized_state_preserves_teacher_and_hides_nuisance()
    test_probabilistic_ensemble_and_risk_calibration()
    test_continuous_cem_and_hard_safety_gate()
    test_gru_and_transformer_baseline()
    test_c3vd_pose_last_row_translation()
    test_cholect50_official_cv_partition()
    print("unit OK")
