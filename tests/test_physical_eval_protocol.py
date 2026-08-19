"""Protocol tests for C3VD Z-depth, strict-future retrieval, and fixed banks."""

from __future__ import annotations

import numpy as np
import torch

from endoworld.world.physical_actions import PhysicalActionDataset, PhysicalSequence


def test_c3vd_z_depth_backprojection():
    from endoworld.eval.c3vd_pose_gate import backproject_z_depth

    rays = np.array([[0.6, 0.0, 0.8], [0.0, 0.0, 1.0]])
    points = backproject_z_depth(rays, np.array([4.0, 2.0]))
    assert np.allclose(points, np.array([[3.0, 0.0, 4.0], [0.0, 0.0, 2.0]]))
    assert np.allclose(points[:, 2], np.array([4.0, 2.0]))


def test_navigation_strict_future_and_pose_error():
    from endoworld.eval.physical_navigation import (
        _pose_error,
        _public_sequence_id,
        _strict_future_nearest_index,
    )

    latents = torch.tensor([[0.0], [5.0], [10.0]])
    assert (
        _strict_future_nearest_index(torch.tensor([0.0]), latents, current_index=0) == 1
    )

    target = np.eye(4)
    target[:3, :3] = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target[:3, 3] = [3.0, 4.0, 0.0]
    translation, rotation = _pose_error(np.eye(4), target)
    assert np.isclose(translation, 5.0)
    assert np.isclose(rotation, 90.0)

    sequence = PhysicalSequence(
        sequence_id="scared:datasets/SCARED/dataset_7/keyframe_3",
        dataset="SCARED",
        case_id="dataset_7",
        latents=torch.zeros(2, 1),
        actions=torch.zeros(1, 6),
    )
    assert _public_sequence_id(sequence) == "SCARED/dataset_7/keyframe_3"


def test_derangement_has_no_fixed_points():
    from endoworld.world.train_continuous_actions import _derangement

    generator = torch.Generator().manual_seed(0)
    for n in range(2, 33):
        perm = _derangement(n, generator)
        assert perm.numel() == n
        assert not torch.any(perm == torch.arange(n))


def test_fixed_bank_uses_distinct_negatives_and_reports_shortfall():
    from endoworld.world.train_continuous_actions import evaluate_fixed_bank

    class Integrator(torch.nn.Module):
        def forward(self, history, actions):
            return history[:, -1:] + actions.cumsum(dim=1)

    actions = torch.arange(42, dtype=torch.float32).reshape(7, 6) / 10
    latents = torch.cat([torch.zeros(1, 6), actions.cumsum(dim=0)])
    sequence = PhysicalSequence(
        sequence_id="scared:datasets/SCARED/dataset_7/keyframe_1",
        dataset="SCARED",
        case_id="dataset_7",
        latents=latents,
        actions=actions,
    )
    dataset = PhysicalActionDataset([sequence], history=2, horizon=1, split="test")
    report = evaluate_fixed_bank(
        Integrator(),
        dataset,
        "cpu",
        n_negatives=10,
        radius=64,
        seed=7,
        batch_size=2,
    )
    expected = len(dataset) - 1
    assert report["actual_unique_negative_counts"] == [expected] * len(dataset)
    assert report["windows_with_reduced_bank"] == len(dataset)
    assert report["actual_unique_negatives_per_window"]["total_pairs"] == (
        len(dataset) * expected
    )
    assert "without replacement" in report["negative_sampling_strategy"]
