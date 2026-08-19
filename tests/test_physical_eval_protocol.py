"""Protocol tests for C3VD ray model, pose error, and hard-negative banks."""
from __future__ import annotations

import numpy as np
import torch

from endoworld.world.physical_actions import PhysicalActionDataset, PhysicalSequence


def test_c3vd_cam_ray_round_trip():
    from endoworld.eval.c3vd_pose_gate import cam2ray, ray2cam

    u = np.array([320.0, 400.0, 511.5])
    v = np.array([240.0, 300.0, 511.5])
    rays = cam2ray(u, v)
    assert np.allclose(np.linalg.norm(rays, axis=-1), 1.0)
    depth = np.array([4.0, 2.0, 6.0])
    points = rays * (depth / rays[..., 2])[..., None]
    assert np.allclose(points[:, 2], depth)
    u2, v2, valid = ray2cam(points)
    assert valid.all()
    assert np.allclose(u2, u, atol=0.5)
    assert np.allclose(v2, v, atol=0.5)


def test_navigation_pose_error():
    from endoworld.eval.physical_navigation import _pose_error

    # Pure translation: twist translation equals the displacement.
    moved = np.eye(4)
    moved[:3, 3] = [3.0, 4.0, 0.0]
    translation, rotation = _pose_error(np.eye(4), moved)
    assert np.isclose(translation, 5.0)
    assert np.isclose(rotation, 0.0)

    # Pure rotation: twist rotation equals the rotation angle in degrees.
    turned = np.eye(4)
    turned[:3, :3] = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    translation, rotation = _pose_error(np.eye(4), turned)
    assert np.isclose(translation, 0.0)
    assert np.isclose(rotation, 90.0)


def _toy_dataset(n_windows: int = 7) -> PhysicalActionDataset:
    actions = torch.arange(42, dtype=torch.float32).reshape(7, 6) / 10
    latents = torch.cat([torch.zeros(1, 6), actions.cumsum(dim=0)])
    sequence = PhysicalSequence(
        sequence_id="scared:E:/private/SCARED/dataset_7/keyframe_1",
        dataset="SCARED",
        case_id="dataset_7",
        latents=latents,
        actions=actions,
    )
    return PhysicalActionDataset([sequence], history=2, horizon=1, split="test")


def test_hard_negative_is_same_sequence_and_never_self():
    dataset = _toy_dataset()
    rng = np.random.default_rng(0)
    for index in range(len(dataset)):
        for _ in range(20):
            negative = dataset.hard_negative_index(index, radius=64, rng=rng)
            assert negative != index
            seq_a, _ = dataset.windows[index]
            seq_b, _ = dataset.windows[negative]
            assert seq_a == seq_b


def test_fixed_bank_reports_pair_and_window_wins():
    from endoworld.world.train_continuous_actions import evaluate_fixed_bank

    class Integrator(torch.nn.Module):
        def forward(self, history, actions):
            return history[:, -1:] + actions.cumsum(dim=1)

    dataset = _toy_dataset()
    report = evaluate_fixed_bank(
        Integrator(), dataset, "cpu", n_negatives=4, radius=64, seed=7)
    assert report["n"] == len(dataset)
    assert report["n_negatives"] == 4
    # A perfect integrator scores zero error for real actions, so it must win
    # every pair against any non-identical negative action sequence.
    assert report["pair_win_fraction"] == 1.0
    assert report["all_negative_win_fraction"] == 1.0
    assert report["mse_real_actions"] < 1e-9
    assert report["mse_negative_actions"] > 1e-3
