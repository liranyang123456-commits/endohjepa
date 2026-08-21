"""Protocol tests for the C3VD ray model, the retrieval proxy, and fixed banks."""

from __future__ import annotations

import numpy as np
import torch

from endoworld.world.physical_actions import PhysicalActionDataset, PhysicalSequence


def test_c3vd_ray_model_roundtrip():
    from endoworld.eval.c3vd_pose_gate import cam2ray, ray2cam

    # keep pixels inside the calibrated region (the polynomial goes negative
    # beyond rho ~ 620, i.e. behind the camera)
    u = np.array([679.54, 700.0, 400.0])
    v = np.array([543.98, 600.0, 300.0])
    points = cam2ray(u, v) * 7.0
    u2, v2, valid = ray2cam(points)
    assert np.all(valid)
    assert np.allclose(u2, u, atol=1.0)
    assert np.allclose(v2, v, atol=1.0)


def test_navigation_retrieval_and_pose_error():
    from endoworld.eval.physical_navigation import _pose_error

    # current proxy retrieval: nearest latent from the current index onward
    latents = torch.tensor([[0.0], [5.0], [10.0]])
    predicted = torch.tensor([[4.9]])
    nearest = int(torch.cdist(predicted, latents[0:]).argmin())
    assert nearest == 1

    # pure translation: the twist translation equals the Euclidean one
    t_only = np.eye(4)
    t_only[:3, 3] = [3.0, 4.0, 0.0]
    translation, rotation = _pose_error(np.eye(4), t_only)
    assert np.isclose(translation, 5.0)
    assert np.isclose(rotation, 0.0)

    # pure rotation about z: translation twist vanishes, angle is exact
    r_only = np.eye(4)
    r_only[:3, :3] = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation, rotation = _pose_error(np.eye(4), r_only)
    assert np.isclose(translation, 0.0)
    assert np.isclose(rotation, 90.0)


def _scared_sequence(n_latents: int) -> PhysicalSequence:
    actions = torch.arange(
        (n_latents - 1) * 6, dtype=torch.float32).reshape(n_latents - 1, 6) / 100
    latents = torch.cat([torch.zeros(1, 6), actions.cumsum(dim=0)])
    return PhysicalSequence(
        sequence_id="scared:datasets/SCARED/dataset_7/keyframe_1",
        dataset="SCARED",
        case_id="dataset_7",
        latents=latents,
        actions=actions,
    )


def test_hard_negative_is_same_sequence_and_local():
    dataset = PhysicalActionDataset(
        [_scared_sequence(40)], history=2, horizon=1, split="test")
    assert len(dataset) > 2
    rng = np.random.default_rng(0)
    for index in range(len(dataset)):
        other = dataset.hard_negative_index(index, radius=64, rng=rng)
        assert other != index
        assert dataset.windows[other][0] == dataset.windows[index][0]


def test_fixed_bank_reports_pair_and_window_rates():
    from endoworld.world.train_continuous_actions import evaluate_fixed_bank

    class Integrator(torch.nn.Module):
        def forward(self, history, actions):
            return history[:, -1:] + actions.cumsum(dim=1)

    dataset = PhysicalActionDataset(
        [_scared_sequence(8)], history=2, horizon=1, split="test")
    report = evaluate_fixed_bank(
        Integrator(), dataset, "cpu", n_negatives=4, radius=64, seed=7)
    assert report["n"] == len(dataset)
    assert report["n_negatives"] == 4
    # the exact integrator has zero real-action error, so it wins every pair
    assert report["pair_win_fraction"] == 1.0
    assert report["all_negative_win_fraction"] == 1.0
