"""Fast regression tests for sampling, protocol, and checkpoint safety fixes."""
from __future__ import annotations

import torch

from endoworld.eval.cholect50_dense_probe import AttentionPhaseProbe, _fit_probe
from endoworld.eval.cholect50_probe import CHALLENGE_TEST_VIDS, split_indices
from endoworld.world.train_util import clone_state_dict


def test_cholect_official_split_matches_challenge_videos():
    vids = [f"VID{i:02d}" for i in range(1, 81)]
    train_idx, test_idx = split_indices(vids, official=True)
    train_vids = {vids[i] for i in train_idx}
    test_vids = {vids[i] for i in test_idx}
    assert train_vids.isdisjoint(test_vids)
    assert test_vids == set(CHALLENGE_TEST_VIDS)


def test_dense_probe_factory_is_seeded_before_initialisation():
    features = (torch.randn(3, 2, 5),)
    targets = (torch.tensor([0, 1, 2]),)

    def initialise(seed):
        return _fit_probe(
            lambda: AttentionPhaseProbe(5),
            features,
            targets,
            "phase",
            "cpu",
            seed,
            steps=0,
        ).state_dict()

    first = initialise(7)
    repeat = initialise(7)
    different = initialise(8)
    assert all(torch.equal(first[key], repeat[key]) for key in first)
    assert any(not torch.equal(first[key], different[key]) for key in first)


def test_build_clip_index_enforces_split_and_exclusions(tmp_path):
    import csv

    from endoworld.data.video_clips import build_clip_index

    frames = tmp_path / "frames_a"
    frames.mkdir()
    for i in range(40):
        (frames / f"{i:04d}.png").write_bytes(b"x")
    manifest = tmp_path / "sequences.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "sequence_id", "frames_dir", "modality",
                        "split", "domain"],
        )
        writer.writeheader()
        writer.writerow({
            "dataset": "DatasetA", "sequence_id": "seq-train",
            "frames_dir": str(frames), "modality": "frames",
            "split": "train", "domain": "laparoscopy",
        })
        writer.writerow({
            "dataset": "DatasetA", "sequence_id": "seq-test",
            "frames_dir": str(frames), "modality": "frames",
            "split": "test", "domain": "laparoscopy",
        })
    train_clips = build_clip_index(manifest, clip_len=4, stride=2, split="train")
    assert train_clips and {c.sequence_id for c in train_clips} == {"seq-train"}
    excluded = build_clip_index(
        manifest, clip_len=4, stride=2, split="train", exclude={"DatasetA"})
    assert excluded == []


def test_stir_endpoint_padding_is_reproducible():
    from endoworld.eval.stir_experiment import _pad_points

    pts = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    padded = _pad_points(pts, 5)
    assert padded.shape == (5, 2)
    assert torch.equal(padded[:2], pts)
    assert torch.equal(padded[2:], pts[-1:].expand(3, 2))
    assert torch.equal(_pad_points(pts, 5), padded)


def test_best_state_snapshot_does_not_share_cpu_storage():
    model = torch.nn.Linear(3, 2)
    snapshot = clone_state_dict(model)
    original = snapshot["weight"].clone()
    with torch.no_grad():
        model.weight.add_(1.0)
    assert torch.equal(snapshot["weight"], original)
    assert not torch.equal(snapshot["weight"], model.weight)
