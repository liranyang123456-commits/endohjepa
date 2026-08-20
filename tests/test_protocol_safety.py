"""Fast regression tests for sampling, protocol, and checkpoint safety fixes."""

from __future__ import annotations

import runpy
from argparse import Namespace
from pathlib import Path

import torch

from endoworld.data.video_clips import ClipSpec
from endoworld.eval.cholect50_dense_probe import AttentionPhaseProbe, _fit_probe
from endoworld.eval.cholect50_probe import _clip_starts
from endoworld.eval.stir_experiment import split_sequences
from endoworld.understanding.adapt import (
    _adaptation_audit,
    _cli_values,
    build_argparser,
    filter_adaptation_clips,
)
from endoworld.world.train_factorized_state import _clone_state_dict


def _clip(dataset: str, sequence_id: str, split: str = "train") -> ClipSpec:
    return ClipSpec(
        dataset=dataset,
        frames_dir=f"/frames/{sequence_id}",
        frame_files=[f"{i}.png" for i in range(64)],
        start=0,
        clip_len=4,
        stride=2,
        sequence_id=sequence_id,
        split=split,
    )


def test_cholect_clip_starts_respect_strided_span():
    # span = (16 - 1) * 2 + 1 = 31
    assert _clip_starts(30, 16, 2, 24) == []
    assert _clip_starts(31, 16, 2, 24) == [0]
    assert _clip_starts(32, 16, 2, 24) == [0, 1]
    starts = _clip_starts(63, 16, 2, 24)
    assert starts[-1] == 63 - 31
    assert all(start + (16 - 1) * 2 < 63 for start in starts)


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


def test_adaptation_filters_and_audit_exclude_held_out_video_ids():
    clips = [
        _clip("CholecT50", r"CholecT50\videos\VID68"),
        _clip("CholecT50", r"CholecT50\videos\VID01"),
        _clip("Other", "secret-video"),
        _clip("Excluded", "ordinary"),
        _clip("Other", "validation-only", split="val"),
    ]
    selected = filter_adaptation_clips(
        clips,
        allowed_splits={"train"},
        excluded_datasets={"excluded"},
        excluded_video_ids={"secret-video"},
        allow_cholect50_held_out=False,
    )
    assert [clip.sequence_id for clip in selected] == [r"CholecT50\videos\VID01"]

    explicitly_allowed = filter_adaptation_clips(
        clips[:2],
        allowed_splits={"train"},
        excluded_datasets=set(),
        excluded_video_ids=set(),
        allow_cholect50_held_out=True,
    )
    assert len(explicitly_allowed) == 2

    args = Namespace(
        manifest="manifest.csv",
        allow_cholect50_held_out=False,
    )
    audit = _adaptation_audit(selected, args, {"train"}, {"excluded"}, {"secret-video"})
    assert audit["n_clips"] == 1
    assert len(audit["clip_ids_sha256"]) == 64
    assert "VID01" in audit["clip_ids"][0]


def test_adaptation_filter_cli_accepts_repeatable_allowlists():
    args = build_argparser().parse_args(
        [
            "--allow-splits",
            "train,val",
            "--exclude-datasets",
            "DatasetA",
            "--exclude-datasets",
            "DatasetB,DatasetC",
            "--exclude-video-ids",
            "VID68,VID70",
        ]
    )
    assert _cli_values(args.allow_splits) == {"train", "val"}
    assert _cli_values(args.exclude_datasets) == {"dataseta", "datasetb", "datasetc"}
    assert _cli_values(args.exclude_video_ids) == {"vid68", "vid70"}


def test_stir_sequence_split_is_disjoint_and_reproducible():
    sequences = [Path(f"patient/sequence-{i}") for i in range(10)]
    train_a, test_a = split_sequences(sequences, train_fraction=0.7, seed=4)
    train_b, test_b = split_sequences(sequences, train_fraction=0.7, seed=4)
    assert train_a == train_b and test_a == test_b
    assert set(train_a).isdisjoint(test_a)
    assert set(train_a) | set(test_a) == set(sequences)


def test_stir_split_keeps_patient_views_together():
    sequences = [
        Path(f"STIRChallenge_2024/{patient}/{view}/seq00")
        for patient in ("01", "02", "03", "04")
        for view in ("left", "right")
    ]
    train, test = split_sequences(sequences, train_fraction=0.5, seed=2)
    train_patients = {path.parent.parent.name for path in train}
    test_patients = {path.parent.parent.name for path in test}
    assert train_patients.isdisjoint(test_patients)
    assert len(train_patients) == len(test_patients) == 2


def test_best_state_snapshot_does_not_share_cpu_storage():
    model = torch.nn.Linear(3, 2)
    snapshot = _clone_state_dict(model)
    original = snapshot["weight"].clone()
    with torch.no_grad():
        model.weight.add_(1.0)
    assert torch.equal(snapshot["weight"], original)
    assert not torch.equal(snapshot["weight"], model.weight)


def test_provenance_sanitizer_redacts_relative_private_paths():
    script = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "endohjepa"
        / "sanitize_provenance.py"
    )
    sanitize = runpy.run_path(str(script))["_sanitize"]
    cleaned = sanitize(
        {
            "dataset": "ION_bronch",
            "path": (
                "datasets/ION_bronch/case_007/intraop_00/"
                "Procedure-20211230/private_folder/frame_001919.jpg"
            ),
        }
    )
    assert cleaned["path"] == (
        "datasets/ION_bronch/case_007/anonymised_sequence/frame_001919.jpg"
    )
    assert "Procedure-" not in cleaned["path"]
