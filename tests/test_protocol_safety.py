"""Fast regression tests for sampling, protocol, and checkpoint safety fixes."""
from __future__ import annotations

import torch

from endoworld.eval.cholect50_dense_probe import AttentionPhaseProbe, _fit_probe


def test_dense_probe_full_batch_fit_is_deterministic_and_converges():
    torch.manual_seed(0)
    features = (torch.randn(6, 2, 5),)
    targets = (torch.tensor([0, 1, 2, 0, 1, 2]),)

    def fit():
        torch.manual_seed(7)
        head = AttentionPhaseProbe(5)
        return _fit_probe(head, features, targets, "phase", "cpu", 7, steps=30)

    first = fit().state_dict()
    repeat = fit().state_dict()
    assert all(torch.equal(first[key], repeat[key]) for key in first)
    trained = fit()
    with torch.no_grad():
        pred = trained(features[0]).argmax(-1)
    assert (pred == targets[0]).float().mean() > 0.5
