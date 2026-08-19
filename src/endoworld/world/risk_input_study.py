"""Risk-head input study on the trained v2 probabilistic ensemble.

The first real-data risk head (future-rollout mean + variances) is at chance
(AUC 0.53). Near-wall state is largely observable from the current view, so we
compare three input choices — future prediction, current latent, both — with
model selection on the calibration split and a single final test evaluation.

    python -m endoworld.world.risk_input_study \
        --data outputs/physical_actions_v2/sequences_risk.pt \
        --checkpoint outputs/probabilistic_risk_v2/probabilistic_risk.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from endoworld.world.continuous_dynamics import ContinuousDynamicsConfig
from endoworld.world.physical_actions import PhysicalActionDataset, load_sequences
from endoworld.world.probabilistic_dynamics import DynamicsEnsemble, RiskCalibrator
from endoworld.world.train_probabilistic_risk import _risk_target


class RiskHead(torch.nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_dim + 2, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, state, alea, epi):
        features = torch.cat(
            [
                state,
                alea.mean(dim=-1, keepdim=True),
                epi.mean(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        return self.network(features).squeeze(-1)


def _features(ensemble, batch, device, mode):
    history = batch["history"].to(device)
    actions = batch["actions"].to(device)
    with torch.no_grad():
        result = ensemble.predict(history, actions)
    alea = result["aleatoric_variance"].mean(dim=1, keepdim=True)
    epi = result["epistemic_variance"].mean(dim=1, keepdim=True)
    future = result["mean"].mean(dim=1, keepdim=True)
    current = history[:, -1:]
    if mode == "future":
        state = future
    elif mode == "current":
        state = current
    else:
        state = torch.cat([current, future], dim=-1)
    return (
        state.expand(-1, actions.size(1), -1),
        alea.expand(-1, actions.size(1), -1),
        epi.expand(-1, actions.size(1), -1),
    )


def _collect(ensemble, loader, device, mode, threshold):
    states, aleas, epis, targets = [], [], [], []
    for batch in loader:
        if "depth_or_risk" not in batch:
            continue
        state, alea, epi = _features(ensemble, batch, device, mode)
        states.append(state.cpu())
        aleas.append(alea.cpu())
        epis.append(epi.cpu())
        targets.append(_risk_target(batch["depth_or_risk"], threshold).flatten().cpu())
    return (torch.cat(states), torch.cat(aleas), torch.cat(epis), torch.cat(targets))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="outputs/physical_actions_v2/sequences_risk.pt"
    )
    parser.add_argument(
        "--checkpoint", default="outputs/probabilistic_risk_v2/probabilistic_risk.pt"
    )
    parser.add_argument("--near-wall-threshold", type=float, default=34.4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--out", default="outputs/probabilistic_risk_v2/risk_input_study.json"
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sequences = [s for s in load_sequences(args.data) if s.depth_or_risk is not None]
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ContinuousDynamicsConfig(**checkpoint["config"])
    members = len(
        {
            int(k.split(".")[1])
            for k in checkpoint["ensemble"]
            if k.startswith("members.")
        }
    )
    ensemble = DynamicsEnsemble(cfg, members).to(device)
    ensemble.load_state_dict(checkpoint["ensemble"])
    ensemble.eval()

    datasets = {
        split: PhysicalActionDataset(sequences, 4, 4, split)
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(ds, batch_size=32, shuffle=False)
        for split, ds in datasets.items()
    }
    cache = {}
    for mode in ("future", "current", "both"):
        for split, loader in loaders.items():
            cache[(split, mode)] = _collect(
                ensemble, loader, device, mode, args.near_wall_threshold
            )
        print(f"[risk-input] features cached for {mode}", flush=True)
    report = {"threshold": args.near_wall_threshold, "variants": {}}
    for mode in ("future", "current", "both"):
        state_tr, alea_tr, epi_tr, target_tr = cache[("train", mode)]
        state_va, alea_va, epi_va, target_va = cache[("val", mode)]
        state_te, alea_te, epi_te, target_te = cache[("test", mode)]
        head = RiskHead(state_tr.size(-1)).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=3e-4)
        for _ in range(args.epochs):
            logits = head(
                state_tr.to(device), alea_tr.to(device), epi_tr.to(device)
            ).flatten()
            loss = F.binary_cross_entropy_with_logits(logits, target_tr.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            logits_va = (
                head(state_va.to(device), alea_va.to(device), epi_va.to(device))
                .flatten()
                .cpu()
            )
            logits_te = (
                head(state_te.to(device), alea_te.to(device), epi_te.to(device))
                .flatten()
                .cpu()
            )
        calibrator = RiskCalibrator(alpha=0.1).fit(logits_va, target_va)
        report["variants"][mode] = {
            "val": calibrator.metrics(logits_va, target_va),
            "test": calibrator.metrics(logits_te, target_te),
            "n_test": len(target_te),
        }
        print(
            f"[risk-input {mode}] val AUC={report['variants'][mode]['val']['auc']:.3f} "
            f"test AUC={report['variants'][mode]['test']['auc']:.3f}",
            flush=True,
        )
    best = max(report["variants"], key=lambda m: report["variants"][m]["val"]["auc"])
    report["selected_on_val"] = best
    report["final_test"] = report["variants"][best]["test"]
    report["gate"] = {"auc": 0.75, "min_transitions": 500}
    report["passed"] = bool(
        report["final_test"]["auc"] >= 0.75 and report["final_test"]["n_test"] >= 500
    )
    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_on_val": best,
                "final_test": report["final_test"],
                "passed": report["passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
