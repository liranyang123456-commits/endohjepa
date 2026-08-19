"""Train the factorised-state adapter on the v2 physical cache.

The adapter is trained with frozen-teacher fidelity, cross-slot separation,
and a geometry->twist supervision that ties the geometry slot to the executed
SE(3) increment. The trained adapter then maps pooled teacher latents to
planner states for slot-space dynamics and MPC.

    python -m endoworld.world.train_factorized_state \
        --data outputs/physical_actions_v2/sequences.pt \
        --out outputs/factorized_state_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from endoworld.world.factorized_state import (
    FactorizedStateAdapter,
    FactorizedStateConfig,
)
from endoworld.world.physical_actions import (
    PhysicalActionDataset,
    load_sequences,
    video_split,
)


def _clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot parameters without sharing storage with the live module."""
    return {
        key: value.detach().cpu().clone() for key, value in module.state_dict().items()
    }


def _action_stats(sequences):
    actions = torch.cat(
        [
            s.actions
            for s in sequences
            if video_split(s.sequence_id, case_id=s.case_id, dataset=s.dataset)
            == "train"
        ]
    )
    return actions.mean(0), actions.std(0).clamp_min(1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/physical_actions_v2/sequences.pt")
    parser.add_argument("--out", default="outputs/factorized_state_v2")
    parser.add_argument("--slot-dim", type=int, default=128)
    parser.add_argument("--adapter-rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sequences = load_sequences(args.data)
    train_data = PhysicalActionDataset(sequences, 4, 4, "train")
    val_data = PhysicalActionDataset(sequences, 4, 4, "val")
    mean, std = _action_stats(sequences)

    cfg = FactorizedStateConfig(
        teacher_dim=sequences[0].latents.size(-1),
        slot_dim=args.slot_dim,
        adapter_rank=args.adapter_rank,
    )
    model = FactorizedStateAdapter(cfg).to(device)
    geometry_twist = torch.nn.Linear(cfg.slot_dim, 6).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(geometry_twist.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    best = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        running, seen = 0.0, 0
        for batch in loader:
            history = batch["history"].to(device)
            actions = batch["actions"].to(device)
            batch["future"].to(device)
            losses = model.losses(history.reshape(-1, history.size(-1)))
            # Geometry slot must predict the executed twist (normalised).
            geometry_slot = model.slot_projectors["geometry"](
                model.slot_norm(model.adapter(history[:, -1].detach()))
            )
            twist_pred = geometry_twist(geometry_slot)
            twist_target = (actions[:, 0] - mean.to(device)) / std.to(device)
            twist_loss = F.smooth_l1_loss(twist_pred, twist_target)
            total = losses["total"] + 0.5 * twist_loss
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(geometry_twist.parameters()), 1.0
            )
            optimizer.step()
            running += total.item() * history.size(0)
            seen += history.size(0)
        # validation: fidelity + twist R^2
        model.eval()
        with torch.no_grad():
            val_loader = DataLoader(val_data, batch_size=args.batch_size)
            fid, preds, targets = [], [], []
            for batch in val_loader:
                history = batch["history"].to(device)
                out = model(history[:, -1])
                fid.append(
                    F.smooth_l1_loss(
                        out["reconstructed_teacher"], history[:, -1]
                    ).item()
                )
                preds.append(
                    geometry_twist(
                        model.slot_projectors["geometry"](
                            model.slot_norm(model.adapter(history[:, -1].detach()))
                        )
                    ).cpu()
                )
                targets.append(((batch["actions"][:, 0] - mean) / std).cpu())
            pred = torch.cat(preds)
            target = torch.cat(targets)
            r2 = 1 - (pred - target).square().sum() / max(
                (target - target.mean(0)).square().sum().item(), 1e-8
            )
            score = float(np_mean(fid)) - 0.1 * float(r2)
        if score < best:
            best = score
            best_state = {
                "adapter": _clone_state_dict(model),
                "geometry_twist": _clone_state_dict(geometry_twist),
            }
        print(
            f"[epoch {epoch + 1}] train={running / max(seen, 1):.4f} "
            f"val_fidelity={np_mean(fid):.4f} twist_R2={r2:.3f}",
            flush=True,
        )
    if best_state is not None:
        model.load_state_dict(best_state["adapter"])
        geometry_twist.load_state_dict(best_state["geometry_twist"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter": model.state_dict(),
            "geometry_twist": geometry_twist.state_dict(),
            "config": cfg.__dict__,
            "action_mean": mean,
            "action_std": std,
        },
        out / "factorized_state.pt",
    )
    report = {"val_fidelity": float(np_mean(fid)), "val_twist_r2": float(r2)}
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def np_mean(values):
    import numpy as np

    return float(np.mean(values)) if values else float("nan")


if __name__ == "__main__":
    main()
