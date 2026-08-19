"""Apply a trained factorised-state adapter to a physical cache.

Writes a new cache whose latents are planner states (geometry+tool+semantic),
so the standard continuous-dynamics training and navigation evaluation run in
slot space without code changes.

    python -m endoworld.world.transform_slot_cache \
        --cache outputs/physical_actions_v2/sequences.pt \
        --adapter outputs/factorized_state_v2/factorized_state.pt \
        --out outputs/physical_actions_v2/sequences_slots.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from endoworld.world.factorized_state import (
    FactorizedStateAdapter,
    FactorizedStateConfig,
)
from endoworld.world.physical_actions import (
    PhysicalSequence,
    load_sequences,
    save_sequences,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="outputs/physical_actions_v2/sequences.pt")
    parser.add_argument(
        "--adapter", default="outputs/factorized_state_v2/factorized_state.pt"
    )
    parser.add_argument(
        "--out", default="outputs/physical_actions_v2/sequences_slots.pt"
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.adapter, map_location=device, weights_only=False)
    model = FactorizedStateAdapter(FactorizedStateConfig(**checkpoint["config"])).to(
        device
    )
    model.load_state_dict(checkpoint["adapter"])
    model.eval()

    sequences = load_sequences(args.cache)
    transformed = []
    with torch.no_grad():
        for seq in sequences:
            states = []
            for start in range(0, seq.latents.size(0), 256):
                chunk = seq.latents[start : start + 256].to(device)
                states.append(model(chunk)["planner_state"].cpu())
            transformed.append(
                PhysicalSequence(
                    sequence_id=seq.sequence_id,
                    dataset=seq.dataset,
                    latents=torch.cat(states),
                    actions=seq.actions,
                    depth_or_risk=seq.depth_or_risk,
                    case_id=seq.case_id,
                )
            )
    save_sequences(transformed, args.out)
    meta = {
        "source_cache": args.cache,
        "adapter": args.adapter,
        "latent_dim": int(transformed[0].latents.size(-1)),
        "space": "planner_state (geometry+tool+semantic slots)",
    }
    Path(args.out).with_suffix(".meta.json").write_text(
        __import__("json").dumps(meta, indent=2), encoding="utf-8"
    )
    print(
        f"[slot-cache] {len(transformed)} sequences -> {args.out} "
        f"(dim {meta['latent_dim']})"
    )


if __name__ == "__main__":
    main()
