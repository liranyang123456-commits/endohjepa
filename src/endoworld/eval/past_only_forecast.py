"""Past-only vs bidirectional forecast audit on a matched 2000-clip subset.

The headline cache encodes each 16-frame clip with the bidirectional V-JEPA 2
window, so history tubelets can attend to future tubelets within the clip.
This script rebuilds a matched subset with leakage-free encoding:
history = encode(frames[0:8]) (four tubelets, no future content), and
future = last four tubelets of encode(frames[0:16]) (targets may see history).
It then trains the same causal-L1 on both caches and reports the delta.

    python -m endoworld.eval.past_only_forecast --max-clips 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from endoworld.data.video_clips import EndoClipDataset, domain_balanced_indices
from endoworld.world.h_jepa import EndoHJEPA, HJEPAConfig, persistence_baseline
from endoworld.world.train import collate_meta


@torch.no_grad()
def encode_cache(enc, dataset, device, past_only: bool, batch_size: int = 8):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_meta,
    )
    latents, domains = [], []
    for clips, doms in loader:
        clips = clips.to(device).float()
        if past_only:
            history = enc.encode_temporal(clips[:, :8])
            future = enc.encode_temporal(clips)[:, -4:]
            z = torch.cat([history, future], dim=1)
        else:
            z = enc.encode_temporal(clips)
        latents.append(z.cpu())
        domains.append(doms.cpu())
    return torch.cat(latents), torch.cat(domains)


def train_and_eval(z_tr, d_tr, z_va, d_va, device, epochs=20, seed=0):
    torch.manual_seed(seed)
    cfg = HJEPAConfig(latent_dim=z_tr.size(-1), ablation="l1", l1_causal=True)
    model = EndoHJEPA(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    history, horizon = 4, 4
    n = z_tr.size(0)
    for _ in range(epochs):
        perm = torch.randperm(n)
        model.train()
        for start in range(0, n, 32):
            idx = perm[start : start + 32]
            z = z_tr[idx].to(device)
            pred = model.forward_l1(z[:, :history], d_tr[idx].to(device))
            loss = F.smooth_l1_loss(pred, z[:, history : history + horizon])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        z_va = z_va.to(device)
        d_va = d_va.to(device)
        pred = model.forward_l1(z_va[:, :history], d_va)
        future = z_va[:, history : history + horizon]
        persist = persistence_baseline(z_va[:, :history], horizon)
        return {
            "cos_model": F.cosine_similarity(pred, future, dim=-1).mean().item(),
            "cos_persistence": F.cosine_similarity(persist, future, dim=-1)
            .mean()
            .item(),
            "mse_model": (pred - future).pow(2).mean().item(),
            "mse_persistence": (persist - future).pow(2).mean().item(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/sequences.csv")
    parser.add_argument("--max-clips", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out", default="outputs/past_only_forecast/audit.json")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from endoworld.understanding.encoders import load_any_encoder

    enc, _, _, _ = load_any_encoder(
        "vjepa2", device, "facebook/vjepa2-vitl-fpc64-256", ""
    )

    ds_tr = EndoClipDataset(
        args.manifest,
        clip_len=16,
        stride=4,
        image_size=256,
        exclude=["EndoVis2019_ROBUST-MIS"],
        return_meta=True,
        split="train",
    )
    idx = domain_balanced_indices(ds_tr.clips, n=args.max_clips, seed=0)
    ds_tr.clips = [ds_tr.clips[i] for i in idx]
    ds_va = EndoClipDataset(
        args.manifest,
        clip_len=16,
        stride=4,
        image_size=256,
        exclude=["EndoVis2019_ROBUST-MIS"],
        return_meta=True,
        split="val",
    )
    idx_va = domain_balanced_indices(
        ds_va.clips, n=min(args.max_clips // 8, len(ds_va.clips)), seed=1
    )
    ds_va.clips = [ds_va.clips[i] for i in idx_va]

    report = {"n_train": len(ds_tr), "n_val": len(ds_va), "epochs": args.epochs}
    cache_dir = Path(args.out).parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("bidirectional", "past_only"):
        cache_path = cache_dir / f"cache_{mode}_{args.max_clips}.pt"
        if cache_path.is_file():
            pack = torch.load(cache_path, map_location="cpu", weights_only=False)
            z_tr, d_tr, z_va, d_va = (
                pack["z_tr"],
                pack["d_tr"],
                pack["z_va"],
                pack["d_va"],
            )
            print(f"[{mode}] loaded cache {cache_path}", flush=True)
        else:
            z_tr, d_tr = encode_cache(enc, ds_tr, device, past_only=mode == "past_only")
            z_va, d_va = encode_cache(enc, ds_va, device, past_only=mode == "past_only")
            torch.save(
                {"z_tr": z_tr, "d_tr": d_tr, "z_va": z_va, "d_va": d_va}, cache_path
            )
        report[mode] = train_and_eval(
            z_tr, d_tr, z_va, d_va, device, epochs=args.epochs
        )
        print(f"[{mode}] {report[mode]}", flush=True)
    report["delta_cos"] = (
        report["past_only"]["cos_model"] - report["bidirectional"]["cos_model"]
    )
    report["interpretation"] = (
        "past_only - bidirectional cosine on the same clips; a negative delta "
        "bounds how much of the headline forecast score came from "
        "within-window future attention."
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
