"""Per-dataset forecast (and optional planning) decomposition.

The latent cache stores only domain IDs. This script reconstructs the same
video-level val clip list used at cache time (domain_balanced_indices, seed=1)
and attaches dataset names to Z_val, then reports forecast cosine / MSE per
dataset. No GPU encoding is required when a matching cache exists.

    python -m endoworld.eval.per_dataset \\
        --ckpt outputs/scale_6000_causal/endohjepa.pt \\
        --latents outputs/scale_6000_causal/latents_cache.pt \\
        --n-val 750 --seed 1

    python -m endoworld.eval.per_dataset \\
        --ckpt outputs/p2000_full_causal/endohjepa.pt \\
        --latents outputs/p2000_full_causal/latents_cache.pt \\
        --n-val 250 --seed 1 --planning
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from endoworld.data.video_clips import EndoClipDataset, domain_balanced_indices
from endoworld.eval.world_benchmark import load_predictor, maybe_pool
from endoworld.world.h_jepa import persistence_baseline


def _rebuild_val_clips(
    manifest: str, n_val: int, seed: int, clip_len: int, stride: int
):
    ds = EndoClipDataset(
        manifest,
        clip_len=clip_len,
        stride=stride,
        image_size=256,
        exclude=["EndoVis2019_ROBUST-MIS"],
        return_meta=True,
        split="val",
    )
    n = min(n_val, len(ds.clips))
    idx = domain_balanced_indices(ds.clips, n=n, seed=seed)
    return [ds.clips[i] for i in idx]


def _forecast_row(pred, persist, z_fut):
    return {
        "n": int(pred.size(0)),
        "cos_model": F.cosine_similarity(pred, z_fut, dim=-1).mean().item(),
        "cos_persist": F.cosine_similarity(persist, z_fut, dim=-1).mean().item(),
        "mse_model": (pred - z_fut).pow(2).mean().item(),
        "mse_persist": (persist - z_fut).pow(2).mean().item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--latents", default="")
    ap.add_argument("--manifest", default="manifests/sequences.csv")
    ap.add_argument("--n-val", type=int, default=750)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--planning", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    model, kind, history, horizon, _ = load_predictor(blob, device)

    latents = args.latents or str(Path(args.ckpt).parent / "latents_cache.pt")
    if not Path(latents).is_file():
        raise SystemExit(f"missing cache: {latents}")
    pack = torch.load(latents, map_location="cpu", weights_only=False)
    if pack.get("Z_val") is not None:
        Z, D = pack["Z_val"], pack["D_val"]
        split_used = "val"
    else:
        Z, D = pack["Z"], pack["D"]
        split_used = "train_fallback"
    Z = maybe_pool(Z)
    clips = _rebuild_val_clips(
        args.manifest, args.n_val, args.seed, args.clip_len, args.stride
    )
    n = min(Z.size(0), len(clips), D.size(0))
    aligned = n == Z.size(0) and n == len(clips)
    Z, D, clips = Z[:n], D[:n], clips[:n]
    names = [c.dataset for c in clips]
    domains = [c.domain for c in clips]

    t = Z.size(1)
    history = min(history, t - 1)
    horizon = min(horizon, t - history)
    z_hist, z_fut = (
        Z[:, :history].to(device),
        Z[:, history : history + horizon].to(device),
    )
    D = D.to(device)
    with torch.no_grad():
        if kind in ("gru", "mamba"):
            pred = model(z_hist)
        else:
            pred = model.forward_l1(z_hist, D)
        persist = persistence_baseline(z_hist, horizon)

    by_ds: dict[str, dict] = {}
    for ds in sorted(set(names)):
        m = torch.tensor([x == ds for x in names], device=device)
        if int(m.sum()) < 1:
            continue
        row = _forecast_row(pred[m], persist[m], z_fut[m])
        row["dataset"] = ds
        row["domain"] = next(d for n_, d in zip(names, domains) if n_ == ds)
        by_ds[ds] = row

    by_dom: dict[str, dict] = {}
    for dom in sorted(set(domains)):
        m = torch.tensor([x == dom for x in domains], device=device)
        if int(m.sum()) < 1:
            continue
        row = _forecast_row(pred[m], persist[m], z_fut[m])
        row["domain"] = dom
        by_dom[dom] = row

    report = {
        "paper": "Endo-HJEPA",
        "ckpt": args.ckpt,
        "latents": latents,
        "split": split_used,
        "n_used": n,
        "n_cache": int(
            pack["Z_val"].size(0)
            if pack.get("Z_val") is not None
            else pack["Z"].size(0)
        ),
        "n_clips_rebuilt": len(
            _rebuild_val_clips(
                args.manifest, args.n_val, args.seed, args.clip_len, args.stride
            )
        ),
        "aligned": aligned,
        "history": history,
        "horizon": horizon,
        "overall": _forecast_row(pred, persist, z_fut),
        "by_dataset": by_ds,
        "by_domain": by_dom,
        "note": "aligned=True means Z_val[i] is assumed to match the rebuilt clip list. "
        "If aligned is False, treat per-dataset rows as approximate.",
    }

    if args.planning and kind == "hjepa":
        from endoworld.eval.eval_ckpt import _reach

        plan_by = {}
        for ds in sorted(set(names)):
            m = torch.tensor([x == ds for x in names], device=device)
            if int(m.sum()) < 2:
                continue
            plan_by[ds] = _reach(model, Z[m.cpu()].to(device), D[m], history, horizon)
        report["planning_by_dataset"] = plan_by

    out = args.out or str(Path(args.ckpt).parent / "per_dataset.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[per-dataset] n={n} aligned={aligned} wrote {out}")
    print(f"{'dataset':28s} {'dom':7s} {'n':>4s} {'cos':>6s} {'persist':>7s} {'Δ':>6s}")
    for ds, r in sorted(by_ds.items(), key=lambda kv: (-kv[1]["n"], kv[0])):
        delta = r["cos_model"] - r["cos_persist"]
        print(
            f"{ds:28s} {r['domain']:7s} {r['n']:4d} {r['cos_model']:6.3f} "
            f"{r['cos_persist']:7.3f} {delta:+6.3f}"
        )


if __name__ == "__main__":
    main()
