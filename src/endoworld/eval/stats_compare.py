"""Paired statistical comparison of two world-model checkpoints on latent forecast.

Per-clip cosine on a video-level val split, then paired bootstrap CI and Wilcoxon
signed-rank test, with Holm correction across horizons. This is what makes a close
margin (e.g. 0.973 vs 0.970) statistically defensible.

    python -m endoworld.eval.stats_compare \
        --a outputs/p2000_full_causal/endohjepa.pt \
        --b outputs/p2000_gru/endohjepa.pt \
        --latents outputs/cache_1000_t16/latents_cache.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from endoworld.eval.world_benchmark import load_predictor
from endoworld.world.h_jepa import persistence_baseline


def _per_clip_cos(model, kind, z_hist, z_fut, D):
    with torch.no_grad():
        pred = model.forward_l1(z_hist, D) if kind == "hjepa" else model(z_hist)
    h = min(pred.size(1), z_fut.size(1))
    # per-clip cosine averaged over horizon steps
    cos = F.cosine_similarity(pred[:, :h], z_fut[:, :h], dim=-1).mean(dim=1)
    return cos.cpu().numpy()


def _bootstrap_diff(a, b, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    d = a - b
    n = len(d)
    means = rng.choice(d, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def _wilcoxon(a, b):
    from scipy.stats import wilcoxon

    d = a - b
    d = d[np.abs(d) > 1e-12]
    if len(d) < 5:
        return float("nan")
    return float(wilcoxon(d).pvalue)


def _holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray([np.nan if v is None else v for v in pvals], dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[ok])]
    m = len(order)
    prev = 0.0
    for rank, i in enumerate(order):
        adj = (m - rank) * p[i]
        prev = max(prev, adj)
        out[i] = min(prev, 1.0)
    return out.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="checkpoint A (e.g. H-JEPA causal)")
    ap.add_argument("--b", required=True, help="checkpoint B (e.g. GRU)")
    ap.add_argument(
        "--latents", required=True, help="cache with Z_val (pooled or dense)"
    )
    ap.add_argument("--horizons", default="1,4,8")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pack = torch.load(args.latents, map_location="cpu", weights_only=False)
    if pack.get("Z_val") is not None:
        Z, D = pack["Z_val"], pack["D_val"]
        split = "val"
    else:
        Z, D = pack["Z"], pack["D"]
        split = "train"
    if Z.dim() == 4:
        Z = Z.mean(dim=2)
    Z, D = Z.to(device), D.to(device)
    t = Z.size(1)

    blob_a = torch.load(args.a, map_location=device, weights_only=False)
    blob_b = torch.load(args.b, map_location=device, weights_only=False)
    ma, ka, ha, _, _ = load_predictor(blob_a, device)
    mb, kb, hb, _, _ = load_predictor(blob_b, device)
    history = min(ha, hb, t - 1)

    horizons = [int(h) for h in args.horizons.split(",") if int(h) <= t - history]
    rows = []
    pvals = []
    for h in horizons:
        z_hist, z_fut = Z[:, :history], Z[:, history : history + h]
        ca = _per_clip_cos(ma, ka, z_hist, z_fut, D)
        cb = _per_clip_cos(mb, kb, z_hist, z_fut, D)
        cp = (
            _per_clip_cos(None, "persist", z_hist, z_fut, D)
            if False
            else F.cosine_similarity(persistence_baseline(z_hist, h), z_fut, dim=-1)
            .mean(1)
            .cpu()
            .numpy()
        )
        mean_d, lo, hi = _bootstrap_diff(ca, cb, args.n_boot)
        p = _wilcoxon(ca, cb)
        pvals.append(p)
        rows.append(
            {
                "horizon": h,
                "n_clips": int(len(ca)),
                "cos_A": float(ca.mean()),
                "cos_B": float(cb.mean()),
                "cos_persist": float(cp.mean()),
                "diff_A_minus_B": mean_d,
                "boot_ci95": [lo, hi],
                "wilcoxon_p": p,
            }
        )
    adj = _holm(pvals)
    for r, ap_ in zip(rows, adj):
        r["holm_p"] = ap_
        r["significant_005"] = bool(ap_ == ap_ and ap_ < 0.05)

    report = {
        "paper": "Endo-HJEPA",
        "not_ablation_planning": True,
        "A": args.a,
        "B": args.b,
        "split": split,
        "history": history,
        "n_bootstrap": args.n_boot,
        "correction": "holm",
        "rows": rows,
    }
    out = args.out or str(Path(args.a).parent / "stats_compare.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
