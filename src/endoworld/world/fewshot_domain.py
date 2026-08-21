"""Few-shot domain adaptation: zero-shot transfer fails -> few-shot domain-token recovery.

Train the world model on a source domain (laparo). Zero-shot forecast on held-out
GI/bronch is below persistence (cross-domain is hard). We then fine-tune ONLY the
domain embedding on a few target-domain clips and measure the forecast recovery.
This is the standard, honest result: zero-shot is hard, few-shot domain tokens adapt.

    python -m endoworld.world.fewshot_domain --ckpt outputs/t16_transfer_laparo/endohjepa.pt \
        --latents outputs/cache_1000_t16/latents_cache.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from endoworld.data.domains import DOMAIN_IDS, ID_TO_DOMAIN
from endoworld.eval.world_benchmark import load_predictor
from endoworld.world.h_jepa import persistence_baseline


def _per_domain_cos(model, Z, D, history, horizon):
    out = {}
    with torch.no_grad():
        for did in D.unique().tolist():
            m = D == did
            if int(m.sum()) < 4:
                continue
            z_hist, z_fut = Z[m, :history], Z[m, history:history + horizon]
            pred = model.forward_l1(z_hist, D[m])
            persist = persistence_baseline(z_hist, horizon)
            out[ID_TO_DOMAIN.get(int(did), str(did))] = {
                "cos_model": float(F.cosine_similarity(pred, z_fut, dim=-1).mean()),
                "cos_persist": float(F.cosine_similarity(persist, z_fut, dim=-1).mean()),
                "n": int(m.sum()),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="model trained on source domain only")
    ap.add_argument("--latents", required=True, help="pooled cache with Z/D (+Z_val)")
    ap.add_argument("--target", default="gi", help="target domain to adapt to")
    ap.add_argument("--shots", type=int, default=32, help="target-domain clips for adaptation")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pack = torch.load(args.latents, map_location="cpu", weights_only=False)
    if pack.get("Z_val") is not None:
        Z, D = pack["Z_val"], pack["D_val"]
    else:
        Z, D = pack["Z"], pack["D"]
    if Z.dim() == 4:
        Z = Z.mean(dim=2)

    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    model, kind, history, horizon, _ = load_predictor(blob, device)
    history = min(history, Z.size(1) - 1)
    horizon = min(horizon, Z.size(1) - history)
    tgt = DOMAIN_IDS[args.target]
    tgt_mask = D == tgt
    Z_t, D_t = Z[tgt_mask].to(device), D[tgt_mask].to(device)
    if len(Z_t) < args.shots + 4:
        print(f"[fewshot] not enough {args.target} clips: {len(Z_t)}")
        return

    # split target into few-shot adapt / held-out eval
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(Z_t), generator=g)
    adapt_idx, eval_idx = perm[: args.shots], perm[args.shots:]
    Z_ad, D_ad = Z_t[adapt_idx], D_t[adapt_idx]
    Z_ev, D_ev = Z_t[eval_idx], D_t[eval_idx]

    # zero-shot baseline on held-out target
    model.eval()
    zero = _per_domain_cos(model, Z_ev, D_ev, history, horizon)

    # few-shot: fine-tune ONLY the domain embedding
    for p in model.parameters():
        p.requires_grad_(False)
    model.domain.embed.weight.requires_grad_(True)
    opt = torch.optim.AdamW([model.domain.embed.weight], lr=args.lr)
    model.train()
    zh, zf = Z_ad[:, :history], Z_ad[:, history:history + horizon]
    for epoch in range(args.epochs):
        pred = model.forward_l1(zh, D_ad)
        loss = F.smooth_l1_loss(pred, zf)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    few = _per_domain_cos(model, Z_ev, D_ev, history, horizon)

    report = {
        "paper": "Endo-HJEPA", "not_ablation_planning": True,
        "task": f"few-shot domain-token adaptation to {args.target}",
        "shots": args.shots, "source_model": args.ckpt,
        "zero_shot": zero.get(args.target), "few_shot": few.get(args.target),
        "recovery": (few.get(args.target, {}).get("cos_model", 0)
                     - zero.get(args.target, {}).get("cos_model", 0)),
    }
    out = args.out or str(Path(args.ckpt).parent / f"fewshot_{args.target}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
