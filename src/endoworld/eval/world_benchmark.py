"""World-model evaluation: multi-horizon latent forecast vs persistence / GRU / H-JEPA.

    python -m endoworld.eval.world_benchmark --ckpt outputs/endohjepa/endohjepa.pt \\
        --latents outputs/endohjepa/latents_cache.pt
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from endoworld.world.baselines import GRUDynamics
from endoworld.world.h_jepa import EndoHJEPA, HJEPAConfig, persistence_baseline


def _cfg_from_blob(blob) -> HJEPAConfig:
    fields = HJEPAConfig.__dataclass_fields__
    kwargs = {k: blob["wcfg"][k] for k in fields if k in blob.get("wcfg", {})}
    return HJEPAConfig(**kwargs)


def load_predictor(blob, device):
    kind = blob.get("kind", "hjepa")
    history, horizon = blob["history"], blob["horizon"]
    if kind in ("gru", "mamba"):
        from endoworld.world.baselines import MambaDynamics
        dim = blob.get("embed_dim") or blob["wcfg"]["latent_dim"]
        hidden = blob["wcfg"].get("hidden_dim", 512)
        cls = GRUDynamics if kind == "gru" else MambaDynamics
        model = cls(dim, hidden, horizon).to(device)
        model.load_state_dict(blob["model"])
        model.eval()
        return model, kind, history, horizon, None
    cfg = _cfg_from_blob(blob)
    model = EndoHJEPA(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, kind, history, horizon, cfg


def horizon_table(pred, persist, z_fut) -> list[dict]:
    horizon = z_fut.size(1)
    rows = []
    wanted = [h for h in (1, 4, 8, 16) if h <= horizon]
    if horizon not in wanted:
        wanted.append(horizon)
    for h in wanted:
        rows.append({
            "horizon": h,
            "cos_model": F.cosine_similarity(pred[:, :h], z_fut[:, :h], dim=-1).mean().item(),
            "cos_persist": F.cosine_similarity(persist[:, :h], z_fut[:, :h], dim=-1).mean().item(),
            "mse_model": (pred[:, :h] - z_fut[:, :h]).pow(2).mean().item(),
            "mse_persist": (persist[:, :h] - z_fut[:, :h]).pow(2).mean().item(),
        })
    return rows


def predict(model, kind, z_hist, D):
    if kind in ("gru", "mamba"):
        return model(z_hist)
    return model.forward_l1(z_hist, D)


def maybe_pool(Z):
    return Z.mean(dim=2) if Z.dim() == 4 else Z


def cross_domain_rows(model, kind, Z, D, history, horizon) -> list[dict]:
    from endoworld.data.domains import ID_TO_DOMAIN
    z = maybe_pool(Z)
    rows = []
    for did in D.unique().tolist():
        m = D == did
        if int(m.sum()) < 2:
            continue
        z_hist, z_fut = z[m, :history], z[m, history:history + horizon]
        persist = persistence_baseline(z_hist, horizon)
        pred = predict(model, kind, z_hist, D[m])
        rows.append({
            "domain": ID_TO_DOMAIN.get(int(did), str(int(did))),
            "n": int(m.sum()),
            "cos_model": F.cosine_similarity(pred, z_fut, dim=-1).mean().item(),
            "cos_persist": F.cosine_similarity(persist, z_fut, dim=-1).mean().item(),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/endohjepa/endohjepa.pt")
    ap.add_argument("--latents", default=None, help="cached (N,T,D) or (N,T,Ntok,D) pt")
    ap.add_argument("--out", default="outputs/endohjepa/benchmark.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    model, kind, history, horizon, _ = load_predictor(blob, device)

    latents_path = args.latents or os.path.join(os.path.dirname(args.ckpt), "latents_cache.pt")
    if not os.path.isfile(latents_path):
        raise SystemExit("Provide --latents cache from training or re-run endoworld.world.train")
    pack = torch.load(latents_path, map_location=device, weights_only=False)
    # prefer the video-level val split when the cache carries it (honest held-out eval)
    if pack.get("Z_val") is not None:
        Z, D = pack["Z_val"].to(device), pack["D_val"].to(device)
        split_used = "val"
    else:
        Z, D = pack["Z"].to(device), pack["D"].to(device)
        split_used = "train"
    z = maybe_pool(Z)
    t = z.size(1)
    history = min(history, t - 1)
    horizon = min(horizon, t - history)
    z_hist, z_fut = z[:, :history], z[:, history:history + horizon]
    with torch.no_grad():
        pred = predict(model, kind, z_hist, D)
        persist = persistence_baseline(z_hist, horizon)
        rows = horizon_table(pred, persist, z_fut)
        by_dom = cross_domain_rows(model, kind, z, D, history, horizon)
    report = {
        "kind": kind,
        "ablation": blob.get("ablation"),
        "split": split_used,
        "dense_cache": bool(pack.get("dense", Z.dim() == 4)),
        "horizons": rows,
        "cross_domain": by_dom,
        "paper": "Endo-HJEPA",
        "not_ablation_planning": True,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
