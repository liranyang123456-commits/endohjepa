"""Self-contained eval of a trained world-model checkpoint.

Encodes a video-level val split on the fly (pooled latents, small memory) and
reports multi-horizon forecast vs persistence plus latent-MPC planning. This avoids
needing a large saved dense cache.

    python -m endoworld.eval.eval_ckpt --ckpt outputs/v2_2000_full/endohjepa.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from endoworld.data.video_clips import EndoClipDataset, domain_balanced_indices
from endoworld.eval.world_benchmark import load_predictor, horizon_table, cross_domain_rows
from endoworld.world.h_jepa import persistence_baseline
from endoworld.world.plan_mpc import rollout_path_energy
from endoworld.world.train import cache_latents, collate_meta


def _reach(model, z, D, history, horizon, n_samples=32):
    z_hist, z_fut = z[:, :history], z[:, history:history + horizon]
    z_goal = z_fut[:, -1]
    persist = persistence_baseline(z_hist, horizon)[:, -1]
    with torch.no_grad():
        pred_l1 = model.forward_l1(z_hist, D)[:, -1]
        plan_a, plan_e = model.plan(z_hist, z_goal, D, n_samples=n_samples, steps=horizon)
        h = min(plan_a.size(1), model.cfg.horizon)
        a = plan_a[:, :h]
        if a.size(1) < model.cfg.horizon:
            pad = torch.zeros(z.size(0), model.cfg.horizon - a.size(1),
                              dtype=torch.long, device=z.device)
            a = torch.cat([a, pad], dim=1)
        pred_plan_path = model.forward_l3(z_hist, a, D)
        pred_plan = pred_plan_path[:, -1]
        _, e_plan = rollout_path_energy(model, z_hist[:, -1], a, pred_plan_path)
        rand_a = torch.randint(
            0, model.cfg.n_actions, (z.size(0), model.cfg.horizon), device=z.device)
        pred_rand_path = model.forward_l3(z_hist, rand_a, D)
        _, e_rand = rollout_path_energy(model, z_hist[:, -1], rand_a, pred_rand_path)
    cos = lambda u, v: F.cosine_similarity(u, v, dim=-1)
    better = (cos(pred_plan, z_goal) > cos(persist, z_goal)).float().mean().item()
    return {
        "cos_plan": cos(pred_plan, z_goal).mean().item(),
        "cos_l1": cos(pred_l1, z_goal).mean().item(),
        "cos_persist": cos(persist, z_goal).mean().item(),
        "plan_better_than_persist": better,
        "mean_plan_energy": plan_e.mean().item(),
        "energy_plan_vs_random": (e_plan - e_rand).mean().item(),
        "energy_plan_lower_frac": (e_plan < e_rand).float().mean().item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="manifests/sequences.csv")
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--max-val", type=int, default=250)
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--vjepa2-id", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--vjepa2-adapted", default="")
    ap.add_argument("--latents", default="",
                    help="optional pooled cache with Z_val/D_val; avoids re-encoding")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    model, kind, history, horizon, _ = load_predictor(blob, device)

    if args.latents and Path(args.latents).is_file():
        pack = torch.load(args.latents, map_location="cpu", weights_only=False)
        Z = pack.get("Z_val")
        D = pack.get("D_val")
        if Z is None or D is None:
            raise RuntimeError("--latents cache must contain Z_val and D_val")
        if Z.dim() == 4:
            Z = Z.mean(dim=2)
        if args.max_val and Z.size(0) > args.max_val:
            Z, D = Z[:args.max_val], D[:args.max_val]
    else:
        # encode the video-level val split on the fly (pooled latents)
        if args.vjepa2_adapted and Path(args.vjepa2_adapted).is_file():
            from endoworld.understanding.encoders import load_adapted_vjepa2
            enc, _, image_size, _ = load_adapted_vjepa2(args.vjepa2_adapted, device)
        else:
            from endoworld.understanding.vjepa2_hf import VJEPA2Encoder
            enc = VJEPA2Encoder(args.vjepa2_id, device=device)
            image_size = enc.image_size
        ds = EndoClipDataset(args.manifest, clip_len=args.clip_len, stride=4,
                             image_size=image_size, exclude=["EndoVis2019_ROBUST-MIS"],
                             return_meta=True, split="val")
        idx = domain_balanced_indices(ds.clips, n=min(args.max_val, len(ds.clips)), seed=1)
        ds.clips = [ds.clips[i] for i in idx]
        dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_meta)
        Z, D = cache_latents(enc, dl, device, dense=False)  # pooled (N, T, D)
    Z, D = Z.to(device), D.to(device)
    t = Z.size(1)
    history = min(history, t - 1)
    horizon = min(horizon, t - history)
    z_hist, z_fut = Z[:, :history], Z[:, history:history + horizon]

    with torch.no_grad():
        pred = model.forward_l1(z_hist, D) if kind == "hjepa" else model(z_hist)
        persist = persistence_baseline(z_hist, horizon)
        rows = horizon_table(pred, persist, z_fut)
        by_dom = cross_domain_rows(model, kind, Z, D, history, horizon)
    report = {
        "ckpt": args.ckpt, "kind": kind, "split": "val", "n_val": int(Z.size(0)),
        "horizons": rows, "cross_domain": by_dom,
        "paper": "Endo-HJEPA", "not_ablation_planning": True,
    }
    if kind == "hjepa" and blob.get("ablation", "full") == "full":
        report["planning"] = _reach(model, Z, D, history, horizon, args.n_samples)
        from endoworld.data.domains import ID_TO_DOMAIN
        plan_by = {}
        for did in D.unique().tolist():
            m = D == did
            if int(m.sum()) >= 2:
                plan_by[ID_TO_DOMAIN.get(int(did), str(int(did)))] = _reach(
                    model, Z[m], D[m], history, horizon, args.n_samples)
        report["planning_by_domain"] = plan_by

    out = args.out or str(Path(args.ckpt).parent / "eval_ckpt.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
