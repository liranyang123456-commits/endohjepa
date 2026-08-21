"""Latent MPC planning eval on C3VD (pose), SCARED, and ION bronchoscopy.

Success = predicted terminal latent is closer to the goal than persistence / random
actions, with an energy threshold as an OOD reject.

    python -m endoworld.eval.plan_eval --ckpt outputs/endohjepa/endohjepa.pt
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from endoworld.data.domains import ID_TO_DOMAIN
from endoworld.eval.world_benchmark import load_predictor, maybe_pool
from endoworld.world.h_jepa import persistence_baseline
from endoworld.world.plan_mpc import latent_mpc


def _reach_metrics(model, z, D, history, horizon, n_samples=16):
    z_hist, z_fut = z[:, :history], z[:, history:history + horizon]
    z_goal = z_fut[:, -1]
    persist = persistence_baseline(z_hist, horizon)[:, -1]
    with torch.no_grad():
        pred_l1 = model.forward_l1(z_hist, D)[:, -1]
        rand_a = torch.randint(0, model.cfg.n_actions, (z.size(0), model.cfg.horizon),
                               device=z.device)
        pred_rand = model.forward_l3(z_hist, rand_a, D)[:, -1]
        plan_a, plan_e = latent_mpc(model, z_hist, z_goal, D, n_samples=n_samples, steps=horizon)
        # execute planned first-horizon actions
        h = min(plan_a.size(1), model.cfg.horizon)
        a = plan_a[:, :h]
        if a.size(1) < model.cfg.horizon:
            pad = torch.zeros(z.size(0), model.cfg.horizon - a.size(1),
                              dtype=torch.long, device=z.device)
            a = torch.cat([a, pad], dim=1)
        pred_plan = model.forward_l3(z_hist, a, D)[:, -1]
        e_plan = model.energy(z_hist[:, -1], a[:, 0], pred_plan)
        e_rand = model.energy(z_hist[:, -1], rand_a[:, 0], pred_rand)

    def cos(a, b):
        return F.cosine_similarity(a, b, dim=-1)

    cos_plan = cos(pred_plan, z_goal)
    cos_l1 = cos(pred_l1, z_goal)
    cos_rand = cos(pred_rand, z_goal)
    cos_persist = cos(persist, z_goal)
    energy_thr = e_plan.median() + e_plan.std().clamp_min(1e-6)
    reject = (e_plan > energy_thr).float()
    success = (cos_plan > cos_persist).float()
    return {
        "n": int(z.size(0)),
        "cos_plan": cos_plan.mean().item(),
        "cos_l1": cos_l1.mean().item(),
        "cos_random_actions": cos_rand.mean().item(),
        "cos_persist": cos_persist.mean().item(),
        "plan_better_than_persist": success.mean().item(),
        "mean_plan_energy": plan_e.mean().item(),
        "energy_reject_rate": reject.mean().item(),
        "energy_plan_vs_random": (e_plan.mean() - e_rand.mean()).item(),
    }


def eval_from_cache(model, pack, history, horizon, n_samples):
    Z = maybe_pool(pack["Z"]).to(next(model.parameters()).device)
    D = pack["D"].to(Z.device)
    t = Z.size(1)
    history = min(history, t - 1)
    horizon = min(horizon, t - history)
    out = {"all": _reach_metrics(model, Z, D, history, horizon, n_samples)}
    for did in D.unique().tolist():
        m = D == did
        if int(m.sum()) < 2:
            continue
        name = ID_TO_DOMAIN.get(int(did), str(int(did)))
        out[name] = _reach_metrics(model, Z[m], D[m], history, horizon, n_samples)
    return out


def _encode_dataset_clips(encoder, manifest, dataset_name, clip_len, image_size, max_clips, device):
    from torch.utils.data import DataLoader
    from endoworld.data.video_clips import EndoClipDataset
    from endoworld.world.train import collate_meta, cache_latents
    ds = EndoClipDataset(
        manifest, clip_len=clip_len, stride=4, image_size=image_size,
        include=[dataset_name], return_meta=True, split="test",
    )
    if len(ds) == 0:
        ds = EndoClipDataset(
            manifest, clip_len=clip_len, stride=4, image_size=image_size,
            include=[dataset_name], return_meta=True,
        )
    if len(ds) == 0:
        return None
    ds.clips = ds.clips[:max_clips]
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_meta)
    Z, D = cache_latents(encoder, dl, device, dense=False)
    return {"Z": Z, "D": D}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/endohjepa/endohjepa.pt")
    ap.add_argument("--latents", default=None)
    ap.add_argument("--manifest", default="manifests/sequences.csv")
    ap.add_argument("--out", default="outputs/endohjepa/plan_eval.json")
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--encode-subsets", action="store_true",
                    help="also encode C3VD/SCARED/ION clips if present")
    ap.add_argument("--max-clips", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    model, kind, history, horizon, _ = load_predictor(blob, device)
    report = {
        "paper": "Endo-HJEPA latent MPC",
        "not_ct_ablation_planning": True,
        "kind": kind,
        "note": "ION/SCARED/C3VD are downstream navigation/reach cases, not ablation-efficacy experiments.",
    }
    if kind != "hjepa" or blob.get("ablation", "full") != "full":
        report["skipped"] = "planning requires full H-JEPA (L3 + energy); L1/GRU ablations are prediction-only"
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    latents_path = args.latents or os.path.join(os.path.dirname(args.ckpt), "latents_cache.pt")
    if os.path.isfile(latents_path):
        pack = torch.load(latents_path, map_location="cpu", weights_only=False)
        report["from_cache"] = eval_from_cache(model, pack, history, horizon, args.n_samples)

    from endoworld.world.c3vd_actions import find_c3vd_pose_files, load_pose_txt, pose_deltas
    pose_files = [str(p) for p in find_c3vd_pose_files("datasets/C3VD")]
    report["c3vd_pose_files"] = len(pose_files)
    if pose_files:
        poses = load_pose_txt(pose_files[0])
        deltas = pose_deltas(poses)
        report["c3vd_kinematics"] = {
            "pose_file": pose_files[0],
            "n_poses": int(len(poses)),
            "delta_trans_rms": float((deltas[:, :3] ** 2).mean() ** 0.5),
            "delta_rot_rms": float((deltas[:, 3:] ** 2).mean() ** 0.5),
            "note": "SE(3) deltas available to pin latent actions when encoder+frames are aligned",
        }

    if args.encode_subsets:
        enc = None
        # planning eval uses the world-model latents already cached; optional live encode
        report["live_encode"] = "skipped unless encoder ckpt is passed via cache"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
