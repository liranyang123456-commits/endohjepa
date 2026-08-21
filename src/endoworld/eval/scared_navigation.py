"""SCARED goal-directed navigation: a *physical* downstream task for the world model.

Task: given a start frame and a goal anatomical viewpoint (a later frame), plan a
latent-action sequence with H-JEPA MPC and roll it out. Success is measured
*physically*: decode the predicted terminal latent to the nearest real frame and
measure the camera-pose distance to the goal pose (from SCARED frame_data), plus the
latent reach rate vs persistence and the energy along the planned path.

This grounds planning in a real, measurable navigation task rather than latent
self-evaluation. In-silico only; not a clinical-navigation claim.

    python -m endoworld.eval.scared_navigation --ckpt outputs/p2000_full_causal/endohjepa.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from endoworld.world.scared_actions import find_scared_keyframes, find_scared_rgb, load_scared_poses


def _encode_frames(enc, frames, device, image_size):
    from PIL import Image
    zs = []
    for fp in frames:
        try:
            im = Image.open(fp).convert("RGB").resize((image_size, image_size))
            zs.append(np.asarray(im, np.float32) / 255.0)
        except Exception:
            continue
    if len(zs) < 4:
        return None
    # encode as one clip (pad/subsample to a multiple of tubelet for dense temporal)
    arr = np.stack(zs).transpose(0, 3, 1, 2)
    clip = torch.from_numpy(arr).unsqueeze(0).to(device).float()
    with torch.no_grad():
        z = enc.encode_temporal(clip)[0].cpu()  # (T', D)
    return z


def _pose_trans(poses):
    return poses[:, :3, 3]  # (N, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/p2000_full_causal/endohjepa.pt")
    ap.add_argument("--scared", default="datasets/SCARED")
    ap.add_argument("--vjepa2-id", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--max-kf", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--history", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--out", default="outputs/p2000_full_causal/scared_navigation.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from endoworld.understanding.vjepa2_hf import VJEPA2Encoder
    from endoworld.eval.world_benchmark import load_predictor
    enc = VJEPA2Encoder(args.vjepa2_id, device=device)
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    model, kind, history, horizon, _ = load_predictor(blob, device)
    if kind != "hjepa" or blob.get("ablation", "full") != "full":
        raise SystemExit("need full H-JEPA ckpt (L3 + energy for planning)")
    history = min(args.history, history)

    rows = []
    for kf in find_scared_keyframes(args.scared)[: args.max_kf]:
        try:
            poses = load_scared_poses(kf)
            video, frames = find_scared_rgb(kf)
            if not frames:
                continue
            frames = frames[: args.max_frames]
            z = _encode_frames(enc, frames, device, enc.image_size)
            if z is None or z.size(0) < history + 4:
                continue
            z = z.to(device)
            trans = _pose_trans(poses)
            # map encoder temporal index -> pose index (tubelet alignment)
            n_z = z.size(0)
            pidx = np.clip(np.round(np.linspace(0, len(trans) - 1, n_z)).astype(int), 0, len(trans) - 1)
            trans_t = torch.from_numpy(trans[pidx]).float().to(device)

            dom = torch.zeros(1, dtype=torch.long, device=device)  # laparo
            # goal = a later frame; start history
            for goal in range(history + horizon, n_z, max(1, (n_z - history - horizon) // 6)):
                z_hist = z[goal - history - horizon + 1: goal - horizon + 1].unsqueeze(0)
                if z_hist.size(1) < history:
                    continue
                z_hist = z_hist[:, -history:]
                z_goal = z[goal].unsqueeze(0)
                # MPC plan toward the goal latent
                plan_a, plan_e = model.plan(z_hist, z_goal, dom, n_samples=args.n_samples, steps=horizon)
                with torch.no_grad():
                    pred = model.forward_l3(z_hist, plan_a[:, :model.cfg.horizon], dom)[0, -1]
                # decode predicted latent -> nearest real frame (by latent distance)
                dist = torch.cdist(pred.unsqueeze(0), z).squeeze(0)
                j_star = int(dist.argmin().item())
                persist_lat = z_hist[0, -1]
                # metrics
                latent_gain = (torch.dist(persist_lat, z_goal) - torch.dist(pred, z_goal)).item()
                pose_err_model = float(torch.dist(trans_t[j_star], trans_t[goal]).item())
                j_persist = int(torch.cdist(persist_lat.unsqueeze(0), z).squeeze(0).argmin().item())
                pose_err_persist = float(torch.dist(trans_t[j_persist], trans_t[goal]).item())
                rows.append({
                    "keyframe": kf.name, "goal_idx": int(goal),
                    "latent_gain_vs_persist": latent_gain,
                    "reach_latent_success": bool(latent_gain > 0),
                    "pose_err_model_mm": pose_err_model,
                    "pose_err_persist_mm": pose_err_persist,
                    "plan_energy": float(plan_e.mean().item()),
                })
        except Exception as e:
            rows.append({"keyframe": str(kf), "error": str(e)})

    ok = [r for r in rows if "pose_err_model_mm" in r]
    if not ok:
        print("[nav] no successful runs"); return
    report = {
        "paper": "Endo-HJEPA", "not_ablation_planning": True,
        "task": "SCARED goal-directed navigation (physical pose success)",
        "n_trials": len(ok),
        "reach_latent_success_rate": float(np.mean([r["reach_latent_success"] for r in ok])),
        "pose_err_model_mm_mean": float(np.mean([r["pose_err_model_mm"] for r in ok])),
        "pose_err_persist_mm_mean": float(np.mean([r["pose_err_persist_mm"] for r in ok])),
        "pose_err_reduction": float(np.mean([r["pose_err_persist_mm"] for r in ok])
                                    - np.mean([r["pose_err_model_mm"] for r in ok])),
        "in_silico_only": True,
        "rows": ok,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
