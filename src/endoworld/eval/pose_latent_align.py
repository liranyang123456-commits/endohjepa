"""Align latent-action residuals with SCARED / C3VD SE(3) deltas.

Uses the same encoder as domain adapt (default: official V-JEPA2). Pose index
follows frame_data{i:06d} / pose.txt row i; extracted 2 fps frames are mapped
linearly onto that index.

    python -m endoworld.eval.pose_latent_align --encoder vjepa2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _encode_seq(enc, clip: torch.Tensor, device: str) -> np.ndarray:
    """Pooled per-timestep latent sequence (T, D)."""
    clip = clip.unsqueeze(0).to(device).float()
    with torch.no_grad():
        z = enc.encode_temporal(clip)[0].cpu().numpy()
    return z


def _encode_residuals(enc, clip: torch.Tensor, device: str) -> np.ndarray:
    z = _encode_seq(enc, clip, device)
    if len(z) < 2:
        return np.zeros((0, z.shape[-1] if z.ndim == 2 else 1), dtype=np.float32)
    return z[1:] - z[:-1]


def _codebook_ids(model, z_seq: np.ndarray, device: str) -> np.ndarray:
    """Assign trained VQ action ids to consecutive latent pairs (T-1,)."""
    if len(z_seq) < 2:
        return np.zeros(0, dtype=np.int64)
    z = torch.from_numpy(z_seq).to(device).float()
    with torch.no_grad():
        idx, _, _ = model.actions(z[:-1], z[1:])
    return idx.cpu().numpy().astype(np.int64)


def _nmi_pack(
    residuals: np.ndarray,
    deltas: np.ndarray,
    n_actions: int,
    lat_ids_override: np.ndarray | None = None,
) -> dict:
    from endoworld.world.pose_align import (
        action_pose_nmi,
        quantise_deltas,
        residual_delta_probe,
    )

    n = min(len(residuals), len(deltas))
    if lat_ids_override is not None:
        n = min(n, len(lat_ids_override))
    # k must be << n; k≈n makes every point its own cluster and NMI collapses to 1
    k = min(n_actions, max(2, n // 4))
    if n < 2 * k:
        return {
            "n": int(n),
            "k": int(k),
            "nmi_latent_pose": float("nan"),
            "note": "too few residuals for NMI; increase --frames",
        }
    pose_ids = quantise_deltas(deltas[:n], k)
    if lat_ids_override is not None:
        lat_ids = lat_ids_override[:n]
        k = int(max(lat_ids.max() + 1, 2)) if len(lat_ids) else k
        pose_ids = quantise_deltas(deltas[:n], k)
    else:
        lat_ids = quantise_deltas(residuals[:n], k)
    rng = np.random.default_rng(0)
    rand = rng.integers(0, k, size=n)
    return {
        "n": int(n),
        "k": int(k),
        "nmi_latent_pose": action_pose_nmi(lat_ids, pose_ids),
        "nmi_random": action_pose_nmi(rand, pose_ids),
        "probe": residual_delta_probe(residuals[:n], deltas[:n]),
        "source": "trained_codebook"
        if lat_ids_override is not None
        else "kmeans_residual",
    }


def run_scared(
    enc,
    device: str,
    root: str,
    n_actions: int,
    max_kf: int,
    frames_per: int,
    image_size: int,
    model=None,
) -> dict:
    from endoworld.world.c3vd_actions import pose_deltas
    from endoworld.world.scared_actions import (
        find_scared_keyframes,
        find_scared_rgb,
        load_scared_poses,
        pose_index_for_frames,
        read_video_frames,
        sample_video_indices,
    )

    rows = []
    for kf in find_scared_keyframes(root)[:max_kf]:
        try:
            poses = load_scared_poses(kf)
            video, frames = find_scared_rgb(kf)
            if video is not None:
                idx_all, total = sample_video_indices(
                    video, min(len(poses), max(frames_per * 4, 64))
                )
                if len(idx_all) < 8:
                    continue
                res_list, d_list, cb_list = [], [], []
                win = min(frames_per, len(idx_all))
                n_win = max(1, (len(idx_all) - win) // max(win // 2, 1) + 1)
                n_win = min(n_win, 8)
                starts = np.linspace(0, max(len(idx_all) - win, 0), n_win).astype(int)
                for s in starts:
                    idx = idx_all[s : s + win]
                    if len(idx) < 4:
                        continue
                    clip = read_video_frames(video, idx, image_size)
                    z = _encode_seq(enc, clip, device)
                    res = z[1:] - z[:-1] if len(z) >= 2 else z[:0]
                    res_list.append(res)
                    if model is not None:
                        cb_list.append(_codebook_ids(model, z, device))
                    pidx = np.clip(idx, 0, len(poses) - 1)
                    d = pose_deltas(poses[pidx])
                    if len(d) > len(res) and len(res) > 0:
                        take = np.round(np.linspace(0, len(d) - 1, len(res))).astype(
                            int
                        )
                        d = d[take]
                    d_list.append(d)
                if not res_list:
                    continue
                res = np.concatenate(res_list, 0)
                d = np.concatenate(d_list, 0)
                cb = np.concatenate(cb_list, 0) if cb_list else None
                pack = _nmi_pack(res, d, n_actions, lat_ids_override=cb)
                pack.update(
                    {
                        "keyframe": str(kf),
                        "n_poses": int(len(poses)),
                        "n_rgb": int(total),
                    }
                )
                rows.append(pack)
                print(
                    f"[scared] {kf.name} n={pack.get('n')} nmi={pack.get('nmi_latent_pose')} src={pack.get('source')}"
                )
                continue
            elif len(frames) >= 4:
                pick = np.round(
                    np.linspace(0, len(frames) - 1, min(frames_per, len(frames)))
                ).astype(int)
                from PIL import Image

                arr = []
                for i in pick:
                    im = (
                        Image.open(frames[i])
                        .convert("RGB")
                        .resize((image_size, image_size))
                    )
                    arr.append(np.asarray(im, np.float32) / 255.0)
                clip = torch.from_numpy(np.stack(arr).transpose(0, 3, 1, 2))
                pidx = pose_index_for_frames(len(pick), len(poses))
                total = len(frames)
            else:
                rows.append({"keyframe": str(kf), "error": "no rgb"})
                continue
            res = _encode_residuals(enc, clip, device)
            # pose delta between consecutive sampled pose indices
            sampled = poses[pidx]
            d = pose_deltas(sampled)
            # tubelet-2 encoder halves time; subsample deltas to residual length
            if len(d) > len(res) and len(res) > 0:
                take = np.round(np.linspace(0, len(d) - 1, len(res))).astype(int)
                d = d[take]
            pack = _nmi_pack(res, d, n_actions)
            pack.update(
                {"keyframe": str(kf), "n_poses": int(len(poses)), "n_rgb": int(total)}
            )
            rows.append(pack)
            print(
                f"[scared] {kf.name} n={pack.get('n')} nmi={pack.get('nmi_latent_pose')}"
            )
        except Exception as e:
            rows.append({"keyframe": str(kf), "error": str(e)})
    return {"n_keyframes": len(rows), "rows": rows}


def run_c3vd(
    enc,
    device: str,
    root: str,
    n_actions: int,
    frames_per: int,
    image_size: int,
    model=None,
) -> dict:
    from endoworld.world.c3vd_actions import (
        find_c3vd_color_frames,
        find_c3vd_pose_files,
        load_pose_txt,
        pose_deltas,
    )
    from PIL import Image

    rows = []
    for pose_path in find_c3vd_pose_files(root):
        seq = pose_path.parent
        poses = load_pose_txt(pose_path)
        color = find_c3vd_color_frames(seq)
        if len(color) < 4:
            rows.append(
                {"pose_file": str(pose_path), "error": f"no color frames in {seq}"}
            )
            continue
        res_list, d_list, cb_list = [], [], []
        win = min(frames_per, len(color))
        n_win = min(8, max(1, (len(color) - win) // max(win // 2, 1) + 1))
        starts = np.linspace(0, max(len(color) - win, 0), n_win).astype(int)
        for s in starts:
            pick = np.arange(s, s + win)
            arr = []
            for i in pick:
                im = (
                    Image.open(color[int(i)])
                    .convert("RGB")
                    .resize((image_size, image_size))
                )
                arr.append(np.asarray(im, np.float32) / 255.0)
            clip = torch.from_numpy(np.stack(arr).transpose(0, 3, 1, 2))
            z = _encode_seq(enc, clip, device)
            res = z[1:] - z[:-1] if len(z) >= 2 else z[:0]
            if model is not None:
                cb_list.append(_codebook_ids(model, z, device))
            pidx = np.clip(pick, 0, len(poses) - 1)
            d = pose_deltas(poses[pidx])
            if len(d) > len(res) and len(res) > 0:
                take = np.round(np.linspace(0, len(d) - 1, len(res))).astype(int)
                d = d[take]
            res_list.append(res)
            d_list.append(d)
        res = np.concatenate(res_list, 0)
        d = np.concatenate(d_list, 0)
        cb = np.concatenate(cb_list, 0) if cb_list else None
        pack = _nmi_pack(res, d, n_actions, lat_ids_override=cb)
        pack.update({"pose_file": str(pose_path), "n_color": int(len(color))})
        rows.append(pack)
        print(
            f"[c3vd] {seq.name} n={pack.get('n')} nmi={pack.get('nmi_latent_pose')} src={pack.get('source')}"
        )
    return {"n_files": len(rows), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["vjepa2", "scratch"], default="vjepa2")
    ap.add_argument("--vjepa2-id", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--scratch-ckpt", default="outputs/vjepa_l1/vjepa_l1_adapt.pt")
    ap.add_argument("--scared", default="datasets/SCARED")
    ap.add_argument("--c3vd", default="datasets/C3VD")
    ap.add_argument("--max-kf", type=int, default=8)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--n-actions", type=int, default=16)
    ap.add_argument(
        "--ckpt",
        default="",
        help="trained H-JEPA ckpt; use its VQ action codebook for latent actions",
    )
    ap.add_argument("--out", default="outputs/endohjepa_vjepa2/pose_latent_align.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from endoworld.understanding.encoders import load_any_encoder

    enc, _, image_size, _ = load_any_encoder(
        args.encoder, device, args.vjepa2_id, args.scratch_ckpt
    )
    model = None
    if args.ckpt and Path(args.ckpt).is_file():
        from endoworld.eval.world_benchmark import load_predictor

        blob = torch.load(args.ckpt, map_location=device, weights_only=False)
        model, _, _, _, _ = load_predictor(blob, device)
        print(f"[pose-align] using trained action codebook <- {args.ckpt}")
    report = {
        "paper": "Endo-HJEPA",
        "not_ct_ablation_planning": True,
        "encoder": args.encoder,
        "action_source": "trained_codebook" if model is not None else "kmeans_residual",
        "main_table_ok": args.encoder == "vjepa2",
        "scared": run_scared(
            enc,
            device,
            args.scared,
            args.n_actions,
            args.max_kf,
            args.frames,
            image_size,
            model=model,
        ),
        "c3vd": run_c3vd(
            enc, device, args.c3vd, args.n_actions, args.frames, image_size, model=model
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[pose-align] wrote {args.out}")


if __name__ == "__main__":
    main()
