"""SCARED wall-proximity vs world-model energy: a *physical* proxy for the energy head.

For each frame we compute the camera's proximity to the nearest tissue surface from
the per-frame scene_points depth map, then test whether the H-JEPA energy head scores
transitions into near-wall (low-depth) states with higher energy. A positive
correlation / AUC > 0.5 grounds the energy prior in a physical signal (not latent
self-evaluation).

    python -m endoworld.eval.scared_collision --ckpt outputs/p2000_full_causal/endohjepa.pt
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import numpy as np
import torch

from endoworld.world.scared_actions import find_scared_keyframes, find_scared_rgb


def _depth_proximities(keyframe_dir: Path, pct: float = 5.0) -> dict[int, float]:
    """Read the scene_points tar ONCE and return frame_idx -> low-percentile depth (mm)."""
    import cv2

    tar = keyframe_dir / "data" / "scene_points.tar.gz"
    out: dict[int, float] = {}
    if not tar.is_file():
        return out
    with tarfile.open(tar, "r:gz") as t:
        for m in t.getmembers():
            if not m.name.endswith(".tiff"):
                continue
            idx = int("".join(ch for ch in m.name if ch.isdigit()) or -1)
            fh = t.extractfile(m)
            if fh is None:
                continue
            try:
                img = cv2.imdecode(
                    np.frombuffer(fh.read(), np.uint8), cv2.IMREAD_UNCHANGED
                )
            except Exception:
                continue
            if img is None or img.ndim != 3:
                continue
            z = img[..., 0].astype(np.float32)
            z = z[np.isfinite(z) & (z > 1.0)]
            if len(z) >= 100:
                out[idx] = float(np.percentile(z, pct))
    return out


def _load_rgb_frame(path):
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/p2000_full_causal/endohjepa.pt")
    ap.add_argument("--scared", default="datasets/SCARED")
    ap.add_argument("--vjepa2-id", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--max-kf", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument(
        "--near-wall-mm",
        type=float,
        default=20.0,
        help="depth percentile threshold below which a frame is 'near wall'",
    )
    ap.add_argument("--out", default="outputs/p2000_full_causal/scared_collision.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from endoworld.understanding.vjepa2_hf import VJEPA2Encoder
    from endoworld.eval.world_benchmark import load_predictor

    enc = VJEPA2Encoder(args.vjepa2_id, device=device)
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    model, kind, _, _, _ = load_predictor(blob, device)
    if kind != "hjepa" or blob.get("ablation", "full") != "full":
        raise SystemExit("need a full H-JEPA ckpt (energy head)")

    energies, proximities = [], []
    kf_used = 0
    for kf in find_scared_keyframes(args.scared)[: args.max_kf]:
        video, frames = find_scared_rgb(kf)
        if not frames:
            continue
        frames = frames[: args.max_frames]
        proxmap = _depth_proximities(Path(kf))
        if len(proxmap) < 9:
            continue
        imgs = []
        prox = []
        for i, fp in enumerate(frames):
            im = _load_rgb_frame(fp)
            p = proxmap.get(i)
            if im is None or p is None:
                continue
            from PIL import Image

            imgs.append(
                np.asarray(
                    Image.fromarray(im).resize((enc.image_size, enc.image_size)),
                    np.float32,
                )
                / 255.0
            )
            prox.append(p)
        if len(imgs) < 9:
            continue
        # encode in sliding windows to get per-frame latents
        zs = []
        clip_len, stride = 8, 1
        for s in range(0, len(imgs) - clip_len, stride):
            clip = np.stack(imgs[s : s + clip_len]).transpose(0, 3, 1, 2)
            zt = enc.encode_temporal(
                torch.from_numpy(clip).unsqueeze(0).to(device).float()
            )[0]
            zs.append(zt[-1].cpu().numpy())  # last-token latent ~ current frame
        if len(zs) < 3:
            continue
        z = torch.from_numpy(np.stack(zs)).to(device).float()  # (M, D)
        with torch.no_grad():
            ids, _, _ = model.actions(z[:-1], z[1:])
            e = (
                model.energy(z[:-1], ids, z[1:]).cpu().numpy()
            )  # energy of each transition
        # align: energy[i] predicts transition into frame (i+1); proximity of frame i+1
        p_next = np.array(
            prox[clip_len : clip_len + len(e)]
        )  # proximity of the "next" frame
        m = min(len(e), len(p_next))
        energies.extend(e[:m].tolist())
        proximities.extend(p_next[:m].tolist())
        kf_used += 1
        print(f"[collision] {kf.name}: {m} transitions")

    if len(energies) < 20:
        print("[collision] too few transitions")
        return
    e = np.array(energies)
    p = np.array(proximities)
    # data-driven near-wall threshold: bottom quartile of observed depths
    thr = args.near_wall_mm
    if (p < thr).sum() < 5:
        thr = float(np.percentile(p, 25))
        print(f"[collision] data-driven near-wall threshold = {thr:.1f} mm (25th pct)")
    near_wall = (p < thr).astype(int)
    # AUC: energy as a ranking score for near-wall transitions (Mann-Whitney)
    from scipy.stats import spearmanr, rankdata

    r = rankdata(e)
    n1 = int(near_wall.sum())
    n0 = len(near_wall) - n1
    auc = (
        float((r[near_wall == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))
        if n1 and n0
        else float("nan")
    )
    sp = spearmanr(
        e, p
    ).correlation  # expect negative: high energy ~ low depth (near wall)
    report = {
        "paper": "Endo-HJEPA",
        "not_ablation_planning": True,
        "task": "SCARED wall-proximity energy proxy",
        "n_transitions": int(len(e)),
        "n_keyframes": kf_used,
        "near_wall_threshold_mm": float(thr),
        "near_wall_frac": float(near_wall.mean()),
        "depth_mm_median": float(np.median(p)),
        "energy_nearwall_auc": auc,
        "spearman_energy_vs_depth": float(sp),
        "interp": "AUC>0.5 and negative Spearman = energy head flags near-wall (collision-risk) transitions",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
