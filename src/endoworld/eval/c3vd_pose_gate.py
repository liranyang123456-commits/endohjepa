"""C3VD pose-convention gate: reproject depth(t) into frame t+1.

C3VD (Bobrow et al., MICCAI 2023) uses a Scaramuzza omnidirectional camera
(Olympus CF-HQ190L calibration, paper Table 3):
    image 1080x1350, centre (679.54, 543.98),
    cam2world polynomial z(rho) = a0 + a2 rho^2 + a3 rho^3 + a4 rho^4
    a0=769.24, a2=-8.13e-4, a3=-6.26e-7, a4=-1.20e-9, stretch ~ identity.

For every candidate pose convention we back-project depth_t to 3D, apply the
relative pose, project into frame t+1 by solving poly(rho)/rho = Z/sqrt(X^2+Y^2)
(bisection), and measure pixel reprojection + depth consistency. The correct
convention must win by a wide margin on synthetic ground truth.

    python -m endoworld.eval.c3vd_pose_gate --seq datasets/C3VD/cecum_t1_a/cecum_t1_a
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

# Olympus CF-HQ190L Scaramuzza intrinsics (C3VD paper, Table 3).
CX, CY = 679.54, 543.98
A0, A2, A3, A4 = 769.24, -8.13e-4, -6.26e-7, -1.20e-9
H, W = 1080, 1350


def _poly(rho: np.ndarray) -> np.ndarray:
    return A0 + A2 * rho**2 + A3 * rho**3 + A4 * rho**4


def cam2ray(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Pixel -> unit ray in camera coordinates (stretch ~ identity)."""
    x, y = u - CX, v - CY
    rho = np.hypot(x, y)
    z = _poly(rho)
    ray = np.stack([x, y, z], axis=-1)
    return ray / np.linalg.norm(ray, axis=-1, keepdims=True)


def ray2cam(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera-frame points -> pixels via bisection on the Scaramuzza model."""
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    r = np.hypot(x, y)
    front = z > 1e-6
    ratio = np.zeros_like(z)
    ratio[front] = z[front] / np.maximum(r[front], 1e-9)
    lo = np.zeros_like(ratio)
    hi = np.full_like(ratio, 800.0)
    # f(rho) = poly(rho) - rho * ratio ; f(0) = A0 > 0, decreasing.
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        f = _poly(mid) - mid * ratio
        hi = np.where(f < 0, mid, hi)
        lo = np.where(f >= 0, mid, lo)
    rho = 0.5 * (lo + hi)
    phi = np.arctan2(y, x)
    u = CX + rho * np.cos(phi)
    v = CY + rho * np.sin(phi)
    return u, v, front & (rho < 799.0)


def _load_pose_txt(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        vals = [float(x) for x in line.replace(",", " ").split()]
        if len(vals) == 16:
            rows.append(np.array(vals, dtype=np.float64).reshape(4, 4))
    return np.stack(rows)


def _depth(path: Path, scale: float) -> np.ndarray:
    return np.asarray(Image.open(path)).astype(np.float64) / scale


def _candidates(raw: np.ndarray) -> dict[str, tuple[np.ndarray, str]]:
    """Each candidate is (relative_pose, application convention).

    'col': column-vector p' = R p + t (translation in [:3, 3]).
    'row': row-vector p' = p R + t (translation in [3, :3]).
    """
    A, B = raw[0], raw[1]
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    rel_col = np.linalg.inv(A.T) @ B.T          # column-major c2w interpretation
    rel_row = np.linalg.inv(A) @ B              # row-vector c2w interpretation
    rel_flip = np.linalg.inv((flip @ A.T @ flip).T) @ (flip @ B.T @ flip).T
    return {
        "transpose, column-vector (current)": (rel_col, "col"),
        "transpose, column-vector, rotation only": (
            np.block([[rel_col[:3, :3], np.zeros((3, 1))],
                      [np.zeros((1, 3)), np.ones((1, 1))]]), "col"),
        "raw, row-vector": (rel_row, "row"),
        "transpose + GL->CV flip, column-vector": (rel_flip, "col"),
        "inverse transpose, column-vector": (np.linalg.inv(rel_col), "col"),
    }


def _apply(points: np.ndarray, rel: np.ndarray, convention: str) -> np.ndarray:
    if convention == "col":
        return points @ rel[:3, :3].T + rel[:3, 3]
    return points @ rel[:3, :3] + rel[3, :3]


def evaluate_pair(
    depth_t: np.ndarray, depth_t1: np.ndarray,
    pose_t: np.ndarray, pose_t1: np.ndarray,
    n_points: int, rng: np.random.Generator,
) -> dict[str, float]:
    valid = np.argwhere(depth_t > 0)
    if len(valid) > n_points:
        valid = valid[rng.choice(len(valid), n_points, replace=False)]
    u = valid[:, 1].astype(np.float64)
    v = valid[:, 0].astype(np.float64)
    rays = cam2ray(u, v)
    points = rays * depth_t[valid[:, 0], valid[:, 1], None]
    out = {}
    for name, (rel, convention) in _candidates(
            np.stack([pose_t, pose_t1])).items():
        moved = _apply(points, rel, convention)
        pu, pv, ok = ray2cam(moved)
        in_frame = ok & (pu >= 0) & (pu < W) & (pv >= 0) & (pv < H)
        if in_frame.sum() < 10:
            out[name] = {"median_px": float("inf"), "in_frame": float(in_frame.mean()),
                         "median_depth_err": float("inf")}
            continue
        pi, pj = np.clip(pv[in_frame].round().astype(int), 0, H - 1), \
            np.clip(pu[in_frame].round().astype(int), 0, W - 1)
        depth_err = np.abs(depth_t1[pi, pj] - moved[in_frame, 2])
        px_err = np.hypot(pu[in_frame] - u[in_frame], pv[in_frame] - v[in_frame])
        out[name] = {
            "median_px": float(np.median(px_err)),
            "p90_px": float(np.quantile(px_err, 0.9)),
            "in_frame": float(in_frame.mean()),
            "median_depth_err": float(np.median(depth_err)),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="datasets/C3VD/cecum_t1_a/cecum_t1_a")
    parser.add_argument("--pairs", type=int, default=40)
    parser.add_argument("--gap", type=int, default=1,
                        help="frame gap per pair; larger gaps amplify motion")
    parser.add_argument("--points", type=int, default=1500)
    parser.add_argument("--depth-scale", type=float, default=1000.0,
                        help="raw uint16 per millimetre (1000 = micrometres)")
    parser.add_argument("--out", default="docs/endohjepa/c3vd_pose_gate.json")
    args = parser.parse_args()
    seq = Path(args.seq)
    poses = _load_pose_txt(seq / "pose.txt")
    depth_files = sorted(seq.glob("*_depth.tiff"))
    n = min(len(poses), len(depth_files))
    rng = np.random.default_rng(0)
    pair_ids = np.linspace(0, n - 1 - args.gap, min(args.pairs, n - args.gap)).astype(int)
    per_candidate: dict[str, list[dict]] = {}
    for i in pair_ids:
        d0 = _depth(depth_files[i], args.depth_scale)
        d1 = _depth(depth_files[i + args.gap], args.depth_scale)
        result = evaluate_pair(d0, d1, poses[i], poses[i + args.gap], args.points, rng)
        for name, metrics in result.items():
            per_candidate.setdefault(name, []).append(metrics)
    summary = {}
    for name, rows in per_candidate.items():
        summary[name] = {
            "median_px": float(np.median([r["median_px"] for r in rows])),
            "p90_px": float(np.median([r["p90_px"] for r in rows])),
            "in_frame": float(np.mean([r["in_frame"] for r in rows])),
            "median_depth_err": float(np.median([r["median_depth_err"] for r in rows])),
        }
    winner = min(summary, key=lambda k: summary[k]["median_px"])
    report = {
        "sequence": str(seq),
        "n_pairs": len(pair_ids),
        "gap": args.gap,
        "depth_scale": args.depth_scale,
        "intrinsics": "Scaramuzza, Olympus CF-HQ190L (C3VD paper Table 3)",
        "candidates": summary,
        "winner": winner,
        "gate_passed": bool(summary[winner]["median_px"] < 5.0),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
