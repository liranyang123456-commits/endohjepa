"""C3VD pose-convention depth-warp diagnostic.

C3VD (Bobrow et al., MICCAI 2023) uses a Scaramuzza omnidirectional camera
(Olympus CF-HQ190L calibration, paper Table 3):
    image 1080x1350, centre (679.54, 543.98),
    cam2world polynomial z(rho) = a0 + a2 rho^2 + a3 rho^3 + a4 rho^4
    a0=769.24, a2=-8.13e-4, a3=-6.26e-7, a4=-1.20e-9, stretch ~ identity.

For every candidate pose convention we back-project depth_t to 3D, apply the
source-to-target camera transform, project into frame t+1 by solving
poly(rho)/rho = Z/sqrt(X^2+Y^2) (bisection), and measure target-depth
consistency. It reports a diagnostic for the convention used by the action
loader; it does not select a pose convention or validate translation without
independent cross-frame correspondences.

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


def backproject_z_depth(rays: np.ndarray, z_depth: np.ndarray) -> np.ndarray:
    """Back-project optical-axis (Z) depth along unit camera rays."""
    scale = np.asarray(z_depth)[..., None] / np.clip(rays[..., 2:3], 1e-9, None)
    return rays * scale


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


def _candidates(raw: np.ndarray) -> dict[str, np.ndarray]:
    """Source-to-target column transforms for diagnostic candidates.

    The action loader uses a row-major pose matrix transposed to column form,
    followed by an OpenGL-to-OpenCV axis flip. For a camera-to-world pose
    ``P``, a source-camera point reaches the target camera as
    ``inv(P_target) @ P_source``. This is the transform required to warp a
    source depth map into the target frame.
    """
    A, B = raw[0], raw[1]
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    transpose = np.stack([A.T, B.T])
    loader = flip @ transpose @ flip

    def source_to_target(poses: np.ndarray) -> np.ndarray:
        return np.linalg.inv(poses[1]) @ poses[0]

    loader_rotation_only = loader.copy()
    loader_rotation_only[:, :3, 3] = 0.0
    return {
        "transpose-only": source_to_target(transpose),
        "loader transpose + GL->CV flip": source_to_target(loader),
        "loader rotation-only diagnostic": source_to_target(loader_rotation_only),
    }


def _apply(points: np.ndarray, rel: np.ndarray) -> np.ndarray:
    """Apply a column-vector rigid transform to row-major point samples."""
    return points @ rel[:3, :3].T + rel[:3, 3]


def evaluate_pair(
    depth_t: np.ndarray,
    depth_t1: np.ndarray,
    pose_t: np.ndarray,
    pose_t1: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    valid = np.argwhere(depth_t > 0)
    if len(valid) > n_points:
        valid = valid[rng.choice(len(valid), n_points, replace=False)]
    u = valid[:, 1].astype(np.float64)
    v = valid[:, 0].astype(np.float64)
    rays = cam2ray(u, v)
    points = backproject_z_depth(rays, depth_t[valid[:, 0], valid[:, 1]])
    out = {}
    for name, rel in _candidates(np.stack([pose_t, pose_t1])).items():
        moved = _apply(points, rel)
        pu, pv, ok = ray2cam(moved)
        in_frame = ok & (pu >= 0) & (pu < W) & (pv >= 0) & (pv < H)
        if in_frame.sum() < 10:
            out[name] = {
                "median_depth_err": float("inf"),
                "median_relative_depth_err": float("inf"),
                "median_flow_px": float("inf"),
                "in_frame": float(in_frame.mean()),
            }
            continue
        pi, pj = (
            np.clip(pv[in_frame].round().astype(int), 0, H - 1),
            np.clip(pu[in_frame].round().astype(int), 0, W - 1),
        )
        target_depth = depth_t1[pi, pj]
        valid_target = target_depth > 0
        if valid_target.sum() < 10:
            out[name] = {
                "median_depth_err": float("inf"),
                "median_relative_depth_err": float("inf"),
                "median_flow_px": float("inf"),
                "in_frame": float(in_frame.mean()),
            }
            continue
        depth_err = np.abs(
            target_depth[valid_target] - moved[in_frame, 2][valid_target]
        )
        # This is the image displacement induced by the candidate, not a
        # correspondence reprojection error; keep it descriptive only.
        flow_px = np.hypot(
            pu[in_frame][valid_target] - u[in_frame][valid_target],
            pv[in_frame][valid_target] - v[in_frame][valid_target],
        )
        out[name] = {
            "median_depth_err": float(np.median(depth_err)),
            "median_relative_depth_err": float(
                np.median(depth_err / np.maximum(target_depth[valid_target], 1e-6))
            ),
            "median_flow_px": float(np.median(flow_px)),
            "in_frame": float(in_frame.mean()),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="datasets/C3VD/cecum_t1_a/cecum_t1_a")
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument(
        "--gap",
        type=int,
        default=5,
        help="frame gap per pair; larger gaps amplify motion",
    )
    parser.add_argument("--points", type=int, default=1500)
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=655.35,
        help="raw uint16 per millimetre (0--100 mm over uint16)",
    )
    parser.add_argument("--out", default="docs/endohjepa/c3vd_pose_gate.json")
    args = parser.parse_args()
    seq = Path(args.seq)
    poses = _load_pose_txt(seq / "pose.txt")
    depth_files = sorted(seq.glob("*_depth.tiff"))
    n = min(len(poses), len(depth_files))
    rng = np.random.default_rng(0)
    pair_ids = np.linspace(0, n - 1 - args.gap, min(args.pairs, n - args.gap)).astype(
        int
    )
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
            "median_depth_err": float(np.median([r["median_depth_err"] for r in rows])),
            "median_relative_depth_err": float(
                np.median([r["median_relative_depth_err"] for r in rows])
            ),
            "median_flow_px": float(np.median([r["median_flow_px"] for r in rows])),
            "in_frame": float(np.mean([r["in_frame"] for r in rows])),
        }
    report = {
        "sequence": str(seq),
        "n_pairs": len(pair_ids),
        "gap": args.gap,
        "depth_scale": args.depth_scale,
        "intrinsics": "Scaramuzza, Olympus CF-HQ190L (C3VD paper Table 3)",
        "candidates": summary,
        "implementation_candidate": "loader transpose + GL->CV flip",
        "interpretation": (
            "Depth-warp diagnostic only. It does not select a convention or "
            "independently validate translation without cross-frame correspondences."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
