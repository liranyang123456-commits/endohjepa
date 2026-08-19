"""Component A - 3D reconstruction into a renderable scene.

Pipeline (per sequence):
  RGB(+depth or stereo) frames --> per-frame point clouds (unproject with intrinsics)
                               --> fuse / init 3D Gaussians
                               --> optimise a 3D Gaussian Splatting scene (photometric)
                               --> render novel views (the "simulated world")

Datasets with usable geometry: SCARED (depth GT), EndoNeRF (depth+mask), Stereo_Lap
(stereo -> disparity -> depth). Heavy deps (open3d, gsplat/nerfstudio) are optional
and imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def depth_to_points(depth, intr: CameraIntrinsics, rgb=None, mask=None):
    """Unproject a depth map to a colored 3D point cloud (numpy in, numpy out)."""
    import numpy as np

    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w]
    z = depth.astype(np.float32)
    x = (xs - intr.cx) * z / intr.fx
    y = (ys - intr.cy) * z / intr.fy
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    cols = rgb.reshape(-1, 3) if rgb is not None else None
    valid = z.reshape(-1) > 0
    if mask is not None:
        valid &= mask.reshape(-1) > 0
    pts = pts[valid]
    cols = cols[valid] if cols is not None else None
    return pts, cols


def write_ply(path: str, points, colors=None) -> None:
    """Write a colored point cloud to a binary PLY file."""
    import numpy as np

    pts = np.asarray(points, dtype=np.float32)
    n = pts.shape[0]
    has_col = colors is not None
    if has_col:
        cols = np.clip(np.asarray(colors), 0, 255).astype(np.uint8)
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_col:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += ["end_header\n"]
    import os as _os

    _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(header)).encode("ascii"))
        if has_col:
            dt = np.dtype(
                [
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("r", "u1"),
                    ("g", "u1"),
                    ("b", "u1"),
                ]
            )
            rec = np.empty(n, dtype=dt)
            rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
            rec["r"], rec["g"], rec["b"] = cols[:, 0], cols[:, 1], cols[:, 2]
        else:
            dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
            rec = np.empty(n, dtype=dt)
            rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        f.write(rec.tobytes())


def read_scared_intrinsics(calib_yaml: str) -> CameraIntrinsics:
    import cv2

    fs = cv2.FileStorage(calib_yaml, cv2.FILE_STORAGE_READ)
    m1 = fs.getNode("M1").mat()
    fs.release()
    return CameraIntrinsics(
        fx=float(m1[0, 0]), fy=float(m1[1, 1]), cx=float(m1[0, 2]), cy=float(m1[1, 2])
    )


def load_scared_keyframe(
    keyframe_dir: str, min_depth_mm: float = 5.0, max_depth_mm: float = 300.0
):
    """Return (points Nx3 in camera frame mm, colors Nx3 rgb, (H,W)).

    SCARED's `left_depth_map.tiff` is an (H,W,3) float map whose channels are stored
    as (Z, Y, X) in the left-camera frame (mm); we reorder to (X, Y, Z).
    """
    import os
    import cv2
    import numpy as np

    raw = cv2.imread(
        os.path.join(keyframe_dir, "left_depth_map.tiff"), cv2.IMREAD_UNCHANGED
    )
    rgb = cv2.imread(os.path.join(keyframe_dir, "Left_Image.png"), cv2.IMREAD_COLOR)
    if raw is None or rgb is None:
        raise FileNotFoundError(f"missing depth/image in {keyframe_dir}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != raw.shape[:2]:
        rgb = cv2.resize(rgb, (raw.shape[1], raw.shape[0]))

    xyz = raw[..., ::-1]  # (Z,Y,X) -> (X,Y,Z)
    pts = xyz.reshape(-1, 3).astype(np.float32)
    cols = rgb.reshape(-1, 3)
    z = pts[:, 2]
    valid = np.isfinite(pts).all(axis=1) & (z > min_depth_mm) & (z < max_depth_mm)
    return pts[valid], cols[valid], raw.shape[:2]


def reconstruct_scared_keyframe(
    keyframe_dir: str, out_ply: str, max_depth_mm: float = 300.0
) -> int:
    """Reconstruct a colored point cloud from a SCARED keyframe and write a PLY."""
    pts, cols, _ = load_scared_keyframe(keyframe_dir, max_depth_mm=max_depth_mm)
    write_ply(out_ply, pts, cols)
    return int(pts.shape[0])


def render_points(
    points,
    colors,
    intr: CameraIntrinsics,
    out_size,
    R=None,
    t=None,
    point_radius: int = 1,
):
    """Splat a colored point cloud through a pinhole camera into an image (z-buffer).

    points/colors: (N,3). R,t: extrinsics (world->cam). Returns (H,W,3) uint8.
    Demonstrates novel-view rendering of the reconstructed world.
    """
    import numpy as np

    H, W = out_size
    if R is None:
        R = np.eye(3, dtype=np.float32)
    if t is None:
        t = np.zeros(3, dtype=np.float32)
    pc = (points @ R.T) + t
    z = pc[:, 2]
    front = z > 1e-6
    pc, cols = pc[front], np.asarray(colors)[front]
    z = pc[:, 2]
    u = (pc[:, 0] * intr.fx / z + intr.cx).round().astype(np.int64)
    v = (pc[:, 1] * intr.fy / z + intr.cy).round().astype(np.int64)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, cols = u[inb], v[inb], z[inb], cols[inb]
    order = np.argsort(-z)  # far first, near overwrite
    u, v, cols = u[order], v[order], cols[order]
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for du in range(-point_radius, point_radius + 1):
        for dv in range(-point_radius, point_radius + 1):
            uu = np.clip(u + du, 0, W - 1)
            vv = np.clip(v + dv, 0, H - 1)
            img[vv, uu] = cols
    return img


def fit_gaussian_splatting(sequence_dir: str, out_dir: str, cfg: dict | None = None):
    """Optimise a 3DGS scene for one sequence and export a renderable checkpoint.

    Hook for a 3DGS backend (`gsplat` / `nerfstudio`, or endoscopy-tuned
    EndoGaussian / Deform3DGS). Initialise Gaussians from the point cloud produced by
    reconstruct_scared_keyframe() / depth_to_points().
    """
    raise NotImplementedError(
        "Integrate a 3DGS backend. Initialise from reconstruct_scared_keyframe() PLY."
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--keyframe", required=True, help="SCARED keyframe dir")
    ap.add_argument("--out", required=True, help="output .ply path")
    ap.add_argument("--max-depth", type=float, default=300.0)
    args = ap.parse_args()
    n = reconstruct_scared_keyframe(args.keyframe, args.out, args.max_depth)
    print(f"[recon] wrote {n} points -> {args.out}")


if __name__ == "__main__":
    main()
