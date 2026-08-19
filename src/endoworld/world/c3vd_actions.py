"""Load C3VD pose.txt into per-frame camera deltas for action-conditioned L3."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from endoworld.world.physical_actions import (
    align_latents_and_poses,
    pose_deltas as _physical_pose_deltas,
)


def load_pose_txt(path: str | Path) -> np.ndarray:
    """Return (N, 4, 4) camera-to-world matrices in OpenCV camera convention.

    C3VD pose.txt stores camera-to-world matrices flattened so that, after
    reshape(4,4), the translation sits in the last row (column-major c2w).
    The camera frame is OpenGL-style (+y up, +z out of the screen), whereas
    this implementation expresses twists in OpenCV coordinates (+y down, +z
    into the screen). We therefore apply the conventional GL-to-CV
    conjugation. ``endoworld.eval.c3vd_pose_gate`` now records only a
    depth-warp diagnostic for this implementation convention; without
    independent cross-frame correspondences it cannot select or validate a
    translation convention. C3VD action results are consequently
    convention-specific external diagnostics, not independently validated
    pose-grounded generalisation.
    """
    rows = []
    text = Path(path).read_text(encoding="utf-8").strip().splitlines()
    for line in text:
        vals = [float(x) for x in line.replace(",", " ").split()]
        if len(vals) == 16:
            rows.append(np.array(vals, dtype=np.float64).reshape(4, 4))
    if not rows:
        raise ValueError(f"no 4x4 poses in {path}")
    poses = np.stack(rows)
    t_col = np.linalg.norm(poses[:, :3, 3], axis=1).mean()
    t_row = np.linalg.norm(poses[:, 3, :3], axis=1).mean()
    if t_row > t_col * 10:
        poses = np.transpose(poses, (0, 2, 1))
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return flip @ poses @ flip


def pose_deltas(poses: np.ndarray) -> np.ndarray:
    """Canonical local SE(3) logarithm [v_x,v_y,v_z,w_x,w_y,w_z]."""
    return _physical_pose_deltas(poses)


def _rodrigues(R: np.ndarray) -> tuple[np.ndarray, float]:
    cos = np.clip((np.trace(R) - 1) * 0.5, -1.0, 1.0)
    ang = float(np.arccos(cos))
    if ang < 1e-8:
        return np.zeros(3), 0.0
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (
        2 * np.sin(ang)
    )
    return w * ang, ang


def find_c3vd_pose_files(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(root.rglob("pose.txt"))


def find_c3vd_color_frames(seq_dir: str | Path) -> list[Path]:
    """RGB frames next to pose.txt (skip depth / occlusion / normals).

    Supports both the v1 flat layout (0000_color.png beside pose.txt) and the
    C3VDv2 layout (rgb/0000.png in a subdirectory).
    """
    root = Path(seq_dir)
    skip = ("occlusion", "depth", "normal", "flow", "mask")
    out = []
    for p in sorted(root.iterdir()) if root.is_dir() else []:
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        low = p.name.lower()
        if any(s in low for s in skip):
            continue
        out.append(p)
    if not out:
        rgb_dir = root / "rgb"
        if rgb_dir.is_dir():
            out = sorted(
                p
                for p in rgb_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
    if not out:
        out = list(root.glob("*color*"))

    def _key(p: Path):
        digits = "".join(ch for ch in p.stem.split("_")[0] if ch.isdigit())
        return (0, int(digits)) if digits else (1, p.name)

    return sorted(out, key=_key)


def align_c3vd_latents(
    latents,
    poses: np.ndarray,
    sampled_frame_indices: np.ndarray,
    tubelet: int,
):
    """Align encoder tubelets to native C3VD pose rows without interpolation."""
    return align_latents_and_poses(latents, poses, sampled_frame_indices, tubelet)
