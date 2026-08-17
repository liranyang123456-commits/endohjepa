"""SCARED per-frame camera poses from frame_data.tar.gz → SE(3) deltas."""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np

from endoworld.world.c3vd_actions import pose_deltas
from endoworld.world.physical_actions import align_latents_and_poses


def _as_matrix44(obj) -> np.ndarray | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        for k in ("camera-pose", "camera_pose", "pose", "matrix"):
            if k in obj:
                return _as_matrix44(obj[k])
        return None
    arr = np.asarray(obj, dtype=np.float64)
    if arr.size == 16:
        return arr.reshape(4, 4)
    return None


def load_scared_poses(keyframe_dir: str | Path) -> np.ndarray:
    """Stack camera-pose matrices from data/frame_data.tar.gz. Returns (T, 4, 4)."""
    root = Path(keyframe_dir)
    tar_path = root / "data" / "frame_data.tar.gz"
    if not tar_path.is_file():
        tar_path = root / "frame_data.tar.gz"
    if not tar_path.is_file():
        raise FileNotFoundError(f"no frame_data.tar.gz under {root}")
    poses = []
    with tarfile.open(tar_path, "r:gz") as tar:
        names = sorted(n for n in tar.getnames() if n.endswith(".json") and "frame_data" in n.replace("\\", "/"))
        for name in names:
            fh = tar.extractfile(name)
            if fh is None:
                continue
            blob = json.loads(fh.read().decode("utf-8", "replace"))
            m = _as_matrix44(blob)
            if m is not None:
                poses.append(m)
    if not poses:
        raise ValueError(f"no camera-pose in {tar_path}")
    return np.stack(poses)


def find_scared_keyframes(root: str | Path = "datasets/SCARED") -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    out = []
    for p in root.rglob("frame_data.tar.gz"):
        kf = p.parent.parent if p.parent.name == "data" else p.parent
        out.append(kf)
    return sorted(set(out))


def scared_pose_deltas(keyframe_dir: str | Path) -> np.ndarray:
    return pose_deltas(load_scared_poses(keyframe_dir))


def align_scared_latents(
    latents, poses: np.ndarray, sampled_frame_indices: np.ndarray, tubelet: int,
):
    """Align encoder tubelets to native SCARED pose rows without interpolation."""
    return align_latents_and_poses(
        latents, poses, sampled_frame_indices, tubelet)


def find_scared_rgb(keyframe_dir: str | Path) -> tuple[Path | None, list[Path]]:
    """Return (rgb.mp4 or None, extracted frame paths). Prefer data/rgb_frames."""
    root = Path(keyframe_dir)
    data = root / "data" if (root / "data").is_dir() else root
    video = data / "rgb.mp4"
    if not video.is_file():
        vids = list(data.glob("*.mp4"))
        video = vids[0] if vids else None
    frames: list[Path] = []
    for d in (data / "rgb_frames", data / "rgb.mp4_frames"):
        if d.is_dir():
            frames = sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"})
            if frames:
                break
    return (video if video and video.is_file() else None), frames


def pose_index_for_frames(n_frames: int, n_poses: int) -> np.ndarray:
    """Map extracted / sampled frame i → pose row. frame_data{i:06d} is native index."""
    if n_frames <= 1 or n_poses <= 1:
        return np.zeros(max(n_frames, 0), dtype=np.int64)
    return np.round(np.linspace(0, n_poses - 1, n_frames)).astype(np.int64)


def sample_video_indices(video_path: str | Path, n: int) -> tuple[np.ndarray, int]:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if total <= 0:
        return np.zeros(0, dtype=np.int64), 0
    n = min(n, total)
    return np.round(np.linspace(0, total - 1, n)).astype(np.int64), total


def crop_stereo_half(rgb: np.ndarray, eye: str = "top") -> np.ndarray:
    """Crop one eye from a vertically stacked SCARED stereo frame.

    SCARED rgb.mp4 stores the two camera views stacked vertically
    (e.g. 2048x1280 = two 2048x640 views). The camera pose in
    frame_data.tar.gz belongs to a single camera; encoding the squashed
    stereo pair breaks the image-motion/SE(3) correspondence.
    """
    if rgb.shape[0] >= 1.4 * rgb.shape[1]:
        half = rgb.shape[0] // 2
        return rgb[:half] if eye == "top" else rgb[half:]
    return rgb


def read_video_frames(
    video_path: str | Path, indices: np.ndarray, image_size: int,
    stereo_eye: str | None = None,
):
    """Read selected frames from rgb.mp4 as (T, C, H, W) float32 in [0, 1].

    Sorted indices are decoded in a single sequential pass; random-access
    seeking per frame is 10-50x slower on long mp4 files.
    """
    import cv2
    from PIL import Image
    import torch
    indices = np.asarray(indices, dtype=np.int64)
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    if len(indices) and np.all(np.diff(indices) >= 0):
        wanted = {int(v): k for k, v in enumerate(indices)}
        decoded: dict[int, np.ndarray] = {}
        frame_id = 0
        last = int(indices[-1])
        while frame_id <= last:
            ok, bgr = cap.read()
            if not ok:
                break
            if frame_id in wanted:
                decoded[frame_id] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame_id += 1
        cap.release()
        ordered = [decoded.get(int(i)) for i in indices]
    else:
        ordered = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, bgr = cap.read()
            if not ok:
                continue
            ordered.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
    for rgb in ordered:
        if rgb is None:
            continue
        if stereo_eye in ("top", "bottom"):
            rgb = crop_stereo_half(rgb, stereo_eye)
        im = Image.fromarray(rgb).resize((image_size, image_size))
        frames.append(np.asarray(im, np.float32) / 255.0)
    if not frames:
        raise FileNotFoundError(f"no frames from {video_path}")
    arr = np.stack(frames).transpose(0, 3, 1, 2)
    return torch.from_numpy(arr)
