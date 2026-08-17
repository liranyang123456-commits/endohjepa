"""STIR Challenge start/end IR segmentation → point sets for L1 track regulariser."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class StirClip:
    seq_dir: Path
    frames: list[Path]
    points_start: np.ndarray  # (N, 2) xy
    points_end: np.ndarray
    calib: dict
    t_start_ms: int
    t_end_ms: int


def _points_from_seg(path: Path, max_points: int = 64) -> np.ndarray:
    from PIL import Image
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    ys, xs = np.where(arr >= 128)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    if len(pts) > max_points:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    return pts


def _ms_from_name(name: str) -> int:
    digits = "".join(ch for ch in Path(name).stem if ch.isdigit())
    return int(digits[:8]) if digits else 0


def load_stir_clip(seq_dir: str | Path, max_points: int = 64) -> StirClip | None:
    seq = Path(seq_dir)
    start_seg = seq / "segmentation" / "icgstartseg.png"
    end_seg = seq / "segmentation" / "icgendseg.png"
    if not (start_seg.is_file() and end_seg.is_file()):
        return None
    frame_dirs = sorted(p for p in seq.rglob("*_frames") if p.is_dir())
    frames: list[Path] = []
    if frame_dirs:
        frames = sorted(frame_dirs[0].glob("frame_*.jpg")) + sorted(frame_dirs[0].glob("frame_*.png"))
    starts = list(seq.glob("*_icgstart.png"))
    ends = list(seq.glob("*_icgend.png"))
    calib = {}
    calib_path = seq.parent.parent / "calib.json"
    if calib_path.is_file():
        calib = json.loads(calib_path.read_text(encoding="utf-8"))
    return StirClip(
        seq_dir=seq,
        frames=frames,
        points_start=_points_from_seg(start_seg, max_points),
        points_end=_points_from_seg(end_seg, max_points),
        calib=calib,
        t_start_ms=_ms_from_name(starts[0].name) if starts else 0,
        t_end_ms=_ms_from_name(ends[0].name) if ends else 0,
    )


def find_stir_sequences(root: str | Path = "datasets/STIR") -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted({p.parent.parent for p in root.rglob("icgstartseg.png")})


def scale_points(pts: np.ndarray, src_wh: tuple[int, int], dst: int) -> np.ndarray:
    if pts.size == 0:
        return pts.reshape(0, 2).astype(np.float32)
    w, h = src_wh
    return pts * np.array([dst / max(w, 1), dst / max(h, 1)], dtype=np.float32)


def stir_clip_tensors(clip: StirClip, image_size: int, n_frames: int = 8):
    """Load visible frames + start/end points resized to the encoder grid."""
    from PIL import Image
    import torch
    if len(clip.frames) < 2:
        return None
    idxs = np.linspace(0, len(clip.frames) - 1, min(n_frames, len(clip.frames)))
    idxs = np.unique(idxs.round().astype(int))
    frames, src_wh = [], None
    for i in idxs:
        im = Image.open(clip.frames[i]).convert("RGB")
        if src_wh is None:
            src_wh = im.size
        frames.append(np.asarray(im.resize((image_size, image_size)), np.float32) / 255.0)
    if src_wh is None:
        return None
    arr = np.stack(frames).transpose(0, 3, 1, 2)
    pts0 = scale_points(clip.points_start, src_wh, image_size)
    pts1 = scale_points(clip.points_end, src_wh, image_size)
    return torch.from_numpy(arr), torch.from_numpy(pts0), torch.from_numpy(pts1)
