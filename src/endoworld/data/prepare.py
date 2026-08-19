"""Stage 0.5 - data preparation.

Sub-commands:
  unzip           extract all *.zip archives in a dataset (Stereo_Lap, SCARED d5-7)
  extract-videos  decode *.mp4 (e.g. SCARED rgb.mp4) into frames at a target fps
  crop-borders    remove endoscopic black borders in-place-copy to a *_cropped dir

De-vignette / black-border cropping matters: endoscopic frames are a bright circle
on black; the black margin (and burned-in UI text at corners) can leak shortcuts
into self-supervised training, so we crop to the content bounding box.

Only stdlib + numpy + opencv are required.
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


# ----------------------------------------------------------------------------- #
# Black-border detection
# ----------------------------------------------------------------------------- #
def content_bbox(img: np.ndarray, thresh: int = 12, margin: int = 2):
    """Return (y0, y1, x0, x1) bounding box of non-black content."""
    gray = img.mean(axis=2) if img.ndim == 3 else img
    ys, xs = np.where(gray > thresh)
    if ys.size == 0:
        return 0, img.shape[0], 0, img.shape[1]
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, img.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, img.shape[1])
    return y0, y1, x0, x1


def crop_content(img: np.ndarray, thresh: int = 12) -> np.ndarray:
    y0, y1, x0, x1 = content_bbox(img, thresh)
    return img[y0:y1, x0:x1]


# ----------------------------------------------------------------------------- #
# unzip
# ----------------------------------------------------------------------------- #
def unzip_all(dataset_dir: str, overwrite: bool = False) -> list[str]:
    root = Path(dataset_dir)
    done = []
    for zp in sorted(root.rglob("*.zip")):
        target = zp.with_suffix("")
        flag = target.parent / (target.name + ".extracted_ok")
        if flag.exists() and not overwrite:
            print(f"[skip] already extracted: {zp.name}")
            continue
        target.mkdir(parents=True, exist_ok=True)
        print(f"[unzip] {zp.name} -> {target}")
        with zipfile.ZipFile(zp) as z:
            z.extractall(target)
        flag.write_text("ok", encoding="utf-8")
        done.append(str(target))
    return done


# ----------------------------------------------------------------------------- #
# video -> frames
# ----------------------------------------------------------------------------- #
def video_to_frames(
    video_path: str,
    out_dir: str,
    target_fps: float = 2.0,
    crop: bool = True,
    max_frames: int | None = None,
) -> int:
    if cv2 is None:
        raise RuntimeError("opencv required: pip install opencv-python")
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(src_fps / target_fps)), 1)
    idx, saved = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            if crop:
                frame = crop_content(frame)
            cv2.imwrite(
                os.path.join(out_dir, f"frame_{saved:06d}.jpg"),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
            saved += 1
            if max_frames and saved >= max_frames:
                break
        idx += 1
    cap.release()
    return saved


def extract_all_videos(
    dataset_dir: str,
    target_fps: float = 2.0,
    crop: bool = True,
    max_frames: int | None = None,
) -> int:
    root = Path(dataset_dir)
    total = 0
    for vp in sorted(root.rglob("*")):
        if vp.suffix.lower() not in VIDEO_EXT:
            continue
        out_dir = vp.parent / (vp.stem + "_frames")
        flag = vp.parent / (vp.stem + "_frames.extracted_ok")
        if flag.exists():
            print(f"[skip] frames exist: {vp}")
            continue
        n = video_to_frames(str(vp), str(out_dir), target_fps, crop, max_frames)
        flag.write_text(str(n), encoding="utf-8")
        print(f"[video] {vp.name}: {n} frames -> {out_dir}")
        total += n
    return total


# ----------------------------------------------------------------------------- #
# CLI
# ----------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="EndoWorld data preparation")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("unzip")
    p1.add_argument("--dir", required=True)
    p1.add_argument("--overwrite", action="store_true")

    p2 = sub.add_parser("extract-videos")
    p2.add_argument("--dir", required=True)
    p2.add_argument("--fps", type=float, default=2.0)
    p2.add_argument("--no-crop", action="store_true")
    p2.add_argument("--max-frames", type=int, default=None)

    args = ap.parse_args()
    if args.cmd == "unzip":
        out = unzip_all(args.dir, args.overwrite)
        print(f"[done] extracted {len(out)} archive(s)")
    elif args.cmd == "extract-videos":
        n = extract_all_videos(args.dir, args.fps, not args.no_crop, args.max_frames)
        print(f"[done] extracted {n} frames total")


if __name__ == "__main__":
    main()
