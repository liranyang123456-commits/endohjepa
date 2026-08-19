"""Ingest an official Cholec80 dump if the user places it under datasets/Cholec80.

Full Cholec80 is distributed by CAMMA after a request form
(https://camma.u-strasbg.fr/datasets). This module does not scrape unofficial
mirrors. It only extracts frames from a locally provided official copy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from endoworld.data.prepare import extract_all_videos

CAMMA = "https://camma.u-strasbg.fr/datasets"
REPO = Path(__file__).resolve().parents[3]
DEST = REPO / "datasets" / "Cholec80"


def status() -> dict:
    boxes = REPO / "datasets" / "Cholec80-Boxes"
    videos = list(DEST.rglob("*.mp4")) + list(DEST.rglob("*.avi"))
    frames = [p for p in DEST.rglob("*_frames") if p.is_dir()]
    box_vids = (
        sorted({p.parent.name for p in boxes.rglob("video_*")})
        if boxes.exists()
        else []
    )
    return {
        "official_request": CAMMA,
        "local_full_dir": str(DEST),
        "local_full_videos": len(videos),
        "local_full_frame_dirs": len(frames),
        "boxes_only": box_vids,
        "complete": len(videos) >= 80 or len(frames) >= 80,
    }


def ingest(src: str | Path | None = None, fps: float = 1.0) -> dict:
    """Copy/extract frames from an official dump dropped at --src or datasets/Cholec80."""
    Path(src) if src else DEST
    DEST.mkdir(parents=True, exist_ok=True)
    if src and Path(src).resolve() != DEST.resolve() and Path(src).exists():
        import shutil

        for p in Path(src).rglob("*"):
            if p.suffix.lower() in {".mp4", ".avi", ".mkv"}:
                dest = DEST / p.name
                if not dest.exists():
                    shutil.copy2(p, dest)
                    print(f"[cholec80] copied {p.name}")
    n = extract_all_videos(str(DEST), target_fps=fps, crop=True)
    print(f"[cholec80] extracted {n} frames into {DEST}")
    return status()


def main():
    ap = argparse.ArgumentParser(
        description="Ingest official Cholec80 (CAMMA request required)"
    )
    ap.add_argument(
        "--src",
        default="",
        help="folder containing official video01.mp4 ... video80.mp4",
    )
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status or not args.src:
        print(status())
        if not args.src:
            print(
                "Place the official dump then: python -m endoworld.data.cholec80 --src <dir>"
            )
            print(f"Request form: {CAMMA}")
        return
    print(ingest(args.src, args.fps))


if __name__ == "__main__":
    main()
