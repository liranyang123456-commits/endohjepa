"""Ingest already-on-disk corpora into datasets/ without pulling the network.

- Targeted unzip of leftover SCARED / Stereo_Lap archives
- Anonymised copy/extract of ION intraoperative videos (numeric case ids only)
- STIR / leftover video frame extraction
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import zipfile
from pathlib import Path

from endoworld.data.prepare import extract_all_videos, video_to_frames, VIDEO_EXT

REPO = Path(__file__).resolve().parents[3]
DATASETS = REPO / "datasets"

# ION source lives outside the repo. Only the numeric prefix is copied into datasets/.
ION_SRC = Path(r"F:\省医-数据") / "省医-数据" / "ION患者整理"

CASE_RE = re.compile(r"^(\d{3})")


def _extract_zip(zp: Path, target: Path, overwrite: bool = False) -> bool:
    flag = target.parent / (target.name + ".extracted_ok")
    if flag.exists() and not overwrite:
        print(f"[skip] {zp.name}")
        return False
    target.mkdir(parents=True, exist_ok=True)
    print(f"[unzip] {zp} -> {target}")
    with zipfile.ZipFile(zp) as z:
        z.extractall(target)
    flag.write_text("ok", encoding="utf-8")
    return True


def unzip_scared(overwrite: bool = False) -> int:
    root = DATASETS / "SCARED"
    n = 0
    for zp in sorted(root.glob("dataset_*.zip")):
        if _extract_zip(zp, zp.with_suffix(""), overwrite):
            n += 1
    return n


def unzip_stereo_lap(overwrite: bool = False) -> int:
    root = DATASETS / "Stereo_Lap"
    n = 0
    for zp in sorted(root.glob("rectified*.zip")):
        if _extract_zip(zp, zp.with_suffix(""), overwrite):
            n += 1
    return n


def ingest_ion(overwrite: bool = False) -> int:
    """Copy/extract ION 术中视频 into datasets/ION_bronch/case_XXX (no patient names)."""
    if not ION_SRC.exists():
        print(f"[ion] source missing: {ION_SRC}")
        return 0
    dest_root = DATASETS / "ION_bronch"
    dest_root.mkdir(parents=True, exist_ok=True)
    n = 0
    for folder in sorted(ION_SRC.iterdir()):
        if not folder.is_dir():
            continue
        m = CASE_RE.match(folder.name)
        if not m:
            continue
        case_id = m.group(1)
        dest = dest_root / f"case_{case_id}"
        dest.mkdir(parents=True, exist_ok=True)
        video_dir = folder / "术中视频"
        if not video_dir.exists():
            continue
        vid_i = 0
        zip_i = 0
        for item in sorted(video_dir.iterdir()):
            if item.suffix.lower() == ".zip":
                out = dest / f"intraop_{zip_i:02d}"
                if _extract_zip(item, out, overwrite):
                    n += 1
                zip_i += 1
            elif item.suffix.lower() in VIDEO_EXT:
                tgt = dest / f"video_{vid_i:02d}{item.suffix.lower()}"
                if not tgt.exists() or overwrite:
                    shutil.copy2(item, tgt)
                    print(f"[ion] copy video case_{case_id}")
                    n += 1
                vid_i += 1
        # extract any copied/extracted videos to frames
        extract_all_videos(str(dest), target_fps=2.0, crop=True)
    return n


def extract_stirs(fps: float = 3.0) -> int:
    root = DATASETS / "STIR"
    if not root.exists():
        return 0
    return extract_all_videos(str(root), target_fps=fps, crop=True)


def extract_scared_videos(fps: float = 2.0) -> int:
    root = DATASETS / "SCARED"
    if not root.exists():
        return 0
    return extract_all_videos(str(root), target_fps=fps, crop=True)


def extract_extra_videos() -> int:
    """Decode TrackVes / EndoVis tracking / own MIS videos into datasets/ extras."""
    from endoworld.data.domains import extra_local_roots

    total = 0
    for name, src in extra_local_roots():
        if not src.exists():
            print(f"[skip extra] {name} {src}")
            continue
        dest = DATASETS / name
        dest.mkdir(parents=True, exist_ok=True)
        # If source already has many frames, just leave a pointer file; scanner
        # will also walk extra roots. Still extract videos found under dest copies.
        n_vid = 0
        for vp in src.rglob("*"):
            if not vp.is_file() or vp.suffix.lower() not in VIDEO_EXT:
                continue
            rel = vp.relative_to(src)
            out_vid = dest / rel
            out_vid.parent.mkdir(parents=True, exist_ok=True)
            if not out_vid.exists():
                try:
                    os.link(vp, out_vid)
                except OSError:
                    shutil.copy2(vp, out_vid)
            frames_dir = out_vid.parent / (out_vid.stem + "_frames")
            flag = out_vid.parent / (out_vid.stem + "_frames.extracted_ok")
            if flag.exists():
                continue
            saved = video_to_frames(str(out_vid), str(frames_dir), target_fps=2.0, crop=True)
            flag.write_text(str(saved), encoding="utf-8")
            print(f"[extra] {name}/{rel}: {saved} frames")
            n_vid += 1
            total += saved
        print(f"[extra] {name}: decoded {n_vid} videos")
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest local endoscopic corpora")
    ap.add_argument("--skip-unzip", action="store_true")
    ap.add_argument("--skip-ion", action="store_true")
    ap.add_argument("--skip-extra", action="store_true")
    ap.add_argument("--skip-frames", action="store_true")
    args = ap.parse_args()

    if not args.skip_unzip:
        print(f"[unzip] SCARED leftover {unzip_scared()}")
        print(f"[unzip] Stereo_Lap leftover {unzip_stereo_lap()}")
    if not args.skip_ion:
        print(f"[ion] ingested {ingest_ion()}")
    if not args.skip_extra:
        print(f"[extra] frames {extract_extra_videos()}")
    if not args.skip_frames:
        print(f"[stir] frames {extract_stirs()}")
        print(f"[scared] frames {extract_scared_videos()}")


if __name__ == "__main__":
    main()
