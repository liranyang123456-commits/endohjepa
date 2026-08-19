"""Scan the copied endoscopy datasets and build a unified, sequence-level manifest.

The manifest is deliberately at *sequence* granularity (one row = one directory
of frames / one video), not per-frame, so it stays small even for large corpora
like SCARED (~150 GB). A video dataloader can expand each row into clips on the fly.

Run:
    python -m endoworld.data.scan_datasets --root <path-to>/datasets --out <path>/manifests
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

from endoworld.data.domains import extra_local_roots, infer_domain

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}
DEPTH_HINTS = ("depth", "disparity", "disp")
MASK_HINTS = ("mask", "seg", "label", "gt", "annotation", "semseg", "insseg")
RIGHT_HINTS = ("right", "_r", "rectified1", "cam1")

# Folders that are outputs / logs / archives rather than raw data.
SKIP_DIR_NAMES = {
    "_copy_logs",
    "__pycache__",
    "_downloads",
    "jpeg",
    "mesh_cache",
    "logs",
    "results",
    "scripts",
    "Comparisons",
}


@dataclass
class SequenceRow:
    dataset: str
    sequence_id: str  # relative path of the frame directory
    modality: str  # rgb | stereo | rgbd | video
    frames_dir: str  # absolute path
    num_frames: int
    has_depth: bool
    has_mask: bool
    has_stereo: bool
    split: str  # train | val | test | unknown
    sample_frame: str  # one example file (for quick preview)
    domain: str = "mixed"  # laparo | gi | bronch | mixed


def _infer_split(rel_path: str) -> str:
    low = rel_path.lower().replace("\\", "/")
    for key in ("train", "val", "test"):
        if f"/{key}" in f"/{low}" or low.startswith(key) or f"_{key}" in low:
            return key
    return "unknown"


def _classify_dir(dir_name: str) -> str:
    low = dir_name.lower()
    if any(h in low for h in DEPTH_HINTS):
        return "depth"
    if any(h in low for h in MASK_HINTS):
        return "mask"
    if any(h in low for h in RIGHT_HINTS):
        return "right"
    return "rgb"


def scan_dataset(
    dataset_dir: Path, dataset_name: str | None = None
) -> list[SequenceRow]:
    """Group image files by their containing directory -> one sequence per dir."""
    name = dataset_name or dataset_dir.name
    rows: list[SequenceRow] = []
    # dir -> list of image files
    dir_images: dict[Path, list[str]] = defaultdict(list)
    dir_videos: dict[Path, list[str]] = defaultdict(list)

    for cur, dirs, files in os.walk(dataset_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES]
        cur_path = Path(cur)
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXT:
                dir_images[cur_path].append(f)
            elif ext in VIDEO_EXT:
                dir_videos[cur_path].append(f)

    # Standalone video files -> one sequence each.
    for d, vids in dir_videos.items():
        for v in vids:
            rel = str((d / v).relative_to(dataset_dir))
            rows.append(
                SequenceRow(
                    dataset=name,
                    sequence_id=rel,
                    modality="video",
                    frames_dir=str(d),
                    num_frames=-1,  # unknown without decoding
                    has_depth=False,
                    has_mask=False,
                    has_stereo=False,
                    split=_infer_split(rel),
                    sample_frame=str(d / v),
                    domain=infer_domain(name, rel),
                )
            )

    # Frame directories: only treat "rgb" dirs as primary sequences; check siblings
    # for depth / mask / right-eye companions.
    for d, imgs in dir_images.items():
        role = _classify_dir(d.name)
        if role != "rgb":
            continue  # depth/mask/right dirs are companions, not primary sequences
        parent = d.parent
        sibling_names = (
            {p.name.lower() for p in parent.iterdir() if p.is_dir()}
            if parent.exists()
            else set()
        )
        has_depth = any(any(h in n for h in DEPTH_HINTS) for n in sibling_names)
        has_mask = any(any(h in n for h in MASK_HINTS) for n in sibling_names)
        has_stereo = any(any(h in n for h in RIGHT_HINTS) for n in sibling_names)
        modality = "rgbd" if has_depth else ("stereo" if has_stereo else "rgb")
        rel = str(d.relative_to(dataset_dir))
        rows.append(
            SequenceRow(
                dataset=name,
                sequence_id=rel,
                modality=modality,
                frames_dir=str(d),
                num_frames=len(imgs),
                has_depth=has_depth,
                has_mask=has_mask,
                has_stereo=has_stereo,
                split=_infer_split(rel),
                sample_frame=str(d / sorted(imgs)[0]),
                domain=infer_domain(name, rel),
            )
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to the datasets/ directory")
    ap.add_argument(
        "--out",
        default=None,
        help="Output dir for manifests (default: <root>/../manifests)",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve() if args.out else root.parent / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = [
        p for p in sorted(root.iterdir()) if p.is_dir() and p.name not in SKIP_DIR_NAMES
    ]
    extras = [(name, path) for name, path in extra_local_roots() if path.exists()]

    all_rows: list[SequenceRow] = []
    summary: list[dict] = []
    jobs: list[tuple[str, Path]] = [(p.name, p) for p in dataset_dirs]
    seen_paths = {p.resolve() for p in dataset_dirs}
    for extra_name, extra_path in extras:
        rp = extra_path.resolve()
        if rp in seen_paths:
            continue
        jobs.append((extra_name, extra_path))
        seen_paths.add(rp)

    all_rows: list[SequenceRow] = []
    summary: list[dict] = []
    for ds_name, ds in jobs:
        rows = scan_dataset(ds, dataset_name=ds_name)
        all_rows.extend(rows)
        n_frames = sum(r.num_frames for r in rows if r.num_frames > 0)
        summary.append(
            {
                "dataset": ds_name,
                "domain": infer_domain(ds_name),
                "sequences": len(rows),
                "total_frames": n_frames,
                "has_depth": any(r.has_depth for r in rows),
                "has_stereo": any(r.has_stereo for r in rows),
                "has_mask": any(r.has_mask for r in rows),
                "modalities": ",".join(sorted({r.modality for r in rows})),
            }
        )
        print(
            f"[scan] {ds_name:28s} domain={infer_domain(ds_name):6s} seq={len(rows):5d}  frames={n_frames:8d}"
        )

    from endoworld.data.splits import apply_splits, assign_split, video_key

    for r in all_rows:
        if (r.split or "unknown") not in ("train", "val", "test"):
            r.split = assign_split(video_key(r.dataset, r.sequence_id))

    manifest_path = out_dir / "sequences.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=list(asdict(all_rows[0]).keys()) if all_rows else []
        )
        w.writeheader()
        for r in all_rows:
            w.writerow(asdict(r))

    summary_path = out_dir / "dataset_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else [])
        w.writeheader()
        for s in summary:
            w.writerow(s)

    print(f"\n[done] {len(all_rows)} sequences across {len(jobs)} datasets")
    print(f"[done] manifest -> {manifest_path}")
    print(f"[done] summary  -> {summary_path}")
    apply_splits(manifest_path)


if __name__ == "__main__":
    main()
