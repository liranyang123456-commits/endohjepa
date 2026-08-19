"""Clip sampling dataset built on top of manifests/sequences.csv.

Expands each frame-directory sequence into fixed-length clips for V-JEPA style
self-supervised training. Kept dependency-light: torch/PIL are imported lazily so
the manifest tooling works without a full training environment.
"""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass

from endoworld.data.domains import infer_domain


IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class ClipSpec:
    dataset: str
    frames_dir: str
    frame_files: list[str]  # sorted frame filenames within frames_dir
    start: int  # index into frame_files
    clip_len: int
    stride: int
    domain: str = "mixed"
    sequence_id: str = ""
    split: str = "unknown"

    def frame_paths(self) -> list[str]:
        idxs = [self.start + i * self.stride for i in range(self.clip_len)]
        idxs = [min(i, len(self.frame_files) - 1) for i in idxs]
        return [os.path.join(self.frames_dir, self.frame_files[i]) for i in idxs]


def domain_balanced_indices(
    clips: list[ClipSpec], n: int | None = None, seed: int = 0
) -> list[int]:
    """Round-robin across domains so laparoscopy does not drown GI/bronch."""
    from collections import defaultdict

    by: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(clips):
        by[c.domain].append(i)
    rng = random.Random(seed)
    for v in by.values():
        rng.shuffle(v)
    keys = list(by.keys())
    out: list[int] = []
    while keys:
        nxt = []
        for k in keys:
            bucket = by[k]
            if bucket:
                out.append(bucket.pop())
                nxt.append(k)
            if n and len(out) >= n:
                return out
        keys = nxt
    return out


def _list_frames(frames_dir: str) -> list[str]:
    try:
        files = [
            f
            for f in os.listdir(frames_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXT
        ]
    except OSError:
        return []
    return sorted(files)


def build_clip_index(
    manifest_csv: str,
    clip_len: int = 16,
    stride: int = 4,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    include_domains: list[str] | None = None,
    split: str | None = None,
) -> list[ClipSpec]:
    """Read the sequence manifest and enumerate all clips (non-overlapping)."""
    include = set(include or [])
    exclude = set(exclude or [])
    include_domains = set(include_domains or [])
    clips: list[ClipSpec] = []
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["modality"] == "video":
                continue  # decode to *_frames first via ingest_local / prepare
            ds = row["dataset"]
            if include and ds not in include:
                continue
            if ds in exclude:
                continue
            row_split = row.get("split") or "unknown"
            if split and row_split != split:
                continue
            domain = row.get("domain") or infer_domain(ds, row.get("frames_dir", ""))
            if include_domains and domain not in include_domains:
                continue
            frames_dir = row["frames_dir"]
            if not os.path.isdir(frames_dir):
                continue
            files = _list_frames(frames_dir)
            span = (clip_len - 1) * stride + 1
            if len(files) < span:
                continue
            seq_id = row.get("sequence_id") or frames_dir
            for start in range(0, len(files) - span + 1, span):
                clips.append(
                    ClipSpec(
                        ds,
                        frames_dir,
                        files,
                        start,
                        clip_len,
                        stride,
                        domain,
                        seq_id,
                        row_split,
                    )
                )
    return clips


class EndoClipDataset:
    """PyTorch-style dataset yielding (T, C, H, W) clip tensors.

    Usable as a torch Dataset when torch/PIL are installed; otherwise raises a
    clear error on first item access.
    """

    def __init__(
        self,
        manifest_csv: str,
        clip_len: int = 16,
        stride: int = 4,
        image_size: int = 224,
        include=None,
        exclude=None,
        include_domains=None,
        return_meta: bool = False,
        split: str | None = None,
    ):
        self.clips = build_clip_index(
            manifest_csv, clip_len, stride, include, exclude, include_domains, split
        )
        self.image_size = image_size
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        try:
            import numpy as np
            from PIL import Image
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "Pillow/numpy required to load clips: pip install pillow numpy"
            ) from e

        from PIL import ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate slightly corrupt JPEGs

        spec = self.clips[idx]
        frames = []
        prev = None
        for p in spec.frame_paths():
            try:
                img = (
                    Image.open(p)
                    .convert("RGB")
                    .resize((self.image_size, self.image_size))
                )
                arr = np.asarray(img, dtype=np.float32) / 255.0
                prev = arr
            except Exception:
                # unreadable frame: reuse previous frame or a black placeholder
                arr = (
                    prev
                    if prev is not None
                    else np.zeros(
                        (self.image_size, self.image_size, 3), dtype=np.float32
                    )
                )
            frames.append(arr)
        arr = np.stack(frames, axis=0)  # (T, H, W, C)
        arr = arr.transpose(0, 3, 1, 2)  # (T, C, H, W)
        try:
            import torch

            tensor = torch.from_numpy(arr)
        except ImportError:
            tensor = arr
        if self.return_meta:
            from endoworld.data.domains import DOMAIN_IDS

            spec = self.clips[idx]
            return tensor, DOMAIN_IDS.get(spec.domain, 3)
        return tensor


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/sequences.csv")
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()
    idx = build_clip_index(args.manifest, args.clip_len, args.stride)
    by_ds: dict[str, int] = {}
    for c in idx:
        by_ds[c.dataset] = by_ds.get(c.dataset, 0) + 1
    print(f"total clips: {len(idx)}")
    for k, v in sorted(by_ds.items()):
        print(f"  {k:24s} {v}")
