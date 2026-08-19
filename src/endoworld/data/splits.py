"""Video-level train/val/test splits.

Clips from the same sequence (same frames_dir / sequence_id) never cross splits.
Assignment is a stable hash of (dataset, sequence_id), independent of scan order.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def video_key(dataset: str, sequence_id: str) -> str:
    return f"{dataset}::{sequence_id.replace(chr(92), '/')}"


def assign_split(key: str, seed: int = 0, train: float = 0.8, val: float = 0.1) -> str:
    h = hashlib.md5(f"{seed}|{key}".encode("utf-8")).hexdigest()
    u = int(h[:8], 16) / 0xFFFFFFFF
    if u < train:
        return "train"
    if u < train + val:
        return "val"
    return "test"


def apply_splits(manifest_csv: str | Path, seed: int = 0) -> Path:
    path = Path(manifest_csv)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return path
    n_tr = n_va = n_te = 0
    for row in rows:
        # keep existing train/val/test from folder names when present and not unknown
        existing = (row.get("split") or "unknown").lower()
        if existing in ("train", "val", "test"):
            split = existing
        else:
            split = assign_split(
                video_key(row["dataset"], row["sequence_id"]), seed=seed
            )
        row["split"] = split
        if split == "train":
            n_tr += 1
        elif split == "val":
            n_va += 1
        else:
            n_te += 1
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[split] video-level  train={n_tr}  val={n_va}  test={n_te}  -> {path}")
    return path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Assign video-level train/val/test on sequences.csv"
    )
    ap.add_argument("--manifest", default="manifests/sequences.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    apply_splits(args.manifest, seed=args.seed)
