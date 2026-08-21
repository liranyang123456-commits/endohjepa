"""Replace rows for selected datasets in sequences.csv without re-walking SCARED."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from endoworld.data.domains import infer_domain
from endoworld.data.scan_datasets import scan_dataset
from endoworld.data.splits import apply_splits, assign_split, video_key


def refresh(manifest: Path, datasets_root: Path, names: list[str]) -> None:
    old = list(csv.DictReader(manifest.open(encoding="utf-8")))
    fields = list(old[0].keys()) if old else []
    keep = [r for r in old if r["dataset"] not in set(names)]
    added = []
    for name in names:
        root = datasets_root / name
        if not root.is_dir():
            print(f"[skip] missing {root}")
            continue
        rows = scan_dataset(root, name)
        for r in rows:
            r.domain = infer_domain(r.dataset, r.frames_dir)
            if (r.split or "unknown") not in ("train", "val", "test"):
                r.split = assign_split(video_key(r.dataset, r.sequence_id))
        n = sum(x.num_frames for x in rows if x.num_frames > 0)
        print(f"[refresh] {name:24s} seq={len(rows):5d}  frames={n:8d}")
        added.extend(asdict(r) for r in rows)
        if not fields and rows:
            fields = list(asdict(rows[0]).keys())
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in keep + added:
            w.writerow({k: r.get(k, "") for k in fields})
    apply_splits(manifest)
    print(f"[refresh] total sequences={len(keep) + len(added)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/sequences.csv")
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--datasets", default="Kvasir-Capsule,ION_bronch")
    args = ap.parse_args()
    refresh(Path(args.manifest), Path(args.root).resolve(), [x.strip() for x in args.datasets.split(",") if x.strip()])


if __name__ == "__main__":
    main()
