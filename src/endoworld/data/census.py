"""Summarise local endoscopic video by domain and split (read-only census).

Does not download data. Records GI / Cholec80 / ION coverage gaps for Endo-HJEPA.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


KNOWN_GAPS = {
    "Cholec80": "Local Cholec80-Boxes holds videos 41-45 only. Full 80 videos: CAMMA request (https://camma.u-strasbg.fr/datasets) then python -m endoworld.data.cholec80 --src <dir>.",
    "Kvasir-Capsule": "Labeled/unlabeled videos are large HTTP dumps; census reports whatever frames exist after ingest.",
    "C3VD": "Google Drive quota often blocks extra trajectories; pose-conditioned L3 uses whatever pose.txt files are present.",
    "ION_bronch": "Private ION cases live under anonymized datasets/ION_bronch/case_XXX only (no patient names in manifests).",
}


def census(manifest_csv: str | Path) -> dict:
    path = Path(manifest_csv)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by_ds: dict[str, dict] = defaultdict(
        lambda: {
            "sequences": 0,
            "frames": 0,
            "domain": "",
            "splits": defaultdict(int),
            "modalities": set(),
        }
    )
    by_dom: dict[str, dict] = defaultdict(
        lambda: {"sequences": 0, "frames": 0, "datasets": set()}
    )
    by_split: dict[str, int] = defaultdict(int)
    missing_dirs = 0
    for r in rows:
        ds = r["dataset"]
        dom = r.get("domain") or "mixed"
        split = r.get("split") or "unknown"
        # Video-only rows use -1 as an "undecoded / unknown" sentinel.
        # They count as sequences but must not reduce the decoded-frame census.
        n = max(0, int(r.get("num_frames") or 0))
        rec = by_ds[ds]
        rec["sequences"] += 1
        rec["frames"] += n
        rec["domain"] = dom
        rec["splits"][split] += 1
        rec["modalities"].add(r.get("modality") or "")
        by_dom[dom]["sequences"] += 1
        by_dom[dom]["frames"] += n
        by_dom[dom]["datasets"].add(ds)
        by_split[split] += 1
        fd = r.get("frames_dir") or ""
        if fd and not Path(fd).is_dir():
            missing_dirs += 1

    def _plain(d):
        out = {}
        for k, v in d.items():
            item = dict(v)
            item["splits"] = dict(item["splits"])
            item["modalities"] = sorted(x for x in item["modalities"] if x)
            out[k] = item
        return out

    report = {
        "manifest": path.as_posix(),
        "n_sequences": len(rows),
        "n_frames": sum(max(0, int(r.get("num_frames") or 0)) for r in rows),
        "missing_frames_dir": missing_dirs,
        "by_dataset": _plain(by_ds),
        "by_domain": {
            k: {
                "sequences": v["sequences"],
                "frames": v["frames"],
                "datasets": sorted(v["datasets"]),
            }
            for k, v in by_dom.items()
        },
        "by_split": dict(by_split),
        "gaps": KNOWN_GAPS,
        "notes": [
            "Sampling must be domain-balanced; laparoscopy otherwise dominates.",
            "Splits are video-level (see endoworld.data.splits); clips from one sequence do not cross splits.",
            "Endo-HJEPA paper is isolated from CT ablation planning (IBM/CBM/BMEO).",
        ],
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/sequences.csv")
    ap.add_argument("--out", default="manifests/domain_census.json")
    args = ap.parse_args()
    report = census(args.manifest)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[census] sequences={report['n_sequences']} frames={report['n_frames']}")
    print("[census] by domain:")
    for k, v in sorted(report["by_domain"].items()):
        print(
            f"  {k:8s} seq={v['sequences']:5d}  frames={v['frames']:8d}  {v['datasets']}"
        )
    print("[census] by split:", report["by_split"])
    print(f"[census] wrote {args.out}")


if __name__ == "__main__":
    main()
