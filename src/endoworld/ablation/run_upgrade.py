"""One-shot runner: public-dataset notes + A seg3d + B masks + patient sim.

    PYTHONPATH=src python -m endoworld.ablation.run_upgrade --limit-a 2 --limit-b 4
"""
from __future__ import annotations

import argparse
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-a", type=int, default=3, help="ION cases for 3D seg")
    ap.add_argument("--limit-b", type=int, default=4, help="follow-up cases")
    ap.add_argument("--skip-unzip", action="store_true")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--skip-sim", action="store_true")
    ap.add_argument("--force-zone", type=float, default=10.0)
    args = ap.parse_args(argv)

    if not args.skip_a:
        print("=" * 60)
        print("[A] segment3d")
        from endoworld.ablation.segment3d import main as seg_main
        rc = seg_main(["--limit", str(args.limit_a),
                       "--out", "outputs/ablation_seg3d",
                       "--cache", "outputs/ct_cache_ion"])
        if rc:
            return rc

    if not args.skip_b:
        print("=" * 60)
        print("[B] followup_masks")
        from endoworld.ablation.followup_masks import main as fu_main
        argv_b = ["--out", "outputs/ablation_followup_masks",
                  "--limit", str(args.limit_b)]
        if not args.skip_unzip:
            argv_b.append("--extract-zips")
        rc = fu_main(argv_b)
        if rc:
            return rc

    if not args.skip_sim:
        print("=" * 60)
        print("[sim] patient_sim")
        from endoworld.ablation.patient_sim import main as sim_main
        rc = sim_main(["--seg", "outputs/ablation_seg3d",
                       "--out", "outputs/ablation_patient_sim",
                       "--limit", str(args.limit_a),
                       "--force-zone", str(args.force_zone)])
        if rc:
            return rc

    print("=" * 60)
    print("[done] upgrade pipeline finished")
    print("  docs/PUBLIC_DATASETS.md")
    print("  outputs/ablation_seg3d/")
    print("  outputs/ablation_followup_masks/")
    print("  outputs/ablation_patient_sim/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
