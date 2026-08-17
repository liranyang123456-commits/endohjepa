"""Cohort B: extract missing vue/zip CTs + per-timepoint ablation-zone masks.

1. Unzip GE ``vue_*.zip`` / case ``*.zip`` that currently block 24h (and other)
   timepoints from having readable DICOM.
2. Re-run longitudinal ROI volumetry and **export** ``M_pre`` / ``M_post`` NPZ
   masks at every usable timepoint (learnable zone-evolution targets).
3. Refresh ``followup_summary.csv`` with any newly recovered timepoints.

    PYTHONPATH=src python -m endoworld.ablation.followup_masks --extract-zips \\
        --out outputs/ablation_followup_masks
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import zipfile
from typing import Any

import numpy as np

from endoworld.ablation.dicom_io import load_largest_series, segment_lung
from endoworld.ablation.followup import (
    classify_response,
    detect_seed,
    find_clinical_root,
    find_followup_subset,
    measure_roi,
    series_dir_with_most_dicoms,
)
from endoworld.ablation.trajectory_schema import save_mask


def _count_dicoms(folder: str) -> int:
    n = 0
    for cur, _, files in os.walk(folder):
        for f in files:
            if f.upper() in ("DICOMDIR", "VERSION", "LOCKFILE"):
                continue
            if "." not in f or f.lower().endswith((".dcm", ".dicom", ".ima")):
                n += 1
    return n


def extract_zips(fu_root: str, min_existing: int = 60) -> list[dict]:
    """Extract vue_*.zip always; extract other *.zip only if folder lacks DICOM."""
    report = []
    for cur, _, files in os.walk(fu_root):
        for f in files:
            if not f.lower().endswith(".zip"):
                continue
            zpath = os.path.join(cur, f)
            stem = f[:-4]
            is_vue = "vue_" in f.lower()
            dest = os.path.join(cur, f"_extracted_{stem[:48]}")
            marker = os.path.join(dest, ".extracted_ok")
            existing = _count_dicoms(cur)
            # Skip non-vue archives when the folder already has a usable series
            if (not is_vue) and existing >= min_existing:
                report.append({"zip": zpath, "status": "skip_has_dicom",
                               "n_dicom": existing})
                continue
            if os.path.isfile(marker):
                report.append({"zip": zpath, "status": "already", "dest": dest,
                               "n_dicom": _count_dicoms(dest)})
                continue
            os.makedirs(dest, exist_ok=True)
            try:
                print(f"[unzip] {zpath} ({os.path.getsize(zpath)//1024//1024} MB)")
                n_ok = n_fail = 0
                with zipfile.ZipFile(zpath, "r") as zf:
                    for info in zf.infolist():
                        try:
                            zf.extract(info, dest)
                            n_ok += 1
                        except Exception:
                            n_fail += 1
                n = _count_dicoms(dest)
                if n < 10:
                    report.append({"zip": zpath, "status": "too_few_dicom",
                                   "dest": dest, "n_dicom": n,
                                   "members_ok": n_ok, "members_fail": n_fail})
                    print(f"  → only {n} dicoms (ok={n_ok} fail={n_fail})")
                    continue
                open(marker, "w").write("ok\n")
                report.append({"zip": zpath, "status": "ok", "dest": dest,
                               "n_dicom": n, "members_ok": n_ok,
                               "members_fail": n_fail})
                print(f"  → {n} dicom-like files (ok={n_ok} fail={n_fail})")
            except Exception as e:
                report.append({"zip": zpath, "status": f"error: {e}"})
                print(f"  ERROR: {e}")
    return report


def _daykey(date: str) -> int:
    return int(date) if date.isdigit() and len(date) == 8 else 0


def process_case(
    case: str,
    case_dir: str,
    out_dir: str,
    slice_stride: int = 2,
    inplane_ds: int = 2,
    seed_override: list[float] | None = None,
) -> dict | None:
    # gather timepoint folders (including _extracted_*)
    tps = []
    for tp in sorted(os.listdir(case_dir)):
        tpp = os.path.join(case_dir, tp)
        if not os.path.isdir(tpp):
            continue
        sdir, ndcm = series_dir_with_most_dicoms(tpp)
        if ndcm < 40:
            continue
        tps.append((tp, sdir, ndcm))
    if len(tps) < 2:
        print(f"[{case}] <2 usable timepoints ({len(tps)}) — skip")
        return None

    print(f"\n[{case}] {len(tps)} timepoints")
    recs = []
    seen = set()
    for tp, sdir, ndcm in tps:
        try:
            vol, sp, meta = load_largest_series(sdir, slice_stride, inplane_ds)
            lung = segment_lung(vol, sp)
            key = (meta["study_date"], vol.shape[0])
            if key in seen:
                continue
            seen.add(key)
            recs.append({
                "tp": tp, "date": meta["study_date"], "vol": vol,
                "spacing": sp, "lung": lung, "meta": meta, "ndcm": ndcm,
                "lung_ml": float(lung.sum() * np.prod(sp) / 1000),
            })
            print(f"  {meta['study_date']:>8}  lung~{recs[-1]['lung_ml']:.0f}mL  "
                  f"({ndcm} slices)  {tp}")
        except Exception as e:
            print(f"  ERROR {tp}: {e}")

    def _tp_rank(tp: str) -> int:
        if "术前" in tp:
            return -100
        if "24h" in tp or "24H" in tp or "24小时" in tp:
            return 1
        if "30天" in tp or "30日" in tp:
            return 2
        if "3个月" in tp or "3月" in tp:
            return 3
        if "6个月" in tp or "6月" in tp:
            return 4
        return 50

    def _nominal_days(tp: str) -> int:
        """Clinical-label day offsets when StudyDate is unreliable."""
        if "术前" in tp:
            return 0
        if "24h" in tp or "24H" in tp or "24小时" in tp:
            return 1
        if "30天" in tp or "30日" in tp:
            return 30
        if "3个月" in tp or "3月" in tp:
            return 90
        if "6个月" in tp or "6月" in tp:
            return 180
        return 999

    # Order by clinical label first (术前→24h→30d→3mo→6mo), then StudyDate
    recs.sort(key=lambda r: (_tp_rank(r["tp"]), _daykey(r["date"]) or 10**9))

    if len(recs) < 2:
        return None

    if seed_override is not None:
        seed = tuple(seed_override)
    else:
        seed = detect_seed(recs[0]["vol"], recs[0]["lung"], recs[0]["spacing"])
        if seed is None:
            seed = (0.5, 0.45, 0.55)
            print("  [warn] seed fallback lung-center")
        else:
            print(f"  [seed] {tuple(round(x, 2) for x in seed)}")

    cdir = os.path.join(out_dir, case)
    os.makedirs(cdir, exist_ok=True)
    traj = []
    mask_paths = []
    d0 = _daykey(recs[0]["date"])
    for i, r in enumerate(recs):
        v, cent, mask = measure_roi(r["vol"], r["lung"], r["spacing"], seed)
        dk = _daykey(r["date"])
        # Always use clinical-label nominal days when the folder name is informative;
        # StudyDate on this cohort is frequently inconsistent with folder labels.
        if i == 0:
            days = 0
        elif _nominal_days(r["tp"]) < 900:
            days = _nominal_days(r["tp"])
        elif d0 and dk and dk >= d0:
            days = dk - d0
        else:
            days = i
        label = "pre" if i == 0 else f"post_d{days}"
        # Deduplicate same nominal day: keep the larger consolidation volume
        if traj and traj[-1][1] == int(days) and i > 0:
            if v <= traj[-1][2]:
                continue
            # replace previous
            traj.pop()
            mask_paths.pop()
        mpath = os.path.join(cdir, f"{label}_zone.npz")
        save_mask(mpath, mask, spacing_mm=float(np.mean(r["spacing"])),
                  label=f"ablation_zone_{label}")
        lpath = os.path.join(cdir, f"{label}_lung.npz")
        save_mask(lpath, r["lung"], spacing_mm=float(np.mean(r["spacing"])),
                  label="lung")
        mask_paths.append({
            "label": label, "days": int(days), "volume_mL": round(v, 3),
            "zone_mask": mpath, "lung_mask": lpath,
            "tp": r["tp"], "date": r["date"],
        })
        traj.append((label, int(days), v))
        r["lesion_ml"] = v
        r["mask"] = mask

    # Re-sort trajectory by day after dedup
    order = sorted(range(len(traj)), key=lambda i: traj[i][1])
    traj = [traj[i] for i in order]
    mask_paths = [mask_paths[i] for i in order]
    # rename pre
    if traj:
        traj[0] = ("pre", 0, traj[0][2])
        mask_paths[0]["label"] = "pre"
        mask_paths[0]["days"] = 0

    verdict = classify_response(traj)
    # preview montage
    _save_montage(os.path.join(cdir, "montage.png"), recs)

    out = {
        "case": case,
        "n_timepoints": len(traj),
        "trajectory_mL": ";".join(f"d{d}:{v:.1f}" for _, d, v in traj),
        "verdict": verdict,
        "seed_norm": list(seed),
        "masks": mask_paths,
        "pre_mask_file": mask_paths[0]["zone_mask"],
        "post_mask_file": mask_paths[-1]["zone_mask"],
    }
    json.dump(out, open(os.path.join(cdir, "meta.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"  verdict: {verdict}")
    print(f"  traj: {out['trajectory_mL']}")
    return out


def _save_montage(path, recs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(recs)
    fig, ax = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    if n == 1:
        ax = [ax]
    for i, r in enumerate(recs):
        z = int(np.argwhere(r["mask"])[:, 0].mean()) if r["mask"].any() else r["vol"].shape[0] // 2
        ax[i].imshow(r["vol"][z], cmap="gray", vmin=-1000, vmax=200)
        ax[i].contour(r["mask"][z], levels=[0.5], colors="red", linewidths=1.0)
        ax[i].set_title(f"{r['date']}\n{r['lesion_ml']:.1f} mL", fontsize=9)
        ax[i].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/ablation_followup_masks")
    ap.add_argument("--extract-zips", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--slice-stride", type=int, default=2)
    ap.add_argument("--inplane-ds", type=int, default=2)
    args = ap.parse_args(argv)

    root = find_clinical_root()
    fu = find_followup_subset(root)
    print("[followup] ", fu)
    os.makedirs(args.out, exist_ok=True)

    if args.extract_zips:
        rep = extract_zips(fu)
        json.dump(rep, open(os.path.join(args.out, "unzip_report.json"), "w"),
                  indent=2, ensure_ascii=False)
        print(f"[unzip] {sum(1 for r in rep if r.get('status')=='ok')} extracted, "
              f"report → {args.out}/unzip_report.json")

    cases = sorted([
        c for c in os.listdir(fu)
        if os.path.isdir(os.path.join(fu, c)) and c.isdigit()
    ])
    if args.cases:
        cases = [c for c in cases if c in args.cases]
    if args.limit > 0:
        cases = cases[: args.limit]

    summary = []
    for case in cases:
        rec = process_case(
            case, os.path.join(fu, case), args.out,
            slice_stride=args.slice_stride, inplane_ds=args.inplane_ds,
        )
        if rec:
            summary.append(rec)

    # write CSV compatible with existing followup_summary
    csv_path = os.path.join(args.out, "followup_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case", "n_timepoints", "trajectory_mL", "verdict"])
        w.writeheader()
        for s in summary:
            w.writerow({k: s[k] for k in w.fieldnames})
    json.dump({"n": len(summary), "cases": summary},
              open(os.path.join(args.out, "index.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    # also refresh the classic followup output if present
    classic = os.path.join("outputs", "ablation_followup", "followup_summary.csv")
    if summary:
        os.makedirs(os.path.dirname(classic), exist_ok=True)
        with open(classic, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["case", "n_timepoints", "trajectory_mL", "verdict"])
            w.writeheader()
            for s in summary:
                w.writerow({k: s[k] for k in w.fieldnames})
    print(f"[done] {len(summary)} cases → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
