"""Post-ablation follow-up analysis: validate real treatment effectiveness from CT.

For each longitudinal case (pre-op + several post-op CTs), we segment the lung, detect
the dominant intrapulmonary consolidation blob (proxy for the tumour pre-op and the
ablation zone / any residual post-op), measure its volume at each timepoint, order the
timepoints by DICOM StudyDate, and classify the treatment response:

  - pre-op tumour volume  ->  early post (24h/30d): ablation zone LARGER than tumour
    (coagulation + margin)  ->  later (3-6 mo): zone should INVOLUTE/shrink.
  - a late-timepoint INCREASE vs the 30-day zone suggests residual/recurrence.

This is a semi-automated volumetric proxy (no manual annotations) and is meant to
quantify and visualise the real plan's effectiveness for radiologist review.

    python -m endoworld.ablation.followup --out outputs/ablation_followup
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from endoworld.ablation.dicom_io import load_largest_series, segment_lung


def find_clinical_root():
    KNOWN = {"World_Agent_Enoscopy", "$RECYCLE.BIN", "System Volume Information"}
    for d in os.listdir("F:\\"):
        p = "F:\\" + d
        if os.path.isdir(p) and d not in KNOWN and not d.startswith("$") and \
                ("\u04bd" in d or "\ufffd" in d or "医" in d):
            while True:
                kids = [k for k in os.listdir(p) if os.path.isdir(os.path.join(p, k))]
                files = [k for k in os.listdir(p) if os.path.isfile(os.path.join(p, k))]
                if len(kids) == 1 and not files:
                    p = os.path.join(p, kids[0]); continue
                return p
    return None


def find_followup_subset(root):
    for sub in os.listdir(root):
        p = os.path.join(root, sub)
        if os.path.isdir(p) and any(k.isdigit() and len(k) == 5 for k in os.listdir(p)):
            return p
    return None


def series_dir_with_most_dicoms(folder):
    best, best_n = None, 0
    for cur, _, files in os.walk(folder):
        n = sum(1 for f in files if "." not in f and f.upper() not in
                ("DICOMDIR", "VERSION", "LOCKFILE"))
        if n > best_n:
            best, best_n = cur, n
    return best, best_n


def _lung_bbox(lung):
    idx = np.argwhere(lung)
    return idx.min(0), idx.max(0)


def detect_seed(vol_hu, lung, spacing_zyx, lo=-150, hi=200):
    """Find a peripheral, compact soft-tissue blob on the PRE-op scan (the nodule).

    Returns normalized centroid within the lung bounding box (0..1 per axis), or None.
    """
    from scipy import ndimage
    dz, py, px = spacing_zyx
    vox_ml = dz * py * px / 1000.0
    lo_b, hi_b = _lung_bbox(lung)
    dense = lung & (vol_hu > lo) & (vol_hu < hi)
    dense = ndimage.binary_opening(dense, iterations=1)
    lbl, n = ndimage.label(dense)
    if n == 0:
        return None
    best = (0.0, None)
    edt = ndimage.distance_transform_edt(lung)   # distance to lung boundary (voxels)
    for i in range(1, n + 1):
        comp = lbl == i
        v = comp.sum() * vox_ml
        if v < 0.05 or v > 30:                     # nodule / small mass
            continue
        idx = np.argwhere(comp)
        bb = np.prod(idx.max(0) - idx.min(0) + 1)
        compactness = comp.sum() / max(bb, 1)
        cent = idx.mean(0).astype(int)
        depth = edt[tuple(cent)]                   # must sit inside parenchyma
        if depth < 1.5:                            # skip pleural rind / wall touching
            continue
        peripheral_pref = 1.0 / (1 + 0.08 * depth) # peripheral nodules preferred
        score = compactness * peripheral_pref * min(v, 6)
        if score > best[0]:
            best = (score, cent)
    if best[1] is None:
        return None
    norm = (best[1] - lo_b) / np.maximum(hi_b - lo_b, 1)
    return tuple(float(x) for x in norm)


def measure_roi(vol_hu, lung, spacing_zyx, seed_norm, radius_mm=22.0, lo=-150, hi=200):
    """Consolidation (soft-tissue) volume inside a fixed anatomical ROI (mL) + mask.

    The ROI is a sphere centred at the lung-bbox-normalized seed location, so the SAME
    region is measured across timepoints without full image registration.
    """
    dz, py, px = spacing_zyx
    vox_ml = dz * py * px / 1000.0
    lo_b, hi_b = _lung_bbox(lung)
    center = lo_b + np.array(seed_norm) * (hi_b - lo_b)
    zc, yc, xc = center
    az = (np.arange(vol_hu.shape[0]) - zc) * dz
    ay = (np.arange(vol_hu.shape[1]) - yc) * py
    ax = (np.arange(vol_hu.shape[2]) - xc) * px
    Z, Y, X = np.meshgrid(az, ay, ax, indexing="ij")
    roi = (Z**2 + Y**2 + X**2) <= radius_mm**2
    dense = roi & (vol_hu > lo) & (vol_hu < hi)    # consolidation within ROI (incl. lesion)
    vol_ml = dense.sum() * vox_ml
    return float(vol_ml), tuple(float(c) for c in center), dense


def classify_response(traj):
    """traj: list of (label, days, vol_ml) sorted by days. Return a verdict string."""
    if len(traj) < 2:
        return "insufficient timepoints"
    vols = [t[2] for t in traj]
    days = [t[1] for t in traj]
    pre = vols[0]
    post = vols[1:]
    peak = max(post)
    last = post[-1]
    max_day = max(days)
    treated = peak >= pre * 1.2                     # post-ablation zone > baseline nodule
    if treated and last <= peak * 0.7:              # clear shrinkage from the peak
        return "complete ablation (zone involuting)"
    if treated and last >= peak * 0.95 and max_day < 120:
        return "post-ablation zone persisting, short follow-up (indeterminate)"
    if last >= peak * 1.05 and max_day >= 120:
        return "possible residual/recurrence (late regrowth)"
    if treated:
        return "ablated, partial involution"
    return "ambiguous - manual review"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/ablation_followup")
    ap.add_argument("--slice-stride", type=int, default=2)
    ap.add_argument("--inplane-ds", type=int, default=2)
    ap.add_argument("--cases", nargs="*", default=None, help="restrict to case ids")
    ap.add_argument("--seed", nargs="*", default=None,
                    help="manual seeds as case:zn,yn,xn (lung-normalized), e.g. 10004:0.4,0.35,0.7")
    args = ap.parse_args()
    if args.seed:
        parsed = {}
        for s in args.seed:
            cid, coords = s.split(":")
            parsed[cid] = [float(x) for x in coords.split(",")]
        args.seed = parsed
    os.makedirs(args.out, exist_ok=True)

    root = find_clinical_root()
    fu = find_followup_subset(root)
    print("[followup] subset:", repr(os.path.basename(fu)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = []
    for case in sorted(os.listdir(fu)):
        cp = os.path.join(fu, case)
        if not os.path.isdir(cp):
            continue
        if args.cases and case not in args.cases:
            continue
        tps = []
        for tp in sorted(os.listdir(cp)):
            tpp = os.path.join(cp, tp)
            if not os.path.isdir(tpp):
                continue
            sdir, ndcm = series_dir_with_most_dicoms(tpp)
            if ndcm < 60:
                continue
            tps.append((tp, sdir, ndcm))
        if len(tps) < 2:
            print(f"[case {case}] <2 usable timepoints, skip")
            continue

        print(f"\n[case {case}] {len(tps)} timepoints")
        recs = []
        seen_dates = set()
        for tp, sdir, ndcm in tps:
            try:
                vol, sp, meta = load_largest_series(sdir, args.slice_stride, args.inplane_ds)
                lung = segment_lung(vol, sp)
                key = (meta["study_date"], vol.shape[0])
                if key in seen_dates:              # skip duplicate same-date series
                    continue
                seen_dates.add(key)
                recs.append({"tp": tp, "date": meta["study_date"], "vol": vol,
                             "spacing": sp, "lung": lung,
                             "lung_ml": float(lung.sum()*np.prod(sp)/1000), "meta": meta})
                print(f"   {meta['study_date']:>8}  lung~{recs[-1]['lung_ml']:.0f}mL  ({ndcm} slices)")
            except Exception as e:
                print(f"   ERROR {tp}: {e}")

        def daykey(r):
            d = r["date"]
            return int(d) if d.isdigit() and len(d) == 8 else 0
        recs.sort(key=daykey)
        if len(recs) < 2:
            print("   <2 distinct timepoints, skip"); continue

        # manual seed override (lung-normalized z,y,x) for cases needing review
        manual = (args.seed or {}).get(case)
        if manual is not None:
            seed = tuple(manual)
            print(f"   [seed] MANUAL lung-normalized {tuple(round(x,2) for x in seed)}")
        else:
            seed = detect_seed(recs[0]["vol"], recs[0]["lung"], recs[0]["spacing"])
            if seed is None:
                # parenchymal fallback: peripheral lung voxel in a central slice (avoid mediastinum)
                from scipy import ndimage as _nd
                lung0 = recs[0]["lung"]
                edt0 = _nd.distance_transform_edt(lung0)
                Zc = lung0.shape[0]
                band = np.zeros_like(lung0)
                band[int(0.35*Zc):int(0.65*Zc)] = True
                cand = (edt0 >= 4) & (edt0 <= 10) & band
                if cand.any():
                    lo_b, hi_b = np.argwhere(lung0).min(0), np.argwhere(lung0).max(0)
                    c = np.argwhere(cand); c = c[len(c)//2]
                    seed = tuple(((c - lo_b) / np.maximum(hi_b - lo_b, 1)).tolist())
                    print(f"   [seed] fallback parenchymal ROI {tuple(round(x,2) for x in seed)}")
                else:
                    seed = (0.5, 0.5, 0.5)
                    print("   [warn] no clear seed; lung-center ROI (unreliable)")
            else:
                print(f"   [seed] pre-op nodule at lung-normalized {tuple(round(x,2) for x in seed)}")
        for r in recs:
            v, cent, mask = measure_roi(r["vol"], r["lung"], r["spacing"], seed)
            r["lesion_ml"] = v; r["cent"] = cent; r["mask"] = mask
        d0 = daykey(recs[0])
        def days(r):
            import datetime as dt
            try:
                a = dt.date(d0//10000, (d0//100)%100, d0%100)
                b = dt.date(daykey(r)//10000, (daykey(r)//100)%100, daykey(r)%100)
                return (b - a).days
            except Exception:
                return 0
        traj = [(r["tp"], days(r), r["lesion_ml"]) for r in recs]
        verdict = classify_response(traj)
        print(f"   => 疗效判定: {verdict}")

        # plot trajectory + lesion montage
        fig, axes = plt.subplots(1, len(recs) + 1, figsize=(3.2*(len(recs)+1), 3.4))
        dd = [days(r) for r in recs]
        vv = [r["lesion_ml"] for r in recs]
        axes[0].plot(dd, vv, "-o", color="tab:red")
        axes[0].set_xlabel("days from baseline"); axes[0].set_ylabel("lesion/ablation vol (mL)")
        axes[0].set_title(f"case {case}", fontsize=10); axes[0].grid(alpha=0.3)
        for ax, r in zip(axes[1:], recs):
            if r["cent"] is not None:
                z = int(r["cent"][0]); sl = r["vol"][z]
                ax.imshow(np.clip((sl+1000)/1200, 0, 1), cmap="gray", origin="lower")
                ax.contour(r["mask"][z], levels=[0.5], colors="red", linewidths=0.8)
            ax.set_title(f"d{days(r)} {r['lesion_ml']:.1f}mL", fontsize=8); ax.axis("off")
        fig.suptitle(f"Case {case}  |  {verdict}", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, f"case_{case}.png"), dpi=110)
        plt.close(fig)

        summary.append({"case": case, "n_timepoints": len(recs),
                        "trajectory_mL": ";".join(f"d{d}:{v:.1f}" for _, d, v in traj),
                        "verdict": verdict})

    with open(os.path.join(args.out, "followup_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["case", "n_timepoints", "trajectory_mL", "verdict"])
        w.writeheader()
        for s in summary:
            w.writerow(s)
    print(f"\n[done] {len(summary)} cases -> {args.out}/followup_summary.csv + case_*.png")


if __name__ == "__main__":
    main()
