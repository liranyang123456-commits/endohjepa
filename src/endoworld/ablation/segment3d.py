"""Cohort A: ION CT → semi-automatic 3D lung / nodule / vessel masks.

Upgrades tabular nodule geometry to patient-specific voxel masks so that
``术前3D → 仿真 burns → 合成术后区`` can run on real anatomy.

Pipeline per case
-----------------
1. Locate CT folder (or extract CT.zip once into a cache).
2. Load largest axial series (downsampled for speed).
3. Segment lung (existing ``segment_lung``).
4. Segment vessels (high-HU inside lung).
5. Place / refine nodule ellipsoid from ``nodule_params.csv`` diameters,
   seeded by automatic soft-tissue blob detection when possible.
6. Export NPZ masks + JSON meta + mid-slice PNG preview.

    PYTHONPATH=src python -m endoworld.ablation.segment3d --limit 3 \\
        --out outputs/ablation_seg3d
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import zipfile
from typing import Any

import numpy as np

from endoworld.ablation.dicom_io import load_largest_series, segment_lung, segment_vessels
from endoworld.ablation.followup import find_clinical_root
from endoworld.ablation.trajectory_schema import save_mask


def ion_root(clinical_root: str | None = None) -> str:
    root = clinical_root or find_clinical_root()
    if root is None:
        raise RuntimeError("clinical root not found")
    p = os.path.join(root, "ION患者整理")
    if not os.path.isdir(p):
        raise RuntimeError(f"ION folder missing under {root}")
    return p


def list_ion_cases(ion: str) -> list[tuple[str, str]]:
    """Return [(case_id '001', folder_name), ...] sorted."""
    out = []
    for name in sorted(os.listdir(ion)):
        p = os.path.join(ion, name)
        if not os.path.isdir(p):
            continue
        # folder names like '001王继深'
        digits = "".join(ch for ch in name[:3] if ch.isdigit())
        if len(digits) >= 2:
            out.append((digits.zfill(3), name))
    return out


def resolve_ct_dir(case_dir: str, cache_root: str) -> str | None:
    """Prefer extracted CT/; else unzip CT.zip into cache once."""
    ct_dir = os.path.join(case_dir, "CT")
    if os.path.isdir(ct_dir) and any(os.scandir(ct_dir)):
        return ct_dir
    zpath = os.path.join(case_dir, "CT.zip")
    if not os.path.isfile(zpath):
        return None
    case_name = os.path.basename(case_dir)
    dest = os.path.join(cache_root, case_name, "CT")
    marker = os.path.join(dest, ".extracted_ok")
    if os.path.isfile(marker):
        return dest
    os.makedirs(dest, exist_ok=True)
    print(f"  [unzip] {zpath} → {dest}")
    try:
        with zipfile.ZipFile(zpath, "r") as zf:
            for info in zf.infolist():
                try:
                    zf.extract(info, dest)
                except Exception as e:
                    print(f"    skip member {info.filename}: {e}")
    except zipfile.BadZipFile as e:
        print(f"  [unzip] BadZipFile {zpath}: {e}")
        return None
    # accept partial extract if we got any files
    n = sum(1 for _ in os.scandir(dest))
    if n < 2:
        print(f"  [unzip] too few files after extract ({n})")
        return None
    open(marker, "w").write("ok\n")
    return dest


def load_params_by_note() -> dict[str, dict]:
    """Map note_XXX → row; also map 001-index via note number."""
    path = os.path.join("manifests", "nodule_params.csv")
    if not os.path.isfile(path):
        return {}
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    by_note = {}
    for r in rows:
        note = (r.get("note") or "").replace(".docx.txt", "").replace(".txt", "")
        by_note[note] = r
    return by_note


def note_for_case_id(case_id: str) -> str:
    # case_id '001' → note_000
    try:
        n = int(case_id) - 1
        return f"note_{n:03d}"
    except ValueError:
        return case_id


def _f(row: dict | None, *keys) -> float | None:
    if not row:
        return None
    for k in keys:
        v = row.get(k)
        if v is None or v == "":
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def detect_nodule_centroid(vol_hu, lung, spacing_zyx, max_vol_ml=25.0):
    """Return voxel centroid of most nodule-like soft-tissue blob, or None."""
    from scipy import ndimage
    dz, py, px = spacing_zyx
    vox_ml = dz * py * px / 1000.0
    dense = lung & (vol_hu > -150) & (vol_hu < 200)
    dense = ndimage.binary_opening(dense, iterations=1)
    lbl, n = ndimage.label(dense)
    if n == 0:
        return None
    edt = ndimage.distance_transform_edt(lung)
    best = (0.0, None)
    for i in range(1, n + 1):
        comp = lbl == i
        v = comp.sum() * vox_ml
        if v < 0.05 or v > max_vol_ml:
            continue
        idx = np.argwhere(comp)
        bb = np.prod(idx.max(0) - idx.min(0) + 1)
        compactness = comp.sum() / max(bb, 1)
        cent = idx.mean(0)
        depth = edt[tuple(cent.astype(int))]
        if depth < 1.0:
            continue
        score = compactness * min(v, 8.0) * (1.0 / (1 + 0.05 * depth))
        if score > best[0]:
            best = (score, cent)
    return None if best[1] is None else best[1]


def ellipsoid_mask(shape, spacing_zyx, center_zyx, semi_axes_zyx_mm):
    """Boolean ellipsoid on a (Z,Y,X) grid."""
    zc, yc, xc = center_zyx
    az, ay, ax = semi_axes_zyx_mm
    dz, py, px = spacing_zyx
    zz = (np.arange(shape[0]) - zc) * dz
    yy = (np.arange(shape[1]) - yc) * py
    xx = (np.arange(shape[2]) - xc) * px
    Z, Y, X = np.meshgrid(zz, yy, xx, indexing="ij")
    a, b, c = max(az, 1e-3), max(ay, 1e-3), max(ax, 1e-3)
    return (Z ** 2 / a ** 2 + Y ** 2 / b ** 2 + X ** 2 / c ** 2) <= 1.0


def build_nodule_mask(vol_hu, lung, spacing_zyx, row: dict | None):
    """Combine auto seed + record diameters into a nodule (+ optional margin) mask."""
    shape = vol_hu.shape
    dz, py, px = spacing_zyx
    cent = detect_nodule_centroid(vol_hu, lung, spacing_zyx)
    if cent is None:
        # lung centroid fallback
        idx = np.argwhere(lung)
        cent = idx.mean(0) if len(idx) else np.array(shape) / 2.0

    # diameters from record (full diameters → semi-axes); map to Z,Y,X ≈ SI, AP/cor, axial/LR
    si = _f(row, "size_SI_mm", "diam_sagittal_mm") or 16.0
    ap = _f(row, "size_AP_mm", "diam_coronal_mm") or 16.0
    lr = _f(row, "size_LR_mm", "diam_axial_mm") or 16.0
    # semi-axes mm in (z,y,x)
    semi = (si / 2.0, ap / 2.0, lr / 2.0)
    nodule = ellipsoid_mask(shape, spacing_zyx, cent, semi) & lung
    # if ellipsoid empty (seed outside lung), drop lung constraint
    if not nodule.any():
        nodule = ellipsoid_mask(shape, spacing_zyx, cent, semi)
    margin_mm = 5.0
    target = ellipsoid_mask(
        shape, spacing_zyx, cent,
        (semi[0] + margin_mm, semi[1] + margin_mm, semi[2] + margin_mm),
    )
    return nodule, target, cent, semi


def save_preview(path, vol, lung, nodule, vessel, z=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if z is None:
        if nodule.any():
            z = int(np.argwhere(nodule)[:, 0].mean())
        else:
            z = vol.shape[0] // 2
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    sl = vol[z]
    ax[0].imshow(sl, cmap="gray", vmin=-1000, vmax=200)
    ax[0].contour(lung[z], levels=[0.5], colors="cyan", linewidths=0.8)
    ax[0].set_title("lung")
    ax[1].imshow(sl, cmap="gray", vmin=-1000, vmax=200)
    ax[1].contour(nodule[z], levels=[0.5], colors="red", linewidths=1.2)
    ax[1].set_title("nodule")
    ax[2].imshow(sl, cmap="gray", vmin=-1000, vmax=200)
    if vessel.any():
        ax[2].contour(vessel[z], levels=[0.5], colors="yellow", linewidths=0.6)
    ax[2].set_title("vessels")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def process_case(
    case_id: str,
    folder_name: str,
    ion: str,
    out_dir: str,
    cache_root: str,
    row: dict | None,
    slice_stride: int = 2,
    inplane_ds: int = 2,
) -> dict | None:
    case_dir = os.path.join(ion, folder_name)
    ct = resolve_ct_dir(case_dir, cache_root)
    if ct is None:
        print(f"[{case_id}] no CT / CT.zip — skip")
        return None
    print(f"[{case_id}] loading {ct}")
    try:
        vol, sp, meta = load_largest_series(ct, slice_stride, inplane_ds)
    except Exception as e:
        print(f"[{case_id}] load failed: {e}")
        return None
    lung = segment_lung(vol, sp)
    vessel = segment_vessels(vol, lung)
    nodule, target, cent, semi = build_nodule_mask(vol, lung, sp, row)

    cdir = os.path.join(out_dir, case_id)
    os.makedirs(cdir, exist_ok=True)
    # spacing for trajectory_schema is isotropic-ish; store full zyx in meta
    save_mask(os.path.join(cdir, "lung.npz"), lung, spacing_mm=float(np.mean(sp)),
              label="lung")
    save_mask(os.path.join(cdir, "nodule.npz"), nodule, spacing_mm=float(np.mean(sp)),
              label="nodule")
    save_mask(os.path.join(cdir, "target.npz"), target, spacing_mm=float(np.mean(sp)),
              label="tumor_plus_margin")
    save_mask(os.path.join(cdir, "vessel.npz"), vessel, spacing_mm=float(np.mean(sp)),
              label="vessel")
    # compact volume stats (not full HU to save disk)
    vox_ml = float(np.prod(sp) / 1000.0)
    rec = {
        "case_id": case_id,
        "folder": folder_name,
        "note": note_for_case_id(case_id),
        "shape": list(vol.shape),
        "spacing_zyx_mm": [float(x) for x in sp],
        "series": meta,
        "centroid_zyx_vox": [float(x) for x in cent],
        "nodule_semi_axes_zyx_mm": [float(x) for x in semi],
        "volumes_mL": {
            "lung": round(float(lung.sum()) * vox_ml, 1),
            "nodule": round(float(nodule.sum()) * vox_ml, 3),
            "target": round(float(target.sum()) * vox_ml, 3),
            "vessel": round(float(vessel.sum()) * vox_ml, 2),
        },
        "masks": {
            "lung": os.path.join(cdir, "lung.npz"),
            "nodule": os.path.join(cdir, "nodule.npz"),
            "target": os.path.join(cdir, "target.npz"),
            "vessel": os.path.join(cdir, "vessel.npz"),
        },
        "record": {
            k: row.get(k) for k in (
                "lobe", "bronchial_segment", "airway_generation",
                "dist_pleura_mm", "dist_chestwall_mm", "dist_vessel_mm",
                "solidity", "malignancy_pct", "n_planned_paths",
            )
        } if row else {},
    }
    json.dump(rec, open(os.path.join(cdir, "meta.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    save_preview(os.path.join(cdir, "preview.png"), vol, lung, nodule, vessel)
    print(f"  lung={rec['volumes_mL']['lung']}mL  nodule={rec['volumes_mL']['nodule']}mL  "
          f"vessel={rec['volumes_mL']['vessel']}mL")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ION CT → 3D lung/nodule/vessel masks")
    ap.add_argument("--out", default="outputs/ablation_seg3d")
    ap.add_argument("--cache", default="outputs/ct_cache_ion")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cases", nargs="*", default=None, help="e.g. 001 002")
    ap.add_argument("--slice-stride", type=int, default=2)
    ap.add_argument("--inplane-ds", type=int, default=2)
    args = ap.parse_args(argv)

    ion = ion_root()
    cases = list_ion_cases(ion)
    if args.cases:
        want = set(c.zfill(3) for c in args.cases)
        cases = [c for c in cases if c[0] in want]
    if args.limit > 0:
        cases = cases[: args.limit]

    by_note = load_params_by_note()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)

    index = []
    # merge with existing index so incremental --cases runs don't wipe prior results
    idx_path = os.path.join(args.out, "index.json")
    if os.path.isfile(idx_path):
        try:
            prev = json.load(open(idx_path, encoding="utf-8"))
            index = list(prev.get("cases") or [])
        except Exception:
            index = []
    by_id = {r["case_id"]: i for i, r in enumerate(index)}

    for cid, folder in cases:
        row = by_note.get(note_for_case_id(cid))
        rec = process_case(
            cid, folder, ion, args.out, args.cache, row,
            slice_stride=args.slice_stride, inplane_ds=args.inplane_ds,
        )
        if rec:
            if cid in by_id:
                index[by_id[cid]] = rec
            else:
                by_id[cid] = len(index)
                index.append(rec)

    summary = {
        "n": len(index),
        "out": args.out,
        "mean_nodule_mL": round(float(np.mean([
            r["volumes_mL"]["nodule"] for r in index])), 3) if index else None,
        "cases": index,
    }
    json.dump(summary, open(idx_path, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False)
    print(f"[done] {len(index)} cases → {idx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
