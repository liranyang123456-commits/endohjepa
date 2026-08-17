"""Load a CT DICOM series into a HU volume, and basic lung/vessel segmentation.

The provincial-hospital CTs are stored as extension-less DICOM files under DICOMDIR
trees. We recursively read files, group by SeriesInstanceUID, take the largest axial
series, sort by slice position, and convert to Hounsfield Units.
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np


def _iter_dicoms(folder):
    import pydicom
    for cur, _, files in os.walk(folder):
        for f in files:
            if f.upper() in ("DICOMDIR", "VERSION", "LOCKFILE"):
                continue
            p = os.path.join(cur, f)
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
                if hasattr(ds, "SeriesInstanceUID") and hasattr(ds, "Rows"):
                    yield p, ds
            except Exception:
                continue


def load_largest_series(folder, slice_stride: int = 1, inplane_ds: int = 1):
    """Return (volume_HU [Z,Y,X], spacing_zyx_mm, meta).

    slice_stride: read every k-th slice; inplane_ds: in-plane downsample factor.
    Both speed up loading of large chest CTs for volumetric follow-up analysis.
    """
    import pydicom
    series = defaultdict(list)
    for p, ds in _iter_dicoms(folder):
        series[ds.SeriesInstanceUID].append((p, ds))
    if not series:
        raise RuntimeError(f"no DICOM series under {folder}")
    uid = max(series, key=lambda k: len(series[k]))
    items = series[uid]

    def zpos(ds):
        if hasattr(ds, "ImagePositionPatient"):
            return float(ds.ImagePositionPatient[2])
        return float(getattr(ds, "InstanceNumber", 0))
    items.sort(key=lambda t: zpos(t[1]))
    sel = items[::slice_stride]

    slices = []
    for p, ds in sel:
        full = pydicom.dcmread(p, force=True)
        arr = full.pixel_array.astype(np.float32)
        slope = float(getattr(full, "RescaleSlope", 1.0))
        inter = float(getattr(full, "RescaleIntercept", 0.0))
        arr = arr * slope + inter
        if inplane_ds > 1:
            arr = arr[::inplane_ds, ::inplane_ds]
        slices.append(arr)
    vol = np.stack(slices, axis=0)

    ref = items[len(items) // 2][1]
    py, px = [float(v) for v in getattr(ref, "PixelSpacing", (1.0, 1.0))]
    py *= inplane_ds; px *= inplane_ds
    if len(items) > 1:
        z0 = zpos(items[0][1]); z1 = zpos(items[-1][1])
        dz = abs(z1 - z0) / (len(items) - 1) or float(getattr(ref, "SliceThickness", 1.0))
    else:
        dz = float(getattr(ref, "SliceThickness", 1.0))
    dz *= slice_stride
    meta = {"series_uid": uid, "n_slices": len(sel), "shape": vol.shape,
            "spacing_zyx": (dz, py, px),
            "series_desc": str(getattr(ref, "SeriesDescription", "")),
            "study_date": str(getattr(ref, "StudyDate", "") or getattr(ref, "AcquisitionDate", "")),
            "n_series_slices": len(items)}
    return vol, (dz, py, px), meta


def segment_lung(vol_hu, spacing_zyx, min_vol_ml=80.0):
    """Lung mask: body (tissue) region, then interior air, excluding outside air.

    Steps: 1) body = largest filled tissue component (>-500 HU);  2) lung = air
    (<-320 HU) INSIDE the body;  3) keep interior air components (drop trachea-only /
    tiny ones by volume);  4) morphological cleanup.
    """
    from scipy import ndimage
    dz, py, px = spacing_zyx
    vox_ml = dz * py * px / 1000.0

    tissue = vol_hu > -500
    # per-axial-slice body fill: robust to lungs venting to the volume border via
    # the trachea/airways (3D fill_holes fails there; 2D in-plane fill does not).
    body = np.zeros_like(tissue)
    for z in range(tissue.shape[0]):
        sl = tissue[z]
        if not sl.any():
            continue
        lbl_s, ns = ndimage.label(sl)
        if ns == 0:
            continue
        szs = np.bincount(lbl_s.ravel()); szs[0] = 0
        biggest = lbl_s == int(np.argmax(szs))    # torso cross-section
        body[z] = ndimage.binary_fill_holes(biggest)

    air_inside = body & (vol_hu < -320)
    lbl2, n2 = ndimage.label(air_inside)
    sizes2 = np.bincount(lbl2.ravel())
    keep = np.where(sizes2 * vox_ml >= min_vol_ml)[0]
    keep = keep[keep != 0]
    lung = np.isin(lbl2, keep)
    lung = ndimage.binary_closing(lung, iterations=1)
    lung = ndimage.binary_fill_holes(lung)
    return lung


def segment_vessels(vol_hu, lung_mask, hu_thresh=-200):
    """High-HU structures inside the lung ~ vessels/soft tissue (heat-sink risk)."""
    return lung_mask & (vol_hu > hu_thresh)
