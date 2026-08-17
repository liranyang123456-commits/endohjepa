"""Bootstrap image->text training pairs from structured labels.

We have no free-text captions, but several datasets carry structured annotations
(CholecSeg8k semantic masks, Cholec80 phase/tool, endovis instruments). This module
turns those into templated clinical captions so we can train/align a VLM before any
manual annotation. Later, replace/augment with clinician-written or model-refined text.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

# CholecSeg8k 13-class palette (class name per label id).
CHOLECSEG8K_CLASSES = {
    0: "black background",
    1: "abdominal wall",
    2: "liver",
    3: "gastrointestinal tract",
    4: "fat",
    5: "grasper",
    6: "connective tissue",
    7: "blood",
    8: "cystic duct",
    9: "L-hook electrocautery",
    10: "gallbladder",
    11: "hepatic vein",
    12: "liver ligament",
}
_INSTRUMENTS = {"grasper", "L-hook electrocautery"}
_IGNORE = {"black background"}


def caption_from_present_classes(present: set[str]) -> str:
    """Compose a templated clinical description from the set of visible structures."""
    anatomy = sorted(present - _INSTRUMENTS - _IGNORE)
    tools = sorted(present & _INSTRUMENTS)
    parts = []
    if anatomy:
        parts.append("Laparoscopic view showing " + ", ".join(anatomy) + ".")
    else:
        parts.append("Laparoscopic surgical view.")
    if tools:
        parts.append("Instrument(s) present: " + ", ".join(tools) + ".")
    if "blood" in present:
        parts.append("Bleeding is visible in the field.")
    return " ".join(parts)


def _present_classes_from_mask(mask_path: str) -> set[str]:
    import numpy as np
    from PIL import Image
    arr = np.asarray(Image.open(mask_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    ids = set(int(v) for v in np.unique(arr))
    return {CHOLECSEG8K_CLASSES[i] for i in ids if i in CHOLECSEG8K_CLASSES}


def build_cholecseg8k_pairs(images_dir: str, masks_dir: str, out_csv: str,
                            limit: int | None = None) -> int:
    """Pair each frame with a caption derived from its 13-class mask.

    Assumes matching filenames between images_dir and masks_dir (adjust the
    `mask_name` mapping to your export naming if needed).
    """
    images = sorted(f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg")))
    n = 0
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "caption"])
        for name in images:
            mask_name = name  # TODO: adapt if mask naming differs (e.g. *_mask.png)
            mask_path = os.path.join(masks_dir, mask_name)
            if not os.path.exists(mask_path):
                continue
            try:
                present = _present_classes_from_mask(mask_path)
            except Exception:
                continue
            caption = caption_from_present_classes(present)
            w.writerow([os.path.join(images_dir, name), caption])
            n += 1
            if limit and n >= limit:
                break
    return n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", default="manifests/caption_pairs_cholecseg8k.csv")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    count = build_cholecseg8k_pairs(args.images, args.masks, args.out, args.limit)
    print(f"wrote {count} image-caption pairs -> {args.out}")
