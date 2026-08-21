"""EndoVis 2017/2018 instrument masks paired with RGB frames."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_CLASSES = {
    0: "background",
    1: "Bipolar Forceps",
    2: "Prograsp Forceps",
    3: "Large Needle Driver",
    4: "Vessel Sealer",
    5: "Grasping Retractor",
    6: "Monopolar Curved Scissors",
    7: "Other",
}


def load_class_map(root: str | Path) -> dict[int, str]:
    p = Path(root) / "instrument_type_mapping.json"
    if not p.is_file():
        return dict(DEFAULT_CLASSES)
    raw = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for k, v in raw.items():
        try:
            out[int(k)] = str(v)
        except ValueError:
            continue
    return out or dict(DEFAULT_CLASSES)


def list_endovis_pairs(root: str | Path, split: str = "train") -> list[tuple[Path, Path]]:
    """(rgb, mask) pairs with identical basenames under image/ and label/."""
    root = Path(root)
    img_dir = root / split / "image"
    lab_dir = root / split / "label"
    if not img_dir.is_dir() or not lab_dir.is_dir():
        return []
    pairs = []
    for img in sorted(img_dir.iterdir()):
        lab = lab_dir / img.name
        if lab.is_file():
            pairs.append((img, lab))
    return pairs


def load_mask(path: str | Path) -> np.ndarray:
    from PIL import Image
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def instrument_binary(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.float32)


def presence_vector(mask: np.ndarray, n_classes: int = 8) -> np.ndarray:
    y = np.zeros(n_classes, dtype=np.float32)
    for i in np.unique(mask):
        ii = int(i)
        if 0 <= ii < n_classes:
            y[ii] = 1.0
    return y


def label_for_image(image_path: str | Path) -> Path | None:
    p = Path(image_path)
    cand = p.parent.parent / "label" / p.name
    return cand if cand.is_file() else None


def load_endovis_clip(pairs: list[tuple[Path, Path]], image_size: int):
    """pairs: (rgb, mask) → clip (T,C,H,W) and binary instrument mask (T,H,W)."""
    from PIL import Image
    import torch
    frames, masks = [], []
    for img, lab in pairs:
        im = Image.open(img).convert("RGB").resize((image_size, image_size))
        frames.append(np.asarray(im, np.float32) / 255.0)
        m = Image.fromarray(instrument_binary(load_mask(lab)))
        m = m.resize((image_size, image_size), Image.NEAREST)
        masks.append(np.asarray(m, np.float32))
    clip = torch.from_numpy(np.stack(frames).transpose(0, 3, 1, 2))
    inst = torch.from_numpy(np.stack(masks))
    return clip, inst


def iter_endovis_clips(root: str | Path, split: str, clip_len: int, image_size: int,
                       limit: int = 64):
    from collections import defaultdict
    pairs = list_endovis_pairs(root, split)
    by_seq: dict[str, list] = defaultdict(list)
    for img, lab in pairs:
        by_seq[img.name.split("_frame")[0]].append((img, lab))
    n = 0
    for seq, items in by_seq.items():
        items = sorted(items, key=lambda x: x[0].name)
        for start in range(0, len(items) - clip_len + 1, clip_len):
            clip, inst = load_endovis_clip(items[start:start + clip_len], image_size)
            yield seq, clip, inst
            n += 1
            if n >= limit:
                return
