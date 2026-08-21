"""Build a real-image atlas covering every dataset in the local census."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_names import display  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "manifests" / "sequences.csv"
OUT = Path(__file__).resolve().parent / "figures"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
REJECT = (
    "depth", "mask", "label", "semantic", "disparity", "normal",
    "flow", "occlusion", "visualization", "trajectory", "plot",
)

ROLES = {
    "C3VD": "pose",
    "Cholec80-Boxes": "forecast",
    "CholecSeg8k": "semantic",
    "CholecT50": "forecast + recognition",
    "EndoNeRF": "RGB-D",
    "EndoVis_InstrumentTracking": "tracking",
    "HyperKvasir": "forecast",
    "ION_bronch": "forecast (private)",
    "Kvasir-Capsule": "forecast",
    "Kvasir-Instrument": "instrument",
    "MIS_own": "forecast",
    "SCARED": "pose + depth",
    "STIR": "deformation",
    "Stereo_Lap": "RGB-D forecast",
    "SurgT": "tracking video",
    "TrackVes": "tracking video",
    "endoscapes": "semantic/CVS",
    "endovis2017_full": "mask + forecast",
    "endovis2018_full": "mask + forecast",
}
DOMAIN_COLORS = {"gi": "#4D8B8B", "laparo": "#557FA8", "bronch": "#9A7A55"}


def _video_frame(path: Path) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(count // 2, 0))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _image(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"))
    except Exception:
        return None
    # Common vertically stacked stereo layout.
    if array.shape[0] >= 1.5 * array.shape[1]:
        array = array[:array.shape[0] // 2]
    return array


def _candidates(row: dict[str, str]) -> list[Path]:
    sample = Path(row["sample_frame"])
    directory = Path(row["frames_dir"])
    paths: list[Path] = []
    if directory.exists():
        files = [
            path for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES
            and not any(token in path.name.lower() for token in REJECT)
        ]
        files.sort()
        if files:
            positions = (0, len(files) // 4, len(files) // 2,
                         3 * len(files) // 4, len(files) - 1)
            paths.extend(files[position] for position in positions)
    if not any(token in sample.name.lower() for token in REJECT):
        paths.append(sample)
    return list(dict.fromkeys(paths))


def _representative(rows: list[dict[str, str]]) -> tuple[np.ndarray, dict[str, str]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            row["split"] not in {"test", "val"},
            -max(int(row["num_frames"]), 0),
        ),
    )
    best = None
    for row in ordered[:4]:
        for path in _candidates(row):
            if not path.exists():
                continue
            image = _video_frame(path) if path.suffix.lower() in VIDEO_SUFFIXES else _image(path)
            if image is not None and image.size:
                small = cv2.resize(image, (96, 96)).astype(np.float32)
                gray = small.mean(axis=-1)
                colorfulness = small.std(axis=-1).mean()
                score = (
                    3.0 * float(colorfulness)
                    + 0.3 * float(gray.std())
                    + 15.0 * float((gray > 12).mean())
                )
                if best is None or score > best[0]:
                    best = (score, image, row)
    if best is not None:
        return best[1], best[2]
    raise RuntimeError(f"no readable visual asset for {rows[0]['dataset']}")


def main():
    by_dataset: dict[str, list[dict[str, str]]] = {}
    with MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            by_dataset.setdefault(row["dataset"], []).append(row)
    names = sorted(by_dataset, key=lambda name: (by_dataset[name][0]["domain"], name.lower()))
    if set(names) != set(ROLES):
        raise RuntimeError(f"atlas role map mismatch: {set(names) ^ set(ROLES)}")

    fig, axes = plt.subplots(4, 5, figsize=(14.2, 10.0))
    for axis, name in zip(axes.flat, names):
        image, row = _representative(by_dataset[name])
        axis.imshow(image)
        axis.axis("off")
        domain = row["domain"]
        axis.set_title(display(name), fontsize=10, fontweight="bold", pad=5)
        axis.text(
            0.02, 0.03, f"{domain} · {ROLES[name]}",
            transform=axis.transAxes, fontsize=7.5, color="white",
            bbox={"facecolor": DOMAIN_COLORS[domain], "edgecolor": "none",
                  "alpha": 0.88, "pad": 2.5},
        )
    summary_axis = axes.flat[len(names)]
    summary_axis.set_facecolor("#F4F6F7")
    summary_axis.set_xticks([])
    summary_axis.set_yticks([])
    for spine in summary_axis.spines.values():
        spine.set_color("#C7CDD2")
        spine.set_linewidth(1.0)
    summary_axis.set_title("Coverage summary", fontsize=10, fontweight="bold", pad=5)
    summary_axis.text(
        0.5, 0.63, "19 datasets\n1,707 sequences\n1.07M decoded frames",
        ha="center", va="center", fontsize=11, fontweight="bold",
        color="#303942", linespacing=1.45,
    )
    summary_axis.text(
        0.5, 0.20, "GI  4  ·  Laparoscopy  14  ·  Bronchoscopy  1",
        ha="center", va="center", fontsize=7.5, color="#52606B",
    )
    fig.suptitle(
        "Local endoscopy corpus atlas: one real representative image per dataset",
        fontsize=13, fontweight="bold",
    )
    fig.text(
        0.5, 0.012,
        "Images document visual coverage; they are not model outputs. "
        "ION cases are anonymised and no identifiers are shown.",
        ha="center", fontsize=8.5, color="#38434F",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.965))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figure9_dataset_atlas.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "figure9_dataset_atlas.png", dpi=220, bbox_inches="tight", facecolor="white")
    print(f"[atlas] wrote {len(names)} datasets")


if __name__ == "__main__":
    main()
