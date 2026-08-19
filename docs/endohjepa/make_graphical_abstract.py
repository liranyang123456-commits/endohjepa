"""Create the standalone, landscape Endo-HJEPA graphical abstract.

This is intentionally a compact conceptual summary rather than a copy of the
multi-row Figure 1 pipeline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
DPI = 500

INK = "#24313A"
MUTED = "#52606B"
BLUE = "#477DA8"
TEAL = "#34847E"
PURPLE = "#77628C"
ORANGE = "#B87333"
SOFT_BLUE = "#EDF4F9"
SOFT_TEAL = "#ECF6F4"
SOFT_PURPLE = "#F3EFF7"
SOFT_ORANGE = "#FBF2E8"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _box(ax, x, y, w, h, face, edge, title, body=None, dashed=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.5,
            linestyle="--" if dashed else "-",
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.035,
        title,
        ha="center",
        va="top",
        fontsize=8.7,
        fontweight="bold",
        color=INK,
        zorder=4,
        linespacing=1.05,
    )
    if body:
        ax.text(
            x + w / 2,
            y + 0.035,
            body,
            ha="center",
            va="bottom",
            fontsize=7.1,
            color=MUTED,
            linespacing=1.15,
            zorder=4,
        )


def _arrow(ax, start, end, colour=INK, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.5,
            color=colour,
            linestyle="--" if dashed else "-",
            zorder=6,
        )
    )


def _thumb(ax, path: Path, cx: float, cy: float, w: float, h: float):
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"))
    image_h, image_w = array.shape[:2]
    scale = min(w / image_w, h / image_h)
    draw_w, draw_h = image_w * scale, image_h * scale
    left, bottom = cx - draw_w / 2, cy - draw_h / 2
    ax.imshow(array, extent=(left, left + draw_w, bottom, bottom + draw_h), zorder=3)
    ax.add_patch(
        Rectangle(
            (left, bottom),
            draw_w,
            draw_h,
            facecolor="none",
            edgecolor=ORANGE,
            linewidth=1.2,
            zorder=4,
        )
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    thumbnails = [
        ("_fig1_in_laparo.png", "Laparo"),
        ("_fig1_in_gi.png", "GI"),
        ("_fig1_in_bronch.png", "Bronch"),
    ]
    for name, _ in thumbnails:
        if not (OUT / name).is_file():
            raise FileNotFoundError(
                f"missing figures/{name}; run make_result_thumbs.py first"
            )

    figure, ax = plt.subplots(figsize=(12.5, 5.0))
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        0.04, 0.95, "Endo-HJEPA", fontsize=18, fontweight="bold", color=INK, va="top"
    )
    ax.text(
        0.04,
        0.875,
        "Frozen video representations across three endoscopic settings",
        fontsize=10.5,
        color=MUTED,
        va="top",
    )

    _box(ax, 0.04, 0.27, 0.53, 0.50, SOFT_ORANGE, ORANGE, "Observed endoscopic video")
    thumb_y = 0.54
    for index, (name, label) in enumerate(thumbnails):
        cx = 0.14 + 0.165 * index
        _thumb(ax, OUT / name, cx, thumb_y, 0.145, 0.19)
        ax.text(
            cx,
            0.405,
            label,
            ha="center",
            va="top",
            fontsize=7.2,
            fontweight="bold",
            color=INK,
        )
    ax.text(
        0.305,
        0.325,
        "real frames · no pixel synthesis",
        ha="center",
        va="center",
        fontsize=9,
        color=MUTED,
    )

    _box(
        ax,
        0.69,
        0.28,
        0.45,
        0.49,
        SOFT_BLUE,
        BLUE,
        "Frozen V-JEPA 2\nencoder",
        "same frozen weights\nprotocol-specific token caches",
    )
    for row in range(3):
        for column in range(7):
            ax.add_patch(
                Rectangle(
                    (0.77 + 0.043 * column, 0.49 + 0.045 * row),
                    0.032,
                    0.028,
                    facecolor="#C9DDEA",
                    edgecolor=BLUE,
                    linewidth=0.65,
                    zorder=4,
                )
            )
    ax.text(
        0.915,
        0.445,
        "pooled spatiotemporal tokens",
        fontsize=7.0,
        color=BLUE,
        ha="center",
        fontweight="bold",
    )
    _arrow(ax, (0.58, 0.525), (0.68, 0.525), BLUE)

    _box(
        ax,
        1.29,
        0.52,
        0.52,
        0.25,
        SOFT_TEAL,
        TEAL,
        "Forecast future\nrepresentations",
        "validated L1 residual forecast;\nL2 implemented, not ablated",
    )
    _arrow(ax, (1.15, 0.60), (1.28, 0.645), TEAL)

    _box(
        ax,
        1.29,
        0.27,
        0.52,
        0.19,
        SOFT_PURPLE,
        PURPLE,
        "Audited physical\nassociation",
        "SE(3)-conditioned dynamics\nmatched-negative evaluation",
    )
    _arrow(ax, (1.15, 0.43), (1.28, 0.37), PURPLE)

    _box(
        ax,
        1.94,
        0.52,
        0.50,
        0.25,
        "#F7FAF8",
        TEAL,
        "Latent forecast\nevidence",
        "mean steps 1--4: 0.978 vs 0.916\nretrieval visualisation, not generation",
    )
    _arrow(ax, (1.82, 0.645), (1.93, 0.645), TEAL)

    _box(
        ax,
        1.94,
        0.27,
        0.50,
        0.19,
        "#F8F5FA",
        PURPLE,
        "Action association\nevidence",
        "87.0% deranged batch\n91.3% fixed-bank pair",
    )
    _arrow(ax, (1.82, 0.37), (1.93, 0.37), PURPLE)

    _box(
        ax,
        1.94,
        0.07,
        0.50,
        0.13,
        "#FAFAFA",
        "#888888",
        "Risk checkpoint inactive",
        dashed=True,
    )
    ax.text(
        2.19,
        0.095,
        "AUC 0.523 < 0.75 gate",
        fontsize=7.1,
        color=MUTED,
        ha="center",
        va="center",
        zorder=5,
    )
    _arrow(ax, (1.55, 0.275), (1.94, 0.145), "#777777", dashed=True)

    ax.text(
        0.04,
        0.10,
        "Predict representations, not pixels.",
        fontsize=12,
        fontweight="bold",
        color=INK,
        va="center",
    )
    ax.text(
        0.04,
        0.045,
        "Risk filtering and closed-loop navigation are not validated.",
        fontsize=9.5,
        color=MUTED,
        va="center",
    )

    png_path = OUT / "graphical_abstract.png"
    pdf_path = OUT / "graphical_abstract.pdf"
    figure.savefig(png_path, dpi=DPI, facecolor="white")
    figure.savefig(pdf_path, dpi=DPI, facecolor="white")
    plt.close(figure)

    with Image.open(png_path) as image:
        pixel_size = list(image.size)
        png_dpi = [float(value) for value in image.info.get("dpi", (DPI, DPI))]
    provenance = {
        "figure": ["figures/graphical_abstract.pdf", "figures/graphical_abstract.png"],
        "design": "standalone simplified landscape conceptual summary; "
        "not a copy of Figure 1",
        "render_dpi": DPI,
        "pixel_size": pixel_size,
        "minimum_declared_font_pt": 8.5,
        "thumbnail_assets": [
            {
                "file": f"figures/{name}",
                "sha256": _sha256(OUT / name),
                "label": label,
            }
            for name, label in thumbnails
        ],
        "png_embedded_dpi": png_dpi,
    }
    (HERE / "graphical_abstract_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"[graphical abstract] wrote {pixel_size[0]}x{pixel_size[1]} at {DPI} dpi")


if __name__ == "__main__":
    main()
