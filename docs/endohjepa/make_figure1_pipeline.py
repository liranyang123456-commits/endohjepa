"""Figure 1: Endo-HJEPA overview — thumbnails, glyphs, result images.

Columns: real orifice frames -> encoder -> module glyphs -> capabilities ->
clinical pathways -> result images. All arrows are orthogonal (elbow
connectors); boxes tightly wrap their text. Bottom strip: four audit gates.

    python docs/endohjepa/make_figure1_pipeline.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle
from PIL import Image

OUT = Path(__file__).resolve().parent / "figures"
ROOT = Path(__file__).resolve().parents[2]

INK = "#2D3741"
LINE = "#9AA4AE"
C_ENC = "#4E7FA6"
C_FC = "#4E8D8D"
C_PHYS = "#8A7AA0"
C_APP = "#3F7A5A"
C_AUD = "#8C6D4F"
SOFT_ENC = "#EFF5FA"
SOFT_FC = "#EFF8F8"
SOFT_PHYS = "#F5F3F8"
SOFT_APP = "#EFF7F2"
SOFT_AUD = "#F8F4EF"
WHITE = "#FFFFFF"

FRAMES = {
    "Laparoscopy": ROOT / "datasets/CholecT50/CholecT50/videos/VID04/000000.png",
    "GI endoscopy": ROOT / "datasets/HyperKvasir/hyper-kvasir-labeled-images/labeled-images/lower-gi-tract/pathological-findings/ulcerative-colitis-grade-1-2/00064260-95ca-47fc-9103-2b526f59fada.jpg",
    "Bronchoscopy": ROOT / "datasets/ION_bronch/case_001/intraop_00/Video_screenshot_Henan1/ion_screenshot-20211026-035827_439.png",
}
RESULTS = [
    ("Forecast retrieval", OUT / "_fig1_forecast_ret.png"),
    ("Action-conditioned", OUT / "_fig1_se3.png"),
    ("Navigation rollout", OUT / "_fig1_nav.png"),
]


def box(ax, x, y, w, h, fc, ec, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.03",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))


def elbow(ax, x1, y1, x2, y2, text=None):
    """Strictly orthogonal connector: horizontal, vertical, horizontal."""
    xm = (x1 + x2) / 2
    ax.plot([x1, xm, xm, x2], [y1, y1, y2, y2], color=LINE, lw=1.3, zorder=1)
    ax.annotate("", xy=(x2, y2), xytext=(x2 - 0.02, y2),
                arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.3,
                                mutation_scale=11), zorder=1)
    if text:
        ax.text(xm, max(y1, y2) + 0.10, text, ha="center", fontsize=7.4,
                color="#4A5560", style="italic")


def harrow(ax, x1, x2, y, text=None):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.3,
                                mutation_scale=11), zorder=1)
    if text:
        ax.text((x1 + x2) / 2, y + 0.10, text, ha="center", fontsize=7.4,
                color="#4A5560", style="italic")


def thumb(ax, path, x, y, w, h, ec):
    img = np.asarray(Image.open(path).convert("RGB"))
    ax.imshow(img, extent=[x, x + w, y, y + h], aspect="auto", zorder=3)
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=ec,
                               linewidth=1.2, zorder=4))


def glyph_forecast(ax, x, y, w, h):
    t = np.linspace(0, 1, 20)
    ax.plot([x, x + w * 0.42], [y + h * 0.32, y + h * 0.32], color=INK, lw=1.4, zorder=5)
    ax.plot(x + w * (0.42 + 0.58 * t), y + h * (0.32 + 0.45 * t), color=C_FC, lw=1.6, zorder=5)
    ax.plot(x + w * (0.42 + 0.58 * t), y + h * np.full_like(t, 0.32), color=LINE,
            lw=1.2, ls="--", zorder=5)
    ax.scatter(x + w * np.linspace(0.05, 0.38, 4), np.full(4, y + h * 0.32),
               s=8, color=INK, zorder=6)
    ax.scatter([x + w * 0.95], [y + h * 0.75], s=16, color=C_FC, zorder=6)


def glyph_se3(ax, x, y, w, h):
    cx, cy = x + w * 0.32, y + h * 0.30
    ax.annotate("", xy=(cx + w * 0.42, cy), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color="#C24D4D", lw=1.6), zorder=5)
    ax.annotate("", xy=(cx, cy + h * 0.5), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color="#3F7A5A", lw=1.6), zorder=5)
    ax.annotate("", xy=(cx + w * 0.26, cy + h * 0.30), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color="#4E7FA6", lw=1.6), zorder=5)


def glyph_navigation(ax, x, y, w, h):
    t = np.linspace(0, 1, 30)
    px = x + w * (0.08 + 0.84 * t)
    py = y + h * (0.25 + 0.45 * np.sin(t * np.pi) + 0.1 * t)
    ax.plot(px, py, color=C_PHYS, lw=1.6, zorder=5)
    ax.scatter([px[0]], [py[0]], s=20, color=INK, zorder=6)
    ax.scatter([px[-1]], [py[-1]], s=60, marker="*", color=C_APP, zorder=6)


def main():
    fig, ax = plt.subplots(figsize=(16.6, 6.7))
    ax.set_xlim(0, 16.6)
    ax.set_ylim(0, 6.7)
    ax.axis("off")

    # ---- col 1: orifice thumbnails ----
    ys = [4.65, 2.90, 1.15]
    for (name, path), y in zip(FRAMES.items(), ys):
        thumb(ax, path, 0.15, y, 1.95, 1.45, C_ENC)
        ax.text(1.12, y - 0.08, name, ha="center", va="top", fontsize=8.6,
                fontweight="bold", color=INK)
    ax.text(1.12, 6.32, "19 datasets · 1,707 sequences", ha="center",
            fontsize=7.8, color="#4A5560", style="italic")

    # ---- col 2: encoder ----
    box(ax, 2.60, 2.80, 2.05, 2.10, SOFT_ENC, C_ENC)
    gx, gy = 2.85, 3.60
    for r in range(3):
        for c in range(4):
            ax.add_patch(plt.Rectangle((gx + c * 0.38, gy + r * 0.30), 0.30, 0.22,
                                       facecolor=WHITE, edgecolor=C_ENC, lw=0.9, zorder=4))
    ax.text(3.62, 2.62, "Shared encoder\nV-JEPA 2 ViT-L (frozen)", ha="center",
            va="top", fontsize=8.6, fontweight="bold", color=INK)

    # ---- col 3: module glyphs ----
    box(ax, 5.10, 4.30, 2.35, 1.50, SOFT_FC, C_FC)
    glyph_forecast(ax, 5.35, 4.60, 1.85, 0.85)
    ax.text(6.27, 4.16, "Causal residual L1 + coarse L2", ha="center", va="top",
            fontsize=8.4, fontweight="bold", color=INK)
    box(ax, 5.10, 1.20, 2.35, 1.50, SOFT_PHYS, C_PHYS)
    glyph_se3(ax, 5.35, 1.40, 1.85, 0.95)
    ax.text(6.27, 1.06, "SE(3) block-causal dynamics", ha="center", va="top",
            fontsize=8.4, fontweight="bold", color=INK)

    # ---- col 4: capabilities ----
    box(ax, 7.95, 4.30, 2.45, 1.50, WHITE, C_FC)
    glyph_forecast(ax, 8.20, 4.60, 1.95, 0.85)
    ax.text(9.17, 4.16, "Capability 1: forecast\n0.978 cos, p<1e-80", ha="center",
            va="top", fontsize=8.4, fontweight="bold", color=INK)
    box(ax, 7.95, 1.20, 2.45, 1.50, WHITE, C_PHYS)
    glyph_navigation(ax, 8.15, 1.40, 2.05, 0.95)
    ax.text(9.17, 1.06, "Capabilities 2+3\n83.1% / 51.5%", ha="center", va="top",
            fontsize=8.4, fontweight="bold", color=INK)

    # ---- col 5: clinical pathways (tight text boxes) ----
    apps = [("Loss-of-view\nwarning", 4.90), ("Camera-handling\ntraining", 3.20),
            ("Navigation\nassistance", 1.50)]
    for title, y in apps:
        box(ax, 10.85, y, 1.85, 0.95, SOFT_APP, C_APP)
        ax.text(11.77, y + 0.475, title, ha="center", va="center", fontsize=8.6,
                fontweight="bold", color=INK, zorder=5)

    # ---- col 6: result images ----
    res_ys = [4.65, 2.90, 1.15]
    for (name, path), y in zip(RESULTS, res_ys):
        thumb(ax, path, 13.15, y, 2.30, 1.45, C_APP)
        ax.text(14.30, y - 0.08, name, ha="center", va="top", fontsize=8.6,
                fontweight="bold", color=INK)
    ax.text(14.30, 6.32, "Result images (retrieval, not generation)",
            ha="center", fontsize=7.8, color="#4A5560", style="italic")

    # ---- arrows (all orthogonal) ----
    for y in (5.38, 3.63, 1.88):
        elbow(ax, 2.10, y, 2.60, 3.85)
    harrow(ax, 4.65, 5.10, 5.05, text="tokens")
    elbow(ax, 4.65, 3.85, 5.10, 1.95, text=None)
    harrow(ax, 7.45, 7.95, 5.05, text="forecast")
    harrow(ax, 7.45, 7.95, 1.95, text="plan")
    elbow(ax, 10.40, 5.05, 10.85, 5.38)
    harrow(ax, 10.40, 10.85, 3.68)
    elbow(ax, 10.40, 1.95, 10.85, 1.98)
    harrow(ax, 12.70, 13.15, 5.38)
    harrow(ax, 12.70, 13.15, 3.68)
    harrow(ax, 12.70, 13.15, 1.98)

    # ---- bottom strip: four audit gates ----
    gates = [
        ("Input-sensitivity tests", "history / action / domain"),
        ("Reprojection pose gate", "10.2 px -> 0.21 px"),
        ("Matched negative banks", "fixed, same-sequence"),
        ("Grouped CV + frozen test", "no test-set tuning"),
    ]
    gx = 0.15
    for title, sub in gates:
        box(ax, gx, 0.12, 3.95, 0.72, SOFT_AUD, C_AUD)
        ax.text(gx + 1.97, 0.62, title, ha="center", fontsize=8.2,
                fontweight="bold", color=INK, zorder=4)
        ax.text(gx + 1.97, 0.32, sub, ha="center", fontsize=7.2,
                color="#4A5560", zorder=4)
        gx += 4.10

    # column labels
    ax.text(6.27, 6.10, "Passive-video forecast (validated)", ha="center",
            fontsize=9.5, fontweight="bold", color=C_FC)
    ax.text(6.27, 2.82, "Physical grounding (audited; pose/depth-gated)",
            ha="center", fontsize=9.5, fontweight="bold", color=C_PHYS)
    ax.text(11.77, 6.10, "Clinical application pathways", ha="center",
            fontsize=9.5, fontweight="bold", color=C_APP)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figure1_pipeline.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "figure1_pipeline.pdf", bbox_inches="tight", facecolor="white")
    print("[figure1] wrote figures/figure1_pipeline.png/.pdf")


if __name__ == "__main__":
    main()
