"""Figure 1: Endo-HJEPA overview with real thumbnails and glyph modules.

Real endoscopic frames for the three orifices (labels below images), vector
glyphs for each module, data-flow labels on the arrows, and a bottom strip
with the four correctness-audit gates. Muted palette, orthogonal arrows.

    python docs/endohjepa/make_figure1_pipeline.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
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


def box(ax, x, y, w, h, fc, ec, lw=1.2):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.04",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)


def label_below(ax, x, y, w, text, color=INK, fs=8.6, bold=True):
    ax.text(x + w / 2, y, text, ha="center", va="top", fontsize=fs,
            fontweight="bold" if bold else "normal", color=color)


def arrow(ax, x1, y1, x2, y2, text=None, text_dy=0.10):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.4,
                                mutation_scale=11), zorder=1)
    if text:
        ax.text((x1 + x2) / 2, max(y1, y2) + text_dy, text, ha="center",
                fontsize=7.4, color="#4A5560", style="italic")


def thumb(ax, path, x, y, w, h, ec):
    img = np.asarray(Image.open(path).convert("RGB"))
    ax.imshow(img, extent=[x, x + w, y, y + h], aspect="auto", zorder=3)
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor=ec,
                               linewidth=1.2, zorder=4))


def glyph_forecast(ax, x, y, w, h):
    """Mini timeline: history dots -> predicted curve vs persistence."""
    t = np.linspace(0, 1, 20)
    ax.plot([x, x + w * 0.42], [y + h * 0.32, y + h * 0.32], color=INK, lw=1.4, zorder=5)
    ax.plot(x + w * (0.42 + 0.58 * t), y + h * (0.32 + 0.45 * t), color=C_FC, lw=1.6, zorder=5)
    ax.plot(x + w * (0.42 + 0.58 * t), y + h * np.full_like(t, 0.32), color=LINE,
            lw=1.2, ls="--", zorder=5)
    ax.scatter(x + w * np.linspace(0.05, 0.38, 4), np.full(4, y + h * 0.32),
               s=8, color=INK, zorder=6)
    ax.scatter([x + w * 0.95], [y + h * 0.75], s=16, color=C_FC, zorder=6)


def glyph_se3(ax, x, y, w, h):
    """Coordinate-frame glyph: three axes."""
    cx, cy = x + w * 0.32, y + h * 0.30
    ax.annotate("", xy=(cx + w * 0.42, cy), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color="#C24D4D", lw=1.6), zorder=5)
    ax.annotate("", xy=(cx, cy + h * 0.5), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color="#3F7A5A", lw=1.6), zorder=5)
    ax.annotate("", xy=(cx + w * 0.26, cy + h * 0.30), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color="#4E7FA6", lw=1.6), zorder=5)
    ax.text(cx + w * 0.46, cy - 2, "x", fontsize=7, color="#C24D4D")
    ax.text(cx - 2, cy + h * 0.54, "z", fontsize=7, color="#3F7A5A")


def glyph_navigation(ax, x, y, w, h):
    """Path from start to goal with waypoints."""
    t = np.linspace(0, 1, 30)
    px = x + w * (0.08 + 0.84 * t)
    py = y + h * (0.25 + 0.45 * np.sin(t * np.pi) + 0.1 * t)
    ax.plot(px, py, color=C_PHYS, lw=1.6, zorder=5)
    ax.scatter([px[0]], [py[0]], s=20, color=INK, zorder=6)
    ax.scatter([px[-1]], [py[-1]], s=60, marker="*", color=C_APP, zorder=6)


def glyph_warning(ax, x, y, w, h):
    """Warning triangle."""
    tri = plt.Polygon([(x + w * 0.5, y + h * 0.85), (x + w * 0.12, y + h * 0.15),
                       (x + w * 0.88, y + h * 0.15)], closed=True,
                      facecolor="#F6E3C5", edgecolor="#B5833A", lw=1.4, zorder=5)
    ax.add_patch(tri)
    ax.text(x + w * 0.5, y + h * 0.30, "!", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#B5833A", zorder=6)


def glyph_training(ax, x, y, w, h):
    """Eye + gauge for camera-handling assessment."""
    eye = plt.Polygon([(x + w * 0.10, y + h * 0.5), (x + w * 0.5, y + h * 0.85),
                       (x + w * 0.90, y + h * 0.5), (x + w * 0.5, y + h * 0.15)],
                      closed=True, facecolor=WHITE, edgecolor=C_APP, lw=1.4, zorder=5)
    ax.add_patch(eye)
    ax.add_patch(Circle((x + w * 0.5, y + h * 0.5), w * 0.10, facecolor=C_APP,
                        edgecolor="none", zorder=6))


def glyph_nav_assist(ax, x, y, w, h):
    """Compass-like target for navigation assistance."""
    ax.add_patch(Circle((x + w * 0.5, y + h * 0.5), w * 0.30, fill=False,
                        edgecolor=C_APP, lw=1.4, zorder=5))
    ax.annotate("", xy=(x + w * 0.68, y + h * 0.68), xytext=(x + w * 0.5, y + h * 0.5),
                arrowprops=dict(arrowstyle="-|>", color=C_APP, lw=1.6), zorder=6)


def main():
    fig, ax = plt.subplots(figsize=(13.8, 6.6))
    ax.set_xlim(0, 13.8)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    # ---- left: three orifice thumbnails, labels below ----
    ys = [4.55, 2.85, 1.15]
    for (name, path), y in zip(FRAMES.items(), ys):
        thumb(ax, path, 0.15, y, 1.95, 1.45, C_ENC)
        label_below(ax, 0.15, y - 0.06, 1.95, name)
    ax.text(1.12, 6.22, "19 datasets · 1,707 sequences", ha="center",
            fontsize=7.8, color="#4A5560", style="italic")

    # ---- shared encoder: token-grid glyph ----
    box(ax, 2.60, 2.75, 2.05, 2.15, SOFT_ENC, C_ENC)
    gx, gy = 2.85, 3.55
    for r in range(3):
        for c in range(4):
            ax.add_patch(plt.Rectangle((gx + c * 0.38, gy + r * 0.30), 0.30, 0.22,
                                       facecolor=WHITE, edgecolor=C_ENC, lw=0.9, zorder=4))
    label_below(ax, 2.60, 2.62, 2.05, "Shared encoder\nV-JEPA 2 ViT-L (frozen)")

    # ---- lane 1: forecast ----
    box(ax, 5.10, 4.20, 2.35, 1.55, SOFT_FC, C_FC)
    glyph_forecast(ax, 5.35, 4.55, 1.85, 0.85)
    label_below(ax, 5.10, 4.06, 2.35, "Causal residual L1 + coarse L2")
    box(ax, 7.95, 4.20, 2.45, 1.55, WHITE, C_FC)
    glyph_forecast(ax, 8.20, 4.55, 1.95, 0.85)
    label_below(ax, 7.95, 4.06, 2.45, "Capability 1: forecast evolution\n0.978 cos, p<1e-80")

    # ---- lane 2: physical grounding ----
    box(ax, 5.10, 1.15, 2.35, 1.55, SOFT_PHYS, C_PHYS)
    glyph_se3(ax, 5.35, 1.35, 1.85, 0.95)
    label_below(ax, 5.10, 1.01, 2.35, "SE(3) block-causal dynamics")
    box(ax, 7.95, 1.15, 2.45, 1.55, WHITE, C_PHYS)
    glyph_navigation(ax, 8.15, 1.35, 2.05, 0.95)
    label_below(ax, 7.95, 1.01, 2.45, "Capabilities 2+3: 83.1% / 51.5%")

    # ---- right: three application pathways with glyphs ----
    apps = [
        ("Loss-of-view warning", glyph_warning, 4.90),
        ("Camera-handling training", glyph_training, 3.20),
        ("Navigation assistance", glyph_nav_assist, 1.50),
    ]
    for title, glyph, y in apps:
        box(ax, 10.85, y, 2.80, 1.30, SOFT_APP, C_APP)
        glyph(ax, 11.05, y + 0.28, 0.75, 0.75)
        ax.text(11.95, y + 0.65, title, ha="left", va="center", fontsize=8.8,
                fontweight="bold", color=INK, zorder=5)

    # ---- arrows ----
    for y in (5.28, 3.58, 1.88):
        ax.plot([2.10, 2.35, 2.35, 2.60], [y, y, 3.82, 3.82], color=LINE, lw=1.2, zorder=1)
    arrow(ax, 2.35, 3.82, 2.60, 3.82, text=None)

    ax.plot([4.65, 4.88, 4.88, 5.10], [3.82, 3.82, 4.97, 4.97], color=LINE, lw=1.3)
    arrow(ax, 4.88, 4.97, 5.10, 4.97)
    ax.plot([4.65, 4.88, 4.88, 5.10], [3.82, 3.82, 1.92, 1.92], color=LINE, lw=1.3)
    arrow(ax, 4.88, 1.92, 5.10, 1.92)
    ax.text(4.88, 4.10, "tokens", ha="center", fontsize=7.4, color="#4A5560", style="italic")

    arrow(ax, 7.45, 4.97, 7.95, 4.97, text="forecast")
    arrow(ax, 7.45, 1.92, 7.95, 1.92, text="plan")

    arrow(ax, 10.40, 4.97, 10.85, 5.35)
    arrow(ax, 10.40, 4.60, 10.85, 3.85)
    arrow(ax, 10.40, 1.92, 10.85, 2.15)

    # ---- bottom strip: four audit gates ----
    gates = [
        ("Input-sensitivity tests", "history / action / domain"),
        ("Reprojection pose gate", "10.2 px -> 0.21 px"),
        ("Matched negative banks", "fixed, same-sequence"),
        ("Grouped CV + frozen test", "no test-set tuning"),
    ]
    gx = 0.15
    for title, sub in gates:
        box(ax, gx, 0.12, 3.28, 0.72, SOFT_AUD, C_AUD)
        ax.text(gx + 1.64, 0.60, title, ha="center", fontsize=8.2,
                fontweight="bold", color=INK, zorder=4)
        ax.text(gx + 1.64, 0.30, sub, ha="center", fontsize=7.2,
                color="#4A5560", zorder=4)
        gx += 3.44

    # lane + column labels
    ax.text(6.30, 6.05, "Passive-video forecast (validated)", ha="center",
            fontsize=9.5, fontweight="bold", color=C_FC)
    ax.text(6.30, 2.72, "Physical grounding (audited; pose/depth-gated)",
            ha="center", fontsize=9.5, fontweight="bold", color=C_PHYS)
    ax.text(12.25, 6.05, "Clinical application pathways", ha="center",
            fontsize=9.5, fontweight="bold", color=C_APP)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figure1_pipeline.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "figure1_pipeline.pdf", bbox_inches="tight", facecolor="white")
    print("[figure1] wrote figures/figure1_pipeline.png/.pdf")


if __name__ == "__main__":
    main()
