"""Figure 1: Endo-HJEPA overview — thumbnails + glyphs + text + result images.

Each module box combines a vector glyph with its text description. All arrows
are orthogonal elbow connectors that start and end exactly at box borders.
Columns: orifice frames -> encoder -> module glyphs -> capabilities ->
clinical pathways -> result images. Bottom strip: four audit gates.

    python docs/endohjepa/make_figure1_pipeline.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image

OUT = Path(__file__).resolve().parent / "figures"
ROOT = Path(__file__).resolve().parents[2]

INK = "#2D3741"
LINE = "#8E979F"
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
                                boxstyle="round,pad=0.004,rounding_size=0.03",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))


def elbow(ax, x1, y1, x2, y2, text=None):
    """Orthogonal connector ending exactly at (x2, y2) with a visible arrow."""
    xm = x1 + 0.18
    ax.plot([x1, xm, xm, x2 - 0.03], [y1, y1, y2, y2], color=LINE, lw=1.3, zorder=1,
            solid_capstyle="butt")
    ax.annotate("", xy=(x2, y2), xytext=(x2 - 0.03, y2),
                arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.3,
                                mutation_scale=11), zorder=2)
    if text:
        ax.text(xm, max(y1, y2) + 0.10, text, ha="center", fontsize=7.4,
                color="#4A5560", style="italic")


def harrow(ax, x1, x2, y, text=None):
    ax.plot([x1, x2 - 0.03], [y, y], color=LINE, lw=1.3, zorder=1,
            solid_capstyle="butt")
    ax.annotate("", xy=(x2, y), xytext=(x2 - 0.03, y),
                arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.3,
                                mutation_scale=11), zorder=2)
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
    fig, ax = plt.subplots(figsize=(16.6, 7.0))
    ax.set_xlim(0, 16.6)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    # ---- col 1: orifice thumbnails (x 0.15-2.10) ----
    ys = [4.85, 3.05, 1.25]
    for (name, path), y in zip(FRAMES.items(), ys):
        thumb(ax, path, 0.15, y, 1.95, 1.45, C_ENC)
        ax.text(1.12, y - 0.08, name, ha="center", va="top", fontsize=8.6,
                fontweight="bold", color=INK)
    ax.text(1.12, 6.55, "19 datasets · 1,707 sequences", ha="center",
            fontsize=7.8, color="#4A5560", style="italic")

    # ---- col 2: encoder (x 2.60-4.65, y 2.95-5.15) ----
    box(ax, 2.60, 2.95, 2.05, 2.20, SOFT_ENC, C_ENC)
    gx, gy = 2.88, 3.95
    for r in range(3):
        for c in range(4):
            ax.add_patch(plt.Rectangle((gx + c * 0.38, gy + r * 0.30), 0.30, 0.22,
                                       facecolor=WHITE, edgecolor=C_ENC, lw=0.9, zorder=4))
    ax.text(3.62, 3.72, "Shared encoder\nV-JEPA 2 ViT-L (frozen)\ndense tokens $z$, $D{=}1024$",
            ha="center", va="top", fontsize=8.2, color=INK, zorder=5)

    # ---- col 3: modules (x 5.15-7.50) ----
    box(ax, 5.15, 4.45, 2.35, 1.70, SOFT_FC, C_FC)
    glyph_forecast(ax, 5.40, 5.30, 1.85, 0.70)
    ax.text(6.32, 5.16, "Causal residual L1 + coarse L2\nshort- and mid-horizon forecast",
            ha="center", va="top", fontsize=8.0, color=INK, zorder=5)
    box(ax, 5.15, 1.30, 2.35, 1.70, SOFT_PHYS, C_PHYS)
    glyph_se3(ax, 5.40, 2.20, 1.85, 0.70)
    ax.text(6.32, 2.06, "SE(3) block-causal dynamics\nensemble, risk + covariance",
            ha="center", va="top", fontsize=8.0, color=INK, zorder=5)

    # ---- col 4: capabilities (x 8.00-10.45) ----
    box(ax, 8.00, 4.45, 2.45, 1.70, WHITE, C_FC)
    glyph_forecast(ax, 8.25, 5.30, 1.95, 0.70)
    ax.text(9.22, 5.16, "Capability 1: forecast evolution\n0.978 cos, $p{<}10^{-80}$",
            ha="center", va="top", fontsize=8.0, color=INK, zorder=5)
    box(ax, 8.00, 1.30, 2.45, 1.70, WHITE, C_PHYS)
    glyph_navigation(ax, 8.20, 2.20, 2.05, 0.70)
    ax.text(9.22, 2.06, "Capabilities 2+3\naction 83.1% · navigation 51.5%",
            ha="center", va="top", fontsize=8.0, color=INK, zorder=5)

    # ---- col 5: clinical pathways (x 10.90-12.75, tight) ----
    apps = [("Loss-of-view\nwarning", 5.05), ("Camera-handling\ntraining", 3.30),
            ("Navigation\nassistance", 1.55)]
    for title, y in apps:
        box(ax, 10.90, y, 1.85, 0.95, SOFT_APP, C_APP)
        ax.text(11.82, y + 0.475, title, ha="center", va="center", fontsize=8.6,
                fontweight="bold", color=INK, zorder=5)

    # ---- col 6: result images (x 13.20-15.50) ----
    res_ys = [4.85, 3.05, 1.25]
    for (name, path), y in zip(RESULTS, res_ys):
        thumb(ax, path, 13.20, y, 2.30, 1.45, C_APP)
        ax.text(14.35, y - 0.08, name, ha="center", va="top", fontsize=8.6,
                fontweight="bold", color=INK)
    ax.text(14.35, 6.55, "Result images (retrieval, not generation)",
            ha="center", fontsize=7.8, color="#4A5560", style="italic")

    # ---- arrows: inputs -> encoder (elbows into encoder left edge, y=4.05) ----
    for y in (5.58, 3.78, 1.98):
        elbow(ax, 2.10, y, 2.60, 4.05)
    # encoder -> lanes
    elbow(ax, 4.65, 4.05, 5.15, 5.30, text="tokens")
    elbow(ax, 4.65, 4.05, 5.15, 2.15)
    # modules -> capabilities
    harrow(ax, 7.50, 8.00, 5.30, text="forecast")
    harrow(ax, 7.50, 8.00, 2.15, text="plan")
    # capabilities -> applications
    elbow(ax, 10.45, 5.30, 10.90, 5.52)
    harrow(ax, 10.45, 10.90, 3.78)
    elbow(ax, 10.45, 2.15, 10.90, 2.02)
    # applications -> results
    harrow(ax, 12.75, 13.20, 5.52)
    harrow(ax, 12.75, 13.20, 3.78)
    harrow(ax, 12.75, 13.20, 2.02)

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
    ax.text(6.32, 6.40, "Passive-video forecast (validated)", ha="center",
            fontsize=9.5, fontweight="bold", color=C_FC)
    ax.text(6.32, 3.30, "Physical grounding (audited; pose/depth-gated)",
            ha="center", fontsize=9.5, fontweight="bold", color=C_PHYS)
    ax.text(11.82, 6.40, "Clinical application pathways", ha="center",
            fontsize=9.5, fontweight="bold", color=C_APP)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figure1_pipeline.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "figure1_pipeline.pdf", bbox_inches="tight", facecolor="white")
    print("[figure1] wrote figures/figure1_pipeline.png/.pdf")


if __name__ == "__main__":
    main()
