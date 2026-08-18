"""Figure 1: Endo-HJEPA overview with domain-matched input->output rows.

Forecast lane: each orifice input row connects to its own same-dataset
retrieval result (true input->output correspondence). Physical lane: SCARED
input connects to the SCARED action-conditioned and navigation results. All
arrows are orthogonal and touch box borders. Bottom strip: four audit gates.

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
SOFT_AUD = "#F8F4EF"
WHITE = "#FFFFFF"

DOMAINS = [
    ("Laparoscopy", "laparo"),
    ("GI endoscopy", "gi"),
    ("Bronchoscopy", "bronch"),
]


def box(ax, x, y, w, h, fc, ec, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.03",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))


def harrow(ax, x1, x2, y, text=None):
    ax.plot([x1, x2 - 0.03], [y, y], color=LINE, lw=1.3, zorder=1,
            solid_capstyle="butt")
    ax.annotate("", xy=(x2, y), xytext=(x2 - 0.03, y),
                arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.3,
                                mutation_scale=11), zorder=2)
    if text:
        ax.text((x1 + x2) / 2, y + 0.10, text, ha="center", fontsize=7.4,
                color="#4A5560", style="italic")


def elbow(ax, x1, y1, x2, y2):
    xm = x1 + 0.18
    ax.plot([x1, xm, xm, x2 - 0.03], [y1, y1, y2, y2], color=LINE, lw=1.3,
            zorder=1, solid_capstyle="butt")
    ax.annotate("", xy=(x2, y2), xytext=(x2 - 0.03, y2),
                arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.3,
                                mutation_scale=11), zorder=2)


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
    fig, ax = plt.subplots(figsize=(16.6, 8.4))
    ax.set_xlim(0, 16.6)
    ax.set_ylim(0, 8.4)
    ax.axis("off")

    # ================= forecast lane: 3 domain-matched rows =================
    row_ys = [6.55, 4.85, 3.15]   # bottom y of each row's thumbnails
    row_cy = [y + 0.72 for y in row_ys]

    # left: domain inputs
    for (name, tag), y, cy in zip(DOMAINS, row_ys, row_cy):
        thumb(ax, OUT / f"_fig1_in_{tag}.png", 0.15, y, 1.95, 1.45, C_ENC)
        ax.text(1.12, y - 0.08, name, ha="center", va="top", fontsize=8.6,
                fontweight="bold", color=INK)
    ax.text(1.12, 8.18, "19 datasets · 1,707 sequences", ha="center",
            fontsize=7.8, color="#4A5560", style="italic")

    # shared encoder (tall, spans the three rows)
    box(ax, 2.65, 3.55, 2.05, 4.05, SOFT_ENC, C_ENC)
    gx, gy = 2.93, 6.20
    for r in range(3):
        for c in range(4):
            ax.add_patch(plt.Rectangle((gx + c * 0.38, gy + r * 0.30), 0.30, 0.22,
                                       facecolor=WHITE, edgecolor=C_ENC, lw=0.9, zorder=4))
    ax.text(3.67, 6.00, "Shared encoder\nV-JEPA 2 ViT-L (frozen)\ndense tokens $z$, $D{=}1024$",
            ha="center", va="top", fontsize=8.2, color=INK, zorder=5)

    # forecast module (tall)
    box(ax, 5.20, 3.55, 2.35, 4.05, SOFT_FC, C_FC)
    glyph_forecast(ax, 5.45, 6.55, 1.85, 0.70)
    ax.text(6.37, 6.40, "Causal residual L1 + coarse L2\nshort- and mid-horizon forecast\nresidual, domain-conditioned",
            ha="center", va="top", fontsize=8.0, color=INK, zorder=5)

    # capability 1 (tall)
    box(ax, 8.05, 3.55, 2.45, 4.05, WHITE, C_FC)
    glyph_forecast(ax, 8.30, 6.55, 1.95, 0.70)
    ax.text(9.27, 6.40, "Capability 1\nforecast scene evolution\n0.978 cos, $p{<}10^{-80}$",
            ha="center", va="top", fontsize=8.2, color=INK, zorder=5)

    # right: domain-matched retrieval results
    for (name, tag), y in zip(DOMAINS, row_ys):
        thumb(ax, OUT / f"_fig1_out_{tag}.png", 13.20, y, 2.30, 1.45, C_APP)
    ax.text(14.35, 8.18, "Forecast retrieval (same dataset)", ha="center",
            fontsize=7.8, color="#4A5560", style="italic")

    # application annotation on the capability->result arrows
    for y in row_cy:
        harrow(ax, 10.50, 13.20, y)
    ax.text(11.85, row_cy[0] + 0.12, "loss-of-view warning", ha="center",
            fontsize=7.4, color=C_APP, style="italic")
    ax.text(11.85, row_cy[1] + 0.12, "camera-handling training", ha="center",
            fontsize=7.4, color=C_APP, style="italic")
    ax.text(11.85, row_cy[2] + 0.12, "navigation assistance", ha="center",
            fontsize=7.4, color=C_APP, style="italic")

    # forecast-lane arrows
    for y in row_cy:
        elbow(ax, 2.10, y, 2.65, 5.57)
    harrow(ax, 4.70, 5.20, 5.57, text="tokens")
    harrow(ax, 7.55, 8.05, 5.57, text="forecast")

    # ================= physical lane (SCARED) =================
    py = 0.95
    pcy = py + 0.72
    thumb(ax, OUT / "_fig1_in_scared.png", 0.15, py, 1.95, 1.45, C_PHYS)
    ax.text(1.12, py - 0.08, "SCARED (pose + depth)", ha="center", va="top",
            fontsize=8.6, fontweight="bold", color=INK)

    box(ax, 5.20, 0.95, 2.35, 1.70, SOFT_PHYS, C_PHYS)
    glyph_se3(ax, 5.45, 2.00, 1.85, 0.60)
    ax.text(6.37, 1.86, "SE(3) block-causal dynamics\nensemble, risk + covariance",
            ha="center", va="top", fontsize=8.0, color=INK, zorder=5)

    box(ax, 8.05, 0.95, 2.45, 1.70, WHITE, C_PHYS)
    glyph_navigation(ax, 8.25, 2.00, 2.05, 0.60)
    ax.text(9.27, 1.86, "Capabilities 2+3\naction 83.1% · navigation 51.5%",
            ha="center", va="top", fontsize=8.0, color=INK, zorder=5)

    thumb(ax, OUT / "_fig1_se3.png", 13.20, py, 2.30, 1.45, C_PHYS)
    ax.text(14.35, py - 0.08, "Action / navigation results (SCARED)",
            ha="center", va="top", fontsize=7.8, color="#4A5560", style="italic")

    # physical-lane arrows (SCARED input enters the SE(3) branch; encoder shared)
    elbow(ax, 2.10, pcy, 5.20, pcy)
    # encoder output feeds down into the physical flow (arrowhead at junction)
    ax.plot([3.67, 3.67], [3.55, pcy + 0.03], color=LINE, lw=1.3, zorder=1,
            solid_capstyle="butt")
    ax.annotate("", xy=(3.67, pcy), xytext=(3.67, pcy + 0.03),
                arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.3, mutation_scale=11), zorder=2)
    harrow(ax, 7.55, 8.05, pcy, text="plan")
    harrow(ax, 10.50, 13.20, pcy)

    # lane labels
    ax.text(6.37, 7.85, "Passive-video forecast (validated, three orifices)",
            ha="center", fontsize=9.5, fontweight="bold", color=C_FC)
    ax.text(6.37, 2.80, "Physical grounding (audited; pose/depth-gated, SCARED/C3VD)",
            ha="center", fontsize=9.5, fontweight="bold", color=C_PHYS)

    # ---- bottom strip: four audit gates ----
    gates = [
        ("Input-sensitivity tests", "history / action / domain"),
        ("Reprojection pose gate", "10.2 px -> 0.21 px"),
        ("Matched negative banks", "fixed, same-sequence"),
        ("Grouped CV + frozen test", "no test-set tuning"),
    ]
    gx = 0.15
    for title, sub in gates:
        box(ax, gx, 0.06, 3.95, 0.62, SOFT_AUD, C_AUD)
        ax.text(gx + 1.97, 0.50, title, ha="center", fontsize=8.0,
                fontweight="bold", color=INK, zorder=4)
        ax.text(gx + 1.97, 0.22, sub, ha="center", fontsize=7.0,
                color="#4A5560", zorder=4)
        gx += 4.10

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figure1_pipeline.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "figure1_pipeline.pdf", bbox_inches="tight", facecolor="white")
    print("[figure1] wrote figures/figure1_pipeline.png/.pdf")


if __name__ == "__main__":
    main()
