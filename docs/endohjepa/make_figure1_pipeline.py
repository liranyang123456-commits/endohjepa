"""Figure 1: Endo-HJEPA overview — final layout.

Real domain thumbnails (labels below images) on the left, domain-matched
retrieval results (labels below images) on the right. Middle modules are
visible boxes containing a glyph plus the full text description (title +
subtitle). All arrows are orthogonal and touch box borders. Bottom strip:
four audit gates.

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
    ("Laparoscopy", "_fig1_in_laparo.png", "_fig1_out_laparo.png"),
    ("GI endoscopy", "_fig1_in_gi.png", "_fig1_out_gi.png"),
    ("Bronchoscopy", "_fig1_in_bronch.png", "_fig1_out_bronch.png"),
]
IMG_H = 1.15


def thumb(ax, path, cx, cy, h, ec, label, sub=None):
    """Image centred at (cx, cy), native aspect; label below the image."""
    img = Image.open(path)
    w = h * (img.size[0] / img.size[1])
    arr = np.asarray(img.convert("RGB"))
    ax.imshow(arr, extent=[cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2],
              aspect="auto", zorder=3)
    ax.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False,
                               edgecolor=ec, linewidth=1.2, zorder=4))
    ax.text(cx, cy - h / 2 - 0.10, label, ha="center", va="top", fontsize=8.4,
            fontweight="bold", color=INK)
    if sub:
        ax.text(cx, cy - h / 2 - 0.34, sub, ha="center", va="top", fontsize=7.2,
                color="#4A5560", style="italic")
    return w


def module(ax, x, y, w, h, title, sub, fc, ec, glyph=None):
    """Visible box with title (top), optional glyph (middle), subtitle (bottom)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.03",
                                linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h - 0.26, title, ha="center", va="top", fontsize=8.6,
            fontweight="bold", color=INK, zorder=5)
    if glyph:
        glyph(ax, x + w / 2, y + h * 0.46)
    ax.text(x + w / 2, y + 0.24, sub, ha="center", va="bottom", fontsize=7.4,
            color="#4A5560", zorder=5)


def harrow(ax, x1, x2, y, text=None):
    ax.plot([x1, x2 - 0.04], [y, y], color=LINE, lw=1.3, zorder=1, solid_capstyle="butt")
    ax.annotate("", xy=(x2, y), xytext=(x2 - 0.04, y),
                arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.3, mutation_scale=11), zorder=2)
    if text:
        ax.text((x1 + x2) / 2, y + 0.12, text, ha="center", fontsize=7.4,
                color="#4A5560", style="italic")


def elbow(ax, x1, y1, x2, y2):
    xm = x1 + 0.22
    ax.plot([x1, xm, xm, x2 - 0.04], [y1, y1, y2, y2], color=LINE, lw=1.3,
            zorder=1, solid_capstyle="butt")
    ax.annotate("", xy=(x2, y2), xytext=(x2 - 0.04, y2),
                arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.3, mutation_scale=11), zorder=2)


def glyph_forecast(ax, cx, cy, s=0.38):
    w, h = s * 2.2, s * 1.4
    x, y = cx - w / 2, cy - h / 2
    t = np.linspace(0, 1, 20)
    ax.plot([x, x + w * 0.42], [y + h * 0.32, y + h * 0.32], color=INK, lw=1.4, zorder=6)
    ax.plot(x + w * (0.42 + 0.58 * t), y + h * (0.32 + 0.45 * t), color=C_FC, lw=1.6, zorder=6)
    ax.plot(x + w * (0.42 + 0.58 * t), y + h * np.full_like(t, 0.32), color=LINE,
            lw=1.2, ls="--", zorder=6)
    ax.scatter(x + w * np.linspace(0.05, 0.38, 4), np.full(4, y + h * 0.32), s=8, color=INK, zorder=7)
    ax.scatter([x + w * 0.95], [y + h * 0.75], s=16, color=C_FC, zorder=7)


def glyph_se3(ax, cx, cy, s=0.38):
    w, h = s * 2.0, s * 1.4
    x0, y0 = cx - w * 0.32, cy - h * 0.30
    ax.annotate("", xy=(x0 + w * 0.42, y0), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#C24D4D", lw=1.6), zorder=6)
    ax.annotate("", xy=(x0, y0 + h * 0.5), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#3F7A5A", lw=1.6), zorder=6)
    ax.annotate("", xy=(x0 + w * 0.26, y0 + h * 0.30), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#4E7FA6", lw=1.6), zorder=6)


def glyph_navigation(ax, cx, cy, s=0.38):
    w, h = s * 2.2, s * 1.4
    x, y = cx - w / 2, cy - h / 2
    t = np.linspace(0, 1, 30)
    px = x + w * (0.08 + 0.84 * t)
    py = y + h * (0.25 + 0.45 * np.sin(t * np.pi) + 0.1 * t)
    ax.plot(px, py, color=C_PHYS, lw=1.6, zorder=6)
    ax.scatter([px[0]], [py[0]], s=20, color=INK, zorder=7)
    ax.scatter([px[-1]], [py[-1]], s=60, marker="*", color=C_APP, zorder=7)


def glyph_tokens(ax, cx, cy):
    for r in range(3):
        for c in range(4):
            ax.add_patch(plt.Rectangle((cx - 0.72 + c * 0.36, cy - 0.42 + r * 0.28),
                                       0.28, 0.20, facecolor=WHITE, edgecolor=C_ENC,
                                       lw=0.9, zorder=6))


def main():
    fig, ax = plt.subplots(figsize=(16.4, 7.8))
    ax.set_xlim(0, 16.4)
    ax.set_ylim(0, 7.8)
    ax.axis("off")

    # grid columns (left edges)
    X_IN, X_ENC, X_MOD, X_CAP, X_RES = 0.25, 3.35, 6.45, 9.60, 13.20
    ROW_FC = [6.10, 4.55, 3.00]     # forecast row centres
    ROW_PHYS = 1.55                 # physical row centre

    # ---- inputs (left) and domain-matched results (right) ----
    for (name, inp, outp), cy in zip(DOMAINS, ROW_FC):
        w_in = thumb(ax, OUT / inp, X_IN + 1.05, cy, IMG_H, C_FC, name, "input clip")
        w_out = thumb(ax, OUT / outp, X_RES + 1.05, cy, IMG_H, C_FC,
                      name, "same-dataset retrieval")
        elbow(ax, X_IN + 1.05 + w_in / 2, cy, X_ENC, cy)
        elbow(ax, X_CAP + 2.30, cy, X_RES + 1.05 - w_out / 2, cy)

    # ---- shared encoder box ----
    enc_cy = sum(ROW_FC) / 3
    box_h = (ROW_FC[0] - ROW_FC[2]) + 1.1
    module(ax, X_ENC, enc_cy - box_h / 2, 1.95, box_h,
           "Shared encoder", "", SOFT_ENC, C_ENC)
    glyph_tokens(ax, X_ENC + 0.97, enc_cy + 0.85)
    ax.text(X_ENC + 0.97, enc_cy - 0.55,
            "V-JEPA 2 ViT-L (frozen)\ndense tokens $z$, $D{=}1024$\n+ domain $e_d$",
            ha="center", va="center", fontsize=7.6, color="#4A5560", zorder=5)

    # ---- forecast module + capability boxes ----
    module(ax, X_MOD, enc_cy - 1.05, 2.30, 2.10,
           "Causal residual L1 + L2", "short- and mid-horizon\nresidual, domain-conditioned",
           SOFT_FC, C_FC, glyph=glyph_forecast)
    module(ax, X_CAP, enc_cy - 1.05, 2.30, 2.10,
           "Capability 1", "forecast scene evolution\n0.978 cos, $p{<}10^{-80}$",
           WHITE, C_FC, glyph=glyph_forecast)
    harrow(ax, X_ENC + 1.95, X_MOD, enc_cy, text="tokens")
    harrow(ax, X_MOD + 2.30, X_CAP, enc_cy, text="forecast")

    # ---- physical lane (SCARED) ----
    w_in = thumb(ax, OUT / "_fig1_in_scared.png", X_IN + 1.05, ROW_PHYS, IMG_H,
                 C_PHYS, "SCARED", "pose + depth input")
    w_out = thumb(ax, OUT / "_fig1_se3.png", X_RES + 1.05, ROW_PHYS, IMG_H,
                  C_PHYS, "SCARED", "action / navigation result")
    module(ax, X_MOD, ROW_PHYS - 1.05, 2.30, 2.10,
           "SE(3) dynamics", "block-causal ensemble\nrisk + covariance",
           SOFT_PHYS, C_PHYS, glyph=glyph_se3)
    module(ax, X_CAP, ROW_PHYS - 1.05, 2.30, 2.10,
           "Capabilities 2 + 3", "action 83.1% · navigation 51.5%\nnormalised CEM, hard gate",
           WHITE, C_PHYS, glyph=glyph_navigation)
    elbow(ax, X_IN + 1.05 + w_in / 2, ROW_PHYS, X_MOD, ROW_PHYS)
    harrow(ax, X_MOD + 2.30, X_CAP, ROW_PHYS, text="plan")
    elbow(ax, X_CAP + 2.30, ROW_PHYS, X_RES + 1.05 - w_out / 2, ROW_PHYS)
    # encoder feeds down into the physical lane
    ax.plot([X_ENC + 0.97, X_ENC + 0.97], [enc_cy - box_h / 2, ROW_PHYS + 0.04],
            color=LINE, lw=1.3, zorder=1, solid_capstyle="butt")
    ax.annotate("", xy=(X_ENC + 0.97, ROW_PHYS), xytext=(X_ENC + 0.97, ROW_PHYS + 0.04),
                arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.3, mutation_scale=11), zorder=2)

    # ---- application labels on capability->result arrows ----
    for label, cy in zip(["loss-of-view warning", "camera-handling training",
                          "navigation assistance"], ROW_FC):
        ax.text((X_CAP + 2.30 + X_RES + 1.05) / 2, cy + 0.16, label, ha="center",
                fontsize=7.4, color=C_APP, style="italic")

    # ---- bottom strip: four audit gates ----
    gates = [
        "Input-sensitivity tests: history / action / domain",
        "Reprojection pose gate: 10.2 px -> 0.21 px",
        "Matched negative banks: fixed, same-sequence",
        "Grouped CV + frozen test: no test-set tuning",
    ]
    gx = 0.35
    for g in gates:
        ax.text(gx, 0.26, g, ha="left", va="center", fontsize=7.8, color=INK, zorder=4,
                bbox=dict(boxstyle="round,pad=0.28", facecolor=SOFT_AUD,
                          edgecolor=C_AUD, linewidth=1.0))
        gx += 4.00

    # lane labels
    ax.text(X_MOD + 1.15, 7.45, "Passive-video forecast (validated, three orifices)",
            ha="center", fontsize=9.5, fontweight="bold", color=C_FC)
    ax.text(X_MOD + 1.15, 2.72, "Physical grounding (audited; pose/depth-gated, SCARED/C3VD)",
            ha="center", fontsize=9.5, fontweight="bold", color=C_PHYS)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figure1_pipeline.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "figure1_pipeline.pdf", bbox_inches="tight", facecolor="white")
    print("[figure1] wrote figures/figure1_pipeline.png/.pdf")


if __name__ == "__main__":
    main()
