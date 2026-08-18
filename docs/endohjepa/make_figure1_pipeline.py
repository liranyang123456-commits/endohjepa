"""Figure 1: capability-driven Endo-HJEPA overview (journal flowchart).

Two horizontal lanes: the validated passive-forecast path (Capability 1) and
the audited physical-grounding path (Capabilities 2/3), with the three
clinical application pathways on the right. Muted palette, orthogonal arrows.

    python docs/endohjepa/make_figure1_pipeline.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"

INK = "#2D3741"
LINE = "#9AA4AE"
C_ENC = "#4E7FA6"    # encoder blue
C_FC = "#4E8D8D"     # forecast teal
C_PHYS = "#8A7AA0"   # physical purple
C_APP = "#3F7A5A"    # application green
SOFT_ENC = "#EFF5FA"
SOFT_FC = "#EFF8F8"
SOFT_PHYS = "#F5F3F8"
SOFT_APP = "#EFF7F2"
WHITE = "#FFFFFF"


def box(ax, x, y, w, h, title, sub, fc, ec, fs=8.5, subfs=7.2):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.04",
                       linewidth=1.15, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=INK, zorder=3)
    if sub:
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                fontsize=subfs, color="#4A5560", zorder=3)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.3,
                                mutation_scale=10), zorder=1)


def main():
    fig, ax = plt.subplots(figsize=(13.6, 5.4))
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    # ---- left: three orifice inputs ----
    box(ax, 0.15, 3.55, 1.75, 0.8, "Laparoscopy", "rigid MIS", WHITE, C_ENC)
    box(ax, 0.15, 2.55, 1.75, 0.8, "GI endoscopy", "flexible / capsule", WHITE, C_ENC)
    box(ax, 0.15, 1.55, 1.75, 0.8, "Bronchoscopy", "ION / airway", WHITE, C_ENC)
    ax.text(1.02, 4.62, "19 datasets · 1,707 sequences", ha="center",
            fontsize=7.5, color="#4A5560", style="italic")

    # ---- shared encoder ----
    box(ax, 2.35, 2.15, 1.9, 1.6, "Shared encoder",
        "V-JEPA 2 ViT-L\nfrozen, $D{=}1024$", SOFT_ENC, C_ENC, 9)

    # ---- lane 1: passive forecast (Capability 1) ----
    box(ax, 4.75, 3.30, 2.15, 1.15, "Causal residual L1",
        "short-horizon forecast", SOFT_FC, C_FC)
    box(ax, 4.75, 2.00, 2.15, 1.0, "Coarse L2",
        "mid-horizon anatomy", SOFT_FC, C_FC)
    box(ax, 7.35, 2.65, 2.3, 1.15, "Capability 1",
        "forecast scene evolution\n0.978 cos, $p{<}10^{-80}$", WHITE, C_FC, 9)

    # ---- lane 2: physical grounding (Capabilities 2/3) ----
    box(ax, 4.75, 0.55, 2.15, 1.05, "SE(3) dynamics",
        "block-causal, ensemble\nrisk + covariance", SOFT_PHYS, C_PHYS)
    box(ax, 7.35, 0.55, 2.3, 1.05, "Capabilities 2 + 3",
        "action evaluation 83.1%\noffline navigation 51.5%", WHITE, C_PHYS, 9)

    # ---- right: clinical pathways ----
    box(ax, 10.35, 3.85, 2.95, 0.95, "Loss-of-view warning",
        "anticipate before losing sight", SOFT_APP, C_APP, 9)
    box(ax, 10.35, 2.55, 2.95, 0.95, "Camera-handling training",
        "score a motion's visual effect", SOFT_APP, C_APP, 9)
    box(ax, 10.35, 1.25, 2.95, 0.95, "Navigation assistance",
        "model-based planning prototype", SOFT_APP, C_APP, 9)

    # ---- arrows ----
    for y in (3.95, 2.95, 1.95):
        ax.plot([1.90, 2.12, 2.12, 2.35], [y, y, 2.95, 2.95], color=LINE, lw=1.2, zorder=1)
    arrow(ax, 2.12, 2.95, 2.35, 2.95)

    # encoder -> lane split
    ax.plot([4.25, 4.5, 4.5, 4.75], [2.95, 2.95, 3.87, 3.87], color=LINE, lw=1.2)
    arrow(ax, 4.5, 3.87, 4.75, 3.87)
    ax.plot([4.25, 4.5, 4.5, 4.75], [2.95, 2.95, 1.07, 1.07], color=LINE, lw=1.2)
    arrow(ax, 4.5, 1.07, 4.75, 1.07)

    arrow(ax, 6.90, 3.87, 7.35, 3.35)   # L1 -> Capability 1
    arrow(ax, 6.90, 2.50, 7.35, 3.05)   # L2 -> Capability 1
    arrow(ax, 6.90, 1.07, 7.35, 1.07)   # SE(3) -> Capabilities 2+3

    arrow(ax, 9.65, 3.35, 10.35, 4.32)  # Capability 1 -> loss-of-view
    arrow(ax, 9.65, 3.10, 10.35, 3.02)  # Capability 1/2 -> training
    arrow(ax, 9.65, 1.07, 10.35, 1.72)  # Capabilities 2+3 -> navigation

    # lane labels
    ax.text(6.05, 4.75, "Passive-video forecast (validated)", ha="center",
            fontsize=9, fontweight="bold", color=C_FC)
    ax.text(6.05, 0.12, "Physical grounding (audited; pose/depth-gated)",
            ha="center", fontsize=9, fontweight="bold", color=C_PHYS)
    ax.text(11.82, 5.05, "Clinical application pathways", ha="center",
            fontsize=9, fontweight="bold", color=C_APP)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figure1_pipeline.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "figure1_pipeline.pdf", bbox_inches="tight", facecolor="white")
    print("[figure1] wrote figures/figure1_pipeline.png/.pdf")


if __name__ == "__main__":
    main()
