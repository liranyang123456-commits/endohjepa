"""Figure 1: capability-driven Endo-HJEPA overview (journal flowchart).

Two dense horizontal lanes (validated forecast / audited physical grounding),
three clinical pathways on the right, and a bottom strip with the four
correctness-audit gates. Muted palette, orthogonal arrows, minimal blank area.

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


def box(ax, x, y, w, h, title, sub, fc, ec, fs=9, subfs=7.4):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.04",
                       linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=INK, zorder=3)
    if sub:
        ax.text(x + w / 2, y + h * 0.32, sub, ha="center", va="center",
                fontsize=subfs, color="#4A5560", zorder=3)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.4,
                                mutation_scale=11), zorder=1)


def main():
    fig, ax = plt.subplots(figsize=(13.8, 6.2))
    ax.set_xlim(0, 13.8)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    # ---- left: three orifice inputs (tall, dense) ----
    box(ax, 0.15, 4.30, 1.95, 1.00, "Laparoscopy", "rigid MIS\nCholecT50 / EndoVis", WHITE, C_ENC)
    box(ax, 0.15, 3.05, 1.95, 1.00, "GI endoscopy", "flexible / capsule\nKvasir / C3VD", WHITE, C_ENC)
    box(ax, 0.15, 1.80, 1.95, 1.00, "Bronchoscopy", "ION / airway\n(private)", WHITE, C_ENC)
    ax.text(1.12, 5.55, "19 datasets · 1,707 sequences", ha="center",
            fontsize=7.8, color="#4A5560", style="italic")

    # ---- shared encoder ----
    box(ax, 2.55, 2.65, 2.05, 1.90, "Shared encoder",
        "V-JEPA 2 ViT-L\nfrozen, $D{=}1024$\ndense tokens $z$", SOFT_ENC, C_ENC, 9.5)

    # ---- lane 1: passive forecast (Capability 1) ----
    box(ax, 5.05, 4.05, 2.35, 1.30, "Causal residual L1",
        "short-horizon forecast\nGPT-style rollout", SOFT_FC, C_FC)
    box(ax, 5.05, 2.75, 2.35, 1.05, "Coarse L2",
        "mid-horizon anatomy", SOFT_FC, C_FC)
    box(ax, 7.85, 3.30, 2.45, 1.35, "Capability 1",
        "forecast scene evolution\n0.978 cos, $p{<}10^{-80}$", WHITE, C_FC, 9.5)

    # ---- lane 2: physical grounding (Capabilities 2/3) ----
    box(ax, 5.05, 1.15, 2.35, 1.30, "SE(3) dynamics",
        "block-causal ensemble\nrisk + covariance", SOFT_PHYS, C_PHYS)
    box(ax, 7.85, 1.15, 2.45, 1.30, "Capabilities 2 + 3",
        "action evaluation 83.1%\noffline navigation 51.5%", WHITE, C_PHYS, 9.5)

    # ---- right: clinical pathways (tall) ----
    box(ax, 10.75, 4.30, 2.90, 1.05, "Loss-of-view warning",
        "anticipate before losing sight", SOFT_APP, C_APP, 9.5)
    box(ax, 10.75, 2.95, 2.90, 1.05, "Camera-handling training",
        "score a motion's visual effect", SOFT_APP, C_APP, 9.5)
    box(ax, 10.75, 1.60, 2.90, 1.05, "Navigation assistance",
        "model-based planning prototype", SOFT_APP, C_APP, 9.5)

    # ---- bottom strip: four audit gates ----
    gates = [
        ("Input-sensitivity tests", "history / action / domain"),
        ("Reprojection pose gate", "10.2 px $\\to$ 0.21 px"),
        ("Matched negative banks", "fixed, same-sequence"),
        ("Grouped CV + frozen test", "no test-set tuning"),
    ]
    gx = 0.15
    for title, sub in gates:
        box(ax, gx, 0.12, 3.28, 0.80, title, sub, SOFT_AUD, C_AUD, 8.5, 7.2)
        gx += 3.44
    ax.text(6.9, 1.02, "Correctness-audit gates (Section: Correctness auditing and evaluation gates)",
            ha="center", fontsize=8.2, color=C_AUD, fontweight="bold")

    # ---- arrows: inputs -> encoder ----
    for y in (4.80, 3.55, 2.30):
        ax.plot([2.10, 2.32, 2.32, 2.55], [y, y, 3.60, 3.60], color=LINE, lw=1.2, zorder=1)
    arrow(ax, 2.32, 3.60, 2.55, 3.60)

    # encoder -> lane split
    ax.plot([4.60, 4.82, 4.82, 5.05], [3.60, 3.60, 4.70, 4.70], color=LINE, lw=1.3)
    arrow(ax, 4.82, 4.70, 5.05, 4.70)
    ax.plot([4.60, 4.82, 4.82, 5.05], [3.60, 3.60, 1.80, 1.80], color=LINE, lw=1.3)
    arrow(ax, 4.82, 1.80, 5.05, 1.80)

    arrow(ax, 7.40, 4.70, 7.85, 4.10)   # L1 -> Capability 1
    arrow(ax, 7.40, 3.28, 7.85, 3.75)   # L2 -> Capability 1
    arrow(ax, 7.40, 1.80, 7.85, 1.80)   # SE(3) -> Capabilities 2+3

    arrow(ax, 10.30, 4.10, 10.75, 4.82)  # Capability 1 -> loss-of-view
    arrow(ax, 10.30, 3.75, 10.75, 3.48)  # Capability 1/2 -> training
    arrow(ax, 10.30, 1.80, 10.75, 2.12)  # Capabilities 2+3 -> navigation

    # lane + column labels
    ax.text(6.17, 5.55, "Passive-video forecast (validated)", ha="center",
            fontsize=9.5, fontweight="bold", color=C_FC)
    ax.text(6.17, 2.58, "Physical grounding (audited; pose/depth-gated)",
            ha="center", fontsize=9.5, fontweight="bold", color=C_PHYS)
    ax.text(12.20, 5.55, "Clinical application pathways", ha="center",
            fontsize=9.5, fontweight="bold", color=C_APP)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figure1_pipeline.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "figure1_pipeline.pdf", bbox_inches="tight", facecolor="white")
    print("[figure1] wrote figures/figure1_pipeline.png/.pdf")


if __name__ == "__main__":
    main()
