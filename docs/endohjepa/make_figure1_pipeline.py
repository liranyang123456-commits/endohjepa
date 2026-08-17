"""Figure 1: horizontal, orthogonal, muted-color pipeline (journal flowchart).

    python docs/endohjepa/make_figure1_pipeline.py
Not a rainbow box salad: one row, right-angle connectors, three chip colours.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent / "figures"

INK = "#2D3741"
LINE = "#A0A8B0"
CHIP_A = "#5A8CBE"   # encoder
CHIP_B = "#50969B"   # predictors
CHIP_C = "#82739B"   # energy / plan
SOFT_A = "#F2F7FC"
SOFT_B = "#F2F9F9"
SOFT_C = "#F7F4FA"
IO = "#FFFFFF"
GOAL = "#E8F5F0"


def box(ax, x, y, w, h, title, sub, fc, ec, fs=8.5):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.04",
                       linewidth=1.15, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=INK, zorder=3)
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
            fontsize=7.2, color="#4A5560", zorder=3)


def h_arrow(ax, x1, x2, y):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.4,
                                mutation_scale=10), zorder=1)


def main():
    fig, ax = plt.subplots(figsize=(13.4, 4.15))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 4.15)
    ax.axis("off")

    # column x positions (grid)
    xs = [0.18, 2.15, 4.35, 6.55, 8.75, 10.95]
    w, h = 1.82, 1.15
    y_mid = 1.85

    # inputs stacked left
    box(ax, 0.18, 2.85, 1.82, 0.85, "Laparoscopy", "rigid MIS", IO, CHIP_A)
    box(ax, 0.18, 1.85, 1.82, 0.85, "GI endoscopy", "flexible / capsule", IO, CHIP_A)
    box(ax, 0.18, 0.85, 1.82, 0.85, "Bronchoscopy", "ION / airway", IO, CHIP_A)

    # encoder
    box(ax, 2.35, 1.45, 1.95, 1.95, "Shared encoder",
        "V-JEPA 2 ViT-L\nfrozen  $D{=}1024$", SOFT_A, CHIP_A, 9)

    # tokens
    box(ax, 4.55, 1.70, 1.70, 1.45, "Tokens $z$",
        r"$T'\times N\times D$"+"\n+ domain $e_d$", IO, INK)

    # validated predictors + physically grounded state path
    box(ax, 6.50, 2.85, 1.95, 0.95, "L1  causal AR",
        "short-horizon  residual", SOFT_B, CHIP_B)
    box(ax, 6.50, 1.70, 1.95, 0.95, "L2  coarse",
        "mid-horizon  pool$\\times$2", SOFT_B, CHIP_B)
    box(ax, 6.50, 0.55, 1.95, 0.95, "Factorised state",
        "$s_g,s_q,s_m$ / nuisance $\\xi$", SOFT_B, CHIP_B)

    # probabilistic physical dynamics
    box(ax, 8.70, 1.55, 1.85, 1.75, "SE(3) dynamics",
        "block-causal ensemble\nrisk + covariance", SOFT_C, CHIP_C)

    # output
    box(ax, 10.80, 1.45, 2.35, 1.95, "Safe continuous MPC",
        "CEM / MPPI\nhard reject + zero motion\noffline only", GOAL, "#2F6B4F", 9)

    # orthogonal connectors: inputs -> encoder
    for y in (3.27, 2.27, 1.27):
        ax.plot([2.00, 2.17, 2.17, 2.35], [y, y, 2.42, 2.42], color=LINE, lw=1.2, zorder=1)
    ax.annotate("", xy=(2.35, 2.42), xytext=(2.28, 2.42),
                arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.2, mutation_scale=9))

    h_arrow(ax, 4.30, 4.55, 2.42)
    # tokens fan to L1/L2/L3
    ax.plot([6.25, 6.40, 6.40], [2.42, 2.42, 3.32], color=LINE, lw=1.2)
    ax.plot([6.40, 6.50], [3.32, 3.32], color=LINE, lw=1.2)
    ax.plot([6.40, 6.50], [2.17, 2.17], color=LINE, lw=1.2)
    ax.plot([6.40, 6.40, 6.50], [2.42, 1.02, 1.02], color=LINE, lw=1.2)
    for y in (3.32, 2.17, 1.02):
        ax.annotate("", xy=(6.50, y), xytext=(6.45, y),
                    arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.2, mutation_scale=8))

    # grounded state -> physical dynamics -> safe MPC
    ax.plot([8.45, 8.58, 8.58, 8.70], [1.02, 1.02, 2.00, 2.00],
            color=LINE, lw=1.2)
    ax.annotate("", xy=(8.70, 2.00), xytext=(8.62, 2.00),
                arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.2,
                                mutation_scale=9))
    h_arrow(ax, 10.55, 10.80, 2.42)

    ax.text(6.75, 3.95, "Predictive foundation + grounded state", ha="center",
            fontsize=9, fontweight="bold", color=INK)
    ax.text(6.7, 0.18,
            "Forecast from passive video; ground control only with aligned pose/depth.  No pixel decoder.",
            ha="center", fontsize=8, style="italic", color="#4A5560")

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figure1_pipeline.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "figure1_pipeline.pdf", bbox_inches="tight", facecolor="white")
    print("[figure1] wrote figures/figure1_pipeline.png/.pdf")


if __name__ == "__main__":
    main()
