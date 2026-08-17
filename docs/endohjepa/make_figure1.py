"""Figure 1: Endo-HJEPA architecture schematic (matplotlib, vector-precise).

    python docs/endohjepa/make_figure1.py
Writes docs/endohjepa/figure1_architecture.png / .pdf
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

# CJK-capable font if present, else default
plt.rcParams["font.size"] = 10
plt.rcParams["axes.unicode_minus"] = False

C_ENC = "#dbeafe"   # encoder blue
C_L1 = "#dcfce7"    # L1 green
C_L2 = "#fef9c3"    # L2 yellow
C_L3 = "#fee2e2"    # L3 red
C_EN = "#f3e8ff"    # energy purple
C_DOM = "#ffedd5"   # domain orange
C_OUT = "#e0e7ff"   # output indigo
C_EDGE = "#334155"


def box(ax, x, y, w, h, text, fc, fs=9, bold=False, ec=C_EDGE):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                       linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=3, wrap=True)


def arrow(ax, x1, y1, x2, y2, text="", color=C_EDGE):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.3, color=color, zorder=1))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.03, text, ha="center", fontsize=7.5,
                color=color, style="italic")


def main():
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 7.2); ax.axis("off")

    # ---- inputs: three orifices ----
    ax.text(1.0, 6.55, "Endoscopic video clips (unified cross-orifice)", ha="center",
            fontsize=10, fontweight="bold")
    box(ax, 0.15, 5.0, 1.7, 1.0, "Laparoscopy\n(rigid)", C_DOM, 9)
    box(ax, 0.15, 3.6, 1.7, 1.0, "GI endoscopy\n(flexible)", C_DOM, 9)
    box(ax, 0.15, 2.2, 1.7, 1.0, "Bronchoscopy\n(ION)", C_DOM, 9)
    box(ax, 0.15, 0.7, 1.7, 1.0, "domain id\nd ∈ {l,g,b}", "#ffffff", 8)

    # ---- encoder ----
    box(ax, 2.4, 3.3, 2.2, 2.5, "Shared encoder\n\nV-JEPA 2 ViT-L\n(official, frozen)\n\n256px, tubelet 2\nD = 1024", C_ENC, 9.5, bold=True)
    ax.text(3.5, 6.05, "dense spatio-temporal tokens  z ∈ R^(T×N×D)", ha="center",
            fontsize=8.5, style="italic")
    box(ax, 2.4, 0.7, 2.2, 1.1, "endo masking:\nspecular ↓\ninstrument ↑\nSTIR chamfer", "#f8fafc", 8)

    # ---- hierarchy ----
    ax.text(8.0, 6.55, "Hierarchical JEPA predictors (domain-conditioned)", ha="center",
            fontsize=10, fontweight="bold")
    box(ax, 5.4, 4.7, 2.4, 1.5, "L1  dense / causal\nshort-horizon\ntissue·tool·camera\n(space+time attn)", C_L1, 8.5)
    box(ax, 8.0, 4.7, 2.2, 1.5, "L2  coarse\nmid-horizon\nanatomy·phase\n(t stride-2 pool)", C_L2, 8.5)
    box(ax, 10.4, 4.7, 2.4, 1.5, "L3  action-cond.\nVQ residual codebook\nz(t+1)-z(t) → K actions", C_L3, 8.5)
    box(ax, 8.0, 2.9, 2.2, 1.2, "Energy head\nE(z,a,z′)\ncontrastive", C_EN, 8.5)
    box(ax, 5.4, 2.9, 2.4, 1.2, "Uncertainty-weighted\nmulti-task loss\n+ VICReg + residual", "#f8fafc", 8)

    # ---- planning / output ----
    box(ax, 10.4, 2.4, 2.5, 1.5, "Latent MPC\nsample K-action seqs\n→ min-energy path\nto goal latent", C_OUT, 8.5, bold=True)
    ax.text(11.65, 1.95, "in-silico only", ha="center", fontsize=8, style="italic", color="#b91c1c")

    # ---- arrows: inputs -> encoder ----
    for y0 in (5.5, 4.1, 2.7):
        arrow(ax, 1.85, y0, 2.4, 4.4)
    arrow(ax, 1.0, 1.2, 2.4, 1.2)            # domain id -> endo masking
    arrow(ax, 1.85, 1.2, 2.4, 3.5, "")       # domain id up to encoder
    ax.text(2.05, 2.25, "domain\nembed", ha="center", fontsize=7, color=C_EDGE)

    # encoder -> L1 (dense tokens)
    arrow(ax, 4.6, 4.55, 5.4, 5.3, "dense z")
    # L1 -> L2 -> L3 (left to right)
    arrow(ax, 7.8, 5.45, 8.0, 5.45)
    arrow(ax, 10.2, 5.45, 10.4, 5.45)
    # L3 -> energy (down) and energy -> MPC (down); L3 -> MPC (right)
    arrow(ax, 11.0, 4.7, 9.6, 4.1, "")       # L3 -> energy
    arrow(ax, 9.1, 3.5, 10.4, 3.1, "energy")  # energy -> MPC
    arrow(ax, 12.2, 4.7, 12.2, 3.9, "")      # L3 -> MPC (planned actions)
    ax.text(12.55, 4.3, "actions", ha="left", fontsize=7.5, color=C_EDGE, style="italic")
    # uncertainty/regularisation feeds L1/L2/L3 (dashed note)
    arrow(ax, 6.6, 4.1, 6.6, 4.7, "")        # reg -> L1

    # footnote
    ax.text(6.75, 0.25,
            "Predict plannable representations, not pixels.  Splits are video-level; sampling is domain-balanced.",
            ha="center", fontsize=9, style="italic", color="#334155")

    plt.tight_layout()
    out_png = "docs/endohjepa/figure1_architecture.png"
    out_pdf = "docs/endohjepa/figure1_architecture.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[figure1] wrote {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
