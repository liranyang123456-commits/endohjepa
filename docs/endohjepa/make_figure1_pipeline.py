"""Figure 1: Endo-HJEPA overview.

Layout rules enforced here:

* four strict rows (three passive-video domains plus the pose-gated physical
  lane) and five aligned columns: input, shared encoder, prediction module,
  audited capability, result;
* every thumbnail is a complete image, letterboxed into a fixed slot so that
  the native aspect ratio is preserved and nothing is stretched or cut;
* no text is drawn on top of an image, all captions sit below their image;
* the shared-encoder column contains a small network diagram in the same
  PlotNeuralNet idiom as Figure 2, not a decorative glyph;
* all connectors are axis-aligned (horizontal / vertical only) and start and
  end exactly on a box or image border;
* column gaps carry a connector and its label and nothing else, so the space
  between two modules is the width of one label rather than a free margin;
  the prediction and capability columns are tall enough to expose the L1/L2
  split and the measured scaling curve instead of leaving the band empty.

    python docs/endohjepa/make_result_thumbs.py    # clean thumbnails first
    python docs/endohjepa/make_figure1_pipeline.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from PIL import Image

OUT = Path(__file__).resolve().parent / "figures"
METRICS = json.loads(
    (Path(__file__).resolve().parent / "verified_metrics.json").read_text(
        encoding="utf-8"
    )
)
FORECAST = METRICS["forecast_6000"]
ACTION_AUDIT = METRICS["grounded_upgrade"]["continuous_action_v2"][
    "scared_action_audit"
]
NAV_PROXY = METRICS["grounded_upgrade"]["navigation_v2"]

INK = "#2D3741"
MUTED = "#4A5560"
LINE = "#8E979F"
C_ENC = "#4E7FA6"
C_FC = "#3F8A8A"
C_PHYS = "#7E6E97"
C_APP = "#3F7A5A"
C_AUD = "#8C6D4F"
SOFT_ENC = "#EDF3F9"
SOFT_FC = "#EDF6F6"
SOFT_PHYS = "#F3F1F7"
SOFT_AUD = "#F8F4EF"
WHITE = "#FFFFFF"

# columns (left, right edges); gaps are one connector label wide
IN_L, IN_R = 0.30, 2.40
ENC_L, ENC_R = 2.90, 5.40
MOD_L, MOD_R = 5.90, 8.60
CAP_L, CAP_R = 9.10, 11.90
RISER = 12.20
RES_L, RES_R = 13.16, 15.26

FIG_W, FIG_H = 15.56, 9.75

# rows (vertical centres)
ROW_FC = [8.55, 6.70, 4.85]
ROW_PHYS = 2.25
ENC_TOP, ENC_BOT = 9.50, 3.80
BRANCH_X = 5.05  # drops from the encoder floor into the physical lane
BRANCH_Y = 3.50
MOD_H_A = 3.56  # tall lane-A modules span all three domain rows
MOD_H_B = 1.90
GATE_Y = 0.38

SLOT_W, SLOT_H = 2.10, 1.30

# Connectors must sit above module fills (zorder 2) and glyphs (zorder 6-7),
# otherwise the segments that run inside a box are hidden by its face colour.
Z_WIRE = 8

# Measured forecast scaling, mirrored from the verified metric ledger. The
# 13,552-clip point comes from a different validation cache, so it is drawn
# open and dashed exactly as in Figure 2 of the experiments section.
SCALE_CLIPS = METRICS["scale_curve"]["clips"]
SCALE_COS = METRICS["scale_curve"]["cos"]
PERSISTENCE_COS = FORECAST["persistence"]["cos"]

ROWS_IN = [
    ("_fig1_in_laparo.png", "CholecT50 (laparoscopy)", "last observed frame"),
    ("_fig1_in_gi.png", "Kvasir-Capsule (GI)", "last observed frame"),
    ("_fig1_in_bronch.png", "ION (bronchoscopy)", "navigation-console capture"),
]
ROWS_OUT = [
    ("_fig1_out_laparo.png", "CholecT50 (laparoscopy)", "predicted-latent retrieval"),
    ("_fig1_out_gi.png", "Kvasir-Capsule (GI)", "predicted-latent retrieval"),
    ("_fig1_out_bronch.png", "ION (bronchoscopy)", "predicted-latent retrieval"),
]
APPS = [
    "loss-of-view\nwarning",
    "camera-handling\ntraining",
    "cross-domain\nrepresentation",
]


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def thumb(ax, name, slot_cx, cy, ec, label, sub):
    """Full image letterboxed into the fixed slot; captions below the frame."""
    with Image.open(OUT / name) as image:
        arr = np.asarray(image.convert("RGB"))
    h_px, w_px = arr.shape[:2]
    scale = min(SLOT_W / w_px, SLOT_H / h_px)
    w, h = w_px * scale, h_px * scale
    x0, y0 = slot_cx - w / 2, cy - h / 2
    ax.imshow(arr, extent=[x0, x0 + w, y0, y0 + h], zorder=3)
    ax.add_patch(
        Rectangle((x0, y0), w, h, fill=False, edgecolor=ec, linewidth=1.1, zorder=4)
    )
    ax.text(
        slot_cx,
        y0 - 0.10,
        label,
        ha="center",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        slot_cx,
        y0 - 0.33,
        sub,
        ha="center",
        va="top",
        fontsize=7.0,
        color=MUTED,
        style="italic",
    )
    return x0, x0 + w


def frame(ax, left, right, cy, h, fc, ec, lw=1.2):
    ax.add_patch(
        FancyBboxPatch(
            (left, cy - h / 2),
            right - left,
            h,
            boxstyle="round,pad=0,rounding_size=0.05",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
            zorder=2,
        )
    )


def module(ax, left, right, cy, title, lines, fc, ec, glyph, h=MOD_H_B):
    """Framed module: bold title, glyph, then description lines."""
    frame(ax, left, right, cy, h, fc, ec)
    cx = (left + right) / 2
    title_size = 8.2 if len(title) > 23 else 9.0
    ax.text(
        cx,
        cy + h / 2 - 0.18,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=INK,
        zorder=6,
    )
    glyph(ax, cx, cy + 0.20)
    ax.text(
        cx,
        cy - h / 2 + 0.16,
        lines,
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=MUTED,
        linespacing=1.45,
        zorder=6,
    )


def subbox(ax, left, right, top, h, title, lines, ec):
    """Nested block inside a tall module."""
    ax.add_patch(
        FancyBboxPatch(
            (left, top - h),
            right - left,
            h,
            boxstyle="round,pad=0,rounding_size=0.04",
            linewidth=0.9,
            edgecolor=ec,
            facecolor=WHITE,
            zorder=5,
        )
    )
    cx = (left + right) / 2
    ax.text(
        cx,
        top - 0.17,
        title,
        ha="center",
        va="top",
        fontsize=7.8,
        fontweight="bold",
        color=INK,
        zorder=6,
    )
    ax.text(
        cx,
        top - h + 0.12,
        lines,
        ha="center",
        va="bottom",
        fontsize=6.9,
        color=MUTED,
        linespacing=1.40,
        zorder=6,
    )


def harrow(ax, x1, x2, y, label=None):
    ax.plot(
        [x1, x2 - 0.05],
        [y, y],
        color=LINE,
        lw=1.25,
        zorder=Z_WIRE,
        solid_capstyle="butt",
    )
    ax.annotate(
        "",
        xy=(x2, y),
        xytext=(x2 - 0.05, y),
        arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.25, mutation_scale=10),
        zorder=Z_WIRE,
    )
    if label:
        ax.text(
            (x1 + x2) / 2,
            y + 0.09,
            label,
            ha="center",
            va="bottom",
            fontsize=6.8,
            color=MUTED,
            style="italic",
        )


def varrow(ax, x, y1, y2):
    """Vertical connector with the arrow head at y2."""
    step = -0.05 if y2 < y1 else 0.05
    ax.plot(
        [x, x],
        [y1, y2 - step],
        color=LINE,
        lw=1.25,
        zorder=Z_WIRE,
        solid_capstyle="butt",
    )
    ax.annotate(
        "",
        xy=(x, y2),
        xytext=(x, y2 - step),
        arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.25, mutation_scale=10),
        zorder=Z_WIRE,
    )


def route_hvh(ax, x1, y1, x2, y2, xm):
    """Horizontal, vertical, horizontal connector ending at (x2, y2)."""
    ax.plot(
        [x1, xm], [y1, y1], color=LINE, lw=1.25, zorder=Z_WIRE, solid_capstyle="butt"
    )
    if abs(y2 - y1) > 1e-6:
        ax.plot(
            [xm, xm],
            [y1, y2],
            color=LINE,
            lw=1.25,
            zorder=Z_WIRE,
            solid_capstyle="butt",
        )
    harrow(ax, xm, x2, y2)


def route_vhv(ax, x1, y1, ym, x2, y2):
    """Vertical, horizontal, vertical connector ending on top of a box."""
    ax.plot(
        [x1, x1], [y1, ym], color=LINE, lw=1.25, zorder=Z_WIRE, solid_capstyle="butt"
    )
    ax.plot(
        [x1, x2], [ym, ym], color=LINE, lw=1.25, zorder=Z_WIRE, solid_capstyle="butt"
    )
    varrow(ax, x2, ym, y2)


def plate(ax, x, y, w, h, depth, fc, ec, lw=0.9):
    """PlotNeuralNet-style extruded box; (x, y) is the front-face lower left."""
    dx, dy = depth * 0.62, depth * 0.48
    ax.add_patch(
        Polygon(
            [
                (x, y + h),
                (x + dx, y + h + dy),
                (x + w + dx, y + h + dy),
                (x + w, y + h),
            ],
            closed=True,
            facecolor=fc,
            edgecolor=ec,
            lw=lw,
            alpha=0.92,
            zorder=6,
        )
    )
    ax.add_patch(
        Polygon(
            [
                (x + w, y),
                (x + w + dx, y + dy),
                (x + w + dx, y + h + dy),
                (x + w, y + h),
            ],
            closed=True,
            facecolor=fc,
            edgecolor=ec,
            lw=lw,
            alpha=0.72,
            zorder=6,
        )
    )
    ax.add_patch(Rectangle((x, y), w, h, facecolor=WHITE, edgecolor="none", zorder=6))
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw, alpha=0.88, zorder=7)
    )


def plate_stack(ax, cx, cy, w, h, depth, n, fc, gap=0.11):
    total = w + (n - 1) * gap
    x = cx - total / 2
    for k in range(n - 1, -1, -1):
        plate(ax, x + k * gap, cy - h / 2, w, h, depth, fc, C_ENC)


def token_grid(ax, cx, cy, cols=8, rows=2, cw=0.16, ch=0.14, gap=0.035):
    w = cols * cw + (cols - 1) * gap
    h = rows * ch + (rows - 1) * gap
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(
                Rectangle(
                    (cx - w / 2 + c * (cw + gap), cy - h / 2 + r * (ch + gap)),
                    cw,
                    ch,
                    facecolor="#DCE9F3",
                    edgecolor=C_ENC,
                    lw=0.7,
                    zorder=7,
                )
            )


# --------------------------------------------------------------------------- #
# module glyphs
# --------------------------------------------------------------------------- #
def glyph_forecast(ax, cx, cy, w=0.98, h=0.42):
    x, y = cx - w / 2, cy - h / 2
    t = np.linspace(0, 1, 24)
    ax.plot(
        [x, x + w * 0.42], [y + h * 0.28, y + h * 0.28], color=INK, lw=1.3, zorder=6
    )
    ax.scatter(
        x + w * np.linspace(0.04, 0.38, 4),
        np.full(4, y + h * 0.28),
        s=7,
        color=INK,
        zorder=7,
    )
    ax.plot(
        x + w * (0.42 + 0.58 * t),
        y + h * (0.28 + 0.64 * t),
        color=C_FC,
        lw=1.5,
        zorder=6,
    )
    ax.plot(
        [x + w * 0.42, x + w],
        [y + h * 0.28, y + h * 0.28],
        color=LINE,
        lw=1.0,
        ls="--",
        zorder=6,
    )
    ax.scatter([x + w], [y + h * 0.92], s=14, color=C_FC, zorder=7)


def glyph_se3(ax, cx, cy, s=0.40):
    x0, y0 = cx - 0.10, cy - 0.14
    ax.annotate(
        "",
        xy=(x0 + s, y0),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color="#B84A4A", lw=1.5),
        zorder=6,
    )
    ax.annotate(
        "",
        xy=(x0, y0 + s * 0.80),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color="#3F7A5A", lw=1.5),
        zorder=6,
    )
    ax.annotate(
        "",
        xy=(x0 - s * 0.46, y0 - s * 0.34),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=C_ENC, lw=1.5),
        zorder=6,
    )


def glyph_navigation(ax, cx, cy, w=1.00, h=0.42):
    x, y = cx - w / 2, cy - h / 2
    t = np.linspace(0, 1, 40)
    px = x + w * t
    py = y + h * (0.16 + 0.74 * np.sin(t * np.pi * 0.85))
    ax.plot(px, py, color=C_PHYS, lw=1.5, zorder=6)
    ax.scatter([px[0]], [py[0]], s=16, color=INK, zorder=7)
    ax.scatter([px[-1]], [py[-1]], s=52, marker="*", color=C_APP, zorder=7)


def scaling_inset(ax, left, right, bottom, top):
    """Measured forecast cosine against training clips, drawn in place."""
    clips = np.log10(np.asarray(SCALE_CLIPS, dtype=float))
    cosine = np.asarray(SCALE_COS)
    low, high = 0.908, 0.984
    px = left + (clips - clips[0]) / (clips[-1] - clips[0]) * (right - left)
    py = bottom + (cosine - low) / (high - low) * (top - bottom)
    baseline = bottom + (PERSISTENCE_COS - low) / (high - low) * (top - bottom)
    ax.plot([left, left], [bottom, top], color=LINE, lw=0.8, zorder=6)
    ax.plot([left, right], [bottom, bottom], color=LINE, lw=0.8, zorder=6)
    ax.plot(
        [left, right], [baseline, baseline], color="#B0888A", lw=1.0, ls="--", zorder=6
    )
    ax.plot(px[:-1], py[:-1], color=C_FC, lw=1.5, zorder=6)
    ax.plot(px[-2:], py[-2:], color=C_FC, lw=1.2, ls="--", zorder=6)
    ax.scatter(px[:-1], py[:-1], s=9, color=C_FC, zorder=7)
    ax.scatter(
        px[-1:],
        py[-1:],
        s=11,
        facecolor=WHITE,
        edgecolor=C_FC,
        linewidths=0.9,
        zorder=7,
    )
    ax.text(
        right,
        baseline - 0.04,
        f"persistence {PERSISTENCE_COS:.3f}",
        ha="right",
        va="top",
        fontsize=5.9,
        color="#9A6C6E",
        zorder=6,
    )
    ax.text(
        right,
        top + 0.01,
        f"{SCALE_COS[-1]:.3f}",
        ha="right",
        va="bottom",
        fontsize=5.9,
        color=C_FC,
        zorder=6,
    )
    ax.text(
        left - 0.04,
        (bottom + top) / 2,
        "cosine",
        ha="center",
        va="center",
        fontsize=5.9,
        color=MUTED,
        rotation=90,
        zorder=6,
    )
    ax.text(
        (left + right) / 2,
        bottom - 0.06,
        "training clips 0.5k $\\to$ 13.5k",
        ha="center",
        va="top",
        fontsize=5.9,
        color=MUTED,
        zorder=6,
    )


# --------------------------------------------------------------------------- #
# tall lane-A modules
# --------------------------------------------------------------------------- #
def hierarchical_predictor(ax, cy):
    frame(ax, MOD_L, MOD_R, cy, MOD_H_A, SOFT_FC, C_FC)
    cx, top = (MOD_L + MOD_R) / 2, cy + MOD_H_A / 2
    ax.text(
        cx,
        top - 0.18,
        "Hierarchical predictor",
        ha="center",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color=INK,
        zorder=6,
    )
    ax.text(
        cx,
        top - 0.42,
        "two temporal scales, encoder space only",
        ha="center",
        va="top",
        fontsize=7.0,
        color=MUTED,
        style="italic",
        zorder=6,
    )
    inner_l, inner_r = MOD_L + 0.20, MOD_R - 0.20
    subbox(
        ax,
        inner_l,
        inner_r,
        top - 0.62,
        1.12,
        "L1: pooled causal residual",
        "GPT-style causal transformer\nhorizon 4 tubelets\nfrozen-teacher targets",
        C_FC,
    )
    subbox(
        ax,
        inner_l,
        inner_r,
        top - 2.08,
        1.12,
        "L2: pooled coarse residual",
        "temporally pooled tokens\nmid-horizon abstraction\ndomain embedding $e_d$",
        C_FC,
    )
    varrow(ax, cx, top - 1.74, top - 1.96)
    ax.text(
        cx + 0.10,
        top - 1.85,
        "pooled state",
        ha="left",
        va="center",
        fontsize=6.4,
        color=MUTED,
        style="italic",
        zorder=9,
    )
    ax.text(
        cx,
        cy - MOD_H_A / 2 + 0.09,
        "cosine target; no pixel decoder",
        ha="center",
        va="bottom",
        fontsize=6.6,
        color=C_FC,
        zorder=6,
        linespacing=1.35,
    )


def forecasting_capability(ax, cy):
    frame(ax, CAP_L, CAP_R, cy, MOD_H_A, WHITE, C_FC)
    cx, top = (CAP_L + CAP_R) / 2, cy + MOD_H_A / 2
    bottom = cy - MOD_H_A / 2
    ax.text(
        cx,
        top - 0.18,
        "Capability 1: forecasting",
        ha="center",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color=INK,
        zorder=6,
    )
    ax.text(
        cx,
        top - 0.42,
        "predict future encoder states",
        ha="center",
        va="top",
        fontsize=7.0,
        color=MUTED,
        style="italic",
        zorder=6,
    )
    scaling_inset(ax, CAP_L + 0.55, CAP_R - 0.30, top - 1.98, top - 0.70)
    ax.text(
        cx,
        bottom + 0.16,
        f"cosine ${FORECAST['causal_l1']['cos']:.3f}$ vs "
        f"${PERSISTENCE_COS:.3f}$ persistence\n"
        "shared 750-clip validation protocol\n"
        "monotone gain, plateau at $13{,}552$ clips\n"
        "video-level split, held-out validation",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=MUTED,
        linespacing=1.45,
        zorder=6,
    )


def shared_encoder(ax):
    """Shared-encoder column drawn as a compact vertical network diagram."""
    frame(ax, ENC_L, ENC_R, (ENC_TOP + ENC_BOT) / 2, ENC_TOP - ENC_BOT, SOFT_ENC, C_ENC)
    cx = (ENC_L + ENC_R) / 2
    ax.text(
        cx,
        ENC_TOP - 0.18,
        "Shared encoder",
        ha="center",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color=INK,
        zorder=6,
    )
    ax.text(
        cx,
        ENC_TOP - 0.44,
        "frozen V-JEPA 2 ViT-L",
        ha="center",
        va="top",
        fontsize=7.3,
        color=MUTED,
        style="italic",
        zorder=6,
    )

    stages = [
        (
            8.45,
            0.32,
            lambda x, y: plate_stack(ax, x, y, 0.40, 0.62, 0.20, 3, "#CFE0EE"),
            "endoscopic clip, 16 frames",
        ),
        (
            7.24,
            0.25,
            lambda x, y: plate_stack(ax, x, y, 0.54, 0.46, 0.20, 1, "#BFD6E9"),
            "tubelet embed $2{\\times}16{\\times}16$",
        ),
        (
            6.03,
            0.33,
            lambda x, y: plate_stack(ax, x, y, 0.32, 0.64, 0.16, 4, "#A9C8E0"),
            "ViT-L $\\times\\,24$, space-time attention",
        ),
        (
            4.82,
            0.18,
            lambda x, y: token_grid(ax, x, y),
            "spatially pooled tokens $\\bar z$, $8{\\times}1024$",
        ),
    ]
    for index, (cy, half, draw, caption) in enumerate(stages):
        draw(cx, cy)
        ax.text(
            cx,
            cy - half - 0.05,
            caption,
            ha="center",
            va="top",
            fontsize=7.0,
            color=MUTED,
            zorder=6,
        )
        if index + 1 < len(stages):
            next_cy, next_half = stages[index + 1][0], stages[index + 1][1]
            varrow(ax, cx, cy - half - 0.30, next_cy + next_half + 0.08)
    ax.text(
        cx,
        ENC_BOT + 0.10,
        "$+$ domain embedding $e_d$\none encoder for all three domains",
        ha="center",
        va="bottom",
        fontsize=6.9,
        color=C_ENC,
        zorder=6,
        linespacing=1.35,
    )


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")

    in_cx, res_cx = (IN_L + IN_R) / 2, (RES_L + RES_R) / 2
    mod_cx = (MOD_L + MOD_R) / 2
    mid_fc = ROW_FC[1]

    shared_encoder(ax)
    hierarchical_predictor(ax, mid_fc)
    forecasting_capability(ax, mid_fc)

    module(
        ax,
        ENC_L,
        ENC_R,
        ROW_PHYS,
        "Tubelet-aligned SE(3) actions",
        "$u_t=\\log(T_t^{-1}T_{t+1})$, camera frame\n"
        "one twist per 2-frame tubelet\ndepth gives near-wall risk labels",
        SOFT_PHYS,
        C_PHYS,
        glyph_se3,
    )
    module(
        ax,
        MOD_L,
        MOD_R,
        ROW_PHYS,
        "SE(3) latent dynamics",
        "block-causal action path\nprobabilistic ensemble, $\\mu,\\Sigma$\n"
        "risk head unused (AUC 0.523 fail)",
        SOFT_PHYS,
        C_PHYS,
        glyph_forecast,
    )
    module(
        ax,
        CAP_L,
        CAP_R,
        ROW_PHYS,
        "SE(3)-conditioned evaluation",
        f"batch-shuffled action win "
        f"{100 * ACTION_AUDIT['batch_shuffled']['real_action_win_fraction']:.1f} % "
        f"($n={ACTION_AUDIT['n_windows']}$)\n"
        f"fixed-bank pair win "
        f"{100 * ACTION_AUDIT['fixed_same_sequence_bank']['pair_win_fraction']:.1f} %\n"
        f"forced-future oracle retrieval "
        f"{100 * NAV_PROXY['proxy_win_fraction']:.1f} % "
        "(diagnostic only)",
        WHITE,
        C_PHYS,
        glyph_navigation,
    )

    # ---- passive-video rows -------------------------------------------------
    for cy, src, dst, app in zip(ROW_FC, ROWS_IN, ROWS_OUT, APPS):
        _, in_right = thumb(ax, src[0], in_cx, cy, C_FC, src[1], src[2])
        out_left, _ = thumb(ax, dst[0], res_cx, cy, C_FC, dst[1], dst[2])
        harrow(ax, in_right, ENC_L, cy)
        route_hvh(ax, CAP_R, mid_fc, out_left, cy, RISER)
        ax.text(
            (RISER + out_left) / 2,
            cy + 0.11,
            app,
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=C_APP,
            style="italic",
            linespacing=1.3,
        )

    harrow(ax, ENC_R, MOD_L, mid_fc, label="tokens $z$")
    harrow(ax, MOD_R, CAP_L, mid_fc, label="forecast $\\hat{z}$")

    # ---- pose-gated physical row -------------------------------------------
    _, in_right = thumb(
        ax,
        "_fig1_in_scared.png",
        in_cx,
        ROW_PHYS,
        C_PHYS,
        "SCARED (laparoscopy)",
        "frame + calibrated pose",
    )
    out_left, _ = thumb(
        ax,
        "_fig1_se3.png",
        res_cx,
        ROW_PHYS,
        C_PHYS,
        "SCARED (laparoscopy)",
        "SE(3)-conditioned retrieval",
    )
    harrow(ax, in_right, ENC_L, ROW_PHYS, label="pose")
    harrow(ax, ENC_R, MOD_L, ROW_PHYS, label="twist $u$")
    harrow(ax, MOD_R, CAP_L, ROW_PHYS, label="$\\hat{z}$ (risk off)")
    harrow(ax, CAP_R, out_left, ROW_PHYS, label="retrieval proxy")
    route_vhv(ax, BRANCH_X, ENC_BOT, BRANCH_Y, mod_cx, ROW_PHYS + MOD_H_B / 2)
    ax.text(
        mod_cx - 0.10,
        BRANCH_Y + 0.09,
        "same frozen tokens",
        ha="right",
        va="bottom",
        fontsize=6.8,
        color=MUTED,
        style="italic",
    )

    # ---- audit gates --------------------------------------------------------
    gates = [
        "Input-sensitivity tests\nhistory / action / domain",
        "C3VD depth-warp diagnostic\nconvention not validated",
        "Action negatives\nbatch shuffle / fixed bank",
        "Video/patient-grouped splits\nlogged audit-set contacts",
    ]
    width = (RES_R - IN_L - 3 * 0.28) / 4
    for index, gate in enumerate(gates):
        left = IN_L + index * (width + 0.28)
        ax.add_patch(
            FancyBboxPatch(
                (left, GATE_Y - 0.32),
                width,
                0.64,
                boxstyle="round,pad=0,rounding_size=0.05",
                linewidth=1.0,
                edgecolor=C_AUD,
                facecolor=SOFT_AUD,
                zorder=2,
            )
        )
        ax.text(
            left + width / 2,
            GATE_Y,
            gate,
            ha="center",
            va="center",
            fontsize=7.2,
            color=INK,
            linespacing=1.4,
            zorder=6,
        )

    # ---- lane titles --------------------------------------------------------
    # The lane-A title is wider than the input column, so it is lifted clear of
    # the encoder box rather than allowed to run across its border.
    ax.text(
        IN_L,
        9.55,
        "Lane A: passive-video forecasting",
        ha="left",
        va="bottom",
        fontsize=8.4,
        fontweight="bold",
        color=C_FC,
    )
    ax.text(
        IN_L,
        9.33,
        "validated across three domains",
        ha="left",
        va="bottom",
        fontsize=7.6,
        color=C_FC,
        style="italic",
    )
    ax.text(
        IN_L,
        3.24,
        "Lane B: physical grounding",
        ha="left",
        va="bottom",
        fontsize=9.6,
        fontweight="bold",
        color=C_PHYS,
    )
    ax.text(
        IN_L,
        3.05,
        "audited on pose-gated data only",
        ha="left",
        va="bottom",
        fontsize=7.6,
        color=C_PHYS,
        style="italic",
    )
    ax.text(
        IN_L,
        0.80,
        "Correctness audits applied to both lanes",
        ha="left",
        va="bottom",
        fontsize=8.4,
        fontweight="bold",
        color=C_AUD,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT / "figure1_pipeline.png", dpi=220, bbox_inches="tight", facecolor="white"
    )
    fig.savefig(OUT / "figure1_pipeline.pdf", bbox_inches="tight", facecolor="white")
    print("[figure1] wrote figures/figure1_pipeline.png/.pdf")


if __name__ == "__main__":
    main()
