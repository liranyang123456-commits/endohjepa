"""Generate all Endo-HJEPA result figures from verified_metrics.json.

    python docs/endohjepa/make_figures.py
Writes PNG+PDF under docs/endohjepa/figures/.
Every plotted number is read from verified_metrics.json (no hard-coded drift).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_names import display  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / "figures"
M = json.loads((HERE / "verified_metrics.json").read_text(encoding="utf-8"))

# Journal-safe muted palette (print-friendly, no neon)
C_OURS = "#2F5D8A"
C_GRU = "#B85C38"
C_MAMBA = "#C48A2A"
C_QUERY = "#6B7280"
C_PERS = "#9AA3AD"
C_GI = "#3D7A6A"
C_LAP = "#2F5D8A"
C_BR = "#8A5A2F"
C_ACC = "#3D7A6A"
C_MAP = "#B85C38"


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] {name}")


def fig_scale_horizon_forecast():
    """Figure 2: scale curve, horizon vs GRU, forecast bars."""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))

    sc = M["scale_curve"]
    ax = axes[0]
    # The last point uses a larger, separately built validation cache. Draw it
    # as an open marker and dashed continuation rather than a homogeneous curve.
    ax.plot(
        sc["clips"][:5],
        sc["cos"][:5],
        "o-",
        color=C_OURS,
        lw=2.0,
        ms=6,
        label="Endo-HJEPA (shared 750-val)",
    )
    ax.plot(sc["clips"][4:], sc["cos"][4:], "--", color=C_OURS, lw=1.2)
    ax.plot(
        sc["clips"][-1],
        sc["cos"][-1],
        marker="o",
        ms=7,
        markerfacecolor="white",
        markeredgecolor=C_OURS,
        markeredgewidth=1.5,
        linestyle="none",
        label="13.6k (different 1,631-val)",
    )
    ax.axhline(
        M["forecast_6000"]["gru"]["cos"],
        color=C_GRU,
        ls="--",
        lw=1.4,
        label="GRU (6k clips)",
    )
    ax.axhline(
        M["forecast_6000"]["persistence"]["cos"],
        color=C_PERS,
        ls=":",
        lw=1.6,
        label="persistence",
    )
    ax.set_xlabel("Training clips (video-level)")
    ax.set_ylabel("Mean forecast cosine (steps 1--4)")
    ax.set_title("(a) Data-scale curve")
    ax.set_ylim(0.90, 0.99)
    ax.set_xscale("log")
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.grid(alpha=0.25)
    ax.set_xticks(sc["clips"])
    ax.set_xticklabels(["0.5k", "1k", "2k", "4k", "6k", "13.6k"])
    ax.tick_params(axis="x", labelsize=7)
    w = M["wilcoxon_vs_gru"]
    ax = axes[1]
    x = np.arange(2)
    width = 0.28
    ours = [w["h1"]["ours"], w["h4"]["ours"]]
    gru = [w["h1"]["gru"], w["h4"]["gru"]]
    pers = [w["h1"]["persist"], w["h4"]["persist"]]
    ax.bar(x - width, ours, width, label="Endo-HJEPA", color=C_OURS)
    ax.bar(x, gru, width, label="GRU", color=C_GRU)
    ax.bar(x + width, pers, width, label="persistence", color=C_PERS)
    ax.set_xticks(x)
    ax.set_xticklabels(["step 1", "mean steps 1--4"])
    ax.set_ylabel("Forecast cosine")
    ax.set_title("(b) Horizon (750-clip val)")
    ax.set_ylim(0.90, 1.0)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(axis="y", alpha=0.25)

    f = M["forecast_6000"]
    ax = axes[2]
    names = ["persist.", "Mamba-\ninspired", "GRU", "causal L1\n(ours)"]
    vals = [
        f["persistence"]["cos"],
        f["mamba"]["cos"],
        f["gru"]["cos"],
        f["causal_l1"]["cos"],
    ]
    cols = [C_PERS, C_MAMBA, C_GRU, C_OURS]
    ax.bar(names, vals, color=cols)
    ax.set_ylabel("Mean forecast cosine (steps 1--4)")
    ax.set_title("(c) Dynamics baselines (6k clips)")
    ax.set_ylim(0.90, 1.0)
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=7)

    fig.tight_layout()
    _save(fig, "figure2_forecast")


def fig_planning_domain():
    """Figure 3: valid action-grounding audit and partial few-shot recovery.

    The legacy 2k ``planning_2000`` record is explicitly invalid in the metric
    ledger because its L2/L3 encoder was not executed.  It must never be
    visualised as a forecast-by-domain result.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8))

    ax = axes[0]
    grounding = M["action_grounding"]
    physical = grounding["physical"]
    keyframes = physical["per_keyframe"]
    x = np.arange(len(keyframes))
    width = 0.38
    latent = [row["nmi_latent"] for row in keyframes]
    random = [row["nmi_random"] for row in keyframes]
    ax.bar(x - width / 2, latent, width, label="latent pose NMI", color=C_OURS)
    ax.bar(x + width / 2, random, width, label="matched random", color=C_PERS)
    ax.axhline(
        grounding["semantic"]["nmi"],
        color=C_GRU,
        ls="--",
        lw=1.1,
        label=f"semantic NMI {grounding['semantic']['nmi']:.3f}",
    )
    ax.axhline(
        grounding["semantic"]["nmi_random"],
        color=C_PERS,
        ls=":",
        lw=1.1,
        label=f"semantic random {grounding['semantic']['nmi_random']:.3f}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Keyframe {index + 1}" for index in range(len(keyframes))], fontsize=6.5
    )
    ax.set_ylabel("Normalised mutual information")
    ax.set_title("(a) Six-keyframe latent-action audit")
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=6.2, loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    fs = M["fewshot"]
    names = ["GI\nzero-shot", "GI\n32-shot", "Bronch\nzero-shot", "Bronch\n32-shot"]
    vals = [
        fs["gi"]["zero"],
        fs["gi"]["few"],
        fs["bronch"]["zero"],
        fs["bronch"]["few"],
    ]
    pers = [
        fs["gi"]["persist"],
        fs["gi"]["persist"],
        fs["bronch"]["persist"],
        fs["bronch"]["persist"],
    ]
    x = np.arange(4)
    ax.bar(x, vals, color=[C_QUERY, C_GI, C_QUERY, C_BR], label="model")
    ax.plot(x, pers, "s--", color=C_PERS, ms=5, label="persistence")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("Forecast cosine")
    ax.set_title("(b) 32-shot token adaptation")
    ax.set_ylim(0.70, 0.95)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    _save(fig, "figure3_planning")


def fig_downstream():
    """Figure 4: official CV, challenge baselines, and per-class phase."""
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.0))

    cv = M["downstream_crossval"]
    ax = axes[0]
    x_cv = np.arange(2)
    w_cv = 0.36
    frozen = cv["vjepa2_frozen"]
    adapted = cv["vjepa2_adapted"]
    ax.bar(
        x_cv - w_cv / 2,
        [frozen["phase"], frozen["instr"]],
        w_cv,
        yerr=[frozen["phase_std_across_folds"], frozen["instr_std_across_folds"]],
        capsize=3,
        label="Frozen",
        color=C_OURS,
    )
    ax.bar(
        x_cv + w_cv / 2,
        [adapted["phase"], adapted["instr"]],
        w_cv,
        yerr=[adapted["phase_std_across_folds"], adapted["instr_std_across_folds"]],
        capsize=3,
        label="Adapted",
        color=C_GRU,
    )
    ax.set_xticks(x_cv)
    ax.set_xticklabels(["Phase acc.", "Instrument mAP"])
    ax.set_ylabel("5-fold mean $\\pm$ std")
    ax.set_title("(a) Official 5-fold CV")
    ax.set_ylim(0.35, 0.78)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25)

    d = M["downstream_official"]
    keys = [
        "imagenet",
        "videomae",
        "dinov2",
        "timesformer",
        "vivit",
        "vjepa2_frozen",
        "vjepa2_adapted",
    ]
    labels = [
        "ImageNet",
        "VideoMAE",
        "DINOv2",
        "TimeSformer",
        "ViViT",
        "V-JEPA2\nfrozen",
        "V-JEPA2\nadapted",
    ]
    phase = [d[k]["phase"] for k in keys]
    instr = [d[k]["instr"] for k in keys]
    pstd = [d[k]["phase_std"] for k in keys]
    istd = [d[k]["instr_std"] for k in keys]
    x = np.arange(len(keys))
    w = 0.38
    ax = axes[1]
    ax.bar(x - w / 2, phase, w, yerr=pstd, capsize=2, label="phase acc", color=C_ACC)
    ax.bar(
        x + w / 2, instr, w, yerr=istd, capsize=2, label="instrument mAP", color=C_MAP
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Score (mean $\\pm$ std, 3 seeds)")
    ax.set_title("(b) Challenge-test baselines")
    ax.set_ylim(0.30, 0.80)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[2]
    pc = M["per_class_frozen"]
    names = [
        "Prep.",
        "Calot",
        "Clip/cut",
        "GB dissect",
        "GB pack",
        "Clean/coag",
        "GB retract",
    ]
    vals = [
        pc["preparation"],
        pc["calot_triangle_dissection"],
        pc["clipping_cutting"],
        pc["gallbladder_dissection"],
        pc["gallbladder_packaging"],
        pc["cleaning_coagulation"],
        pc["gallbladder_retraction"],
    ]
    ax.barh(names, vals, color=C_OURS)
    ax.set_xlabel("Per-class accuracy (V-JEPA 2 frozen)")
    ax.set_xlim(0, 1.0)
    ax.set_title("(c) Challenge phase by class")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    for i, v in enumerate(vals):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8)

    fig.tight_layout()
    _save(fig, "figure4_downstream")


def fig_aux():
    """Figure 5: pixel contrast, corrected risk, and grouped STIR."""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))

    ax = axes[0]
    pix = M["pixel"]
    names = [
        "Copy last\n(CNN)",
        "CNN\nnext-frame",
        "Copy last\n(DDPM)",
        "DDPM\nnext-frame",
    ]
    vals = [
        pix["cnn"]["copy"],
        pix["cnn"]["psnr"],
        pix["diffusion"]["copy"],
        pix["diffusion"]["psnr"],
    ]
    cols = [C_PERS, C_GRU, C_PERS, C_MAMBA]
    ax.bar(names, vals, color=cols)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("(a) Pixel generation vs copy-last")
    ax.set_ylim(0, 36)
    ax.tick_params(axis="x", labelsize=7)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.6, f"{v:.1f}", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    risk = M["grounded_upgrade"]["risk_real_v2"]["test"]
    names = ["Corrected risk", "Chance", "Prespecified\ngate"]
    vals = [risk["auc"], 0.50, 0.75]
    cols = [C_GRU, C_PERS, C_QUERY]
    ax.bar(names, vals, color=cols)
    ax.axhline(0.75, color=C_QUERY, ls="--", lw=1.0)
    ax.set_ylim(0, 1.0)
    ax.set_title("(b) Corrected near-wall risk\nAUC 0.523 FAIL")
    ax.set_ylabel("AUC")
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=7)

    ax = axes[2]
    s = M["stir"]
    ax.bar(
        ["Before STIR", "After STIR"], [s["before"], s["after"]], color=[C_PERS, C_OURS]
    )
    ax.set_ylabel("Latent feature-set distance")
    ax.set_title("(c) STIR endpoint regulariser")
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate([s["before"], s["after"]]):
        ax.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=8)

    fig.tight_layout()
    _save(fig, "figure5_aux")


def fig_per_dataset(path: Path | None = None):
    """Figure 7: per-dataset forecast from the declared 6k evaluation."""
    cands = [
        path,
        ROOT / "outputs/scale_6000_causal/per_dataset.json",
        ROOT / "results/scale_6000_causal__per_dataset.json",
        ROOT / "results/forecast_per_dataset.json",
    ]
    src = next((p for p in cands if p is not None and p.is_file()), None)
    if src is None:
        raise FileNotFoundError(
            "Figure 7 requires the declared 6k per-dataset report; refusing "
            "to substitute a legacy result."
        )
    rep = json.loads(src.read_text(encoding="utf-8"))
    rows = sorted(rep["by_dataset"].values(), key=lambda r: -r["n"])
    if not rows:
        return
    names = [display(r["dataset"]) for r in rows]
    ours = [r["cos_model"] for r in rows]
    pers = [r["cos_persist"] for r in rows]
    n_vals = [int(r["n"]) for r in rows]
    fig, ax = plt.subplots(figsize=(max(8.5, 0.55 * len(names) + 2), 4.2))
    x = np.arange(len(names))
    w = 0.38
    bars_ours = ax.bar(x - w / 2, ours, w, label="Endo-HJEPA", color=C_OURS)
    bars_pers = ax.bar(x + w / 2, pers, w, label="persistence", color=C_PERS)
    for bar, n in zip(bars_ours, n_vals):
        if n == 1:
            bar.set_hatch("///")
    for bar, n in zip(bars_pers, n_vals):
        if n == 1:
            bar.set_hatch("///")
    if any(n == 1 for n in n_vals):
        ax.bar(
            [],
            [],
            color="white",
            edgecolor=C_OURS,
            hatch="///",
            label="$n{=}1$ descriptive only",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean forecast cosine (steps 1--4)")
    ax.set_title("Per-dataset latent forecast (video-level val)")
    ax.set_ylim(0.70, 1.0)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, "figure7_per_dataset")


def main():
    fig_scale_horizon_forecast()
    fig_planning_domain()
    fig_downstream()
    fig_aux()
    fig_per_dataset()
    print(f"[figures] wrote {OUT}")


if __name__ == "__main__":
    main()
