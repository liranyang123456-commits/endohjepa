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

HERE = Path(__file__).resolve().parent
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
    ax.plot(sc["clips"][:5], sc["cos"][:5], "o-", color=C_OURS, lw=2.0, ms=6,
            label="Endo-HJEPA (shared 750-val)")
    ax.plot(sc["clips"][4:], sc["cos"][4:], "--", color=C_OURS, lw=1.2)
    ax.plot(sc["clips"][-1], sc["cos"][-1], marker="o", ms=7,
            markerfacecolor="white", markeredgecolor=C_OURS, markeredgewidth=1.5,
            linestyle="none", label="13.6k (different 1,631-val)")
    ax.axhline(M["forecast_6000"]["gru"]["cos"], color=C_GRU, ls="--", lw=1.4, label="GRU (6k clips)")
    ax.axhline(M["forecast_6000"]["persistence"]["cos"], color=C_PERS, ls=":", lw=1.6, label="persistence")
    ax.set_xlabel("Training clips (video-level)")
    ax.set_ylabel("Forecast cosine ($h{=}4$)")
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
    ax.set_xticklabels(["$h{=}1$", "$h{=}4$"])
    ax.set_ylabel("Forecast cosine")
    ax.set_title("(b) Horizon (750-clip val)")
    ax.set_ylim(0.90, 1.0)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(axis="y", alpha=0.25)

    f = M["forecast_6000"]
    ax = axes[2]
    names = ["persist.", "query L1", "Mamba", "GRU", "causal L1\n(ours)"]
    vals = [f["persistence"]["cos"], f["query_l1"]["cos"], f["mamba"]["cos"],
            f["gru"]["cos"], f["causal_l1"]["cos"]]
    cols = [C_PERS, C_QUERY, C_MAMBA, C_GRU, C_OURS]
    ax.bar(names, vals, color=cols)
    ax.set_ylabel("Forecast cosine ($h{=}4$)")
    ax.set_title("(c) Dynamics baselines (6k clips)")
    ax.set_ylim(0.90, 1.0)
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=7)

    fig.tight_layout()
    _save(fig, "figure2_forecast")


def fig_planning_domain():
    """Figure 3: per-domain forecast + audited grounding + few-shot."""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))
    p = M["planning_2000"]["by_domain"]
    doms = ["laparo", "gi", "bronch"]
    labels = ["Laparoscopy", "GI", "Bronchoscopy"]
    cols = [C_LAP, C_GI, C_BR]

    ax = axes[0]
    x = np.arange(3)
    w = 0.36
    ax.bar(x - w / 2, [p[d]["forecast"] for d in doms], w, label="Endo-HJEPA", color=C_OURS)
    ax.bar(x + w / 2, [p[d]["persist"] for d in doms], w, label="persistence", color=C_PERS)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Forecast cosine ($h{=}4$)")
    ax.set_title("(a) Forecast by orifice")
    ax.set_ylim(0.84, 1.0)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    grounding = M["action_grounding"]
    physical = grounding["physical"]
    values = [
        grounding["semantic"]["nmi"],
        grounding["semantic"]["nmi_random"],
        np.mean(physical["nmi_range"]),
        np.mean(physical["random_range"]),
    ]
    ax.bar(
        ["Semantic\nlatent", "Semantic\nrandom", "Physical\nlatent", "Physical\nrandom"],
        values,
        color=[C_OURS, C_PERS, C_OURS, C_PERS],
    )
    ax.set_ylabel("Normalised mutual information")
    ax.set_title("(b) Action grounding audit")
    ax.set_ylim(0, 0.6)
    for i, v in enumerate(values):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=7)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[2]
    fs = M["fewshot"]
    names = ["GI\nzero-shot", "GI\n32-shot", "Bronch\nzero-shot", "Bronch\n32-shot"]
    vals = [fs["gi"]["zero"], fs["gi"]["few"], fs["bronch"]["zero"], fs["bronch"]["few"]]
    pers = [fs["gi"]["persist"], fs["gi"]["persist"], fs["bronch"]["persist"], fs["bronch"]["persist"]]
    x = np.arange(4)
    ax.bar(x, vals, color=[C_QUERY, C_GI, C_QUERY, C_BR], label="model")
    ax.plot(x, pers, "s--", color=C_PERS, ms=5, label="persistence")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("Forecast cosine")
    ax.set_title("(c) Few-shot domain token")
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
    ax.bar(x_cv - w_cv / 2, [frozen["phase"], frozen["instr"]], w_cv,
           yerr=[frozen["phase_std_across_folds"], frozen["instr_std_across_folds"]],
           capsize=3, label="Frozen", color=C_OURS)
    ax.bar(x_cv + w_cv / 2, [adapted["phase"], adapted["instr"]], w_cv,
           yerr=[adapted["phase_std_across_folds"], adapted["instr_std_across_folds"]],
           capsize=3, label="Adapted", color=C_GRU)
    ax.set_xticks(x_cv)
    ax.set_xticklabels(["Phase acc.", "Instrument mAP"])
    ax.set_ylabel("5-fold mean $\\pm$ std")
    ax.set_title("(a) Official 5-fold CV")
    ax.set_ylim(0.35, 0.78)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25)

    d = M["downstream_official"]
    keys = ["imagenet", "videomae", "dinov2", "timesformer", "vivit", "vjepa2_frozen", "vjepa2_adapted"]
    labels = ["ImageNet", "VideoMAE", "DINOv2", "TimeSformer", "ViViT", "V-JEPA2\nfrozen", "V-JEPA2\nadapted"]
    phase = [d[k]["phase"] for k in keys]
    instr = [d[k]["instr"] for k in keys]
    pstd = [d[k]["phase_std"] for k in keys]
    istd = [d[k]["instr_std"] for k in keys]
    x = np.arange(len(keys))
    w = 0.38
    ax = axes[1]
    ax.bar(x - w / 2, phase, w, yerr=pstd, capsize=2, label="phase acc", color=C_ACC)
    ax.bar(x + w / 2, instr, w, yerr=istd, capsize=2, label="instrument mAP", color=C_MAP)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Score (mean $\\pm$ std, 3 seeds)")
    ax.set_title("(b) Challenge-test baselines")
    ax.set_ylim(0.30, 0.80)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[2]
    pc = M["per_class_frozen"]
    names = ["Prep.", "Calot", "Clip/cut", "GB dissect", "GB pack", "Clean/coag", "GB retract"]
    vals = [pc["preparation"], pc["calot_triangle_dissection"], pc["clipping_cutting"],
            pc["gallbladder_dissection"], pc["gallbladder_packaging"],
            pc["cleaning_coagulation"], pc["gallbladder_retraction"]]
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
    """Figure 5: pixel contrast, energy grounding, STIR."""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))

    ax = axes[0]
    pix = M["pixel"]
    names = ["Copy last\n(CNN proto.)", "CNN\nnext-frame", "Copy last\n(diff. proto.)", "DDPM\nnext-frame"]
    vals = [pix["cnn"]["copy"], pix["cnn"]["psnr"], pix["diffusion"]["copy"], pix["diffusion"]["psnr"]]
    cols = [C_PERS, C_GRU, C_PERS, C_MAMBA]
    ax.bar(names, vals, color=cols)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("(a) Pixel generation vs copy-last")
    ax.set_ylim(0, 36)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.6, f"{v:.1f}", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    c = M["collision"]
    ax.bar(["Near-wall AUC", "Chance"], [c["auc"], 0.50], color=[C_OURS, C_PERS])
    ax.set_ylim(0, 1.0)
    ax.set_title(f"(b) Energy vs wall proximity\nSpearman $={c['spearman']:.2f}$")
    ax.set_ylabel("AUC")
    ax.grid(axis="y", alpha=0.25)
    ax.text(0, c["auc"] + 0.03, f"{c['auc']:.3f}", ha="center", fontsize=8)

    ax = axes[2]
    s = M["stir"]
    ax.bar(["Before STIR", "After STIR"], [s["before"], s["after"]], color=[C_PERS, C_OURS])
    ax.set_ylabel("Token chamfer")
    ax.set_title("(c) STIR deformation regulariser")
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate([s["before"], s["after"]]):
        ax.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=8)

    fig.tight_layout()
    _save(fig, "figure5_aux")


def fig_latent():
    """Figure 6: PCA of cached val latents (if cache exists)."""
    cache = Path("outputs/scale_6000_causal/latents_cache.pt")
    if not cache.is_file():
        cache = Path("outputs/p2000_full_causal/latents_cache.pt")
    if not cache.is_file():
        print("[fig] skip figure6_latent (no cache)")
        return
    import torch
    from endoworld.data.domains import ID_TO_DOMAIN
    pack = torch.load(cache, map_location="cpu", weights_only=False)
    Z = pack.get("Z_val") if pack.get("Z_val") is not None else pack["Z"]
    D = pack.get("D_val") if pack.get("D_val") is not None else pack["D"]
    if Z.dim() == 4:
        Z = Z.mean(dim=2)
    feat = Z.mean(dim=1).numpy()
    feat = feat - feat.mean(0, keepdims=True)
    _, s, vt = np.linalg.svd(feat, full_matrices=False)
    P = feat @ vt[:2].T
    var = s / s.sum()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    ax = axes[0]
    colours = {0: C_LAP, 1: C_GI, 2: C_BR, 3: C_PERS}
    for did in np.unique(D.numpy()):
        m = D.numpy() == did
        ax.scatter(P[m, 0], P[m, 1], s=9, alpha=0.55, color=colours.get(int(did), "#000"),
                   label=ID_TO_DOMAIN.get(int(did), str(did)))
    ax.set_xlabel(f"PC1 ({100 * var[0]:.0f}%)")
    ax.set_ylabel(f"PC2 ({100 * var[1]:.0f}%)")
    ax.set_title("(a) Clip latents by orifice")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    z0 = Z[0].numpy()
    z0 = z0 - z0.mean(0, keepdims=True)
    _, _, vt2 = np.linalg.svd(z0, full_matrices=False)
    tr = z0 @ vt2[:2].T
    ax.plot(tr[:, 0], tr[:, 1], "o-", color=C_OURS, lw=1.6, ms=5, label="actual trajectory")
    ax.plot([tr[-1, 0]], [tr[-1, 1]], "s", color=C_PERS, ms=8, label="persistence (last token)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("(b) One clip: smooth latent path")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "figure6_latent")


def fig_per_dataset(path: Path | None = None):
    """Figure 7: per-dataset forecast if per_dataset.json exists."""
    cands = [
        path,
        Path("outputs/scale_6000_causal/per_dataset.json"),
        Path("outputs/p2000_full_causal/per_dataset.json"),
        HERE / "per_dataset.json",
    ]
    src = next((p for p in cands if p is not None and p.is_file()), None)
    if src is None:
        print("[fig] skip figure7_per_dataset (run endoworld.eval.per_dataset first)")
        return
    rep = json.loads(src.read_text(encoding="utf-8"))
    rows = sorted(rep["by_dataset"].values(), key=lambda r: -r["n"])
    if not rows:
        return
    names = [r["dataset"] for r in rows]
    ours = [r["cos_model"] for r in rows]
    pers = [r["cos_persist"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(8.5, 0.55 * len(names) + 2), 4.2))
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, ours, w, label="Endo-HJEPA", color=C_OURS)
    ax.bar(x + w / 2, pers, w, label="persistence", color=C_PERS)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Forecast cosine ($h{=}4$)")
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
    fig_latent()
    fig_per_dataset()
    print(f"[figures] wrote {OUT}")


if __name__ == "__main__":
    main()
