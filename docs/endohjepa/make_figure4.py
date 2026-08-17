"""Figure 4: comprehensive results panels (per-domain, per-class, planning, energy).

    python docs/endohjepa/make_figure4.py
All numbers from verified JSON / runs — CPU only.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# per-domain (verified from p2000_full_causal/eval_ckpt.json)
DOMS = ["laparo", "GI", "bronch"]
FORE_MODEL = [0.960, 0.970, 0.984]
FORE_PERSIST = [0.895, 0.935, 0.918]
PLAN_REACH = [95.2, 98.8, 100.0]

# per-class phase (V-JEPA2 frozen, official split)
PHASES = ["prep", "calot", "clip/cut", "gb_dissect", "gb_pack", "clean/coag", "gb_retract"]
PHASE_ACC = [0.80, 0.69, 0.67, 0.74, 0.67, 0.50, 0.80]

# forecast baselines (6000-clip consistent)
MODELS = ["persist", "query", "Mamba", "GRU", "causal\n(ours)"]
FCOS = [0.916, 0.936, 0.971, 0.974, 0.978]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))

    # (a) per-domain forecast
    ax = axes[0, 0]
    x = np.arange(len(DOMS)); w = 0.38
    ax.bar(x - w/2, FORE_MODEL, w, label="Endo-HJEPA", color="#2563eb")
    ax.bar(x + w/2, FORE_PERSIST, w, label="persistence", color="#94a3b8")
    ax.set_xticks(x); ax.set_xticklabels(DOMS)
    ax.set_ylabel("forecast cosine"); ax.set_ylim(0.85, 1.0)
    ax.set_title("(a) Forecast by orifice domain"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # (b) planning reach by domain
    ax = axes[0, 1]
    ax.bar(DOMS, PLAN_REACH, color="#059669")
    ax.set_ylabel("plan reach vs persistence (%)"); ax.set_ylim(0, 105)
    ax.set_title("(b) Latent-MPC planning reach by domain")
    for i, v in enumerate(PLAN_REACH):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # (c) per-class phase accuracy
    ax = axes[1, 0]
    ax.barh(PHASES, PHASE_ACC, color="#d97706")
    ax.set_xlabel("accuracy"); ax.set_xlim(0, 1.0)
    ax.set_title("(c) Per-class phase recognition (V-JEPA 2)")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)

    # (d) forecast model comparison
    ax = axes[1, 1]
    cols = ["#94a3b8", "#64748b", "#d97706", "#dc2626", "#2563eb"]
    ax.bar(MODELS, FCOS, color=cols)
    ax.set_ylabel("forecast cosine @ h=4"); ax.set_ylim(0.88, 1.0)
    ax.set_title("(d) Forecast: ours vs baselines")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(FCOS):
        ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)

    fig.suptitle("Endo-HJEPA results: per-domain, per-class, planning, and baselines", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("docs/endohjepa/figure4_results.png", dpi=200, bbox_inches="tight")
    fig.savefig("docs/endohjepa/figure4_results.pdf", bbox_inches="tight")
    print("[figure4] wrote docs/endohjepa/figure4_results.png/.pdf")


if __name__ == "__main__":
    main()
