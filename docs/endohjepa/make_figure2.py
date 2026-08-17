"""Figure 2: data-scale curve + horizon robustness + external baselines (Endo-HJEPA).

    python docs/endohjepa/make_figure2.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# measured on video-level val
SCALE_CLIPS = [500, 1000, 2000, 4000, 6000]
SCALE_COS = [0.962, 0.968, 0.972, 0.976, 0.978]
PERSIST_COS = 0.916
GRU_COS = 0.949

HORIZONS = [1, 4, 8]
HJEPA_H = [0.9601, 0.9541, 0.9522]
GRU_H = [0.9587, 0.9514, 0.9494]

ENCODERS = ["ImageNet\n(supervised)", "VideoMAE\n(Kinetics)", "V-JEPA2\nfrozen", "V-JEPA2\nadapted (ours)"]
PHASE = [0.625, 0.550, 0.688, 0.688]
INSTR = [0.422, 0.495, 0.414, 0.498]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) scale curve
    ax = axes[0]
    ax.plot(SCALE_CLIPS, SCALE_COS, "o-", color="#2563eb", lw=2, label="Endo-HJEPA (causal L1)")
    ax.axhline(GRU_COS, color="#dc2626", ls="--", label="GRU baseline")
    ax.axhline(PERSIST_COS, color="#64748b", ls=":", label="persistence")
    ax.set_xlabel("# training clips (video-level)")
    ax.set_ylabel("forecast cosine @ h=4")
    ax.set_title("(a) Data-scale curve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.88, 1.0)

    # (b) horizon robustness
    ax = axes[1]
    x = np.arange(len(HORIZONS))
    w = 0.35
    ax.bar(x - w / 2, HJEPA_H, w, label="Endo-HJEPA", color="#2563eb")
    ax.bar(x + w / 2, GRU_H, w, label="GRU", color="#dc2626")
    ax.set_xticks(x); ax.set_xticklabels([f"h={h}" for h in HORIZONS])
    ax.set_ylabel("forecast cosine")
    ax.set_title("(b) Forecast vs horizon (Wilcoxon p<1e-6 at h=4,8)")
    ax.set_ylim(0.94, 0.965)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # (c) external baselines
    ax = axes[2]
    x = np.arange(len(ENCODERS))
    w = 0.38
    ax.bar(x - w / 2, PHASE, w, label="phase acc", color="#059669")
    ax.bar(x + w / 2, INSTR, w, label="instrument mAP", color="#d97706")
    ax.set_xticks(x); ax.set_xticklabels(ENCODERS, fontsize=8)
    ax.set_ylabel("score")
    ax.set_title("(c) External baselines (CholecT50 official split)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Endo-HJEPA: scale, horizon robustness, and external-baseline comparison", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("docs/endohjepa/figure2_results.png", dpi=200, bbox_inches="tight")
    fig.savefig("docs/endohjepa/figure2_results.pdf", bbox_inches="tight")
    print("[figure2] wrote docs/endohjepa/figure2_results.png/.pdf")


if __name__ == "__main__":
    main()
