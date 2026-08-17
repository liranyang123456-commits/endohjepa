"""Figure 3: qualitative latent-space structure (PCA by domain) + forecast trajectory.

    python docs/endohjepa/make_figure3.py
Uses the pooled cached latents (tiny) — CPU only.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from endoworld.data.domains import ID_TO_DOMAIN

CACHE = "outputs/cache_6000_pool/latents_cache.pt"


def pca2(Z):
    Z = Z - Z.mean(0, keepdims=True)
    U, S, V = np.linalg.svd(Z, full_matrices=False)
    return Z @ V[:2].T, S / S.sum()


def main():
    pack = torch.load(CACHE, map_location="cpu", weights_only=False)
    Z = pack.get("Z_val") if pack.get("Z_val") is not None else pack["Z"]
    D = pack.get("D_val") if pack.get("D_val") is not None else pack["D"]
    if Z.dim() == 4:
        Z = Z.mean(dim=2)
    # per-clip feature = mean over time; colour by domain
    feat = Z.mean(dim=1).numpy()
    doms = D.numpy()
    P, var = pca2(feat)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    colours = {0: "#2563eb", 1: "#059669", 2: "#d97706", 3: "#64748b"}
    for did in np.unique(doms):
        m = doms == did
        ax.scatter(P[m, 0], P[m, 1], s=10, alpha=0.6, color=colours.get(int(did), "#000"),
                   label=ID_TO_DOMAIN.get(int(did), str(did)))
    ax.set_xlabel(f"PC1 ({var[0]*100:.0f}%)"); ax.set_ylabel(f"PC2 ({var[1]*100:.0f}%)")
    ax.set_title("(a) Latent space by orifice domain (PCA of pooled clip latents)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) latent forecast trajectory: model vs persistence vs actual, 2 PCs over time
    ax = axes[1]
    n_show = min(40, Z.shape[0])
    # mean |cos| to actual at each future step is in RESULTS; here show a sample clip trajectory
    z0 = Z[0].numpy()  # (T, D) one clip
    T = z0.shape[0]
    zp, _ = pca2(z0)
    ax.plot(zp[:, 0], zp[:, 1], "o-", color="#2563eb", label="actual clip trajectory")
    persist = np.repeat(zp[-1:], 4, axis=0)  # persistence = stay at last
    ax.plot(np.r_[zp[-1, 0], persist[:, 0]], np.r_[zp[-1, 1], persist[:, 1]],
            "s--", color="#64748b", label="persistence (frozen at last)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("(b) A clip's latent trajectory (PCA) — smooth, low-dim")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Endo-HJEPA latent space: domain structure and smooth dynamics", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("docs/endohjepa/figure3_latent.png", dpi=200, bbox_inches="tight")
    fig.savefig("docs/endohjepa/figure3_latent.pdf", bbox_inches="tight")
    print("[figure3] wrote docs/endohjepa/figure3_latent.png/.pdf")


if __name__ == "__main__":
    main()
