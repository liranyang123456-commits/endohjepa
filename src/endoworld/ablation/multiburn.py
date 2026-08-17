"""Multi-burn coverage experiment: empirical validation of the submodular greedy
(1-1/e) behaviour for large tumours with a fixed (capped) applicator.

Two studies:
  (S1) Burns-vs-size: with a fixed ablation ellipsoid (transverse radius r0), sweep the
       tumour diameter and record the number of greedy burns needed to cover the
       tumour + margin to >= gamma, plus the achieved coverage.
  (S2) Coverage-vs-k on a large tumour: compare greedy farthest-point placement to
       random placement (mean over trials). Greedy exhibits concave diminishing
       returns (submodularity) and dominates random; we annotate the (1-1/e) fraction
       of the converged coverage as the theoretical greedy guarantee.

    python -m endoworld.ablation.multiburn --out outputs/ablation_multiburn
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy import ndimage

from endoworld.ablation.planner import _voxelize_ellipsoid


def _grid(target_axes, zone_axes, spacing):
    half = np.array(target_axes) + np.array(zone_axes) + 4
    shape = tuple(int(2 * h / spacing) + 1 for h in half)
    center = tuple(s // 2 for s in shape)
    coords = np.stack(np.meshgrid(
        (np.arange(shape[0]) - center[0]) * spacing,
        (np.arange(shape[1]) - center[1]) * spacing,
        (np.arange(shape[2]) - center[2]) * spacing, indexing="ij"), -1)
    return shape, center, coords


def _ellipsoid_kernel(zone_axes, spacing):
    r = [max(a / spacing, 1) for a in zone_axes]
    rng = [np.arange(-int(np.ceil(x)), int(np.ceil(x)) + 1) for x in r]
    Z, Y, X = np.meshgrid(*rng, indexing="ij")
    return ((Z / r[0])**2 + (Y / r[1])**2 + (X / r[2])**2 <= 1.0).astype(np.float32)


def greedy_cover(target, coords, shape, center, spacing, zone_axes, k_max, gamma=0.99):
    """Max-marginal-gain greedy (the submodular greedy of Theorem 1): each step places
    the burn whose zone covers the most *new* target voxels, via an FFT-convolved gain
    field."""
    from scipy.signal import fftconvolve
    kernel = _ellipsoid_kernel(zone_axes, spacing)
    covered = np.zeros(shape, bool)
    cov_curve = []
    n_t = max(int(target.sum()), 1)
    for _ in range(k_max):
        frac = (covered & target).sum() / n_t
        cov_curve.append(frac)
        if frac >= gamma:
            break
        unc = (target & ~covered).astype(np.float32)
        if unc.sum() == 0:
            break
        gain = fftconvolve(unc, kernel, mode="same")   # gain[c] = new target voxels if centred at c
        gain[~target] = -1                              # only place centres inside target
        idx = np.unravel_index(np.argmax(gain), gain.shape)
        covered |= _voxelize_ellipsoid(shape, spacing, idx, zone_axes)
    cov_curve.append((covered & target).sum() / n_t)
    return covered, cov_curve


def random_cover(target, shape, center, spacing, zone_axes, k, trials=8, seed=0):
    rng = np.random.default_rng(seed)
    n_t = max(int(target.sum()), 1)
    tgt_idx = np.argwhere(target)
    best_curves = []
    for _ in range(trials):
        covered = np.zeros(shape, bool); curve = [0.0]
        for _j in range(k):
            c = tgt_idx[rng.integers(len(tgt_idx))]
            covered |= _voxelize_ellipsoid(shape, spacing, tuple(c), zone_axes)
            curve.append((covered & target).sum() / n_t)
        best_curves.append(curve)
    return np.array(best_curves).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/ablation_multiburn")
    ap.add_argument("--r0", type=float, default=10.0, help="fixed applicator transverse radius (mm)")
    ap.add_argument("--margin", type=float, default=5.0)
    ap.add_argument("--spacing", type=float, default=1.5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    zone = (args.r0, args.r0, args.r0 * 1.3)   # fixed prolate applicator

    # S1: burns vs tumour size
    diams = [15, 20, 25, 30, 35, 40, 45, 50]
    s1 = []
    for d in diams:
        a = d / 2.0
        tgt_axes = (a + args.margin, a + args.margin, a + args.margin)
        shape, center, coords = _grid(tgt_axes, zone, args.spacing)
        target = _voxelize_ellipsoid(shape, args.spacing, center, tgt_axes)
        _, curve = greedy_cover(target, coords, shape, center, args.spacing, zone,
                                k_max=40, gamma=0.99)
        s1.append({"diam_mm": d, "burns": len(curve) - 1, "final_cov": round(curve[-1], 4)})
        print(f"[S1] tumour {d}mm -> burns={len(curve)-1} cov={curve[-1]*100:.1f}%")

    # S2: coverage vs k on a large tumour (45 mm), greedy vs random
    a = 45 / 2.0
    tgt_axes = (a + args.margin, a + args.margin, a + args.margin)
    shape, center, coords = _grid(tgt_axes, zone, args.spacing)
    target = _voxelize_ellipsoid(shape, args.spacing, center, tgt_axes)
    _, g_curve = greedy_cover(target, coords, shape, center, args.spacing, zone,
                              k_max=25, gamma=1.01)   # force full curve
    k_full = len(g_curve) - 1
    r_curve = random_cover(target, shape, center, args.spacing, zone, k=k_full, trials=10)
    conv = g_curve[-1]
    print(f"[S2] 45mm tumour: greedy reaches {conv*100:.1f}% in {k_full} burns; "
          f"random@{k_full}={r_curve[-1]*100:.1f}%")

    res = {"r0_mm": args.r0, "margin": args.margin, "S1": s1,
           "S2": {"greedy": [round(x, 4) for x in g_curve],
                  "random_mean": [round(float(x), 4) for x in r_curve],
                  "converged": round(float(conv), 4),
                  "one_minus_1_over_e_bound": round(float((1 - 1/np.e) * conv), 4)}}
    json.dump(res, open(os.path.join(args.out, "multiburn.json"), "w"), indent=2)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].plot([r["diam_mm"] for r in s1], [r["burns"] for r in s1], "-o", color="tab:blue")
    ax[0].set_xlabel("tumour diameter (mm)"); ax[0].set_ylabel("greedy burns for $\\geq$99% coverage")
    ax[0].set_title(f"(a) Burns vs tumour size (fixed {args.r0:.0f} mm applicator)")
    ax[0].grid(alpha=0.3)

    kk = np.arange(len(g_curve))
    ax[1].plot(kk, np.array(g_curve) * 100, "-o", label="greedy (max-marginal-gain)", color="tab:green")
    ax[1].plot(np.arange(len(r_curve)), r_curve * 100, "-s", label="random placement", color="tab:orange")
    ax[1].axhline((1 - 1/np.e) * conv * 100, ls="--", color="gray",
                  label=r"$(1-1/e)\times$ converged")
    ax[1].set_xlabel("number of burns $k$"); ax[1].set_ylabel("target coverage (%)")
    ax[1].set_title("(b) Coverage vs #burns (45 mm tumour)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle("Fig. 7  Multi-burn coverage: submodular greedy behaviour and $(1-1/e)$ guarantee", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "multiburn.png"), dpi=150)
    fig.savefig(os.path.join("docs/paper/figures", "fig7_multiburn.png"), dpi=150)
    plt.close(fig)
    print(f"[done] -> {args.out}/multiburn.json + docs/paper/figures/fig7_multiburn.png")


if __name__ == "__main__":
    main()
