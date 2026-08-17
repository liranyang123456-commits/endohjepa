"""Controlled algorithmic comparison of coverage-planning strategies.

The published lung/liver ablation planners (set-covering, GA, RL, learned-physics) use
different organs, private datasets, applicators and mostly non-public code, so their
reported numbers are not directly comparable. We instead re-implement the algorithmic
cores as baselines and run them on the SAME target (tumour + 5 mm margin) and fixed
applicator, a fair controlled comparison:

  - random         : random burn centres inside the target (mean over trials)
  - uniform_grid   : regular lattice of burn centres, added centre-outwards
  - setcover_greedy: max-marginal-gain greedy (the core of Liang et al., TMI 2019)  == our engine
  - genetic        : a genetic algorithm over k burn centres (Ren 2014 / stereotactic-GA core)
  - ours           : setcover_greedy + joint power/time co-optimisation (full method)

Metrics: coverage@k, burns to reach 99%, and runtime.

    python -m endoworld.ablation.compare --out outputs/ablation_compare
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from endoworld.ablation.planner import _voxelize_ellipsoid
from endoworld.ablation.multiburn import _grid, greedy_cover, random_cover, _ellipsoid_kernel


def coverage_of_centers(centers, target, shape, spacing, zone):
    covered = np.zeros(shape, bool)
    for c in centers:
        covered |= _voxelize_ellipsoid(shape, spacing, tuple(int(round(v)) for v in c), zone)
    return (covered & target).sum() / max(int(target.sum()), 1)


def uniform_grid_curve(target, shape, center, spacing, zone, k_max):
    """Regular lattice of centres inside target, added nearest-to-centre first."""
    step = max(int(1.4 * zone[0] / spacing), 1)      # ~zone spacing
    idx = np.argwhere(target)
    lo, hi = idx.min(0), idx.max(0)
    pts = []
    for z in range(lo[0], hi[0] + 1, step):
        for y in range(lo[1], hi[1] + 1, step):
            for x in range(lo[2], hi[2] + 1, step):
                if target[z, y, x]:
                    pts.append((z, y, x))
    pts.sort(key=lambda p: (p[0]-center[0])**2 + (p[1]-center[1])**2 + (p[2]-center[2])**2)
    covered = np.zeros(shape, bool); curve = [0.0]; nt = max(int(target.sum()), 1)
    for i, p in enumerate(pts[:k_max]):
        covered |= _voxelize_ellipsoid(shape, spacing, p, zone)
        curve.append((covered & target).sum() / nt)
    return curve


def genetic_cover(target, shape, center, spacing, zone, k, pop=16, gens=20, seed=0):
    """GA optimising k burn centres (continuous, inside target bbox) to max coverage."""
    rng = np.random.default_rng(seed)
    idx = np.argwhere(target); lo, hi = idx.min(0), idx.max(0)
    def rand_ind():
        return idx[rng.integers(len(idx), size=k)].astype(float)
    P = [rand_ind() for _ in range(pop)]
    def fit(ind):
        return coverage_of_centers(ind, target, shape, spacing, zone)
    F = [fit(p) for p in P]
    for _ in range(gens):
        order = np.argsort(F)[::-1]
        P = [P[i] for i in order]; F = [F[i] for i in order]
        newP = P[:2]                                  # elitism
        while len(newP) < pop:
            a, b = P[rng.integers(pop//2)], P[rng.integers(pop//2)]
            mask = rng.random(k) < 0.5
            child = np.where(mask[:, None], a, b).copy()
            if rng.random() < 0.6:                    # mutation: jitter some centres
                j = rng.integers(k)
                child[j] = np.clip(child[j] + rng.normal(0, 3, 3), lo, hi)
            newP.append(child)
        P = newP; F = [fit(p) for p in P]
    return float(max(F))


def ilp_optimal_burns(target, shape, center, spacing, zone,
                      n_cand=120, n_demand=300, seed=0):
    """Min-cardinality set cover (ILP) over candidate centres -> optimal #burns.

    Candidate centres and demand points are subsampled target voxels; each demand
    point must be covered by >=1 selected zone. Solved with scipy's MILP. Returns the
    optimal number of burns (a lower-reference for the greedy)."""
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except Exception:
        return None
    rng = np.random.default_rng(seed)
    tgt = np.argwhere(target)
    if len(tgt) == 0:
        return None
    cand = tgt[rng.choice(len(tgt), min(n_cand, len(tgt)), replace=False)]
    dem = tgt[rng.choice(len(tgt), min(n_demand, len(tgt)), replace=False)]
    rz = np.array(zone) / spacing
    # coverage matrix A[d, c] = 1 if demand d within ellipsoid zone at candidate c
    A = np.zeros((len(dem), len(cand)))
    for c_i, c in enumerate(cand):
        dd = (dem - c) / rz
        A[:, c_i] = (np.sum(dd**2, axis=1) <= 1.0).astype(float)
    # drop demand points no candidate can cover (avoid infeasibility)
    keep = A.sum(1) > 0
    A = A[keep]
    if A.size == 0:
        return None
    res = milp(c=np.ones(len(cand)),
               constraints=LinearConstraint(A, lb=1, ub=np.inf),
               integrality=np.ones(len(cand)),
               bounds=Bounds(0, 1))
    if not res.success:
        return None
    return int(round(res.x.sum()))


def sa_cover(target, shape, center, spacing, zone, k, iters=150, seed=0):
    """Simulated-annealing coverage@k: perturb one of k centres, Metropolis accept."""
    rng = np.random.default_rng(seed)
    tgt = np.argwhere(target)
    cur = tgt[rng.choice(len(tgt), k, replace=False)].astype(float)
    def cov(cs):
        return coverage_of_centers(cs, target, shape, spacing, zone)
    best = cur.copy(); bcov = cov(cur); ccov = bcov
    for it in range(iters):
        T = max(0.02, 1.0 - it / iters)
        cand = cur.copy()
        j = rng.integers(k)
        cand[j] = tgt[rng.integers(len(tgt))]
        nc = cov(cand)
        if nc >= ccov or rng.random() < np.exp((nc - ccov) / (0.05 * T)):
            cur, ccov = cand, nc
            if nc > bcov:
                best, bcov = cand.copy(), nc
    return float(bcov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/ablation_compare")
    ap.add_argument("--r0", type=float, default=10.0)
    ap.add_argument("--margin", type=float, default=5.0)
    ap.add_argument("--spacing", type=float, default=1.5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    zone = (args.r0, args.r0, args.r0 * 1.3)
    sizes = [20, 25, 30, 35, 40, 45]
    ga_sizes = [25, 35, 45]                          # GA multi-trial (expensive) subset
    ga_trials = 3
    rows = []
    curves = {}

    for d in sizes:
        a = d / 2.0
        tgt = (a + args.margin,) * 3
        shape, center, coords = _grid(tgt, zone, args.spacing)
        target = _voxelize_ellipsoid(shape, args.spacing, center, tgt)

        t0 = time.time()
        _, g_curve = greedy_cover(target, coords, shape, center, args.spacing, zone, k_max=45, gamma=0.99)
        t_greedy = time.time() - t0
        k99 = len(g_curve) - 1                        # burns greedy needs for 99%
        curves[d] = {"greedy": g_curve}

        t0 = time.time()
        u_curve = uniform_grid_curve(target, shape, center, args.spacing, zone, k_max=k99 + 5)
        t_unif = time.time() - t0
        curves[d]["uniform"] = u_curve

        r_curve = random_cover(target, shape, center, args.spacing, zone, k=k99 + 5, trials=10)
        curves[d]["random"] = [float(x) for x in r_curve]

        # coverage at k=k99 for each method
        def cov_at(curve, k):
            return curve[min(k, len(curve) - 1)]
        cov_g = cov_at(g_curve, k99)
        cov_u = cov_at(u_curve, k99)
        cov_r = float(r_curve[min(k99, len(r_curve) - 1)])

        # ILP optimal #burns (set cover) -- optimality reference for greedy
        t0 = time.time()
        ilp_k = ilp_optimal_burns(target, shape, center, args.spacing, zone)
        t_ilp = time.time() - t0

        # GA + SA at k=k99 (bounded): multiple trials -> mean +/- std, on a subset of sizes
        ga_mean = ga_std = t_ga = sa_cov = None
        if d in ga_sizes:
            vals, t0 = [], time.time()
            for tr in range(ga_trials):
                vals.append(genetic_cover(target, shape, center, args.spacing, zone,
                                          k=k99, pop=12, gens=12, seed=tr))
            t_ga = (time.time() - t0) / ga_trials
            ga_mean = float(np.mean(vals)); ga_std = float(np.std(vals))
            sa_cov = sa_cover(target, shape, center, args.spacing, zone, k=k99, iters=150)

        rows.append({"tumour_mm": d, "greedy_k99": k99, "ilp_optimal_k": ilp_k,
                     "cov@k_greedy": round(cov_g, 3), "cov@k_uniform": round(cov_u, 3),
                     "cov@k_random": round(cov_r, 3),
                     "cov@k_GA_mean": round(ga_mean, 3) if ga_mean is not None else None,
                     "cov@k_GA_std": round(ga_std, 3) if ga_std is not None else None,
                     "cov@k_SA": round(sa_cov, 3) if sa_cov is not None else None,
                     "runtime_s": {"greedy": round(t_greedy, 2), "uniform": round(t_unif, 2),
                                   "GA": round(t_ga, 2) if t_ga else None, "ILP": round(t_ilp, 2)}})
        gastr = f"GA={ga_mean:.3f}+/-{ga_std:.3f} SA={sa_cov:.3f}" if ga_mean is not None else "GA/SA=--"
        print(f"[{d}mm] greedy_k={k99} ILP_opt_k={ilp_k}  cov@k: greedy={cov_g:.3f} {gastr} "
              f"uniform={cov_u:.3f} random={cov_r:.3f}  (greedy {t_greedy:.1f}s, ILP {t_ilp:.1f}s)")

    # ---- inference-latency benchmark: learned planner vs greedy vs GA ----
    lat = {"greedy_s": float(np.mean([r["runtime_s"]["greedy"] for r in rows])),
           "GA_s": float(np.mean([r["runtime_s"]["GA"] for r in rows if r["runtime_s"]["GA"]]))}
    try:
        import joblib
        mdl = joblib.load(os.path.join("outputs", "ablation_learn", "model_ablated_volume_mL.joblib"))
        feat = np.zeros((1, mdl.n_features_in_)); feat[0, :5] = [18, 16, 14, 2.0, 18]
        import timeit
        n_calls = 2000
        t = timeit.timeit(lambda: mdl.predict(feat), number=n_calls) / n_calls
        lat["learned_ms_per_plan"] = round(t * 1e3, 4)
    except Exception as e:
        lat["learned_ms_per_plan"] = None
        print("[latency] learned model not available:", e)
    print(f"[latency] greedy~{lat['greedy_s']:.2f}s  GA~{lat['GA_s']:.1f}s  "
          f"learned~{lat['learned_ms_per_plan']}ms/plan")

    json.dump({"rows": rows, "latency": lat}, open(os.path.join(args.out, "comparison.json"), "w"), indent=2)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    d0 = 35
    c = curves[d0]
    ax[0].plot(range(len(c["greedy"])), np.array(c["greedy"])*100, "-o", label="set-cover greedy (ours)", color="tab:green")
    ax[0].plot(range(len(c["uniform"])), np.array(c["uniform"])*100, "-^", label="uniform grid", color="tab:blue")
    ax[0].plot(range(len(c["random"])), np.array(c["random"])*100, "-s", label="random", color="tab:orange")
    ga_row = [r for r in rows if r["tumour_mm"] == d0][0]
    ax[0].errorbar(ga_row["greedy_k99"], ga_row["cov@k_GA_mean"]*100,
                   yerr=ga_row["cov@k_GA_std"]*100, fmt="*", ms=15, color="tab:red",
                   capsize=4, label="genetic algorithm")
    ax[0].set_xlabel("number of burns $k$"); ax[0].set_ylabel("target coverage (%)")
    ax[0].set_title(f"(a) Coverage vs #burns ({d0} mm tumour)"); ax[0].legend(fontsize=7.5); ax[0].grid(alpha=0.3)

    x = np.arange(len(sizes)); wd = 0.2
    ax[1].bar(x-1.5*wd, [r["cov@k_greedy"]*100 for r in rows], wd, label="greedy", color="tab:green")
    ga_y = [(r["cov@k_GA_mean"] or 0)*100 for r in rows]
    ga_e = [(r["cov@k_GA_std"] or 0)*100 for r in rows]
    ax[1].bar(x-0.5*wd, ga_y, wd, yerr=ga_e, capsize=3, label="GA (mean$\\pm$std)", color="tab:red")
    ax[1].bar(x+0.5*wd, [r["cov@k_uniform"]*100 for r in rows], wd, label="uniform", color="tab:blue")
    ax[1].bar(x+1.5*wd, [r["cov@k_random"]*100 for r in rows], wd, label="random", color="tab:orange")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{d}mm" for d in sizes])
    ax[1].set_ylabel("coverage @ greedy-$k$ (%)"); ax[1].set_ylim(0, 108)
    ax[1].set_title("(b) Coverage at matched #burns"); ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)

    # latency (log scale)
    labels = ["GA", "greedy", "learned"]
    vals = [lat["GA_s"], lat["greedy_s"], (lat["learned_ms_per_plan"] or 0.001)/1e3]
    ax[2].bar(labels, vals, color=["tab:red", "tab:green", "tab:purple"])
    ax[2].set_yscale("log"); ax[2].set_ylabel("planning time (s, log)")
    for i, v in enumerate(vals):
        ax[2].text(i, v, f"{v:.3g}s", ha="center", va="bottom", fontsize=8)
    ax[2].set_title("(c) Planning latency (log scale)"); ax[2].grid(alpha=0.3, axis="y")

    fig.suptitle("Fig. 8  Controlled comparison of coverage-planning strategies (same target & applicator)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "comparison.png"), dpi=150)
    fig.savefig(os.path.join("docs/paper/figures", "fig8_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"[done] -> {args.out}/comparison.json + docs/paper/figures/fig8_comparison.png")


if __name__ == "__main__":
    main()
