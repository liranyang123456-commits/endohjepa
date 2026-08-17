"""CLI for ablation trajectory schema + simulation environment.

Examples
--------
Print the JSON schema example::

    python -m endoworld.ablation.run_sim --print-schema

Greedy rollout on a synthetic 16×14×18 mm nodule::

    python -m endoworld.ablation.run_sim --axes 8,7,9 --policy greedy \\
        --out outputs/ablation_sim

Roll out all cases in nodule_params.csv with the greedy policy::

    python -m endoworld.ablation.run_sim --params manifests/nodule_params.csv \\
        --policy greedy --out outputs/ablation_sim --limit 5

Compare greedy env vs classical optimiser on one case::

    python -m endoworld.ablation.run_sim --axes 10,9,11 --compare
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time


def _parse_axes(s: str) -> tuple:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("axes must be a,b,c (semi-axes mm)")
    return tuple(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ablation sim env / trajectory schema CLI")
    ap.add_argument("--print-schema", action="store_true",
                    help="print example trajectory JSON and exit")
    ap.add_argument("--axes", type=_parse_axes, default=None,
                    help="tumour semi-axes mm, e.g. 8,7,9")
    ap.add_argument("--params", default=None,
                    help="nodule_params.csv for cohort rollouts")
    ap.add_argument("--policy", choices=["greedy", "random", "optimiser"],
                    default="greedy")
    ap.add_argument("--device", default="MWA")
    ap.add_argument("--margin", type=float, default=5.0)
    ap.add_argument("--spacing", type=float, default=1.5)
    ap.add_argument("--coverage", type=float, default=0.99)
    ap.add_argument("--max-burns", type=int, default=12)
    ap.add_argument("--use-fdm", action="store_true",
                    help="refine zones with Pennes FDM (slow)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="max cases from --params (0=all)")
    ap.add_argument("--out", default="outputs/ablation_sim")
    ap.add_argument("--save-masks", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="also run classical planner and print side-by-side metrics")
    ap.add_argument("--smoke", action="store_true",
                    help="fast self-check then exit")
    args = ap.parse_args(argv)

    if args.print_schema:
        from endoworld.ablation.trajectory_schema import schema_example_dict
        print(json.dumps(schema_example_dict(), indent=2, ensure_ascii=False))
        return 0

    if args.smoke:
        return _smoke()

    os.makedirs(args.out, exist_ok=True)
    mask_dir = os.path.join(args.out, "masks") if args.save_masks else None

    if args.params:
        return _cohort(args, mask_dir)
    if args.axes is None:
        args.axes = (8.0, 7.0, 9.0)
        print(f"[run_sim] no --axes/--params; using default axes={args.axes}")
    return _single(args, mask_dir)


def _single(args, mask_dir) -> int:
    from endoworld.ablation.sim_env import (
        make_env_from_axes, optimiser_demo_trajectory, rollout,
    )
    from endoworld.ablation.trajectory_schema import save_trajectory

    if args.policy == "optimiser":
        traj = optimiser_demo_trajectory(
            args.axes, margin_mm=args.margin, device=args.device,
            case_id="optimiser_demo")
        path = save_trajectory(traj, os.path.join(args.out, "optimiser_demo.json"))
        print(f"[optimiser] n_burns={traj.n_burns()} "
              f"cov={traj.metrics.get('target_coverage_incl_margin')} "
              f"-> {path}")
        if args.compare:
            _print_compare(args.axes, args)
        return 0

    env = make_env_from_axes(
        args.axes, margin_mm=args.margin, device=args.device,
        spacing_mm=args.spacing, coverage_target=args.coverage,
        max_burns=args.max_burns, use_fdm=args.use_fdm, seed=args.seed,
        case_id="synthetic",
    )
    t0 = time.time()
    traj, hist = rollout(env, policy=args.policy, seed=args.seed)
    if mask_dir:
        traj = env.to_trajectory(case_id=traj.case_id, save_masks_dir=mask_dir)
    elapsed = time.time() - t0
    path = save_trajectory(traj, os.path.join(args.out, f"{args.policy}_rollout.json"))
    total_r = sum(h["reward"] for h in hist)
    print(f"[{args.policy}] burns={traj.n_burns()}  "
          f"cov={traj.metrics.get('target_coverage_incl_margin')}  "
          f"over={traj.metrics.get('healthy_overtreated_mL')}mL  "
          f"time={traj.metrics.get('total_ablation_time_min')}min  "
          f"Σr={total_r:.3f}  ({elapsed:.2f}s)  -> {path}")
    print(env.render())
    if args.compare:
        _print_compare(args.axes, args)
    return 0


def _cohort(args, mask_dir) -> int:
    from endoworld.ablation.sim_env import make_env_from_record, rollout
    from endoworld.ablation.trajectory_schema import save_trajectory

    rows = list(csv.DictReader(open(args.params, encoding="utf-8-sig")))
    if args.limit > 0:
        rows = rows[: args.limit]
    summary = []
    for i, row in enumerate(rows):
        cid = str(row.get("case_id") or row.get("id") or f"case_{i:03d}")
        env = make_env_from_record(
            row, device=args.device, margin_mm=args.margin,
            spacing_mm=args.spacing, coverage_target=args.coverage,
            max_burns=args.max_burns, use_fdm=args.use_fdm, seed=args.seed,
        )
        if env is None:
            print(f"[skip] {cid}: missing diameters")
            continue
        if args.policy == "optimiser":
            from endoworld.ablation.sim_env import optimiser_demo_trajectory
            traj = optimiser_demo_trajectory(
                env.geometry.tumor_axes_mm, margin_mm=args.margin,
                device=args.device, case_id=cid)
        else:
            traj, _ = rollout(env, policy=args.policy, seed=args.seed)
            if mask_dir:
                traj = env.to_trajectory(case_id=cid, save_masks_dir=mask_dir)
            else:
                traj.case_id = cid
                traj.geometry.case_id = cid
        path = save_trajectory(
            traj, os.path.join(args.out, f"{cid}_{args.policy}.json"))
        summary.append({
            "case_id": cid,
            "n_burns": traj.n_burns(),
            **{k: traj.metrics.get(k) for k in (
                "tumor_coverage", "target_coverage_incl_margin",
                "healthy_overtreated_mL", "total_ablation_time_min")},
            "path": path,
        })
        print(f"[{i+1}/{len(rows)}] {cid}: burns={traj.n_burns()} "
              f"cov={traj.metrics.get('target_coverage_incl_margin')}")

    sum_path = os.path.join(args.out, "summary.json")
    json.dump(summary, open(sum_path, "w", encoding="utf-8"), indent=2)
    print(f"[done] {len(summary)} trajectories -> {args.out}  summary={sum_path}")
    return 0


def _print_compare(axes, args) -> None:
    from endoworld.ablation.sim_env import make_env_from_axes, rollout
    from endoworld.ablation.planner import plan_ablation

    plan = plan_ablation(axes, margin_mm=args.margin, device=args.device,
                         spacing_mm=args.spacing, coverage_target=args.coverage,
                         max_burns=args.max_burns)
    env = make_env_from_axes(
        axes, margin_mm=args.margin, device=args.device,
        spacing_mm=args.spacing, coverage_target=args.coverage,
        max_burns=args.max_burns, seed=args.seed)
    traj, _ = rollout(env, policy="greedy", seed=args.seed)
    print("--- compare ---")
    print(f"  planner: burns={len(plan.burns)}  "
          f"cov={plan.metrics.get('target_coverage_incl_margin')}  "
          f"over={plan.metrics.get('healthy_overtreated_mL')}")
    print(f"  env/greedy: burns={traj.n_burns()}  "
          f"cov={traj.metrics.get('target_coverage_incl_margin')}  "
          f"over={traj.metrics.get('healthy_overtreated_mL')}")


def _smoke() -> int:
    """Fast self-check used in CI / local verification."""
    from endoworld.ablation.trajectory_schema import (
        schema_example_dict, dict_to_trajectory, trajectory_to_dict,
        save_trajectory, load_trajectory, save_mask, load_mask,
    )
    from endoworld.ablation.sim_env import make_env_from_axes, rollout

    # schema round-trip
    ex = schema_example_dict()
    traj = dict_to_trajectory(ex)
    assert traj.n_burns() == 1
    assert abs(traj.total_energy_kJ() - 45 * 420 / 1000) < 1e-6

    out = os.path.join("outputs", "ablation_sim", "_smoke")
    os.makedirs(out, exist_ok=True)
    p = save_trajectory(traj, os.path.join(out, "example.json"))
    traj2 = load_trajectory(p)
    assert traj2.case_id == traj.case_id

    # mask I/O
    m = __import__("numpy").zeros((11, 11, 11), dtype=bool)
    m[4:7, 4:7, 4:7] = True
    mp = os.path.join(out, "mask.npz")
    save_mask(mp, m, spacing_mm=1.5, label="demo")
    m2, sp, _ = load_mask(mp)
    assert m2.shape == m.shape and sp == 1.5 and m2.sum() == m.sum()

    # env greedy reaches high coverage on a small nodule
    env = make_env_from_axes((6, 5, 6), margin_mm=5.0, spacing_mm=2.0,
                             max_burns=8, seed=0, case_id="smoke")
    traj_r, hist = rollout(env, policy="greedy", seed=0)
    cov = traj_r.metrics["target_coverage_incl_margin"]
    assert cov >= 0.95, f"greedy coverage too low: {cov}"
    assert traj_r.n_burns() >= 1
    assert len(hist) >= 2
    save_trajectory(traj_r, os.path.join(out, "greedy_smoke.json"))
    env.to_trajectory(case_id="smoke", save_masks_dir=os.path.join(out, "masks"))

    # random policy runs without crash
    env2 = make_env_from_axes((6, 5, 6), margin_mm=5.0, spacing_mm=2.0,
                              max_burns=4, seed=1)
    rollout(env2, policy="random", seed=1)

    print("[smoke] OK — schema round-trip, mask I/O, greedy/random rollout")
    print(f"[smoke] artefacts in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
