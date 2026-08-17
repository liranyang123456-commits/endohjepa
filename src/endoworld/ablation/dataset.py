"""Build a mixed trajectory dataset for ablation path learning.

Sources
-------
1. ``optimiser``  — classical ``plan_ablation`` demonstrations (safe, high coverage)
2. ``simulated``  — greedy / random / noisy-greedy rollouts in ``AblationSimEnv``
3. ``clinical``   — follow-up labelled cases (outcome only; geometry from cohort
                    when available) and any JSON trajectories already on disk

The resulting folder::

    outputs/ablation_learn_traj/
      optimiser/*.json
      simulated/*.json
      clinical/*.json
      index.json          # manifest of all trajectories + preference scores
      steps.npz           # flattened (state, action) pairs for BC training

    python -m endoworld.ablation.dataset --params manifests/nodule_params.csv \\
        --followup outputs/ablation_followup/followup_summary.csv \\
        --out outputs/ablation_learn_traj --force-zone 10
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Iterable

import numpy as np

from endoworld.ablation.sim_env import (
    AblationAction,
    make_env_from_axes,
    make_env_from_record,
    optimiser_demo_trajectory,
    rollout,
)
from endoworld.ablation.trajectory_schema import (
    AblationTrajectory,
    BurnStep,
    DeviceParams,
    LesionGeometry,
    OutcomeLabel,
    geometry_from_record_row,
    load_trajectory,
    save_trajectory,
    trajectory_to_dict,
)


# --------------------------------------------------------------------------- #
# Preference / outcome helpers
# --------------------------------------------------------------------------- #
def preference_from_metrics(metrics: dict) -> float:
    """Scalar preference used for ranking trajectories (higher = better)."""
    cov = float(metrics.get("target_coverage_incl_margin")
                or metrics.get("target_coverage") or 0.0)
    over = float(metrics.get("healthy_overtreated_mL") or 0.0)
    burns = float(metrics.get("n_burns") or 0.0)
    tmin = float(metrics.get("total_ablation_time_min") or 0.0)
    return round(cov - 0.04 * over - 0.015 * burns - 0.01 * tmin, 4)


def parse_followup_row(row: dict) -> OutcomeLabel:
    traj = row.get("trajectory_mL") or ""
    days, vols = [], []
    for part in traj.split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        d, v = part.split(":", 1)
        d = d.strip().lstrip("d")
        try:
            days.append(float(d))
            vols.append(float(v))
        except ValueError:
            continue
    verdict_raw = (row.get("verdict") or "unknown").lower()
    if "complete" in verdict_raw:
        verdict = "complete_ablation"
        pref = 1.0
    elif "indeterminate" in verdict_raw:
        verdict = "indeterminate"
        pref = 0.3
    elif "incomplete" in verdict_raw or "residual" in verdict_raw:
        verdict = "incomplete"
        pref = 0.0
    else:
        verdict = "unknown"
        pref = 0.5
    return OutcomeLabel(
        verdict=verdict,  # type: ignore[arg-type]
        pre_volume_mL=vols[0] if vols else None,
        peak_volume_mL=max(vols) if vols else None,
        late_volume_mL=vols[-1] if vols else None,
        followup_days=days,
        followup_volumes_mL=vols,
        preference_score=pref,
        note=row.get("verdict") or "",
    )


# --------------------------------------------------------------------------- #
# State / action featurisation for BC
# --------------------------------------------------------------------------- #
STATE_DIM = 16
ACTION_DIM = 5  # dx, dy, dz, power_W, time_s


def state_features(obs: dict, geometry: LesionGeometry) -> np.ndarray:
    """Fixed-length state vector from env observation + lesion geometry."""
    a, b, c = geometry.tumor_axes_mm
    uc = obs.get("uncovered_centroid_mm") or (0.0, 0.0, 0.0)
    last = obs.get("last_zone_axes_mm") or (0.0, 0.0, 0.0)
    feat = [
        float(a), float(b), float(c),
        float(geometry.margin_mm),
        float(geometry.tumor_volume_mL),
        float(obs.get("coverage") or 0.0),
        float(obs.get("tumor_coverage") or 0.0),
        float(obs.get("overtreat_mL") or 0.0),
        float(obs.get("n_burns") or 0.0),
        float(obs.get("remaining_budget") or 1.0),
        float(obs.get("uncovered_fraction") or 1.0),
        float(uc[0]), float(uc[1]), float(uc[2]),
        float(last[0]) if last else 0.0,
        float(geometry.airway_generation or 0.0),
    ]
    return np.asarray(feat, dtype=np.float32)


def action_features(action: AblationAction | dict) -> np.ndarray:
    if isinstance(action, AblationAction):
        d = action.as_dict()
    else:
        d = action
    pos = d.get("position_mm") or (0, 0, 0)
    return np.asarray([
        float(pos[0]), float(pos[1]), float(pos[2]),
        float(d.get("power_W") or 0.0),
        float(d.get("time_s") or 0.0),
    ], dtype=np.float32)


def replay_trajectory_to_pairs(
    traj: AblationTrajectory,
    force_zone_mm: float | None = None,
    spacing_mm: float | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Re-simulate a stored trajectory to recover (state, action) pairs."""
    sp = spacing_mm or traj.geometry.spacing_mm
    env = make_env_from_axes(
        traj.geometry.tumor_axes_mm,
        margin_mm=traj.geometry.margin_mm,
        device=traj.device.device_type,
        spacing_mm=sp,
        coverage_target=0.99,
        max_burns=max(12, len(traj.steps) + 2),
        force_zone_mm=force_zone_mm,
        case_id=traj.case_id,
    )
    # copy anatomy fields
    for k in ("lobe", "airway_generation", "dist_pleura_mm",
              "dist_chestwall_mm", "dist_vessel_mm"):
        setattr(env.geometry, k, getattr(traj.geometry, k))
    obs, _ = env.reset()
    pairs = []
    for step in traj.steps:
        act = AblationAction(
            position_mm=tuple(step.position_mm),
            power_W=step.power_W,
            time_s=step.time_s,
            temperature_C=step.temperature_C,
        )
        pairs.append((state_features(obs, env.geometry), action_features(act)))
        obs, _, term, trunc, _ = env.step(act)
        if term or trunc:
            break
    return pairs


# --------------------------------------------------------------------------- #
# Dataset builders
# --------------------------------------------------------------------------- #
def _ensure_dirs(root: str) -> dict:
    paths = {
        "root": root,
        "optimiser": os.path.join(root, "optimiser"),
        "simulated": os.path.join(root, "simulated"),
        "clinical": os.path.join(root, "clinical"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def build_optimiser_demos(
    rows: list[dict],
    out_dir: str,
    device: str = "MWA",
    margin_mm: float = 5.0,
    spacing_mm: float = 1.5,
) -> list[dict]:
    entries = []
    for i, row in enumerate(rows):
        geom = geometry_from_record_row(row, margin_mm=margin_mm, spacing_mm=spacing_mm)
        if geom is None:
            continue
        cid = geom.case_id or f"opt_{i:03d}"
        traj = optimiser_demo_trajectory(
            geom.tumor_axes_mm, margin_mm=margin_mm, device=device, case_id=cid)
        traj.geometry = geom
        traj.geometry.case_id = cid
        traj.outcome.preference_score = preference_from_metrics(traj.metrics)
        path = save_trajectory(traj, os.path.join(out_dir, f"{cid}.json"))
        entries.append({
            "case_id": cid, "source": "optimiser", "path": path,
            "n_burns": traj.n_burns(),
            "preference": traj.outcome.preference_score,
            **{k: traj.metrics.get(k) for k in (
                "tumor_coverage", "target_coverage_incl_margin",
                "healthy_overtreated_mL")},
        })
    return entries


def build_simulated_rollouts(
    rows: list[dict],
    out_dir: str,
    policies: Iterable[str] = ("greedy", "random"),
    device: str = "MWA",
    margin_mm: float = 5.0,
    spacing_mm: float = 1.5,
    force_zone_mm: float | None = 10.0,
    max_burns: int = 20,
    n_random: int = 1,
    seed: int = 0,
) -> list[dict]:
    entries = []
    for i, row in enumerate(rows):
        for pol in policies:
            n_rep = n_random if pol == "random" else 1
            for r in range(n_rep):
                env = make_env_from_record(
                    row, device=device, margin_mm=margin_mm,
                    spacing_mm=spacing_mm, max_burns=max_burns,
                    force_zone_mm=force_zone_mm, seed=seed + i * 17 + r,
                    coverage_target=0.99,
                )
                if env is None:
                    continue
                # noisy-greedy: greedy position + random power/time jitter
                if pol == "noisy_greedy":
                    def _noisy(obs, e, rng=np.random.default_rng(seed + i + r)):
                        a = e.greedy_action()
                        pw = float(rng.choice(e.device.power_presets_W))
                        ts = float(rng.choice(e.device.time_presets_s))
                        jitter = rng.normal(0, 1.5, size=3)
                        pos = tuple(float(x + j) for x, j in zip(a.position_mm, jitter))
                        return AblationAction(pos, pw, ts)
                    traj, _ = rollout(env, policy=_noisy, seed=seed + i + r)
                    tag = "noisy_greedy"
                else:
                    traj, _ = rollout(env, policy=pol, seed=seed + i + r)
                    tag = pol
                cid = env.geometry.case_id or f"sim_{i:03d}"
                traj.case_id = f"{cid}_{tag}" + (f"_r{r}" if n_rep > 1 else "")
                traj.geometry.case_id = traj.case_id
                traj.source = "simulated"
                traj.outcome.preference_score = preference_from_metrics(traj.metrics)
                traj.meta["policy"] = tag
                path = save_trajectory(
                    traj, os.path.join(out_dir, f"{traj.case_id}.json"))
                entries.append({
                    "case_id": traj.case_id, "source": "simulated",
                    "policy": tag, "path": path,
                    "n_burns": traj.n_burns(),
                    "preference": traj.outcome.preference_score,
                    **{k: traj.metrics.get(k) for k in (
                        "tumor_coverage", "target_coverage_incl_margin",
                        "healthy_overtreated_mL")},
                })
    return entries


def build_clinical_from_followup(
    followup_csv: str,
    params_csv: str | None,
    out_dir: str,
    device: str = "MWA",
    margin_mm: float = 5.0,
) -> list[dict]:
    """Attach follow-up outcomes to nearest geometry (or synthetic axes from volume)."""
    fu_rows = list(csv.DictReader(open(followup_csv, encoding="utf-8-sig")))
    geom_pool: list[LesionGeometry] = []
    if params_csv and os.path.isfile(params_csv):
        for row in csv.DictReader(open(params_csv, encoding="utf-8-sig")):
            g = geometry_from_record_row(row, margin_mm=margin_mm)
            if g is not None:
                geom_pool.append(g)

    entries = []
    for row in fu_rows:
        cid = str(row.get("case") or row.get("case_id") or "fu")
        outcome = parse_followup_row(row)
        # pick geometry: match by volume if possible, else first / synthetic
        if geom_pool and outcome.pre_volume_mL:
            def _vol_err(g):
                return abs(g.tumor_volume_mL - float(outcome.pre_volume_mL))
            geom = min(geom_pool, key=_vol_err)
            # copy with new id
            geom = LesionGeometry(**{
                **{f: getattr(geom, f) for f in LesionGeometry.__dataclass_fields__},
                "case_id": cid,
            })
        elif outcome.pre_volume_mL and outcome.pre_volume_mL > 0:
            # isotropic ball from volume
            r = (outcome.pre_volume_mL * 1000.0 * 3 / (4 * np.pi)) ** (1 / 3)
            geom = LesionGeometry(
                case_id=cid, tumor_axes_mm=(r, r, r), margin_mm=margin_mm)
        else:
            geom = LesionGeometry(
                case_id=cid, tumor_axes_mm=(8.0, 7.0, 8.0), margin_mm=margin_mm)

        # Clinical burn logs are unavailable → leave steps empty, but store
        # an optimiser demo as a *proposed* plan annotated with real outcome.
        demo = optimiser_demo_trajectory(
            geom.tumor_axes_mm, margin_mm=margin_mm, device=device, case_id=cid)
        demo.geometry = geom
        demo.outcome = outcome
        demo.source = "clinical"
        demo.meta["note"] = (
            "steps from optimiser proposal; outcome from real follow-up "
            "(no intra-op burn log available)"
        )
        path = save_trajectory(demo, os.path.join(out_dir, f"{cid}.json"))
        entries.append({
            "case_id": cid, "source": "clinical", "path": path,
            "n_burns": demo.n_burns(),
            "preference": outcome.preference_score,
            "verdict": outcome.verdict,
            **{k: demo.metrics.get(k) for k in (
                "tumor_coverage", "target_coverage_incl_margin",
                "healthy_overtreated_mL")},
        })
    return entries


def flatten_steps(
    index: list[dict],
    force_zone_mm: float | None = None,
    sources: Iterable[str] = ("optimiser", "simulated"),
) -> dict:
    """Build supervised (X, y) arrays from trajectory JSON files."""
    Xs, ys, meta = [], [], []
    for e in index:
        if e["source"] not in sources:
            continue
        # Keep optimiser + greedy demos always; drop only clearly failed randoms
        pol = e.get("policy")
        cov = float(e.get("target_coverage_incl_margin") or 0.0)
        if pol == "random" and cov < 0.5:
            continue
        if pol == "noisy_greedy" and cov < 0.7:
            continue
        traj = load_trajectory(e["path"])
        if not traj.steps:
            continue
        # force-zone only for multi-burn simulated policies
        fz = None
        if traj.source == "simulated" and traj.meta.get("policy") in (
            "greedy", "random", "noisy_greedy",
        ):
            fz = force_zone_mm
        try:
            pairs = replay_trajectory_to_pairs(traj, force_zone_mm=fz)
        except Exception as ex:
            print(f"[warn] replay failed {e['case_id']}: {ex}")
            continue
        for s, a in pairs:
            Xs.append(s)
            ys.append(a)
            meta.append({
                "case_id": e["case_id"], "source": e["source"],
                "policy": pol,
            })
    X = np.stack(Xs) if Xs else np.zeros((0, STATE_DIM), np.float32)
    y = np.stack(ys) if ys else np.zeros((0, ACTION_DIM), np.float32)
    return {"X": X, "y": y, "meta": meta,
            "state_dim": STATE_DIM, "action_dim": ACTION_DIM}


def build_dataset(
    params_csv: str,
    out_dir: str,
    followup_csv: str | None = None,
    device: str = "MWA",
    margin_mm: float = 5.0,
    spacing_mm: float = 1.5,
    force_zone_mm: float | None = 10.0,
    limit: int = 0,
    seed: int = 0,
    include_noisy: bool = True,
) -> dict:
    paths = _ensure_dirs(out_dir)
    rows = list(csv.DictReader(open(params_csv, encoding="utf-8-sig")))
    if limit > 0:
        rows = rows[:limit]

    print(f"[dataset] {len(rows)} planning cases → {out_dir}")
    index: list[dict] = []
    index += build_optimiser_demos(
        rows, paths["optimiser"], device=device,
        margin_mm=margin_mm, spacing_mm=spacing_mm)
    print(f"  optimiser demos: {sum(1 for e in index if e['source']=='optimiser')}")

    pols = ["greedy", "random"]
    if include_noisy:
        pols.append("noisy_greedy")
    sim = build_simulated_rollouts(
        rows, paths["simulated"], policies=pols, device=device,
        margin_mm=margin_mm, spacing_mm=spacing_mm,
        force_zone_mm=force_zone_mm, seed=seed, n_random=2)
    index += sim
    print(f"  simulated rollouts: {len(sim)}")

    if followup_csv and os.path.isfile(followup_csv):
        clin = build_clinical_from_followup(
            followup_csv, params_csv, paths["clinical"],
            device=device, margin_mm=margin_mm)
        index += clin
        print(f"  clinical (follow-up labelled): {len(clin)}")

    # flatten for BC
    pack = flatten_steps(index, force_zone_mm=force_zone_mm)
    npz_path = os.path.join(out_dir, "steps.npz")
    np.savez_compressed(npz_path, X=pack["X"], y=pack["y"])
    meta_path = os.path.join(out_dir, "steps_meta.json")
    json.dump(pack["meta"], open(meta_path, "w", encoding="utf-8"), indent=2)

    index_path = os.path.join(out_dir, "index.json")
    summary = {
        "n_trajectories": len(index),
        "n_steps": int(pack["X"].shape[0]),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "by_source": {
            s: sum(1 for e in index if e["source"] == s)
            for s in ("optimiser", "simulated", "clinical")
        },
        "force_zone_mm": force_zone_mm,
        "entries": index,
    }
    json.dump(summary, open(index_path, "w", encoding="utf-8"), indent=2)
    print(f"[dataset] {summary['n_trajectories']} traj, {summary['n_steps']} steps "
          f"→ {index_path}")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build mixed ablation trajectory dataset")
    ap.add_argument("--params", default="manifests/nodule_params.csv")
    ap.add_argument("--followup",
                    default="outputs/ablation_followup/followup_summary.csv")
    ap.add_argument("--out", default="outputs/ablation_learn_traj")
    ap.add_argument("--device", default="MWA")
    ap.add_argument("--margin", type=float, default=5.0)
    ap.add_argument("--spacing", type=float, default=1.5)
    ap.add_argument("--force-zone", type=float, default=10.0,
                    help="cap transverse zone radius for multi-burn sim (0=off)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-noisy", action="store_true")
    args = ap.parse_args(argv)

    fz = args.force_zone if args.force_zone > 0 else None
    build_dataset(
        args.params, args.out, followup_csv=args.followup,
        device=args.device, margin_mm=args.margin, spacing_mm=args.spacing,
        force_zone_mm=fz, limit=args.limit, seed=args.seed,
        include_noisy=not args.no_noisy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
