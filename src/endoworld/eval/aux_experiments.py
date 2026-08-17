"""Paper auxiliary experiments: STIR tracks, SCARED pose alignment, EndoVis masks.

    python -m endoworld.eval.aux_experiments
These are Endo-HJEPA world-model experiments, not CT ablation planning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def run_stir(root: str) -> dict:
    from endoworld.data.stir_tracks import find_stir_sequences, load_stir_clip
    seqs = find_stir_sequences(root)
    rows = []
    for seq in seqs[:32]:
        clip = load_stir_clip(seq)
        if clip is None:
            continue
        rows.append({
            "seq": str(clip.seq_dir),
            "n_start": int(len(clip.points_start)),
            "n_end": int(len(clip.points_end)),
            "n_frames": len(clip.frames),
            "dt_ms": int(clip.t_end_ms - clip.t_start_ms),
        })
    return {"n_sequences": len(seqs), "loaded": rows}


def run_scared(root: str, n_actions: int = 16) -> dict:
    from endoworld.world.pose_align import action_pose_nmi, quantise_deltas
    from endoworld.world.scared_actions import find_scared_keyframes, scared_pose_deltas
    kfs = find_scared_keyframes(root)
    out = []
    for kf in kfs:
        try:
            d = scared_pose_deltas(kf)
        except Exception as e:
            out.append({"keyframe": str(kf), "error": str(e)})
            continue
        pose_ids = quantise_deltas(d, n_actions)
        # surrogate latent actions = random permutation baseline + identity
        rng = np.random.default_rng(0)
        rand = rng.integers(0, n_actions, size=len(pose_ids))
        out.append({
            "keyframe": str(kf),
            "n_deltas": int(len(d)),
            "trans_rms": float((d[:, :3] ** 2).mean() ** 0.5),
            "rot_rms": float((d[:, 3:] ** 2).mean() ** 0.5),
            "nmi_identity": action_pose_nmi(pose_ids, pose_ids),
            "nmi_random": action_pose_nmi(rand, pose_ids),
        })
    return {"n_keyframes": len(kfs), "rows": out}


def run_c3vd(root: str, n_actions: int = 16) -> dict:
    from endoworld.world.c3vd_actions import find_c3vd_pose_files, load_pose_txt, pose_deltas
    from endoworld.world.pose_align import action_pose_nmi, quantise_deltas
    files = find_c3vd_pose_files(root)
    rows = []
    for p in files:
        d = pose_deltas(load_pose_txt(p))
        ids = quantise_deltas(d, n_actions)
        rows.append({
            "pose_file": str(p),
            "n_deltas": int(len(d)),
            "trans_rms": float((d[:, :3] ** 2).mean() ** 0.5),
            "nmi_identity": action_pose_nmi(ids, ids),
        })
    return {"n_files": len(files), "rows": rows}


def run_endovis(root: str, limit: int = 200) -> dict:
    from endoworld.data.endovis_masks import (
        instrument_binary, list_endovis_pairs, load_class_map, load_mask, presence_vector,
    )
    root_p = Path(root)
    # 2017 lives under endovis2017/ subfolder
    cand = [root_p, root_p / "endovis2017", root_p / "endovis2018"]
    picked = next((c for c in cand if (c / "train" / "image").is_dir()), None)
    if picked is None:
        return {"error": f"no EndoVis train/image under {root}"}
    pairs = list_endovis_pairs(picked, "train")[:limit]
    classes = load_class_map(picked)
    present = np.zeros(8, dtype=np.int64)
    inst_frac = []
    for _, lab in pairs:
        m = load_mask(lab)
        y = presence_vector(m, 8)
        present += y.astype(np.int64)
        inst_frac.append(float(instrument_binary(m).mean()))
    return {
        "root": str(picked),
        "n_pairs": len(pairs),
        "class_map": classes,
        "positives": {classes.get(i, str(i)): int(present[i]) for i in range(8)},
        "mean_instrument_fraction": float(np.mean(inst_frac)) if inst_frac else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/endohjepa/aux_experiments.json")
    ap.add_argument("--stir", default="datasets/STIR")
    ap.add_argument("--scared", default="datasets/SCARED")
    ap.add_argument("--c3vd", default="datasets/C3VD")
    ap.add_argument("--endovis", default="datasets/endovis2017_full")
    args = ap.parse_args()
    report = {
        "paper": "Endo-HJEPA",
        "not_ct_ablation_planning": True,
        "main_table_ok": False,
        "note": "Loader stats only. Encoder-aligned STIR/SCARED/EndoVis live in pose_latent_align / stir_experiment / instrument_mask_exp.",
        "stir": run_stir(args.stir),
        "scared_pose": run_scared(args.scared),
        "c3vd_pose": run_c3vd(args.c3vd),
        "endovis_masks": run_endovis(args.endovis),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)[:4000])
    print(f"[aux] wrote {args.out}")


if __name__ == "__main__":
    main()
