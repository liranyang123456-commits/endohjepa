"""Paper evaluation protocol for Endo-HJEPA (isolated from CT ablation planning).

Runs video-level world-model horizons, L1 vs H-JEPA if both ckpts exist,
cross-domain transfer, and planning proxies. Writes a single JSON the paper
tables can cite. Does **not** load IBM/CBM/BMEO ablation artefacts.

    python -m endoworld.eval.paper_protocol
    python -m endoworld.eval.paper_protocol --smoke
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PAPER = "Endo-HJEPA: A Hierarchical Joint-Embedding World Model for Unified Endoscopic Video"
NOT_THIS_PAPER = [
    "docs/paper IBM/CBM/BMEO CT thermal ablation planning",
    "endoworld.ablation bioheat / needle coverage / follow-up CT efficacy",
]
PROTOCOL = {
    "representation": [
        "Cholec phase / EndoVis instrument / CholecSeg8k semantic linear probes",
        "frozen encoder + video-level (not clip-level) train/test",
        "do not report the leaked 0.992 CholecSeg8k mAP as a main result",
    ],
    "world_model": [
        "horizons 1/4/8/16 cosine + MSE vs persistence and GRU",
        "ablation: L1-only vs L1+L2 vs full H-JEPA",
        "dense L1 tokens vs spatially pooled tokens",
    ],
    "cross_domain": [
        "laparo -> gi and laparo -> bronch zero-shot dynamics",
        "few-shot domain token adaptation is a separate row",
    ],
    "planning": [
        "latent MPC reach vs persistence / random actions on SCARED, C3VD, ION",
        "energy threshold as reject / wall-collision proxy",
        "in-silico only; not a claim of replacing clinical navigation",
    ],
}


def _load_json(path: str | Path):
    p = Path(path)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _try_benchmark(ckpt: Path, latents: Path, out: Path):
    if not ckpt.is_file() or not latents.is_file():
        return None
    from endoworld.eval.world_benchmark import (
        load_predictor, maybe_pool, horizon_table, predict, cross_domain_rows,
    )
    from endoworld.world.h_jepa import persistence_baseline
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    model, kind, history, horizon, _ = load_predictor(blob, device)
    pack = torch.load(latents, map_location=device, weights_only=False)
    Z, D = maybe_pool(pack["Z"]).to(device), pack["D"].to(device)
    t = Z.size(1)
    history = min(history, t - 1)
    horizon = min(horizon, t - history)
    z_hist, z_fut = Z[:, :history], Z[:, history:history + horizon]
    with torch.no_grad():
        pred = predict(model, kind, z_hist, D)
        persist = persistence_baseline(z_hist, horizon)
        rows = horizon_table(pred, persist, z_fut)
        by_dom = cross_domain_rows(model, kind, Z, D, history, horizon)
    report = {"kind": kind, "ablation": blob.get("ablation"), "horizons": rows, "cross_domain": by_dom}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _smoke_synthetic():
    import torch
    from endoworld.world.h_jepa import EndoHJEPA, HJEPAConfig, persistence_baseline
    from endoworld.eval.world_benchmark import horizon_table
    import torch.nn.functional as F
    device = "cpu"
    cfg = HJEPAConfig(latent_dim=32, hidden_dim=64, n_heads=4, n_layers=1,
                      history=4, horizon=4, n_actions=8)
    l1 = EndoHJEPA(cfg).to(device).eval()
    cfg_full = HJEPAConfig(latent_dim=32, hidden_dim=64, n_heads=4, n_layers=1,
                           history=4, horizon=4, n_actions=8, ablation="full")
    full = EndoHJEPA(cfg_full).to(device).eval()
    Z = torch.randn(16, 8, 32)
    D = torch.zeros(16, dtype=torch.long)
    z_hist, z_fut = Z[:, :4], Z[:, 4:8]
    persist = persistence_baseline(z_hist, 4)
    with torch.no_grad():
        p1 = l1.forward_l1(z_hist, D)
        pf = full.forward_l1(z_hist, D)
        plan_a, plan_e = full.plan(z_hist, z_fut[:, -1], D, n_samples=4, steps=4)
    return {
        "l1": horizon_table(p1, persist, z_fut),
        "full": horizon_table(pf, persist, z_fut),
        "plan_energy": plan_e.mean().item(),
        "plan_actions": list(plan_a.shape),
        "l1_vs_full_cos": {
            "l1": F.cosine_similarity(p1, z_fut, dim=-1).mean().item(),
            "full": F.cosine_similarity(pf, z_fut, dim=-1).mean().item(),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/endohjepa_vjepa2")
    ap.add_argument("--l1-ckpt", default="outputs/endohjepa_vjepa2_l1/endohjepa.pt")
    ap.add_argument("--full-ckpt", default="outputs/endohjepa_vjepa2/endohjepa.pt")
    ap.add_argument("--gru-ckpt", default="outputs/endohjepa_vjepa2_gru/endohjepa.pt")
    ap.add_argument("--out", default="outputs/endohjepa_vjepa2/paper_protocol.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    report = {
        "title": PAPER,
        "isolated_from": NOT_THIS_PAPER,
        "protocol": PROTOCOL,
        "main_tables_must_not_include": [
            "CholecSeg8k linear-probe mAP 0.992 from clip-leaky splits",
            "3DGS PSNR as a primary world-model result",
            "pixel video generation FID as the core contribution",
            "CT ablation coverage / CEM43 / follow-up volume curves",
            "scratch 9M V-JEPA numbers (outputs/endohjepa, outputs/vjepa_l1) — debug only",
        ],
        "main_checkpoint": "outputs/endohjepa_vjepa2/endohjepa.pt",
        "debug_only_scratch": True,
        "results": {},
    }

    if args.smoke:
        report["results"]["smoke_synthetic"] = _smoke_synthetic()
    else:
        pairs = {
            "full": (Path(args.full_ckpt), Path(args.full_ckpt).parent / "latents_cache.pt"),
            "l1": (Path(args.l1_ckpt), Path(args.l1_ckpt).parent / "latents_cache.pt"),
            "gru": (Path(args.gru_ckpt), Path(args.gru_ckpt).parent / "latents_cache.pt"),
        }
        for name, (ckpt, lat) in pairs.items():
            got = _try_benchmark(ckpt, lat, Path(args.root) / f"benchmark_{name}.json")
            if got is not None:
                report["results"][name] = got

        plan_json = Path(args.root) / "plan_eval.json"
        if not plan_json.is_file() and Path(args.full_ckpt).is_file():
            from endoworld.eval import plan_eval as pe
            import sys
            sys.argv = ["plan_eval", "--ckpt", args.full_ckpt, "--out", str(plan_json)]
            try:
                pe.main()
            except SystemExit:
                pass
        report["results"]["planning"] = _load_json(plan_json)
        report["results"]["pose_align"] = _load_json(Path(args.root) / "pose_latent_align.json")
        report["results"]["stir"] = _load_json(Path(args.root) / "stir_experiment.json")
        report["results"]["instrument_mask"] = _load_json(Path(args.root) / "instrument_mask_exp.json")
        report["results"]["instrument_probe"] = _load_json(Path(args.root) / "instrument_probe.json")
        report["results"]["scratch_debug_do_not_cite"] = _load_json("outputs/endohjepa/val_metrics.json")

        census = _load_json("manifests/domain_census.json")
        report["data_census"] = census

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[paper_protocol] wrote {args.out}")
    print(f"[paper_protocol] this is {PAPER}")
    print("[paper_protocol] CT ablation planning is a different manuscript.")


if __name__ == "__main__":
    main()
