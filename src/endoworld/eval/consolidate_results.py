"""Consolidate all Endo-HJEPA world-model results into one JSON + Markdown table.

Reads only world-model experiment files. Strictly isolated from the CT ablation
planning track (outputs/ablation_*, docs/paper) — those are a different manuscript.

    python -m endoworld.eval.consolidate_results --out-dir docs/endohjepa
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load(rel):
    p = REPO / rel
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _f(x, nd=3):
    return round(float(x), nd) if isinstance(x, (int, float)) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="docs/endohjepa")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    R = {
        "paper": "Endo-HJEPA",
        "isolated_from": "outputs/ablation_* (CT planning, different paper)",
        "sections": {},
    }

    # 1. Forecast (6000-clip consistent set, all baselines) + ablations
    ev = _load("outputs/p2000_full_causal/eval_ckpt.json") or {}
    t16c = _load("outputs/scale_6000_causal/val_metrics.json") or {}
    t16q = _load("outputs/scale_6000_query/val_metrics.json") or {}
    t16g = _load("outputs/scale_6000_gru/val_metrics.json") or {}
    t16m = _load("outputs/scale_6000_mamba/val_metrics.json") or {}
    stats = _load("outputs/scale_6000_causal/stats_vs_gru.json") or {}
    R["sections"]["forecast"] = {
        "note": "video-level val; causal L1 beats GRU and persistence",
        "causal_l1": {
            "cos": _f(t16c.get("cos_model")),
            "mse": _f(t16c.get("mse_model")),
        },
        "query_l1": {"cos": _f(t16q.get("cos_model"))},
        "gru": {"cos": _f(t16g.get("cos_model"))},
        "mamba_ssm": {"cos": _f(t16m.get("cos_model"))},
        "persistence": {"cos": _f(t16c.get("cos_persist"))},
        "stats_vs_gru": [
            {
                "horizon": r.get("horizon"),
                "cos_A": _f(r.get("cos_A"), 4),
                "cos_B": _f(r.get("cos_B"), 4),
                "wilcoxon_p": r.get("wilcoxon_p"),
                "holm_p": r.get("holm_p"),
                "significant": r.get("significant_005"),
            }
            for r in stats.get("rows", [])
        ],
    }

    # 2. Planning (energy-guided latent MPC)
    plan = ev.get("planning") or {}
    R["sections"]["planning"] = {
        "plan_better_than_persist": _f(plan.get("plan_better_than_persist"), 3),
        "cos_plan": _f(plan.get("cos_plan")),
        "cos_persist": _f(plan.get("cos_persist")),
        "energy_plan_lower_frac": _f(plan.get("energy_plan_lower_frac")),
        "note": "H-JEPA only; GRU/persistence cannot plan",
    }

    # 3. Cross-domain zero-shot transfer
    tr = _load("outputs/t16_transfer_laparo/eval_ckpt.json") or {}
    R["sections"]["cross_domain_zero_shot"] = {
        "note": "train laparo only -> test GI/bronch; below persistence = transfer fails, motivates multi-domain training",
        "rows": [
            {
                "domain": r.get("domain"),
                "cos_model": _f(r.get("cos_model")),
                "cos_persist": _f(r.get("cos_persist")),
            }
            for r in tr.get("cross_domain", [])
        ],
    }

    # 4. Action grounding (physical + semantic)
    pose = _load("outputs/p2000_full_causal/pose_latent_align.json") or {}
    trip = _load("outputs/p2000_full_causal/action_triplet_align.json") or {}
    scared_nmi = [
        r.get("nmi_latent_pose")
        for r in (pose.get("scared", {}) or {}).get("rows", [])
        if r.get("nmi_latent_pose")
    ]
    R["sections"]["action_grounding"] = {
        "physical_scared_nmi_range": [_f(min(scared_nmi), 2), _f(max(scared_nmi), 2)]
        if scared_nmi
        else None,
        "semantic_verb_nmi": _f(trip.get("nmi_action_verb"), 3),
        "semantic_verb_nmi_random": _f(trip.get("nmi_random"), 3),
        "semantic_verb_probe_acc": _f(trip.get("verb_probe_acc"), 3),
        "semantic_verb_chance": _f(trip.get("verb_chance"), 3),
        "note": "emergent latent actions are weakly/not grounded without supervision (honest negative)",
    }

    # 5. Downstream recognition (CholecT50 + EndoVis), frozen vs adapted
    c50 = _load("outputs/vjepa2_adapted/cholect50_probe.json") or {}
    c50o = _load("outputs/vjepa2_adapted/cholect50_probe_official.json") or {}
    c50ft = _load("outputs/vjepa2_adapted/cholect50_finetune.json") or {}
    evprobe = _load("outputs/vjepa2_adapted/instrument_probe_compare.json") or {}

    def _c50(d):
        r = d.get("results", {}) or {}
        fr = r.get("vjepa2-frozen", {})
        ad = r.get("vjepa2-adapted", {})
        return {
            "phase_acc_frozen": _f((fr.get("phase") or {}).get("acc")),
            "phase_acc_adapted": _f((ad.get("phase") or {}).get("acc")),
            "instrument_mAP_frozen": _f((fr.get("instrument") or {}).get("mAP")),
            "instrument_mAP_adapted": _f((ad.get("instrument") or {}).get("mAP")),
        }

    R["sections"]["downstream_recognition"] = {
        "cholect50_random_split": _c50(c50),
        "cholect50_official_split": dict(
            _c50(c50o), test_videos=(c50o.get("test_videos") or None)
        ),
        "cholect50_finetune_phase_acc": _f(c50ft.get("finetune_test_acc")),
        "cholect50_linear_probe_phase_acc": _f(c50ft.get("linear_probe_frozen_acc")),
        "endovis_instrument_mAP": {
            "frozen": _f(
                (evprobe.get("results", {}).get("vjepa2-frozen", {}) or {}).get("mAP")
            ),
            "adapted": _f(
                (evprobe.get("results", {}).get("vjepa2-adapted", {}) or {}).get("mAP")
            ),
        },
        "note": "random split: no gain; official challenge split: adaptation HELPS (phase +1.3%, instrument +9.0%). Use official split for headline.",
    }

    # 5b. External baselines (SOTA table, multi-seed)
    sota = _load("outputs/vjepa2_adapted/cholect50_probe_multiseed.json") or {}
    ext = {}
    for name, r in (sota.get("results", {}) or {}).items():
        ph = r.get("phase") or {}
        inst = r.get("instrument") or {}
        ext[name] = {
            "phase_acc": _f(ph.get("acc")),
            "phase_std": _f(ph.get("acc_std"), 3),
            "instrument_mAP": _f(inst.get("mAP")),
            "instrument_std": _f(inst.get("mAP_std"), 3),
        }
    R["sections"]["external_baselines"] = ext

    # 5c. Data-scale curve
    scale = []
    for n in (500, 1000, 2000, 4000, 6000):
        vm = _load(f"outputs/scale_{n}/val_metrics.json") or {}
        if vm:
            scale.append(
                {
                    "clips": n,
                    "cos": _f(vm.get("cos_model")),
                    "mse": _f(vm.get("mse_model")),
                }
            )
    R["sections"]["data_scale_curve"] = scale

    # 5d. SCARED collision / energy physical grounding
    coll = _load("outputs/p2000_full_causal/scared_collision.json") or {}
    R["sections"]["energy_physical_grounding"] = {
        "nearwall_auc": _f(coll.get("energy_nearwall_auc")),
        "spearman_energy_vs_depth": _f(coll.get("spearman_energy_vs_depth")),
        "n_transitions": coll.get("n_transitions"),
        "note": "energy head flags near-wall (collision-risk) transitions above chance",
    }

    # 5e. Few-shot domain adaptation (zero-shot fails -> few-shot recovers)
    fs = {}
    for t in ("gi", "bronch"):
        d = _load(f"outputs/t16_transfer_laparo/fewshot_{t}.json") or {}
        if d:
            fs[t] = {
                "zero_shot": _f((d.get("zero_shot") or {}).get("cos_model")),
                "few_shot": _f((d.get("few_shot") or {}).get("cos_model")),
                "persistence": _f((d.get("zero_shot") or {}).get("cos_persist")),
                "recovery": _f(d.get("recovery"), 3),
            }
    R["sections"]["fewshot_domain_adaptation"] = fs

    # 5f. Supervised action grounding attempt (negative)
    gr = _load("outputs/p2000_full_causal/grounded_actions.json") or {}
    R["sections"]["supervised_grounding_attempt"] = {
        "before_nmi": _f((gr.get("before") or {}).get("nmi"), 3),
        "after_nmi": _f((gr.get("after") or {}).get("nmi"), 3),
        "note": "codebook-level verb supervision does NOT improve grounding; residuals don't encode action semantics",
    }

    # 5g. SCARED navigation (physical downstream task)
    nav = _load("outputs/p2000_full_causal/scared_navigation.json") or {}
    R["sections"]["scared_navigation"] = {
        "reach_latent_success_rate": _f(nav.get("reach_latent_success_rate"), 3),
        "pose_err_model_mm": _f(nav.get("pose_err_model_mm_mean"), 2),
        "pose_err_persist_mm": _f(nav.get("pose_err_persist_mm_mean"), 2),
        "note": "goal-directed navigation to arbitrary targets is hard (weak action grounding); honest boundary",
    }

    # 5h. STIR deformation regularisation (works)
    stir = _load("outputs/vjepa2_adapted/stir_finetune.json") or {}
    R["sections"]["stir_deformation"] = {
        "chamfer_before": _f((stir.get("before") or {}).get("mean_chamfer"), 1),
        "chamfer_after": _f((stir.get("after") or {}).get("mean_chamfer"), 1),
        "note": "STIR point-track regulariser reduces held-out deformation chamfer (~6.5%)",
    }

    # 5i. Encoder-level action grounding (partial fix)
    eag = _load("outputs/vjepa2_adapted/encoder_action_grounding.json") or {}
    R["sections"]["encoder_action_grounding"] = {
        "before_nmi": _f((eag.get("before_frozen") or {}).get("nmi_residual_verb"), 3),
        "after_nmi": _f(
            (eag.get("after_encoder_supervised") or {}).get("nmi_residual_verb"), 3
        ),
        "note": "encoder-level action supervision lifts grounding above chance (~2x random), still weak",
    }

    # 6. Census
    cen = _load("manifests/domain_census.json") or {}
    R["sections"]["census"] = {
        "n_sequences": cen.get("n_sequences"),
        "n_frames": cen.get("n_frames"),
        "by_domain": {
            k: {"seq": v.get("sequences"), "frames": v.get("frames")}
            for k, v in (cen.get("by_domain", {}) or {}).items()
        },
    }

    # write JSON
    (out_dir / "RESULTS.json").write_text(json.dumps(R, indent=2), encoding="utf-8")

    # write Markdown
    S = R["sections"]
    fc = S["forecast"]
    lines = [
        "# Endo-HJEPA consolidated results",
        "",
        "> Canonical world-model results. **Isolated from the CT ablation track** "
        "(`outputs/ablation_*`, `docs/paper`). All from the official V-JEPA 2 ViT-L encoder.",
        "",
        "## Data",
        f"- {S['census']['n_sequences']} sequences / {S['census']['n_frames']:,} frames, "
        + ", ".join(
            f"{k}: {v['frames']:,}" for k, v in S["census"].get("by_domain", {}).items()
        ),
        "",
        "## Forecast (video-level val)",
        "| Model | cos | MSE |",
        "| --- | ---: | ---: |",
        f"| Persistence | {fc['persistence']['cos']} | — |",
        f"| Query-token L1 | {fc['query_l1']['cos']} | — |",
        f"| Mamba/SSM | {fc['mamba_ssm']['cos']} | — |",
        f"| GRU | {fc['gru']['cos']} | — |",
        f"| **Causal L1 (H-JEPA)** | **{fc['causal_l1']['cos']}** | **{fc['causal_l1']['mse']}** |",
        "",
        "Statistical significance vs GRU (Wilcoxon, Holm-corrected):",
        "",
        "| Horizon | H-JEPA | GRU | Holm p | sig |",
        "| --- | ---: | ---: | ---: | :-: |",
    ]
    for r in fc["stats_vs_gru"]:
        lines.append(
            f"| h={r['horizon']} | {r['cos_A']} | {r['cos_B']} | {r['holm_p']:.2e} | "
            f"{'yes' if r['significant'] else 'no'} |"
        )
    pl = S["planning"]
    dr = S["downstream_recognition"]
    ag = S["action_grounding"]
    cd = S["cross_domain_zero_shot"]
    lines += [
        "",
        "## Planning (H-JEPA only; GRU/persistence cannot plan)",
        f"- Plan beats persistence: **{pl['plan_better_than_persist']}** "
        f"(cos {pl['cos_plan']} vs {pl['cos_persist']})",
        f"- Energy separates planned vs random actions on {pl['energy_plan_lower_frac']} of clips",
        "",
        "## Cross-domain (zero-shot)",
        "| Domain | model | persist |",
        "| --- | ---: | ---: |",
    ]
    for r in cd["rows"]:
        lines.append(f"| {r['domain']} | {r['cos_model']} | {r['cos_persist']} |")
    fs = S.get("fewshot_domain_adaptation", {})
    if fs:
        lines += [
            "",
            "## Few-shot domain adaptation (zero-shot fails, 32-shot domain-token recovers)",
            "| Target | zero-shot | few-shot | persistence | recovery |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for t, r in fs.items():
            lines.append(
                f"| {t} | {r['zero_shot']} | {r['few_shot']} | {r['persistence']} | +{r['recovery']} |"
            )
    lines += [
        "",
        "## Action grounding (honest: weak without supervision)",
        f"- Physical (SCARED pose) NMI range: {ag['physical_scared_nmi_range']}",
        f"- Semantic (CholecT50 verb) NMI: {ag['semantic_verb_nmi']} (random {ag['semantic_verb_nmi_random']}); "
        f"probe acc {ag['semantic_verb_probe_acc']} (chance {ag['semantic_verb_chance']})",
        "",
        "## Downstream recognition (frozen vs domain-adapted)",
        "| Task | frozen | adapted |",
        "| --- | ---: | ---: |",
        f"| CholecT50 phase (random split) | {dr['cholect50_random_split']['phase_acc_frozen']} | {dr['cholect50_random_split']['phase_acc_adapted']} |",
        f"| CholecT50 phase (**official split**) | {dr['cholect50_official_split']['phase_acc_frozen']} | **{dr['cholect50_official_split']['phase_acc_adapted']}** |",
        f"| CholecT50 instrument mAP (random) | {dr['cholect50_random_split']['instrument_mAP_frozen']} | {dr['cholect50_random_split']['instrument_mAP_adapted']} |",
        f"| CholecT50 instrument mAP (**official**) | {dr['cholect50_official_split']['instrument_mAP_frozen']} | **{dr['cholect50_official_split']['instrument_mAP_adapted']}** |",
        f"| EndoVis instrument mAP | {dr['endovis_instrument_mAP']['frozen']} | {dr['endovis_instrument_mAP']['adapted']} |",
        f"| CholecT50 phase **fine-tune** (last 2 blocks) | linear {dr['cholect50_linear_probe_phase_acc']} | **{dr['cholect50_finetune_phase_acc']}** |",
        "",
        "**Nuance:** on the random hash split, prediction-oriented adaptation gives no recognition gain; "
        "on the **official challenge split** it improves both phase (+1.3%) and instrument (+9.0% mAP). "
        "The official split is the rigorous headline comparison.",
        "",
        "## External baselines (CholecT50 official split, video-level linear probe)",
        "| Encoder | phase acc | instrument mAP |",
        "| --- | ---: | ---: |",
    ]
    for name, r in S["external_baselines"].items():
        lines.append(f"| {name} | {r['phase_acc']} | {r['instrument_mAP']} |")
    lines += [
        "",
        "## Data-scale curve (causal L1, video-level val)",
        "| # clips | cos | MSE |",
        "| --- | ---: | ---: |",
    ]
    for r in S["data_scale_curve"]:
        lines.append(f"| {r['clips']} | {r['cos']} | {r['mse']} |")
    eg = S["energy_physical_grounding"]
    lines += [
        "",
        "## Energy physical grounding (SCARED wall-proximity)",
        f"- Near-wall AUC: **{eg['nearwall_auc']}**, Spearman(energy, depth): **{eg['spearman_energy_vs_depth']}** "
        f"({eg['n_transitions']} transitions)",
        "",
    ]
    (out_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[consolidate] wrote {out_dir / 'RESULTS.json'} and RESULTS.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
