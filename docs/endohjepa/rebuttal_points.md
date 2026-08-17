# Anticipated Reviewer Concerns & Rebuttal Points

**Document purpose:** Pre-emptive rebuttal preparation for the *Medical Image
Analysis* submission *"Endo-HJEPA: A Hierarchical Joint-Embedding World Model
for Unified Endoscopic Video."* Each concern is paired with the strongest
defensible response grounded in the paper's actual results. Numbers are
reproduced verbatim from `RESULTS.json` / `docs/endohjepa/RESULTS.md` and must
not be altered.

---

## Concern 1 — "The forecast margin over GRU is tiny (0.978 vs 0.974)."

**Likely phrasing:** *"The improvement over a simple GRU is within noise."*

**Rebuttal:**
- The margin is small in absolute terms but **statistically decisive**: paired
  bootstrap (1,000 resamples) + Wilcoxon signed-rank with Holm correction on 750
  video-level held-out sequences gives $p<10^{-100}$ at horizon 4 against both
  GRU and the Mamba/SSM baseline (`eval/stats_compare.py`).
- The GRU/Mamba are *strong* baselines precisely because endoscopic latent
  dynamics are smooth; our point (§6.2) is that forecaster *form* (causal
  autoregressive) matters more than capacity, which the ablation isolates.
- The world model's value is not marginal forecast gains but what it *enables*:
  planning (98% reach) and an energy/OOD signal that flat forecasters lack.

---

## Concern 2 — "Latent actions are not grounded; the 'action' claim is overstated."

**Likely phrasing:** *"The latent actions do not correspond to physical or
semantic actions, so the action-conditioned framing is misleading."*

**Rebuttal:**
- We agree and say so explicitly (§5.4): emergent latent actions are a
  *planning device*, not interpretable surgical actions. This is a reported
  negative result, not an overclaim.
- We quantify grounding honestly: physical (SCARED pose NMI 0.41–0.52) and
  semantic (CholecT50 verb at/below chance), plus a supervised-grounding
  analysis showing encoder-level supervision is required.
- This is a contribution in itself: it corrects a tacit assumption in prior
  latent-action world models that emergent codes are action-meaningful.

---

## Concern 3 — "Goal-directed navigation only reaches 10%; planning is weak."

**Likely phrasing:** *"The physical navigation task shows 10% reach—doesn't
this contradict the 98% planning claim?"*

**Rebuttal:**
- The 98% figure is *forecast-style* reach (predict the likely continuation),
  which is what the energy+MPC stack is designed for. The 10% is the strictly
  harder task of reaching an *arbitrary* anatomical target with no action
  labels—bounded by the weak action grounding (§5.4), which we state openly.
- The two numbers measure different things and we do not conflate them; the
  navigation boundary is reported precisely to bound the planning claim.

---

## Concern 4 — "No comparison to V-JEPA 2-AC / SurgRec-JEPA numbers."

**Likely phrasing:** *"The paper does not compare numerically to the most
relevant published world models."*

**Rebuttal:**
- V-JEPA 2-AC and SurgRec-JEPA do not release public checkpoints, so a
  like-for-like re-implementation on our data is not currently possible.
- We therefore compare against reproducible baselines we *can* run (ImageNet,
  DINOv2, VideoMAE, TimeSformer, ViViT, GRU, Mamba, persistence) with multi-seed
  means and standard deviations, and cite the published figures of the closed
  models as reference points with the protocol differences noted.

---

## Concern 5 — "Downstream recognition does not beat DINOv2 on instrument presence."

**Likely phrasing:** *"DINOv2 beats the proposed encoder on instrument mAP
(0.575 vs 0.485)."*

**Rebuttal:**
- Correct, and we report it. The downstream probe is a *representation-quality
  check*, not the world-model claim. V-JEPA 2 is the strongest *video* backbone
  for phase recognition (0.704); DINOv2 (image SSL) is best for instrument
  presence—an expected, honest split between temporal and spatial strengths.
- The world-model contribution (forecast/planning/energy) is orthogonal and is
  where Endo-HJEPA leads.

---

## Concern 6 — "Reproducibility: are splits, caches, and checkpoints released?"

**Rebuttal:**
- Yes. Video-level splits are a deterministic hash of `dataset::sequence_id`;
  encoder caches are shared across ablations; probe heads use a fixed seed; all
  evaluation is single-command (`eval/consolidate_results.py` regenerates
  RESULTS.md/json). Code and protocols are released.

---

## Quick-reference: strongest one-line defenses

| Concern | One-liner |
|---|---|
| Tiny margin over GRU | $p<10^{-100}$ on 750 video-level holds; forecaster *form*, not capacity, is the finding. |
| Ungrounded actions | Reported as a negative result; latent actions are a planning device, not interpretable actions. |
| 10% navigation | That is the harder arbitrary-target task; 98% is forecast-reach—we do not conflate them. |
| No V-JEPA2-AC numbers | No public checkpoints exist; we beat reproducible baselines with multi-seed CIs. |
| DINOv2 > ours on instrument | Recognition is a probe, not the world-model claim; V-JEPA2 is the best *video* backbone for phase. |

---

## Recommended revision actions if reviewers demand more

1. **Full-scale training** on the complete ~1M-frame corpus (the data-scale
   curve is still rising at 6,000 clips) — moderate compute, addresses scale.
2. **Complete Cholec80 phase recognition** after CAMMA approval — the standard
   SOTA benchmark for phase, directly addresses the recognition axis.
3. **A real navigation loop** with action supervision (encoder-level), which
   our analysis shows is required to ground latent actions for goal-directed
   control.
