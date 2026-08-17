# Response to Reviewers — Template

**Manuscript:** Endo-HJEPA: A Hierarchical Joint-Embedding World Model for Unified
Endoscopic Video (submitted to Medical Image Analysis).

We thank the editor and reviewers for their constructive comments. We respond to
each point below; reviewer comments are in *italic*, our responses in plain text,
and manuscript changes are marked **[Change]**. All numbers are reproduced from
`RESULTS.json` / `docs/endohjepa/RESULTS.md` and the runnable `endoworld` package.

---

## Reviewer 1

**R1.1** *<comment>*

**Response.** <response>

**[Change]** <what changed in the manuscript, with section>

---

*(Pre-filled from rebuttal_points.md — the most likely concerns)*

**R1.x — "The forecast margin over GRU is tiny."**
**Response.** The margin is small in absolute terms but statistically decisive:
paired bootstrap + Wilcoxon with Holm correction on 750 video-level held-out
sequences gives p<1e-100 at horizon 4 against both GRU and the Mamba baseline
(Section 5.1). The finding is that forecaster *form* (causal autoregressive)
matters more than capacity; the world model's value is the planning and energy
capabilities flat forecasters lack.

**R1.x — "Latent actions are not grounded."**
**Response.** We agree and report this as a negative result (Section 5.4):
latent actions are a planning device, not interpretable surgical actions. We
quantify grounding honestly and show encoder-level supervision is required.
This corrects a tacit assumption in prior latent-action world models.

**R1.x — "Goal-directed navigation only reaches 10%."**
**Response.** That is the harder arbitrary-target task, bounded by the action-
grounding gap; the 98% figure is forecast-style reach. We do not conflate them
and state the boundary explicitly (Section 5.2).

**R1.x — "No numeric comparison to V-JEPA 2-AC / SurgRec-JEPA."**
**Response.** No public checkpoints exist for those; we compare against
reproducible baselines (ImageNet, DINOv2, VideoMAE, TimeSformer, ViViT, GRU,
Mamba, persistence) with multi-seed means/std, and cite the closed models'
published figures as reference points (Section 6.4).

---

## Reviewer 2

*(to be filled on receipt)*

---

## Summary of changes
- <list of edits made in response>
