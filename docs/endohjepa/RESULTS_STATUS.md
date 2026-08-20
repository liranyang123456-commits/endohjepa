# Current result and comparison status

Historical smoke-scale and debug outputs are not submission evidence. For the
current paper, use `verified_metrics.json` for registered metrics,
`audit_contact_protocol.json` for the audit-contact history, and
`endohjepa.tex` for the scoped interpretation of positive and negative results.

## Evidence supported by the current release

- Strict past-only forecasting: mean steps-1--4 cosine 0.9578 versus 0.9102
  persistence on the registered sensitivity audit.
- Separate bidirectional-cache comparison: 0.978 versus 0.916 persistence,
  0.974 GRU and 0.971 custom gated recurrence on one shared 750-clip validation
  cache.
- SCARED action association: 88.5% deranged-batch wins over 958 overlapping
  contacted-audit windows; a distinct fixed bank gives 92.2% pair wins and
  66.4% all-negative wins.
- Near-wall risk is a negative result: AUC 0.523, below the registered 0.75
  gate.
- The 97.5% value is a forced-future, oracle-goal latent-retrieval diagnostic;
  it is not closed-loop navigation or executed control.

## SOTA interpretation

The current evidence does **not** establish external state of the art. It
supports only an internal, same-cache advantage over the reproduced
persistence, GRU and custom gated-recurrence baselines. Protocols and endpoints
reported by SurgicalMamba, EndoWAM, SurgVista/SurgWorld-Bench, Surg-UniWorld and
SurgWMBench are not directly interchangeable with the pooled-latent and
camera-motion evaluations in this paper.

## Experiments required for a future external-SOTA claim

1. Rerun persistence, GRU, the custom gated recurrence and the Transformer on
   the same strict past-only cache with matched parameter budgets, tuning
   budgets, seeds and sequence-grouped splits.
2. Evaluate action association and risk on an untouched trajectory/case-level
   external set; do not reuse the contacted SCARED audit partition for model
   selection.
3. Run the official SurgWMBench transition, rollout and perturbation-recovery
   protocols, or another directly matched public benchmark.
4. Compare with EndoWAM under the same physical phantom, goal definition and
   closed-loop success criterion; compare phase recognition with SurgicalMamba
   under the complete official Cholec80 protocol.
5. Report controlled ablations for the causal mask, residual anchor, domain
   conditioning and L2 head, with sequence-level uncertainty estimates.

## Submission blocker unrelated to model metrics

Formal review-board/ethics approval or exemption details, decision date and
consent-waiver determination for the private ION and in-house MIS cohorts must
be supplied by the responsible institution before submission.
