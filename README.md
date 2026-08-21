# Endo-HJEPA: Hierarchical Latent Prediction for Cross-Orifice Endoscopic Video with Audited SE(3)-Conditioned Evaluation

Official code and reproducibility package for the Endo-HJEPA manuscript.

Endo-HJEPA is a joint-embedding predictive architecture (JEPA) world model for
endoscopic video, built on a frozen V-JEPA 2 ViT-L encoder with a
domain-conditioned causal L1 forecaster, an implemented (but not ablated)
coarse L2 head, a continuous SE(3) action-conditioned branch, and an audited
physical-grounding evaluation protocol across 19 local endoscopic datasets
(1,707 sequences, 1,067,734 decoded frames).

## Repository layout

```
src/endoworld/          # the endoworld package
  world/                # H-JEPA, continuous SE(3) dynamics, MPC, risk, cache builders
  eval/                 # forecast / probe / navigation / pose-diagnostic evaluation
  understanding/        # V-JEPA 2 encoder wrapper and adaptation
  data/                 # manifests, video-level splits, clip datasets
tests/                  # unit tests (action-path sensitivity, geometry, risk, planner)
docs/endohjepa/         # paper (endohjepa.tex), figures + generators, metric ledger
manifests/              # sequence manifest and domain census
results/                # every JSON cited by docs/endohjepa/verified_metrics.json
requirements.txt
```

## Reproduce the headline numbers

All numbers in the paper come from `docs/endohjepa/verified_metrics.json`, whose
entries point at the JSON files under `results/`.

```bash
pip install -r requirements.txt   # install torch matching your CUDA separately
pip install -e .                  # install the endoworld package (src layout)
python -m pytest tests/ -q        # 28 unit tests
```

Key entry points:

| Result | Command |
|---|---|
| Latent forecast (6k causal L1) | `python -m endoworld.world.train --encoder vjepa2 --ablation l1 --l1-causal --max-clips 6000` |
| Per-dataset decomposition | `python -m endoworld.eval.per_dataset --ckpt outputs/scale_6000_causal/endohjepa.pt` |
| Physical cache (v2, mono + past-only) | `python -m endoworld.world.build_physical_actions --encoder vjepa2 --stereo-eye top --past-only` |
| Continuous SE(3) dynamics | `python -m endoworld.world.train_continuous_actions --data outputs/physical_actions_v2/sequences.pt --negatives local --skip-test` |
| Grouped CV (development protocol) | `python -m endoworld.world.train_grouped_cv --data outputs/physical_actions_v2/sequences.pt --negatives local` |
| Oracle-goal retrieval proxy (normalised CEM) | `python -m endoworld.eval.physical_navigation --data outputs/physical_actions_v2/sequences.pt --checkpoint outputs/continuous_actions_v2/continuous_dynamics.pt --trials 200 --normalised-actions` |
| C3VD depth-warp diagnostic | `python -m endoworld.eval.c3vd_pose_gate` |
| CholecT50 dense MIL probe | `python -m endoworld.eval.cholect50_dense_probe` |
| Paper figures | `python docs/endohjepa/make_figures.py` (reads only `verified_metrics.json`) |

Datasets are not redistributed; see `docs/endohjepa/c3vd_download_links.json` for
the C3VD/C3VDv2 download recipe and `manifests/` for the expected layout. Private
ION bronchoscopy data cannot be shared; all ION-derived numbers are marked in the
paper.

## Headline results (from `verified_metrics.json`)

- Latent forecast, horizon 4, video-level held-out (6,000-clip protocol):
  **0.978 cosine** vs 0.916 persistence, 0.974 GRU, 0.971 Mamba; a strictly
  past-only audit remains above persistence (0.9578 vs 0.9102) on a separate
  cache.
- Continuous SE(3) action association (v2 corrected pipeline): **85.2%**
  batch-shuffled real-action win on 958 overlapping windows from four held-out
  SCARED sequences (audit-selected, descriptive); grouped-CV macro fixed-bank
  pair wins 88.9% (worst fold 84.1%). External ten-trajectory C3VD pooled
  58.3% (n=798): convention-specific diagnostic, below the 80% gate.
- Oracle-goal latent-retrieval proxy (single-shot normalised-action CEM, 200
  windows): retrieval beats persistence in **60.0%** of windows with 33.8%
  retrieved-pose translation-error reduction; not executed navigation.
- Corrected-label near-wall risk AUC 0.523 over 3,832 transitions: below the
  0.75 gate; the risk head is inactive in the reported proxy.
- CholecT50 official 5-fold CV: frozen V-JEPA 2 dense MIL probe 0.719 phase /
  0.586 instrument mAP.

## License

Code: MIT (see LICENSE). Datasets remain under their original licences; C3VD is
CC BY-NC-SA 4.0.
