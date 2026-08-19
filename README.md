# Endo-HJEPA: Multi-Domain Latent Forecasting with Audited SE(3)-Conditioned Dynamics for Endoscopic Video

Official code and reproducibility package for the Endo-HJEPA manuscript.

Endo-HJEPA studies latent forecasting across laparoscopic, gastrointestinal and
bronchoscopic video. It combines a frozen V-JEPA 2 ViT-L encoder, a
domain-conditioned causal residual forecaster and a separately audited
continuous SE(3)-conditioned branch. The 19-dataset census contains 1,707
sequences and 1.07M decoded frames; the main forecast caches use the eligible
RGB-video subsets.

## Repository layout

```
src/endoworld/          # the endoworld package
  world/                # H-JEPA, continuous SE(3) dynamics, MPC, risk, cache builders
  eval/                 # forecast / probe / navigation / pose-gate evaluation
  understanding/        # V-JEPA 2 encoder wrapper and adaptation
  data/                 # manifests, video-level splits, clip datasets
tests/                  # unit tests (action-path sensitivity, geometry, risk, planner)
docs/endohjepa/         # paper (endohjepa.tex), figures + generators, metric ledger
manifests/              # released aggregate census; local manifest is generated from data
results/                # every JSON cited by docs/endohjepa/verified_metrics.json
requirements.txt
```

## Reproduce the headline numbers

All numbers in the paper come from `docs/endohjepa/verified_metrics.json`, whose
entries point at the JSON files under `results/`.

```bash
pip install -r requirements.txt   # install torch matching your CUDA separately
pip install -e .
python -m pytest tests -q
```

Key entry points:

| Result | Command |
|---|---|
| Latent forecast (6k causal L1) | `python -m endoworld.world.train --encoder vjepa2 --ablation l1 --l1-causal --max-clips 6000` |
| Per-dataset decomposition | `python -m endoworld.eval.per_dataset --ckpt outputs/scale_6000_causal/endohjepa.pt` |
| Physical cache (v2, mono + past-only) | `python -m endoworld.world.build_physical_actions --encoder vjepa2 --stereo-eye top --past-only` |
| Continuous SE(3) dynamics | `python -m endoworld.world.train_continuous_actions --data outputs/physical_actions_v2/sequences.pt --negatives local --skip-test` |
| Oracle-goal retrieval diagnostic | `python -m endoworld.eval.physical_navigation --data outputs/physical_actions_v2/sequences.pt --checkpoint outputs/continuous_actions_v2/continuous_dynamics.pt --trials 200 --normalised-actions` |
| C3VD pose-convention gate | `python -m endoworld.eval.c3vd_pose_gate` |
| CholecT50 dense MIL probe | `python -m endoworld.eval.cholect50_dense_probe` |
| Paper figures | `python docs/endohjepa/make_figures.py` (reads only `verified_metrics.json`) |

Datasets are not redistributed; see `docs/endohjepa/c3vd_download_links.json` for
the C3VD/C3VDv2 download recipe and `docs/endohjepa/DATA_ACCESS.md` for the
expected layout and local-manifest instructions. Private
ION bronchoscopy data cannot be shared; all ION-derived numbers are marked in the
paper.

## Headline results (from `verified_metrics.json`)

- Latent forecast, mean over steps 1--4: **0.978 cosine** vs 0.916
  persistence, 0.974 GRU and 0.971 Mamba on the same 750-clip validation set.
- The separate past-only sensitivity audit is **0.9578** vs 0.9102 persistence.
- The audit-selected SCARED action result is **87.0%** deranged-batch wins;
  a distinct same-sequence bank gives 91.3% pair wins and 66.5%
  all-negative wins (958 overlapping windows, four sequences).
- Corrected near-wall risk fails the prespecified gate: AUC **0.523**.
- The 98.0% forced-future oracle retrieval result is a structurally advantaged
  diagnostic, not navigation or robot control.
- CholecT50 official 5-fold CV: frozen V-JEPA 2 dense MIL probe 0.719 phase /
  0.586 instrument mAP; this is below task-specific supervised SOTA.

No external SOTA claim is made for latent forecasting because no standard
multi-domain endoscopic latent-forecast benchmark was identified. C3VD action
percentages are archived until rerun with the final negative evaluator.

## License

Code: MIT (see LICENSE). Datasets remain under their original licences; C3VD is
CC BY-NC-SA 4.0.
