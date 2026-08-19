# Endo-HJEPA: Predictive Foundations and Physical-Grounding Limits of a Cross-Orifice Endoscopic World Model

Official code and reproducibility package for the Endo-HJEPA manuscript.

Endo-HJEPA is a hierarchical joint-embedding predictive architecture (JEPA) world
model for endoscopic video, built on a frozen V-JEPA 2 ViT-L encoder with
domain-conditioned causal predictors, a continuous SE(3) action-conditioned
branch, and an audited physical-grounding evaluation protocol across 19 local
endoscopic datasets (1,707 sequences, 1.07M decoded frames).

## Repository layout

```
src/endoworld/          # the endoworld package
  world/                # H-JEPA, continuous SE(3) dynamics, MPC, risk, cache builders
  eval/                 # forecast / probe / navigation / pose-gate evaluation
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
python -m pytest tests/test_endohjepa_units.py -q
```

Key entry points:

| Result | Command |
|---|---|
| Latent forecast (6k causal L1) | `python -m endoworld.world.train --encoder vjepa2 --ablation l1 --l1-causal --max-clips 6000` |
| Per-dataset decomposition | `python -m endoworld.eval.per_dataset --ckpt outputs/scale_6000_causal/endohjepa.pt` |
| Physical cache (v2, mono + past-only) | `python -m endoworld.world.build_physical_actions --encoder vjepa2 --stereo-eye top --past-only` |
| Continuous SE(3) dynamics | `python -m endoworld.world.train_continuous_actions --data outputs/physical_actions_v2/sequences.pt --negatives local --skip-test` |
| Grouped CV (development protocol) | `python -m endoworld.world.train_grouped_cv --data outputs/physical_actions_v2/sequences.pt --negatives local` |
| Offline planning proxy (single-shot oracle-goal latent retrieval) | `python -m endoworld.eval.physical_navigation --data outputs/physical_actions_v2/sequences.pt --checkpoint outputs/continuous_actions_v2/continuous_dynamics.pt --trials 200` |
| C3VD depth-warp diagnostic | `python -m endoworld.eval.c3vd_pose_gate` |
| CholecT50 dense MIL probe | `python -m endoworld.eval.cholect50_dense_probe` |
| Paper figures | `python docs/endohjepa/make_figures.py` (reads only `verified_metrics.json`) |

Datasets are not redistributed; see `docs/endohjepa/c3vd_download_links.json` for
the C3VD/C3VDv2 download recipe and `manifests/` for the expected layout. Private
ION bronchoscopy data cannot be shared; all ION-derived numbers are marked in the
paper.

## Headline results (from `verified_metrics.json`)

- Latent forecast, horizon 4, video-level held-out: **0.978 cosine** vs 0.916
  persistence, 0.974 GRU, 0.971 Mamba (Holm-corrected Wilcoxon p < 1e-80).
- Continuous SE(3) action sensitivity (v2 corrected pipeline): **83.1%**
  real-action win on the frozen SCARED test case (grouped-CV macro 88.9%),
  passing the 80% gate; external C3VD ten-trajectory pooled 58.3% (above
  chance, below gate).
- Calibrated risk screen: **fails its gate** (AUC 0.523 vs 0.75 target) and is
  reported as a negative result; the risk head is inactive in the proxy.
- Offline planning proxy (single-shot oracle-goal latent retrieval, 200
  trials): one-step reach 51.5% (descriptive only; no receding-horizon control
  or robot execution is claimed).
- CholecT50 official 5-fold CV: frozen V-JEPA 2 dense MIL probe 0.719 phase /
  0.586 instrument mAP.

## License

Code: MIT (see LICENSE). Datasets remain under their original licences; C3VD is
CC BY-NC-SA 4.0.
