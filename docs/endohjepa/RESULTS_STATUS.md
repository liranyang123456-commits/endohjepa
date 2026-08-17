# Endo-HJEPA result status (not a submission table)

All numbers below are **protocol / smoke-scale** on official V-JEPA 2 ViT-L unless marked debug. Do not paste them into a MedIA/MICCAI main table.

## Main track (`outputs/endohjepa_vjepa2/`)

| Item | Setting | Observation |
| --- | --- | --- |
| L1 adapt | 48 clips, 4 epochs, frozen ViT-L | loss 1.58 → 1.43; cache `(48, 8, 256, 1024)` |
| H-JEPA full | same cache, 6 epochs | val cos 0.781 vs persist **0.948** |
| H-JEPA L1 | same cache | val cos 0.816 vs persist 0.951 |
| GRU | same cache | val cos 0.812 vs persist 0.918 |
| Latent MPC | cached latents | plan better than persist **14.6%**; energy reject 0.40 |
| EndoVis presence | video-level, 80 clips / 6 seq | mAP **0.314** (not 0.992) |
| EndoVis token motion | 24 clips | inst/bg MSE ratio **0.97** |
| STIR chamfer | 24 seq, frozen tokens | mean **178.7** (unnormalised token space) |
| SCARED pose↔latent | 8 keyframes, 49 residuals, k=12 | NMI **0.41–0.47** vs random ~0.40; residual→6D \(R^2\) negative |
| C3VD pose↔latent | `cecum_t1_a`, 56 residuals | NMI **0.40** |

Persistence still wins at this scale. That is expected: 48 clips × 6 epochs is not a paper run.

## Debug only (do not cite as main)

- Scratch 9M V-JEPA: `outputs/vjepa_l1/`, `outputs/endohjepa/` (including any 0.996 cosine).
- CholecSeg8k clip-leaky mAP 0.992.
- EndoVis scratch probe mAP 0.546 (`outputs/endohjepa/instrument_probe.json`).

## Blocked on access (code ready)

- Full Cholec80: CAMMA form → `python -m endoworld.data.cholec80 --src <dir>`.
- Other C3VD trajectories: Google Drive quota. Local RGB+pose: `cecum_t1_a` only (`{i}_color.png` + `pose.txt`).
