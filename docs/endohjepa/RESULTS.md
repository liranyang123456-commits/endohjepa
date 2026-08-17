# Endo-HJEPA consolidated results

> Human-readable summary only. Numerical source of truth:
> `verified_metrics.json`. Isolated from the CT ablation track.

## Data
- 1,707 sequences / 1,067,734 decoded frames; GI: 416,262,
  laparoscopy: 488,029, bronchoscopy: 163,443

## Forecast (video-level val)
| Model | cos | MSE |
| --- | ---: | ---: |
| Persistence | 0.916 | — |
| Query-token L1 | 0.936 | — |
| Mamba/SSM | 0.971 | — |
| GRU | 0.974 | — |
| **Causal L1 (H-JEPA)** | **0.978** | **0.139** |

Statistical significance vs GRU (Wilcoxon, Holm-corrected):

| Horizon | H-JEPA | GRU | Holm p | sig |
| --- | ---: | ---: | ---: | :-: |
| h=1 | 0.9811 | 0.9777 | 1.24e-84 | yes |
| h=4 | 0.9776 | 0.974 | 4.06e-102 | yes |

## Planning (H-JEPA only)
- Plan beats persistence on **0.98** of clips (cos 0.920 vs 0.874), but causal
  L1 is higher at 0.970.
- Cumulative path energy is lower than random trajectories on 0.928 of clips.

## Cross-domain (zero-shot)
| Domain | model | persist |
| --- | ---: | ---: |
| laparo | 0.922 | 0.85 |
| gi | 0.791 | 0.913 |
| bronch | 0.776 | 0.893 |

## Few-shot domain adaptation (partial recovery; still below persistence)
| Target | zero-shot | few-shot | persistence | recovery |
| --- | ---: | ---: | ---: | ---: |
| gi (eval n=10) | 0.793 | 0.839 | 0.911 | +0.045 |
| bronch (eval n=9) | 0.781 | 0.83 | 0.894 | +0.049 |

## Action grounding (honest: weak without supervision)
- Physical SCARED pose NMI: [0.41, 0.52], random [0.49, 0.53];
  latent exceeds random on 2/6 keyframes.
- Semantic (CholecT50 verb) NMI: 0.044 (random 0.056); probe acc 0.306 (chance 0.39)

## Downstream recognition (official five-fold CV; primary)
| Task | frozen | adapted |
| --- | ---: | ---: |
| CholecT50 phase | **0.683 ± 0.040** | 0.679 ± 0.038 |
| CholecT50 instrument mAP | **0.531 ± 0.046** | 0.489 ± 0.048 |

Standard deviation is across five fold means; every fold averages three probe
seeds. Adaptation improves neither task.

## Downstream recognition (3-seed challenge-test probe; secondary)
| Task | frozen | adapted |
| --- | ---: | ---: |
| CholecT50 phase | **0.704 ± 0.006** | 0.688 ± 0.057 |
| CholecT50 instrument mAP | 0.406 ± 0.002 | **0.485 ± 0.005** |

The apparent challenge-split instrument gain does not survive official CV.

## External baselines (CholecT50 challenge-test split, video-level linear probe)
| Encoder | phase acc | instrument mAP |
| --- | ---: | ---: |
| vjepa2-frozen | 0.704 | 0.406 |
| vjepa2-adapted | 0.688 | 0.485 |
| imagenet | 0.592 | 0.43 |
| videomae | 0.558 | 0.497 |
| dinov2 | 0.658 | 0.575 |
| timesformer | 0.683 | 0.49 |
| vivit | 0.692 | 0.443 |

## Per-dataset forecast (6k causal L1, 750-clip val, aligned)

| Dataset | Domain | n | cos | persist | Δ |
| --- | --- | ---: | ---: | ---: | ---: |
| ION_bronch | bronch | 250 | 0.988 | 0.918 | +0.071 |
| Kvasir-Capsule | gi | 206 | 0.978 | 0.941 | +0.037 |
| CholecT50 | laparo | 89 | 0.963 | 0.894 | +0.069 |
| Stereo_Lap | laparo | 75 | 0.982 | 0.889 | +0.093 |
| endoscapes | laparo | 72 | 0.961 | 0.893 | +0.068 |
| HyperKvasir | gi | 44 | 0.970 | 0.924 | +0.046 |
| endovis2018_full | laparo | 7 | 0.961 | 0.882 | +0.079 |
| endovis2017_full | laparo | 5 | 0.965 | 0.909 | +0.056 |
| **Overall** | --- | **750** | **0.978** | **0.916** | **+0.062** |

Source: `outputs/scale_6000_causal/per_dataset.json`. SCARED/STIR $n{=}1$ omitted.

## Data-scale curve (causal L1, video-level val)
| # clips | cos | MSE |
| --- | ---: | ---: |
| 500 | 0.962 | 0.236 |
| 1000 | 0.968 | 0.201 |
| 2000 | 0.972 | 0.174 |
| 4000 | 0.976 | 0.15 |
| 6000 | 0.978 | 0.139 |
| 13552 | 0.978 | 0.135 |

## Energy physical grounding (SCARED wall-proximity)
- Near-wall AUC: **0.682**, Spearman(energy, depth): **-0.471** (132 transitions)
