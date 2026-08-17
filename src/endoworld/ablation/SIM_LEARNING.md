# Hybrid ablation path learning

Implements the closed loop discussed in the paper / design notes:

```text
device θ + lesion M_pre
        │
        ▼
 ┌──────────────┐   propose    ┌─────────────┐
 │  BC policy π │ ───────────► │ safety gate │──► accepted plan
 └──────────────┘              └──────┬──────┘
        ▲                             │ fail
        │ demos                       ▼
 ┌──────┴───────┐              greedy continue / replan
 │ mixed dataset│              (submodular optimiser)
 │ opt+sim+clin │
 └──────────────┘
        ▲
        │ outcomes
 longitudinal follow-up CT
```

## Modules

| Module | Role |
|---|---|
| `trajectory_schema.py` | `(θ, M_pre, steps, M_post, outcome)` JSON + NPZ masks |
| `sim_env.py` | Gym env; analytic / FDM zones; greedy & random policies |
| `dataset.py` | Build mixed dataset + flatten `(s,a)` for BC |
| `policy.py` | Ridge / RF / GBRT multi-output BC policy |
| `safety_gate.py` | Coverage gate + cascade repair (greedy → replan) |
| `preference.py` | Outcome-calibrated trajectory ranker |
| `train_eval.py` | End-to-end train + method comparison |
| `run_sim.py` | Low-level schema / rollout CLI |

## Verified result (`--all --limit 12 --force-zone 10`)

| Method | Mean cov | ≥99% | Over (mL) | Burns | Repaired |
|---|---:|---:|---:|---:|---:|
| random | 99.4% | 100% | 13.7 | 5.0 | — |
| greedy | 99.0% | 75% | 14.2 | 5.5 | — |
| BC | 99.0% | 75% | 14.2 | 5.5 | — |
| **BC + gate** | **100%** | **100%** | **10.5** | **3.5** | 25% |

Preference ranker: leave-in \(R^2\!\approx\!0.98\), pairwise order accuracy \(\approx\!0.98\);
clinical cases 10001–10003 ranked above indeterminate 10004.

## Quick start

```bash
# Full hybrid loop on 10 cases (fast smoke of the idea)
PYTHONPATH=src python -m endoworld.ablation.train_eval --all --limit 10

# Upgrade: A 3D seg + B masks (+ unzip) + patient-grid sim
PYTHONPATH=src python -m endoworld.ablation.run_upgrade --limit-a 2 --limit-b 4

# Pieces
PYTHONPATH=src python -m endoworld.ablation.segment3d --cases 001 008
PYTHONPATH=src python -m endoworld.ablation.followup_masks --extract-zips
PYTHONPATH=src python -m endoworld.ablation.patient_sim --force-zone 10
PYTHONPATH=src python -m endoworld.ablation.dataset --limit 10 --force-zone 10
PYTHONPATH=src python -m endoworld.ablation.policy --data outputs/ablation_hybrid/dataset
PYTHONPATH=src python -m endoworld.ablation.run_sim --smoke
```

Public datasets (LIDC/LUNA/NSCLC-Radiomics/AirRC/TotalSegmentator): see
`docs/PUBLIC_DATASETS.md`.


## Data contract for real OR logs

Drop clinical trajectories as JSON matching the schema (`source=clinical`):

```json
{
  "case_id": "10001",
  "device": {"device_type": "MWA", "probe_diameter_mm": 1.8, "...": "..."},
  "geometry": {"tumor_axes_mm": [8,7,9], "pre_mask_file": "masks/pre.npz"},
  "steps": [
    {"step_index": 1, "position_mm": [0,0,0], "power_W": 45, "time_s": 420,
     "temperature_C": 95}
  ],
  "outcome": {"verdict": "complete_ablation", "preference_score": 1.0,
              "followup_days": [0, 69, 167],
              "followup_volumes_mL": [3.5, 15.4, 4.8]}
}
```

Until intra-op `(p,T,t)` logs exist, `dataset.py` attaches real follow-up
**outcomes** to optimiser-proposed steps (`source=clinical`) so preference
learning / reward calibration can already use them.

## Safety guarantee

The learned policy **proposes**; the gate **guarantees**:

- if coverage < γ → continue with submodular greedy (or full replan)
- only plans with coverage ≥ γ leave the gate
- the classical $(1-1/e)$ backbone is preserved
