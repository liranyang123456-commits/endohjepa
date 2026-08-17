# Data access (Endo-HJEPA)

## Already ingested locally

HyperKvasir, Kvasir-Capsule (labeled + unlabeled), Kvasir-Instrument, C3VD `cecum_t1_a`,
SCARED (datasets 1–3, 5–7), EndoVis 2017/2018, CholecSeg8k, Cholec80-Boxes (videos 41–45),
STIR Challenge 2024, ION bronchoscopy (anonymized `case_XXX` only).

## Cholec80 full (not in this repo)

Official distribution: CAMMA request form  
https://camma.u-strasbg.fr/datasets

After approval, place `video01.mp4` … `video80.mp4` (or the official folder) somewhere local and run:

```bash
python -m endoworld.data.cholec80 --src <official_dir>
python -m endoworld.data.refresh_manifest --datasets Cholec80 --root datasets
```

This repository does not download Cholec80 from unofficial mirrors.

## C3VD remaining trajectories

```bash
python scripts/09_download_public.py --set c3vd --skip-screening
```

Google Drive quota may block extra zips; retry later. Pose files (`pose.txt`) pin L3 actions.

Retry 2026-08-14: all remaining *trajectory* zips (`cecum_t1_b` … `trans_t4_b`) still returned Drive “Too many users…”. Screening/mold zips (`cecum_mold`, `desc_mold`, `sigmoid_mold`, `trans_mold`) downloaded; they are geometry assets, not world-model sequences. Local `cecum_t1_a` has `pose.txt`, `{i}_color.png`, depth/normals/occlusion (276 RGB frames). Extra trajectories still blocked.

## ION

Private. Only numeric `datasets/ION_bronch/case_XXX/` is used. No patient names in manifests.
Requires an ethics approval number before any submission that discusses these cases.
In-silico navigation only; not a claim of replacing clinical bronchoscopy.
