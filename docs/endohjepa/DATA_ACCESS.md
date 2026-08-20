# Data access (Endo-HJEPA)

## Dataset configuration used in the study

HyperKvasir, Kvasir-Capsule (labeled + unlabeled), Kvasir-Instrument, the
fixed ten-trajectory C3VD/C3VDv2 external-evaluation manifest,
SCARED (datasets 1–3, 5–7), EndoVis 2017/2018, CholecSeg8k, Cholec80-Boxes (videos 41–45),
STIR Challenge 2024, the public Hamlyn rectified stereo corpus, private ION
bronchoscopy (anonymised `case_XXX` only), and private in-house MIS video.

Generate a local sequence manifest after configuring the data roots:

```bash
python -m endoworld.data.scan_datasets
```

## Cholec80 full (not in this repo)

Official distribution: CAMMA request form  
https://camma.u-strasbg.fr/datasets

After approval, place `video01.mp4` … `video80.mp4` (or the official folder) somewhere local and run:

```bash
python -m endoworld.data.cholec80 --src <official_dir>
python -m endoworld.data.refresh_manifest --datasets Cholec80 --root datasets
```

This repository does not download Cholec80 from unofficial mirrors.

## C3VD external-evaluation manifest

```bash
python -m endoworld.data.download_public --set c3vd --skip-screening
```

Google Drive quota may block extra zips. Pose files (`pose.txt`) pin the
continuous SE(3) actions.

The archived external result (`n=798`) uses the fixed manifest expressed in the
implementation's OpenGL-to-OpenCV pose convention:
`cecum_t1_a`, `c1_ascending_t4_v4`, `c1_cecum_t1_v4`,
`c1_descending_t4_v4`, `c1_sigmoid1_t4_v4`, `c1_sigmoid2_t4_v4`,
`c1_transverse1_t1_v4`, `c1_transverse1_t4_v4`, `c2_cecum_t1_v4`, and
`c2_transverse1_t1_v4`. These are the only C3VD trajectories claimed as
screened in the archived export. It predates the final negative evaluator and
is not used as external-generalisation evidence. Additional Drive-registered trajectories remain
unavailable under the host download limits; screening/mold zips are geometry
assets, not world-model sequences.

## Private clinical data

ION and in-house MIS video require institutional authorisation. Only numeric
`datasets/ION_bronch/case_XXX/` identifiers are used for ION, and no patient
names are included in manifests or the public repository. The reported study
uses video only; associated CT volumes are not model inputs. The CEM analysis
is an offline retrieval diagnostic on public SCARED data, not a claim of
replacing clinical bronchoscopy.
The available ION export has no patient grouping key. Its hash split therefore
guarantees sequence-level, but not independently verified patient-level,
non-overlap.
