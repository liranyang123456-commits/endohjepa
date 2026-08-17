"""Domain tags for the unified endoscopic world model.

Domains:
  laparo   laparoscopic / rigid MIS
  gi       flexible GI endoscopy (gastro/colon) including capsule
  bronch   bronchoscopy / ION navigation
  mixed    unknown or mixed-orifice
"""
from __future__ import annotations

from pathlib import Path

# Canonical dataset folder name -> domain
DATASET_DOMAIN: dict[str, str] = {
    "Cholec80-Boxes": "laparo",
    "Cholec80": "laparo",
    "CholecSeg8k": "laparo",
    "endoscapes": "laparo",
    "endovis2017_full": "laparo",
    "endovis2018_full": "laparo",
    "EndoVis2017": "laparo",
    "EndoVis2019_ROBUST-MIS": "laparo",
    "EndoNeRF": "laparo",
    "SCARED": "laparo",
    "Stereo_Lap": "laparo",
    "STIR": "laparo",
    "SurgT": "laparo",
    "SurgT_SoftTissueTracking": "laparo",
    "TrackVes": "laparo",
    "EndoVis_InstrumentTracking": "laparo",
    "MIS_own": "laparo",
    "Kvasir-Instrument": "gi",
    "HyperKvasir": "gi",
    "Kvasir-Capsule": "gi",
    "C3VD": "gi",
    "ION_bronch": "bronch",
}

DOMAIN_IDS = {"laparo": 0, "gi": 1, "bronch": 2, "mixed": 3}
ID_TO_DOMAIN = {v: k for k, v in DOMAIN_IDS.items()}


def infer_domain(dataset: str, path: str = "") -> str:
    if dataset in DATASET_DOMAIN:
        return DATASET_DOMAIN[dataset]
    blob = f"{dataset} {path}".lower().replace("\\", "/")
    if any(k in blob for k in ("bronch", "ion_", "ion-", "/ion/")):
        return "bronch"
    if any(k in blob for k in ("kvasir", "hyperkvasir", "capsule", "c3vd", "colon", "gastro")):
        return "gi"
    if any(k in blob for k in ("cholec", "endovis", "scared", "lapar", "endoscapes", "endonerf", "stir", "surgt", "trackves")):
        return "laparo"
    return "mixed"


def domain_id(dataset: str, path: str = "") -> int:
    return DOMAIN_IDS[infer_domain(dataset, path)]


def extra_local_roots() -> list[tuple[str, Path]]:
    """Named extra corpora that live outside datasets/ but are already on disk.

    Returns (dataset_name, root_path) pairs. Missing paths are skipped by the scanner.
    """
    return [
        ("TrackVes", Path(r"E:\Surgical_Tracking_Datasets\TrackVes\TrackVes\TrackVes")),
        ("EndoVis_InstrumentTracking", Path(r"E:\Surgical_Tracking_Datasets\EndoVis_InstrumentTracking")),
        ("SurgT", Path(r"E:\Surgical_Tracking_Datasets\Benchmarks\SurgT_benchmarking\data")),
        ("MIS_own", Path(r"D:\Exp_MIS_ChessBox_Datas\MIS_Videso_1_6")),
        ("MIS_own", Path(r"F:\Exp_Videos")),
    ]
