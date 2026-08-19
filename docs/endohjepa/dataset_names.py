"""Display names for the corpora used in the manuscript.

Manifest keys are storage identifiers (snake case, internal suffixes). Figures
and tables must show the published corpus name instead, so every figure script
routes its labels through :func:`display`.
"""
from __future__ import annotations

DISPLAY_NAMES = {
    "C3VD": "C3VD",
    "Cholec80-Boxes": "Cholec80",
    "CholecSeg8k": "CholecSeg8k",
    "CholecT50": "CholecT50",
    "EndoNeRF": "EndoNeRF",
    "EndoVis2019_ROBUST-MIS": "ROBUST-MIS 2019",
    "EndoVis_InstrumentTracking": "EndoVis instrument tracking",
    "HyperKvasir": "HyperKvasir",
    "ION_bronch": "ION bronchoscopy",
    "Kvasir-Capsule": "Kvasir-Capsule",
    "Kvasir-Instrument": "Kvasir-Instrument",
    "MIS_own": "In-house MIS",
    "SCARED": "SCARED",
    "STIR": "STIR",
    "Stereo_Lap": "Stereo laparoscopy",
    "SurgT": "SurgT",
    "TrackVes": "TrackVes",
    "endoscapes": "Endoscapes",
    "endovis2017_full": "EndoVis 2017",
    "endovis2018_full": "EndoVis 2018",
}


def display(key: str) -> str:
    """Published corpus name for a manifest key, falling back to the key."""
    return DISPLAY_NAMES.get(key, key)
