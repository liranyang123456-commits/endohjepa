"""CholecT50 loader: phase recognition + instrument-presence + action triplets.

CholecT50 JSON annotation record (per detection, 15 fields); empirically decoded:
  [0] triplet_id (0-99)  [1] instrument_id (0-5)  [7] verb_id (0-9)
  [8] target_id (0-14)   [14] phase_id (0-6)
A frame has 1-3 records. Phase is constant across a frame's records.

    python -m endoworld.data.cholect50 --status
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "datasets" / "CholecT50" / "CholecT50"

N_PHASE = 7
N_INSTR = 6
N_VERB = 10
N_TARGET = 15
PHASE_NAMES = ["preparation", "calot_triangle_dissection", "clipping_cutting",
               "gallbladder_dissection", "gallbladder_packaging",
               "cleaning_coagulation", "gallbladder_retraction"]
INSTR_NAMES = ["grasper", "bipolar", "hook", "scissors", "clipper", "irrigator"]
VERB_NAMES = ["grasp", "retract", "dissect", "coagulate", "clip", "cut",
              "aspirate", "irrigate", "pack", "null_verb"]
TARGET_NAMES = ["gallbladder", "cystic_plate", "cystic_duct", "cystic_artery",
                "cystic_pedicle", "blood_vessel", "fluid", "abdominal_wall_cavity",
                "liver", "adhesion", "omentum", "peritoneum", "gut",
                "specimen_bag", "null_target"]


# Official CholecT50 challenge held-out (test) videos — the designated val set.
OFFICIAL_TEST_VIDS = ["VID68", "VID70", "VID73", "VID74", "VID75"]


def official_split(vids: list[str]):
    """Train on the 45 non-challenge videos, test on the 5 official held-out videos."""
    test = set(OFFICIAL_TEST_VIDS)
    tr = [i for i, v in enumerate(vids) if v not in test]
    te = [i for i, v in enumerate(vids) if v in test]
    return tr, te


# Official CholecTriplet-challenge validation videos = designated held-out test set.
CHALLENGE_TEST_VIDS = ["VID68", "VID70", "VID73", "VID74", "VID75"]

# Recommended research split from Nwoye and Padoy (2022), Table 4.
CHOLECT50_CV_FOLDS = {
    1: ["VID79", "VID02", "VID51", "VID06", "VID25", "VID14", "VID66", "VID23", "VID50", "VID111"],
    2: ["VID80", "VID32", "VID05", "VID15", "VID40", "VID47", "VID26", "VID48", "VID70", "VID96"],
    3: ["VID31", "VID57", "VID36", "VID18", "VID52", "VID68", "VID10", "VID08", "VID73", "VID103"],
    4: ["VID42", "VID29", "VID60", "VID27", "VID65", "VID75", "VID22", "VID49", "VID12", "VID110"],
    5: ["VID78", "VID43", "VID62", "VID35", "VID74", "VID01", "VID56", "VID04", "VID13", "VID92"],
}


def official_test_vids() -> list[str]:
    return list(CHALLENGE_TEST_VIDS)


def split_official(vids: list[str]) -> tuple[list[str], list[str]]:
    """(train, test) using the official challenge validation videos as the test set."""
    test = [v for v in vids if v in CHALLENGE_TEST_VIDS]
    train = [v for v in vids if v not in CHALLENGE_TEST_VIDS]
    return train, test


def list_videos(labels_dir: Path | None = None) -> list[str]:
    ld = labels_dir or (ROOT / "labels")
    return sorted(p.stem for p in ld.glob("VID*.json"))


def load_video_labels(vid: str, labels_dir: Path | None = None):
    """Return dict frame_idx -> dict with phase, instr/verb/target multi-hot, triplet ids."""
    ld = labels_dir or (ROOT / "labels")
    d = json.loads((ld / f"{vid}.json").read_text(encoding="utf-8"))
    out = {}
    for fr, records in d["annotations"].items():
        phase = -1
        instr = np.zeros(N_INSTR, dtype=np.float32)
        verb = np.zeros(N_VERB, dtype=np.float32)
        target = np.zeros(N_TARGET, dtype=np.float32)
        trips = []
        for r in records:
            if len(r) < 15:
                continue
            if r[14] >= 0:
                phase = int(r[14])
            if 0 <= r[1] < N_INSTR:
                instr[int(r[1])] = 1.0
            if 0 <= r[7] < N_VERB:
                verb[int(r[7])] = 1.0
            if 0 <= r[8] < N_TARGET:
                target[int(r[8])] = 1.0
            if r[0] >= 0:
                trips.append(int(r[0]))
        out[int(fr)] = {"phase": phase, "instr": instr, "verb": verb,
                        "target": target, "triplets": trips}
    return out


def video_frames_dir(vid: str, videos_dir: Path | None = None) -> Path:
    vd = videos_dir or (ROOT / "videos")
    return vd / vid


def status() -> dict:
    vids = list_videos()
    vdir = ROOT / "videos"
    n_frames = {v: len(list((vdir / v).glob("*.png"))) for v in vids if (vdir / v).is_dir()}
    return {
        "root": str(ROOT),
        "n_videos_labeled": len(vids),
        "videos_with_frames": len(n_frames),
        "total_frames": int(sum(n_frames.values())),
        "phases": PHASE_NAMES,
        "instruments": INSTR_NAMES,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    print(json.dumps(status(), indent=2))


if __name__ == "__main__":
    main()
