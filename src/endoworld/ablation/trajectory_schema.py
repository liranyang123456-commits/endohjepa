"""Trajectory data schema for ablation path learning.

A single case is represented as
    (device θ,  M_pre,  {(p_i, T_i / P_i, t_i)}_i,  M_post,  outcome)

This module defines typed dataclasses, JSON/NPZ serialisation, and converters
from the existing optimiser ``AblationPlan`` into the trajectory format so that
clinical logs, simulated rollouts and optimiser demos share one schema.

Schema version: 1
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np

SCHEMA_VERSION = 1

SourceKind = Literal["clinical", "simulated", "optimiser", "augmented"]
ApproachKind = Literal["percutaneous", "transbronchial", "unknown"]
VerdictKind = Literal[
    "complete_ablation", "incomplete", "indeterminate", "unknown", "simulated"
]


# --------------------------------------------------------------------------- #
# Device parameters θ
# --------------------------------------------------------------------------- #
@dataclass
class DeviceParams:
    """Applicator / generator parameters that condition the zone model."""

    device_type: str = "MWA"           # MWA | RFA | CRYO
    probe_diameter_mm: float = 1.8     # e.g. 14G ≈ 2.1 mm, 17G ≈ 1.5 mm
    tip_length_mm: float = 15.0        # active antenna length
    lam_mm: float = 4.0                # deposition decay length
    power_presets_W: tuple = (30, 45, 60, 80)
    time_presets_s: tuple = (180, 300, 420, 600)
    temp_cap_C: float = 105.0          # vaporisation plateau
    cem43_thresh: float = 240.0
    max_power_W: float = 100.0
    min_power_W: float = 20.0
    note: str = ""

    def clamp_power(self, p: float) -> float:
        return float(np.clip(p, self.min_power_W, self.max_power_W))

    def nearest_preset(self, power_W: float, time_s: float) -> tuple[float, float]:
        pw = min(self.power_presets_W, key=lambda x: abs(x - power_W))
        ts = min(self.time_presets_s, key=lambda x: abs(x - time_s))
        return float(pw), float(ts)

    @classmethod
    def from_device_name(cls, name: str = "MWA") -> "DeviceParams":
        from endoworld.ablation.planner import DEVICES
        d = DEVICES.get(name, DEVICES["MWA"])
        return cls(
            device_type=name,
            power_presets_W=tuple(sorted({
                d["pmin"], int(d["power_W"]), d["pmax"],
                (d["pmin"] + int(d["power_W"])) // 2,
            })),
            time_presets_s=tuple(d["times_s"]),
            min_power_W=float(d["pmin"]),
            max_power_W=float(d["pmax"]),
        )


# --------------------------------------------------------------------------- #
# Single burn step: (position, temperature/power, timing)
# --------------------------------------------------------------------------- #
@dataclass
class BurnStep:
    """One sequential ablation activation."""

    step_index: int
    position_mm: tuple                 # (x, y, z) in tumour-local frame
    power_W: float
    time_s: float
    temperature_C: float | None = None  # measured tip/tissue temp if available
    zone_axes_mm: tuple | None = None   # (rt, rt, ra); filled by simulator
    approach: ApproachKind = "unknown"
    note: str = ""

    def as_action_dict(self) -> dict:
        return {
            "position_mm": list(self.position_mm),
            "power_W": float(self.power_W),
            "time_s": float(self.time_s),
            "temperature_C": self.temperature_C,
        }


# --------------------------------------------------------------------------- #
# Lesion / anatomy geometry (+ optional dense masks)
# --------------------------------------------------------------------------- #
@dataclass
class LesionGeometry:
    """Patient-specific lesion description used as M_pre / planning input."""

    case_id: str
    tumor_axes_mm: tuple               # semi-axes (a, b, c)
    margin_mm: float = 5.0
    spacing_mm: float = 1.5
    lobe: str = ""
    bronchial_segment: str = ""
    airway_generation: float | None = None
    dist_pleura_mm: float | None = None
    dist_chestwall_mm: float | None = None
    dist_vessel_mm: float | None = None
    solidity: str = ""
    malignancy_pct: float | None = None
    # Optional dense mask references (relative paths inside a case folder)
    pre_mask_file: str | None = None   # .npz bool volume, tumour(+margin) frame
    post_mask_file: str | None = None
    lung_mask_file: str | None = None
    vessel_mask_file: str | None = None
    frame: str = "tumour_local"        # coordinate frame of positions / masks

    @property
    def tumor_volume_mL(self) -> float:
        a, b, c = self.tumor_axes_mm
        return float(4.0 / 3.0 * np.pi * a * b * c / 1000.0)


@dataclass
class OutcomeLabel:
    """Post-operative / follow-up quality label used as learning signal."""

    verdict: VerdictKind = "unknown"
    pre_volume_mL: float | None = None
    peak_volume_mL: float | None = None
    late_volume_mL: float | None = None
    followup_days: list = field(default_factory=list)
    followup_volumes_mL: list = field(default_factory=list)
    preference_score: float | None = None  # higher = better (for preference learning)
    note: str = ""


# --------------------------------------------------------------------------- #
# Full trajectory record
# --------------------------------------------------------------------------- #
@dataclass
class AblationTrajectory:
    """Complete case record for imitation / RL / preference learning."""

    case_id: str
    device: DeviceParams
    geometry: LesionGeometry
    steps: list = field(default_factory=list)          # list[BurnStep]
    outcome: OutcomeLabel = field(default_factory=OutcomeLabel)
    source: SourceKind = "simulated"
    schema_version: int = SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metrics: dict = field(default_factory=dict)        # coverage, overtreat, ...
    meta: dict = field(default_factory=dict)

    # ---- convenience ------------------------------------------------------ #
    def n_burns(self) -> int:
        return len(self.steps)

    def total_time_s(self) -> float:
        return float(sum(s.time_s for s in self.steps))

    def total_energy_kJ(self) -> float:
        return float(sum(s.power_W * s.time_s for s in self.steps) / 1000.0)

    def action_sequence(self) -> list[dict]:
        return [s.as_action_dict() for s in self.steps]


# --------------------------------------------------------------------------- #
# Mask I/O (NPZ)
# --------------------------------------------------------------------------- #
def save_mask(path: str, mask: np.ndarray, spacing_mm: float = 1.5,
              origin_mm: tuple = (0.0, 0.0, 0.0), label: str = "mask") -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        mask=np.asarray(mask, dtype=bool),
        spacing_mm=np.asarray(spacing_mm, dtype=np.float32),
        origin_mm=np.asarray(origin_mm, dtype=np.float32),
        label=np.asarray(label),
    )
    return path


def load_mask(path: str) -> tuple[np.ndarray, float, tuple]:
    z = np.load(path, allow_pickle=True)
    mask = z["mask"].astype(bool)
    spacing = float(z["spacing_mm"]) if "spacing_mm" in z.files else 1.5
    origin = tuple(z["origin_mm"].tolist()) if "origin_mm" in z.files else (0.0, 0.0, 0.0)
    return mask, spacing, origin


# --------------------------------------------------------------------------- #
# JSON serialisation (masks stored as file refs, not inline)
# --------------------------------------------------------------------------- #
def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, AblationTrajectory):
        d = asdict(obj)
        return d
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def trajectory_to_dict(traj: AblationTrajectory) -> dict:
    return _to_jsonable(traj)


def dict_to_trajectory(d: dict) -> AblationTrajectory:
    raw_dev = d["device"]
    dev_kwargs = {}
    for k in DeviceParams.__dataclass_fields__:
        if k not in raw_dev:
            continue
        v = raw_dev[k]
        if k in ("power_presets_W", "time_presets_s") and isinstance(v, (list, tuple)):
            dev_kwargs[k] = tuple(v)
        else:
            dev_kwargs[k] = v
    device = DeviceParams(**dev_kwargs)

    g = d["geometry"]
    geometry = LesionGeometry(
        case_id=g["case_id"],
        tumor_axes_mm=tuple(g["tumor_axes_mm"]),
        margin_mm=float(g.get("margin_mm", 5.0)),
        spacing_mm=float(g.get("spacing_mm", 1.5)),
        lobe=g.get("lobe", ""),
        bronchial_segment=g.get("bronchial_segment", ""),
        airway_generation=g.get("airway_generation"),
        dist_pleura_mm=g.get("dist_pleura_mm"),
        dist_chestwall_mm=g.get("dist_chestwall_mm"),
        dist_vessel_mm=g.get("dist_vessel_mm"),
        solidity=g.get("solidity", ""),
        malignancy_pct=g.get("malignancy_pct"),
        pre_mask_file=g.get("pre_mask_file"),
        post_mask_file=g.get("post_mask_file"),
        lung_mask_file=g.get("lung_mask_file"),
        vessel_mask_file=g.get("vessel_mask_file"),
        frame=g.get("frame", "tumour_local"),
    )
    steps = []
    for s in d.get("steps", []):
        steps.append(BurnStep(
            step_index=int(s["step_index"]),
            position_mm=tuple(s["position_mm"]),
            power_W=float(s["power_W"]),
            time_s=float(s["time_s"]),
            temperature_C=s.get("temperature_C"),
            zone_axes_mm=tuple(s["zone_axes_mm"]) if s.get("zone_axes_mm") else None,
            approach=s.get("approach", "unknown"),
            note=s.get("note", ""),
        ))
    oc = d.get("outcome") or {}
    outcome = OutcomeLabel(
        verdict=oc.get("verdict", "unknown"),
        pre_volume_mL=oc.get("pre_volume_mL"),
        peak_volume_mL=oc.get("peak_volume_mL"),
        late_volume_mL=oc.get("late_volume_mL"),
        followup_days=list(oc.get("followup_days") or []),
        followup_volumes_mL=list(oc.get("followup_volumes_mL") or []),
        preference_score=oc.get("preference_score"),
        note=oc.get("note", ""),
    )
    return AblationTrajectory(
        case_id=d["case_id"],
        device=device,
        geometry=geometry,
        steps=steps,
        outcome=outcome,
        source=d.get("source", "simulated"),
        schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
        created_at=d.get("created_at", datetime.now(timezone.utc).isoformat()),
        metrics=dict(d.get("metrics") or {}),
        meta=dict(d.get("meta") or {}),
    )


def save_trajectory(traj: AblationTrajectory, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trajectory_to_dict(traj), f, indent=2, ensure_ascii=False)
    return path


def load_trajectory(path: str) -> AblationTrajectory:
    with open(path, encoding="utf-8") as f:
        return dict_to_trajectory(json.load(f))


def schema_example_dict() -> dict:
    """Human-readable example used in docs / CLI --print-schema."""
    ex = AblationTrajectory(
        case_id="demo_001",
        device=DeviceParams.from_device_name("MWA"),
        geometry=LesionGeometry(
            case_id="demo_001",
            tumor_axes_mm=(8.0, 7.0, 9.0),
            margin_mm=5.0,
            spacing_mm=1.5,
            lobe="LUL",
            airway_generation=6,
            dist_pleura_mm=12.0,
            pre_mask_file="masks/pre.npz",
            post_mask_file="masks/post.npz",
        ),
        steps=[
            BurnStep(1, (0.0, 0.0, 0.0), 45.0, 420.0, temperature_C=95.0,
                     zone_axes_mm=(11.0, 11.0, 14.3), approach="percutaneous"),
        ],
        outcome=OutcomeLabel(
            verdict="simulated",
            pre_volume_mL=2.1,
            peak_volume_mL=12.0,
            late_volume_mL=3.5,
            followup_days=[0, 30, 180],
            followup_volumes_mL=[2.1, 12.0, 3.5],
            preference_score=1.0,
        ),
        source="simulated",
        metrics={"tumor_coverage": 1.0, "target_coverage_incl_margin": 0.995},
        meta={"comment": "example trajectory record"},
    )
    return trajectory_to_dict(ex)


# --------------------------------------------------------------------------- #
# Converters: optimiser plan → trajectory
# --------------------------------------------------------------------------- #
def plan_to_trajectory(plan, case_id: str = "optimiser",
                       geometry: LesionGeometry | None = None,
                       device: DeviceParams | None = None) -> AblationTrajectory:
    """Convert an ``AblationPlan`` from ``planner.plan_ablation`` into schema."""
    device = device or DeviceParams.from_device_name(getattr(plan, "device", "MWA"))
    if geometry is None:
        geometry = LesionGeometry(
            case_id=case_id,
            tumor_axes_mm=tuple(plan.tumor_axes_mm),
            margin_mm=float(plan.margin_mm),
        )
    approach = "unknown"
    if plan.trajectory is not None:
        approach = plan.trajectory.approach  # type: ignore[assignment]
    steps = []
    for i, bn in enumerate(plan.burns):
        steps.append(BurnStep(
            step_index=i + 1,
            position_mm=tuple(bn.center_mm),
            power_W=float(bn.power_W),
            time_s=float(bn.time_s),
            zone_axes_mm=tuple(bn.zone_axes_mm) if bn.zone_axes_mm else None,
            approach=approach,
        ))
    return AblationTrajectory(
        case_id=case_id,
        device=device,
        geometry=geometry,
        steps=steps,
        outcome=OutcomeLabel(verdict="simulated", note="from optimiser"),
        source="optimiser",
        metrics=dict(plan.metrics or {}),
        meta={"approach_note": getattr(plan.trajectory, "note", "") if plan.trajectory else ""},
    )


def geometry_from_record_row(row: dict, case_id: str | None = None,
                             margin_mm: float = 5.0,
                             spacing_mm: float = 1.5) -> LesionGeometry | None:
    """Build LesionGeometry from a nodule_params.csv row."""
    def _f(k, *alts):
        for key in (k, *alts):
            v = row.get(key)
            if v is None or v == "":
                continue
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
        return None

    ap = _f("size_AP_mm", "diam_coronal_mm") or 0.0
    si = _f("size_SI_mm", "diam_sagittal_mm") or 0.0
    lr = _f("size_LR_mm", "diam_axial_mm") or 0.0
    if min(ap, si, lr) <= 0:
        return None
    cid = case_id or str(
        row.get("case_id") or row.get("id") or row.get("note") or "unknown"
    )
    # strip file extensions from note-based ids
    if cid.endswith(".txt"):
        cid = cid[: -len(".txt")]
    if cid.endswith(".docx"):
        cid = cid[: -len(".docx")]
    return LesionGeometry(
        case_id=cid,
        tumor_axes_mm=(lr / 2, ap / 2, si / 2),
        margin_mm=margin_mm,
        spacing_mm=spacing_mm,
        lobe=str(row.get("lobe") or ""),
        bronchial_segment=str(row.get("bronchial_segment") or ""),
        airway_generation=_f("airway_generation"),
        dist_pleura_mm=_f("dist_pleura_mm"),
        dist_chestwall_mm=_f("dist_chestwall_mm"),
        dist_vessel_mm=_f("dist_vessel_mm"),
        solidity=str(row.get("solidity") or ""),
        malignancy_pct=_f("malignancy_pct"),
    )


if __name__ == "__main__":
    print(json.dumps(schema_example_dict(), indent=2, ensure_ascii=False))
