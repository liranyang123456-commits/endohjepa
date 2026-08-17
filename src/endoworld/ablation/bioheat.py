"""Pennes bioheat simulation of a thermal-ablation zone + fast analytic surrogate.

Pennes bioheat equation (per unit volume):
    rho*c dT/dt = k*lap(T) + w_b*rho_b*c_b*(T_a - T) + Q(x)
where Q is the power deposited by the applicator (RF/microwave), modelled here as a
radially-decaying source around the antenna tip. We integrate explicitly on a 3D grid,
accumulate CEM43 thermal dose, and threshold it to obtain the coagulation (ablation)
zone. A fast ellipsoid surrogate calibrated from these simulations lets the planner
place many candidate burns without re-running the PDE each time.

References: Pennes 1948; CEM43 thermal dose (Sapareto & Dewey 1984). Tissue values are
typical lung/tumour ranges from the ablation-simulation literature (see docs/LITERATURE.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Tissue:
    rho: float = 1050.0      # kg/m^3
    c: float = 3600.0        # J/(kg K)
    k: float = 0.51          # W/(m K)
    w_b: float = 0.02        # effective perfusion 1/s (aerated lung heat loss; bounds zone)
    rho_b: float = 1050.0
    c_b: float = 3600.0
    T_a: float = 37.0        # arterial/body temp (C)


@dataclass
class Applicator:
    power_W: float = 40.0    # net deposited power
    time_s: float = 300.0    # ablation duration
    lam_mm: float = 4.0      # radial decay length of deposition
    tip_len_mm: float = 15.0 # active length of antenna (line source along z)


def simulate_zone(tissue: Tissue, appl: Applicator, spacing_mm: float = 1.0,
                  grid_mm: float = 80.0, dt: float | None = None,
                  cem43_thresh: float = 240.0, t_cap: float = 105.0):
    """Return (ablation_mask, T_final, cem43) on a cubic grid centred on the antenna.

    ablation_mask: bool grid where CEM43 >= threshold (coagulative necrosis).
    Tissue temperature is capped at `t_cap` (~vaporisation plateau) for stability.
    """
    n = int(grid_mm / spacing_mm)
    n += (n % 2 == 0)  # make odd so there is a centre voxel
    dx = spacing_mm / 1000.0                       # m
    c0 = n // 2

    # coordinate radius (mm) from the antenna (line source along z through centre)
    ax = (np.arange(n) - c0) * spacing_mm
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    r_line = np.sqrt(X**2 + Y**2)                   # radial distance to z-axis
    along = np.clip(np.abs(Z) - appl.tip_len_mm / 2, 0, None)
    r_eff = np.sqrt(r_line**2 + along**2)           # distance to the active segment

    # deposition density Q (W/m^3), normalised so total integrates to power_W
    q = np.exp(-r_eff / appl.lam_mm)
    vox_vol = dx**3
    q *= appl.power_W / (q.sum() * vox_vol)

    alpha = tissue.k / (tissue.rho * tissue.c)
    if dt is None:
        dt = 0.12 * dx**2 / alpha                   # 3D explicit CFL: factor < 1/6
    steps = int(np.ceil(appl.time_s / dt))
    dt = appl.time_s / steps

    T = np.full((n, n, n), tissue.T_a, dtype=np.float64)
    cem43 = np.zeros_like(T)
    perf = tissue.w_b * tissue.rho_b * tissue.c_b
    coef = dt / (tissue.rho * tissue.c)

    for _ in range(steps):
        lap = (
            np.roll(T, 1, 0) + np.roll(T, -1, 0) +
            np.roll(T, 1, 1) + np.roll(T, -1, 1) +
            np.roll(T, 1, 2) + np.roll(T, -1, 2) - 6 * T
        ) / dx**2
        T = T + coef * (tissue.k * lap + perf * (tissue.T_a - T) + q)
        np.minimum(T, t_cap, out=T)                 # vaporisation plateau cap
        T[0, :, :] = T[-1, :, :] = tissue.T_a       # Dirichlet far-field
        T[:, 0, :] = T[:, -1, :] = tissue.T_a
        T[:, :, 0] = T[:, :, -1] = tissue.T_a
        # CEM43 accumulation (R=0.5 above 43C, 0.25 below); clip exponent for stability
        Rc = np.where(T >= 43.0, 0.5, 0.25)
        expo = np.clip(43.0 - T, -60.0, 60.0)
        cem43 += (Rc ** expo) * (dt / 60.0)

    mask = cem43 >= cem43_thresh
    return mask, T, cem43


def zone_radii_mm(mask: np.ndarray, spacing_mm: float = 1.0):
    """Principal radii (mm) of the ablation zone along x,y,z through its centroid."""
    if not mask.any():
        return (0.0, 0.0, 0.0)
    idx = np.argwhere(mask)
    ext = idx.max(0) - idx.min(0) + 1
    return tuple((ext * spacing_mm / 2.0).tolist())


# --------------------------------------------------------------------------- #
# Fast analytic surrogate: ablation ellipsoid vs (power, time, perfusion)
# Calibrated to published microwave/RF ablation zone charts (e.g. ~3 cm short-axis
# at 40-45 W / 5 min; prolate along the antenna). See docs/LITERATURE.md.
# --------------------------------------------------------------------------- #
AXIAL_RATIO = 1.3  # ablation zones are elongated along the antenna axis


def analytic_zone_radius_mm(power_W: float, time_s: float,
                            perfusion: float = 0.005) -> float:
    """Transverse (short-axis) ablation radius (mm) for one applicator."""
    t_min = time_s / 60.0
    base = 16.0 * np.sqrt(power_W / 40.0)          # ~3.2 cm short-axis @ 40 W plateau
    time_factor = 1.0 - np.exp(-t_min / 3.0)       # saturates ~ 6-9 min
    heat_sink = 1.0 / (1.0 + 40.0 * max(perfusion - 0.003, 0.0))
    return float(base * time_factor * heat_sink)


def analytic_zone_axes_mm(power_W: float, time_s: float, perfusion: float = 0.005):
    """Return (r_transverse, r_transverse, r_axial) mm for the ablation ellipsoid."""
    rt = analytic_zone_radius_mm(power_W, time_s, perfusion)
    return (rt, rt, rt * AXIAL_RATIO)


def calibrate(powers=(30, 40, 50), times=(300, 600), spacing_mm=2.0, grid_mm=90.0):
    """Run a few simulations and report zone radii to sanity-check the surrogate."""
    tis = Tissue()
    out = []
    for p in powers:
        for t in times:
            mask, _, _ = simulate_zone(tis, Applicator(power_W=p, time_s=t),
                                       spacing_mm=spacing_mm, grid_mm=grid_mm)
            rx, ry, rz = zone_radii_mm(mask, spacing_mm)
            out.append((p, t, round(rx, 1), round(ry, 1), round(rz, 1),
                        round(analytic_zone_radius_mm(p, t, tis.w_b), 1)))
    return out


if __name__ == "__main__":
    # NOTE: the explicit-FDM Pennes solver here is a qualitative physics illustration;
    # precise device calibration needs coupled EM + porous-media modelling (COMSOL-grade).
    # For planning we use the literature-calibrated analytic zone model below.
    print("[analytic zone model] power_W time_s -> short-axis radius (mm)")
    for p in (30, 40, 50, 60):
        for t in (300, 480, 600):
            print(f"  {p}W {t//60}min -> r={analytic_zone_radius_mm(p, t):.1f}mm "
                  f"axes={tuple(round(x,1) for x in analytic_zone_axes_mm(p, t))}")
