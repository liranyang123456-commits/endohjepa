"""Component C - deformable soft-tissue simulation via Position-Based Dynamics (PBD).

A rectangular sheet of particles connected by distance constraints (structural +
shear + bending) approximates elastic tissue. An instrument tip presses into it; PBD
projects constraints each substep so the surface deforms physically and springs back.
This is the "physics engine" the world model can eventually be coupled to.

Pure numpy; renders the deformed mesh to images for visualisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PBDConfig:
    nx: int = 40
    ny: int = 40
    spacing: float = 1.0
    gravity: float = 0.0          # tissue is roughly neutrally supported
    stiffness: float = 0.9        # constraint stiffness in [0,1]
    substeps: int = 20
    damping: float = 0.02
    pin_border: bool = True       # clamp the sheet edges (attached tissue)


class SoftBodyPBD:
    def __init__(self, cfg: PBDConfig):
        self.cfg = cfg
        nx, ny, s = cfg.nx, cfg.ny, cfg.spacing
        gx, gy = np.meshgrid(np.arange(nx) * s, np.arange(ny) * s, indexing="xy")
        z = np.zeros_like(gx)
        self.X0 = np.stack([gx, gy, z], axis=-1).reshape(-1, 3).astype(np.float64)
        self.X = self.X0.copy()
        self.V = np.zeros_like(self.X)
        self.w = np.ones(len(self.X))          # inverse mass
        self.nx, self.ny = nx, ny

        if cfg.pin_border:
            ii = np.arange(len(self.X)).reshape(ny, nx)
            border = np.concatenate([ii[0], ii[-1], ii[:, 0], ii[:, -1]])
            self.w[border] = 0.0

        self.constraints = self._build_constraints()

    def _idx(self, r, c):
        return r * self.nx + c

    def _build_constraints(self):
        cons = []
        nx, ny = self.nx, self.ny
        def add(a, b):
            rest = np.linalg.norm(self.X0[a] - self.X0[b])
            cons.append((a, b, rest))
        for r in range(ny):
            for c in range(nx):
                if c + 1 < nx:
                    add(self._idx(r, c), self._idx(r, c + 1))     # structural
                if r + 1 < ny:
                    add(self._idx(r, c), self._idx(r + 1, c))
                if c + 1 < nx and r + 1 < ny:
                    add(self._idx(r, c), self._idx(r + 1, c + 1)) # shear
                    add(self._idx(r + 1, c), self._idx(r, c + 1))
                if c + 2 < nx:
                    add(self._idx(r, c), self._idx(r, c + 2))     # bending
                if r + 2 < ny:
                    add(self._idx(r, c), self._idx(r + 2, c))
        return cons

    def _project(self):
        k = self.cfg.stiffness
        for a, b, rest in self.constraints:
            wa, wb = self.w[a], self.w[b]
            if wa + wb == 0:
                continue
            d = self.X[a] - self.X[b]
            L = np.linalg.norm(d)
            if L < 1e-9:
                continue
            corr = k * (L - rest) / L * d
            self.X[a] -= wa / (wa + wb) * corr
            self.X[b] += wb / (wa + wb) * corr

    def press(self, center_xy, radius, depth):
        """Instrument pushes particles within `radius` down by up to `depth` (z-)."""
        xy = self.X[:, :2]
        dist = np.linalg.norm(xy - np.asarray(center_xy), axis=1)
        m = dist < radius
        fall = np.clip(1 - dist[m] / radius, 0, 1)
        self.X[m, 2] = -depth * fall
        self.w_saved = self.w.copy()

    def step(self, dt=1.0):
        g = np.array([0, 0, -self.cfg.gravity])
        self.V += dt * g
        self.V *= (1 - self.cfg.damping)
        Xprev = self.X.copy()
        self.X = self.X + dt * self.V * self.w[:, None]
        for _ in range(self.cfg.substeps):
            self._project()
        self.V = (self.X - Xprev) / dt

    def render(self, size=256, elev=35.0, azim=30.0):
        """Orthographic render of the mesh coloured by depression depth."""
        R = _rot(elev, azim)
        P = self.X @ R.T
        xy = P[:, :2]
        mn, mx = xy.min(0), xy.max(0)
        span = (mx - mn).max() + 1e-6
        uv = ((xy - mn) / span * (size - 8) + 4).astype(int)
        img = np.zeros((size, size, 3), np.uint8)
        depth = -self.X[:, 2]
        dn = (depth - depth.min()) / (np.ptp(depth) + 1e-6)
        cols = (np.stack([dn, 0.3 + 0.4 * (1 - dn), 1 - dn], -1) * 255).astype(np.uint8)
        order = np.argsort(P[:, 2])
        for i in order:
            u, v = uv[i]
            img[np.clip(v-1, 0, size-1):v+2, np.clip(u-1, 0, size-1):u+2] = cols[i]
        return img


def _rot(elev, azim):
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return Rx @ Rz
