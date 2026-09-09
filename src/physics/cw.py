"""Analytical Clohessy-Wiltshire (Hill) relative orbital dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np


def propagate_cw_analytical(
    x0: float, y0: float, z0: float,
    vx0: float, vy0: float, vz0: float,
    t_s: float, n_rad_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Analytical closed-form solution to linearized Clohessy-Wiltshire equations.

    LVLH Frame:
    - x: Radial direction (Zenith / altitude offset)
    - y: Along-track direction (orbital velocity direction)
    - z: Cross-track direction (out-of-plane, orbit normal)
    """
    nt = n_rad_s * t_s
    sin_nt = math.sin(nt)
    cos_nt = math.cos(nt)

    # Position equations
    x = (4.0 - 3.0 * cos_nt) * x0 + (sin_nt / n_rad_s) * vx0 + (2.0 * (1.0 - cos_nt) / n_rad_s) * vy0
    y = (
        6.0 * (sin_nt - nt) * x0
        + y0
        - (2.0 * (1.0 - cos_nt) / n_rad_s) * vx0
        + ((4.0 * sin_nt - 3.0 * nt) / n_rad_s) * vy0
    )
    z = cos_nt * z0 + (sin_nt / n_rad_s) * vz0

    # Velocity equations (time derivative)
    vx = 3.0 * n_rad_s * sin_nt * x0 + cos_nt * vx0 + 2.0 * sin_nt * vy0
    vy = 6.0 * n_rad_s * (cos_nt - 1.0) * x0 - 2.0 * sin_nt * vx0 + (4.0 * cos_nt - 3.0) * vy0
    vz = -n_rad_s * sin_nt * z0 + cos_nt * vz0

    return np.array([x, y, z], dtype=float), np.array([vx, vy, vz], dtype=float)


@dataclass
class ClohessyWiltshire:
    """Propagate [radial x, along-track y, cross-track z] position and velocity analytical state."""

    mean_motion_rad_s: float
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    initial_position_m: np.ndarray | None = None
    initial_velocity_m_s: np.ndarray | None = None
    t_s: float = 0.0

    def __post_init__(self) -> None:
        if self.initial_position_m is None:
            self.initial_position_m = self.position_m.copy()
        if self.initial_velocity_m_s is None:
            self.initial_velocity_m_s = self.velocity_m_s.copy()

    def reset(self, pos: np.ndarray | None = None, vel: np.ndarray | None = None) -> None:
        if pos is not None:
            self.initial_position_m = pos.copy()
        if vel is not None:
            self.initial_velocity_m_s = vel.copy()
        self.t_s = 0.0
        self.position_m = self.initial_position_m.copy()
        self.velocity_m_s = self.initial_velocity_m_s.copy()

    def step(self, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Advance time by dt_s and return analytical CW position and velocity."""
        self.t_s += dt_s
        x0, y0, z0 = self.initial_position_m
        vx0, vy0, vz0 = self.initial_velocity_m_s
        pos, vel = propagate_cw_analytical(
            x0, y0, z0, vx0, vy0, vz0, self.t_s, self.mean_motion_rad_s
        )
        self.position_m = pos
        self.velocity_m_s = vel
        return self.position_m.copy(), self.velocity_m_s.copy()


def get_cw_preset_initial_state(case_key: str, n_rad_s: float = 0.00113) -> tuple[np.ndarray, np.ndarray]:
    """Get physically meaningful initial conditions [x0, y0, z0] and [vx0, vy0, vz0] for CW cases."""
    if case_key == "case1":
        # Bounded Natural Ellipse (vy0 = -2 * n * x0)
        x0, y0, z0 = 10.0, 0.0, 0.0
        vx0, vy0, vz0 = 0.0, -2.0 * n_rad_s * x0, 0.0
    elif case_key == "case2":
        # In-Plane Elliptical Orbit
        x0, y0, z0 = 15.0, 30.0, 0.0
        vx0, vy0, vz0 = 0.04, -2.0 * n_rad_s * x0, 0.0
    elif case_key == "case3":
        # Along-Track Drift (vy0 != -2 * n * x0)
        x0, y0, z0 = 10.0, 0.0, 0.0
        vx0, vy0, vz0 = 0.0, -1.2 * n_rad_s * x0, 0.0
    elif case_key == "case4":
        # Cross-Track Oscillation
        x0, y0, z0 = 0.0, 0.0, 25.0
        vx0, vy0, vz0 = 0.0, 0.0, 0.05
    elif case_key == "case5":
        # General 3D Relative Orbit
        x0, y0, z0 = 12.0, 20.0, 15.0
        vx0, vy0, vz0 = 0.03, -2.0 * n_rad_s * x0, 0.04
    else:
        # Default Bounded
        x0, y0, z0 = 10.0, 0.0, 0.0
        vx0, vy0, vz0 = 0.0, -2.0 * n_rad_s * x0, 0.0

    return np.array([x0, y0, z0], dtype=float), np.array([vx0, vy0, vz0], dtype=float)
