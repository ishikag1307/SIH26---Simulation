"""UAV kinematics and atmospheric disturbance physics model."""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np


@dataclass
class UAVDynamics:
    """3D trajectory propagation for UAV platforms with Dryden turbulence and vibration."""

    position_m: np.ndarray
    velocity_m_s: np.ndarray
    wind_speed_m_s: float = 10.0
    vibration_amplitude_mrad: float = 2.0
    seed: int = 42

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.t = 0.0

    def gust(self, seed_offset: float, mag: float) -> float:
        """Dryden wind gust approximation using sum of incommensurate sinusoids."""
        if mag <= 0:
            return 0.0
        return mag * (
            0.52 * math.sin(self.t * 0.21 + seed_offset)
            + 0.31 * math.sin(self.t * 0.53 + seed_offset * 2.1)
            + 0.15 * math.sin(self.t * 1.12 + seed_offset * 3.7)
            + 0.07 * math.sin(self.t * 2.31 + seed_offset * 5.4)
        )

    def step(self, dt_s: float, scenario: str = "uav-ground") -> tuple[np.ndarray, np.ndarray]:
        """Advance UAV state forward by dt_s seconds."""
        self.t += dt_s
        wind_mag = self.wind_speed_m_s / 10.0

        if scenario == "uav-ground":
            # Self UAV hovering/patrolling in wind
            x = 9.0 * math.cos(self.t * 0.05) + self.gust(1.1, wind_mag * 0.45)
            y = 6.0 + self.gust(1.7, wind_mag * 0.12)
            z = 9.0 * math.sin(self.t * 0.05) + self.gust(2.3, wind_mag * 0.45)
        elif scenario == "uav-uav":
            # Dual UAV relative movement
            x = 10.0 * math.cos(self.t * 0.07 + 1.4) + self.gust(6.2, wind_mag * 0.35)
            y = 5.5 + self.gust(6.9, wind_mag * 0.14)
            z = 10.0 * math.sin(self.t * 0.09) + self.gust(7.3, wind_mag * 0.35)
        else:
            x, y, z = self.position_m

        new_pos = np.array([x, y, z], dtype=float)
        velocity = (new_pos - self.position_m) / max(dt_s, 1e-4)
        self.position_m = new_pos
        self.velocity_m_s = velocity
        return self.position_m.copy(), self.velocity_m_s.copy()

    def get_vibration_jitter(self) -> tuple[float, float]:
        """Generate high-frequency mechanical vibration angular jitter in radians."""
        amp_rad = math.radians(self.vibration_amplitude_mrad / 1000.0)
        yaw_jitter = float(self.rng.normal(0.0, amp_rad))
        pitch_jitter = float(self.rng.normal(0.0, amp_rad))
        return yaw_jitter, pitch_jitter
