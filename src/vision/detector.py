"""Synthetic optical camera renderer and OpenCV centroid detector."""

from __future__ import annotations

import cv2
import numpy as np


def render_optical_frame(
    width: int,
    height: int,
    projected: tuple[float, float, float],
    noise_sigma: float = 15.0,
    cloud_opacity: float = 0.0,
    sun_glare: float = 0.0,
    laser_divergence: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Render a synthetic camera frame with optical disturbances and sensor noise."""
    rng = np.random.default_rng(seed)
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Background starfield or sky illumination
    star_x = rng.integers(0, width, 40)
    star_y = rng.integers(0, height, 40)
    star_b = rng.integers(20, 80, 40)
    frame[star_y, star_x] = np.column_stack((star_b, star_b, star_b))

    # Sunlight glare overlay
    if sun_glare > 0.05:
        glare_mask = np.zeros((height, width), dtype=np.float32)
        cv2.circle(glare_mask, (int(width * 0.85), int(height * 0.15)), int(160 * sun_glare), 1.0, -1)
        glare_mask = cv2.GaussianBlur(glare_mask, (99, 99), 0)
        frame[:, :, 0] = np.clip(frame[:, :, 0] + glare_mask * 200 * sun_glare, 0, 255).astype(np.uint8)
        frame[:, :, 1] = np.clip(frame[:, :, 1] + glare_mask * 170 * sun_glare, 0, 255).astype(np.uint8)

    # Laser Beacon Glow
    u, v, z_camera = projected
    if z_camera > 0 and -30 <= u < width + 30 and -30 <= v < height + 30:
        eff_intensity = max(0.0, 1.0 - cloud_opacity)
        radius = int(round((12.0 * laser_divergence) * eff_intensity + 4.0))
        brightness = int(round(255 * eff_intensity))
        if brightness > 10:
            cv2.circle(
                frame,
                (int(round(u)), int(round(v))),
                radius,
                (brightness, brightness, brightness),
                -1,
                cv2.LINE_AA,
            )

    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Sensor Shot Noise & Read Noise
    if noise_sigma > 0:
        noise = rng.normal(0.0, noise_sigma, frame.shape)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return frame


def detect_beacon_centroid(frame: np.ndarray, threshold: int = 85) -> tuple[float, float] | None:
    """Detect optical beacon centroid using intensity-weighted spatial moments."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(binary)
    if moments["m00"] > 10.0:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        return cx, cy
    return None
