"""Integrated optical-tracking simulation engine for UAV and Satellite FSOC scenarios using analytical Clohessy-Wiltshire relative orbit dynamics."""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.control.pid import PIDAxis, PanTiltActuator
from src.metrics.performance import PerformanceTracker
from src.physics.cw import ClohessyWiltshire, get_cw_preset_initial_state
from src.physics.uav import UAVDynamics
from src.tracking.kalman import ImageKalmanFilter
from src.vision.detector import detect_beacon_centroid, render_optical_frame


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "integrated.json"


class SimulationEngine:
    def __init__(self, config_path: Path = DEFAULT_CONFIG) -> None:
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as config_file:
                self.config = json.load(config_file)
        else:
            self.config = {
                "scenario": "uav-uav",
                "cw_case": "case1",
                "fps": 30,
                "time_scale": 15.0,
                "seed": 42,
                "camera": {
                    "width_px": 640,
                    "height_px": 480,
                    "focal_length_px": 800,
                    "threshold": 85,
                    "noise_sigma": 15.0,
                    "beacon_radius_px": 12.0,
                },
                "kalman": {"process_noise": 4.0, "measurement_noise": 30.0},
                "pid": {
                    "kp": 5.5,
                    "ki": 0.1,
                    "kd": 0.5,
                    "max_rate_rad_s": 1.6,
                    "integral_limit": 0.5,
                    "actuator_response_time_s": 0.05,
                },
                "orbit": {
                    "mean_motion_rad_s": 0.00113,
                    "initial_hill_position_m": [10.0, 0.0, 0.0],
                    "initial_hill_velocity_m_s": [0.0, -0.0226, 0.0],
                },
            }

        self.scenario = self.config.get("scenario", "uav-uav")
        self.cw_case = self.config.get("cw_case", "case1")
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.latest_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.latest_telemetry: dict = {}
        self.trajectory: deque[list[float]] = deque(maxlen=300)
        self.frame_number = 0
        self.tracker = PerformanceTracker(scenario=self.scenario)
        self.reset()

    def set_scenario(self, scenario: str) -> None:
        with self.lock:
            self.scenario = scenario
            self.config["scenario"] = scenario
            self.tracker = PerformanceTracker(scenario=scenario)
            self.reset()

    def set_cw_case(self, case_key: str) -> None:
        with self.lock:
            self.cw_case = case_key
            self.config["cw_case"] = case_key
            n = float(self.config["orbit"].get("mean_motion_rad_s", 0.00113))
            p0, v0 = get_cw_preset_initial_state(case_key, n)
            self.config["orbit"]["initial_hill_position_m"] = p0.tolist()
            self.config["orbit"]["initial_hill_velocity_m_s"] = v0.tolist()
            self.reset()

    def reset(self) -> None:
        orbit = self.config["orbit"]
        mean_motion = float(orbit.get("mean_motion_rad_s", 0.00113))
        pos0 = np.asarray(orbit.get("initial_hill_position_m", [10.0, 0.0, 0.0]), dtype=float)
        vel0 = np.asarray(orbit.get("initial_hill_velocity_m_s", [0.0, -0.0226, 0.0]), dtype=float)

        self.cw = ClohessyWiltshire(
            mean_motion_rad_s=mean_motion,
            position_m=pos0,
            velocity_m_s=vel0,
        )

        self.uav = UAVDynamics(
            position_m=pos0,
            velocity_m_s=vel0,
            seed=int(self.config.get("seed", 42)),
        )

        kalman = self.config["kalman"]
        self.kalman = ImageKalmanFilter(kalman["process_noise"], kalman["measurement_noise"])

        pid = self.config["pid"]
        self.pan_pid = PIDAxis(pid["kp"], pid["ki"], pid["kd"], pid["max_rate_rad_s"], pid["integral_limit"])
        self.tilt_pid = PIDAxis(pid["kp"], pid["ki"], pid["kd"], pid["max_rate_rad_s"], pid["integral_limit"])
        self.actuator = PanTiltActuator(pid["actuator_response_time_s"], pid["max_rate_rad_s"])
        self.rng = np.random.default_rng(int(self.config.get("seed", 42)))
        self.trajectory.clear()
        self.frame_number = 0
        self.tracker.reset()

    @staticmethod
    def camera_basis(pan_rad: float, tilt_rad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        forward = np.array(
            [math.sin(pan_rad) * math.cos(tilt_rad), math.sin(tilt_rad), math.cos(pan_rad) * math.cos(tilt_rad)],
            dtype=float,
        )
        right = np.array([math.cos(pan_rad), 0.0, -math.sin(pan_rad)], dtype=float)
        up = np.cross(forward, right)
        return right, up, forward

    def project_target(self, target_world_m: np.ndarray) -> tuple[float, float, float]:
        camera = self.config["camera"]
        right, up, forward = self.camera_basis(self.actuator.pan_rad, self.actuator.tilt_rad)
        x_camera = float(target_world_m @ right)
        y_camera = float(target_world_m @ up)
        z_camera = float(target_world_m @ forward)
        if z_camera <= 0.05:
            return float("nan"), float("nan"), z_camera
        focal = float(camera["focal_length_px"])
        u = camera["width_px"] / 2.0 + focal * x_camera / z_camera
        v = camera["height_px"] / 2.0 - focal * y_camera / z_camera
        return float(u), float(v), z_camera

    def step(self) -> tuple[np.ndarray, dict]:
        fps = float(self.config["fps"])
        control_dt = 1.0 / fps
        physics_dt = control_dt * float(self.config.get("time_scale", 15.0))

        # Deterministic analytical Clohessy-Wiltshire state propagation
        cw_pos, cw_vel = self.cw.step(physics_dt)

        # Coordinate Frame Mapping: CW LVLH [x_radial, y_along, z_cross] -> Three.js World [X, Y_height, Z_depth]
        if self.scenario == "uav-uav":
            target_world = np.array([cw_pos[0], cw_pos[2], cw_pos[1]], dtype=float)
        else:
            # UAV to Ground (fixed ground station at origin, airborne UAV follows CW relative orbit)
            target_world = np.array([cw_pos[0], cw_pos[2], cw_pos[1]], dtype=float)

        self.trajectory.append(target_world.tolist())
        projected = self.project_target(target_world)

        camera = self.config["camera"]
        frame = render_optical_frame(
            width=int(camera["width_px"]),
            height=int(camera["height_px"]),
            projected=projected,
            noise_sigma=float(camera["noise_sigma"]),
            seed=self.rng.integers(0, 1000000),
        )

        measurement = detect_beacon_centroid(frame, int(camera["threshold"]))
        estimate = self.kalman.step(measurement, control_dt)

        centre_u, centre_v = camera["width_px"] / 2.0, camera["height_px"] / 2.0
        if estimate is not None:
            pan_command = self.pan_pid.step(float(estimate[0] - centre_u), control_dt)
            tilt_command = self.tilt_pid.step(float(centre_v - estimate[1]), control_dt)
        else:
            pan_command = tilt_command = 0.0
        self.actuator.step(pan_command, tilt_command, control_dt)

        # Draw visual overlays
        if measurement is not None:
            cv2.drawMarker(frame, tuple(np.rint(measurement).astype(int)), (0, 255, 90), cv2.MARKER_CROSS, 20, 2)
        if estimate is not None:
            cv2.circle(frame, tuple(np.rint(estimate[:2]).astype(int)), 11, (255, 210, 40), 2, cv2.LINE_AA)
        cv2.drawMarker(frame, (int(centre_u), int(centre_v)), (50, 80, 255), cv2.MARKER_CROSS, 24, 1)

        ground_truth = None if not np.isfinite(projected[0]) else [projected[0], projected[1]]
        tracking_error = None if estimate is None else float(np.linalg.norm(estimate[:2] - [centre_u, centre_v]))
        locked = bool(measurement is not None and tracking_error is not None and tracking_error < 25.0)

        self.tracker.record_frame(
            t_s=self.frame_number * physics_dt,
            status="TRACKING" if locked else "ACQUIRING",
            tracking_error_px=tracking_error,
            range_m=float(np.linalg.norm(target_world)),
            fps=fps,
            locked=locked,
        )

        telemetry = {
            "frame": self.frame_number,
            "scenario": self.scenario,
            "cw_case": self.cw_case,
            "simulation_time_s": round(self.frame_number * physics_dt, 3),
            "cw_position_m": [round(float(v), 4) for v in cw_pos],
            "cw_velocity_m_s": [round(float(v), 6) for v in cw_vel],
            "target_world_m": [round(float(v), 4) for v in target_world],
            "opencv_pixel": None if measurement is None else [round(float(v), 2) for v in measurement],
            "kalman_pixel": None if estimate is None else [round(float(v), 2) for v in estimate[:2]],
            "tracking_error_px": None if tracking_error is None else round(tracking_error, 2),
            "camera_pan_rad": round(self.actuator.pan_rad, 5),
            "camera_tilt_rad": round(self.actuator.tilt_rad, 5),
            "locked": locked,
            "acquisition_time_s": self.tracker.acquisition_time_s,
            "lock_retention_pct": round((self.tracker.lock_frames / max(1, self.tracker.total_frames)) * 100.0, 1),
            "trajectory_m": list(self.trajectory),
        }
        self.frame_number += 1
        return frame, telemetry

    def run(self) -> None:
        period = 1.0 / float(self.config["fps"])
        while not self.stop_event.is_set():
            started = time.perf_counter()
            frame, telemetry = self.step()
            with self.lock:
                self.latest_frame = frame
                self.latest_telemetry = telemetry
            self.stop_event.wait(max(0.0, period - (time.perf_counter() - started)))

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.run, name="simulation-engine", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)

    def telemetry_snapshot(self) -> dict:
        with self.lock:
            return dict(self.latest_telemetry)

    def jpeg_snapshot(self) -> bytes:
        with self.lock:
            frame = self.latest_frame.copy()
        encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not encoded:
            raise RuntimeError("Could not encode camera frame")
        return jpeg.tobytes()
