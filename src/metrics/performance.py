"""Performance metrics recorder and CSV report generator for FSOC coarse alignment."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PerformanceTracker:
    """Logs simulation metrics and calculates acquisition, error, and lock retention statistics."""

    scenario: str = "uav-ground"
    records: list[dict] = field(default_factory=list)
    acquisition_time_s: float | None = None
    lock_hits: int = 0
    total_frames: int = 0
    lock_frames: int = 0

    def reset(self) -> None:
        self.records.clear()
        self.acquisition_time_s = None
        self.lock_hits = 0
        self.total_frames = 0
        self.lock_frames = 0

    def record_frame(
        self,
        t_s: float,
        status: str,
        tracking_error_px: float | None,
        range_m: float,
        fps: float,
        locked: bool,
    ) -> None:
        self.total_frames += 1
        if locked:
            self.lock_frames += 1
            if self.acquisition_time_s is None:
                self.acquisition_time_s = t_s

        self.records.append(
            {
                "t_s": round(t_s, 2),
                "status": status,
                "tracking_error_px": "" if tracking_error_px is None else round(tracking_error_px, 2),
                "range_m": round(range_m, 2),
                "fps": round(fps, 1),
            }
        )

    def summary(self) -> dict:
        lock_retention_pct = (self.lock_frames / max(1, self.total_frames)) * 100.0
        errors = [r["tracking_error_px"] for r in self.records if isinstance(r["tracking_error_px"], (int, float))]
        avg_error = float(np.mean(errors)) if errors else None
        max_error = float(np.max(errors)) if errors else None

        return {
            "scenario": self.scenario,
            "total_frames": self.total_frames,
            "acquisition_time_s": self.acquisition_time_s,
            "lock_retention_pct": round(lock_retention_pct, 2),
            "average_tracking_error_px": None if avg_error is None else round(avg_error, 2),
            "maximum_tracking_error_px": None if max_error is None else round(max_error, 2),
        }

    def generate_csv(self) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["t_s", "status", "tracking_error_px", "range_m", "fps"])
        for r in self.records:
            writer.writerow([r["t_s"], r["status"], r["tracking_error_px"], r["range_m"], r["fps"]])

        writer.writerow([])
        writer.writerow(["SUMMARY METRICS"])
        summ = self.summary()
        for k, v in summ.items():
            writer.writerow([k, v if v is not None else "N/A"])

        return output.getvalue()
