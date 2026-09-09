from __future__ import annotations

import math
import numpy as np

from src.core.engine import SimulationEngine
from src.physics.cw import ClohessyWiltshire, get_cw_preset_initial_state, propagate_cw_analytical


def test_cw_analytical_propagation_remains_finite_and_deterministic() -> None:
    n = 0.00113
    p0, v0 = get_cw_preset_initial_state("case1", n)
    x0, y0, z0 = p0
    vx0, vy0, vz0 = v0

    # Test deterministic evaluation at t = 1000s
    pos1, vel1 = propagate_cw_analytical(x0, y0, z0, vx0, vy0, vz0, 1000.0, n)
    pos2, vel2 = propagate_cw_analytical(x0, y0, z0, vx0, vy0, vz0, 1000.0, n)

    assert np.all(np.isfinite(pos1))
    assert np.all(np.isfinite(vel1))
    assert np.array_equal(pos1, pos2)
    assert np.array_equal(vel1, vel2)


def test_cw_bounded_vs_drift_cases() -> None:
    n = 0.00113
    period = 2.0 * math.pi / n

    # Bounded case (vy0 = -2 * n * x0)
    p_b, v_b = get_cw_preset_initial_state("case1", n)
    pos_b_0, _ = propagate_cw_analytical(p_b[0], p_b[1], p_b[2], v_b[0], v_b[1], v_b[2], 0.0, n)
    pos_b_1, _ = propagate_cw_analytical(p_b[0], p_b[1], p_b[2], v_b[0], v_b[1], v_b[2], period, n)

    # After 1 full period T = 2pi/n, bounded ellipse returns to initial position
    assert np.isclose(pos_b_0[0], pos_b_1[0], atol=1e-3)
    assert np.isclose(pos_b_0[1], pos_b_1[1], atol=1e-3)

    # Drift case (vy0 != -2 * n * x0)
    p_d, v_d = get_cw_preset_initial_state("case3", n)
    pos_d_1, _ = propagate_cw_analytical(p_d[0], p_d[1], p_d[2], v_d[0], v_d[1], v_d[2], period, n)
    # Drift case shows along-track displacement after 1 period
    assert not np.isclose(p_d[1], pos_d_1[1], atol=1.0)


def test_simulation_engine_cw_cases() -> None:
    engine = SimulationEngine()
    for case_key in ["case1", "case2", "case3", "case4", "case5"]:
        engine.set_cw_case(case_key)
        frame, telemetry = engine.step()
        assert frame.shape == (480, 640, 3)
        assert telemetry["cw_case"] == case_key
        assert len(telemetry["cw_position_m"]) == 3
