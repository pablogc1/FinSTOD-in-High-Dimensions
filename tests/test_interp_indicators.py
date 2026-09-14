# -*- coding: utf-8 -*-
"""Time-aware indicators, clip-to-grid, and Duffing forcing."""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from stod_nd import BickleyJetFlow, CoupledDuffing, build_paths, flow_indicator_bundle, spearman
from stod_nd.fields import slice_cells


def test_duffing_forcing_advances():
    sys = CoupledDuffing(d=1, dt_out=0.01, n_substeps=4)
    sys.reset()
    y = np.zeros((1, 2))
    period_half = np.pi / sys.omega
    n = int(round(period_half / sys.dt_out))
    for _ in range(n):
        y = sys.step(y)
    assert abs(sys._t - n * sys.dt_out) < 1e-9
    # At t ≈ π/ω, cos(ωt) ≈ -1 so the force at the origin is -γ.
    sys.reset()
    y0 = np.zeros((1, 2))
    h = period_half
    sys._rk4(y0, h)
    force = sys.velocity(np.zeros((1, 2)))[0, 1]
    assert abs(force - (-sys.gamma)) < 1e-6, force


def test_bickley_time_advances():
    sys = BickleyJetFlow(eps=(0.075, 0.15, 0.3), dt_out=0.04, n_substeps=4)
    sys.reset()
    y = np.array([[1.0, 0.0]])
    y = sys.step(y)
    assert abs(sys._t - sys.dt_out) < 1e-12


def test_pure_shear_ld_tracks_speed():
    sys = BickleyJetFlow(eps=(0.0, 0.0, 0.0), dt_out=0.05, n_substeps=4)
    cells, _ticks = slice_cells(2, 40, (0, 1), stride=8)
    n_levels = 20
    bundle = flow_indicator_bundle(
        sys, cells, 40, n_levels, chunk_size=None, extras=(),
    )
    y = sys.cell_to_state(cells, 40)[:, 1]
    u = sys.U / np.cosh(np.clip(y / sys.L, -50, 50)) ** 2
    tau = n_levels * sys.dt_out
    pred = np.abs(u) * tau
    rho = spearman(bundle["ld_arc"], pred)
    assert rho > 0.95, rho
    # No exponential stretching: FTLE near 0.
    assert np.nanmax(np.abs(bundle["ftle"])) < 0.05, np.nanmax(np.abs(bundle["ftle"]))


def test_clip_to_grid_clamps():
    sys = CoupledDuffing(d=1, dt_out=0.05, n_substeps=2)
    cells = np.array([[20, 39]], dtype=np.int64)
    n_cells, n_levels = 40, 30
    raw = build_paths(sys, cells, n_cells, n_levels, clip_to_grid=False)
    clipped = build_paths(sys, cells, n_cells, n_levels, clip_to_grid=True)
    assert clipped.min() >= 0
    assert clipped.max() <= n_cells - 1
    # Unclipped paths are allowed outside the box.
    assert raw.min() < 0 or raw.max() >= n_cells or np.array_equal(raw, clipped)


def test_format_duration_uses_largest_unit():
    from stod_nd.progress import format_duration
    assert format_duration(12) == "12s"
    assert format_duration(90) == "1.5m"
    assert format_duration(7200) == "2.0h"
    assert format_duration(90000) == "1.0d"
    assert format_duration(-1) == "?"


def test_format_array_watch_is_shell_safe():
    from stod_nd.progress import format_array_watch
    line = format_array_watch(0, 60, 12, 60)
    assert "0/60" in line
    assert "ETA" in line
    assert "12s" in line


if __name__ == "__main__":
    test_duffing_forcing_advances()
    test_bickley_time_advances()
    test_pure_shear_ld_tracks_speed()
    test_clip_to_grid_clamps()
    test_format_duration_uses_largest_unit()
    test_format_array_watch_is_shell_safe()
    print("test_interp_indicators: all passed")
