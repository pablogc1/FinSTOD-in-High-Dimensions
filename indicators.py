# -*- coding: utf-8 -*-
"""
Tangent-space reference indicator for the map systems.
=====================================================

A finite-time Lyapunov exponent for maps, evolving the tangent map with periodic
QR reorthonormalisation.  It serves two purposes in this study.

**A baseline.**  The question "does the FinSTOD field still resolve structure as
``m`` grows" needs something to be compared against.  FTLE is the natural
reference, and the rank correlation between the two fields is a threshold-free
way to ask whether FinSTOD still sees what the tangent-space method sees.

**A cost measurement.**  This is where the central practical claim of the
higher-dimensional programme gets tested rather than asserted.  FinSTOD advances
``m`` state components per trajectory; a variational FTLE advances ``m + m^2``,
because it carries the ``m x m`` tangent matrix alongside the state.  The
predicted per-trajectory cost ratio is therefore ``1 : (1 + m)``, growing without
bound in ``m``, and :mod:`experiments.exp01_dimension_scaling` measures it
directly.
"""

from __future__ import annotations

import time

import numpy as np

from .progress import format_duration, print_chunk_progress


def _renormalise(tangent, log_growth):
    """QR-reorthonormalise and accumulate the leading log stretching factor."""
    qmat, rmat = np.linalg.qr(tangent)
    diag = np.einsum("kii->ki", rmat)
    magnitude = np.abs(diag)
    # A vanishing diagonal means the tangent has collapsed numerically; floor it
    # rather than emit -inf.
    np.maximum(magnitude, 1e-300, out=magnitude)
    log_growth += np.log(magnitude[:, 0])
    signs = np.sign(diag)
    signs[signs == 0] = 1.0
    return qmat * signs[:, None, :], log_growth


def flow_ftle(system, cells, n_cells, n_levels, qr_every=1, chunk_size=2000,
              progress=False, label=""):
    """Finite-time Lyapunov exponent of a flow, per seed cell.

    The tangent matrix is co-integrated with the state by the same RK4 scheme the
    flow uses, solving ``M' = Df(y) M`` alongside ``y' = f(y)``, with QR
    reorthonormalisation every ``qr_every`` output levels.

    This is the variational FTLE, matching ``use_variational_ftle: true`` in the
    published configs, and it is the honest cost comparison: ``m + m^2`` components
    advanced per trajectory against FinSTOD's ``m``.

    Returns the exponent per unit time, ``log(growth) / (n_levels * dt_out)``.
    """
    if not hasattr(system, "velocity_jacobian"):
        raise TypeError("%r provides no velocity_jacobian" % (system,))

    cells = np.atleast_2d(np.asarray(cells, dtype=np.int64))
    k, m = cells.shape
    if k == 0:
        return np.empty(0, dtype=np.float64)

    if chunk_size is not None and k > chunk_size:
        out = np.empty(k, dtype=np.float64)
        t0 = time.time()
        for start in range(0, k, chunk_size):
            stop = min(start + chunk_size, k)
            out[start:stop] = flow_ftle(
                system, cells[start:stop], n_cells, n_levels,
                qr_every=qr_every, chunk_size=None, progress=False,
            )
            if progress:
                print_chunk_progress(label, stop, k, time.time() - t0, kind="FTLE")
        return out

    bundle = flow_indicator_bundle(
        system, cells, n_cells, n_levels, qr_every=qr_every,
        chunk_size=None, progress=False, extras=(),
    )
    return bundle["ftle"]


def flow_indicator_bundle(system, cells, n_cells, n_levels, qr_every=1,
                          chunk_size=2000, progress=False, label="",
                          extras=()):
    """FTLE, FLI, Lagrangian descriptors and finite-time rotation number.

    One variational integration.  Time-dependent systems (Bickley, Duffing)
    advance ``system._t`` at RK4 stages, matching ``Flow.step``.

    ``extras`` may contain ``"fput_modes"`` or ``"duffing_basins"``.
    """
    if not hasattr(system, "velocity_jacobian"):
        raise TypeError("%r provides no velocity_jacobian" % (system,))

    cells = np.atleast_2d(np.asarray(cells, dtype=np.int64))
    k, m = cells.shape
    extras = tuple(extras)
    if k == 0:
        return _empty_bundle(m, extras)

    if chunk_size is not None and k > chunk_size:
        parts = []
        t0 = time.time()
        for start in range(0, k, chunk_size):
            stop = min(start + chunk_size, k)
            parts.append(flow_indicator_bundle(
                system, cells[start:stop], n_cells, n_levels,
                qr_every=qr_every, chunk_size=None, progress=False,
                extras=extras,
            ))
            if progress:
                print_chunk_progress(label, stop, k, time.time() - t0, kind="LD/FLI")
        return _concat_bundles(parts)

    system.reset()
    y = system.cell_to_state(cells, n_cells)
    tangent = np.broadcast_to(np.eye(m), (k, m, m)).copy()
    log_growth = np.zeros(k, dtype=np.float64)
    fli = np.full(k, -np.inf, dtype=np.float64)
    w = np.full((k, m), 1.0 / np.sqrt(m), dtype=np.float64)
    w_log = np.zeros(k, dtype=np.float64)
    ld_arc = np.zeros(k, dtype=np.float64)
    ld_p05 = np.zeros(k, dtype=np.float64)
    ftrn = np.zeros(k, dtype=np.float64)
    vel = system.velocity(y)
    ang_prev = np.arctan2(vel[:, min(1, m - 1)], vel[:, 0])

    n_levels = int(n_levels)
    h = system.direction * system.dt_out / system.n_substeps
    dt = abs(h)
    want_fput = "fput_modes" in extras and hasattr(system, "modal_energies")
    want_basins = "duffing_basins" in extras
    n_modes = 0
    acc_E = None
    n_acc = 0
    if want_fput:
        n_modes = int(system.modal_energies(y).shape[1])
        acc_E = np.zeros((k, n_modes), dtype=np.float64)

    sample_every = 10
    for level in range(n_levels):
        for _ in range(system.n_substeps):
            vel = system.velocity(y)
            speed = np.linalg.norm(vel, axis=1)
            ld_arc += speed * dt
            ld_p05 += np.sqrt(np.maximum(speed, 0.0)) * dt
            ang = np.arctan2(vel[:, min(1, m - 1)], vel[:, 0])
            dphi = ang - ang_prev
            dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi
            ftrn += dphi
            ang_prev = ang
            with np.errstate(over="ignore", invalid="ignore"):
                y, tangent, w = _rk4_variational(system, y, tangent, w, h)
            bad = (~np.isfinite(y).all(axis=1)
                   | ~np.isfinite(tangent).all(axis=(1, 2))
                   | ~np.isfinite(w).all(axis=1))
            if bad.any():
                y[bad] = np.nan_to_num(y[bad], nan=0.0, posinf=0.0, neginf=0.0)
                tangent[bad] = np.eye(m)
                w[bad] = 1.0 / np.sqrt(m)
            nrm = np.linalg.norm(w, axis=1)
            nrm = np.maximum(nrm, 1e-300)
            w_log += np.log(nrm)
            w /= nrm[:, None]
            np.maximum(fli, w_log, out=fli)
        if (level + 1) % qr_every == 0 or level == n_levels - 1:
            tangent, log_growth = _renormalise(tangent, log_growth)
        if want_fput and level % sample_every == 0 and level >= n_levels // 2:
            acc_E += system.modal_energies(y)
            n_acc += 1

    total_time = n_levels * abs(system.dt_out)
    out = {
        "ftle": log_growth / max(total_time, 1e-300),
        "fli": fli,
        "ld_arc": ld_arc,
        "ld_p05": ld_p05,
        "ftrn": ftrn / (2.0 * np.pi),
    }
    if want_fput:
        mean_E = acc_E / max(n_acc, 1)
        out["mean_E"] = mean_E
        order = np.argsort(-mean_E, axis=1)
        out["mode0"] = order[:, 0].astype(np.int16)
        out["mode1"] = order[:, 1].astype(np.int16) if n_modes > 1 else order[:, 0].astype(np.int16)
    if want_basins:
        d = int(getattr(system, "d", m // 2))
        x = y[:, :d]
        wells = np.sign(x)
        wells[np.abs(x) < 0.2] = 0.0
        code = np.zeros(k, dtype=np.int32)
        for j in range(d):
            code += (wells[:, j].astype(np.int32) + 1) * (3 ** j)
        out["well_code"] = code
        out["x_end"] = x.astype(np.float32)
    return out


def _empty_bundle(m, extras):
    z = np.empty(0, dtype=np.float64)
    out = {"ftle": z, "fli": z, "ld_arc": z, "ld_p05": z, "ftrn": z}
    if "fput_modes" in extras:
        out["mean_E"] = np.empty((0, 1), dtype=np.float64)
        out["mode0"] = np.empty(0, dtype=np.int16)
        out["mode1"] = np.empty(0, dtype=np.int16)
    if "duffing_basins" in extras:
        out["well_code"] = np.empty(0, dtype=np.int32)
        out["x_end"] = np.empty((0, max(1, m // 2)), dtype=np.float32)
    return out


def _concat_bundles(parts):
    keys = parts[0].keys()
    out = {}
    for key in keys:
        out[key] = np.concatenate([p[key] for p in parts], axis=0)
    return out


def _rk4_variational(system, y, mat, w, h):
    """One RK4 step of state, tangent matrix, and a single FLI vector.

    Advances ``system._t`` at the RK4 stages when the flow is time-dependent.
    """
    t0 = getattr(system, "_t", None)

    def rates(state, tang, vec):
        jac = system.velocity_jacobian(state)
        return (system.velocity(state),
                np.matmul(jac, tang),
                np.matmul(jac, vec[:, :, None])[:, :, 0])

    ky1, km1, kw1 = rates(y, mat, w)
    if t0 is not None:
        system._t = t0 + 0.5 * h
    ky2, km2, kw2 = rates(y + 0.5 * h * ky1, mat + 0.5 * h * km1, w + 0.5 * h * kw1)
    ky3, km3, kw3 = rates(y + 0.5 * h * ky2, mat + 0.5 * h * km2, w + 0.5 * h * kw2)
    if t0 is not None:
        system._t = t0 + h
    ky4, km4, kw4 = rates(y + h * ky3, mat + h * km3, w + h * kw3)
    y_new = y + (h / 6.0) * (ky1 + 2.0 * ky2 + 2.0 * ky3 + ky4)
    m_new = mat + (h / 6.0) * (km1 + 2.0 * km2 + 2.0 * km3 + km4)
    w_new = w + (h / 6.0) * (kw1 + 2.0 * kw2 + 2.0 * kw3 + kw4)
    return y_new, m_new, w_new


def map_ftle(system, cells, n_cells, n_steps, qr_every=1):
    """Finite-time Lyapunov exponent of a map, per seed cell.

    Parameters
    ----------
    system
        Must provide ``step`` and ``jacobian``.
    cells : (k, m) int array
        Seed cells; the trajectory starts at each cell centre.
    n_steps : int
        Number of map iterations.
    qr_every : int
        Reorthonormalise every this many steps.  1 is safest and, for the sizes
        here, not measurably slower.

    Returns
    -------
    (k,) float array
        ``(1/n_steps) * sum log|R_00|``, the leading finite-time exponent.
    """
    if not hasattr(system, "jacobian"):
        raise TypeError("%r provides no jacobian; FTLE needs the tangent map"
                        % (system,))

    cells = np.atleast_2d(np.asarray(cells, dtype=np.int64))
    k, m = cells.shape
    states = system.cell_to_state(cells, n_cells)

    tangent = np.broadcast_to(np.eye(m), (k, m, m)).copy()
    log_growth = np.zeros(k, dtype=np.float64)

    for step_index in range(int(n_steps)):
        jac = system.jacobian(states)                 # (k, m, m)
        tangent = np.matmul(jac, tangent)
        states = system.step(states)
        if (step_index + 1) % qr_every == 0 or step_index == n_steps - 1:
            tangent, log_growth = _renormalise(tangent, log_growth)

    return log_growth / float(n_steps)


def spearman(x, y):
    """Spearman rank correlation, with average ranks for ties.

    Implemented here so the package needs nothing beyond NumPy.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if x.size < 2:
        return float("nan")

    rx = _average_ranks(x)
    ry = _average_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt(np.dot(rx, rx) * np.dot(ry, ry))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(rx, ry) / denom)


def _average_ranks(values):
    n = values.size
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)

    sorted_vals = values[order]
    start = 0
    for i in range(1, n + 1):
        if i == n or sorted_vals[i] != sorted_vals[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


def finite_difference_ftle(system, cells, n_cells, n_levels, grid_shape,
                           obs_noise_sigma=0.0):
    """Compute trajectory-based finite-difference FTLE on a 2D slice grid.

    Measures final flow map displacements from integrated trajectories and
    computes deformation tensor via spatial central differences. Simulates
    the standard experimental LCS method on observational trajectory data.

    Parameters
    ----------
    system : Flow or GridSystem
    cells : (k, m) int array
        Grid cells corresponding to a 2D slice of shape ``grid_shape = (nx, ny)``.
    n_cells : int
    n_levels : int
    grid_shape : (int, int)
    obs_noise_sigma : float
        Standard deviation of Gaussian observation noise added to final states.
    """
    nx, ny = grid_shape
    k = nx * ny
    if cells.shape[0] != k:
        raise ValueError(f"cells count {cells.shape[0]} does not match grid_shape {grid_shape}")

    system.reset()
    states = system.cell_to_state(cells, n_cells)

    for _ in range(int(n_levels)):
        states = system.step(states)

    if obs_noise_sigma > 0.0:
        states = states + np.random.normal(0.0, obs_noise_sigma, size=states.shape)

    # Initial grid spacing in state space
    span = (system.hi - system.lo) / float(n_cells)

    final_2d = states.reshape(nx, ny, -1)

    # Compute deformation gradient via central differences (along axis 0 and 1)
    df_dx0 = np.gradient(final_2d, axis=0) / span[0]
    df_dx1 = np.gradient(final_2d, axis=1) / span[1]

    # Cauchy-Green tensor C = J^T J of shape (nx, ny, 2, 2)
    c00 = np.sum(df_dx0 * df_dx0, axis=-1)
    c01 = np.sum(df_dx0 * df_dx1, axis=-1)
    c11 = np.sum(df_dx1 * df_dx1, axis=-1)

    # Max eigenvalue of 2x2 symmetric matrix
    tr = c00 + c11
    det = c00 * c11 - c01 ** 2
    disc = np.maximum(tr ** 2 - 4.0 * det, 0.0)
    lam_max = np.maximum(0.5 * (tr + np.sqrt(disc)), 1e-300)

    total_time = n_levels * abs(getattr(system, "dt_out", 1.0))
    ftle = (0.5 * np.log(lam_max)) / max(total_time, 1e-300)
    return ftle.ravel()

