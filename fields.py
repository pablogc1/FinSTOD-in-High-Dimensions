# -*- coding: utf-8 -*-
"""
Path construction, neighbourhood evaluation and scalar-field assembly.
=====================================================================

Everything here is the ``m``-dimensional counterpart of what
``calc_metrics.cpp`` does per pixel in the published pipeline: build the path for
a cell, build the paths for its von Neumann neighbours, run the pair comparison
against each, and reduce the results to one value per cell.

Two differences from the published pipeline are worth flagging.

*Memory.*  The paper reports storage scaling as ``O(N^m * L)`` and estimates
40 GB for a 1000x1000 grid.  That figure describes the *pipeline*, whose Stage 1
writes every trajectory to HDF5 before Stage 2 reads it.  Local STOD does not
need it: a cell and its ``2m`` neighbours are the only paths in play at any
moment, so the working set is ``O(m * L)`` per cell regardless of ``N`` or ``m``.
:func:`evaluate_cells` streams in chunks accordingly, which is what makes an
``m = 6`` study feasible on a laptop.

*Neighbours at chunk boundaries.*  ``calc_metrics.cpp`` looks neighbours up with
local chunk row indices, so cells on a shard edge silently lose neighbours.  Here
neighbour cells are resolved in the global grid before chunking, so chunking has
no effect on any result.  On the unit torus the grid is periodic and neighbours
wrap, so no cell has fewer than ``2m`` neighbours at all.
"""

from __future__ import annotations

import time

import numpy as np

from .core import STRICT, TYPE_T, TYPE_UC, TYPE_UU, stod_pair
from .progress import print_chunk_progress


# ----------------------------------------------------------------------------
# neighbourhoods
# ----------------------------------------------------------------------------

def von_neumann_offsets(m):
    """The ``2m`` unit offsets of the von Neumann neighbourhood in ``m`` dimensions.

    One pair per coordinate axis, which is the generalisation the paper describes
    in Section 3.2: each comparison probes a single coordinate direction, so the
    neighbourhood grows linearly rather than as ``3^m - 1`` for the Moore case.
    """
    m = int(m)
    offsets = np.zeros((2 * m, m), dtype=np.int64)
    for i in range(m):
        offsets[2 * i, i] = 1
        offsets[2 * i + 1, i] = -1
    return offsets


def build_paths(system, cells, n_cells, n_levels, reverse=True, obs_noise_sigma=0.0,
                clip_to_grid=False):
    """Discretised paths for a batch of seed cells.

    Returns ``(k, n_levels, m)`` integer cell indices.  With ``reverse=True``
    (the FinSTOD convention) level 0 is the *final* cell and the last level is the
    seed cell.

    If ``obs_noise_sigma > 0``, Gaussian measurement noise is added to the observed
    state before cell binning, simulating noisy trajectory data.

    If ``clip_to_grid`` is true, cell indices are clamped to ``[0, n_cells)`` so
    trajectories that leave the box do not carry distinct out-of-domain labels.
    That is the domain-aware curation discussed for backward Duffing in the
    original paper's reviewer reply.  Default is false (published behaviour).
    """
    cells = np.atleast_2d(np.asarray(cells, dtype=np.int64))
    k, m = cells.shape
    n_levels = int(n_levels)
    if n_levels < 1:
        raise ValueError("n_levels must be >= 1")

    system.reset()
    states = system.cell_to_state(cells, n_cells)

    buf = np.empty((n_levels, k, m), dtype=np.int32)
    # Level 0 of the forward path is the seed cell itself
    if obs_noise_sigma > 0.0:
        noise = np.random.normal(0.0, obs_noise_sigma, size=states.shape)
        buf[0] = system.state_to_cell(states + noise, n_cells).astype(np.int32)
    else:
        buf[0] = cells.astype(np.int32)

    for t in range(1, n_levels):
        states = system.step(states)
        if obs_noise_sigma > 0.0:
            noise = np.random.normal(0.0, obs_noise_sigma, size=states.shape)
            idx = system.state_to_cell(states + noise, n_cells)
        else:
            idx = system.state_to_cell(states, n_cells)
        # Default: keep distinct out-of-domain signatures (published UU mechanism).
        # clip_to_grid: reviewer-suggested domain curation — discard excursions
        # by clamping to the computational box.
        if clip_to_grid:
            np.clip(idx, 0, n_cells - 1, out=idx)
        else:
            np.clip(idx, -4 * n_cells, 4 * n_cells, out=idx)
        buf[t] = idx.astype(np.int32)

    paths = np.transpose(buf, (1, 0, 2))
    if reverse:
        paths = paths[:, ::-1, :]
    return np.ascontiguousarray(paths)


def path_locality(paths, n_cells):
    """How the paths sit on the grid -- the regime diagnostic.

    The published 2-D runs integrate with ``dt = 0.001`` on a 1000-cell axis, so a
    trajectory occupies each cell for several consecutive levels and its
    per-coordinate value set covers only part of the axis.  Both numbers control
    how easily cancellations happen, so they are confounds for any ``m``-sweep
    unless they are matched across configurations.  This function measures them so
    that a comparison across ``m`` can be reported alongside the regime it was
    made in.

    Returns a dict with

    ``levels_per_cell``
        Mean run length of consecutive identical cells.  1.0 means every level
        jumps to a new cell (the map regime, where the cancellation mechanism
        degenerates into a birthday problem); the published runs sit near 4.
    ``fill_fraction``
        Mean fraction of the ``n_cells`` values on an axis that a single path's
        coordinate actually visits.  As this approaches 1, every component
        cancels and every pair terminates immediately.
    ``mean_step_cells``
        Mean absolute change in cell index per level per coordinate.  Should be
        well under 1 for an oversampled continuous curve.
    """
    paths = np.asarray(paths)
    if paths.ndim == 2:
        paths = paths[None, :, :]
    k, L, m = paths.shape

    if L > 1:
        same = np.all(paths[:, 1:, :] == paths[:, :-1, :], axis=2)
        n_transitions = int((~same).sum())
        levels_per_cell = (k * L) / float(n_transitions + k)
        steps = np.abs(paths[:, 1:, :].astype(np.int64)
                       - paths[:, :-1, :].astype(np.int64))
        mean_step_cells = float(steps.mean())
    else:
        levels_per_cell = float(L)
        mean_step_cells = 0.0

    distinct = np.empty(k * m, dtype=np.float64)
    idx = 0
    for a in range(k):
        for i in range(m):
            distinct[idx] = np.unique(paths[a, :, i]).size
            idx += 1

    return {
        "levels_per_cell": float(levels_per_cell),
        "fill_fraction": float(distinct.mean() / float(n_cells)),
        "distinct_per_axis": float(distinct.mean()),
        "mean_step_cells": mean_step_cells,
        "n_levels": int(L),
        "n_cells": int(n_cells),
    }


def auto_chunk_size(m, n_levels, budget_bytes=1.2e8):
    """Centre cells per chunk that keep the path buffer inside ``budget_bytes``.

    Each centre cell drags in up to ``2m`` neighbours, and every path costs
    ``n_levels * m`` int32 entries.  This is the whole memory story for local
    STOD: nothing needs to be held beyond the chunk, so the working set never
    scales with ``N^m``.
    """
    per_cell = (1 + 2 * m) * n_levels * m * 4.0
    return int(max(16, min(4000, budget_bytes / per_cell)))


def select_neighbourhood(types, scores, valid=None):
    """Reduce ``2m`` pair results per cell to one value, as the paper prescribes.

    Highest type wins under the priority ``UU > UC > T``, ties broken by the
    larger score (Section 3.2.1, step 3).

    Note for the ``m``-dependence: this is a maximum over ``2m`` samples, so its
    expectation drifts upward with ``m`` for purely combinatorial reasons.  Fields
    computed at different ``m`` are not directly comparable without accounting for
    that; :func:`pairwise_summary` reports the underlying pair statistics, which
    are free of the effect.
    """
    types = np.asarray(types)
    scores = np.asarray(scores, dtype=np.float64)
    if valid is None:
        valid = np.ones(types.shape, dtype=bool)

    effective = np.where(valid, types, -1)
    best_type = effective.max(axis=1)
    is_best = effective == best_type[:, None]
    best_score = np.where(is_best, scores, -np.inf).max(axis=1)

    empty = best_type < 0
    if empty.any():
        best_score = np.where(empty, 0.0, best_score)
    return best_type, best_score


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------

def evaluate_cells(system, cells, n_cells, n_levels, q=None, rule=STRICT,
                   chunk_size=None, progress=True, label="", locality=False,
                   obs_noise_sigma=0.0, clip_to_grid=False):
    """Run the full von Neumann comparison for a list of seed cells.

    Returns a dict with the per-pair arrays (``pair_types``, ``pair_scores``,
    ``pair_kstar``, each ``(S, 2m)``) and the reduced per-cell arrays
    (``cell_types``, ``cell_scores``, each ``(S,)``).  With ``locality=True`` the
    regime diagnostics from :func:`path_locality` are included, measured on the
    first chunk.
    """
    cells = np.atleast_2d(np.asarray(cells, dtype=np.int64))
    n_sites, m = cells.shape
    offsets = von_neumann_offsets(m)

    if chunk_size is None:
        chunk_size = auto_chunk_size(m, n_levels)

    neighbours = cells[:, None, :] + offsets[None, :, :]
    if getattr(system, "periodic", False):
        neighbours = neighbours % n_cells
    # For a bounded system a neighbour index may fall outside [0, n_cells); that
    # is the same situation the published pipeline handles by skipping the pair,
    # so those pairs are marked invalid rather than wrapped.
    valid = np.all((neighbours >= 0) & (neighbours < n_cells), axis=2)
    neighbours = np.clip(neighbours, 0, n_cells - 1)

    pair_types = np.empty((n_sites, 2 * m), dtype=np.int8)
    pair_scores = np.empty((n_sites, 2 * m), dtype=np.float64)
    pair_kstar = np.empty((n_sites, 2 * m), dtype=np.int32)

    locality_info = None
    t0 = time.time()
    for start in range(0, n_sites, chunk_size):
        stop = min(start + chunk_size, n_sites)
        n_chunk = stop - start

        centre_cells = cells[start:stop]
        nb_cells = neighbours[start:stop].reshape(-1, m)

        # One path per distinct cell in the chunk; a cell that is both a centre
        # and a neighbour is integrated once.
        all_cells = np.concatenate((centre_cells, nb_cells), axis=0)
        unique_cells, inverse = np.unique(all_cells, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).ravel()

        paths = build_paths(system, unique_cells, n_cells, n_levels,
                            obs_noise_sigma=obs_noise_sigma,
                            clip_to_grid=clip_to_grid)

        if locality and locality_info is None:
            locality_info = path_locality(paths[: min(64, paths.shape[0])], n_cells)

        idx_centre = inverse[:n_chunk]
        idx_nb = inverse[n_chunk:].reshape(n_chunk, 2 * m)

        for a in range(n_chunk):
            path_a = paths[idx_centre[a]]
            row = start + a
            for b in range(2 * m):
                if not valid[start + a, b]:
                    pair_types[row, b] = TYPE_T
                    pair_scores[row, b] = 0.0
                    pair_kstar[row, b] = -1
                    continue
                res = stod_pair(path_a, paths[idx_nb[a, b]], q=q, rule=rule)
                pair_types[row, b] = res.type_code
                pair_scores[row, b] = res.score
                pair_kstar[row, b] = res.k_star

        if progress:
            print_chunk_progress(label, stop, n_sites, time.time() - t0, kind="FinSTOD")

    cell_types, cell_scores = select_neighbourhood(pair_types, pair_scores, valid)

    return {
        "cells": cells,
        "pair_types": pair_types,
        "pair_scores": pair_scores,
        "pair_kstar": pair_kstar,
        "pair_valid": valid,
        "cell_types": cell_types,
        "cell_scores": cell_scores,
        "m": m,
        "q": int(m if q is None else q),
        "rule": rule,
        "n_levels": int(n_levels),
        "n_cells": int(n_cells),
        "chunk_size": int(chunk_size),
        "locality": locality_info,
        "seconds": time.time() - t0,
    }


def sample_cells(m, n_cells, n_samples, rng, system=None, max_tries=200):
    """Uniformly random seed cells, restricted to the admissible domain.

    For the statistical questions -- how the type split, the termination level and
    the score distribution respond to ``m`` -- sampling is enough, and its cost is
    independent of ``m``.  That is what keeps the scaling study off the ``N^m``
    grid entirely: nothing here ever enumerates the full grid.

    If ``system`` defines a restricted domain via ``valid_cells`` (an energy
    surface, for instance), cells are drawn by rejection until enough admissible
    ones are found.
    """
    n_samples = int(n_samples)
    m = int(m)
    if system is None:
        return rng.integers(0, n_cells, size=(n_samples, m), dtype=np.int64)

    kept = []
    total = 0
    for _ in range(max_tries):
        batch = rng.integers(0, n_cells, size=(4 * n_samples, m), dtype=np.int64)
        ok = np.asarray(system.valid_cells(batch, n_cells), dtype=bool)
        kept.append(batch[ok])
        total += int(ok.sum())
        if total >= n_samples:
            break
    if total == 0:
        raise RuntimeError(
            "no admissible cells found for %r at n_cells=%d; the analysis box is "
            "probably outside the system's physical domain" % (system, n_cells)
        )
    return np.concatenate(kept, axis=0)[:n_samples]


def slice_cells(m, n_cells, axes, anchor=None, stride=1, cell_offset=0):
    """Seed cells covering a 2-D coordinate slice of the ``m``-dimensional grid.

    ``axes`` is the pair of coordinate indices to vary; the others are held at
    ``anchor`` (default: the middle cell).  ``stride`` subsamples the evaluated
    cells without changing the discretisation, so the cell size -- and therefore
    the cancellation statistics -- stay exactly as they are at full resolution.

    Neighbours are still taken in all ``2m`` directions, including the ``2(m-2)``
    that leave the slice.  That matters: restricting the neighbourhood to the
    plane would not be the same diagnostic.
    """
    i0, i1 = int(axes[0]), int(axes[1])
    if i0 == i1 or not (0 <= i0 < m and 0 <= i1 < m):
        raise ValueError("axes must be two distinct coordinate indices in [0, m)")

    if anchor is None:
        anchor = np.full(m, n_cells // 2, dtype=np.int64)
    anchor = np.asarray(anchor, dtype=np.int64).copy()

    ticks = np.arange(0, n_cells, int(stride), dtype=np.int64) + int(cell_offset)
    ticks = np.clip(ticks, 0, n_cells - 1)
    g0, g1 = np.meshgrid(ticks, ticks, indexing="ij")
    n_side = ticks.size

    cells = np.tile(anchor, (n_side * n_side, 1))
    cells[:, i0] = g0.ravel()
    cells[:, i1] = g1.ravel()
    return cells, ticks


def slice_cells_3d(m, n_cells, axes, anchor=None, stride=1):
    """Seed cells covering a 3-D volumetric slice of the ``m``-dimensional grid.

    ``axes`` is the triplet of coordinate indices to vary; the remaining ``m - 3``
    are held at ``anchor`` (default: the middle cell). Evaluates all ``2m``
    von Neumann neighbours for every voxel in the 3D cube.
    """
    i0, i1, i2 = int(axes[0]), int(axes[1]), int(axes[2])
    unique_axes = {i0, i1, i2}
    if len(unique_axes) != 3 or not all(0 <= a < m for a in unique_axes):
        raise ValueError("axes must be three distinct coordinate indices in [0, m)")

    if anchor is None:
        anchor = np.full(m, n_cells // 2, dtype=np.int64)
    anchor = np.asarray(anchor, dtype=np.int64).copy()

    ticks = np.arange(0, n_cells, int(stride), dtype=np.int64)
    g0, g1, g2 = np.meshgrid(ticks, ticks, ticks, indexing="ij")
    n_side = ticks.size
    total_voxels = n_side ** 3

    cells = np.tile(anchor, (total_voxels, 1))
    cells[:, i0] = g0.ravel()
    cells[:, i1] = g1.ravel()
    cells[:, i2] = g2.ravel()
    return cells, ticks


# ----------------------------------------------------------------------------
# normalisation and summaries
# ----------------------------------------------------------------------------

def segmented_normalise(types, scores, log=False):
    """The paper's segmented mapping, pivoted on ``P = N_T / (N_T + N_UC)``.

    T cells are mapped into ``[0, P]``, UC cells into ``[P, 0.99]``, and UU cells
    are set to 1.0 (Section 3.2.1). With ``log=True``, values within each tier
    are log-transformed via ``log1p`` before min-max scaling to preserve relative
    hierarchy while resolving subtle multi-scale gradients across wide dynamic ranges.

    Returns ``(normalised, P)``.
    """
    types = np.asarray(types)
    scores = np.asarray(scores, dtype=np.float64)
    out = np.zeros(scores.shape, dtype=np.float64)

    is_t = types == TYPE_T
    is_uc = types == TYPE_UC
    is_uu = types == TYPE_UU

    n_t = int(is_t.sum())
    n_uc = int(is_uc.sum())
    P = (n_t / float(n_t + n_uc)) if (n_t + n_uc) > 0 else 0.0

    def rescale(mask, lo, hi):
        if not mask.any():
            return
        vals = scores[mask]
        if log:
            vals = np.log1p(np.maximum(0.0, vals))
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            out[mask] = lo + (hi - lo) * (vals - vmin) / (vmax - vmin)
        else:
            out[mask] = lo

    rescale(is_t, 0.0, P)
    rescale(is_uc, P, 0.99)
    out[is_uu] = 1.0
    return out, P


def pairwise_summary(result):
    """Per-pair statistics: where the dimension-scaling question actually lives.

    Reported before the max-over-``2m`` reduction, which has its own
    ``m``-dependence.

    The quantity to watch is the *termination level* ``k*``, not the type split.
    The score is ``sum_{k<=k*} k * U_k``, so it is dominated by how long the pair
    survives, and the field carries information only while ``k*`` sits in the
    interior of ``[0, L-1]``.  Both extremes are degenerate: ``k*`` near 0 means
    every pair is killed immediately and every score is ~0, while ``k*`` pinned at
    ``L-1`` means nothing ever terminates and the score is set by ``L`` rather
    than by the dynamics.  ``frac_kstar_floor`` and ``frac_kstar_maxed`` measure
    the two failure modes, and ``kstar_interior`` is the fraction that is
    informative.
    """
    keep = np.asarray(result.get("pair_valid",
                                 np.ones(result["pair_types"].shape, bool))).ravel()
    types = np.asarray(result["pair_types"]).ravel()[keep]
    scores = np.asarray(result["pair_scores"], dtype=np.float64).ravel()[keep]
    kstar = np.asarray(result["pair_kstar"]).ravel()[keep]
    n = types.size

    frac = {}
    for code, name in ((TYPE_T, "T"), (TYPE_UC, "UC"), (TYPE_UU, "UU")):
        frac[name] = float((types == code).sum()) / n if n else 0.0

    positive = scores[scores > 0]
    if positive.size >= 4:
        q1, q3 = np.percentile(positive, [25, 75])
        iqr_ratio = float(q3 / q1) if q1 > 0 else float("inf")
    else:
        iqr_ratio = float("nan")

    out = {
        "m": result["m"],
        "q": result["q"],
        "rule": result["rule"],
        "n_pairs": int(n),
        "frac_T": frac["T"],
        "frac_UC": frac["UC"],
        "frac_UU": frac["UU"],
        "median_kstar": float(np.median(kstar)),
        "kstar_over_L": float(np.median(kstar)) / max(result["n_levels"] - 1, 1),
        "frac_kstar_floor": float((kstar <= 2).mean()),
        "frac_kstar_maxed": float((kstar >= result["n_levels"] - 1).mean()),
        "kstar_interior": float(((kstar > 2)
                                 & (kstar < result["n_levels"] - 1)).mean()),
        "score_median": float(np.median(scores)),
        "score_iqr_ratio": iqr_ratio,
        "n_distinct_scores": int(np.unique(scores).size),
        "seconds": result["seconds"],
    }
    if result.get("locality"):
        out["levels_per_cell"] = result["locality"]["levels_per_cell"]
        out["fill_fraction"] = result["locality"]["fill_fraction"]
    return out


def summary_table(rows, columns=None, floatfmt="%10.4g"):
    """Format a list of summary dicts as fixed-width text for the console."""
    if not rows:
        return "(no rows)"
    if columns is None:
        columns = list(rows[0].keys())
    widths = [max(len(c), 10) for c in columns]
    lines = ["  ".join(c.rjust(w) for c, w in zip(columns, widths))]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        cells = []
        for c, w in zip(columns, widths):
            v = row.get(c, "")
            if isinstance(v, float):
                cells.append((floatfmt % v).rjust(w))
            else:
                cells.append(str(v).rjust(w))
        lines.append("  ".join(cells))
    return "\n".join(lines)
