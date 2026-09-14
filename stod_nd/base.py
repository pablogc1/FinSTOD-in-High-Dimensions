# -*- coding: utf-8 -*-
"""
Common grid/state interface shared by the map and flow systems.
==============================================================

A system supplies three things to the rest of the package: the size and topology
of its analysis box, a way to turn integer cell indices into states, and a way to
advance a batch of states by one output level.  Everything else -- path
construction, neighbourhoods, pair comparison -- is written against this
interface only.
"""

from __future__ import annotations

import numpy as np


class GridSystem(object):
    """Base class for anything that can be discretised onto an ``m``-dimensional grid.

    Attributes
    ----------
    m : int
        Number of grid coordinates.
    lo, hi : (m,) float arrays
        The analysis box.
    periodic : bool
        Whether the box wraps.  Periodic boxes have no grid boundary, so every
        cell has a full complement of ``2m`` von Neumann neighbours; this removes
        the edge-effect confound the published 2-D pipeline has to live with.
    coord_names : tuple of str
        Labels for plotting.
    """

    m = 0
    name = "grid_system"
    coord_names = ()
    periodic = False
    lo = None
    hi = None

    def _init_box(self, lo, hi):
        self.lo = np.broadcast_to(np.asarray(lo, dtype=np.float64), (self.m,)).copy()
        self.hi = np.broadcast_to(np.asarray(hi, dtype=np.float64), (self.m,)).copy()
        if np.any(self.hi <= self.lo):
            raise ValueError("need hi > lo on every axis")
        self.span = self.hi - self.lo

    def cell_to_state(self, cells, n_cells):
        """Cell-centre states, matching the pipeline's ``x_min + (c + 0.5) * dx``."""
        cells = np.asarray(cells, dtype=np.int64)
        if cells.ndim == 1:
            cells = cells[None, :]
        frac = (cells.astype(np.float64) + 0.5) / float(n_cells)
        return self.lo[None, :] + frac * self.span[None, :]

    def state_to_cell(self, states, n_cells):
        """Integer cell indices for a batch of states.

        Periodic systems wrap.  Bounded systems are *not* clamped by default,
        following the deliberate choice in ``discretize_path``: a trajectory that
        leaves the analysis box keeps a distinct visitation pattern rather than
        being folded onto the boundary cells.
        """
        states = np.asarray(states, dtype=np.float64)
        frac = (states - self.lo[None, :]) / self.span[None, :]
        scaled = frac * n_cells
        # An unbounded flow can diverge or overflow to non-finite values.  Those
        # levels must still yield *some* index, and it must not collide with a
        # real cell, so they are pushed far outside the box.
        bad = ~np.isfinite(scaled)
        if bad.any():
            scaled = np.where(bad, 8.0 * n_cells, scaled)
        scaled = np.clip(scaled, -8.0 * n_cells, 8.0 * n_cells)
        idx = np.floor(scaled).astype(np.int64)
        if self.periodic:
            np.mod(idx, n_cells, out=idx)
        return idx

    def valid_cells(self, cells, n_cells):
        """Mask of cells whose centre is an admissible initial condition.

        Defaults to everything.  Systems with a restricted physical domain -- an
        energy surface, say -- override this so that seed cells are not placed
        where trajectories immediately escape.
        """
        cells = np.atleast_2d(np.asarray(cells, dtype=np.int64))
        return np.ones(cells.shape[0], dtype=bool)

    def reset(self):
        """Restore any internal state before a fresh batch of trajectories.

        A no-op for autonomous systems.  Non-autonomous ones carry a clock, and
        without rewinding it a chunked field computation would integrate later
        chunks at a later phase, making a cell's result depend on which chunk it
        happened to fall in.
        """

    def step(self, states):  # pragma: no cover - abstract
        """Advance a batch of states by one output level."""
        raise NotImplementedError

    def __repr__(self):
        return "<%s m=%d>" % (self.name, self.m)
