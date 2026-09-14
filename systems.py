# -*- coding: utf-8 -*-
"""
Test systems for the higher-dimensional STOD study.
==================================================

Everything here is a **map on the unit torus** ``[0, 1)^m`` rather than a flow.
That is a deliberate choice for the scaling study, for three reasons.

1.  The published 2-D paper measures 89% of its runtime in the ODE integration.
    Maps remove that entirely, which is what makes ``N^m`` grids reachable on a
    laptop instead of on 600 cores.
2.  Coupled standard maps are the standard testbed in the literature on
    higher-dimensional Hamiltonian chaos, so results here can be compared
    against existing work rather than existing in isolation.
3.  On the unit torus every coordinate has the same range and the same units, so
    the axis-scaling non-objectivity that Reviewer #3 raised against the 2-D
    paper is neutralised by construction.  A mixed position/momentum grid in
    physical units would confound the ``m``-dependence we are trying to measure
    with an arbitrary choice of relative scaling.

Both systems below expose the same small interface, used by :mod:`stod_nd.fields`:

``m``
    Number of grid coordinates.
``coord_names``
    Labels, for plotting.
``cell_to_state(cells, n_cells)``
    Cell-centre states for integer cell indices, shape ``(k, m)``.
``step(states)``
    One iteration, vectorised over the leading axis.

Adding a flow means providing a ``step`` that advances by one output interval;
nothing else in the package assumes discrete time.
"""

from __future__ import annotations

import numpy as np

from .base import GridSystem

TWO_PI = 2.0 * np.pi

# Fractional parts of sqrt(prime) -- badly approximable enough to serve as
# rotation numbers without landing on low-order resonances.
_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def default_frequencies(d):
    """``d`` distinct irrational rotation numbers in ``(0, 1)``."""
    if d > len(_PRIMES):
        raise ValueError("only %d built-in frequencies available" % len(_PRIMES))
    return np.array([np.sqrt(p) % 1.0 for p in _PRIMES[:d]], dtype=np.float64)


class TorusSystem(GridSystem):
    """A map on the unit torus ``[0, 1)^m``; periodic, so no grid boundary."""

    periodic = True

    def _init_torus(self):
        self._init_box(0.0, 1.0)


class CoupledStandardMap(TorusSystem):
    """A ring of ``d`` coupled standard maps; symplectic, ``m = 2d``.

    Coordinates are laid out blocked as ``(x_1..x_d, p_1..p_d)``, so ``x_j`` is
    column ``j`` and ``p_j`` is column ``d + j``.

    One iteration, with indices cyclic in ``j``::

        p_j' = p_j + (K/2pi) sin(2pi x_j) + (xi/2pi) sin(2pi (x_j + x_{j+1}))
        x_j' = x_j + p_j'

    both reduced mod 1.  At ``d = 1`` the coupling term is dropped and this is the
    Chirikov standard map.  At ``d = 2`` it is the 4-D Froeschle map.  ``K``
    controls the kick strength, so small ``K`` is near-integrable (invariant tori
    of dimension ``d``) and large ``K`` is strongly chaotic.
    """

    def __init__(self, d, K=0.9, xi=0.3, name=None):
        d = int(d)
        if d < 1:
            raise ValueError("d must be >= 1")
        self.d = d
        self.m = 2 * d
        self.K_vec = np.broadcast_to(
            np.asarray(K, dtype=np.float64), (d,)
        ).astype(np.float64).copy()
        self.K = float(self.K_vec[0])
        self.xi = float(xi)
        self._init_torus()
        self.name = name or ("coupled_standard_map_d%d" % d)
        self.coord_names = tuple(
            ["x%d" % (j + 1) for j in range(d)] + ["p%d" % (j + 1) for j in range(d)]
        )

    def step(self, states):
        states = np.asarray(states, dtype=np.float64)
        d = self.d
        x = states[:, :d]
        p = states[:, d:]

        kick = (self.K_vec[None, :] / TWO_PI) * np.sin(TWO_PI * x)
        if d >= 2 and self.xi != 0.0:
            partner = np.roll(x, -1, axis=1)
            kick = kick + (self.xi / TWO_PI) * np.sin(TWO_PI * (x + partner))

        p_new = np.mod(p + kick, 1.0)
        x_new = np.mod(x + p_new, 1.0)

        out = np.empty_like(states)
        out[:, :d] = x_new
        out[:, d:] = p_new
        return out

    def jacobian(self, states):
        """Tangent map at each state, shape ``(k, m, m)``.

        With the state ordered ``(x, p)`` and ``A_jk = d p_j' / d x_k``::

            J = [[ I + A,  I ],
                 [   A  ,  I ]]

        since ``x_j' = x_j + p_j'`` and ``d p_j' / d p_k = delta_jk``.
        """
        states = np.asarray(states, dtype=np.float64)
        k = states.shape[0]
        d = self.d
        x = states[:, :d]

        A = np.zeros((k, d, d), dtype=np.float64)
        diag = self.K_vec[None, :] * np.cos(TWO_PI * x)
        if d >= 2 and self.xi != 0.0:
            partner = np.roll(x, -1, axis=1)
            cross = self.xi * np.cos(TWO_PI * (x + partner))
            diag = diag + cross
            # d p_j' / d x_{j+1}
            j_idx = np.arange(d)
            A[:, j_idx, (j_idx + 1) % d] += cross
        j_idx = np.arange(d)
        A[:, j_idx, j_idx] += diag

        eye = np.eye(d)[None, :, :]
        jac = np.empty((k, 2 * d, 2 * d), dtype=np.float64)
        jac[:, :d, :d] = eye + A
        jac[:, :d, d:] = eye
        jac[:, d:, :d] = A
        jac[:, d:, d:] = eye
        return jac


class TorusTranslation(TorusSystem):
    """Rigid translation on ``[0, 1)^m`` with only ``d`` coordinates moving.

    ``y' = (y + omega) mod 1`` where ``omega_i`` is irrational for ``i < d`` and
    zero for ``i >= d``.  Every orbit therefore fills a ``d``-dimensional torus
    embedded in the ``m``-dimensional grid, with the remaining ``m - d``
    coordinates frozen at their initial values.

    This is the calibration system for the ``STOD_q`` / torus-dimension
    hypothesis: ``d`` is known exactly and set independently of ``m``, with no
    dynamical ambiguity about what the "right answer" is.

    ``shear`` optionally adds a linear coupling ``y_i += shear * y_{i-1}`` among
    the moving coordinates, which makes neighbouring orbits separate instead of
    staying rigidly parallel.  ``shear = 0`` is the pure translation.
    """

    def __init__(self, m, d, freqs=None, shear=0.0, name=None):
        m = int(m)
        d = int(d)
        if not 1 <= d <= m:
            raise ValueError("need 1 <= d <= m; got d=%d, m=%d" % (d, m))
        self.m = m
        self.d = d
        self.shear = float(shear)
        omega = np.zeros(m, dtype=np.float64)
        omega[:d] = default_frequencies(d) if freqs is None else np.asarray(freqs)[:d]
        self.omega = omega
        self._init_torus()
        self.name = name or ("torus_translation_m%d_d%d" % (m, d))
        self.coord_names = tuple("y%d" % (i + 1) for i in range(m))

    def step(self, states):
        states = np.asarray(states, dtype=np.float64)
        out = states + self.omega[None, :]
        if self.shear != 0.0 and self.d >= 2:
            # Couple each moving coordinate to the previous one; leaves the
            # frozen coordinates frozen.
            out[:, 1:self.d] += self.shear * states[:, 0:self.d - 1]
        return np.mod(out, 1.0)

    def jacobian(self, states):
        """Tangent map, shape ``(k, m, m)``.

        A rigid translation has the identity as its tangent map; ``shear`` adds
        the sub-diagonal coupling among the moving coordinates.
        """
        states = np.asarray(states, dtype=np.float64)
        k = states.shape[0]
        jac = np.broadcast_to(np.eye(self.m), (k, self.m, self.m)).copy()
        if self.shear != 0.0 and self.d >= 2:
            rows = np.arange(1, self.d)
            jac[:, rows, rows - 1] += self.shear
        return jac


def make_system(kind, **kwargs):
    """Small factory so experiment scripts can name systems in a config block."""
    kinds = {
        "coupled_standard_map": CoupledStandardMap,
        "torus_translation": TorusTranslation,
    }
    if kind not in kinds:
        raise ValueError("unknown system %r; available: %r" % (kind, sorted(kinds)))
    return kinds[kind](**kwargs)
