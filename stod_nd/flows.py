# -*- coding: utf-8 -*-
"""
Continuous-time test systems.
============================

Why flows and not just maps
---------------------------
The published 2-D runs integrate with ``dt = 0.001`` on a 1000-cell axis, so a
trajectory spends several consecutive levels in the same cell and moves to a
*neighbouring* cell when it moves at all.  The path is an oversampled continuous
curve through the grid.

That is not incidental.  A cancellation happens when one trajectory's coordinate
value reappears in the other's history, and for continuous curves that is
governed by whether the two curves sweep overlapping ranges -- a geometric
property of how they separate.  In a map, every level lands on an essentially
uncorrelated cell, so cancellations become a birthday-problem coincidence
instead: with ``L`` levels on an ``N``-cell axis, the per-coordinate collision
probability is about ``1 - (1 - L/N)^L``, which is close to 1 for any usable
``L``.  Termination then fires almost immediately for every pair at every ``m``,
and the field carries no information.

So the ``m``-scaling question has to be asked with flows, where the mechanism the
published indicator actually relies on is present.  The maps in
:mod:`stod_nd.systems` remain useful as the contrasting case, and that contrast is
a result in its own right: FinSTOD needs trajectory continuity in the grid, not
merely a sequence of cell labels.

Systems provided
----------------
``Lorenz96``
    ``y_i' = (y_{i+1} - y_{i-2}) * y_{i-1} - y_i + F`` on a ring of ``m`` sites.
    The dimension is a free parameter, which is exactly what a dimension-scaling
    study needs: the same physics at ``m = 4, 5, 6, ..., 12``, so the only thing
    varying is ``m``.  It is a standard high-dimensional chaos benchmark, and ``F``
    tunes it from decaying (``F < ~0.9``) through periodic to strongly chaotic
    (``F = 8``).  Dissipative.

``CoupledDuffing``
    A chain of ``d`` forced Duffing oscillators, ``m = 2d``.  The direct
    continuation of the paper's own Duffing system, so the ``m = 2`` case *is* the
    published system and the higher ``m`` cases add oscillators to it.

``HenonHeiles``
    Hénon-Heiles at 2 or 3 degrees of freedom, ``m = 4`` or ``6``.  Autonomous and
    Hamiltonian, with the 3-DoF case being the standard setting for
    higher-dimensional invariant-manifold work, so results here are comparable
    against an existing literature.
"""

from __future__ import annotations

import numpy as np

from .base import GridSystem


class Flow(GridSystem):
    """A flow discretised onto a grid, advanced by fixed-step RK4.

    ``step`` advances by ``dt_out``, internally taking ``n_substeps`` RK4 steps.
    Separating the two matters: ``dt_out`` sets how finely the path samples the
    grid (the oversampling factor that gives the cancellation mechanism its
    geometry), while ``n_substeps`` sets the integration accuracy.  They are
    conflated in the published pipeline, where the output spacing equals the
    integration step.

    Fixed-step RK4 rather than the pipeline's adaptive RKF45: paths must be
    sampled at uniform times so that level ``k`` means the same thing for a cell
    and for its neighbours, which is what makes the level-by-level comparison
    meaningful.  An adaptive integrator with per-trajectory step sizes would
    break that, and the pipeline sidesteps it by emitting at fixed spacing anyway.
    """

    def __init__(self, dt_out=0.05, n_substeps=5, direction=1.0,
                 escape_factor=4.0):
        self.dt_out = float(dt_out)
        self.n_substeps = int(n_substeps)
        if self.n_substeps < 1:
            raise ValueError("n_substeps must be >= 1")
        self.direction = float(direction)
        self.escape_factor = float(escape_factor)

    def velocity(self, states):  # pragma: no cover - abstract
        raise NotImplementedError

    def velocity_jacobian(self, states):  # pragma: no cover - abstract
        """``d f_i / d y_j`` at each state, shape ``(k, m, m)``."""
        raise NotImplementedError

    def _escaped(self, y):
        """Which states have left the analysis box by more than ``escape_factor``.

        Escaping trajectories are frozen rather than integrated further.  This is
        a numerical guard, but it also *preserves* a mechanism the published
        pipeline depends on.  ``discretize_path`` deliberately does not clamp cell
        indices, so a trajectory that leaves the domain keeps a distinct
        visitation pattern far outside ``[0, N)`` and therefore cancels nothing
        against an in-domain neighbour -- which is where UU pairs come from.
        Clamping the state would destroy that; freezing it keeps a distinct
        out-of-domain signature while preventing the overflow to non-finite values
        that unbounded systems otherwise reach in a few hundred levels.
        """
        margin = self.escape_factor * self.span
        return (~np.isfinite(y).all(axis=1)
                | np.any(y < (self.lo - margin)[None, :], axis=1)
                | np.any(y > (self.hi + margin)[None, :], axis=1))

    def _rk4(self, y, h):
        """Non-autonomous RK4: if the system has ``_t``, stages use t, t+h/2, t+h.

        CoupledDuffing and the Bickley jet are time-dependent.  Evaluating every
        stage at the same frozen ``_t`` makes the forcing a constant and makes
        variational FTLE a different dynamical system from FinSTOD (which goes
        through ``step``).  Autonomous flows have no ``_t`` and are unchanged.
        """
        t0 = getattr(self, "_t", None)
        k1 = self.velocity(y)
        if t0 is not None:
            self._t = t0 + 0.5 * h
        k2 = self.velocity(y + 0.5 * h * k1)
        k3 = self.velocity(y + 0.5 * h * k2)
        if t0 is not None:
            self._t = t0 + h
        k4 = self.velocity(y + h * k3)
        return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step(self, states):
        y = np.asarray(states, dtype=np.float64)
        h = self.direction * self.dt_out / self.n_substeps
        for _ in range(self.n_substeps):
            frozen = self._escaped(y)
            if frozen.all():
                break
            with np.errstate(over="ignore", invalid="ignore"):
                advanced = self._rk4(y, h)
            # A step that produced a non-finite state is rejected outright, so the
            # trajectory stays at its last well-defined position.
            keep = frozen | ~np.isfinite(advanced).all(axis=1)
            y = np.where(keep[:, None], y, advanced)
        return y

    def escaped_fraction(self, cells, n_cells, n_levels):
        """Fraction of trajectories that leave the box within ``n_levels``.

        Worth reporting alongside any result: escapees are exactly the pairs that
        cancel nothing and come out as UU.
        """
        states = self.cell_to_state(cells, n_cells)
        for _ in range(int(n_levels)):
            states = self.step(states)
        return float(self._escaped(states).mean())


class Lorenz96(Flow):
    """Lorenz-96 on a ring of ``m`` sites; ``m`` is a free parameter.

    ``F = 8`` is the standard strongly chaotic setting.  The attractor lives
    roughly in ``[-10, 15]`` per coordinate for that forcing, which is the default
    analysis box.
    """

    def __init__(self, m, F=8.0, dt_out=0.02, n_substeps=4, direction=1.0,
                 box=(-10.0, 15.0), name=None):
        m = int(m)
        if m < 4:
            raise ValueError("Lorenz-96 needs m >= 4 for the ring to make sense")
        Flow.__init__(self, dt_out=dt_out, n_substeps=n_substeps,
                      direction=direction)
        self.m = m
        self.F = float(F)
        self._init_box(box[0], box[1])
        self.name = name or ("lorenz96_m%d_F%g" % (m, F))
        self.coord_names = tuple("y%d" % (i + 1) for i in range(m))

    def velocity(self, states):
        y = np.asarray(states, dtype=np.float64)
        return ((np.roll(y, -1, axis=1) - np.roll(y, 2, axis=1))
                * np.roll(y, 1, axis=1) - y + self.F)

    def velocity_jacobian(self, states):
        """Analytic velocity Jacobian, ``(k, m, m)``.

        ``d f_i / d y_j`` is nonzero only for ``j`` in ``{i-2, i-1, i+1, i}``::

            d f_i / d y_{i-2} = -y_{i-1}
            d f_i / d y_{i-1} = y_{i+1} - y_{i-2}
            d f_i / d y_{i+1} = y_{i-1}
            d f_i / d y_i     = -1
        """
        y = np.asarray(states, dtype=np.float64)
        k, m = y.shape
        i = np.arange(m)
        jac = np.zeros((k, m, m), dtype=np.float64)
        y_m1 = np.roll(y, 1, axis=1)     # y_{i-1}
        y_m2 = np.roll(y, 2, axis=1)     # y_{i-2}
        y_p1 = np.roll(y, -1, axis=1)    # y_{i+1}

        jac[:, i, (i - 2) % m] += -y_m1
        jac[:, i, (i - 1) % m] += y_p1 - y_m2
        jac[:, i, (i + 1) % m] += y_m1
        jac[:, i, i] += -1.0
        return jac

    def reference_state(self):
        """The usual Lorenz-96 reference point: uniform ``F`` with one site nudged."""
        y = np.full(self.m, self.F, dtype=np.float64)
        y[0] += 0.01
        return y


class IntegrableTorusFlow(Flow):
    """An integrable flow whose orbits fill tori of a *prescribed* dimension ``d``.

    The first ``d`` coordinates are angles and the remaining ``m - d`` are actions,
    frozen for all time::

        theta_i' = omega_i + twist * sum(actions)      i < d
        J_i'     = 0                                   i >= d

    Every orbit therefore lies on a ``d``-dimensional torus embedded in the
    ``m``-dimensional grid, and ``d`` is known exactly rather than inferred.  The
    ``twist`` term makes the frequencies depend on the actions, which is what makes
    neighbouring tori separate: with ``twist = 0`` the flow is a rigid translation
    and neighbours stay exactly parallel forever, which is a degenerate special
    case rather than the generic integrable picture.

    This is the calibration system for the ``q``-versus-torus-dimension question in
    ``experiments/exp03_torus_dimension.py``.  If ``STOD_q`` carries information
    about the dimension of the invariant structure -- the property that makes
    ``GALI_k`` useful in higher dimensions -- then sweeping ``q`` at known ``d``
    should show it here, in a system where there is no ambiguity about the answer.

    Periodic, so there is no boundary and no escape.
    """

    periodic = True

    def __init__(self, m, d, freqs=None, twist=0.35, dt_out=0.002,
                 n_substeps=1, direction=1.0, basis="identity", name=None):
        m = int(m)
        d = int(d)
        if not 1 <= d <= m:
            raise ValueError("need 1 <= d <= m; got d=%d, m=%d" % (d, m))
        Flow.__init__(self, dt_out=dt_out, n_substeps=n_substeps,
                      direction=direction)
        self.m = m
        self.d = d
        self.twist = float(twist)
        self._init_box(0.0, 1.0)
        if freqs is None:
            freqs = _default_frequencies(d)
        self.omega = np.zeros(m, dtype=np.float64)
        self.omega[:d] = np.asarray(freqs, dtype=np.float64)[:d]

        # ``basis`` decides whether the invariant tori are aligned with the grid
        # axes.  Alignment is a strong and unrealistic special case: it makes each
        # frozen action *literally* a grid coordinate.  A non-aligned basis is the
        # honest test of any dimension estimator built on axis-aligned neighbours.
        #
        # The change of basis has to be an integer unimodular matrix, so that it is
        # a genuine automorphism of the torus and both it and its inverse are
        # well defined mod 1.
        self.basis = basis
        self.A, self.Ainv = _torus_basis(m, basis)
        self.axis_aligned = (basis == "identity")

        self.name = name or ("integrable_torus_m%d_d%d_%s" % (m, d, basis))
        if self.axis_aligned:
            self.coord_names = (tuple("th%d" % (i + 1) for i in range(d))
                                + tuple("J%d" % (i + 1) for i in range(m - d)))
        else:
            self.coord_names = tuple("y%d" % (i + 1) for i in range(m))

    def _internal(self, y):
        """Angle/action coordinates ``z = A^{-1} y`` (mod 1)."""
        if self.axis_aligned:
            return y
        return np.mod(y.dot(self.Ainv.T), 1.0)

    def velocity(self, states):
        y = np.asarray(states, dtype=np.float64)
        z = self._internal(y)
        rate = np.zeros_like(y)
        rate[:, :self.d] = self.omega[None, :self.d]
        if self.twist != 0.0 and self.d < self.m:
            actions = z[:, self.d:].sum(axis=1)
            rate[:, :self.d] += self.twist * actions[:, None]
        if self.axis_aligned:
            return rate
        return rate.dot(self.A.T)

    def velocity_jacobian(self, states):
        y = np.asarray(states, dtype=np.float64)
        k = y.shape[0]
        inner = np.zeros((k, self.m, self.m), dtype=np.float64)
        if self.twist != 0.0 and self.d < self.m:
            inner[:, :self.d, self.d:] = self.twist
        if self.axis_aligned:
            return inner
        # d(A rate(A^-1 y))/dy = A (d rate/dz) A^-1
        return np.matmul(self.A[None, :, :], np.matmul(inner, self.Ainv[None, :, :]))

    def step(self, states):
        # The rate depends only on the frozen actions, so the exact solution is
        # available and RK4 would be wasted effort.  Wrapping keeps the state on
        # the torus, which the generic Flow.step does not do.
        y = np.asarray(states, dtype=np.float64)
        h = self.direction * self.dt_out
        return np.mod(y + h * self.velocity(y), 1.0)

    def true_dimension(self):
        return self.d


def _torus_basis(m, basis):
    """An integer unimodular change of basis on the ``m``-torus, and its inverse.

    Built as a product of elementary shears ``I + e_i e_j^T``, each of which has
    determinant 1 and an integer inverse, so the product and its inverse are both
    integer matrices and the map is an automorphism of the torus.
    """
    eye = np.eye(m, dtype=np.float64)
    if basis == "identity":
        return eye, eye
    if basis not in ("mixed", "strong"):
        raise ValueError("basis must be 'identity', 'mixed' or 'strong'")

    shears = [(i, (i + 1) % m) for i in range(m)]
    if basis == "strong":
        shears += [(i, (i + 2) % m) for i in range(m)]

    A = eye.copy()
    for i, j in shears:
        if i == j:
            continue
        elementary = eye.copy()
        elementary[i, j] = 1.0
        A = elementary.dot(A)

    Ainv = np.linalg.inv(A)
    Ainv_int = np.rint(Ainv)
    if not np.allclose(Ainv, Ainv_int, atol=1e-8):
        raise RuntimeError("basis inverse is not integral; not a torus automorphism")
    return A, Ainv_int


_PRIMES_FLOW = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _default_frequencies(d):
    """``d`` distinct irrational frequencies, scaled to order one."""
    if d > len(_PRIMES_FLOW):
        raise ValueError("only %d built-in frequencies available" % len(_PRIMES_FLOW))
    return np.array([1.0 + (np.sqrt(p) % 1.0) for p in _PRIMES_FLOW[:d]],
                    dtype=np.float64)


class CoupledDuffing(Flow):
    """A ring of ``d`` forced Duffing oscillators; ``m = 2d``.

    Coordinates are blocked as ``(x_1..x_d, v_1..v_d)``::

        x_j' = v_j
        v_j' = -delta v_j - alpha x_j - beta x_j^3
               + gamma cos(omega t) + kappa (x_{j+1} - 2 x_j + x_{j-1})

    At ``d = 1`` with ``kappa = 0`` and the paper's parameters
    (``alpha=-1, beta=1, delta=0.3, gamma=0.5, omega=1.2``) this *is* the
    published Duffing system, so the ``m = 2`` end of any sweep is anchored to a
    system whose FinSTOD field is already documented.

    Non-autonomous, so the state carries time as an extra component internally;
    the grid coordinates are the ``2d`` physical ones only.
    """

    def __init__(self, d, alpha=-1.0, beta=1.0, delta=0.3, gamma=0.5,
                 omega=1.2, kappa=0.15, dt_out=0.02, n_substeps=4,
                 direction=1.0, box=(-2.0, 2.0), name=None):
        d = int(d)
        if d < 1:
            raise ValueError("d must be >= 1")
        Flow.__init__(self, dt_out=dt_out, n_substeps=n_substeps,
                      direction=direction)
        self.d = d
        self.m = 2 * d
        self.alpha, self.beta, self.delta = float(alpha), float(beta), float(delta)
        self.gamma, self.omega = float(gamma), float(omega)
        self.kappa = float(kappa) if d >= 2 else 0.0
        self._init_box(box[0], box[1])
        self.name = name or ("coupled_duffing_d%d" % d)
        self.coord_names = tuple(
            ["x%d" % (j + 1) for j in range(d)] + ["v%d" % (j + 1) for j in range(d)]
        )
        self._t = 0.0

    def reset(self):
        """Rewind the forcing phase to ``t = 0``.

        Called by :func:`stod_nd.fields.build_paths` before every batch.  Without
        it, a chunked field computation would integrate later chunks at a later
        forcing phase, so a cell's result would depend on which chunk it landed
        in.
        """
        self._t = 0.0

    def velocity(self, states):
        y = np.asarray(states, dtype=np.float64)
        d = self.d
        x = y[:, :d]
        v = y[:, d:]

        force = (-self.delta * v - self.alpha * x - self.beta * x ** 3
                 + self.gamma * np.cos(self.omega * self._t))
        if self.kappa != 0.0:
            force = force + self.kappa * (np.roll(x, -1, axis=1)
                                          - 2.0 * x
                                          + np.roll(x, 1, axis=1))
        out = np.empty_like(y)
        out[:, :d] = v
        out[:, d:] = force
        return out

    def velocity_jacobian(self, states):
        """``(k, m, m)``.  The forcing is state-independent, so it drops out.

        With the state ordered ``(x, v)``::

            d x'/d x = 0                d x'/d v = I
            d v'/d x = -alpha I - 3 beta diag(x^2) + kappa * ring Laplacian
            d v'/d v = -delta I
        """
        y = np.asarray(states, dtype=np.float64)
        k = y.shape[0]
        d = self.d
        x = y[:, :d]

        dv_dx = np.zeros((k, d, d), dtype=np.float64)
        j = np.arange(d)
        dv_dx[:, j, j] = -self.alpha - 3.0 * self.beta * x ** 2
        if self.kappa != 0.0:
            dv_dx[:, j, j] += -2.0 * self.kappa
            dv_dx[:, j, (j + 1) % d] += self.kappa
            dv_dx[:, j, (j - 1) % d] += self.kappa

        eye = np.eye(d)[None, :, :]
        jac = np.zeros((k, 2 * d, 2 * d), dtype=np.float64)
        jac[:, :d, d:] = eye
        jac[:, d:, :d] = dv_dx
        jac[:, d:, d:] = -self.delta * eye
        return jac

    def _rk4(self, y, h):
        # The forcing depends on absolute time, so the clock advances in lockstep
        # with the stages rather than the system being treated as autonomous.
        t0 = self._t
        k1 = self.velocity(y)
        self._t = t0 + 0.5 * h
        k2 = self.velocity(y + 0.5 * h * k1)
        k3 = self.velocity(y + 0.5 * h * k2)
        self._t = t0 + h
        k4 = self.velocity(y + h * k3)
        return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class HenonHeiles(Flow):
    """Hénon-Heiles at ``dof`` degrees of freedom; ``m = 2 * dof``.

    Coordinates are blocked as ``(q_1..q_n, p_1..p_n)`` with

        H = 1/2 sum p_i^2 + 1/2 sum q_i^2 + lam * sum_i q_i^2 q_{i+1} ...

    For ``dof = 2`` this is the classical Hénon-Heiles Hamiltonian

        H = 1/2 (p1^2 + p2^2) + 1/2 (q1^2 + q2^2) + lam (q1^2 q2 - q2^3 / 3),

    and for ``dof = 3`` the standard chain extension is used, which is the setting
    most of the higher-dimensional invariant-manifold literature works in.
    Autonomous and Hamiltonian, so unlike Lorenz-96 there is no attractor and the
    dynamics is volume-preserving.
    """

    def __init__(self, dof=2, lam=1.0, dt_out=0.02, n_substeps=4,
                 direction=1.0, box=(-0.9, 0.9), energy_max=None, name=None):
        dof = int(dof)
        if dof not in (2, 3):
            raise ValueError("dof must be 2 or 3")
        Flow.__init__(self, dt_out=dt_out, n_substeps=n_substeps,
                      direction=direction)
        self.dof = dof
        self.m = 2 * dof
        self.lam = float(lam)
        self._init_box(box[0], box[1])
        # Above the escape energy 1/(6 lam^2) the Henon-Heiles potential opens and
        # trajectories run off to infinity, which produces non-finite states
        # rather than a bounded visitation pattern.  Seed cells are restricted to
        # the bound region.
        self.escape_energy = 1.0 / (6.0 * self.lam ** 2)
        self.energy_max = (0.95 * self.escape_energy if energy_max is None
                           else float(energy_max))
        self.name = name or ("henon_heiles_dof%d" % dof)
        self.coord_names = tuple(
            ["q%d" % (j + 1) for j in range(dof)] + ["p%d" % (j + 1) for j in range(dof)]
        )

    def valid_cells(self, cells, n_cells):
        states = self.cell_to_state(cells, n_cells)
        return self.energy(states) <= self.energy_max

    def _dV_dq(self, q):
        """Gradient of the potential."""
        lam = self.lam
        if self.dof == 2:
            q1, q2 = q[:, 0], q[:, 1]
            out = np.empty_like(q)
            out[:, 0] = q1 + 2.0 * lam * q1 * q2
            out[:, 1] = q2 + lam * (q1 ** 2 - q2 ** 2)
            return out
        q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2]
        out = np.empty_like(q)
        out[:, 0] = q1 + 2.0 * lam * q1 * q2
        out[:, 1] = q2 + lam * (q1 ** 2 - q2 ** 2) + 2.0 * lam * q2 * q3
        out[:, 2] = q3 + lam * (q2 ** 2 - q3 ** 2)
        return out

    def velocity(self, states):
        y = np.asarray(states, dtype=np.float64)
        n = self.dof
        q = y[:, :n]
        p = y[:, n:]
        out = np.empty_like(y)
        out[:, :n] = p
        out[:, n:] = -self._dV_dq(q)
        return out

    def velocity_jacobian(self, states):
        """``(k, m, m)``, built from the Hessian of the potential.

            d q'/d q = 0        d q'/d p = I
            d p'/d q = -Hess V  d p'/d p = 0
        """
        y = np.asarray(states, dtype=np.float64)
        k = y.shape[0]
        n = self.dof
        q = y[:, :n]
        lam = self.lam

        hess = np.zeros((k, n, n), dtype=np.float64)
        if n == 2:
            q1, q2 = q[:, 0], q[:, 1]
            hess[:, 0, 0] = 1.0 + 2.0 * lam * q2
            hess[:, 0, 1] = 2.0 * lam * q1
            hess[:, 1, 0] = 2.0 * lam * q1
            hess[:, 1, 1] = 1.0 - 2.0 * lam * q2
        else:
            q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2]
            hess[:, 0, 0] = 1.0 + 2.0 * lam * q2
            hess[:, 0, 1] = 2.0 * lam * q1
            hess[:, 1, 0] = 2.0 * lam * q1
            hess[:, 1, 1] = 1.0 - 2.0 * lam * q2 + 2.0 * lam * q3
            hess[:, 1, 2] = 2.0 * lam * q2
            hess[:, 2, 1] = 2.0 * lam * q2
            hess[:, 2, 2] = 1.0 - 2.0 * lam * q3

        jac = np.zeros((k, 2 * n, 2 * n), dtype=np.float64)
        jac[:, :n, n:] = np.eye(n)[None, :, :]
        jac[:, n:, :n] = -hess
        return jac

    def energy(self, states):
        """Total energy, for checking that the integrator is behaving."""
        y = np.asarray(states, dtype=np.float64)
        n = self.dof
        q, p = y[:, :n], y[:, n:]
        lam = self.lam
        kinetic = 0.5 * np.sum(p ** 2, axis=1)
        harmonic = 0.5 * np.sum(q ** 2, axis=1)
        if self.dof == 2:
            cubic = lam * (q[:, 0] ** 2 * q[:, 1] - q[:, 1] ** 3 / 3.0)
        else:
            cubic = lam * (q[:, 0] ** 2 * q[:, 1] - q[:, 1] ** 3 / 3.0
                           + q[:, 1] ** 2 * q[:, 2] - q[:, 2] ** 3 / 3.0)
        return kinetic + harmonic + cubic


class ABCFlow(Flow):
    """Arnold-Beltrami-Childress (ABC) exact 3D Euler flow.

    Autonomous flow on the 3-torus :math:`[0, 2\\pi]^3`:
        x' = A sin(z) + C cos(y)
        y' = B sin(x) + A cos(z)
        z' = C sin(y) + B cos(x)

    A foundational benchmark in fluid mechanics and Lagrangian Coherent Structure (LCS)
    theory. Exhibits helical KAM vortex tubes embedded in chaotic streamline webs.
    """

    def __init__(self, A=1.0, B=1.0, C=1.0, dt_out=0.02, n_substeps=4,
                 direction=1.0, name=None):
        Flow.__init__(self, dt_out=dt_out, n_substeps=n_substeps,
                      direction=direction, escape_factor=1.0)
        self.m = 3
        self.A = float(A)
        self.B = float(B)
        self.C = float(C)
        self._init_box(0.0, 2.0 * np.pi)
        self.periodic = True
        self.name = name or ("abc_flow_A%.1f_B%.1f_C%.1f" % (A, B, C))
        self.coord_names = ("x", "y", "z")

    def velocity(self, states):
        y = np.asarray(states, dtype=np.float64)
        x_c, y_c, z_c = y[:, 0], y[:, 1], y[:, 2]
        out = np.empty_like(y)
        out[:, 0] = self.A * np.sin(z_c) + self.C * np.cos(y_c)
        out[:, 1] = self.B * np.sin(x_c) + self.A * np.cos(z_c)
        out[:, 2] = self.C * np.sin(y_c) + self.B * np.cos(x_c)
        return out

    def velocity_jacobian(self, states):
        y = np.asarray(states, dtype=np.float64)
        k = y.shape[0]
        x_c, y_c, z_c = y[:, 0], y[:, 1], y[:, 2]
        jac = np.zeros((k, 3, 3), dtype=np.float64)
        jac[:, 0, 1] = -self.C * np.sin(y_c)
        jac[:, 0, 2] = self.A * np.cos(z_c)
        jac[:, 1, 0] = self.B * np.cos(x_c)
        jac[:, 1, 2] = -self.A * np.sin(z_c)
        jac[:, 2, 0] = -self.B * np.sin(x_c)
        jac[:, 2, 1] = self.C * np.cos(y_c)
        return jac


class Froeschle(Flow):
    """Coupled Symplectic Flow / Froeschlé Hamiltonian system.

    Autonomous Hamiltonian on :math:`m = 2d` phase space with :math:`d` degrees of freedom:
        H = 1/2 sum p_i^2 - sum cos(q_i) - K sum_{i=1}^{d-1} cos(q_{i+1} - q_i)

    Canonical benchmark for high-dimensional KAM islands, resonance junctions, and
    Arnold diffusion in symplectic systems.
    """

    def __init__(self, dof=2, K=0.3, dt_out=0.02, n_substeps=4,
                 direction=1.0, box_p=(-3.0, 3.0), name=None):
        Flow.__init__(self, dt_out=dt_out, n_substeps=n_substeps,
                      direction=direction, escape_factor=2.0)
        self.dof = int(dof)
        self.m = 2 * self.dof
        self.K = float(K)
        lo = np.concatenate([np.full(self.dof, -np.pi), np.full(self.dof, box_p[0])])
        hi = np.concatenate([np.full(self.dof, np.pi), np.full(self.dof, box_p[1])])
        self._init_box(lo, hi)
        self.name = name or ("froeschle_dof%d_K%.2f" % (self.dof, self.K))
        self.coord_names = tuple(
            ["q%d" % (j + 1) for j in range(self.dof)] + ["p%d" % (j + 1) for j in range(self.dof)]
        )

    def velocity(self, states):
        y = np.asarray(states, dtype=np.float64)
        n = self.dof
        q, p = y[:, :n], y[:, n:]
        out = np.empty_like(y)
        out[:, :n] = p

        # Forces -dV/dq
        forces = -np.sin(q)
        if n > 1 and self.K != 0:
            diff = q[:, 1:] - q[:, :-1]  # q_{i+1} - q_i
            sin_diff = self.K * np.sin(diff)
            forces[:, :-1] += sin_diff
            forces[:, 1:] -= sin_diff

        out[:, n:] = forces
        return out

    def velocity_jacobian(self, states):
        y = np.asarray(states, dtype=np.float64)
        k = y.shape[0]
        n = self.dof
        q = y[:, :n]

        hess = np.zeros((k, n, n), dtype=np.float64)
        for i in range(n):
            hess[:, i, i] = np.cos(q[:, i])

        if n > 1 and self.K != 0:
            diff = q[:, 1:] - q[:, :-1]
            cos_diff = self.K * np.cos(diff)
            for i in range(n - 1):
                hess[:, i, i] += cos_diff[:, i]
                hess[:, i + 1, i + 1] += cos_diff[:, i]
                hess[:, i, i + 1] -= cos_diff[:, i]
                hess[:, i + 1, i] -= cos_diff[:, i]

        jac = np.zeros((k, 2 * n, 2 * n), dtype=np.float64)
        jac[:, :n, n:] = np.eye(n)[None, :, :]
        jac[:, n:, :n] = -hess
        return jac

    def energy(self, states):
        y = np.asarray(states, dtype=np.float64)
        n = self.dof
        q, p = y[:, :n], y[:, n:]
        kinetic = 0.5 * np.sum(p ** 2, axis=1)
        pot = -np.sum(np.cos(q), axis=1)
        if n > 1 and self.K != 0:
            pot -= self.K * np.sum(np.cos(q[:, 1:] - q[:, :-1]), axis=1)
        return kinetic + pot


class CR3BP(Flow):
    """Spatial Circular Restricted 3-Body Problem (m=6).

    Coordinates: :math:`(x, y, z, p_x, p_y, p_z)` in the rotating barycentric frame.
    Governs spacecraft and celestial dynamics in the gravitational field of two primaries.
    Resolves Lagrange point invariant manifolds and spatial halo/Lyapunov orbit tubes.
    """

    def __init__(self, mu=0.01215, dt_out=0.01, n_substeps=4,
                 direction=1.0, box_pos=(-1.8, 1.8), box_vel=(-1.8, 1.8),
                 eps=1e-3, name=None):
        Flow.__init__(self, dt_out=dt_out, n_substeps=n_substeps,
                      direction=direction, escape_factor=2.0)
        self.m = 6
        self.mu = float(mu)
        self.eps2 = float(eps ** 2)
        lo = np.array([box_pos[0], box_pos[0], box_pos[0], box_vel[0], box_vel[0], box_vel[0]])
        hi = np.array([box_pos[1], box_pos[1], box_pos[1], box_vel[1], box_vel[1], box_vel[1]])
        self._init_box(lo, hi)
        self.name = name or ("cr3bp_mu%.5f" % self.mu)
        self.coord_names = ("x", "y", "z", "px", "py", "pz")

    def velocity(self, states):
        y = np.asarray(states, dtype=np.float64)
        x, y_pos, z = y[:, 0], y[:, 1], y[:, 2]
        px, py, pz = y[:, 3], y[:, 4], y[:, 5]
        mu = self.mu

        r1_sq = (x + mu) ** 2 + y_pos ** 2 + z ** 2 + self.eps2
        r2_sq = (x - 1.0 + mu) ** 2 + y_pos ** 2 + z ** 2 + self.eps2
        r1_3 = r1_sq ** 1.5
        r2_3 = r2_sq ** 1.5

        out = np.empty_like(y)
        out[:, 0] = px + y_pos
        out[:, 1] = py - x
        out[:, 2] = pz

        gx = (1.0 - mu) * (x + mu) / r1_3 + mu * (x - 1.0 + mu) / r2_3
        gy = (1.0 - mu) * y_pos / r1_3 + mu * y_pos / r2_3
        gz = (1.0 - mu) * z / r1_3 + mu * z / r2_3

        out[:, 3] = py - gx
        out[:, 4] = -px - gy
        out[:, 5] = -gz
        return out

    def velocity_jacobian(self, states):
        y = np.asarray(states, dtype=np.float64)
        k = y.shape[0]
        x, y_pos, z = y[:, 0], y[:, 1], y[:, 2]
        mu = self.mu

        r1_sq = (x + mu) ** 2 + y_pos ** 2 + z ** 2 + self.eps2
        r2_sq = (x - 1.0 + mu) ** 2 + y_pos ** 2 + z ** 2 + self.eps2
        r1_3 = r1_sq ** 1.5
        r2_3 = r2_sq ** 1.5
        r1_5 = r1_sq ** 2.5
        r2_5 = r2_sq ** 2.5

        dx1 = x + mu
        dx2 = x - 1.0 + mu

        g_xx = (1.0 - mu) * (1.0 / r1_3 - 3.0 * dx1 ** 2 / r1_5) + mu * (1.0 / r2_3 - 3.0 * dx2 ** 2 / r2_5)
        g_yy = (1.0 - mu) * (1.0 / r1_3 - 3.0 * y_pos ** 2 / r1_5) + mu * (1.0 / r2_3 - 3.0 * y_pos ** 2 / r2_5)
        g_zz = (1.0 - mu) * (1.0 / r1_3 - 3.0 * z ** 2 / r1_5) + mu * (1.0 / r2_3 - 3.0 * z ** 2 / r2_5)
        g_xy = -(1.0 - mu) * 3.0 * dx1 * y_pos / r1_5 - mu * 3.0 * dx2 * y_pos / r2_5
        g_xz = -(1.0 - mu) * 3.0 * dx1 * z / r1_5 - mu * 3.0 * dx2 * z / r2_5
        g_yz = -(1.0 - mu) * 3.0 * y_pos * z / r1_5 - mu * 3.0 * y_pos * z / r2_5

        jac = np.zeros((k, 6, 6), dtype=np.float64)
        jac[:, 0, 1] = 1.0
        jac[:, 0, 3] = 1.0
        jac[:, 1, 0] = -1.0
        jac[:, 1, 4] = 1.0
        jac[:, 2, 5] = 1.0

        jac[:, 3, 0] = -g_xx
        jac[:, 3, 1] = -g_xy
        jac[:, 3, 2] = -g_xz
        jac[:, 3, 4] = 1.0

        jac[:, 4, 0] = -g_xy
        jac[:, 4, 1] = -g_yy
        jac[:, 4, 2] = -g_yz
        jac[:, 4, 3] = -1.0

        jac[:, 5, 0] = -g_xz
        jac[:, 5, 1] = -g_yz
        jac[:, 5, 2] = -g_zz
        return jac


class FPUT(Flow):
    """Fermi-Pasta-Ulam-Tsingou (FPUT) non-linear lattice chain (m = 2d).

    A chain of :math:`d` non-linearly coupled oscillators:
        H = 1/2 sum p_i^2 + sum [ 1/2 (q_{i+1} - q_i)^2 + alpha/3 (q_{i+1} - q_i)^3 + beta/4 (q_{i+1} - q_i)^4 ]

    Foundational for understanding energy localization, quasiperiodicity, and the onset of
    equipartition in high-dimensional non-linear physics.
    """

    def __init__(self, dof=4, alpha=0.25, beta=0.1, dt_out=0.02, n_substeps=4,
                 direction=1.0, box_q=(-2.0, 2.0), box_p=(-2.0, 2.0), name=None):
        Flow.__init__(self, dt_out=dt_out, n_substeps=n_substeps,
                      direction=direction, escape_factor=2.0)
        self.dof = int(dof)
        self.m = 2 * self.dof
        self.alpha = float(alpha)
        self.beta = float(beta)
        lo = np.concatenate([np.full(self.dof, box_q[0]), np.full(self.dof, box_p[0])])
        hi = np.concatenate([np.full(self.dof, box_q[1]), np.full(self.dof, box_p[1])])
        self._init_box(lo, hi)
        self.name = name or ("fput_dof%d_a%.2f_b%.2f" % (self.dof, self.alpha, self.beta))
        self.coord_names = tuple(
            ["q%d" % (j + 1) for j in range(self.dof)] + ["p%d" % (j + 1) for j in range(self.dof)]
        )

    def velocity(self, states):
        y = np.asarray(states, dtype=np.float64)
        n = self.dof
        q, p = y[:, :n], y[:, n:]
        out = np.empty_like(y)
        out[:, :n] = p

        # Periodic boundaries on chain
        q_prev = np.roll(q, 1, axis=1)
        q_next = np.roll(q, -1, axis=1)

        delta_next = q_next - q
        delta_prev = q - q_prev

        # Linear + Alpha + Beta nonlinear forces
        force = (delta_next - delta_prev
                 + self.alpha * (delta_next ** 2 - delta_prev ** 2)
                 + self.beta * (delta_next ** 3 - delta_prev ** 3))

        out[:, n:] = force
        return out

    def velocity_jacobian(self, states):
        y = np.asarray(states, dtype=np.float64)
        k = y.shape[0]
        n = self.dof
        q = y[:, :n]

        q_prev = np.roll(q, 1, axis=1)
        q_next = np.roll(q, -1, axis=1)
        delta_next = q_next - q
        delta_prev = q - q_prev

        k_next = 1.0 + 2.0 * self.alpha * delta_next + 3.0 * self.beta * delta_next ** 2
        k_prev = 1.0 + 2.0 * self.alpha * delta_prev + 3.0 * self.beta * delta_prev ** 2

        hess = np.zeros((k, n, n), dtype=np.float64)
        for i in range(n):
            i_prev = (i - 1) % n
            i_next = (i + 1) % n
            hess[:, i, i] = k_next[:, i] + k_prev[:, i]
            hess[:, i, i_next] -= k_next[:, i]
            hess[:, i, i_prev] -= k_prev[:, i]

        jac = np.zeros((k, 2 * n, 2 * n), dtype=np.float64)
        jac[:, :n, n:] = np.eye(n)[None, :, :]
        jac[:, n:, :n] = -hess
        return jac

    def modal_energies(self, states):
        """Compute harmonic mode energies E_k for a batch of states."""
        y = np.asarray(states, dtype=np.float64)
        n = self.dof
        q, p = y[:, :n], y[:, n:]

        # Discrete Fourier Transform along spatial lattice
        Q_k = np.fft.rfft(q, axis=1) / np.sqrt(n)
        P_k = np.fft.rfft(p, axis=1) / np.sqrt(n)

        # Harmonic frequencies omega_k = 2 * sin(pi * k / n)
        k_indices = np.arange(Q_k.shape[1])
        omega_k = 2.0 * np.sin(np.pi * k_indices / float(n))

        # Modal harmonic energy E_k = 1/2 (|P_k|^2 + omega_k^2 |Q_k|^2)
        E_k = 0.5 * (np.abs(P_k) ** 2 + (omega_k ** 2)[None, :] * np.abs(Q_k) ** 2)
        return E_k

    def spectral_entropy(self, states):
        """Normalized spectral entropy S / ln(n_modes) in [0, 1].

        S = 0 represents complete energy trapping in a single mode.
        S = 1 represents full thermal equipartition across all modes.
        """
        E_k = self.modal_energies(states)
        E_tot = np.sum(E_k, axis=1, keepdims=True)
        E_tot = np.maximum(E_tot, 1e-300)
        p_k = E_k / E_tot

        with np.errstate(divide="ignore", invalid="ignore"):
            log_p = np.where(p_k > 1e-15, np.log(p_k), 0.0)
            entropy = -np.sum(p_k * log_p, axis=1)

        n_modes = E_k.shape[1]
        max_entropy = np.log(n_modes) if n_modes > 1 else 1.0
        return entropy / max_entropy


class Kuramoto(Flow):
    """Kuramoto Network of coupled phase oscillators on the m-torus.

    Governed by:
        theta_i' = omega_i + sum_j W_ij sin(theta_j - theta_i)

    Two coupling topologies are available.  ``coupling="global"`` is all-to-all
    with W_ij = K/m, the classical mean-field model.  ``coupling="ring"`` couples
    each oscillator to the ``R`` neighbours on each side of a periodic ring with
    W_ij = K/(2R); this is the topology that supports twisted states, chimeras
    and genuine multistability, and therefore has a phase space worth mapping.

    Regime matters.  For a uniform natural-frequency spread on [-w, w] the
    mean-field locking threshold is K_c = 4w/pi.  Well above K_c every initial
    condition collapses onto a single phase-locked attractor, the tangent
    dynamics is then identical everywhere, and the FTLE field is constant to
    machine precision -- an uninformative benchmark.  Choose K near or below K_c
    to retain a mixed phase space.
    """

    def __init__(self, m=8, K=2.0, natural_freqs=None, dt_out=0.02,
                 n_substeps=4, direction=1.0, name=None,
                 coupling="global", radius=None, freq_spread=1.0):
        Flow.__init__(self, dt_out=dt_out, n_substeps=n_substeps,
                      direction=direction, escape_factor=1.0)
        self.m = int(m)
        self.K = float(K)
        self.coupling = str(coupling)
        self.freq_spread = float(freq_spread)

        if natural_freqs is None:
            self.omega = np.linspace(-self.freq_spread, self.freq_spread, self.m)
        else:
            self.omega = np.asarray(natural_freqs, dtype=np.float64)
            if self.omega.shape != (self.m,):
                raise ValueError("natural_freqs must have length m=%d" % self.m)

        if self.coupling == "global":
            self.radius = self.m
            self.W = np.full((self.m, self.m), self.K / float(self.m))
        elif self.coupling == "ring":
            self.radius = int(radius if radius is not None else max(1, self.m // 4))
            if not (1 <= self.radius <= self.m // 2):
                raise ValueError("ring radius must lie in [1, m//2]")
            idx = np.arange(self.m)
            sep = np.abs(idx[None, :] - idx[:, None])
            sep = np.minimum(sep, self.m - sep)  # circular separation
            self.W = np.where(
                (sep > 0) & (sep <= self.radius),
                self.K / float(2 * self.radius),
                0.0,
            )
        else:
            raise ValueError("coupling must be 'global' or 'ring'")
        np.fill_diagonal(self.W, 0.0)

        self._init_box(-np.pi, np.pi)
        self.periodic = True
        if name:
            self.name = name
        elif self.coupling == "ring":
            self.name = "kuramoto_m%d_K%.2f_ringR%d" % (self.m, self.K, self.radius)
        else:
            self.name = "kuramoto_m%d_K%.2f" % (self.m, self.K)
        self.coord_names = tuple(["th%d" % (j + 1) for j in range(self.m)])

    @property
    def critical_coupling(self):
        """Mean-field locking threshold K_c = 4w/pi for a uniform spread [-w, w]."""
        return 4.0 * self.freq_spread / np.pi

    def order_parameter(self, states):
        """Kuramoto order parameter r in [0, 1]; 1 is full phase locking."""
        y = np.asarray(states, dtype=np.float64)
        return np.abs(np.mean(np.exp(1j * y), axis=-1))

    def velocity(self, states):
        y = np.asarray(states, dtype=np.float64)
        # Pairwise differences theta_j - theta_i: shape (k, m, m)
        diff = y[:, None, :] - y[:, :, None]
        coupling = np.einsum("ij,kij->ki", self.W, np.sin(diff))
        return self.omega[None, :] + coupling

    def velocity_jacobian(self, states):
        y = np.asarray(states, dtype=np.float64)
        k, m = y.shape
        diff = y[:, None, :] - y[:, :, None]
        jac = self.W[None, :, :] * np.cos(diff)

        # Diagonal: -sum_{j != i} W_ij cos(theta_j - theta_i). The stored W has a
        # zero diagonal, so the row sum already excludes j == i.
        diag = jac.sum(axis=2) - np.diagonal(jac, axis1=1, axis2=2)
        idx = np.arange(m)
        jac[:, idx, idx] = -diag
        return jac


class BickleyJetFlow(Flow):
    """Bickley Jet shear flow model for geophysical transport barriers.

    Two-dimensional incompressible flow on (x, y) with a central jet stream:
        u(x, y, t) = U sech^2(y/L) - 2 U sum_i eps_i sech^2(y/L) tanh(y/L) cos(k_i x - sigma_i t)
        v(x, y, t) = -U L sum_i eps_i k_i sech^2(y/L) sin(k_i x - sigma_i t)

    In the unperturbed limit (eps = 0), this is a pure shear flow with zero Lyapunov
    exponent (lambda = 0) everywhere, where FTLE fails to resolve transport barriers
    bounding the jet core while FinSTOD sharply detects shear inflection barriers.
    """

    def __init__(self, U=1.0, L=1.0, L_x=6.371, eps=(0.075, 0.15, 0.3),
                 c_ratios=(0.1446, 0.2053, 0.461), dt_out=0.05, n_substeps=4,
                 direction=1.0, box_x=(0.0, None), box_y=(-3.0, 3.0), name=None):
        Flow.__init__(self, dt_out=dt_out, n_substeps=n_substeps,
                      direction=direction, escape_factor=2.0)
        self.m = 2
        self.U = float(U)
        self.L = float(L)
        self.L_x = float(L_x)
        self.eps = np.asarray(eps, dtype=np.float64)
        self.c_ratios = np.asarray(c_ratios, dtype=np.float64)
        self.k_vec = np.array([2.0, 4.0, 6.0], dtype=np.float64) / self.L_x
        self.sigma_vec = self.k_vec * (self.c_ratios * self.U)

        xmax = float(np.pi * self.L_x) if box_x[1] is None else float(box_x[1])
        xmin = float(box_x[0])
        lo = np.array([xmin, float(box_y[0])], dtype=np.float64)
        hi = np.array([xmax, float(box_y[1])], dtype=np.float64)
        self._init_box(lo, hi)
        self.name = name or ("bickley_jet_U%.1f" % self.U)
        self.coord_names = ("x", "y")
        self._t = 0.0

    def reset(self):
        self._t = 0.0

    def velocity(self, states):
        y = np.asarray(states, dtype=np.float64)
        x_pos, y_pos = y[:, 0], y[:, 1]
        y_scaled = y_pos / self.L
        sech = 1.0 / np.cosh(np.clip(y_scaled, -50.0, 50.0))
        sech2 = sech ** 2
        tanh = np.tanh(y_scaled)

        u = self.U * sech2
        v = np.zeros_like(x_pos)

        if np.any(self.eps != 0.0):
            phases = self.k_vec[None, :] * x_pos[:, None] - self.sigma_vec[None, :] * self._t
            cos_p = np.cos(phases)
            sin_p = np.sin(phases)

            wave_u = -2.0 * self.U * sech2[:, None] * tanh[:, None] * self.eps[None, :] * cos_p
            wave_v = -self.U * self.L * sech2[:, None] * (self.eps * self.k_vec)[None, :] * sin_p
            u = u + np.sum(wave_u, axis=1)
            v = v + np.sum(wave_v, axis=1)

        out = np.empty_like(y)
        out[:, 0] = u
        out[:, 1] = v
        return out

    def velocity_jacobian(self, states):
        y = np.asarray(states, dtype=np.float64)
        k = y.shape[0]
        x_pos, y_pos = y[:, 0], y[:, 1]
        y_scaled = y_pos / self.L
        sech = 1.0 / np.cosh(np.clip(y_scaled, -50.0, 50.0))
        sech2 = sech ** 2
        tanh = np.tanh(y_scaled)

        du_dy = -2.0 * (self.U / self.L) * sech2 * tanh
        du_dx = np.zeros(k, dtype=np.float64)
        dv_dx = np.zeros(k, dtype=np.float64)
        dv_dy = np.zeros(k, dtype=np.float64)

        if np.any(self.eps != 0.0):
            phases = self.k_vec[None, :] * x_pos[:, None] - self.sigma_vec[None, :] * self._t
            cos_p = np.cos(phases)
            sin_p = np.sin(phases)

            du_dx_waves = 2.0 * self.U * sech2[:, None] * tanh[:, None] * (self.eps * self.k_vec)[None, :] * sin_p
            du_dx = np.sum(du_dx_waves, axis=1)

            d_sech2tanh = (1.0 / self.L) * sech2[:, None] * (1.0 - 3.0 * tanh[:, None] ** 2)
            du_dy_waves = -2.0 * self.U * d_sech2tanh * self.eps[None, :] * cos_p
            du_dy = du_dy + np.sum(du_dy_waves, axis=1)

            dv_dx_waves = -self.U * self.L * sech2[:, None] * (self.eps * (self.k_vec ** 2))[None, :] * cos_p
            dv_dx = np.sum(dv_dx_waves, axis=1)

            dv_dy = -du_dx

        jac = np.zeros((k, 2, 2), dtype=np.float64)
        jac[:, 0, 0] = du_dx
        jac[:, 0, 1] = du_dy
        jac[:, 1, 0] = dv_dx
        jac[:, 1, 1] = dv_dy
        return jac


