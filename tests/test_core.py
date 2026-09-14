# -*- coding: utf-8 -*-
"""
Correctness tests for the m-dimensional STOD implementation.
===========================================================

Runnable three ways:

    python tests/test_core.py          # from the project root
    pytest tests/test_core.py
    (open in Spyder and press F5)

The two tests that matter most are

``test_published_rule_matches_original_2d``
    The vectorised implementation, at ``m = 2``, ``q = 2``, ``rule="published"``,
    must reproduce the released ``sod_logic.py`` bit for bit.  Without this the
    new work is not continuous with the published results.

``test_vectorised_matches_reference``
    The closed-form reformulation in ``core.py`` must agree with the
    level-by-level simulation in ``reference.py`` for every ``(m, q, rule)``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import numpy as np

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # interactive paste
    _HERE = os.path.abspath(os.path.join(os.getcwd(), "tests"))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from stod_nd import (  # noqa: E402
    PUBLISHED,
    STRICT,
    TYPE_T,
    TYPE_UU,
    CoupledStandardMap,
    TorusTranslation,
    build_paths,
    evaluate_cells,
    path_locality,
    sample_cells,
    segmented_normalise,
    select_neighbourhood,
    slice_cells,
    stod_max_score,
    stod_pair,
    stod_pair_reference,
    von_neumann_offsets,
)

RULES = (STRICT, PUBLISHED)


# ----------------------------------------------------------------------------
# loading the published 2-D implementation as ground truth
# ----------------------------------------------------------------------------

def _load_original_module():
    """Import ``reference/pipeline_core/sod_logic_original.py``.

    That file imports numba at module scope purely to decorate the accelerated
    variant.  We only need its pure-Python function, so if numba is absent we
    stand in a no-op ``njit``.
    """
    if "numba" not in sys.modules:
        try:
            import numba  # noqa: F401
        except ImportError:
            stub = types.ModuleType("numba")

            def njit(*args, **kwargs):
                if len(args) == 1 and callable(args[0]) and not kwargs:
                    return args[0]

                def decorate(func):
                    return func

                return decorate

            stub.njit = njit
            sys.modules["numba"] = stub

    candidates = [
        os.path.join(_ROOT, "Original_FinSTOD_Paper_Folder", "generic_holistic_stod_pipeline_10", "pipeline_core", "sod_logic.py"),
        os.path.join(_ROOT, "reference", "pipeline_core", "sod_logic_original.py"),
        os.path.join(_ROOT, "reference", "pipeline_core", "sod_logic.py"),
    ]
    path = None
    for cand in candidates:
        if os.path.isfile(cand):
            path = cand
            break
    if path is None:
        raise FileNotFoundError("Could not find original sod_logic.py in %r" % (candidates,))
    spec = importlib.util.spec_from_file_location("sod_logic_original", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------
# input generators
# ----------------------------------------------------------------------------

def _random_paths(rng, n, m, n_cells):
    """Unstructured paths -- exercises the bookkeeping."""
    return (rng.integers(0, n_cells, size=(n, m), dtype=np.int64),
            rng.integers(0, n_cells, size=(n, m), dtype=np.int64))


def _drifting_paths(rng, n, m, n_cells):
    """Paths that start adjacent and separate -- the realistic neighbour case."""
    a = np.cumsum(rng.integers(-1, 2, size=(n, m)), axis=0) % n_cells
    b = a.copy()
    b[:, 0] = (b[:, 0] + 1) % n_cells
    drift = np.cumsum(rng.integers(0, 2, size=(n, m)), axis=0)
    b = (b + drift) % n_cells
    return a.astype(np.int64), b.astype(np.int64)


def _sticky_paths(rng, n, m, n_cells):
    """Few distinct values, so cancellations are dense and termination is early."""
    small = max(2, n_cells // 8)
    return (rng.integers(0, small, size=(n, m), dtype=np.int64),
            rng.integers(0, small, size=(n, m), dtype=np.int64))


_GENERATORS = (_random_paths, _drifting_paths, _sticky_paths)


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------

def test_vectorised_matches_reference():
    rng = np.random.default_rng(20260817)
    checked = 0
    for gen in _GENERATORS:
        for m in (1, 2, 3, 4, 6):
            for n in (1, 2, 5, 23, 40):
                for n_cells in (3, 16, 64):
                    pa, pb = gen(rng, n, m, n_cells)
                    for q in range(1, m + 1):
                        for rule in RULES:
                            fast = stod_pair(pa, pb, q=q, rule=rule)
                            slow = stod_pair_reference(pa, pb, q=q, rule=rule)
                            assert fast == slow, (
                                "mismatch: gen=%s m=%d n=%d n_cells=%d q=%d rule=%s\n"
                                "  core      = %r\n  reference = %r"
                                % (gen.__name__, m, n, n_cells, q, rule, fast, slow)
                            )
                            checked += 1
    print("    cross-checked %d (m, q, rule, input) combinations" % checked)


def test_prefix_early_exit_is_exact():
    """The doubling early-exit must give bit-identical results to the full scan."""
    rng = np.random.default_rng(555)
    checked = 0
    for gen in _GENERATORS:
        for m in (1, 2, 4, 6):
            for n in (1, 5, 33, 200, 700):
                for n_cells in (4, 50, 400):
                    pa, pb = gen(rng, n, m, n_cells)
                    for q in (1, max(1, m // 2), m):
                        for rule in RULES:
                            full = stod_pair(pa, pb, q=q, rule=rule, probe_start=0)
                            for start in (1, 2, 16, 256):
                                fast = stod_pair(pa, pb, q=q, rule=rule,
                                                 probe_start=start)
                                assert fast == full, (
                                    "early exit changed the answer: gen=%s m=%d "
                                    "n=%d n_cells=%d q=%d rule=%s start=%d\n"
                                    "  full = %r\n  fast = %r"
                                    % (gen.__name__, m, n, n_cells, q, rule,
                                       start, full, fast)
                                )
                                checked += 1
    print("    early exit verified exact on %d configurations" % checked)


def test_published_rule_matches_original_2d():
    original = _load_original_module()
    rng = np.random.default_rng(777)
    checked = 0
    for gen in _GENERATORS:
        for n in (1, 2, 3, 9, 30, 75):
            for n_cells in (2, 5, 20, 100):
                pa, pb = gen(rng, n, 2, n_cells)

                mine = stod_pair(pa, pb, q=2, rule=PUBLISHED)
                theirs = original.stod_pair_python(pa, pb)
                their_type, their_score = theirs[0], theirs[1]

                assert mine.type_code == their_type, (
                    "type mismatch (n=%d n_cells=%d gen=%s): %d vs %d"
                    % (n, n_cells, gen.__name__, mine.type_code, their_type)
                )
                assert abs(mine.score - their_score) < 1e-9, (
                    "score mismatch (n=%d n_cells=%d gen=%s): %r vs %r"
                    % (n, n_cells, gen.__name__, mine.score, their_score)
                )
                checked += 1
    print("    reproduced the published 2-D implementation on %d inputs" % checked)


def test_serpentine_canary_published_and_strict():
    """Canonical 6x6 Serpentine Canary from the published paper.

    Verifies the published reference ground truth:
    - Forward STOD: score = 21.0, type = T (0), k* = 4
    - Reversed FinSTOD: score = 15.0, type = T (0), k* = 3
    Both strict and published rules must yield these exact values.
    """
    size = 6
    length = 11  # 10 steps + initial position
    path = []
    for r in range(size):
        cols = range(size) if r % 2 == 0 else range(size - 1, -1, -1)
        for c in cols:
            path.append((r, c))
    full_snake = np.array(path, dtype=np.int64)
    n_cells = full_snake.shape[0]

    idx_a, idx_b = 5, 10
    idxs_a = (idx_a + np.arange(length)) % n_cells
    idxs_b = (idx_b + np.arange(length)) % n_cells
    traj_a = full_snake[idxs_a]
    traj_b = full_snake[idxs_b]
    traj_a_rev = traj_a[::-1].copy()
    traj_b_rev = traj_b[::-1].copy()

    for rule in (PUBLISHED, STRICT):
        res_stod = stod_pair(traj_a, traj_b, q=2, rule=rule)
        res_finstod = stod_pair(traj_a_rev, traj_b_rev, q=2, rule=rule)

        assert res_stod.type_code == TYPE_T, "STOD type mismatch for rule %s" % rule
        assert abs(res_stod.score - 21.0) < 1e-9, "STOD score %r != 21.0" % res_stod.score
        assert res_stod.k_star == 4, "STOD k* %r != 4" % res_stod.k_star

        assert res_finstod.type_code == TYPE_T, "FinSTOD type mismatch for rule %s" % rule
        assert abs(res_finstod.score - 15.0) < 1e-9, "FinSTOD score %r != 15.0" % res_finstod.score
        assert res_finstod.k_star == 3, "FinSTOD k* %r != 3" % res_finstod.k_star
    print("    verified Serpentine Canary (STOD=21.0, FinSTOD=15.0)")


def test_strict_terminates_no_later_than_published():
    """The published rule can only miss terminations, never invent them."""
    rng = np.random.default_rng(4242)
    n_differ = 0
    n_total = 0
    for gen in _GENERATORS:
        for m in (2, 3, 4):
            for _ in range(40):
                pa, pb = gen(rng, 30, m, 24)
                strict = stod_pair(pa, pb, q=m, rule=STRICT)
                published = stod_pair(pa, pb, q=m, rule=PUBLISHED)
                assert strict.k_star <= published.k_star, (
                    "strict terminated later than published: %r vs %r"
                    % (strict, published)
                )
                if strict.terminated and not published.terminated:
                    n_differ += 1
                n_total += 1
    print("    %d/%d cases where the published rule missed a termination"
          % (n_differ, n_total))


def test_q_monotone_in_termination_level():
    rng = np.random.default_rng(99)
    for gen in _GENERATORS:
        for m in (3, 4, 6):
            for _ in range(25):
                pa, pb = gen(rng, 35, m, 20)
                for rule in RULES:
                    ks = [stod_pair(pa, pb, q=q, rule=rule).k_star
                          for q in range(1, m + 1)]
                    assert all(ks[i] <= ks[i + 1] for i in range(len(ks) - 1)), (
                        "k* not monotone in q (m=%d rule=%s): %r" % (m, rule, ks)
                    )


def test_identical_start_cell_terminates_at_zero():
    for m in (1, 2, 5):
        pa = np.arange(10 * m).reshape(10, m) % 7
        pb = pa.copy()
        pb[1:] = (pb[1:] + 3) % 7          # same level-0 cell, different afterwards
        for rule in RULES:
            res = stod_pair(pa, pb, q=m, rule=rule)
            assert res.terminated
            assert res.type_code == TYPE_T
            assert res.score == 0.0
            assert res.k_star == 0


def test_empty_path():
    for rule in RULES:
        res = stod_pair(np.zeros((0, 3), dtype=int), np.zeros((5, 3), dtype=int),
                        rule=rule)
        assert res.type_code == TYPE_T
        assert res.score == 0.0
        assert res.n_levels == 0
        assert res.k_star == -1


def test_disjoint_paths_are_uu_at_max_score():
    """No shared value in any coordinate: UU, and the score hits m*L*(L+1)."""
    for m in (1, 2, 4, 6):
        for n in (1, 4, 25):
            pa = np.arange(n * m).reshape(n, m)
            pb = pa + 10 ** 6
            for rule in RULES:
                res = stod_pair(pa, pb, q=m, rule=rule)
                assert not res.terminated
                assert res.type_code == TYPE_UU
                expected = stod_max_score(n, m)
                assert abs(res.score - expected) < 1e-9, (
                    "m=%d n=%d: %r != %r" % (m, n, res.score, expected)
                )


def test_score_is_level_weighted_uncancelled_count():
    """Hand-checked case, verifying the score convention directly."""
    # m=2.  Coordinate 0 shares the value 5; coordinate 1 shares nothing.
    pa = np.array([[5, 0], [6, 1], [7, 2]])
    pb = np.array([[5, 50], [8, 51], [9, 52]])
    res = stod_pair(pa, pb, q=2, rule=STRICT)
    # Level 0: a=(5,0) -> coord0 cancelled (5 in B), coord1 not.
    #          b=(5,50) -> coord0 cancelled (5 in A), coord1 not.  => 2 of 4 cancelled
    # Level 1: nothing shared -> 4 of 4 uncancelled
    # Level 2: nothing shared -> 4 of 4 uncancelled
    # No point ever has both coordinates cancelled, so the pair is unterminated.
    assert not res.terminated
    assert res.k_star == 2
    assert res.score == 0 * 2 + 1 * 4 + 2 * 4


def test_von_neumann_offsets():
    for m in (1, 2, 3, 5):
        offs = von_neumann_offsets(m)
        assert offs.shape == (2 * m, m)
        assert np.all(np.abs(offs).sum(axis=1) == 1)
        assert len({tuple(r) for r in offs}) == 2 * m


def test_build_paths_finstod_ordering():
    sysm = CoupledStandardMap(d=2, K=1.1, xi=0.3)
    cells = np.array([[3, 4, 5, 6], [10, 11, 12, 13]], dtype=np.int64)
    n_cells, n_levels = 32, 12

    fwd = build_paths(sysm, cells, n_cells, n_levels, reverse=False)
    rev = build_paths(sysm, cells, n_cells, n_levels, reverse=True)

    assert fwd.shape == (2, n_levels, 4)
    assert np.array_equal(fwd[:, 0, :], cells)          # forward starts at the seed
    assert np.array_equal(rev[:, -1, :], cells)         # reversed ends at the seed
    assert np.array_equal(rev, fwd[:, ::-1, :])
    assert fwd.min() >= 0 and fwd.max() < n_cells


def test_frozen_coordinates_stay_frozen():
    sysm = TorusTranslation(m=5, d=2)
    cells = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    path = build_paths(sysm, cells, 40, 30, reverse=False)[0]
    # Coordinates 2..4 have zero rotation number and must never move.
    assert np.all(path[:, 2:] == path[0, 2:])
    # The moving coordinates must actually move.
    assert np.unique(path[:, 0]).size > 1


def test_select_neighbourhood_priority():
    types = np.array([[0, 1, 2, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
    scores = np.array([[9.0, 8.0, 1.0, 7.0], [5.0, 6.0, 2.0, 3.0], [4.0, 1.0, 2.0, 3.0]])
    best_type, best_score = select_neighbourhood(types, scores)
    # Row 0: a UU is present, so it wins despite its low score.
    assert best_type[0] == TYPE_UU and best_score[0] == 1.0
    # Row 1: best type is UC, and among UC pairs the larger score wins.
    assert best_type[1] == 1 and best_score[1] == 3.0
    # Row 2: all T, so the largest score wins.
    assert best_type[2] == TYPE_T and best_score[2] == 4.0


def _all_systems():
    from stod_nd.flows import (
        CoupledDuffing,
        HenonHeiles,
        IntegrableTorusFlow,
        Lorenz96,
    )
    return [
        Lorenz96(m=5, F=8.0, dt_out=0.005),
        CoupledDuffing(d=2, dt_out=0.005),
        HenonHeiles(dof=3, dt_out=0.005),
        IntegrableTorusFlow(m=5, d=2, dt_out=0.005),
        IntegrableTorusFlow(m=5, d=2, dt_out=0.005, basis="mixed"),
        CoupledStandardMap(d=2, K=1.2, xi=0.3),
        TorusTranslation(m=4, d=2),
    ]


def test_cell_state_roundtrip_for_every_system():
    """Discretising a cell centre must return the same cell, for every system."""
    for system in _all_systems():
        for n_cells in (16, 101, 400):
            cells = np.array([[0] * system.m,
                              [n_cells - 1] * system.m,
                              [n_cells // 3] * system.m], dtype=np.int64)
            back = system.state_to_cell(system.cell_to_state(cells, n_cells),
                                        n_cells)
            assert np.array_equal(back, cells), (
                "%r at n_cells=%d: %r != %r" % (system, n_cells, back, cells)
            )


def test_paths_stay_finite_for_unbounded_flows():
    """Escaping trajectories must freeze, never emit NaN or overflow."""
    from stod_nd.flows import HenonHeiles

    system = HenonHeiles(dof=2, dt_out=0.01)
    # Deliberately seed at high energy, where the potential opens and orbits run
    # away.  These are exactly the cells valid_cells() rejects.
    n_cells = 64
    cells = np.array([[1, 1, 1, 1], [62, 62, 62, 62], [0, 63, 0, 63]],
                     dtype=np.int64)
    with np.errstate(all="raise"):
        paths = build_paths(system, cells, n_cells, 400)
    assert np.all(np.isfinite(paths))
    assert paths.dtype == np.int32
    # Frozen escapees must land outside the box, not on a boundary cell, so that
    # they remain distinguishable from bounded orbits.
    assert paths.max() > n_cells or paths.min() < 0


def test_valid_cells_restricts_to_bound_region():
    from stod_nd.flows import HenonHeiles

    system = HenonHeiles(dof=2, dt_out=0.01)
    rng = np.random.default_rng(11)
    cells = sample_cells(system.m, 64, 200, rng, system=system)
    assert cells.shape == (200, system.m)
    assert np.all(system.valid_cells(cells, 64))
    states = system.cell_to_state(cells, 64)
    assert np.all(system.energy(states) <= system.energy_max + 1e-12)


def test_torus_basis_is_a_valid_automorphism():
    """The change of basis must be integral with an integral inverse."""
    from stod_nd.flows import IntegrableTorusFlow

    for basis in ("identity", "mixed", "strong"):
        for m in (3, 4, 6, 8):
            system = IntegrableTorusFlow(m=m, d=2, basis=basis)
            assert np.allclose(system.A.dot(system.Ainv), np.eye(m))
            assert abs(abs(np.linalg.det(system.A)) - 1.0) < 1e-9
            assert np.allclose(system.A, np.rint(system.A))
            assert np.allclose(system.Ainv, np.rint(system.Ainv))


def test_integrable_torus_conserves_its_actions():
    """Exactly ``m - d`` independent quantities must stay frozen."""
    from stod_nd.flows import IntegrableTorusFlow

    for basis in ("identity", "mixed"):
        for m, d in ((5, 2), (6, 3), (4, 1)):
            system = IntegrableTorusFlow(m=m, d=d, dt_out=0.01, basis=basis)
            rng = np.random.default_rng(5)
            states = rng.random((8, m))
            internal0 = system._internal(states)
            for _ in range(200):
                states = system.step(states)
            internal1 = system._internal(states)
            drift = np.abs(internal1[:, d:] - internal0[:, d:]).max()
            assert drift < 1e-9, (
                "basis=%s m=%d d=%d: actions drifted by %.3g" % (basis, m, d, drift)
            )


def test_slice_cells_geometry():
    m, n_cells, stride = 5, 40, 4
    anchor = np.array([7, 7, 7, 7, 7], dtype=np.int64)
    cells, ticks = slice_cells(m, n_cells, (1, 3), anchor=anchor, stride=stride)
    n_side = ticks.size
    assert cells.shape == (n_side * n_side, m)
    # The two chosen axes vary; every other coordinate is pinned to the anchor.
    for i in (0, 2, 4):
        assert np.all(cells[:, i] == 7)
    assert set(np.unique(cells[:, 1])) == set(ticks.tolist())
    assert set(np.unique(cells[:, 3])) == set(ticks.tolist())
    assert len({tuple(r) for r in cells}) == n_side * n_side


def test_path_locality_detects_the_two_regimes():
    """A map must report ~1 level per cell; an oversampled flow must report more."""
    from stod_nd.flows import Lorenz96

    n_cells = 400
    rng = np.random.default_rng(2)

    mapping = CoupledStandardMap(d=2, K=2.5, xi=0.3)
    cells = sample_cells(mapping.m, n_cells, 16, rng)
    map_info = path_locality(build_paths(mapping, cells, n_cells, 200), n_cells)
    assert map_info["levels_per_cell"] < 1.2, map_info

    flow = Lorenz96(m=4, F=8.0, dt_out=0.0002)
    cells = sample_cells(flow.m, n_cells, 16, rng)
    flow_info = path_locality(build_paths(flow, cells, n_cells, 200), n_cells)
    assert flow_info["levels_per_cell"] > 2.0, flow_info
    assert flow_info["mean_step_cells"] < 1.0, flow_info


def test_chunking_does_not_change_results():
    """Neighbours are resolved globally, so chunk size must be irrelevant."""
    from stod_nd.flows import CoupledDuffing

    system = CoupledDuffing(d=2, dt_out=0.005)
    rng = np.random.default_rng(13)
    cells = sample_cells(system.m, 128, 24, rng, system=system)
    baseline = evaluate_cells(system, cells, 128, 300, progress=False,
                              chunk_size=1000)
    for chunk in (1, 5, 7):
        other = evaluate_cells(system, cells, 128, 300, progress=False,
                               chunk_size=chunk)
        assert np.array_equal(baseline["pair_types"], other["pair_types"])
        assert np.allclose(baseline["pair_scores"], other["pair_scores"])
        assert np.array_equal(baseline["cell_types"], other["cell_types"])


def test_jacobian_matches_finite_differences():
    """The analytic tangent map must agree with a central difference of ``step``.

    Guards the FTLE baseline: a wrong Jacobian would silently corrupt every
    comparison against FinSTOD.
    """
    rng = np.random.default_rng(31337)
    eps = 1e-6
    for sysm in (CoupledStandardMap(d=1, K=1.3, xi=0.0),
                 CoupledStandardMap(d=2, K=1.1, xi=0.4),
                 CoupledStandardMap(d=3, K=0.7, xi=0.25),
                 TorusTranslation(m=4, d=2, shear=0.3)):
        m = sysm.m
        # Stay clear of the mod-1 wrap, where finite differences are meaningless.
        states = 0.2 + 0.6 * rng.random((6, m))
        analytic = sysm.jacobian(states)

        numeric = np.empty_like(analytic)
        for j in range(m):
            bump = np.zeros(m)
            bump[j] = eps
            plus = sysm.step(states + bump[None, :])
            minus = sysm.step(states - bump[None, :])
            # Undo any wrap introduced by the perturbation itself.
            diff = plus - minus
            diff -= np.round(diff)
            numeric[:, :, j] = diff / (2 * eps)

        err = np.abs(analytic - numeric).max()
        assert err < 1e-5, "%r: max Jacobian error %.3g" % (sysm, err)


def test_ftle_is_zero_for_rigid_translation():
    """A rigid rotation has no stretching, so its FTLE must vanish."""
    from stod_nd.indicators import map_ftle

    sysm = TorusTranslation(m=4, d=3)
    cells = np.array([[2, 3, 4, 5], [9, 8, 7, 6]], dtype=np.int64)
    ftle = map_ftle(sysm, cells, n_cells=32, n_steps=50)
    assert np.abs(ftle).max() < 1e-10, ftle


def test_ftle_standard_map_matches_known_regimes():
    """Large K must give a clearly positive exponent; K=0 must give zero."""
    from stod_nd.indicators import map_ftle

    cells = np.arange(16, dtype=np.int64)[:, None] * np.array([[1, 1]])
    integrable = map_ftle(CoupledStandardMap(d=1, K=0.0, xi=0.0),
                          cells, n_cells=16, n_steps=200)
    chaotic = map_ftle(CoupledStandardMap(d=1, K=5.0, xi=0.0),
                       cells, n_cells=16, n_steps=200)
    assert np.abs(integrable).max() < 1e-8, integrable
    # For the standard map at K=5 the exponent is around log(K/2) ~ 0.9.
    assert np.median(chaotic) > 0.5, np.median(chaotic)


def test_spearman_basics():
    from stod_nd.indicators import spearman

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(spearman(x, 2 * x + 1) - 1.0) < 1e-12
    assert abs(spearman(x, -x) + 1.0) < 1e-12
    # Monotone but nonlinear -> still exactly 1 for a rank correlation.
    assert abs(spearman(x, np.exp(x)) - 1.0) < 1e-12
    # Ties handled without blowing up.
    assert abs(spearman(np.array([1.0, 1.0, 2.0, 2.0]),
                        np.array([1.0, 2.0, 3.0, 4.0])) - 0.894427191) < 1e-6


def test_segmented_normalise_ranges():
    types = np.array([0, 0, 0, 1, 1, 2])
    scores = np.array([10.0, 20.0, 30.0, 100.0, 300.0, 999.0])
    out, P = segmented_normalise(types, scores)
    assert abs(P - 3.0 / 5.0) < 1e-12
    assert np.all(out[:3] >= 0.0) and np.all(out[:3] <= P + 1e-12)
    assert np.all(out[3:5] >= P - 1e-12) and np.all(out[3:5] <= 0.99 + 1e-12)
    assert out[5] == 1.0
    assert np.all(np.diff(out[:3]) > 0)      # order preserved inside a class

    out_log, P_log = segmented_normalise(types, scores, log=True)
    assert abs(P_log - 3.0 / 5.0) < 1e-12
    assert np.all(out_log[:3] >= 0.0) and np.all(out_log[:3] <= P_log + 1e-12)
    assert np.all(out_log[3:5] >= P_log - 1e-12) and np.all(out_log[3:5] <= 0.99 + 1e-12)
    assert out_log[5] == 1.0
    assert np.all(np.diff(out_log[:3]) > 0)


def test_slice_cells_3d_geometry():
    from stod_nd.fields import slice_cells_3d

    cells, ticks = slice_cells_3d(m=6, n_cells=100, axes=(0, 2, 4), stride=5)
    assert ticks.size == 20
    assert cells.shape == (20 * 20 * 20, 6)
    # Check anchor on non-varying axes (1, 3, 5)
    assert np.all(cells[:, 1] == 50)
    assert np.all(cells[:, 3] == 50)
    assert np.all(cells[:, 5] == 50)


def test_new_flow_systems_step_and_jacobian():
    from stod_nd.flows import ABCFlow, CR3BP, FPUT, Froeschle, Kuramoto

    systems = [
        ABCFlow(A=1.0, B=1.0, C=1.0, dt_out=0.01),
        Froeschle(dof=2, K=0.3, dt_out=0.01),
        Froeschle(dof=3, K=0.2, dt_out=0.01),
        CR3BP(mu=0.01215, dt_out=0.005),
        FPUT(dof=4, alpha=0.25, beta=0.1, dt_out=0.01),
        Kuramoto(m=8, K=1.5, dt_out=0.01),
    ]

    eps = 1e-6
    for sysm in systems:
        m = sysm.m
        states = 0.5 * (sysm.lo + sysm.hi)[None, :] + 0.1 * np.random.default_rng(42).standard_normal((4, m))
        v = sysm.velocity(states)
        assert v.shape == (4, m)
        assert np.all(np.isfinite(v))

        # Test velocity Jacobian against finite differences
        analytic_jac = sysm.velocity_jacobian(states)
        numeric_jac = np.empty_like(analytic_jac)
        for j in range(m):
            bump = np.zeros(m)
            bump[j] = eps
            v_plus = sysm.velocity(states + bump[None, :])
            v_minus = sysm.velocity(states - bump[None, :])
            numeric_jac[:, :, j] = (v_plus - v_minus) / (2 * eps)

        err = np.abs(analytic_jac - numeric_jac).max()
        assert err < 1e-4, "%s: max velocity Jacobian error %.3g" % (sysm.name, err)


# ----------------------------------------------------------------------------

def run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for func in tests:
        name = func.__name__
        try:
            func()
        except AssertionError as exc:
            failures += 1
            print("FAIL  %s" % name)
            print("      %s" % str(exc).replace("\n", "\n      "))
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print("ERROR %s: %s: %s" % (name, type(exc).__name__, exc))
        else:
            print("ok    %s" % name)
    print("\n%d/%d passed" % (len(tests) - failures, len(tests)))
    return failures


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)
