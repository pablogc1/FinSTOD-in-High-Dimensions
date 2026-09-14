# -*- coding: utf-8 -*-
"""
m-dimensional STOD / FinSTOD pair logic.
=======================================

This is the generalisation of the published 2-D implementation
(``reference/pipeline_core/sod_logic_original.py`` and
``reference/cpp_backend/sod_logic_original.hpp``) to an arbitrary number of
discretised grid coordinates ``m``.

A *path* is an integer array of shape ``(L, m)``: one row per level, one column
per grid coordinate.  At ``m = 2`` the columns are the ``(row, col)`` pair of the
original code.

Two things are generalised relative to the published version.

**The number of coordinates, m.**  Cancellation is component-wise: coordinate
``i`` of one path is cancelled when its value appears anywhere in coordinate
``i`` of the other path.  Each coordinate keeps its own independent set of seen
values, exactly as rows and columns do at ``m = 2``.

**The termination threshold, q.**  The published rule terminates when *all*
components of some visited point have been cancelled.  Here termination happens
when *q* of the *m* components have been cancelled, with ``q = m`` recovering the
published rule.  The motivation is that requiring all *m* components makes
termination geometrically rare as *m* grows (see the project README), so ``q``
turns that scaling problem into a tunable parameter.

The score is unchanged from the published definition,

    STOD = sum_k  k * U_k ,

where ``U_k`` is the number of uncancelled components at level ``k`` counted over
both paths, so ``U_k`` ranges over ``0 .. 2m``.  Only the level at which the sum
stops, ``k*``, depends on ``q``.

Two termination rules are available
-----------------------------------

``"strict"``
    Definitions 3 and 4 of the paper as written: the pair terminates at the
    first level at which any visited point has ``q`` of its components
    cancelled, regardless of when those cancellations happened.

``"published"``
    Reproduces what the released 2-D code actually computes.  Both the Python
    and the C++ implementations insert a point into the "waiting" structures
    only at its own level, and only if at least one of its components was
    already covered at that moment; a point registered with nothing covered is
    never revisited and so can never terminate the pair.  This rule is
    therefore ``"strict"`` plus the extra condition that a point may only
    trigger termination if at least one of its components was already cancelled
    by its own level.

``"strict"`` never terminates later than ``"published"``: the extra condition can
only suppress terminations, never create them, so ``k*_strict <= k*_published``.
The scores are *not* ordered, however, because a longer scoring window also
admits more cancellations and so lowers the per-level contributions.  At ``m = 2``
the two rules rarely differ, since for a von Neumann neighbour one coordinate is
shared and cancels immediately; the gap grows with ``m``.  ``"published"`` exists
so that results here can be tied back to the published 2-D figures, and so the
size of that gap can be measured rather than assumed.

Implementation note
-------------------
The published code walks the two paths level by level, maintaining the seen-sets
and the waiting maps as it goes.  That is necessary if the paths arrive as a
stream, but here both paths are known up front, which allows a closed-form
reformulation that vectorises cleanly in NumPy and needs no Numba:

    Let first_b[i][v] be the first level at which coordinate i of path B takes
    value v.  Coordinate i of A's point j is cancelled from level
    max(j, first_b[i][a[j, i]]) onwards.  Point j therefore reaches q cancelled
    components at level

        t_a[j] = max( j, q-th smallest of { first_b[i][a[j, i]] : i } )

    and the pair terminates at k* = min over j and over both paths of t[j].

``stod_pair`` implements this.  ``stod_nd.reference.stod_pair_reference``
implements the level-by-level simulation directly from the definitions, and the
test suite checks that the two agree, and that ``"published"`` at ``m = 2``,
``q = 2`` matches the original released code exactly.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

TYPE_T = 0   # Terminated
TYPE_UC = 1  # Unterminated, at least one cancellation
TYPE_UU = 2  # Unterminated, no cancellation at all

TYPE_NAME = {TYPE_T: "T", TYPE_UC: "UC", TYPE_UU: "UU"}
TYPE_ORDER = (TYPE_T, TYPE_UC, TYPE_UU)

STRICT = "strict"
PUBLISHED = "published"
_RULES = (STRICT, PUBLISHED)

# Stands in for "never cancelled".  Large enough to lose any comparison against a
# level index, small enough that adding to it cannot overflow int64.
_NEVER = np.int64(1) << 40

PairResult = namedtuple(
    "PairResult",
    ["type_code", "score", "terminated", "k_star", "n_levels"],
)


# ----------------------------------------------------------------------------
# internals
# ----------------------------------------------------------------------------

def _factorise(pa, pb):
    """Map the values of each coordinate onto dense codes shared by both paths.

    Cell indices may be any integers -- the C++ ``discretize_path`` deliberately
    does not clamp to the grid, so trajectories leaving the domain carry indices
    outside ``[0, n_cells)``.  Factorising per coordinate removes any assumption
    about their range while keeping the membership tests as array lookups.

    Returns ``(codes_a, codes_b, width)`` with codes in ``[0, width)``.
    """
    n = pa.shape[0]
    comb = np.concatenate((pa, pb), axis=0)
    order = np.argsort(comb, axis=0, kind="stable")
    srt = np.take_along_axis(comb, order, axis=0)

    is_new = np.empty(srt.shape, dtype=bool)
    is_new[0, :] = True
    np.not_equal(srt[1:], srt[:-1], out=is_new[1:])

    codes_srt = np.cumsum(is_new, axis=0, dtype=np.int64) - 1
    codes = np.empty_like(codes_srt)
    np.put_along_axis(codes, order, codes_srt, axis=0)
    return codes[:n], codes[n:], 2 * n


def _flat_index(codes, width):
    """Fold ``(level, coordinate)`` codes into one flat index per coordinate."""
    m = codes.shape[1]
    offsets = (np.arange(m, dtype=np.int64) * np.int64(width))[None, :]
    return codes + offsets


def _first_occurrence(codes, width):
    """First level at which each (coordinate, code) pair occurs; ``_NEVER`` if absent."""
    n, m = codes.shape
    out = np.full(m * width, _NEVER, dtype=np.int64)
    flat = _flat_index(codes, width)
    # Write levels in descending order so that the level-0 write lands last and
    # duplicate codes end up holding their *first* occurrence.
    levels_desc = np.repeat(np.arange(n - 1, -1, -1, dtype=np.int64), m)
    out[flat[::-1].ravel()] = levels_desc
    return out


def _as_path(path, name):
    arr = np.asarray(path, dtype=np.int64)
    if arr.ndim == 1:  # a single-coordinate path
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("%s must be (L, m); got shape %r" % (name, arr.shape))
    return arr


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------

def stod_pair(path_a, path_b, q=None, rule=STRICT, probe_start=256):
    """STOD value and pair type for one pair of discretised paths.

    Parameters
    ----------
    path_a, path_b : (L, m) int array_like
        Discretised paths.  For FinSTOD these must already be time-reversed, so
        that level 0 is the final cell.  Lengths may differ; the comparison uses
        the first ``min(L_a, L_b)`` levels, as the published code does.
    q : int, optional
        Number of cancelled components required to terminate.  Defaults to ``m``,
        the published rule.
    rule : {"strict", "published"}
        Termination rule; see the module docstring.
    probe_start : int
        Length of the first prefix tried by the early-exit schedule described
        below.  Set to 0 to disable it.  Results are unaffected either way.

    Returns
    -------
    PairResult
        ``type_code`` (0 = T, 1 = UC, 2 = UU), ``score``, ``terminated``,
        ``k_star`` (last level included in the score, ``-1`` for empty input) and
        ``n_levels`` (the number of comparable levels).

    Notes
    -----
    Realistic settings need long paths: the published configs integrate to
    ``t = 20`` with ``dt = 0.001``, so ``L`` is around ``2 x 10^4``.  Most pairs,
    though, terminate in the first per cent or so of that.  So the computation is
    tried on a prefix first and the prefix length doubled until termination is
    found.

    That shortcut is exact, not approximate.  Restricting to a prefix can only
    raise a first-occurrence level (a value not present in the prefix looks like
    "never"), so it can only push ``k*`` later.  Conversely, if the pair truly
    terminates at ``k0``, every cancellation responsible for that sits at a level
    ``<= k0``, so any prefix reaching ``k0`` observes them all and reports the same
    ``k*``.  A termination found inside a prefix is therefore the true one, and the
    score depends only on levels ``<= k*``, so it is identical too.  Only pairs
    that never terminate pay for the full length, and they pay at most twice.
    """
    if rule not in _RULES:
        raise ValueError("rule must be one of %r; got %r" % (_RULES, rule))

    pa = _as_path(path_a, "path_a")
    pb = _as_path(path_b, "path_b")

    m = pa.shape[1]
    if pb.shape[1] != m:
        raise ValueError(
            "both paths need the same number of coordinates; got %d and %d"
            % (m, pb.shape[1])
        )

    if q is None:
        q = m
    q = int(q)
    if not 1 <= q <= m:
        raise ValueError("q must satisfy 1 <= q <= m = %d; got %d" % (m, q))

    n = min(pa.shape[0], pb.shape[0])
    if n == 0:
        # Matches the published code, which returns type T with score 0.
        return PairResult(TYPE_T, 0.0, False, -1, 0)

    probe_start = int(probe_start)
    if probe_start > 0:
        limit = probe_start
        while limit < n:
            probe = _stod_pair_window(pa[:limit], pb[:limit], m, q, rule)
            if probe.terminated:
                return probe._replace(n_levels=n)
            limit *= 2

    return _stod_pair_window(pa[:n], pb[:n], m, q, rule)


def _stod_pair_window(pa, pb, m, q, rule):
    """The pair computation over exactly the levels supplied."""
    n = pa.shape[0]

    codes_a, codes_b, width = _factorise(pa, pb)
    first_a = _first_occurrence(codes_a, width)
    first_b = _first_occurrence(codes_b, width)

    # cover_a[j, i] = level from which coordinate i of A's point j is cancelled.
    cover_a = first_b[_flat_index(codes_a, width)]
    cover_b = first_a[_flat_index(codes_b, width)]

    levels = np.arange(n, dtype=np.int64)

    # Level at which each point reaches q cancelled components.
    if q == m:
        qth_a = cover_a.max(axis=1)
        qth_b = cover_b.max(axis=1)
    else:
        qth_a = np.partition(cover_a, q - 1, axis=1)[:, q - 1]
        qth_b = np.partition(cover_b, q - 1, axis=1)[:, q - 1]

    t_a = np.maximum(levels, qth_a)
    t_b = np.maximum(levels, qth_b)

    if rule == PUBLISHED:
        # A point may only terminate the pair if at least one of its components
        # was already cancelled at its own level -- the registration condition
        # baked into the released implementation.
        eligible_a = cover_a.min(axis=1) <= levels
        eligible_b = cover_b.min(axis=1) <= levels
        t_a = np.where(eligible_a, t_a, _NEVER)
        t_b = np.where(eligible_b, t_b, _NEVER)

    k_term = min(int(t_a.min()), int(t_b.min()))
    terminated = k_term <= n - 1
    k_star = k_term if terminated else n - 1

    # Scoring uses the seen-sets accumulated over levels 0..k_star, which is what
    # the published code's final sets contain when it breaks at k_star.  A
    # component is cancelled in that window exactly when its first occurrence in
    # the other path is at or before k_star.
    canc_a = cover_a[: k_star + 1] <= k_star
    canc_b = cover_b[: k_star + 1] <= k_star
    canc_count = canc_a.sum(axis=1) + canc_b.sum(axis=1)
    uncancelled = (2 * m) - canc_count

    score = float(np.dot(uncancelled, levels[: k_star + 1]))

    if terminated:
        type_code = TYPE_T
    elif canc_count.any():
        type_code = TYPE_UC
    else:
        type_code = TYPE_UU

    return PairResult(type_code, score, bool(terminated), int(k_star), int(n))


def stod_max_score(n_levels, m):
    """Theoretical maximum score, ``m * L * (L + 1)``, for a UU pair.

    Same expression as in the paper (Section 3.2.1), with ``L = n_levels - 1``
    the largest level index: every level ``k`` contributes ``2m * k``, and
    ``2m * sum(k, k=0..L) = m * L * (L + 1)``.
    """
    L = int(n_levels) - 1
    if L < 0:
        return 0.0
    return float(m) * L * (L + 1)
