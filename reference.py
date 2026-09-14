# -*- coding: utf-8 -*-
"""
Straightforward, slow, obviously-correct STOD pair logic.
========================================================

This module exists only to validate :func:`stod_nd.core.stod_pair`.  It walks the
two paths level by level with Python sets, in the same order of operations as the
released C++ implementation, and makes no attempt to be fast: it is O(n^2 * m).

:func:`stod_nd.core.stod_pair` computes the same quantities through a closed-form
reformulation that vectorises.  Reformulations are exactly where subtle mistakes
hide, so the test suite cross-checks the two on random and structured inputs.
"""

from __future__ import annotations

import numpy as np

from .core import PUBLISHED, STRICT, TYPE_T, TYPE_UC, TYPE_UU, PairResult, _RULES


def stod_pair_reference(path_a, path_b, q=None, rule=STRICT):
    """Level-by-level STOD pair logic, transcribed from the definitions.

    Same signature and return value as :func:`stod_nd.core.stod_pair`.
    """
    if rule not in _RULES:
        raise ValueError("rule must be one of %r; got %r" % (_RULES, rule))

    pa = np.asarray(path_a, dtype=np.int64)
    pb = np.asarray(path_b, dtype=np.int64)
    if pa.ndim == 1:
        pa = pa[:, None]
    if pb.ndim == 1:
        pb = pb[:, None]

    m = pa.shape[1]
    if pb.shape[1] != m:
        raise ValueError("coordinate count mismatch")

    if q is None:
        q = m
    q = int(q)
    if not 1 <= q <= m:
        raise ValueError("q must satisfy 1 <= q <= m")

    n = min(pa.shape[0], pb.shape[0])
    if n == 0:
        return PairResult(TYPE_T, 0.0, False, -1, 0)

    seen_a = [set() for _ in range(m)]
    seen_b = [set() for _ in range(m)]

    # Under the "published" rule a point can only ever terminate the pair if at
    # least one of its components was already cancelled at its own level.
    eligible_a = [False] * n
    eligible_b = [False] * n

    terminated = False
    final_level = 0

    for level in range(n):
        final_level = level

        for i in range(m):
            seen_a[i].add(int(pa[level, i]))
            seen_b[i].add(int(pb[level, i]))

        # Coverage of the points entering at this level, evaluated now.
        cov_a_now = sum(1 for i in range(m) if int(pa[level, i]) in seen_b[i])
        cov_b_now = sum(1 for i in range(m) if int(pb[level, i]) in seen_a[i])
        eligible_a[level] = cov_a_now >= 1
        eligible_b[level] = cov_b_now >= 1

        # Any point visited so far, including this level's, may complete.
        for j in range(level + 1):
            cov_a = sum(1 for i in range(m) if int(pa[j, i]) in seen_b[i])
            if cov_a >= q and (rule == STRICT or eligible_a[j]):
                terminated = True
                break
            cov_b = sum(1 for i in range(m) if int(pb[j, i]) in seen_a[i])
            if cov_b >= q and (rule == STRICT or eligible_b[j]):
                terminated = True
                break

        if terminated:
            break

    # Scoring, against the sets as they stand at the level where we stopped.
    total_score = 0.0
    has_cancellation = False
    for level in range(final_level + 1):
        cancelled = 0
        for i in range(m):
            if int(pa[level, i]) in seen_b[i]:
                cancelled += 1
            if int(pb[level, i]) in seen_a[i]:
                cancelled += 1
        if cancelled > 0:
            has_cancellation = True
        total_score += level * ((2 * m) - cancelled)

    if terminated:
        type_code = TYPE_T
    elif has_cancellation:
        type_code = TYPE_UC
    else:
        type_code = TYPE_UU

    return PairResult(type_code, float(total_score), bool(terminated),
                      int(final_level), int(n))
