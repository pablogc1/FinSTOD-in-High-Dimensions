# -*- coding: utf-8 -*-
"""
stod_nd -- m-dimensional STOD / FinSTOD.
=======================================

Generalisation of the published 2-D FinSTOD indicator to an arbitrary number of
discretised grid coordinates, built to answer one question first: does the
cancellation mechanism still produce a graded scalar field as the number of
coordinates grows, or does the requirement that *all* components cancel make
termination so rare that the field saturates?

See ``README.md`` for the reasoning, and ``experiments/`` for the runnable
studies.
"""

from __future__ import annotations

from .core import (
    PUBLISHED,
    STRICT,
    TYPE_NAME,
    TYPE_T,
    TYPE_UC,
    TYPE_UU,
    PairResult,
    stod_max_score,
    stod_pair,
)
from .base import GridSystem
from .fields import (
    auto_chunk_size,
    build_paths,
    evaluate_cells,
    pairwise_summary,
    path_locality,
    sample_cells,
    segmented_normalise,
    select_neighbourhood,
    slice_cells,
    slice_cells_3d,
    summary_table,
    von_neumann_offsets,
)
from .flows import (
    ABCFlow,
    BickleyJetFlow,
    CR3BP,
    FPUT,
    CoupledDuffing,
    Froeschle,
    HenonHeiles,
    Kuramoto,
    Lorenz96,
)
from .indicators import finite_difference_ftle, flow_ftle, flow_indicator_bundle, map_ftle, spearman
from .reference import stod_pair_reference
from .systems import CoupledStandardMap, TorusTranslation, make_system

__version__ = "0.1.0"

__all__ = [
    "PUBLISHED",
    "STRICT",
    "TYPE_NAME",
    "TYPE_T",
    "TYPE_UC",
    "TYPE_UU",
    "PairResult",
    "stod_pair",
    "stod_pair_reference",
    "stod_max_score",
    "GridSystem",
    "auto_chunk_size",
    "build_paths",
    "evaluate_cells",
    "pairwise_summary",
    "path_locality",
    "sample_cells",
    "segmented_normalise",
    "select_neighbourhood",
    "slice_cells",
    "slice_cells_3d",
    "summary_table",
    "von_neumann_offsets",
    "CoupledStandardMap",
    "TorusTranslation",
    "make_system",
    "CoupledDuffing",
    "HenonHeiles",
    "Lorenz96",
    "ABCFlow",
    "BickleyJetFlow",
    "Froeschle",
    "CR3BP",
    "FPUT",
    "Kuramoto",
    "map_ftle",
    "flow_ftle",
    "flow_indicator_bundle",
    "finite_difference_ftle",
    "spearman",
]
