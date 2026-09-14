# -*- coding: utf-8 -*-
"""
Task generator for CESVIMA / Slurm high-dimensional FinSTOD pipeline.
=====================================================================

Splits an m-dimensional grid slice (or sampling) into K shards for parallel
array execution across cluster nodes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Add project root to sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PROJ_ROOT = os.path.dirname(_ROOT)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from stod_nd import (
    ABCFlow,
    BickleyJetFlow,
    CR3BP,
    FPUT,
    CoupledDuffing,
    Froeschle,
    HenonHeiles,
    Kuramoto,
    Lorenz96,
    sample_cells,
    slice_cells,
    slice_cells_3d,
)
from stod_nd.flows import IntegrableTorusFlow


def load_config(path):
    """Load configuration from YAML or JSON."""
    if not os.path.isfile(path):
        raise FileNotFoundError("Config file not found: %s" % path)
    
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except ImportError:
            pass

        # Fallback lightweight parser for nested YAML configs
        import ast
        cfg = {}
        current_section = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.rstrip()
                stripped = raw.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                indent = len(raw) - len(raw.lstrip())
                if ":" in stripped:
                    k, v = stripped.split(":", 1)
                    k, v = k.strip(), v.strip()
                    val = None
                    if v != "" and not v.startswith("#"):
                        # strip comments if any
                        if " #" in v:
                            v = v.split(" #", 1)[0].strip()
                        if v.lower() == "null" or v.lower() == "none":
                            val = None
                        elif v.lower() == "true":
                            val = True
                        elif v.lower() == "false":
                            val = False
                        else:
                            try:
                                val = ast.literal_eval(v)
                            except Exception:
                                val = v.strip('"').strip("'")
                    
                    if indent == 0:
                        if val is None and (v == "" or v.startswith("#")):
                            current_section = k
                            cfg[current_section] = {}
                        else:
                            current_section = None
                            cfg[k] = val
                    else:
                        if current_section is not None:
                            cfg[current_section][k] = val
                        else:
                            cfg[k] = val
        return cfg
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def build_system(sys_cfg):
    """Instantiate dynamical system from config."""
    name = sys_cfg.get("name", "lorenz96").lower()
    dt_out = float(sys_cfg.get("dt_out", 0.002))
    
    if name == "lorenz96":
        m = int(sys_cfg.get("m", 6))
        F = float(sys_cfg.get("F", 8.0))
        return Lorenz96(m=m, F=F, dt_out=dt_out)
    elif name in ("henon_heiles", "henonheiles"):
        dof = int(sys_cfg.get("dof", 3))
        energy = float(sys_cfg.get("energy", sys_cfg.get("energy_max", 0.12)))
        return HenonHeiles(dof=dof, energy_max=energy, dt_out=dt_out)
    elif name in ("coupled_duffing", "duffing"):
        d = int(sys_cfg.get("d", 3))
        kappa = float(sys_cfg.get("kappa", 0.15))
        return CoupledDuffing(d=d, kappa=kappa, dt_out=dt_out)
    elif name in ("integrable_torus", "torus"):
        m = int(sys_cfg.get("m", 6))
        d = int(sys_cfg.get("d", 3))
        basis = sys_cfg.get("basis", "identity")
        twist = float(sys_cfg.get("twist", 0.35))
        return IntegrableTorusFlow(m=m, d=d, basis=basis, twist=twist, dt_out=dt_out)
    elif name in ("abc", "abc_flow"):
        A = float(sys_cfg.get("A", 1.0))
        B = float(sys_cfg.get("B", 1.0))
        C = float(sys_cfg.get("C", 1.0))
        return ABCFlow(A=A, B=B, C=C, dt_out=dt_out)
    elif name in ("froeschle", "froeschle_map"):
        dof = int(sys_cfg.get("dof", 2))
        K = float(sys_cfg.get("K", 0.3))
        return Froeschle(dof=dof, K=K, dt_out=dt_out)
    elif name in ("cr3bp", "restricted_three_body"):
        mu = float(sys_cfg.get("mu", 0.01215))
        return CR3BP(mu=mu, dt_out=dt_out)
    elif name in ("fput", "fermi_pasta_ulam"):
        dof = int(sys_cfg.get("dof", 4))
        alpha = float(sys_cfg.get("alpha", 0.25))
        beta = float(sys_cfg.get("beta", 0.1))
        return FPUT(dof=dof, alpha=alpha, beta=beta, dt_out=dt_out)
    elif name in ("bickley_jet", "bickley", "bickley_flow"):
        U = float(sys_cfg.get("U", 1.0))
        L = float(sys_cfg.get("L", 1.0))
        eps = sys_cfg.get("eps", [0.075, 0.15, 0.3])
        if isinstance(eps, (int, float)):
            eps = [float(eps), float(eps), float(eps)]
        return BickleyJetFlow(U=U, L=L, eps=eps, dt_out=dt_out)
    elif name in ("kuramoto", "kuramoto_network"):
        m = int(sys_cfg.get("m", 8))
        K = float(sys_cfg.get("K", 2.0))
        coupling = str(sys_cfg.get("coupling", "global"))
        radius = sys_cfg.get("radius", None)
        freq_spread = float(sys_cfg.get("freq_spread", 1.0))
        return Kuramoto(m=m, K=K, dt_out=dt_out, coupling=coupling,
                        radius=None if radius is None else int(radius),
                        freq_spread=freq_spread)
    else:
        raise ValueError("Unknown system name: %s" % name)


def resolve_run_dir(cfg, outdir=None, run_tag=None):
    """Absolute run directory for a config (optional tag override)."""
    base_out = outdir or cfg.get("output_dir", "cesvima_output")
    if not os.path.isabs(base_out):
        base_out = os.path.join(_PROJ_ROOT, base_out)
    tag = run_tag or cfg.get("run_tag", "run_default")
    return os.path.join(base_out, tag), tag


def main():
    parser = argparse.ArgumentParser(description="Generate task shards for CESVIMA")
    parser.add_argument("--config", required=True, help="Path to config YAML or JSON")
    parser.add_argument("--outdir", default=None, help="Output directory for tasks (optional)")
    parser.add_argument("--run_tag", default=None,
                        help="Override run_tag (writes to a different directory)")
    parser.add_argument("--n_levels", type=int, default=None,
                        help="Override n_levels / path length L")
    args = parser.parse_args()

    cfg = load_config(args.config)
    system = build_system(cfg["system"])
    m = system.m

    grid_cfg = cfg["grid"]
    mode = grid_cfg.get("mode", "slice")
    n_cells = int(grid_cfg.get("n_cells", 400))
    n_levels = int(args.n_levels if args.n_levels is not None else cfg.get("n_levels", 2000))
    num_shards = int(cfg.get("num_shards", 20))
    # Safety clamp: CESVIMA has a strict MaxSubmitJobsPerAccount limit of 200 across the cluster
    num_shards = min(num_shards, 75)

    if mode == "slice":
        axes = tuple(grid_cfg.get("axes", [0, 1]))
        stride = int(grid_cfg.get("stride", 1))
        anchor = grid_cfg.get("anchor", None)
        cells, ticks = slice_cells(m, n_cells, axes, anchor=anchor, stride=stride,
                                  cell_offset=int(grid_cfg.get("cell_offset", 0)))
        n_side = ticks.size
        grid_shape = [n_side, n_side]
        extent = [float(system.lo[axes[0]]), float(system.hi[axes[0]]),
                  float(system.lo[axes[1]]), float(system.hi[axes[1]])]
    elif mode in ("volume_3d", "slice_3d", "3d"):
        axes = tuple(grid_cfg.get("axes", [0, 1, 2]))
        stride = int(grid_cfg.get("stride", 1))
        anchor = grid_cfg.get("anchor", None)
        cells, ticks = slice_cells_3d(m, n_cells, axes, anchor=anchor, stride=stride)
        n_side = ticks.size
        grid_shape = [n_side, n_side, n_side]
        extent = [float(system.lo[axes[0]]), float(system.hi[axes[0]]),
                  float(system.lo[axes[1]]), float(system.hi[axes[1]]),
                  float(system.lo[axes[2]]), float(system.hi[axes[2]])]
    elif mode == "sample":
        n_samples = int(grid_cfg.get("n_samples", 10000))
        seed = int(cfg.get("seed", 20260818))
        rng = np.random.default_rng(seed)
        cells = sample_cells(m, n_cells, n_samples, rng, system=system)
        ticks = np.arange(cells.shape[0])
        grid_shape = [cells.shape[0]]
        extent = [0.0, 1.0, 0.0, 1.0]
    else:
        raise ValueError("Unknown grid mode: %s" % mode)

    total_cells = cells.shape[0]

    # Restrict seed cells to the system's admissible domain.  For Henon-Heiles
    # this is the energy condition H <= energy_max, so the accessible region --
    # and hence the whole figure -- changes with the configured energy.  Without
    # this the energy parameter is inert in slice/volume mode and an energy sweep
    # silently recomputes the same field.
    admissible = np.asarray(system.valid_cells(cells, n_cells), dtype=bool)
    admissible_idx = np.flatnonzero(admissible)
    n_admissible = int(admissible_idx.size)
    if n_admissible == 0:
        raise ValueError(
            "No admissible seed cells: every cell in the %s grid fails "
            "%s.valid_cells(). Check the energy/domain settings in the config."
            % (mode, type(system).__name__)
        )

    num_shards = min(num_shards, n_admissible)
    
    run_dir, run_tag = resolve_run_dir(cfg, outdir=args.outdir, run_tag=args.run_tag)
    consolidated = os.path.join(run_dir, "results_consolidated.npz")
    if os.path.isfile(consolidated):
        raise SystemExit(
            "Refusing to overwrite existing run %s (%s). "
            "Pick a new run_tag (tau series uses <tag>_tauNN)."
            % (run_tag, consolidated)
        )

    shards_dir = os.path.join(run_dir, "shards")
    results_dir = os.path.join(run_dir, "results")
    figures_dir = os.path.join(run_dir, "figures")

    os.makedirs(shards_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Partition the admissible cells into shards.  Indices stay in full-grid
    # coordinates so the aggregator can scatter results back onto the grid and
    # leave inadmissible cells as NaN.
    shard_indices = np.array_split(admissible_idx, num_shards)
    
    for shard_id, idxs in enumerate(shard_indices):
        shard_file = os.path.join(shards_dir, f"shard_{shard_id:04d}.npz")
        np.savez_compressed(
            shard_file,
            shard_id=shard_id,
            indices=idxs,
            cells=cells[idxs]
        )

    # Save meta info
    meta = {
        "run_tag": run_tag,
        "system": cfg["system"],
        "m": m,
        "coord_names": list(system.coord_names),
        "grid": grid_cfg,
        "n_levels": n_levels,
        "dt_out": float(system.dt_out),
        "total_time": float(n_levels * system.dt_out),
        "total_cells": int(total_cells),
        "admissible_cells": n_admissible,
        "grid_shape": grid_shape,
        "extent": extent,
        "num_shards": int(num_shards),
        "q": cfg.get("q", None),
        "rule": cfg.get("rule", "strict"),
        "obs_noise_sigma": float(cfg.get("obs_noise_sigma", 0.0)),
        "compute_ftle": bool(cfg.get("compute_ftle", True)),
        "store_pair_diagnostics": bool(cfg.get("store_pair_diagnostics", False)),
        "clip_to_grid": bool(cfg.get("clip_to_grid", False)),
    }

    meta_file = os.path.join(run_dir, "meta.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("=" * 78)
    print(f"TASK GENERATION COMPLETE: {run_tag}")
    print("=" * 78)
    print(f"  System       : {system.name} (m={m})")
    print(f"  Total cells  : {total_cells} ({grid_shape})")
    print(f"  Admissible   : {n_admissible} ({100.0 * n_admissible / total_cells:.1f}% of grid)")
    print(f"  Shards       : {num_shards} array tasks")
    print(f"  Run dir      : {run_dir}")
    print(f"  Slurm array  : 0-{num_shards - 1}")
    print("=" * 78)


if __name__ == "__main__":
    main()
