# -*- coding: utf-8 -*-
"""
Slurm Array Worker for CESVIMA High-Dimensional FinSTOD Pipeline.
================================================================

Computes FinSTOD and variational FTLE for a single shard of cells,
saving compact binary results for subsequent aggregation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# Add project root to sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_ROOT = os.path.dirname(_HERE)
_PROJ_ROOT = os.path.dirname(_PIPELINE_ROOT)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from stod_nd import (
    ABCFlow,
    BickleyJetFlow,
    CR3BP,
    FPUT,
    PUBLISHED,
    STRICT,
    TYPE_T,
    CoupledDuffing,
    Froeschle,
    HenonHeiles,
    Kuramoto,
    Lorenz96,
    evaluate_cells,
    flow_ftle,
)
from stod_nd.flows import IntegrableTorusFlow
from stod_nd.progress import format_duration


def build_system(sys_cfg):
    """Instantiate dynamical system from config dictionary."""
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


def main():
    parser = argparse.ArgumentParser(description="CESVIMA FinSTOD Worker")
    parser.add_argument("--run_dir", required=True, help="Path to run directory containing meta.json")
    parser.add_argument("--task_id", type=int, default=None,
                        help="Task / Shard ID (defaults to SLURM_ARRAY_TASK_ID)")
    args = parser.parse_args()

    task_id = args.task_id
    if task_id is None:
        slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if slurm_id is None:
            raise ValueError("Task ID not provided and SLURM_ARRAY_TASK_ID not set.")
        task_id = int(slurm_id)

    run_dir = os.path.abspath(args.run_dir)
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"meta.json not found in {run_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    out_file = os.path.join(run_dir, "results", f"result_{task_id:04d}.npz")
    if os.path.isfile(out_file):
        print(f"TASK {task_id:04d} already complete ({out_file}), skip")
        return

    shard_path = os.path.join(run_dir, "shards", f"shard_{task_id:04d}.npz")
    if not os.path.isfile(shard_path):
        raise FileNotFoundError(f"Shard file not found: {shard_path}")

    shard_data = np.load(shard_path)
    shard_indices = shard_data["indices"]
    cells = shard_data["cells"]
    n_cells_shard = cells.shape[0]

    system = build_system(meta["system"])
    n_cells_grid = int(meta["grid"]["n_cells"])
    n_levels = int(meta["n_levels"])
    q = meta.get("q", None)
    rule_str = meta.get("rule", "strict").lower()
    rule = STRICT if rule_str == "strict" else PUBLISHED
    compute_ftle = bool(meta.get("compute_ftle", True))
    obs_noise_sigma = float(meta.get("obs_noise_sigma", 0.0))
    store_pairs = bool(meta.get("store_pair_diagnostics", False))
    clip_to_grid = bool(meta.get("clip_to_grid", False))

    print("=" * 78)
    print(f"STARTING WORKER TASK {task_id:04d} | Total Cells in Shard: {n_cells_shard}")
    print(f"System: {system.name} (m={system.m}) | n_levels={n_levels} | dt={system.dt_out:.4f}")
    if obs_noise_sigma > 0.0:
        print(f"Trajectory Observation Noise: sigma={obs_noise_sigma}")
    print(f"Host: {os.uname().nodename} | PID: {os.getpid()}")
    print("=" * 78)
    sys.stdout.flush()

    t_start = time.time()

    # 1. Evaluate FinSTOD
    print(f"[Task {task_id:04d}] Evaluating FinSTOD across 2m={2*system.m} neighbours...")
    sys.stdout.flush()
    
    stod_res = evaluate_cells(
        system=system,
        cells=cells,
        n_cells=n_cells_grid,
        n_levels=n_levels,
        q=q,
        rule=rule,
        progress=True,
        label=f"Task {task_id:04d}",
        locality=False,
        obs_noise_sigma=obs_noise_sigma,
        clip_to_grid=clip_to_grid,
    )

    cell_types = stod_res["cell_types"]
    cell_scores = stod_res["cell_scores"]
    pair_types = stod_res["pair_types"]
    pair_kstar = stod_res["pair_kstar"]
    pair_valid = stod_res["pair_valid"]

    # Termination statistics & local torus dimension count
    terminating_count = np.sum((pair_types == TYPE_T) & pair_valid, axis=1)
    
    # Best neighbour k_star
    effective_types = np.where(pair_valid, pair_types, -1)
    best_type = effective_types.max(axis=1)
    is_best = effective_types == best_type[:, None]
    k_star_best = np.where(is_best, pair_kstar, 0).max(axis=1)

    t_stod = time.time() - t_start
    print(f"[Task {task_id:04d}] FinSTOD done in {format_duration(t_stod)} ({n_cells_shard / max(1e-6, t_stod):.1f} cells/s)")
    sys.stdout.flush()

    # 2. Evaluate FTLE (optional)
    ftle_vals = np.zeros(n_cells_shard, dtype=np.float64)
    t_ftle = 0.0
    if compute_ftle:
        print(f"[Task {task_id:04d}] Evaluating Variational FTLE benchmark...")
        sys.stdout.flush()
        t_ftle_start = time.time()
        ftle_vals = flow_ftle(
            system=system,
            cells=cells,
            n_cells=n_cells_grid,
            n_levels=n_levels,
            chunk_size=1000,
            progress=True,
            label=f"Task {task_id:04d}",
        )
        t_ftle = time.time() - t_ftle_start
        print(f"[Task {task_id:04d}] FTLE done in {format_duration(t_ftle)} ({n_cells_shard / max(1e-6, t_ftle):.1f} cells/s)")
        sys.stdout.flush()

    total_time = time.time() - t_start

    # Save shard result
    payload = dict(
        task_id=task_id,
        indices=shard_indices,
        cells=cells,
        cell_types=cell_types,
        cell_scores=cell_scores,
        k_star_best=k_star_best,
        terminating_count=terminating_count,
        ftle=ftle_vals,
        wall_seconds=total_time,
        stod_seconds=t_stod,
        ftle_seconds=t_ftle,
    )
    if store_pairs:
        payload["pair_kstar"] = pair_kstar
        payload["pair_valid"] = pair_valid
        payload["pair_types"] = pair_types
    np.savez_compressed(out_file, **payload)

    print("=" * 78)
    print(f"TASK {task_id:04d} COMPLETED in {format_duration(total_time)} -> Saved to {out_file}")
    print("=" * 78)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
