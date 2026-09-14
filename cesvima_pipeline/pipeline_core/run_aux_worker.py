# -*- coding: utf-8 -*-
"""Auxiliary CESVIMA workers.

Re-integrate existing production shards without overwriting FinSTOD fields.

Kinds
-----
frequency_plane   NAFF-style ω1, ω2 (Phase 10)
fput_lifetime     mean S-bar + τ_eq (Phase 12)
interp_bundle     time-aware FTLE, FLI, LD (p=1 and p=0.5), FTRN;
                  plus FPUT mode energies or Duffing wells when the system fits
duffing_basins    long state-only integration; well identity of each oscillator
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_ROOT = os.path.dirname(_HERE)
_PROJ_ROOT = os.path.dirname(_PIPELINE_ROOT)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from generate_tasks import load_config
from run_worker import build_system
from stod_nd.indicators import flow_indicator_bundle
from stod_nd.progress import format_duration, print_loop_progress


def iter_aux_jobs(config_path):
    """Yield (source_tag, aux_tag, kind, n_levels_or_empty)."""
    cfg = load_config(config_path)
    kind_cfg = cfg.get("kind")
    if kind_cfg == "fput_sbar_aux":
        kind = cfg.get("aux_kind", "fput_lifetime")
        prefix = cfg.get("aux_prefix", "aux_sbar_")
        n_ov = ""
        for src in cfg["sources"]:
            yield str(src), prefix + str(src), kind, n_ov
        return
    if kind_cfg != "interp_aux":
        raise ValueError("%s is not fput_sbar_aux or interp_aux" % config_path)
    kind = cfg.get("aux_kind", "interp_bundle")
    prefix = cfg.get("aux_prefix", "aux_interp_")
    n_ov = "" if cfg.get("n_levels") in (None, "") else str(int(cfg["n_levels"]))
    for src in cfg["sources"]:
        yield str(src), prefix + str(src), kind, n_ov


def iter_sbar_aux_jobs(config_path):
    """Backward-compatible 3-tuple iterator."""
    for src, aux, kind, _n in iter_aux_jobs(config_path):
        yield src, aux, kind


def dominant_omega(signal, dt):
    x = signal - np.mean(signal, axis=0)
    spec = np.fft.rfft(x, axis=0)
    mag = np.abs(spec)
    mag[0] = 0.0
    k = np.argmax(mag, axis=0)
    freqs = np.fft.rfftfreq(signal.shape[0], d=dt)
    k0 = np.clip(k, 1, mag.shape[0] - 2)
    rows = np.arange(signal.shape[1])
    a, b, c = mag[k0 - 1, rows], mag[k0, rows], mag[k0 + 1, rows]
    denom = (a - 2 * b + c)
    delta = np.where(np.abs(denom) > 1e-12, 0.5 * (a - c) / denom, 0.0)
    f = (k0 + delta) * freqs[1]
    return 2.0 * np.pi * f


def frequency_plane_shard(system, cells, n_cells_grid, n_levels, label=""):
    dt = float(system.dt_out)
    system.reset()
    states = system.cell_to_state(cells, n_cells_grid)
    q1 = np.empty((n_levels, cells.shape[0]), dtype=np.float32)
    q2 = np.empty_like(q1)
    t0 = time.time()
    for t in range(n_levels):
        states = system.step(states)
        q1[t] = states[:, 0]
        q2[t] = states[:, 1]
        print_loop_progress(label, t + 1, n_levels, time.time() - t0, kind="omega")
    return dominant_omega(q1.astype(np.float64), dt), dominant_omega(q2.astype(np.float64), dt)


def fput_lifetime_shard(system, cells, n_cells_grid, n_levels, sample_every=10,
                        s_thresh=0.5, label=""):
    system.reset()
    states = system.cell_to_state(cells, n_cells_grid)
    k = cells.shape[0]
    tau = np.full(k, n_levels, dtype=np.float64)
    found = np.zeros(k, dtype=bool)
    acc = np.zeros(k, dtype=np.float64)
    n_acc = 0
    t0 = time.time()
    for t in range(n_levels):
        states = system.step(states)
        print_loop_progress(label, t + 1, n_levels, time.time() - t0, kind="S-bar")
        if t % sample_every != 0:
            continue
        S = system.spectral_entropy(states)
        hit = (~found) & (S >= s_thresh)
        tau[hit] = float(t + 1)
        found[hit] = True
        if t >= n_levels // 2:
            acc += S
            n_acc += 1
    return acc / max(n_acc, 1), tau


def duffing_basins_shard(system, cells, n_cells_grid, n_levels, label=""):
    """Long integration; pack each oscillator's well as a base-3 integer."""
    system.reset()
    states = system.cell_to_state(cells, n_cells_grid)
    t0 = time.time()
    for t in range(int(n_levels)):
        states = system.step(states)
        print_loop_progress(label, t + 1, n_levels, time.time() - t0, kind="basins")
    d = int(getattr(system, "d", system.m // 2))
    x = states[:, :d]
    wells = np.sign(x)
    wells[np.abs(x) < 0.2] = 0.0
    code = np.zeros(cells.shape[0], dtype=np.int32)
    for j in range(d):
        code += (wells[:, j].astype(np.int32) + 1) * (3 ** j)
    return code, x.astype(np.float32)


def _extras_for(system):
    extras = []
    name = type(system).__name__.lower()
    if "fput" in name:
        extras.append("fput_modes")
    if "duffing" in name:
        extras.append("duffing_basins")
    return tuple(extras)


def worker(source_dir, aux_dir, kind, task_id, n_levels_override=None):
    meta = json.load(open(os.path.join(source_dir, "meta.json")))
    shard_path = os.path.join(source_dir, "shards", f"shard_{task_id:04d}.npz")
    shard = np.load(shard_path)
    cells = shard["cells"]
    idxs = shard["indices"]
    system = build_system(meta["system"])
    n_cells_grid = int(meta["grid"]["n_cells"])
    n_levels = int(n_levels_override or meta["n_levels"])
    os.makedirs(os.path.join(aux_dir, "results"), exist_ok=True)
    out_path = os.path.join(aux_dir, "results", f"result_{task_id:04d}.npz")
    if os.path.isfile(out_path):
        print("AUX %s task %04d already complete, skip" % (kind, task_id))
        return
    t0 = time.time()
    out = dict(task_id=task_id, indices=idxs, cells=cells)
    label = "Task %04d" % task_id
    if kind == "frequency_plane":
        w1, w2 = frequency_plane_shard(system, cells, n_cells_grid, n_levels, label=label)
        out["omega1"] = w1
        out["omega2"] = w2
    elif kind == "fput_lifetime":
        mean_S, tau = fput_lifetime_shard(system, cells, n_cells_grid, n_levels, label=label)
        out["mean_S"] = mean_S
        out["tau_eq"] = tau
    elif kind == "interp_bundle":
        bundle = flow_indicator_bundle(
            system, cells, n_cells_grid, n_levels,
            chunk_size=1000, progress=True,
            label=label,
            extras=_extras_for(system),
        )
        out.update(bundle)
    elif kind == "duffing_basins":
        code, x_end = duffing_basins_shard(
            system, cells, n_cells_grid, n_levels, label=label)
        out["well_code"] = code
        out["x_end"] = x_end
    else:
        raise ValueError("unknown aux kind: %s" % kind)
    out["wall_seconds"] = time.time() - t0
    np.savez_compressed(out_path, **out)
    print(f"AUX {kind} task {task_id:04d} done in {format_duration(out['wall_seconds'])}")


def aggregate(source_dir, aux_dir, kind):
    meta = json.load(open(os.path.join(source_dir, "meta.json")))
    total = int(meta["total_cells"])
    num_shards = int(meta["num_shards"])
    files = sorted(glob.glob(os.path.join(aux_dir, "results", "result_*.npz")))
    if len(files) != num_shards:
        raise RuntimeError("aux incomplete: %d/%d shards" % (len(files), num_shards))
    skip = {"task_id", "indices", "cells", "wall_seconds"}
    first = np.load(files[0])
    payload = {"cells": np.zeros((total, int(meta["m"])), dtype=np.int64)}
    for key in first.files:
        if key in skip:
            continue
        arr = first[key]
        shape = (total,) + tuple(arr.shape[1:])
        if np.issubdtype(arr.dtype, np.floating):
            payload[key] = np.full(shape, np.nan, dtype=arr.dtype)
        else:
            payload[key] = np.full(shape, -1, dtype=arr.dtype)
    for fpath in files:
        d = np.load(fpath)
        idxs = d["indices"]
        payload["cells"][idxs] = d["cells"]
        for key in payload:
            if key != "cells" and key in d:
                payload[key][idxs] = d[key]
    os.makedirs(aux_dir, exist_ok=True)
    np.savez_compressed(os.path.join(aux_dir, "results_consolidated.npz"), **payload)
    aux_meta = {
        "run_tag": os.path.basename(aux_dir.rstrip("/")),
        "aux_kind": kind,
        "source_tag": meta.get("run_tag"),
        "source_dir": source_dir,
        "n_levels": meta["n_levels"],
        "grid_shape": meta["grid_shape"],
        "num_shards": num_shards,
    }
    with open(os.path.join(aux_dir, "meta.json"), "w") as f:
        json.dump(aux_meta, f, indent=2)
    print("Wrote", os.path.join(aux_dir, "results_consolidated.npz"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list_config", default=None,
                   help="List aux jobs from a YAML and exit")
    p.add_argument("--source_dir", default=None)
    p.add_argument("--aux_dir", default=None)
    p.add_argument("--kind", default=None,
                   choices=["frequency_plane", "fput_lifetime",
                            "interp_bundle", "duffing_basins"])
    p.add_argument("--task_id", type=int, default=None)
    p.add_argument("--n_levels", type=int, default=None)
    p.add_argument("--aggregate", action="store_true")
    args = p.parse_args()
    if args.list_config:
        for src, aux, kind, n_ov in iter_aux_jobs(args.list_config):
            print("%s\t%s\t%s\t%s" % (src, aux, kind, n_ov))
        return
    if not args.source_dir or not args.aux_dir or not args.kind:
        p.error("--source_dir, --aux_dir, and --kind are required unless --list_config")
    source_dir = os.path.abspath(args.source_dir)
    aux_dir = os.path.abspath(args.aux_dir)
    n_ov = args.n_levels
    if n_ov is None:
        env = os.environ.get("AUX_N_LEVELS", "").strip()
        n_ov = int(env) if env else None
    if args.aggregate:
        aggregate(source_dir, aux_dir, args.kind)
        return
    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    worker(source_dir, aux_dir, args.kind, task_id, n_levels_override=n_ov)


if __name__ == "__main__":
    main()
