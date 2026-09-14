# -*- coding: utf-8 -*-
"""
Result Aggregator & Publication Figure Generator for CESVIMA Pipeline.
======================================================================

Stitches worker shard outputs into complete grid fields, computes all
normalisations (raw, log, segmented linear, segmented log, k*/L, d_hat),
evaluates Spearman correlations against FTLE, and renders publication figures.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

# Set matplotlib backend to non-interactive Agg for headless HPC clusters
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_ROOT = os.path.dirname(_HERE)
_PROJ_ROOT = os.path.dirname(_PIPELINE_ROOT)
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from stod_nd import segmented_normalise, spearman


def main():
    parser = argparse.ArgumentParser(description="Aggregate CESVIMA results and generate figures")
    parser.add_argument("--run_dir", required=True, help="Path to run directory containing meta.json")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for publication figures")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Aggregate even when worker shards are missing. The run is then flagged "
             "incomplete in summary.json and excluded from cross-system synthesis.",
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    meta_path = os.path.join(run_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"meta.json not found in {run_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    total_cells = int(meta["total_cells"])
    # Cells outside the system's admissible domain are never dispatched, so the
    # completeness target is the admissible count, not the full grid.
    admissible_cells = int(meta.get("admissible_cells", total_cells))
    num_shards = int(meta["num_shards"])
    grid_shape = meta["grid_shape"]
    extent = meta.get("extent", [0.0, 1.0, 0.0, 1.0])
    system_cfg = meta["system"]
    m = int(meta["m"])
    n_levels = int(meta["n_levels"])
    compute_ftle = bool(meta.get("compute_ftle", True))

    results_dir = os.path.join(run_dir, "results")
    figures_dir = os.path.join(run_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 78)
    print(f"AGGREGATING RESULTS FOR: {meta.get('run_tag', 'run')}")
    print(f"Run dir: {run_dir}")
    print(f"Total cells: {total_cells} | Expected shards: {num_shards}")
    print("=" * 78)

    # Find result files
    result_files = sorted(glob.glob(os.path.join(results_dir, "result_*.npz")))
    found_shards = len(result_files)
    print(f"Found {found_shards}/{num_shards} shard result files.")

    missing = []
    if found_shards < num_shards:
        present_ids = set()
        for fpath in result_files:
            fname = os.path.basename(fpath)
            try:
                sid = int(fname.replace("result_", "").replace(".npz", ""))
                present_ids.add(sid)
            except ValueError:
                pass
        missing = sorted(set(range(num_shards)) - present_ids)
        shown = ", ".join(str(i) for i in missing[:20])
        if len(missing) > 20:
            shown += ", ... (%d more)" % (len(missing) - 20)
        msg = (
            "INCOMPLETE RUN: %d of %d worker shards produced no result file.\n"
            "  Missing task IDs: %s\n"
            "  Aggregating anyway would yield a scalar field defined on only %.1f%% of "
            "the grid, and any correlation computed from it is not interpretable.\n"
            "  Inspect the worker logs (a TIMEOUT at the walltime limit is the usual "
            "cause), resubmit the missing array indices, then re-run this aggregator.\n"
            "  To aggregate the partial field deliberately, pass --allow-incomplete."
            % (
                num_shards - found_shards,
                num_shards,
                shown,
                100.0 * found_shards / max(1, num_shards),
            )
        )
        if not args.allow_incomplete:
            raise RuntimeError(msg)
        print("WARNING: " + msg)

    # Allocate arrays
    all_cells = np.zeros((total_cells, m), dtype=np.int64)
    all_types = np.full(total_cells, -1, dtype=np.int8)
    all_scores = np.full(total_cells, np.nan, dtype=np.float64)
    all_kstar = np.full(total_cells, -1, dtype=np.int32)
    all_terminating = np.full(total_cells, 0, dtype=np.int32)
    all_ftle = np.full(total_cells, np.nan, dtype=np.float64)
    all_pair_kstar = None
    all_pair_valid = None
    all_pair_types = None
    wall_times = []
    stod_times = []
    ftle_times = []

    filled_mask = np.zeros(total_cells, dtype=bool)

    for fpath in result_files:
        d = np.load(fpath)
        idxs = d["indices"]
        all_cells[idxs] = d["cells"]
        all_types[idxs] = d["cell_types"]
        all_scores[idxs] = d["cell_scores"]
        all_kstar[idxs] = d["k_star_best"]
        all_terminating[idxs] = d["terminating_count"]
        if "ftle" in d:
            all_ftle[idxs] = d["ftle"]
        if "pair_kstar" in d:
            n_pair = int(d["pair_kstar"].shape[1])
            if all_pair_kstar is None:
                all_pair_kstar = np.full((total_cells, n_pair), -1, dtype=np.int32)
                all_pair_valid = np.zeros((total_cells, n_pair), dtype=bool)
                all_pair_types = np.full((total_cells, n_pair), -1, dtype=np.int8)
            all_pair_kstar[idxs] = d["pair_kstar"]
            all_pair_valid[idxs] = d["pair_valid"]
            all_pair_types[idxs] = d["pair_types"]
        if "wall_seconds" in d:
            wall_times.append(float(d["wall_seconds"]))
        if "stod_seconds" in d:
            stod_times.append(float(d["stod_seconds"]))
        if "ftle_seconds" in d:
            ftle_times.append(float(d["ftle_seconds"]))
        filled_mask[idxs] = True

    n_filled = int(filled_mask.sum())
    print(f"Successfully assembled {n_filled}/{admissible_cells} admissible cells "
          f"({100.0 * n_filled / max(1, admissible_cells):.1f}%) "
          f"out of a {total_cells}-cell grid.")
    if n_filled < admissible_cells and not args.allow_incomplete:
        raise RuntimeError(
            "INCOMPLETE RUN: %d admissible cells have no result even though all "
            "%d shard files are present. The shard files are likely truncated or "
            "were written by a mismatched task generation. Re-generate tasks and "
            "resubmit, or pass --allow-incomplete to proceed anyway."
            % (admissible_cells - n_filled, found_shards)
        )
    if wall_times:
        print(f"Total worker compute time: {sum(wall_times):.1f}s | Max shard time: {max(wall_times):.1f}s")
    if stod_times and ftle_times and sum(ftle_times) > 0:
        t_s, t_f = sum(stod_times), sum(ftle_times)
        print(f"Cost split: FinSTOD {t_s:.1f}s vs FTLE {t_f:.1f}s (FinSTOD/FTLE = {t_s / t_f:.2f}x)")

    # 1. Normalisations & Field Calculations
    print("Computing transformations (raw, log, linear segmented, log segmented, k*/L, d_hat)...")
    raw_scores = all_scores
    log_raw_scores = np.log1p(np.maximum(0.0, raw_scores))
    
    linear_segmented, P_lin = segmented_normalise(all_types, raw_scores, log=False)
    log_segmented, P_log = segmented_normalise(all_types, raw_scores, log=True)
    
    kstar_ratio = all_kstar / max(1.0, float(n_levels - 1))
    d_hat = all_terminating / 2.0
    d_hat_graded = None
    if all_pair_kstar is not None:
        denom = float(max(n_levels - 1, 1))
        depth = np.clip(all_pair_kstar.astype(np.float64), 0.0, denom) / denom
        contrib = np.where(all_pair_valid, 1.0 - depth, np.nan)
        d_hat_graded = 0.5 * np.nansum(contrib, axis=1)

    # Type breakdown
    n_t = int(np.sum(all_types == 0))
    n_uc = int(np.sum(all_types == 1))
    n_uu = int(np.sum(all_types == 2))
    
    # 2. Correlations with FTLE
    correlations = {}
    if compute_ftle and np.any(~np.isnan(all_ftle)):
        valid_c = ~np.isnan(all_ftle) & ~np.isnan(raw_scores)
        if valid_c.any():
            correlations["spearman_raw_vs_ftle"] = float(spearman(raw_scores[valid_c], all_ftle[valid_c]))
            correlations["spearman_log_raw_vs_ftle"] = float(spearman(log_raw_scores[valid_c], all_ftle[valid_c]))
            correlations["spearman_linear_seg_vs_ftle"] = float(spearman(linear_segmented[valid_c], all_ftle[valid_c]))
            correlations["spearman_log_seg_vs_ftle"] = float(spearman(log_segmented[valid_c], all_ftle[valid_c]))
            correlations["spearman_kstar_vs_ftle"] = float(spearman(kstar_ratio[valid_c], all_ftle[valid_c]))

            print("-" * 50)
            print("SPEARMAN RANK CORRELATIONS WITH FTLE:")
            for k, v in correlations.items():
                print(f"  {k:32s}: {v:+.4f}")
            print("-" * 50)

    # 3. Save Consolidated NPZ
    consolidated_path = os.path.join(run_dir, "results_consolidated.npz")
    payload = dict(
        cells=all_cells,
        cell_types=all_types,
        raw_scores=raw_scores,
        log_raw_scores=log_raw_scores,
        linear_segmented=linear_segmented,
        log_segmented=log_segmented,
        kstar_best=all_kstar,
        kstar_ratio=kstar_ratio,
        terminating_count=all_terminating,
        d_hat=d_hat,
        ftle=all_ftle,
        P_linear=P_lin,
        P_log=P_log,
    )
    if all_pair_kstar is not None:
        payload["pair_kstar"] = all_pair_kstar
        payload["pair_valid"] = all_pair_valid
        payload["pair_types"] = all_pair_types
        payload["d_hat_graded"] = d_hat_graded
    np.savez_compressed(consolidated_path, **payload)
    print(f"Saved consolidated data to: {consolidated_path}")

    # Summary JSON
    summary = {
        "run_tag": meta.get("run_tag", "run"),
        "system": system_cfg,
        "m": m,
        "total_cells": total_cells,
        "filled_cells": n_filled,
        "admissible_cells": admissible_cells,
        "complete": bool(found_shards == num_shards and n_filled == admissible_cells),
        "shards_expected": num_shards,
        "shards_found": found_shards,
        "missing_task_ids": missing,
        "type_counts": {"T": n_t, "UC": n_uc, "UU": n_uu},
        "P_linear": float(P_lin),
        "P_log": float(P_log),
        "correlations": correlations,
        "total_worker_cpu_seconds": sum(wall_times) if wall_times else 0.0,
        "finstod_cpu_seconds": sum(stod_times) if stod_times else 0.0,
        "ftle_cpu_seconds": sum(ftle_times) if ftle_times else 0.0,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 4. Generate Publication Figures (if 2D slice)
    if len(grid_shape) == 2:
        nx, ny = grid_shape
        print(f"Generating high-resolution 2D publication figures ({nx} x {ny})...")
        
        # Reshape fields
        mat_raw = raw_scores.reshape(nx, ny)
        mat_log_raw = log_raw_scores.reshape(nx, ny)
        mat_lin_seg = linear_segmented.reshape(nx, ny)
        mat_log_seg = log_segmented.reshape(nx, ny)
        mat_kstar = kstar_ratio.reshape(nx, ny)
        mat_dhat = d_hat.reshape(nx, ny)
        mat_ftle = all_ftle.reshape(nx, ny)

        axes_info = meta["grid"].get("axes", [0, 1])
        cnames = meta.get("coord_names", [f"x_{i}" for i in range(m)])
        xlab = f"${cnames[axes_info[0]]}$"
        ylab = f"${cnames[axes_info[1]]}$"
        sys_name = system_cfg.get("name", "System").replace("_", " ").title()

        # Figure 1: Comprehensive 6-Panel Suite
        fig, axs = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
        fig.suptitle(f"{sys_name} ($m={m}$) — High-Definition Coherent Structure Analysis\n"
                     f"Resolution: {nx}$\\times${ny} | $L={n_levels}$ | $\\Delta t={meta['dt_out']}$",
                     fontsize=15, fontweight="bold")

        # Robust clim for FTLE to avoid escape singularities (percentile 2 to 98)
        finite_ftle = mat_ftle[np.isfinite(mat_ftle)]
        clim_ftle = None
        if finite_ftle.size > 0:
            p2, p98 = np.percentile(finite_ftle, [2, 98])
            if p98 > p2:
                clim_ftle = (float(p2), float(p98))

        panels = [
            (axs[0, 0], mat_log_seg, f"FinSTOD Log-Segmented ($P={P_log:.2f}$)", "magma", (0.0, 1.0)),
            (axs[0, 1], mat_lin_seg, f"FinSTOD Linear Segmented ($P={P_lin:.2f}$)", "viridis", (0.0, 1.0)),
            (axs[0, 2], mat_log_raw, r"FinSTOD Raw $\log(1+s)$", "inferno", None),
            (axs[1, 0], mat_ftle, f"Variational FTLE ($\\rho={correlations.get('spearman_log_seg_vs_ftle', 0.0):+.3f}$)", "plasma", clim_ftle),
            (axs[1, 1], mat_kstar, r"Termination Ratio $k^*/L$", "coolwarm", (0.0, 1.0)),
            (axs[1, 2], mat_dhat, r"Torus Dimension $\hat{d}$ ($2\hat{d}$ Terminating Neighbours)", "tab10", (0, m)),
        ]

        for ax, mat, title, cmap, clim in panels:
            im = ax.imshow(mat.T, origin="lower", extent=extent, cmap=cmap, aspect="auto")
            if clim:
                im.set_clim(*clim)
            ax.set_title(title, fontsize=12, fontweight="semibold")
            ax.set_xlabel(xlab, fontsize=11)
            ax.set_ylabel(ylab, fontsize=11)
            cbar = fig.colorbar(im, ax=ax, shrink=0.85)
            cbar.ax.tick_params(labelsize=9)

        fig_path_6p = os.path.join(figures_dir, "figure_comparison_6panel.png")
        fig_path_6p_pdf = os.path.join(figures_dir, "figure_comparison_6panel.pdf")
        fig.savefig(fig_path_6p, dpi=args.dpi)
        fig.savefig(fig_path_6p_pdf)
        plt.close(fig)
        print(f"Saved 6-panel figure to: {fig_path_6p}")

        # Figure 2: FinSTOD Log-Segmented vs FTLE High-Res Dual Panel
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
        im1 = ax1.imshow(mat_log_seg.T, origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
        ax1.set_title(f"FinSTOD (Log-Segmented Hierarchy)\n$P={P_log:.3f}$", fontsize=12, fontweight="bold")
        ax1.set_xlabel(xlab, fontsize=11)
        ax1.set_ylabel(ylab, fontsize=11)
        fig.colorbar(im1, ax=ax1, shrink=0.85, label="Normalised FinSTOD [0, 1]")

        im2 = ax2.imshow(mat_ftle.T, origin="lower", extent=extent, cmap="plasma", aspect="auto")
        if clim_ftle:
            im2.set_clim(*clim_ftle)
        ax2.set_title(f"Variational FTLE (Tangent-Space)\nSpearman $\\rho = {correlations.get('spearman_log_seg_vs_ftle', 0.0):+.3f}$",
                      fontsize=12, fontweight="bold")
        ax2.set_xlabel(xlab, fontsize=11)
        ax2.set_ylabel(ylab, fontsize=11)
        fig.colorbar(im2, ax=ax2, shrink=0.85, label="FTLE Exponent")

        fig_path_dual = os.path.join(figures_dir, "figure_finstod_vs_ftle_dual.png")
        fig_path_dual_pdf = os.path.join(figures_dir, "figure_finstod_vs_ftle_dual.pdf")
        fig.savefig(fig_path_dual, dpi=args.dpi)
        fig.savefig(fig_path_dual_pdf)
        plt.close(fig)
        print(f"Saved dual comparison figure to: {fig_path_dual}")

        # Figure 3: Correlation Scatter / Hexbin Plot
        if compute_ftle and np.any(~np.isnan(all_ftle)):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
            
            # Log raw vs FTLE
            hb1 = ax1.hexbin(all_ftle, log_raw_scores, gridsize=60, cmap="Blues", mincnt=1, bins="log")
            ax1.set_xlabel("Variational FTLE", fontsize=11)
            ax1.set_ylabel(r"FinSTOD $\log(1+s)$", fontsize=11)
            ax1.set_title(f"Log-FinSTOD vs FTLE ($\\rho = {correlations.get('spearman_log_raw_vs_ftle', 0.0):+.3f}$)", fontsize=11, fontweight="bold")
            fig.colorbar(hb1, ax=ax1, label="log10(count)")

            # Log segmented vs FTLE
            hb2 = ax2.hexbin(all_ftle, log_segmented, gridsize=60, cmap="Reds", mincnt=1, bins="log")
            ax2.set_xlabel("Variational FTLE", fontsize=11)
            ax2.set_ylabel("Log-Segmented FinSTOD [0, 1]", fontsize=11)
            ax2.set_title(f"Segmented FinSTOD vs FTLE ($\\rho = {correlations.get('spearman_log_seg_vs_ftle', 0.0):+.3f}$)", fontsize=11, fontweight="bold")
            fig.colorbar(hb2, ax=ax2, label="log10(count)")

            fig_path_scatter = os.path.join(figures_dir, "figure_correlation_density.png")
            fig.savefig(fig_path_scatter, dpi=args.dpi)
            plt.close(fig)
            print(f"Saved correlation density figure to: {fig_path_scatter}")

    elif len(grid_shape) == 3:
        nx, ny, nz = grid_shape
        print(f"Generating 3D volumetric publication figures & interactive visualizer ({nx} x {ny} x {nz})...")

        vol_finstod = log_segmented.reshape(nx, ny, nz)
        vol_kstar = kstar_ratio.reshape(nx, ny, nz)
        vol_ftle = all_ftle.reshape(nx, ny, nz)

        axes_info = meta["grid"].get("axes", [0, 1, 2])
        cnames = meta.get("coord_names", [f"x_{i}" for i in range(m)])
        lab_x = f"${cnames[axes_info[0]]}$"
        lab_y = f"${cnames[axes_info[1]]}$"
        lab_z = f"${cnames[axes_info[2]]}$"
        sys_name = system_cfg.get("name", "System").replace("_", " ").title()

        # Figure 1: Orthogonal 3-Slice Tri-Panel (XY, XZ, YZ midplanes)
        fig, axs = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
        fig.suptitle(f"{sys_name} ($m={m}$) — 3D Volumetric FinSTOD Orthogonal Cross-Sections",
                     fontsize=14, fontweight="bold")

        im0 = axs[0].imshow(vol_finstod[:, :, nz // 2].T, origin="lower", cmap="magma", vmin=0, vmax=1)
        axs[0].set_title(f"XY Midplane ({lab_z} = mid)", fontsize=11, fontweight="semibold")
        axs[0].set_xlabel(lab_x)
        axs[0].set_ylabel(lab_y)
        fig.colorbar(im0, ax=axs[0], shrink=0.85)

        im1 = axs[1].imshow(vol_finstod[:, ny // 2, :].T, origin="lower", cmap="magma", vmin=0, vmax=1)
        axs[1].set_title(f"XZ Midplane ({lab_y} = mid)", fontsize=11, fontweight="semibold")
        axs[1].set_xlabel(lab_x)
        axs[1].set_ylabel(lab_z)
        fig.colorbar(im1, ax=axs[1], shrink=0.85)

        im2 = axs[2].imshow(vol_finstod[nx // 2, :, :].T, origin="lower", cmap="magma", vmin=0, vmax=1)
        axs[2].set_title(f"YZ Midplane ({lab_x} = mid)", fontsize=11, fontweight="semibold")
        axs[2].set_xlabel(lab_y)
        axs[2].set_ylabel(lab_z)
        fig.colorbar(im2, ax=axs[2], shrink=0.85)

        fig_path_3slice = os.path.join(figures_dir, "figure_volume_ortho_slices.png")
        fig_path_3slice_pdf = os.path.join(figures_dir, "figure_volume_ortho_slices.pdf")
        fig.savefig(fig_path_3slice, dpi=args.dpi)
        fig.savefig(fig_path_3slice_pdf)
        plt.close(fig)
        print(f"Saved 3D orthoslice figure to: {fig_path_3slice}")

        # Figure 2: Interactive 3D HTML Volume with Plotly (if installed)
        try:
            import plotly.graph_objects as go
            X, Y, Z = np.mgrid[0:nx, 0:ny, 0:nz]
            fig_3d = go.Figure(data=go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=vol_finstod.flatten(),
                isomin=0.4,
                isomax=0.95,
                surface_count=5,
                colorscale='Magma',
                caps=dict(x_show=False, y_show=False, z_show=False),
            ))
            fig_3d.update_layout(
                title=f"Interactive 3D FinSTOD Isosurfaces: {sys_name} ($m={m}$)",
                scene=dict(
                    xaxis_title=lab_x,
                    yaxis_title=lab_y,
                    zaxis_title=lab_z,
                ),
                margin=dict(l=0, r=0, b=0, t=40),
            )
            html_path = os.path.join(figures_dir, "figure_interactive_3d.html")
            fig_3d.write_html(html_path)
            print(f"Generated interactive 3D HTML viewer: {html_path}")
        except ImportError:
            pass

    print("=" * 78)
    print("AGGREGATION & FIGURE GENERATION COMPLETE!")
    print(f"Results stored in: {run_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
