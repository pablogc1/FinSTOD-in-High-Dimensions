# -*- coding: utf-8 -*-
"""
Cross-System Synthesizer for High-Dimensional FinSTOD Paper.
============================================================

Reads all completed study directories in `cesvima_output/`, computes comparative
metrics, formats LaTeX and Markdown publication tables, and generates a
multi-system summary panel.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def format_duration(seconds):
    """Format seconds into readable Xh Ym Zs format."""
    if seconds is None or np.isnan(seconds):
        return "N/A"
    sec = float(seconds)
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    rem_sec = int(sec % 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {rem_sec:02d}s"
    elif minutes > 0:
        return f"{minutes}m {rem_sec:02d}s"
    else:
        return f"{sec:.1f}s"


def main():
    parser = argparse.ArgumentParser(description="Cross-system synthesis and summary generator")
    parser.add_argument("--output_dir", default="../cesvima_output", help="Path to cesvima_output directory")
    args = parser.parse_args()

    out_base = os.path.abspath(args.output_dir)
    if not os.path.isdir(out_base):
        print(f"Directory not found: {out_base}")
        return

    summary_files = sorted(glob.glob(os.path.join(out_base, "*/summary.json")))
    if not summary_files:
        print(f"No summary.json files found in {out_base}")
        return

    print("=" * 78)
    print("SYNTHESIZING CROSS-SYSTEM FINSTOD HPC RESULTS")
    print(f"Base folder: {out_base}")
    print(f"Found {len(summary_files)} study summaries.")
    print("=" * 78)

    rows = []
    run_dirs = []
    skipped = []

    for sfile in summary_files:
        with open(sfile, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        rdir = os.path.dirname(sfile)
        tag = data.get("run_tag", os.path.basename(rdir))

        # A run whose workers did not all finish yields a scalar field defined on
        # only part of the grid; its type counts and correlations are computed
        # from that fragment and would be misleading alongside complete runs.
        filled = data.get("filled_cells", 0)
        expected = data.get("total_cells", 0)
        complete = data.get("complete")
        if complete is None:  # summaries written before completeness tracking
            complete = expected > 0 and filled == expected
        if not complete:
            skipped.append((tag, filled, expected, data.get("missing_task_ids", [])))
            continue

        run_dirs.append(rdir)
        sys_cfg = data.get("system", {})
        sys_name = sys_cfg.get("name", "Unknown")
        m = data.get("m", 0)
        total_cells = data.get("total_cells", 0)
        types = data.get("type_counts", {})
        p_lin = data.get("P_linear", 0.0)
        p_log = data.get("P_log", 0.0)
        corr = data.get("correlations", {})
        cpu_time = data.get("total_worker_cpu_seconds", 0.0)

        rows.append({
            "tag": tag,
            "system": sys_name,
            "m": m,
            "cells": total_cells,
            "N_T": types.get("T", 0),
            "N_UC": types.get("UC", 0),
            "N_UU": types.get("UU", 0),
            "P": p_log,
            "rho_raw": corr.get("spearman_raw_vs_ftle", np.nan),
            "rho_log": corr.get("spearman_log_raw_vs_ftle", np.nan),
            "rho_seg_lin": corr.get("spearman_linear_seg_vs_ftle", np.nan),
            "rho_seg_log": corr.get("spearman_log_seg_vs_ftle", np.nan),
            "rho_kstar": corr.get("spearman_kstar_vs_ftle", np.nan),
            "cpu_time": cpu_time,
            "dir": rdir,
        })

    if skipped:
        print("")
        print("!" * 78)
        print("EXCLUDED FROM SYNTHESIS -- %d incomplete run(s):" % len(skipped))
        for tag, filled, expected, missing in skipped:
            pct = 100.0 * filled / expected if expected else 0.0
            print("  %-40s %s/%s cells (%.1f%%), %d shard(s) missing"
                  % (tag, f"{filled:,}", f"{expected:,}", pct, len(missing)))
        print("Resubmit the missing array indices and re-aggregate before publishing.")
        print("!" * 78)

    if not rows:
        raise RuntimeError(
            "No complete runs available to synthesise (%d incomplete run(s) excluded)."
            % len(skipped)
        )

    # 1. Output Markdown Table
    md_lines = [
        "# High-Dimensional FinSTOD Paper: Cross-System Synthesis",
        "",
        "| Run Tag | System | $m$ | $2m$ Neighbours | Grid Cells | $N_T$ | $N_{UC}$ | $N_{UU}$ | $P$ | $\\rho(\\text{FinSTOD}_{\\log}, \\text{FTLE})$ | $\\rho(\\text{FinSTOD}_{\\text{seg}}, \\text{FTLE})$ | Total CPU Core Compute Time |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        rho_str = f"{r['rho_seg_log']:+.3f}" if not np.isnan(r['rho_seg_log']) else "N/A"
        rho_raw_str = f"{r['rho_log']:+.3f}" if not np.isnan(r['rho_log']) else "N/A"
        md_lines.append(
            f"| `{r['tag']}` | {r['system']} | {r['m']} | {2*r['m']} | {r['cells']:,} | {r['N_T']:,} | {r['N_UC']:,} | {r['N_UU']:,} | {r['P']:.2f} | {rho_raw_str} | {rho_str} | {format_duration(r['cpu_time'])} |"
        )
    md_content = "\n".join(md_lines)
    md_path = os.path.join(out_base, "synthesis_summary_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\nWrote Markdown summary table to: {md_path}")
    print("\n" + md_content + "\n")

    # 2. Output LaTeX Table
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Computational scaling, classification breakdown, and rank correlation against variational FTLE across high-dimensional dynamical systems ($m=4, 6, 8$).}",
        r"\label{tab:highdim_finstod_synthesis}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"\textbf{System} & $m$ & $2m$ & \textbf{Resolution} & $N_T$ & $N_{UC}$ & $P$ & $\rho(\text{FinSTOD}_{\text{log}}, \text{FTLE})$ & \textbf{Compute Time} \\",
        r"\midrule",
    ]
    for r in rows:
        sname = r['system'].replace("_", " ").title()
        rho_tex = f"{r['rho_seg_log']:+.3f}" if not np.isnan(r['rho_seg_log']) else r"\text{N/A}"
        tex_lines.append(
            f"{sname} & {r['m']} & {2*r['m']} & {r['cells']:,} & {r['N_T']:,} & {r['N_UC']:,} & {r['P']:.2f} & {rho_tex} & {format_duration(r['cpu_time'])} \\\\"
        )
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    tex_path = os.path.join(out_base, "synthesis_summary_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines))
    print(f"Wrote LaTeX summary table to: {tex_path}")

    # 3. Cross-System Synthesis Multi-Panel Figure
    # Filter 2D slices with figures
    slice_dirs = []
    for r in rows:
        npz_file = os.path.join(r["dir"], "results_consolidated.npz")
        meta_file = os.path.join(r["dir"], "meta.json")
        if os.path.isfile(npz_file) and os.path.isfile(meta_file):
            with open(meta_file, "r") as mf:
                mdata = json.load(mf)
            if len(mdata.get("grid_shape", [])) == 2:
                slice_dirs.append((r, npz_file, mdata))

    if slice_dirs:
        n_panels = len(slice_dirs)
        fig, axs = plt.subplots(3, n_panels, figsize=(4.2 * n_panels, 11.5), constrained_layout=True)
        if n_panels == 1:
            axs = axs[:, None]

        fig.suptitle("High-Dimensional FinSTOD ($m$-Dimensional Phase Space Slices)\n"
                     "Row 1: Log-Segmented FinSTOD | Row 2: Termination Level $k^*/L$ | Row 3: Variational FTLE",
                     fontsize=14, fontweight="bold")

        for col, (r, npz_file, mdata) in enumerate(slice_dirs):
            d = np.load(npz_file)
            nx, ny = mdata["grid_shape"]
            extent = mdata.get("extent", [0, 1, 0, 1])
            axes_info = mdata["grid"].get("axes", [0, 1])
            cnames = mdata.get("coord_names", [f"x_{i}" for i in range(r["m"])])
            xlab = f"${cnames[axes_info[0]]}$"
            ylab = f"${cnames[axes_info[1]]}$"

            mat_finstod = d["log_segmented"].reshape(nx, ny)
            mat_kstar = d["kstar_ratio"].reshape(nx, ny)
            mat_ftle = d["ftle"].reshape(nx, ny)

            # Row 0: FinSTOD Log-Segmented
            im0 = axs[0, col].imshow(mat_finstod.T, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=1, aspect="auto")
            axs[0, col].set_title(f"{r['system'].replace('_', ' ').title()} ($m={r['m']}$)\nFinSTOD Log-Seg ($P={r['P']:.2f}$)", fontsize=10, fontweight="bold")
            axs[0, col].set_xlabel(xlab, fontsize=9)
            axs[0, col].set_ylabel(ylab, fontsize=9)
            fig.colorbar(im0, ax=axs[0, col], shrink=0.85)

            # Row 1: k*/L
            im1 = axs[1, col].imshow(mat_kstar.T, origin="lower", extent=extent, cmap="coolwarm", vmin=0, vmax=1, aspect="auto")
            axs[1, col].set_title(r"Termination Level $k^*/L$", fontsize=10)
            axs[1, col].set_xlabel(xlab, fontsize=9)
            axs[1, col].set_ylabel(ylab, fontsize=9)
            fig.colorbar(im1, ax=axs[1, col], shrink=0.85)

            # Row 2: FTLE
            if np.any(np.isfinite(mat_ftle)):
                clim = None
                p2, p98 = np.percentile(mat_ftle[np.isfinite(mat_ftle)], [2, 98])
                if p98 > p2:
                    clim = (float(p2), float(p98))
                im2 = axs[2, col].imshow(mat_ftle.T, origin="lower", extent=extent, cmap="plasma", aspect="auto")
                if clim:
                    im2.set_clim(*clim)
                ftle_title = f"Variational FTLE\n$\\rho={r['rho_seg_log']:+.3f}$" if not np.isnan(r['rho_seg_log']) else "Variational FTLE"
            else:
                im2 = axs[2, col].imshow(np.zeros_like(mat_finstod).T, origin="lower", extent=extent, cmap="plasma", vmin=0, vmax=1, aspect="auto")
                ftle_title = f"Variational FTLE\n(N/A: Prohibitive $m={r['m']}$)"

            axs[2, col].set_title(ftle_title, fontsize=10)
            axs[2, col].set_xlabel(xlab, fontsize=9)
            axs[2, col].set_ylabel(ylab, fontsize=9)
            fig.colorbar(im2, ax=axs[2, col], shrink=0.85)

        synth_fig_png = os.path.join(out_base, "figure_synthesis_cross_system.png")
        synth_fig_pdf = os.path.join(out_base, "figure_synthesis_cross_system.pdf")
        fig.savefig(synth_fig_png, dpi=300)
        fig.savefig(synth_fig_pdf)
        plt.close(fig)
        print(f"Generated cross-system synthesis publication figure: {synth_fig_png}")


if __name__ == "__main__":
    main()
