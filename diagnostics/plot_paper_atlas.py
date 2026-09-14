"""Paper atlas + repository 3×1 galleries.

Each field is stretched to its own 2–98 percentile and mapped to [0, 1].
Chrome follows the original-paper Latest Figure Generator: panel titles at
fontsize 18, axis labels only on the bottom-left panel, tick labels only
there. The colourbar is drawn only when requested (Froeschlé indicators, the first
paper figure), attached to that figure's bottom-right panel.

Lorenz-96 m=32 is FinSTOD only (FTLE was never computed). HH energy v1 and
Duffing noise are excluded.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "cesvima_output"
OUT = "figures/atlas"


def _paper_root():
    for name in ("FinSTOD_high_dim_paper", "FinSTOD_high_dim", "paper_draft"):
        if os.path.isdir(name):
            return name
    return "FinSTOD_high_dim_paper"


_PAPER_ROOT = _paper_root()
PAPER_FIG = os.path.join(_PAPER_ROOT, "figures")
CMAP = "inferno"
DPI = 300
TITLE_FS = 18
LABEL_FS = 18
TICK_FS = 16
CBAR_FS = 14
SUPTITLE_FS = 16
# Side length of one square panel, in inches. Layouts scale from the data
# aspect ratio so fields are never stretched.
PANEL = 5.5
MAX_FIG = (12.4, 22.0)


def to01(a, lo_pct=2, hi_pct=98):
    """Independent 2–98 stretch, clipped to [0, 1]."""
    out = np.full(np.shape(a), np.nan, dtype=np.float64)
    m = np.isfinite(a)
    if m.sum() < 8:
        return out
    lo, hi = np.percentile(a[m], [lo_pct, hi_pct])
    if not np.isfinite(lo) or hi <= lo:
        out[m] = 0.0
        return out
    out[m] = np.clip((a[m] - lo) / (hi - lo), 0.0, 1.0)
    return out


def load(tag):
    meta = json.load(open(os.path.join(ROOT, tag, "meta.json")))
    z = np.load(os.path.join(ROOT, tag, "results_consolidated.npz"))
    nx, ny = meta["grid_shape"]
    raw = np.asarray(z["raw_scores"], dtype=float).reshape(nx, ny)
    fin = np.log1p(np.clip(raw, 0, None))
    fin[~np.isfinite(raw)] = np.nan
    ftle = np.asarray(z["ftle"], dtype=float).reshape(nx, ny)
    kstar = np.asarray(z["kstar_ratio"], dtype=float).reshape(nx, ny)
    log_seg = np.asarray(z["log_segmented"], dtype=float).reshape(nx, ny)
    axes = meta["grid"].get("axes", [0, 1])
    names = meta.get("coord_names", ["x", "y"])
    finite_ftle = ftle[np.isfinite(ftle)]
    ftle_ok = bool(meta.get("compute_ftle", True)) and finite_ftle.size > 0
    ftle_var = bool(ftle_ok and np.ptp(finite_ftle) > 1e-15)
    return {
        "raw": raw,
        "fin": fin,
        "ftle": ftle,
        "kstar": kstar,
        "log_seg": log_seg,
        "extent": meta.get("extent", [0, 1, 0, 1]),
        "xlab": "$%s$" % names[axes[0]],
        "ylab": "$%s$" % names[axes[1]],
        "m": int(meta["m"]),
        "L": int(meta["n_levels"]),
        "dt": float(meta.get("dt_out", meta.get("system", {}).get("dt_out", np.nan))),
        "tau": float(meta["total_time"]),
        "ftle_ok": ftle_ok,
        "ftle_var": ftle_var,
        "tag": tag,
    }


def fmt_tau(t):
    t = float(t)
    if abs(t - round(t)) < 1e-9:
        return "%d" % int(round(t))
    return "%g" % t


def with_tau(prefix, d):
    if r"\tau" in prefix:
        return prefix
    return r"%s, $\tau=%s$" % (prefix, fmt_tau(d["tau"]))


def row_tau(d, lab):
    """Row label that already carries L or K, plus τ."""
    if r"\tau" in lab:
        return lab
    return r"%s  ($\tau=%s$)" % (lab, fmt_tau(d["tau"]))


def _panel(ax, field, extent, title, xlab, ylab, *, is_bl=False, empty=None):
    ax.set_title(title, fontsize=TITLE_FS)
    ax.tick_params(
        axis="both", labelsize=TICK_FS, direction="out",
        labelbottom=is_bl, labelleft=is_bl,
    )
    if is_bl:
        ax.set_xlabel(xlab, fontsize=LABEL_FS)
        ax.set_ylabel(ylab, fontsize=LABEL_FS)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
    if empty == "omit":
        ax.set_axis_off()
        ax.set_title("")
        return None
    if empty:
        ax.set_facecolor("#f2f2f2")
        ax.text(0.5, 0.5, empty, ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="0.3", wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])
        return None
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad("#e6e6e6")
    im = ax.imshow(
        np.ma.masked_invalid(to01(field)).T,
        origin="lower", extent=extent, cmap=cmap, aspect="equal",
        interpolation="nearest", rasterized=True, vmin=0.0, vmax=1.0,
    )
    ax.set_aspect("equal", adjustable="box")
    return im


def _add_cbar(fig, im, axs):
    """Reference colourbar on the bottom-right panel of the first paper figure."""
    ax_br = np.atleast_2d(axs)[-1, -1]
    cb = fig.colorbar(im, ax=ax_br, fraction=0.046, pad=0.04)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=CBAR_FS)


def _paint_grid(fig, axs, cells, xlab, ylab, extent, *, colorbar=False):
    """cells: list of (field, title, empty_or_None), row-major matching axs.

    Axis titles and tick labels only on the bottom-left panel. Colourbar
    only when requested, attached to the bottom-right panel.
    """
    axs = np.atleast_2d(axs)
    nrows, ncols = axs.shape
    assert len(cells) == nrows * ncols
    last_im = None
    for k, (field, title, empty) in enumerate(cells):
        r, c = divmod(k, ncols)
        ax = axs[r, c]
        im = _panel(
            ax, field, extent, title, xlab, ylab,
            is_bl=(r == nrows - 1 and c == 0),
            empty=empty,
        )
        if im is not None:
            last_im = im
    if colorbar and last_im is not None:
        _add_cbar(fig, last_im, axs)


def fig_size(nrows, ncols, extent, extra_w=0.0):
    """Page-sized canvas of square panel slots, except for wide domains.

    A shear flow (Bickley, aspect ~3) keeps its physical aspect and is
    stacked rather than letterboxed into postage stamps. Square domains
    (Froeschlé, ABC, FPUT) use square slots so a 2×2 fills the page.
    """
    x0, x1, y0, y1 = extent
    ar = abs(x1 - x0) / max(abs(y1 - y0), 1e-12)
    if ar > 1.8:
        w = ncols * PANEL * ar + extra_w
        h = nrows * PANEL
    else:
        w = ncols * PANEL + extra_w
        h = nrows * PANEL
    scale = min(1.0, MAX_FIG[0] / w, MAX_FIG[1] / h)
    return (w * scale, h * scale)


def save_both(fig, fname, outdir=None):
    """Write PNG and PDF. Rasters inside the PDF use DPI so the 1000×1000
    field is not downsampled below one pixel per cell on a dual panel."""
    dest = OUT if outdir is None else outdir
    os.makedirs(dest, exist_ok=True)
    base = os.path.join(dest, os.path.splitext(fname)[0])
    fig.savefig(base + ".png", dpi=DPI)
    fig.savefig(base + ".pdf", dpi=DPI)
    print("  wrote", os.path.basename(base) + ".{png,pdf}")


def ftle_title(d):
    return "FTLE" + ("" if d["ftle_var"] else r"  (constant)")


def save_dual(d, title, fname, ftle_note=None, colorbar=False):
    """Stacked FinSTOD over FTLE, physical aspect preserved."""
    extra = 0.8 if colorbar else 0.0
    fig, axs = plt.subplots(
        2, 1, figsize=fig_size(2, 1, d["extent"], extra_w=extra),
        constrained_layout=True,
    )
    fig.suptitle(with_tau(title, d), fontsize=SUPTITLE_FS)
    ftle_cell = (d["ftle"], ftle_title(d), None) if d["ftle_ok"] else (
        None, "FTLE", ftle_note or "FTLE not computed")
    _paint_grid(
        fig, axs,
        [(d["fin"], r"FinSTOD  $\log(1+s)$", None), ftle_cell],
        d["xlab"], d["ylab"], d["extent"],
        colorbar=colorbar,
    )
    save_both(fig, fname)
    plt.close(fig)


def save_solo(d, title, fname, field=None, panel_title=None, colorbar=False):
    """Single panel — used for Lorenz-96 m=32 (no FTLE)."""
    extra = 0.8 if colorbar else 0.0
    fig, ax = plt.subplots(
        1, 1, figsize=fig_size(1, 1, d["extent"], extra_w=extra),
        constrained_layout=True,
    )
    fig.suptitle(with_tau(title, d), fontsize=SUPTITLE_FS)
    field = d["fin"] if field is None else field
    lab = panel_title or r"FinSTOD  $\log(1+s)$"
    _paint_grid(
        fig, np.array([[ax]]),
        [(field, lab, None)],
        d["xlab"], d["ylab"], d["extent"],
        colorbar=colorbar,
    )
    save_both(fig, fname)
    plt.close(fig)


def save_quad(d, title, fname, panels, outdir=None, colorbar=False):
    """Paper layout: 2×2 for four panels, otherwise a vertical stack.
    Three-panel figures (ABC) are stacked; they are not padded with a blank cell."""
    cells = []
    for p in panels:
        if len(p) == 2:
            cells.append((p[0], p[1], None))
        else:
            cells.append(p)
    n = len(cells)
    if n == 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = n, 1
    extra = 0.8 if colorbar else 0.0
    fig, axs = plt.subplots(
        nrows, ncols, figsize=fig_size(nrows, ncols, d["extent"], extra_w=extra),
        constrained_layout=True, squeeze=False,
    )
    fig.suptitle(with_tau(title, d), fontsize=SUPTITLE_FS)
    _paint_grid(
        fig, axs, cells, d["xlab"], d["ylab"], d["extent"],
        colorbar=colorbar,
    )
    save_both(fig, fname, outdir=outdir)
    plt.close(fig)


def save_row(d, title, fname, panels, outdir=None, figsize=None,
             colorbar=False):
    """1×n row. Default canvas is the paper page (wide triples stay readable)."""
    n = len(panels)
    extra = 0.8 if colorbar else 0.0
    if figsize is None:
        figsize = fig_size(1, n, d["extent"], extra_w=extra)
    fig, axs = plt.subplots(1, n, figsize=figsize, constrained_layout=True)
    fig.suptitle(with_tau(title, d), fontsize=SUPTITLE_FS)
    cells = [(field, lab, None) for field, lab in panels]
    _paint_grid(fig, axs, cells, d["xlab"], d["ylab"], d["extent"],
                colorbar=colorbar)
    save_both(fig, fname, outdir=outdir)
    plt.close(fig)


def save_triples(d, title):
    """Repository 3×1 galleries. Skipped when FTLE was never computed."""
    if not d["ftle_ok"]:
        return
    ft = ftle_title(d)
    save_row(
        d, title, "triple_kstar_%s.png" % d["tag"],
        [
            (d["kstar"], r"$k^{*}/L$"),
            (1.0 - d["kstar"], r"$1-k^{*}/L$"),
            (d["ftle"], ft),
        ],
    )
    save_row(
        d, title, "triple_rawlog_%s.png" % d["tag"],
        [
            (d["raw"], r"FinSTOD  $s$"),
            (d["fin"], r"FinSTOD  $\log(1+s)$"),
            (d["ftle"], ft),
        ],
    )


def save_montage(rows, fname, suptitle, colorbar=False):
    """rows: list of (d, row_title). Each row is FinSTOD | FTLE."""
    n = len(rows)
    extra = 0.8 if colorbar else 0.0
    fig, axs = plt.subplots(
        n, 2, figsize=fig_size(n, 2, rows[0][0]["extent"], extra_w=extra),
        constrained_layout=True,
    )
    fig.suptitle(suptitle, fontsize=SUPTITLE_FS)
    cells = []
    xlab = rows[0][0]["xlab"]
    ylab = rows[0][0]["ylab"]
    extent = rows[0][0]["extent"]
    for d, row_title in rows:
        lab = row_tau(d, row_title)
        cells.append((d["fin"], r"%s  —  FinSTOD $\log(1+s)$" % lab, None))
        if not d["ftle_ok"]:
            cells.append((None, "%s  —  FTLE" % lab, "FTLE not computed"))
        else:
            cells.append((d["ftle"], "%s  —  %s" % (lab, ftle_title(d)), None))
    _paint_grid(fig, axs, cells, xlab, ylab, extent, colorbar=colorbar)
    save_both(fig, fname)
    plt.close(fig)


def save_fput_interior_panel(get, fname="panel_fput_m8_tau_sbar.png"):
    """4×3: τ=5,10,15,20 rows; FinSTOD | FTLE | modal S-bar."""
    rows = [
        (attach_sbar(get("prod_fput_m8_tau05")), r"$\tau=5$"),
        (attach_sbar(get("prod_fput_m8_tau10")), r"$\tau=10$"),
        (attach_sbar(get("prod_fput_m8_tau15")), r"$\tau=15$"),
        (attach_sbar(get("prod_fput_m8_tau20")), r"$\tau=20$"),
    ]
    n = len(rows)
    fig, axs = plt.subplots(
        n, 3, figsize=fig_size(n, 3, rows[0][0]["extent"]),
        constrained_layout=True,
    )
    fig.suptitle(
        r"FPUT $m=8$ — FinSTOD, FTLE, and modal $\bar S$",
        fontsize=SUPTITLE_FS,
    )
    cells = []
    for d, lab in rows:
        cells.append((d["fin"], r"%s  —  FinSTOD $\log(1+s)$" % lab, None))
        cells.append((d["ftle"], r"%s  —  FTLE" % lab, None))
        cells.append((d["sbar"], r"%s  —  modal $\bar S$" % lab, None))
    d0 = rows[0][0]
    _paint_grid(fig, axs, cells, d0["xlab"], d0["ylab"], d0["extent"])
    save_both(fig, fname)
    plt.close(fig)


def copy_selected_pdf(fname):
    """Copy a PDF already written to the repo atlas into the Overleaf figures folder."""
    os.makedirs(PAPER_FIG, exist_ok=True)
    base = os.path.splitext(os.path.basename(fname))[0]
    src = os.path.join(OUT, base + ".pdf")
    dst = os.path.join(PAPER_FIG, base + ".pdf")
    shutil.copy2(src, dst)
    print("  paper", base + ".pdf")


# Paper-atlas selection. Colourbar on the first remaining atlas figure (Froeschlé).
SELECTED_PDFS = [
    "dual_prod_abc_flow_m3",
    "extra_abc_log_ftle",
    "triple_rawlog_prod_abc_flow_m3",
    "triple_kstar_prod_abc_flow_m3",
    "dual_prod_cr3bp_m6",
    "pair_fput",
    "panel_fput_m8_tau_sbar",
    "pair_kuramoto",
    "pair_froeschle_L0250_L0500",
    "pair_froeschle_L1000_L2000",
    "pair_froeschle_L2500_L4000",
    "dual_prod_froeschle_L8000",
    "dual_prod_froeschle_K005",
    "pair_froeschle_K010_K020",
    "pair_froeschle_K030_K050",
    "dual_prod_froeschle_K100",
    "dual_prod_froeschle_dof3_m6",
    "sweep_hh_energy",
    "sweep_lorenz_m",
    "sweep_lorenz_m4_L",
    "sweep_lorenz_m6_L",
    "solo_prod_lorenz96_m32",
    "sweep_torus_mixed",
]


def export_selected(get):
    """Build every Overleaf PDF (new chrome) and copy into PAPER_FIG.

    L-sweep (K=0.30): 250+500, 1000+2000, 2500+4000, 8000 alone.
    K-sweep (L=2500): 0.05, 0.10+0.20, 0.30+0.50, 1.00; then m=6 alone.
    Lorenz-96 m-ladder omits m=32 (shown as a FinSTOD-only solo).
    ABC: production dual plus log-transform and $k^{*}$ extras.
    """
    abc = get("prod_abc_flow_m3")
    save_dual(abc, r"ABC flow, $m=3$",
              "dual_prod_abc_flow_m3.png")
    save_quad(
        abc, r"ABC flow, $m=3$",
        "triple_rawlog_prod_abc_flow_m3.png",
        [
            (abc["raw"], r"FinSTOD  $s$"),
            (abc["fin"], r"FinSTOD  $\log(1+s)$"),
            (abc["ftle"], ftle_title(abc)),
        ],
    )
    save_quad(
        abc, r"ABC flow, $m=3$",
        "triple_kstar_prod_abc_flow_m3.png",
        [
            (abc["kstar"], r"$k^{*}/L$"),
            (1.0 - abc["kstar"], r"$1-k^{*}/L$"),
            (abc["ftle"], ftle_title(abc)),
        ],
    )
    ftle = abc["ftle"]
    shift = np.nanmin(ftle)
    log_ftle = np.log1p(ftle - shift)
    save_quad(
        abc, r"ABC flow — linear FTLE vs $\log(1+x)$ FTLE",
        "extra_abc_log_ftle.png",
        [
            (abc["fin"], r"FinSTOD  $\log(1+s)$"),
            (ftle, "FTLE  linear"),
            (log_ftle, r"FTLE  $\log(1+x)$"),
        ],
    )

    save_dual(get("prod_cr3bp_m6"), r"CR3BP, $m=6$",
              "dual_prod_cr3bp_m6.png")
    save_montage(
        [(get("prod_fput_m8"), r"$m=8$"), (get("prod_fput_m16"), r"$m=16$")],
        "pair_fput.png", r"FPUT lattice, $\tau=25$",
    )
    save_fput_interior_panel(get)
    save_montage(
        [(get("prod_kuramoto_m8"), r"$m=8$"),
         (get("prod_kuramoto_m16"), r"$m=16$")],
        "pair_kuramoto.png", r"Kuramoto ring ($K=0.7$), $\tau=15$",
    )

    save_montage(
        [(get("prod_froeschle_L0250"), r"$L=250$"),
         (get("prod_froeschle_L0500"), r"$L=500$")],
        "pair_froeschle_L0250_L0500.png",
        r"Froeschlé $m=4$, $K=0.30$ — $L=250$, $500$",
    )
    save_montage(
        [(get("prod_froeschle_L1000"), r"$L=1000$"),
         (get("prod_froeschle_L2000"), r"$L=2000$")],
        "pair_froeschle_L1000_L2000.png",
        r"Froeschlé $m=4$, $K=0.30$ — $L=1000$, $2000$",
    )
    save_montage(
        [(get("prod_froeschle_dof2_m4"), r"$L=2500$"),
         (get("prod_froeschle_L4000"), r"$L=4000$")],
        "pair_froeschle_L2500_L4000.png",
        r"Froeschlé $m=4$, $K=0.30$ — $L=2500$, $4000$",
    )
    save_dual(get("prod_froeschle_L8000"),
              r"Froeschlé $m=4$, $K=0.30$, $L=8000$",
              "dual_prod_froeschle_L8000.png")

    save_dual(get("prod_froeschle_K005"),
              r"Froeschlé $m=4$, $K=0.05$, $L=2500$",
              "dual_prod_froeschle_K005.png")
    save_montage(
        [(get("prod_froeschle_K010"), r"$K=0.10$"),
         (get("prod_froeschle_K020"), r"$K=0.20$")],
        "pair_froeschle_K010_K020.png",
        r"Froeschlé $m=4$, $L=2500$ — $K=0.10$, $0.20$",
    )
    save_montage(
        [(get("prod_froeschle_dof2_m4"), r"$K=0.30$"),
         (get("prod_froeschle_K050"), r"$K=0.50$")],
        "pair_froeschle_K030_K050.png",
        r"Froeschlé $m=4$, $L=2500$ — $K=0.30$, $0.50$",
    )
    save_dual(get("prod_froeschle_K100"),
              r"Froeschlé $m=4$, $K=1.00$, $L=2500$",
              "dual_prod_froeschle_K100.png")
    save_dual(get("prod_froeschle_dof3_m6"),
              r"Froeschlé $m=6$, $K=0.30$, $L=2500$",
              "dual_prod_froeschle_dof3_m6.png")

    hh_rows = [
        ("prod_hh_energy_006_v2", r"$E=0.06$"),
        ("prod_hh_energy_010_v2", r"$E=0.10$"),
        ("prod_hh_energy_014_v2", r"$E=0.14$"),
        ("prod_hh_energy_016_v2", r"$E=0.16$"),
    ]
    save_montage(
        [(get(t), lab) for t, lab in hh_rows],
        "sweep_hh_energy.png",
        r"Hénon–Heiles $m=6$ — energy ceiling $H\leq E$, $\tau=5$",
    )

    lz_m = [
        ("prod_lorenz96_m4_1000x1000", r"$m=4$"),
        ("prod_lorenz96_m6_1000x1000", r"$m=6$"),
        ("prod_lorenz96_m8_1000x1000", r"$m=8$"),
        ("prod_lorenz96_m16", r"$m=16$"),
    ]
    save_montage(
        [(get(t), lab) for t, lab in lz_m],
        "sweep_lorenz_m.png",
        r"Lorenz-96 — dimension $m$  ($m=32$ in a separate panel)",
    )
    lz4 = [
        ("prod_lorenz96_m4_L0250", r"$m=4$, $L=250$"),
        ("prod_lorenz96_m4_L0500", r"$m=4$, $L=500$"),
        ("prod_lorenz96_m4_L1000", r"$m=4$, $L=1000$"),
        ("prod_lorenz96_m4_1000x1000", r"$m=4$, $L=2000$"),
    ]
    lz6 = [
        ("prod_lorenz96_m6_L0250", r"$m=6$, $L=250$"),
        ("prod_lorenz96_m6_L0500", r"$m=6$, $L=500$"),
        ("prod_lorenz96_m6_L1000", r"$m=6$, $L=1000$"),
        ("prod_lorenz96_m6_1000x1000", r"$m=6$, $L=2000$"),
    ]
    save_montage([(get(t), lab) for t, lab in lz4],
                 "sweep_lorenz_m4_L.png",
                 r"Lorenz-96 $m=4$ — window $L$")
    save_montage([(get(t), lab) for t, lab in lz6],
                 "sweep_lorenz_m6_L.png",
                 r"Lorenz-96 $m=6$ — window $L$")
    save_solo(get("prod_lorenz96_m32"),
              r"Lorenz-96, $m=32$, $L=2500$",
              "solo_prod_lorenz96_m32.png")

    torus = [
        ("prod_torus_mixed_d1_kstar", r"$d=1$"),
        ("prod_torus_mixed_d2_kstar", r"$d=2$"),
        ("prod_torus_mixed_d3_kstar", r"$d=3$"),
        ("prod_torus_mixed_d4_kstar", r"$d=4$"),
        ("prod_torus_mixed_d5_kstar", r"$d=5$"),
    ]
    save_montage(
        [(get(t), lab) for t, lab in torus],
        "sweep_torus_mixed.png",
        r"Integrable torus $m=6$, mixed basis — true dimension $d$, $\tau=4$",
    )

    missing = []
    for base in SELECTED_PDFS:
        src = os.path.join(OUT, base + ".pdf")
        if not os.path.isfile(src):
            missing.append(base)
            continue
        copy_selected_pdf(base + ".pdf")
    if missing:
        raise SystemExit("missing PDFs for selection: " + ", ".join(missing))

    stale = {
        "dual_prod_froeschle_L1000.pdf",
        "pair_froeschle_L2000_L2500.pdf",
        "pair_froeschle_L4000_L8000.pdf",
        "dual_prod_lorenz96_m32.pdf",
        "panel_fput_m8_tau05_tau10.pdf",
        "dual_prod_duffing_m6_1000x1000.pdf",
        "dual_test2_bickley_jet_shear.pdf",
    }
    for name in stale:
        for folder in (PAPER_FIG,):
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                os.remove(path)
                print("  removed stale", name)


def main():
    os.makedirs(OUT, exist_ok=True)
    cache = {}

    def get(tag):
        if tag not in cache:
            print("loading", tag)
            cache[tag] = load(tag)
        return cache[tag]

    singles = [
        ("prod_abc_flow_m3", r"ABC flow, $m=3$"),
        ("prod_cr3bp_m6", r"CR3BP, $m=6$"),
        ("prod_duffing_m6_1000x1000", r"Coupled Duffing, $m=6$"),
        ("prod_fput_m8", r"FPUT, $m=8$"),
        ("prod_fput_m16", r"FPUT, $m=16$"),
        ("prod_froeschle_dof2_m4", r"Froeschlé $m=4$, $K=0.30$, $L=2500$"),
        ("prod_froeschle_dof3_m6", r"Froeschlé $m=6$, $K=0.30$, $L=2500$"),
        ("prod_kuramoto_m8", r"Kuramoto ring, $m=8$"),
        ("prod_kuramoto_m16", r"Kuramoto ring, $m=16$"),
        ("test2_bickley_jet_shear", r"Bickley jet (perturbed)"),
        ("test2_bickley_pure_shear", r"Bickley jet (pure shear)"),
        ("prod_lorenz96_m4_1000x1000", r"Lorenz-96, $m=4$, $L=2000$"),
        ("prod_lorenz96_m6_1000x1000", r"Lorenz-96, $m=6$, $L=2000$"),
        ("prod_lorenz96_m8_1000x1000", r"Lorenz-96, $m=8$, $L=2000$"),
        ("prod_lorenz96_m16", r"Lorenz-96, $m=16$, $L=2500$"),
    ]
    def emit(d, title, fname, ftle_note=None):
        save_dual(d, title, fname, ftle_note=ftle_note)
        save_triples(d, title)

    for tag, title in singles:
        emit(get(tag), title, "dual_%s.png" % tag)

    emit(
        get("prod_lorenz96_m32"),
        r"Lorenz-96, $m=32$, $L=2500$",
        "dual_prod_lorenz96_m32.png",
        ftle_note="FTLE not computed\n(variational cost prohibitive)",
    )
    save_solo(
        get("prod_lorenz96_m32"),
        r"Lorenz-96, $m=32$, $L=2500$",
        "solo_prod_lorenz96_m32.png",
    )

    # Froeschlé K-sweep (m=4, L=2500)
    k_rows = [
        ("prod_froeschle_K005", r"$K=0.05$"),
        ("prod_froeschle_K010", r"$K=0.10$"),
        ("prod_froeschle_K020", r"$K=0.20$"),
        ("prod_froeschle_dof2_m4", r"$K=0.30$"),
        ("prod_froeschle_K050", r"$K=0.50$"),
        ("prod_froeschle_K100", r"$K=1.00$"),
    ]
    save_montage(
        [(get(t), lab) for t, lab in k_rows],
        "sweep_froeschle_K.png",
        r"Froeschlé $m=4$, $L=2500$ — coupling $K$",
    )
    for tag, lab in k_rows:
        if tag != "prod_froeschle_dof2_m4":
            emit(get(tag), r"Froeschlé $m=4$, %s, $L=2500$" % lab,
                 "dual_%s.png" % tag)

    # Froeschlé L-sweep (m=4, K=0.30)
    l_rows = [
        ("prod_froeschle_L0250", r"$L=250$"),
        ("prod_froeschle_L0500", r"$L=500$"),
        ("prod_froeschle_L1000", r"$L=1000$"),
        ("prod_froeschle_L2000", r"$L=2000$"),
        ("prod_froeschle_dof2_m4", r"$L=2500$"),
        ("prod_froeschle_L4000", r"$L=4000$"),
        ("prod_froeschle_L8000", r"$L=8000$"),
    ]
    save_montage(
        [(get(t), lab) for t, lab in l_rows],
        "sweep_froeschle_L.png",
        r"Froeschlé $m=4$, $K=0.30$ — window $L$",
    )
    for tag, lab in l_rows:
        if tag != "prod_froeschle_dof2_m4":
            emit(get(tag), r"Froeschlé $m=4$, $K=0.30$, %s" % lab,
                 "dual_%s.png" % tag)

    # HH energy v2
    hh_rows = [
        ("prod_hh_energy_006_v2", r"$E=0.06$"),
        ("prod_hh_energy_010_v2", r"$E=0.10$"),
        ("prod_hh_energy_014_v2", r"$E=0.14$"),
        ("prod_hh_energy_016_v2", r"$E=0.16$"),
    ]
    save_montage(
        [(get(t), lab) for t, lab in hh_rows],
        "sweep_hh_energy.png",
        r"Hénon–Heiles $m=6$ — energy ceiling $H\leq E$",
    )
    for tag, lab in hh_rows:
        emit(get(tag), r"Hénon–Heiles $m=6$, %s" % lab,
             "dual_%s.png" % tag)

    # Lorenz m-ladder (m=32 is its own dual; not in this sweep)
    lz_m = [
        ("prod_lorenz96_m4_1000x1000", r"$m=4$"),
        ("prod_lorenz96_m6_1000x1000", r"$m=6$"),
        ("prod_lorenz96_m8_1000x1000", r"$m=8$"),
        ("prod_lorenz96_m16", r"$m=16$"),
    ]
    save_montage(
        [(get(t), lab) for t, lab in lz_m],
        "sweep_lorenz_m.png",
        r"Lorenz-96 — dimension $m$",
    )

    # Lorenz L census
    lz4 = [
        ("prod_lorenz96_m4_L0250", r"$m=4$, $L=250$"),
        ("prod_lorenz96_m4_L0500", r"$m=4$, $L=500$"),
        ("prod_lorenz96_m4_L1000", r"$m=4$, $L=1000$"),
        ("prod_lorenz96_m4_1000x1000", r"$m=4$, $L=2000$"),
    ]
    lz6 = [
        ("prod_lorenz96_m6_L0250", r"$m=6$, $L=250$"),
        ("prod_lorenz96_m6_L0500", r"$m=6$, $L=500$"),
        ("prod_lorenz96_m6_L1000", r"$m=6$, $L=1000$"),
        ("prod_lorenz96_m6_1000x1000", r"$m=6$, $L=2000$"),
    ]
    save_montage([(get(t), lab) for t, lab in lz4],
                 "sweep_lorenz_m4_L.png",
                 r"Lorenz-96 $m=4$ — window $L$")
    save_montage([(get(t), lab) for t, lab in lz6],
                 "sweep_lorenz_m6_L.png",
                 r"Lorenz-96 $m=6$ — window $L$")
    for tag, lab in lz4 + lz6:
        if "1000x1000" not in tag:
            emit(get(tag), r"Lorenz-96 %s" % lab, "dual_%s.png" % tag)

    # Torus mixed
    torus = [
        ("prod_torus_mixed_d1_kstar", r"$d=1$"),
        ("prod_torus_mixed_d2_kstar", r"$d=2$"),
        ("prod_torus_mixed_d3_kstar", r"$d=3$"),
        ("prod_torus_mixed_d4_kstar", r"$d=4$"),
        ("prod_torus_mixed_d5_kstar", r"$d=5$"),
    ]
    save_montage(
        [(get(t), lab) for t, lab in torus],
        "sweep_torus_mixed.png",
        r"Integrable torus $m=6$, mixed basis — true dimension $d$",
    )
    for tag, lab in torus:
        emit(get(tag), r"Integrable torus mixed basis, %s" % lab,
             "dual_%s.png" % tag)

    # ABC extras: hierarchy vs raw vs k*/L
    abc = get("prod_abc_flow_m3")
    save_row(
        abc, r"ABC flow — raw, segmented hierarchy, and $k^{*}/L$",
        "extra_abc_hierarchy.png",
        [
            (abc["fin"], r"raw  $\log(1+s)$"),
            (abc["log_seg"], r"log-segmented  ($P=1$)"),
            (abc["kstar"], r"$k^{*}/L$"),
        ],
    )
    ftle = abc["ftle"]
    shift = np.nanmin(ftle)
    log_ftle = np.log1p(ftle - shift)
    save_row(
        abc, r"ABC flow — FTLE under the same $\log(1+x)$ stretch as FinSTOD",
        "extra_abc_log_ftle.png",
        [
            (abc["fin"], r"FinSTOD  $\log(1+s)$"),
            (ftle, "FTLE  linear"),
            (log_ftle, r"FTLE  $\log(1+x)$"),
        ],
    )

    # FPUT / Kuramoto / Bickley pair montages for the paper body
    save_montage(
        [(get("prod_fput_m8"), r"$m=8$"), (get("prod_fput_m16"), r"$m=16$")],
        "pair_fput.png", r"FPUT lattice",
    )
    save_montage(
        [(get("prod_kuramoto_m8"), r"$m=8$"),
         (get("prod_kuramoto_m16"), r"$m=16$")],
        "pair_kuramoto.png", r"Kuramoto ring ($K=0.7$)",
    )
    save_montage(
        [(get("test2_bickley_jet_shear"), "perturbed"),
         (get("test2_bickley_pure_shear"), "pure shear")],
        "pair_bickley.png", r"Bickley jet",
    )

    print("Wrote", OUT)
    export_selected(get)


TAU_SERIES_JOBS = [
    ("prod_abc_flow_m3_tau05", r"ABC flow, $m=3$"),
    ("prod_abc_flow_m3_tau10", r"ABC flow, $m=3$"),
    ("prod_abc_flow_m3_tau15", r"ABC flow, $m=3$"),
    ("prod_abc_flow_m3_tau20", r"ABC flow, $m=3$"),
    ("prod_fput_m8_tau05", r"FPUT, $m=8$"),
    ("prod_fput_m8_tau10", r"FPUT, $m=8$"),
    ("prod_fput_m8_tau15", r"FPUT, $m=8$"),
    ("prod_fput_m8_tau20", r"FPUT, $m=8$"),
]

FPUT_SBAR_JOBS = [
    ("prod_fput_m8_tau05", r"FPUT, $m=8$"),
    ("prod_fput_m8_tau10", r"FPUT, $m=8$"),
    ("prod_fput_m8_tau15", r"FPUT, $m=8$"),
    ("prod_fput_m8_tau20", r"FPUT, $m=8$"),
    ("prod_fput_m8", r"FPUT, $m=8$"),
]
SBAR_STRIDE = 1


def _spearman(a, b):
    from scipy.stats import spearmanr
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 20:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).statistic)


def _sbar_workers():
    n = os.cpu_count() or 2
    return max(1, min(8, n - 1 if n > 2 else n))


def _sbar_chunk(payload):
    """Picklable worker: integrate one cell batch to mean_S / tau_eq / last_S."""
    sys_cfg, dt_out, n_cells, L, cells, sample_every, s_thresh = payload
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    if root not in sys.path:
        sys.path.insert(0, root)
    from stod_nd import FPUT
    system = FPUT(
        dof=int(sys_cfg.get("dof", 4)),
        alpha=float(sys_cfg.get("alpha", 0.25)),
        beta=float(sys_cfg.get("beta", 0.1)),
        dt_out=float(sys_cfg.get("dt_out", dt_out)),
    )
    system.reset()
    states = system.cell_to_state(cells, n_cells)
    k = cells.shape[0]
    tau = np.full(k, L, dtype=np.float64)
    found = np.zeros(k, dtype=bool)
    acc = np.zeros(k, dtype=np.float64)
    n_acc = 0
    last_S = np.zeros(k, dtype=np.float64)
    for t in range(L):
        states = system.step(states)
        if t % sample_every != 0:
            continue
        S = system.spectral_entropy(states)
        last_S = S
        hit = (~found) & (S >= s_thresh)
        tau[hit] = float(t + 1)
        found[hit] = True
        if t >= L // 2:
            acc += S
            n_acc += 1
    return acc / max(n_acc, 1), tau, last_S


def compute_fput_sbar(tag, stride=SBAR_STRIDE, sample_every=10, s_thresh=0.5):
    """Second-half mean modal spectral entropy on the production slice.

    Prefers the CESVIMA Phase 12 aux (``aux_sbar_<tag>``). Full-grid local
    integration is off unless ``FPUT_SBAR_LOCAL=1``.
    """
    meta = json.load(open(os.path.join(ROOT, tag, "meta.json")))
    L = int(meta["n_levels"])
    nx, ny = meta["grid_shape"]
    aux = os.path.join(ROOT, "aux_sbar_%s" % tag, "results_consolidated.npz")
    if os.path.isfile(aux):
        z = np.load(aux)
        mean_S = np.asarray(z["mean_S"], dtype=float).reshape(nx, ny)
        tau_eq = np.asarray(z["tau_eq"], dtype=float).reshape(nx, ny)
        last_S = (np.asarray(z["last_S"], dtype=float).reshape(nx, ny)
                  if "last_S" in z.files else np.full((nx, ny), np.nan))
        print("  loaded CESVIMA aux S-bar %s  %dx%d" % (tag, nx, ny))
        return {
            "mean_S": mean_S, "tau_eq": tau_eq, "last_S": last_S,
            "stride": 1, "L": L,
        }
    cache = os.path.join(ROOT, tag, "entropy_subsample_s%d.npz" % stride)
    if os.path.isfile(cache):
        z = np.load(cache)
        if int(z["L"]) == L and int(z["stride"]) == stride:
            return {
                "mean_S": np.asarray(z["mean_S"], dtype=float),
                "tau_eq": np.asarray(z["tau_eq"], dtype=float),
                "last_S": np.asarray(z["last_S"], dtype=float),
                "stride": stride,
                "L": L,
            }
    if stride <= 2 and os.environ.get("FPUT_SBAR_LOCAL") != "1":
        raise SystemExit(
            "full-grid FPUT S-bar is CESVIMA Phase 12 "
            "(sbatch --export=ALL,PHASES=\"12\" master_pipeline.slurm). "
            "Set FPUT_SBAR_LOCAL=1 to integrate on this machine."
        )
    from concurrent.futures import ProcessPoolExecutor, as_completed

    z = np.load(os.path.join(ROOT, tag, "results_consolidated.npz"))
    nx, ny = meta["grid_shape"]
    m = int(meta["m"])
    n_cells = int(meta["grid"]["n_cells"])
    cells = np.asarray(z["cells"]).reshape(nx, ny, m)[::stride, ::stride]
    sy, sx = cells.shape[:2]
    flat = np.ascontiguousarray(cells.reshape(-1, m))
    n_workers = _sbar_workers()
    print("  integrating %s  %dx%d stride=%d  L=%d  n=%d  workers=%d" % (
        tag, sy, sx, stride, L, flat.shape[0], n_workers))
    sys_cfg = meta["system"]
    dt_out = float(sys_cfg.get("dt_out", meta.get("dt_out", 0.01)))
    splits = np.array_split(flat, n_workers)
    parts = [None] * n_workers
    payloads = [
        (sys_cfg, dt_out, n_cells, L, ch, sample_every, s_thresh)
        for ch in splits
    ]
    if n_workers == 1:
        parts[0] = _sbar_chunk(payloads[0])
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_sbar_chunk, p): i for i, p in enumerate(payloads)}
            done = 0
            for fut in as_completed(futs):
                i = futs[fut]
                parts[i] = fut.result()
                done += 1
                print("    chunk %d/%d" % (done, n_workers), flush=True)
    mean_S = np.concatenate([p[0] for p in parts]).reshape(sy, sx)
    tau = np.concatenate([p[1] for p in parts]).reshape(sy, sx)
    last_S = np.concatenate([p[2] for p in parts]).reshape(sy, sx)
    np.savez_compressed(
        cache, mean_S=mean_S, tau_eq=tau, last_S=last_S,
        stride=np.int32(stride), L=np.int32(L),
    )
    print("  cached", cache)
    return {
        "mean_S": mean_S, "tau_eq": tau, "last_S": last_S,
        "stride": stride, "L": L,
    }


def attach_sbar(d, stride=SBAR_STRIDE):
    pack = compute_fput_sbar(d["tag"], stride=stride)
    s = pack["stride"]
    fin_s = d["fin"][::s, ::s]
    ftle_s = d["ftle"][::s, ::s]
    raw_s = d["raw"][::s, ::s]
    out = dict(d)
    out["sbar"] = pack["mean_S"]
    out["tau_eq"] = pack["tau_eq"]
    out["last_S"] = pack["last_S"]
    out["sbar_stride"] = s
    out["rho_fin_sbar"] = _spearman(fin_s, pack["mean_S"])
    out["rho_raw_sbar"] = _spearman(raw_s, pack["mean_S"])
    out["rho_ftle_sbar"] = _spearman(ftle_s, pack["mean_S"])
    out["rho_fin_tau"] = _spearman(fin_s, pack["tau_eq"])
    out["rho_ftle_tau"] = _spearman(ftle_s, pack["tau_eq"])
    out["frac_never_eq"] = float(np.mean(pack["tau_eq"] >= pack["L"]))
    return out


def save_sbar_dual(d, title, fname):
    fig, axs = plt.subplots(
        2, 1, figsize=fig_size(2, 1, d["extent"]),
        constrained_layout=True,
    )
    fig.suptitle(with_tau(title, d), fontsize=SUPTITLE_FS)
    _paint_grid(
        fig, axs,
        [(d["fin"], r"FinSTOD  $\log(1+s)$", None),
         (d["sbar"], r"modal $\bar S$", None)],
        d["xlab"], d["ylab"], d["extent"],
    )
    save_both(fig, fname)
    plt.close(fig)


def save_sbar_montage(rows, fname, suptitle):
    n = len(rows)
    fig, axs = plt.subplots(
        n, 2, figsize=fig_size(n, 2, rows[0][0]["extent"]),
        constrained_layout=True,
    )
    fig.suptitle(suptitle, fontsize=SUPTITLE_FS)
    cells = []
    xlab, ylab, extent = rows[0][0]["xlab"], rows[0][0]["ylab"], rows[0][0]["extent"]
    for d, row_title in rows:
        lab = row_tau(d, row_title)
        cells.append((d["fin"], r"%s  —  FinSTOD $\log(1+s)$" % lab, None))
        cells.append((d["sbar"], r"%s  —  modal $\bar S$" % lab, None))
    _paint_grid(fig, axs, cells, xlab, ylab, extent)
    save_both(fig, fname)
    plt.close(fig)


def export_tau_series(get):
    """Dual + both triples for the ABC / FPUT m=8 tau envelope (not tau=25)."""
    os.makedirs(OUT, exist_ok=True)
    for tag, title in TAU_SERIES_JOBS:
        d = get(tag)
        print("  %s  L=%s  tau=%s  ftle=%s" % (tag, d["L"], fmt_tau(d["tau"]), d["ftle_ok"]))
        save_dual(d, title, "dual_%s.png" % tag)
        save_triples(d, title)
    save_montage(
        [(get("prod_abc_flow_m3_tau05"), r"$\tau=5$"),
         (get("prod_abc_flow_m3_tau10"), r"$\tau=10$"),
         (get("prod_abc_flow_m3_tau15"), r"$\tau=15$"),
         (get("prod_abc_flow_m3_tau20"), r"$\tau=20$")],
        "sweep_abc_tau.png",
        r"ABC flow $m=3$ — $\tau$ envelope (production $\tau=25$ already in the atlas)",
    )
    save_montage(
        [(get("prod_fput_m8_tau05"), r"$\tau=5$"),
         (get("prod_fput_m8_tau10"), r"$\tau=10$"),
         (get("prod_fput_m8_tau15"), r"$\tau=15$"),
         (get("prod_fput_m8_tau20"), r"$\tau=20$")],
        "sweep_fput_m8_tau.png",
        r"FPUT $m=8$ — $\tau$ envelope (production $\tau=25$ already in the atlas)",
    )
    print("Wrote tau-series figures to", OUT)


def export_fput_sbar(get):
    """FinSTOD vs physical modal spectral entropy for the FPUT τ envelope."""
    os.makedirs(OUT, exist_ok=True)
    rows = []
    print("%-24s %5s  rho(log s, S)  rho(FTLE, S)  rho(s, tau_eq)  never-eq" % (
        "tag", "tau"))
    for tag, title in FPUT_SBAR_JOBS:
        d = attach_sbar(get(tag))
        print("  %s  L=%s  tau=%s  stride=%d" % (
            tag, d["L"], fmt_tau(d["tau"]), d["sbar_stride"]))
        print("%-24s %5s  %+13.3f  %+12.3f  %+13.3f  %7.3f" % (
            tag, fmt_tau(d["tau"]),
            d["rho_fin_sbar"], d["rho_ftle_sbar"],
            d["rho_fin_tau"], d["frac_never_eq"]))
        save_sbar_dual(d, title, "dual_sbar_%s.png" % tag)
        if tag != "prod_fput_m8":
            rows.append((d, r"$\tau=%s$" % fmt_tau(d["tau"])))
    save_sbar_montage(
        rows, "sweep_fput_m8_sbar.png",
        r"FPUT $m=8$ — FinSTOD vs modal spectral entropy $\bar S$",
    )
    save_fput_interior_panel(get)


if __name__ == "__main__":
    import sys
    os.makedirs(OUT, exist_ok=True)
    cache = {}

    def get(tag):
        if tag not in cache:
            print("loading", tag)
            cache[tag] = load(tag)
        return cache[tag]

    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "selected":
        export_selected(get)
    elif mode in ("tau-series", "tau_series"):
        export_tau_series(get)
    elif mode in ("fput-sbar", "fput_sbar", "tau-series-sbar"):
        export_fput_sbar(get)
    else:
        main()
