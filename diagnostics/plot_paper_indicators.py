"""Paper-style 2×2 indicator panels (FinSTOD, FTLE, LD p=1, FLI).

Chrome is identical to plot_paper_atlas.py: inferno, 2–98 stretch, titles at
18 pt, axis titles and tick labels only on the bottom-left panel. Colourbar
only on the first of these figures in the compiled paper (Froeschlé), on
the bottom-right panel.

Constant panels (pure-shear FTLE, FPUT FLI) are omitted rather than
plotted as a flat colour.

Writes PNG+PDF to figures/atlas/ and copies PDFs into the paper figures/ folder.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "diagnostics"))

import plot_paper_atlas as atlas  # noqa: E402


def _varies(field):
    """True if the field has structure, not just round-off on a constant."""
    finite = np.asarray(field, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return False
    scale = max(abs(float(np.median(finite))), 1.0)
    return float(np.ptp(finite)) > max(1e-12, 1e-8 * scale)


def _const_title(name, field):
    return name if _varies(field) else r"%s  (constant)" % name


def load_pair(src_tag, aux_tag):
    """FinSTOD from *src_tag*; time-aware FTLE / FLI / LD from *aux_tag*."""
    d = atlas.load(src_tag)
    z = np.load(os.path.join(atlas.ROOT, aux_tag, "results_consolidated.npz"))
    shape = d["fin"].shape

    def field(key):
        a = np.asarray(z[key], dtype=float).reshape(shape)
        return a

    d["ftle"] = field("ftle")
    d["ld"] = field("ld_arc")
    d["ld_p05"] = field("ld_p05")
    d["fli"] = field("fli")
    d["ftle_ok"] = True
    d["ftle_var"] = _varies(d["ftle"])
    d["src_tag"] = src_tag
    d["aux_tag"] = aux_tag
    return d


def save_indicators(d, title, fname, panels, *, colorbar=False):
    """Varying fields only. Four square panels stay 2×2; three-panel
    figures and wide domains (Bickley) stack vertically, with no blank cell."""
    cells = []
    for field, lab in panels:
        if field is None or not _varies(field):
            continue
        cells.append((field, lab, None))
    if not cells:
        raise RuntimeError("no varying panels for %s" % fname)
    x0, x1, y0, y1 = d["extent"]
    ar = abs(x1 - x0) / max(abs(y1 - y0), 1e-12)
    n = len(cells)
    if ar > 1.8 or n != 4:
        nrows, ncols = n, 1
    else:
        nrows, ncols = 2, 2
    extra = 0.8 if colorbar else 0.0
    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=atlas.fig_size(nrows, ncols, d["extent"], extra_w=extra),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle(atlas.with_tau(title, d), fontsize=atlas.SUPTITLE_FS)
    atlas._paint_grid(
        fig, axs, cells, d["xlab"], d["ylab"], d["extent"],
        colorbar=colorbar,
    )
    atlas.save_both(fig, fname)
    plt.close(fig)
    atlas.copy_selected_pdf(fname)


def four(d):
    return [
        (d["fin"], r"FinSTOD  $\log(1+s)$"),
        (d["ftle"], atlas.ftle_title(d)),
        (d["ld"], r"LD  $p=1$"),
        (d["fli"], _const_title("FLI", d["fli"])),
    ]


def main():
    os.makedirs(atlas.OUT, exist_ok=True)
    os.makedirs(atlas.PAPER_FIG, exist_ok=True)

    fro = load_pair("prod_froeschle_dof2_m4",
                    "aux_interp_prod_froeschle_dof2_m4")
    save_indicators(
        fro,
        r"Froeschlé $m=4$, $K=0.30$",
        "indicators_froeschle_m4.png",
        four(fro),
        colorbar=True,
    )

    pure = load_pair("test2_bickley_pure_shear",
                     "aux_interp_test2_bickley_pure_shear")
    save_indicators(
        pure,
        r"Bickley jet (pure shear), $m=2$",
        "indicators_bickley_pure.png",
        four(pure),
    )

    jet = load_pair("test2_bickley_jet_shear",
                    "aux_interp_test2_bickley_jet_shear")
    save_indicators(
        jet,
        r"Bickley jet (perturbed), $m=2$",
        "indicators_bickley_jet.png",
        four(jet),
    )

    duff4 = load_pair("prod_duffing_m6_1000x1000",
                      "aux_interp_prod_duffing_m6_1000x1000")
    save_indicators(
        duff4,
        r"Coupled Duffing $m=6$, $(x_1,x_2)$",
        "indicators_duffing_tau4.png",
        four(duff4),
    )

    duff20 = load_pair("interp_duffing_m6_tau20",
                       "aux_interp_interp_duffing_m6_tau20")
    save_indicators(
        duff20,
        r"Coupled Duffing $m=6$, $(x_1,x_2)$",
        "indicators_duffing_tau20_xx.png",
        four(duff20),
    )

    duff_xv = load_pair("interp_duffing_m6_xv",
                        "aux_interp_interp_duffing_m6_xv")
    save_indicators(
        duff_xv,
        r"Coupled Duffing $m=6$, $(x_1,v_1)$",
        "indicators_duffing_tau20_xv.png",
        four(duff_xv),
    )

    fput = load_pair("prod_fput_m8", "aux_interp_prod_fput_m8")
    save_indicators(
        fput,
        r"FPUT $m=8$",
        "indicators_fput_m8.png",
        four(fput),
    )

    save_indicators(
        fro,
        r"Froeschlé $m=4$, $K=0.30$",
        "ld_p1_vs_p05_froeschle.png",
        [
            (fro["ld"], r"LD  $p=1$"),
            (fro["ld_p05"], r"LD  $p=0.5$"),
        ],
    )


if __name__ == "__main__":
    main()
