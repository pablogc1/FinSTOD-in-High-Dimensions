"""FinSTOD volume as a solid cube.

The scalar field is painted on the six faces of the sampled cube —
not three interior slices. Faces are interpolated only for display.

Stills (PDF/PNG) draw the three camera-facing faces. The HTML Plotly
figure draws all six so rotation stays a closed cube.

  python diagnostics/plot_volume_cube.py
  python diagnostics/plot_volume_cube.py --only lorenz
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.ndimage import zoom

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "diagnostics"))

import plot_paper_atlas as atlas  # noqa: E402

FACE_N = 240
HTML_N = 160
CAMERAS = [
    (20, -55, ""),
    (22, 35, "_azim035"),
    (18, 125, "_azim125"),
    (28, -140, "_azim140"),
]
RUNS = {
    "abc": {
        "tag": "prod_abc_3d_volume",
        "stem": "abc_volume_cube",
        "still_title": r"ABC flow, $m=3$, $\tau=%s$: FinSTOD $\log(1+s)$",
        "html_title": "ABC FinSTOD cube — drag to rotate, then pick a still",
    },
    "lorenz": {
        "tag": "prod_lorenz96_m6_3d_volume",
        "stem": "lorenz_volume_cube",
        "still_title": r"Lorenz-96, $m=6$, $(y_1,y_2,y_3)$, $\tau=%s$: FinSTOD $\log(1+s)$",
        "html_title": "Lorenz-96 FinSTOD cube — drag to rotate, then pick a still",
    },
}


def _load_volume(tag):
    meta = json.load(open(os.path.join(atlas.ROOT, tag, "meta.json")))
    z = np.load(os.path.join(atlas.ROOT, tag, "results_consolidated.npz"))
    nx, ny, nz = meta["grid_shape"]
    raw = np.asarray(z["raw_scores"], dtype=float).reshape(nx, ny, nz)
    fin = np.log1p(np.clip(raw, 0, None))
    fin[~np.isfinite(raw)] = np.nan
    return {
        "vol": atlas.to01(fin),
        "extent": meta["extent"],
        "names": meta.get("coord_names", ["x", "y", "z"]),
        "tau": float(meta["total_time"]),
        "shape": (nx, ny, nz),
        "tag": tag,
    }


def _upsample2(face, n):
    face = np.array(face, dtype=float, copy=True)
    bad = ~np.isfinite(face)
    if bad.any():
        fill = np.nanmedian(face)
        if not np.isfinite(fill):
            fill = 0.0
        face[bad] = fill
    zoom_f = (n / face.shape[0], n / face.shape[1])
    return np.clip(zoom(face, zoom_f, order=3), 0.0, 1.0)


def _coords(n, lo, hi):
    return np.linspace(lo, hi, n)


def _cube_edges(extent):
    x0, x1, y0, y1, z0, z1 = extent
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    segs = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return [[corners[a], corners[b]] for a, b in segs]


def _boundary_faces(vol, n):
    return {
        "z0": _upsample2(vol[:, :, 0], n),
        "z1": _upsample2(vol[:, :, -1], n),
        "y0": _upsample2(vol[:, 0, :], n),
        "y1": _upsample2(vol[:, -1, :], n),
        "x0": _upsample2(vol[0, :, :], n),
        "x1": _upsample2(vol[-1, :, :], n),
    }


def _face_xyz(d, key, face, n):
    x0, x1, y0, y1, z0, z1 = d["extent"]
    xs, ys, zs = _coords(n, x0, x1), _coords(n, y0, y1), _coords(n, z0, z1)
    if key == "z0":
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        return X, Y, np.full_like(X, z0), face
    if key == "z1":
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        return X, Y, np.full_like(X, z1), face
    if key == "y0":
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        return X, np.full_like(X, y0), Z, face
    if key == "y1":
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        return X, np.full_like(X, y1), Z, face
    if key == "x0":
        Y, Z = np.meshgrid(ys, zs, indexing="ij")
        return np.full_like(Y, x0), Y, Z, face
    if key == "x1":
        Y, Z = np.meshgrid(ys, zs, indexing="ij")
        return np.full_like(Y, x1), Y, Z, face
    raise KeyError(key)


def _view_vec(elev, azim):
    """Camera location relative to the origin, matplotlib view_init convention."""
    e, a = np.radians(elev), np.radians(azim)
    return np.array([
        np.cos(e) * np.cos(a),
        np.cos(e) * np.sin(a),
        np.sin(e),
    ])


def _visible_keys(elev, azim):
    """The three faces of the cube that point toward the camera."""
    v = _view_vec(elev, azim)
    return [
        "x1" if v[0] >= 0 else "x0",
        "y1" if v[1] >= 0 else "y0",
        "z1" if v[2] >= 0 else "z0",
    ]


def _style_axes(ax, d):
    x0, x1, y0, y1, z0, z1 = d["extent"]
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel(r"$%s$" % d["names"][0], fontsize=12)
    ax.set_ylabel(r"$%s$" % d["names"][1], fontsize=12)
    ax.set_zlabel(r"$%s$" % d["names"][2], fontsize=12)
    ax.tick_params(labelsize=9)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("none")
    ax.grid(False)


def _draw_faces(ax, d, faces, keys, n, cmap):
    for key in keys:
        X, Y, Z, face = _face_xyz(d, key, faces[key], n)
        ax.plot_surface(
            X, Y, Z,
            rstride=1, cstride=1,
            facecolors=cmap(face),
            shade=False,
            linewidth=0,
            antialiased=False,
            zorder=1,
        )


def render_still(d, faces, elev, azim, stem, still_title):
    cmap = plt.get_cmap(atlas.CMAP)
    fig = plt.figure(figsize=(7.2, 6.6))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle(still_title % atlas.fmt_tau(d["tau"]), fontsize=13)
    _draw_faces(ax, d, faces, _visible_keys(elev, azim), FACE_N, cmap)
    ax.add_collection3d(Line3DCollection(
        _cube_edges(d["extent"]), colors="0.15", linewidths=1.2,
    ))
    _style_axes(ax, d)
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    atlas.save_both(fig, stem + ".png")
    plt.close(fig)
    atlas.copy_selected_pdf(stem + ".pdf")


def render_html(d, path, html_title):
    import plotly.graph_objects as go

    faces = _boundary_faces(d["vol"], HTML_N)
    traces = []
    for key, face in faces.items():
        X, Y, Z, col = _face_xyz(d, key, face, HTML_N)
        traces.append(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=col,
            colorscale="Inferno",
            cmin=0.0, cmax=1.0,
            showscale=False,
            opacity=1.0,
            hoverinfo="skip",
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
            contours=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
        ))
    for a, b in _cube_edges(d["extent"]):
        traces.append(go.Scatter3d(
            x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]],
            mode="lines",
            line=dict(color="#222222", width=5),
            hoverinfo="skip",
            showlegend=False,
        ))
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=html_title,
        scene=dict(
            xaxis_title=d["names"][0],
            yaxis_title=d["names"][1],
            zaxis_title=d["names"][2],
            aspectmode="cube",
            camera=dict(eye=dict(x=1.55, y=-1.55, z=1.15)),
            bgcolor="white",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor="white",
    )
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    print("  wrote", path)


def render_run(key, *, stills=True, html_dir=None):
    cfg = RUNS[key]
    d = _load_volume(cfg["tag"])
    print("loaded", d["tag"], d["shape"], "tau", d["tau"])
    if stills:
        faces = _boundary_faces(d["vol"], FACE_N)
        for elev, azim, suffix in CAMERAS:
            stem = cfg["stem"] + suffix
            print("still elev=%s azim=%s faces=%s -> %s"
                  % (elev, azim, ",".join(_visible_keys(elev, azim)), stem))
            render_still(d, faces, elev, azim, stem, cfg["still_title"])
    dest = html_dir or atlas.OUT
    os.makedirs(dest, exist_ok=True)
    render_html(d, os.path.join(dest, cfg["stem"] + ".html"), cfg["html_title"])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(RUNS), default=None)
    parser.add_argument("--html-only", action="store_true")
    parser.add_argument("--html-dir", default=None)
    args = parser.parse_args()
    os.makedirs(atlas.OUT, exist_ok=True)
    os.makedirs(atlas.PAPER_FIG, exist_ok=True)
    keys = [args.only] if args.only else list(RUNS)
    for key in keys:
        render_run(key, stills=not args.html_only, html_dir=args.html_dir)


if __name__ == "__main__":
    main()
