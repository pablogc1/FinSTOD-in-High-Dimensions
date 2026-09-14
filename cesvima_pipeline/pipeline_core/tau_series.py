# -*- coding: utf-8 -*-
"""Expand a tau_series YAML into (template, run_tag, n_levels, tau) jobs.

Each member inherits its template system/grid/FTLE settings. Output goes to a
new directory ``<template_run_tag>_tauNN`` so existing productions are never
touched. ``tau = 25`` for ABC and FPUT is already on disk and is omitted from
the ABC/FPUT m=8 series config.
"""
from __future__ import annotations

import os

from pipeline_core.generate_tasks import load_config, resolve_run_dir

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.dirname(_HERE)


def series_tag(base, tau):
    t = float(tau)
    if abs(t - round(t)) < 1e-9:
        return "%s_tau%02d" % (base, int(round(t)))
    return "%s_tau%g" % (base, t)


def n_levels_for_tau(dt_out, tau):
    dt = float(dt_out)
    tau = float(tau)
    n_levels = int(round(tau / dt))
    if n_levels < 1:
        raise ValueError("tau=%s / dt_out=%s gives n_levels=%s" % (tau, dt, n_levels))
    if abs(n_levels * dt - tau) > 1e-6:
        raise ValueError(
            "tau=%s is not an integer number of steps at dt_out=%s (got L=%s)"
            % (tau, dt, n_levels)
        )
    return n_levels


def iter_series_jobs(series_path):
    """Yield dicts: template, run_tag, n_levels, tau, dt_out."""
    series = load_config(series_path)
    if series.get("kind") != "tau_series":
        raise ValueError("%s is not kind: tau_series" % series_path)
    taus = series["taus"]
    series_dir = os.path.dirname(os.path.abspath(series_path))
    for member in series["members"]:
        if isinstance(member, str):
            rel = member
            prefix = None
        else:
            rel = member["config"]
            prefix = member.get("run_tag_prefix")
        candidates = []
        if os.path.isabs(rel):
            candidates.append(rel)
        else:
            candidates.append(os.path.normpath(os.path.join(series_dir, rel)))
            candidates.append(os.path.normpath(os.path.join(_PIPE, rel)))
            candidates.append(os.path.normpath(os.path.join(_PIPE, "configs", rel)))
        tmpl_path = next((p for p in candidates if os.path.isfile(p)), None)
        if tmpl_path is None:
            raise FileNotFoundError("series member config not found: %s" % rel)
        tmpl = load_config(tmpl_path)
        base = prefix or tmpl.get("run_tag", "run")
        dt = float(tmpl["system"]["dt_out"])
        for tau in taus:
            n_levels = n_levels_for_tau(dt, tau)
            tag = series_tag(base, tau)
            if tag == tmpl.get("run_tag"):
                raise ValueError("series tag %s collides with the production run_tag" % tag)
            yield {
                "template": tmpl_path,
                "run_tag": tag,
                "n_levels": n_levels,
                "tau": float(tau),
                "dt_out": dt,
                "production_tag": tmpl.get("run_tag"),
            }


def job_run_dir(job):
    tmpl = load_config(job["template"])
    run_dir, _ = resolve_run_dir(tmpl, run_tag=job["run_tag"])
    return run_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="List tau-series jobs")
    parser.add_argument("series")
    args = parser.parse_args()
    for j in iter_series_jobs(args.series):
        print("%s  L=%s  tau=%s  (template %s, leaves %s untouched)"
              % (j["run_tag"], j["n_levels"], j["tau"],
                 os.path.basename(j["template"]), j["production_tag"]))
