# -*- coding: utf-8 -*-
"""Stdlib progress / ETA for Phase 12 FPUT S-bar aux (no numpy).

Wall time scales with L because each tag is a 60-way array. Default job list
matches configs/aux_fput_sbar.yaml.
"""

import argparse
import glob
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PIPE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

JOBS = [
    ("prod_fput_m8_tau05", 500),
    ("prod_fput_m8_tau10", 1000),
    ("prod_fput_m8_tau15", 1500),
    ("prod_fput_m8_tau20", 2000),
    ("prod_fput_m8", 2500),
]
N_SHARDS = 60


def _root():
    here = os.path.dirname(os.path.abspath(__file__))
    pipe = os.path.dirname(here)
    return os.path.abspath(os.path.join(pipe, "..", "cesvima_output"))


def fmt(seconds):
    try:
        from stod_nd.progress import format_duration
        return format_duration(seconds)
    except Exception:
        if seconds is None or seconds < 0:
            return "?"
        seconds = int(seconds)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h:
            return "%d:%02d:%02d" % (h, m, s)
        return "%d:%02d" % (m, s)


def bar(frac, width=24):
    frac = max(0.0, min(1.0, float(frac)))
    fill = int(round(frac * width))
    return "#" * fill + "-" * (width - fill)


def snapshot(root):
    rows = []
    work_left = 0.0
    work_tot = 0.0
    for tag, L in JOBS:
        aux = os.path.join(root, "aux_sbar_%s" % tag)
        cons = os.path.isfile(os.path.join(aux, "results_consolidated.npz"))
        n = len(glob.glob(os.path.join(aux, "results", "result_*.npz")))
        if cons:
            n = N_SHARDS
        frac = n / float(N_SHARDS)
        work_tot += L
        work_left += L * (1.0 - frac)
        rows.append((tag, L, n, cons, frac))
    return rows, work_left, work_tot


def line(t0, rows, work_left, work_tot):
    elapsed = time.time() - t0
    # 60-way arrays: wall scales with remaining L. tau=5 was ~150 s / L=500.
    eta = work_left * (150.0 / 500.0) if work_left > 0 else 0
    overall = 1.0 - (work_left / work_tot if work_tot else 0.0)
    bits = []
    for tag, L, n, cons, frac in rows:
        if tag == "prod_fput_m8":
            lab = "25"
        else:
            lab = tag.split("tau")[-1]
        mark = "done" if cons else "%d/%d" % (n, N_SHARDS)
        bits.append("t%s=%s" % (lab, mark))
    return (
        "[%s] S-bar  [%s] %d%%  elapsed=%s  ETA=%s  %s"
        % (
            time.strftime("%H:%M:%S"),
            bar(overall),
            int(round(100 * overall)),
            fmt(elapsed),
            fmt(eta),
            "  ".join(bits),
        )
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=15.0)
    p.add_argument("--once", action="store_true")
    p.add_argument("--append", default=None, help="Also append each line to this file")
    args = p.parse_args()
    root = _root()
    t0 = time.time()
    while True:
        rows, work_left, work_tot = snapshot(root)
        msg = line(t0, rows, work_left, work_tot)
        print(msg, flush=True)
        if args.append:
            with open(args.append, "a") as f:
                f.write(msg + "\n")
        if args.once or work_left <= 0:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
