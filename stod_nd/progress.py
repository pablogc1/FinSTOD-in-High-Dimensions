# -*- coding: utf-8 -*-
"""Shared progress bar and human duration formatting."""
from __future__ import annotations


def format_duration(seconds):
    """Largest unit the duration reaches: s, m, h, or d.

    45 → ``45s``, 90 → ``1.5m``, 7200 → ``2.0h``, 90000 → ``1.0d``.
    Negative or non-finite values are ``?``.
    """
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "?"
    if s < 0.0 or s != s:  # NaN
        return "?"
    if s < 60.0:
        return "%.0fs" % s if s >= 9.95 else "%.1fs" % s
    if s < 3600.0:
        return "%.1fm" % (s / 60.0)
    if s < 86400.0:
        return "%.1fh" % (s / 3600.0)
    return "%.1fd" % (s / 86400.0)


def format_array_watch(done, tot, elapsed, running, bar_width=24):
    """One master-log line for a Slurm array (ints only; safe from the shell)."""
    done = int(done or 0)
    tot = max(int(tot or 0), 1)
    elapsed = int(elapsed or 0)
    running = int(running or 0)
    frac = done / float(tot)
    if done <= 0:
        eta = -1.0
    elif done >= tot:
        eta = 0.0
    else:
        eta = elapsed * (tot - done) / float(done)
    bar = ascii_bar(frac, bar_width, filled_char="#", tip="", empty="-")
    return "[%s] %d/%d (%d%%)  queue=%d  elapsed=%s  ETA=%s" % (
        bar, done, tot, int(100.0 * frac), running,
        format_duration(elapsed), format_duration(eta),
    )


def ascii_bar(frac, width=20, filled_char="=", tip=">", empty="."):
    frac = 0.0 if frac != frac else max(0.0, min(1.0, float(frac)))
    filled = int(width * frac)
    if filled >= width:
        return filled_char * width
    if filled <= 0:
        return empty * width
    return filled_char * (filled - (1 if tip else 0)) + (tip or "") + empty * (width - filled)


def print_chunk_progress(label, done, total, elapsed, kind="", bar_len=20):
    """Worker-log line matching the existing FinSTOD/FTLE chrome."""
    total = max(int(total), 1)
    done = int(done)
    frac = done / float(total)
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else -1.0
    bar = ascii_bar(frac, bar_len)
    who = " ".join(p for p in (label, kind) if p)
    prefix = ("%s " % who) if who else ""
    print(
        "    [%s] %5.1f%% | %s%d/%d cells | %s elapsed | ETA %s"
        % (bar, 100.0 * frac, prefix, done, total,
           format_duration(elapsed), format_duration(eta)),
        flush=True,
    )


def print_loop_progress(label, step, n_steps, elapsed, kind="", bar_len=20, every=None):
    """Progress along an integration window (basins, S-bar, frequency plane)."""
    n_steps = max(int(n_steps), 1)
    step = int(step)
    every = int(every) if every else max(1, n_steps // 20)
    if step % every != 0 and step != n_steps:
        return
    frac = step / float(n_steps)
    rate = step / elapsed if elapsed > 0 else 0.0
    eta = (n_steps - step) / rate if rate > 0 else -1.0
    bar = ascii_bar(frac, bar_len)
    who = " ".join(p for p in (label, kind) if p)
    prefix = ("%s " % who) if who else ""
    print(
        "    [%s] %5.1f%% | %s%d/%d steps | %s elapsed | ETA %s"
        % (bar, 100.0 * frac, prefix, step, n_steps,
           format_duration(elapsed), format_duration(eta)),
        flush=True,
    )
