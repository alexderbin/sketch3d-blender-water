"""Move timing: the progress curve, the hold at the end, the frame-edge speed, fitting the move under the format ceiling.

Curves: linear (trailer: an even move, edges cut by the cut), ease (an even move with short edges — the fraction ramp of the move, as measured
0.10–0.15), smooth (an S-curve over the whole move — "arrival", documentary), settle (the move at the start — the fraction move_share, then
a hold: a history short), accel (toward a hit). The edge speed is measured as in the moves_probe script: rays through the frame edge hit the
place field, the points are projected into the next frame; shift — % of the frame width per second, scale — % per second. Pure numpy."""
import math

import numpy as np

from components.camera.move import frame as F

MODES = ("linear", "ease", "smooth", "settle", "accel")


def progress(n, mode="ease", ramp=0.12, hold_end=0.0, move_share=0.6, fps=24.0):
    """The fraction of the path (0 → 1, monotonic) for each of n frames; hold_end — seconds of hold at the end (the camera has settled before the cut)."""
    if mode not in MODES:
        raise ValueError(f"curve '{mode}': available are {', '.join(MODES)}")
    if n <= 1:
        return np.zeros(1)
    t = np.arange(n) / (n - 1)
    hold = min(0.9, max(0.0, float(hold_end) * float(fps) / (n - 1)))
    tm = np.clip(t / (1.0 - hold), 0.0, 1.0)
    if mode == "linear":
        return tm
    if mode == "smooth":
        return tm * tm * (3 - 2 * tm)
    if mode == "accel":
        return tm * tm
    if mode == "settle":
        x = np.clip(tm / max(1e-6, float(move_share)), 0.0, 1.0)
        return x * x * (3 - 2 * x)
    r = min(0.5, max(1e-3, float(ramp)))                              # ease: the speed is a trapezoid with cosine edges
    s = np.linspace(0.0, 1.0, 2001); v = np.ones_like(s)
    v = np.where(s < r, (1 - np.cos(math.pi * s / r)) / 2, v); v = np.where(s > 1 - r, (1 - np.cos(math.pi * (1 - s) / r)) / 2, v)
    P = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2)]); P /= P[-1]
    return np.interp(tm, s, P)


def _rim(n=4):
    """Frame-edge points (fractions of the screen) — as in moves_probe, a 5 % inset."""
    a = np.linspace(0.05, 0.95, n)
    return np.array([(x, y) for x in a for y in a if x in (a[0], a[-1]) or y in (a[0], a[-1])])


def edge_motion(cams, field, fps=24.0, aspect=16 / 9):
    """Movement of the frame edge between neighbouring frames: (shift, % of width/s; scale, %/s) — arrays of length N−1. cams — [{pos, look,
    lens}]; an edge point is a ray hit on the place field, a miss — a point at 5 km (sky: only the rotation matters)."""
    if len(cams) < 2:
        return np.zeros(0), np.zeros(0)
    rim = _rim(); sh, sc = [], []
    for a, b in zip(cams[:-1], cams[1:]):
        r, u, f = F.basis(a["pos"], a["look"]); th, tv = F.tans(a["lens"], aspect)
        D = f[None] + r[None] * ((rim[:, :1] - 0.5) * 2 * th) - u[None] * ((rim[:, 1:] - 0.5) * 2 * tv)
        t, pt, _ = field.cast(np.repeat(np.asarray(a["pos"], float)[None], len(D), 0), D)
        far = np.asarray(a["pos"], float)[None] + D / np.linalg.norm(D, axis=1, keepdims=True) * 5000.0
        pt = np.where(np.isfinite(t)[:, None], pt, far)
        x, y, z = F.project(b["pos"], b["look"], b["lens"], aspect, pt)
        ok = z > 0
        if ok.sum() < 3:
            sh.append(np.nan); sc.append(np.nan); continue
        dx = (x - rim[:, 0])[ok]; dy = (y - rim[:, 1])[ok] / aspect   # frame height → in fractions of the width
        sh.append(math.hypot(float(np.mean(dx)), float(np.mean(dy))) * 100.0 * fps)
        c = rim[ok] - 0.5; c[:, 1] /= aspect
        p1 = np.stack([x[ok] - 0.5, (y[ok] - 0.5) / aspect], 1)
        sc.append(float(np.mean(np.sum(p1 * c, 1) / np.sum(c * c, 1)) - 1.0) * 100.0 * fps)
    return np.asarray(sh), np.asarray(sc)


def speeds(cams, field, fps=24.0, aspect=16 / 9):
    """Edge speed summary: p50, p90 of the shift (%/s) and p90 of |scale| (%/s)."""
    sh, sc = edge_motion(cams, field, fps, aspect)
    sh, sc = sh[np.isfinite(sh)], sc[np.isfinite(sc)]
    if not len(sh):
        return {"shift_p50": 0.0, "shift_p90": 0.0, "scale_p90": 0.0}
    return {"shift_p50": float(np.percentile(sh, 50)), "shift_p90": float(np.percentile(sh, 90)),
            "scale_p90": float(np.percentile(np.abs(sc), 90)) if len(sc) else 0.0}


def fit(build, field, ceil_shift, ceil_scale, fps=24.0, aspect=16 / 9, tries=8):
    """Fitting the move amplitude k ∈ (0, 1] under the edge-speed ceiling (p90): build(k) → cameras. Exceeding it makes the move shorter, not the shot
    faster. → (k, cameras, speeds)."""
    cams = build(1.0); v = speeds(cams, field, fps, aspect)
    if v["shift_p90"] <= ceil_shift and v["scale_p90"] <= ceil_scale:
        return 1.0, cams, v
    lo, hi, best = 0.05, 1.0, None
    for _ in range(tries):
        k = (lo + hi) / 2; c = build(k); w = speeds(c, field, fps, aspect)
        if w["shift_p90"] <= ceil_shift and w["scale_p90"] <= ceil_scale:
            lo, best = k, (k, c, w)
        else:
            hi = k
    if best:
        return best
    c = build(lo)
    return lo, c, speeds(c, field, fps, aspect)
