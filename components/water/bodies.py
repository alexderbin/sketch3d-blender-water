"""Water bodies of a site: the water of the Site relief (the near grid level, Ground.level) → connected water bodies: area, depth,
kind (sea — the site flag sea, river — the flag river, otherwise a lake or a pond by area), fetch — the distance to the shore
against the wind (the median over the water body; a water body at the plot edge is open water, with the sea's fetch), for a
river — the level slope, the Manning speed v = R^⅔·S^½ / n and the Froude number Fr = v/√(g·h) (calm, riffle, rapid). Pure numpy
+ scipy (the project's Python)."""
import math

import numpy as np

G_ = 9.81
MANNING_N = 0.035                       # a channel with pebbles and grass at the bank (Chow 1959: 0.03–0.05)


def _label(mask):
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    return lab, n


def fetch_field(wet, res, to, step=4.0, far=50000.0):
    """Fetch in each wet cell: the path against the wind (to — where it blows toward) to land, m; if it leaves the plot edge — far
    (open water beyond the edge). On a coarse grid of step m — cheap."""
    k = max(int(round(step / res)), 1); W = wet[::k, ::k]; r = res * k
    ny, nx = W.shape; Y, X = np.mgrid[0:ny, 0:nx].astype(float)
    F = np.full(W.shape, np.nan); open_ = np.zeros(W.shape, bool); hit = ~W.copy()
    dx, dy = -to[0], -to[1]
    for s in range(1, int(max(nx, ny) * 1.5)):
        xi, yi = np.round(X + dx * s).astype(int), np.round(Y + dy * s).astype(int)
        out = (xi < 0) | (yi < 0) | (xi >= nx) | (yi >= ny)
        land = np.zeros(W.shape, bool); ok = ~out
        land[ok] = ~W[yi[ok], xi[ok]]
        new = W & np.isnan(F) & (land | out)
        F[new & land] = s * r; open_[new & out] = True; F[new & out] = far
        if not np.isnan(F[W]).any(): break
    F[~W] = 0.0
    full = np.repeat(np.repeat(np.nan_to_num(F, nan=far), k, 0), k, 1)[:wet.shape[0], :wet.shape[1]]
    return full


def _dense(path, step):
    P = np.asarray(path, float)
    if len(P) < 2: return P
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(int(np.linalg.norm(b - a) / step), 1); out += [a + (b - a) * (i / n) for i in range(1, n + 1)]
    return np.array(out)


def bodies(Gr, to=(1.0, 0.0), min_area=4.0):
    """Water bodies from Ground: [{id, kind, area, cells, depth_mean, depth_max, fetch, open, center, bbox, river: {slope, v, froude, regime}}]."""
    lv = Gr.level; res = float(Gr.nres)
    ny, nx = lv.shape
    X, Y = np.meshgrid(Gr.nx0 + np.arange(nx) * res, Gr.ny0 + np.arange(ny) * res)
    D = np.where(np.isfinite(lv), lv - Gr.Hn[:ny, :nx], 0.0)
    wet = np.isfinite(lv) & (D > 0.02)
    if not wet.any(): return []
    lab, n = _label(wet)
    F = fetch_field(wet, res, to)
    feats = Gr.loc.get("features") or []
    sea_ids = [f for f in feats if f.get("id") == "sea" or str(f.get("id", "")).endswith("_sea") or f.get("sea")]
    rivers = [{"path": [q[:2] for q in v.get("pts") or []], "w": v.get("w", 10.0)}                 # river paths — location.json: paths
              for v in (Gr.loc.get("paths") or {}).values() if isinstance(v, dict) and v.get("kind") == "river"]
    rivers += [f for f in feats if f.get("kind") == "river" and f.get("path")]
    out = []
    for i in range(1, n + 1):
        m = lab == i; area = float(m.sum()) * res * res
        if area < min_area: continue
        edge = bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
        d = D[m]; cx, cy = float(X[m].mean()), float(Y[m].mean())
        b = {"id": f"w{i}", "area": round(area, 1), "cells": int(m.sum()), "depth_mean": round(float(d.mean()), 2),
             "depth_max": round(float(d.max()), 2), "center": [round(cx, 1), round(cy, 1)],
             "bbox": [round(float(X[m].min()), 1), round(float(Y[m].min()), 1), round(float(X[m].max()), 1), round(float(Y[m].max()), 1)]}
        fe = F[m]; b["open"] = bool(edge and (fe >= 49999).mean() > 0.2)
        b["fetch"] = None if b["open"] else round(float(np.median(fe)), 1)
        sub = max(1, int(m.sum()) // 400); px, py = X[m][::sub], Y[m][::sub]
        near_river = False
        for f in rivers:                                            # the fraction of the water body's cells near the channel (the path is densified to 2 m)
            P = _dense(f.get("path") or [], 2.0)
            if len(P) and (np.min(np.hypot(P[:, 0][:, None] - px, P[:, 1][:, None] - py), 0) < f.get("w", 10) / 2 + 3 * res).mean() > 0.5:
                near_river = True
        if sea_ids and b["open"]: b["kind"] = "sea"
        elif near_river: b["kind"] = "river"
        elif area >= 5000: b["kind"] = "lake"
        else: b["kind"] = "pond"
        if b["kind"] == "river":                                    # level slope → speed and regime
            gy, gx = np.gradient(np.where(m, lv, np.nan), res)
            s_ = np.hypot(gx, gy)[m]; s_ = s_[np.isfinite(s_)]
            S = float(np.median(s_)) if len(s_) else 1e-4; S = max(S, 1e-5)
            h = max(float(np.median(d)), 0.05); v = h ** (2 / 3) * math.sqrt(S) / MANNING_N; fr = v / math.sqrt(G_ * h)
            b["river"] = {"slope": round(S, 5), "v": round(v, 2), "froude": round(fr, 2),
                          "regime": "glide" if fr < 0.3 else "current" if fr < 0.7 else "riffle" if fr < 1.0 else "threshold"}
        out.append(b)
    out.sort(key=lambda r: -r["area"])
    return out
