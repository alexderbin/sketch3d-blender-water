"""Man-made site relief: ramparts and moats (contour hillforts, promontory forts, ring forts, town walls, bastioned), settlement
mounds (tell, terp, platform), the cultural layer and fill, field terraces (lynchets), sunken roads, ridge-and-furrow arable
land, quarries with benches and spoil heaps, dams, ash dunes. One rule for all shapes: a fresh profile by soil (angle of repose;
a rampart from the moat's spoil, volume × bulking) → aging by linear slope diffusion (Culling 1960, Hanks 1984: the shape's
increment is smoothed with a Gaussian σ = √(2κt), κ by soil, checked against lidar measurements) → gullies at mounds and spoil
heaps (Forti 2023: deeper toward the foot, an alluvial fan). The knowledge is in library/terrain/works.yaml. Part of terrain.py:
the KINDS shapes (ORDER — after stone on steep ground, before platforms and roads), the ARCH archetypes. numpy; cv2 and yaml —
only when building a site (.venv)."""
import math, zlib

import numpy as np

from components.site import terrain as T
import core.paths

KINDS = T.WORKS
_K = {}


def know():
    if not _K:
        import os, yaml
        _K.update(yaml.safe_load(open(os.path.join(core.paths.ROOT, "library", "terrain", "works.yaml"))))
    return _K


def soil(f):
    S = know()["soils"]; return S.get(f.get("soil", "loam"), S["loam"])


def sigma(f):
    """Aging: σ = √(2κt), m (κ in m²/thousand years, t in years)."""
    return math.sqrt(2.0 * float(f.get("kappa", soil(f)["kappa"])) * float(f.get("age", 0)) / 1000.0)


def blur(A, s):
    """Gaussian with σ = s cells by three box windows (cumulative sums; numpy only)."""
    if s < 0.35: return A
    w = int(math.floor(math.sqrt(12 * s * s / 3 + 1))); w -= (w % 2 == 0); w = max(w, 1); r = w // 2
    for ax in (0, 1):
        for _ in range(3):
            P = np.pad(A, [(r + 1, r) if a == ax else (0, 0) for a in (0, 1)], mode="edge"); C = np.cumsum(P, axis=ax)
            A = (np.take(C, range(w, C.shape[ax]), axis=ax) - np.take(C, range(0, C.shape[ax] - w), axis=ax)) / w
    return A


def _merge(f, group="works"):
    """The kind's defaults (type) from works.yaml + the shape's fields; nested bank/ditch are merged."""
    base = dict(know()[group].get(f.get("type"), {}) if f.get("type") else {}); out = {**base, **f}
    for k in ("bank", "ditch"):
        if isinstance(base.get(k), dict) or isinstance(f.get(k), dict): out[k] = {**(base.get(k) or {}), **(f.get(k) or {})}
    return out


# ---------------------------------------------------------------- ramparts and moats
def section_params(f):
    """Line profile: rampart, berm, moat, counterscarp (m, °); the rampart height comes from the moat's volume (balance), if there is a moat."""
    p = _merge(f, "works"); S = soil(p); B = p.get("bank") or {}; D = p.get("ditch") or {}
    rep = float(S["repose"]); a_o = float(B.get("out") or rep - 3); a_i = float(B.get("in") or rep); a_c = float(D.get("out") or rep - 6)   # tuned against lidar
    c = float(B.get("crest", 2.5)); d = float(D.get("depth", 0.0)); wd = float(D.get("w", 1.0)); cs = float(p.get("cs", 0.15))
    to, ti, tc = (math.tan(math.radians(a)) for a in (a_o, a_i, a_c))
    Ad = d * (wd + 0.5 * d / to + 0.5 * d / tc) if d > 0 else 0.0
    h = float(B.get("h", 2.5))
    if d > 0 and p.get("balance", True) and not B.get("fix"):     # rampart — from the moat spoil: h·c + h²/2·(1/tg_about + 1/tg_within) = (1 − cs)·bulking·A_moat
        A = (1 - cs) * float(S["bulk"]) * Ad; a = 0.5 / to + 0.5 / ti; h = (-c + math.sqrt(c * c + 4 * a * A)) / (2 * a)
    Acs = cs * float(S["bulk"]) * Ad; ccs = 1.5; t30 = math.tan(math.radians(28)); hcs = (-ccs + math.sqrt(ccs * ccs + 4 * Acs / t30)) / (2 / t30) if Acs > 0 else 0.0
    return {"h": h, "c": c, "to": to, "ti": ti, "tc": tc, "b": float(p.get("berm", 0.5)), "d": d, "wd": wd, "hcs": hcs, "ccs": ccs, "t30": t30,
            "inner": bool(D.get("inner")), "revet": bool(p.get("revet")), "Ad": Ad}


def section(u, q, k=1.0):
    """Fresh profile across a line (u — from the crest axis, outward is +), an addition to the ground, m; k — the line scale (outer ones are smaller)."""
    h, c, d, wd, b, hcs = q["h"] * k, q["c"], q["d"] * k, q["wd"], q["b"], q["hcs"] * k
    to = 30.0 if q["revet"] else q["to"]                            # a revetted face is almost sheer (Naarden: 60–72°)
    z = np.where(u >= 0, np.clip(h - np.maximum(0, u - c / 2) * to, 0, h), np.clip(h - np.maximum(0, -u - c / 2) * q["ti"], 0, h))
    if d > 0:
        sg = -1.0 if q["inner"] else 1.0; v = u * sg                 # moat inside (a henge) — mirrored
        t_side = q["ti"] if q["inner"] else q["to"]
        u0 = c / 2 + h / t_side + b; u3 = u0 + d / t_side + wd + d / q["tc"]
        dz = np.where((v > u0) & (v < u3), -np.minimum(d, np.minimum((v - u0) * t_side, (u3 - v) * q["tc"])), 0.0)
        z = z + dz
        if hcs > 0:
            um = u3 + hcs / q["t30"] + q["ccs"] / 2
            z = z + np.clip(hcs - np.maximum(0, np.abs(v - um) - q["ccs"] / 2) * q["t30"], 0, hcs)
    return z


def reach(q, lines=1, spacing=30.0):
    """The greatest distance of the profile from the axis, m (inward, outward)."""
    out = q["c"] / 2 + q["h"] / q["to"] + q["b"] + (q["d"] / q["to"] + q["wd"] + q["d"] / q["tc"] if q["d"] > 0 else 0) + (2 * q["hcs"] / q["t30"] + q["ccs"] + 1 if q["hcs"] else 0)
    inn = q["c"] / 2 + q["h"] / q["ti"] + (out if q["inner"] else 0)
    return inn, out + (lines - 1) * spacing


def _smooth_closed(P, win):
    n = len(P); k = max(1, int(win)) | 1
    if n < k + 2: return P
    Q = np.concatenate([P[-(k // 2):], P, P[:k // 2]]); ker = np.ones(k) / k
    return np.stack([np.convolve(Q[:, 0], ker, "valid"), np.convolve(Q[:, 1], ker, "valid")], 1)


def contour_line(ctx, c, area, rmax=None):
    """The contour line around the summit near c that encloses ≈ area m² (a contour hillfort): level search by bisection, the connected
    region containing the summit, the outline, smoothing (the rampart follows a smooth curve, not every bend of the contour)."""
    import cv2
    rmax = rmax or max(80.0, 2.6 * math.sqrt(area / math.pi) + 60)
    w = ctx.g.window((c[0] - rmax, c[1] - rmax, c[0] + rmax, c[1] + rmax))
    if w is None: return None
    sl, X, Y = w; res = ctx.g.res; Hs = blur(ctx.g.H[sl].astype(np.float64), 3.0 / res)
    d2 = (X - c[0]) ** 2 + (Y - c[1]) ** 2; near = d2 < 20.0 ** 2
    it = np.unravel_index(np.argmax(np.where(near, Hs, -1e9)), Hs.shape); z_top = float(Hs[it]); lo, hi = z_top - 120.0, z_top; best = None
    for _ in range(28):
        zc = 0.5 * (lo + hi); M = (Hs >= zc).astype(np.uint8); n, lab = cv2.connectedComponents(M); comp = lab == lab[it]
        a = comp.sum() * res * res; edge = comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any()
        if a > area or edge: lo = zc
        else: hi = zc; best = (comp, zc)
    if best is None: return None
    cs, _ = cv2.findContours(best[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    C = max(cs, key=len)[:, 0, :].astype(float); P = np.stack([X[0, 0] + C[:, 0] * res, Y[0, 0] + C[:, 1] * res], 1)
    P = _smooth_closed(P, 22.0 / res); P = _smooth_closed(P, 14.0 / res)
    Q = T.resample(np.concatenate([P, P[:1]]), 3.0)[0][:-1]
    return Q, best[1]


def neck_line(ctx, c, face, rng=(15.0, 220.0), drop=4.0, wmax=260.0, wmin=14.0, area=None):
    """The cut-off line of a promontory fort: across the promontory behind the platform near c. The promontory width — where the ground
    is not lower than the level at c by drop m. area (m²) — a yard of the required area: back from the tip of the promontory until
    the area is accumulated, then the narrowest spot within 40 m (a shorter rampart); without area — the narrowest spot beyond c.
    Returns (line ends, level)."""
    a = T.ang(face); d = np.array([math.cos(a), math.sin(a)]); n = np.array([-d[1], d[0]]); c = np.asarray(c, float)
    zc = float(ctx.g.sample(np.array([c[0]]), np.array([c[1]]))[0]); s = np.arange(-wmax, wmax + 0.1, 1.0)

    def width(t):
        p = c + d * t; Q = p + np.outer(s, n); z = ctx.g.sample(Q[:, 0], Q[:, 1]); ok = z >= zc - drop; i0 = len(s) // 2
        if not ok[i0]: return None
        a0 = i0
        while a0 > 0 and ok[a0 - 1]: a0 -= 1
        b0 = i0
        while b0 < len(s) - 1 and ok[b0 + 1]: b0 += 1
        if a0 == 0 or b0 == len(s) - 1: return None                  # not closed — farther on is a plateau, not a promontory
        return s[b0] - s[a0], Q[a0], Q[b0]
    best = None
    if area:
        tt = 0.0
        while tt < 600 and width(tt + 3.0) is not None: tt += 3.0    # the tip of the promontory, forward from c
        acc, t = 0.0, tt
        while t > -rng[1]:
            w = width(t)
            if w is None: break
            acc += w[0] * 3.0
            if acc >= area:
                for t2 in np.arange(t, t - 40.0, -3.0):
                    w2 = width(t2)
                    if w2 is not None and w2[0] >= wmin and (best is None or w2[0] < best[0] - 1.0): best = (w2[0], w2[1], w2[2], t2)
                break
            t -= 3.0
    if best is None:
        for t in np.arange(-rng[0], -rng[1], -3.0):
            w = width(t)
            if w is not None and w[0] >= wmin and (best is None or w[0] < best[0] - 2.0): best = (w[0], w[1], w[2], t)
    if best is None: return None
    _, A, B, t = best; e = (B - A) / max(np.linalg.norm(B - A), 1e-9)
    return np.array([A - e * 8, B + e * 8]), zc


def _orient_out(P):
    """Outward normal at the points of a closed contour."""
    Q = np.asarray(P, float); t = np.roll(Q, -1, 0) - np.roll(Q, 1, 0); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    area = 0.5 * np.sum(Q[:, 0] * np.roll(Q[:, 1], -1) - np.roll(Q[:, 0], -1) * Q[:, 1])
    nrm = np.stack([t[:, 1], -t[:, 0]], 1) if area > 0 else np.stack([-t[:, 1], t[:, 0]], 1)
    return nrm


def _gates(f, P, closed, ctx):
    """Gates: given (path fractions | directions | points) or by approach — where the outside is gentlest (the approach is easy, so that
    is what the gate blocks); the second ones — on the opposite side among the gentle ones."""
    G = f.get("gates", 1); Q = np.asarray(P, float)
    if not closed: N = np.tile(np.array([[0.0, 0.0]]), (len(Q), 1))
    else: N = _orient_out(Q)
    out = []
    if isinstance(G, list):
        for g in G:
            if isinstance(g, (int, float)) and not isinstance(g, bool) and 0 <= g <= 1:
                j = int(round(g * (len(Q) - 1)))
            elif isinstance(g, str) or isinstance(g, (int, float)):
                cc = Q.mean(0); a = T.ang(g); u = np.array([math.cos(a), math.sin(a)]); j = int(np.argmax((Q - cc) @ u))
            else: j = int(np.argmin(np.linalg.norm(Q - np.asarray(g, float)[:2], axis=1)))
            out.append(j)
    elif G and closed:
        z0 = ctx.g.sample(Q[:, 0], Q[:, 1]); z1 = ctx.g.sample(Q[:, 0] + N[:, 0] * 30, Q[:, 1] + N[:, 1] * 30); drop = z0 - z1
        if f.get("gate_to") is not None:
            tq = np.asarray(f["gate_to"], float)[:2]; dist = np.linalg.norm(Q - tq, axis=1); drop = drop + 0.05 * (dist - dist.min())
        j1 = int(np.argmin(drop)); out.append(j1)
        if int(G) > 1:
            L = T.cumlen(np.concatenate([Q, Q[:1]]))[:-1]; per = L[-1] + np.linalg.norm(Q[-1] - Q[0])
            far = np.minimum(np.abs(L - L[j1]), per - np.abs(L - L[j1])) > 0.3 * per; ok = far & (drop <= np.median(drop))
            if ok.any(): out.append(int(np.argmin(np.where(ok, drop, 1e9))))
    return [(Q[j], N[j]) for j in out]


def _seg_dist(X, Y, a, b):
    ab = b - a; t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / max(ab @ ab, 1e-9), 0, 1)
    return np.hypot(X - a[0] - t * ab[0], Y - a[1] - t * ab[1])


def works_geom(f, ctx):
    """Rampart line: path (open) | poly | ring {at, r | rx, ry, rot} | contour {at, area_ha} | neck {at, face}. Cached — for the belt and the anchors."""
    key = f["id"] + ":line"
    if key in ctx.cache: return ctx.cache[key]
    closed, P, out_pt = True, None, None
    if "contour" in f:
        cc = f["contour"]; r = contour_line(ctx, cc["at"], float(cc.get("area_ha", 2.0)) * 1e4)
        if r is None: return None
        P = r[0]
    elif "neck" in f:
        nk = f["neck"]; r = neck_line(ctx, nk["at"], nk.get("face", "N"), tuple(nk.get("range", (15, 220))), float(nk.get("drop", 4.0)),
                                       area=float(nk["area_ha"]) * 1e4 if nk.get("area_ha") else None)
        if r is None: return None
        P, closed = T.resample(r[0], 3.0)[0], False; a = T.ang(nk.get("face", "N")); out_pt = P.mean(0) - 30 * np.array([math.cos(a), math.sin(a)])   # outward — toward the field
    elif "ring" in f:
        rg = f["ring"]; c = np.asarray(rg["at"], float); rx, ry = float(rg.get("rx", rg.get("r", 50))), float(rg.get("ry", rg.get("r", 50)))
        rot = math.radians(rg.get("rot", 0)); t = np.linspace(0, 2 * math.pi, max(24, int(2 * math.pi * max(rx, ry) / 3)), endpoint=False)
        E = np.stack([rx * np.cos(t), ry * np.sin(t)], 1); R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]]); P = c + E @ R.T
        if rg.get("jag", 0.08):                                     # a circle on the terrain — not drawn with a compass
            P = P + _orient_out(P) * (rg.get("jag", 0.08) * min(rx, ry) * T.fbm(P[:, 0] / 60.0, P[:, 1] / 60.0, 2, zlib.crc32(f["id"].encode()) & 0xFFF))[:, None]
    elif "poly" in f:
        P = T.resample(np.concatenate([np.asarray(f["poly"], float), np.asarray(f["poly"][:1], float)]), 3.0)[0][:-1]
    else:
        P, closed = T.resample(T.chaikin(np.asarray(f["path"], float), 2), 3.0)[0], bool(f.get("closed", False))
        if f.get("out") is not None: out_pt = np.asarray(f["out"], float) if not isinstance(f["out"], (str, int, float)) else P.mean(0) + 30 * np.array([math.cos(T.ang(f["out"])), math.sin(T.ang(f["out"]))])
    G = _gates(f, P, closed, ctx)
    ctx.cache[key] = (P, closed, out_pt, G); f["_line"] = P.tolist(); f["_closed"] = closed
    return ctx.cache[key]


def apply_works(f, ctx, seed):
    p = _merge(f, "works"); q = section_params(p); n = int(p.get("lines", 1)); sp = float(p.get("spacing", 30)); shr = float(p.get("shrink", 0.85))
    sg = sigma(p); G_ = works_geom(f, ctx)
    if G_ is None: return
    P, closed, out_pt, G = G_; inn, out = reach(q, n, sp); m = out + 3 * sg + 4
    w_ = ctx.g.window(T._bbox_pts(P, m + inn))
    if w_ is None: return
    sl, X, Y = w_; H = ctx.g.H[sl]
    if closed: u = T.sdf_poly(X, Y, P)
    else:
        d, s, side, ext, L = T.line_info(X, Y, P); sgn = 1.0
        if out_pt is not None:
            dd, ss, sd_, ee, _ = T.line_info(np.array([out_pt[0]]), np.array([out_pt[1]]), P); sgn = float(sd_[0])
        u = d * side * sgn
    D = np.zeros_like(H)
    for k in range(n): D = D + section(u - k * sp, q, shr ** k)
    if not closed: D = D * (1 - T.smoothstep(0, 4 + q["h"], ext))   # the ends of the cut-off rampart fade out (the rampart abuts the edge of the promontory)
    gw = float(p.get("gate_w", 4.0))
    for g, nrm in G:                                                # a gate is a passage through all the lines: the rampart and moat are interrupted (a causeway across the moat)
        nv = nrm if np.linalg.norm(nrm) > 0 else np.zeros(2)
        a, b = g - nv * (inn + 6), g + nv * (out + 10)
        dist = _seg_dist(X, Y, a, b) if np.linalg.norm(nv) > 0 else np.hypot(X - g[0], Y - g[1])
        D = D * T.smoothstep(gw / 2, gw / 2 + 2 + q["h"], dist)
    if sg > 0: D = blur(D, sg / ctx.g.res)
    ctx.g.H[sl] = H + D
    if p.get("water") is not None and ctx.near:                     # water in the moat: the level is a number or "auto" (the lowest point of the moat + 0.6 m)
        ditch = D < -0.3
        if ditch.any():
            lv = float(p["water"]) if p["water"] != "auto" else float(np.min((H + D)[ditch]) + 0.6)
            ctx.put("level", sl, np.where(ditch & (H + D < lv), lv, np.nan), np.fmax)


# ---------------------------------------------------------------- settlement mounds, spoil heaps, gullies
def _gullies(D, X, Y, c, r_a, r_b, n, g0, rs, fan=0.45):
    """Gullies down a mound slope: n rays, the depth grows toward the foot (Forti 2023), an alluvial fan at the foot."""
    if n <= 0 or g0 <= 0: return D
    rng = np.random.default_rng(rs); dx, dy = X - c[0], Y - c[1]; r = np.hypot(dx, dy); th = np.arctan2(dy, dx)
    for a in rng.uniform(0, 2 * math.pi, int(n)):
        dth = np.abs((th - a + np.pi) % (2 * np.pi) - np.pi); lat = r * np.sin(np.minimum(dth, np.pi / 2)) * (dth < np.pi / 2) + 1e9 * (dth >= np.pi / 2)
        g = g0 * rng.uniform(0.6, 1.3); w = 1.5 + 0.1 * np.maximum(r - r_a, 0); Lf = 0.18 * (r_b - r_a) + 6
        dep = g * T.smoothstep(r_a, r_b, r) * (1 - T.smoothstep(r_b, r_b + Lf * 0.6, r)) * np.exp(-(lat / w) ** 2)
        lobe = fan * g * np.exp(-((r - r_b - Lf * 0.5) / (Lf * 0.5)) ** 2) * np.exp(-(lat / (2.2 * w)) ** 2)
        D = D - dep + lobe
    return D


def apply_mound(f, ctx, seed):
    """Settlement mound: a flat top, the facet by soil, aging by diffusion (the top rounds off, the foot spreads), gullies, a steep
    side; tiers — a platform in steps."""
    p = {**know()["mounds"].get(f.get("type", "tell"), {}), **f}; c = np.asarray(p["at"], float); h = float(p.get("h", 15))
    top = p.get("top", 50); rx, ry = (float(top[0]), float(top[1])) if isinstance(top, list) else (float(top), float(top))
    rx, ry = float(p.get("rx", rx)), float(p.get("ry", ry)); rot = math.radians(p.get("rot", 0)); tf = math.tan(math.radians(float(p.get("face", 30))))
    sg = sigma(p); R = max(rx, ry) + h / tf + 4 * sg + 30
    w_ = ctx.g.window((c[0] - R, c[1] - R, c[0] + R, c[1] + R))
    if w_ is None: return
    sl, X, Y = w_; H = ctx.g.H[sl]; dx, dy = X - c[0], Y - c[1]
    lx, ly = dx * math.cos(rot) + dy * math.sin(rot), -dx * math.sin(rot) + dy * math.cos(rot)
    if p.get("shape") == "square": rho = np.maximum(np.abs(lx) - rx, np.abs(ly) - ry)
    else: rho = (np.hypot(lx / rx, ly / ry) - 1) * min(rx, ry)        # distance from the edge of the top, m (for an ellipse — along the minor axis)
    if "steep" in p:                                                 # steep side: the foot is closer
        a, fac = T.ang(p["steep"][0]), float(p["steep"][1]); th = np.arctan2(dy, dx); tfx = tf * (1 + (fac - 1) * np.maximum(0, np.cos(th - a)))
    else: tfx = tf
    D = np.clip(h - np.maximum(rho, 0) * tfx, 0, h)
    for t_h, t_k in p.get("tiers") or []:                            # platform tiers: the upper ones are smaller
        D = D + np.clip(float(t_h) - np.maximum((np.maximum(np.abs(lx), np.abs(ly)) if p.get("shape") == "square" else np.hypot(lx, ly)) - float(t_k) * min(rx, ry), 0) * tf * 1.2, 0, float(t_h))
    if sg > 0: D = blur(D, sg / ctx.g.res)
    gl = p.get("gullies") or [0, 0]
    if ctx.near and gl[0]: D = _gullies(D, X, Y, c, min(rx, ry) * 0.9, min(rx, ry) + h / tf * 1.15, gl[0], float(gl[1]), seed + (zlib.crc32(f["id"].encode()) & 0xFFFF))
    ctx.g.H[sl] = H + D
    if ctx.near: ctx.cache[f["id"] + ":mound"] = {"top": c.tolist(), "z": float(ctx.g.sample(c[:1], c[1:])[0]), "r_top": min(rx, ry), "r_base": min(rx, ry) + h / tf + 2 * sg}


def apply_fill(f, ctx, seed):
    """Cultural layer and fill: inside the contour the ground is raised by t and smoothed — pits and small gullies are filled up to a
    common surface (cities grow upward: Novgorod has 6–8 m of layer); the edge is a slope."""
    Pp = T.footprint(f); t = float(f.get("t", 2.0)); edge = float(f.get("edge", t / math.tan(math.radians(14)) + 6)); sm = float(f.get("smooth", 18.0))
    w_ = ctx.g.window(T._bbox_pts(Pp, edge + 3 * sm))
    if w_ is None: return
    sl, X, Y = w_; H = ctx.g.H[sl]; sd = T.sdf_poly(X, Y, Pp)
    S = blur(H.astype(np.float64), sm / ctx.g.res); Tt = (S + t) if f.get("flat") else np.maximum(H, S) + t   # flat — on the smoothed surface: it cuts as well
    wgt = 1 - T.smoothstep(0, edge, sd)
    ctx.g.H[sl] = H + (Tt - H) * wgt


def apply_lynchets(f, ctx, seed):
    """Field terraces along a slope: ledges along contours, the shelf runs almost along the slope (the fraction tread), the ledge face
    riser°; the shelf width follows from the slope itself: narrower on steep ground. The field is a contour poly | at+r."""
    p = {**know()["lynchets"], **f}; Pp = T.footprint(p); hr = float(p["step"]); ta = math.tan(math.radians(float(p["riser"]))); tau = float(p["tread"])
    w_ = ctx.g.window(T._bbox_pts(Pp, 6))
    if w_ is None: return
    sl, X, Y = w_; H = ctx.g.H[sl]; res = ctx.g.res; m = 1 - T.smoothstep(-4, 4, T.sdf_poly(X, Y, Pp))
    S = blur(H.astype(np.float64), 8.0 / res); gy, gx = np.gradient(S, res); tb = np.hypot(gx, gy)
    rs = seed + (zlib.crc32(f["id"].encode()) & 0xFFFF); zw = S + 0.3 * hr * T.fbm(X / 45.0, Y / 45.0, 3, rs)
    qz = zw / hr; fl = np.floor(qz); tt = qz - fl
    tr = np.clip((ta - tb) / np.maximum(ta - tau * tb, 1e-6), 0.05, 0.95)
    zt = np.where(tt < tr, fl + tau * tt, fl + tau * tr + (tt - tr) / np.maximum(1 - tr, 1e-6) * (1 - tau * tr)) * hr
    ok = T.smoothstep(math.radians(2.5), math.radians(5), np.arctan(tb)) * (1 - T.smoothstep(math.atan(ta) - 0.12, math.atan(ta) - 0.03, np.arctan(tb)))
    Dl = (zt - zw) * m * ok; sg = sigma(p)
    if sg > 0: Dl = blur(Dl, sg / res)
    ctx.g.H[sl] = H + Dl


def apply_furrows(f, ctx, seed):
    """Ridge-and-furrow arable land: ridges along the slope (drainage), spacing step, amplitude amp, a reverse S in plan, headlands at the ends."""
    p = {**know()["furrows"], **f}; Pp = T.footprint(p)
    w_ = ctx.g.window(T._bbox_pts(Pp, 2))
    if w_ is None or not ctx.near: return
    sl, X, Y = w_; H = ctx.g.H[sl]; res = ctx.g.res; sd = T.sdf_poly(X, Y, Pp); m = 1 - T.smoothstep(-2, 1, sd)
    if p.get("dir") is not None: th = T.ang(p["dir"])
    else:
        S = blur(H.astype(np.float64), 10.0 / res); gy, gx = np.gradient(S, res); ins = sd < 0
        th = math.atan2(float(gy[ins].mean()), float(gx[ins].mean())) if ins.any() else 0.0
    cx, cy = Pp.mean(0); v = (X - cx) * math.cos(th) + (Y - cy) * math.sin(th); w = -(X - cx) * math.sin(th) + (Y - cy) * math.cos(th)
    vi = v[sd < 0]; v0, v1 = (float(vi.min()), float(vi.max())) if vi.size else (-50.0, 50.0); Lf = max(v1 - v0, 10.0)
    w2 = w + float(p["s_curve"]) * np.sin(2 * math.pi * (v - v0) / Lf)          # reverse S: the plough team turns in toward the end of the strip
    hd = 7.0; inner = T.smoothstep(v0, v0 + hd, v) * (1 - T.smoothstep(v1 - hd, v1, v))
    z = 0.5 * float(p["amp"]) * np.cos(2 * math.pi * w2 / float(p["step"])) * inner
    z = z + float(p["headland"]) * (np.exp(-((v - v0 - hd * 0.5) / 2.5) ** 2) + np.exp(-((v - v1 + hd * 0.5) / 2.5) ** 2))
    ctx.g.H[sl] = H + z * m


def apply_pit(f, ctx, seed):
    """Quarry in benches: horizontal benches from the brow level (the lowest ground point along the contour) down to the floor; face
    angle face°, shelf berm; water on the floor (water — water depth). The excavation volume goes into the cache (a nearby spoil
    heap takes it)."""
    p = {**know()["pits"].get(f.get("type", "quarry"), {}), **f}; Pp = T.footprint(p)
    hb, bw, tf, D_ = float(p["bench"]), float(p["berm"]), math.tan(math.radians(float(p["face"]))), float(p["depth"])
    w_ = ctx.g.window(T._bbox_pts(Pp, 40))
    if w_ is None: return
    sl, X, Y = w_; H = ctx.g.H[sl]; u = -T.sdf_poly(X, Y, Pp); key = f["id"] + ":rim"
    if key not in ctx.cache:
        Q = T.resample(np.concatenate([Pp, Pp[:1]]), 2.0)[0]; ctx.cache[key] = float(ctx.g.sample(Q[:, 0], Q[:, 1]).min()) - 0.3
    zt = ctx.cache[key]; per = hb / tf + bw; k = np.floor(np.maximum(u, 0) / per); r = np.maximum(u, 0) - k * per
    fall = np.minimum(k * hb + np.clip(r * tf / hb, 0, 1) * hb, D_)
    Z = np.where(u >= 0, zt - fall, zt + (-u) * tf); H2 = np.minimum(H, Z)
    if ctx.near:
        ctx.cache[f["id"] + ":V"] = float((H - H2).sum() * ctx.g.res ** 2); ctx.cache[f["id"] + ":floor"] = zt - D_
        if p.get("water"):
            lv = zt - D_ + float(p["water"]); ctx.put("level", sl, np.where((u > 0) & (H2 < lv), lv, np.nan), np.fmax)
    ctx.g.H[sl] = H2


def apply_heap(f, ctx, seed):
    """Spoil heap: a cone (spoil tip) | flat (platform top) | ridge (along a path) — the face at the soil's angle of repose; the height
    is h or comes from the volume of the quarry of (× bulking); aging and gullies — as for a mound."""
    p = {**know()["heaps"].get(f.get("type", "flat_top"), {}), **f}; S_ = soil(p); ta = math.tan(math.radians(float(p.get("face", S_["repose"]))))
    shape = p.get("shape", "flat"); h = float(p.get("h", 15)); top = float(p.get("top", 30))
    if p.get("of") and p["of"] + ":V" in ctx.cache:                  # volume from the excavation: V·bulking
        V = ctx.cache[p["of"] + ":V"] * float(S_["bulk"])
        if shape == "cone": h = (3 * V * ta * ta / math.pi) ** (1 / 3)
        else:
            lo, hi = 0.5, 200.0
            for _ in range(40):                                     # flat: V(h) = π·h·(R² + R·r + r²)/3, r = top, R = top + h/tg
                hm = 0.5 * (lo + hi); Rb = top + hm / ta; Vm = math.pi * hm * (Rb * Rb + Rb * top + top * top) / 3
                lo, hi = (hm, hi) if Vm < V else (lo, hm)
            h = 0.5 * (lo + hi)
        ctx.cache[f["id"] + ":h"] = h
    elif f["id"] + ":h" in ctx.cache: h = ctx.cache[f["id"] + ":h"]
    sg = sigma(p)
    if shape == "ridge":
        P = T.chaikin(np.asarray(p["path"], float), 2); m = top / 2 + h / ta + 3 * sg + 20; w_ = ctx.g.window(T._bbox_pts(P, m))
        if w_ is None: return
        sl, X, Y = w_; d, s, side, ext, L = T.line_info(X, Y, P); rho = np.maximum(d - top / 2, 0) + np.maximum(ext, 0); c = P.mean(0)
    else:
        c = np.asarray(p["at"], float); R = (0 if shape == "cone" else top) + h / ta + 3 * sg + 25
        w_ = ctx.g.window((c[0] - R, c[1] - R, c[0] + R, c[1] + R))
        if w_ is None: return
        sl, X, Y = w_; rho = np.maximum(np.hypot(X - c[0], Y - c[1]) - (0 if shape == "cone" else top), 0)
    H = ctx.g.H[sl]; D = np.clip(h - rho * ta, 0, h)
    if p.get("benches"):                                            # in benches (a tailings dam): a shelf every bench m
        bh, bw = p["benches"]; k = np.floor((h - D) / bh); D = np.clip(h - (rho - k * bw).clip(0) * ta, 0, h) * (D > 0)
    if sg > 0: D = blur(D, sg / ctx.g.res)
    gl = p.get("gullies") or [0, 0]
    if ctx.near and gl[0] and shape != "ridge": D = _gullies(D, X, Y, c, (0 if shape == "cone" else top) + 2, (0 if shape == "cone" else top) + h / ta, gl[0], float(gl[1]), seed + (zlib.crc32(f["id"].encode()) & 0xFFFF))
    ctx.g.H[sl] = H + D


def apply_dike(f, ctx, seed):
    """Dam, mole, embankment: the crest at a level (crest_z) or at h above the ground along the path (smoothed), slopes out° (toward
    the water, side sea: L|R) and in°; raises the ground up to the profile (makes no cuts)."""
    p = {**know()["dike"], **f}; P = T.chaikin(np.asarray(p["path"], float), 2); h, c = float(p["h"]), float(p["crest"])
    to, ti = math.tan(math.radians(float(p["out"]))), math.tan(math.radians(float(p["in"]))); m = c / 2 + h / min(to, ti) + 6
    w_ = ctx.g.window(T._bbox_pts(P, m))
    if w_ is None: return
    sl, X, Y = w_; H = ctx.g.H[sl]; d, s, side, ext, L = T.line_info(X, Y, P); key = f["id"] + ":z"
    if key not in ctx.cache:
        Q, sq, zq = T._profile_along(ctx, f, P, 2.0, 30.0); ctx.cache[key] = (sq, np.full_like(zq, float(p["crest_z"])) if p.get("crest_z") is not None else zq + h)
    sq, zc = ctx.cache[key]; zt = np.interp(s, sq, zc)
    sgn = 1 if p.get("sea", "L") == "L" else -1; tt = np.where(side * sgn > 0, to, ti)
    Tz = zt - np.maximum(0, d - c / 2) * tt; wgt = 1 - T.smoothstep(0, 3 + h, ext)
    ctx.g.H[sl] = np.where(Tz > H, H + (Tz - H) * wgt, H)


def apply_dunes(f, ctx, seed):
    """Dunes (ash of a hive wasteland, sand): ridges across the wind, the windward slope stoss°, the leeward — the angle of repose
    lee°; the ridge phase drifts with noise (crests not ruler-straight), the height varies along the crest — ridges break into arcs
    (barchan chains), flat in patches."""
    p = {**know()["dunes"], **f}; lam, a = float(p["step"]), float(p["h"]); ts, tl = math.tan(math.radians(float(p["stoss"]))), math.tan(math.radians(float(p["lee"])))
    if "poly" in p or "at" in p:
        Pp = T.footprint(p); w_ = ctx.g.window(T._bbox_pts(Pp, 10))
    else: Pp = None; w_ = ctx.g.window((-1e9, -1e9, 1e9, 1e9))
    if w_ is None: return
    sl, X, Y = w_; H = ctx.g.H[sl]; rs = seed + (zlib.crc32(f["id"].encode()) & 0xFFFF)
    m = (1 - T.smoothstep(-25, 0, T.sdf_poly(X, Y, Pp))) if Pp is not None else ctx.fades(X, Y)[1]
    th = T.ang(p.get("wind", "E")); v = X * math.cos(th) + Y * math.sin(th); w = -X * math.sin(th) + Y * math.cos(th)
    ph = v / lam + float(p["sinuosity"]) * T.fbm(w / (1.6 * lam), v / (2.5 * lam), 3, rs) + 0.35 * T.fbm(X / (5 * lam), Y / (5 * lam), 2, rs + 7)
    seg = T.fbm(w / (1.2 * lam) + 3.3, np.floor(ph) * 1.7, 2, rs + 5)       # along the crest: arcs and gaps, each ridge has its own
    A = a * np.clip(0.5 + 0.9 * seg + float(p["patch"]) * T.fbm(X / (3.2 * lam), Y / (3.2 * lam), 3, rs + 3), 0, 1.3)
    Ls, Ll = A / ts, A / tl; sc = np.minimum(1.0, 0.92 * lam / np.maximum(Ls + Ll, 1e-6)); A, Ls, Ll = A * sc, Ls * sc, Ll * sc
    q = np.mod(ph, 1.0) * lam; z = np.where(q < Ls, q * ts, np.maximum(A - (q - Ls) * tl, 0))   # windward rise, leeward drop-off
    z = np.minimum(z, A); D = blur(z * m, 1.2 / ctx.g.res) if ctx.near else z * m
    ctx.g.H[sl] = H + D


def apply_holloway(f, ctx, seed):
    """Sunken road: a road trampled and washed out over centuries — deeper on steep ground (depth × the slope along it), floor w, sides
    side°, banks along the brows (hedgerows)."""
    if not ctx.near: return
    p = {**know()["holloway"], **f}; P = T.chaikin(np.asarray(p["path"], float), 3); d0, wb, tsd = float(p["depth"]), float(p["w"]), math.tan(math.radians(float(p["side"])))
    m = wb / 2 + 2.2 * d0 / tsd + 6; w_ = ctx.g.window(T._bbox_pts(P, m))
    if w_ is None: return
    sl, X, Y = w_; H = ctx.g.H[sl]; d, s, side, ext, L = T.line_info(X, Y, P); key = f["id"] + ":dep"
    if key not in ctx.cache:
        Q, sq, zq = T._profile_along(ctx, f, P, 2.0, 20.0); gr = np.abs(np.gradient(zq, sq)); dep = d0 * np.clip(0.45 + gr / 0.10, 0.45, 2.2)
        k = 9; dep = np.convolve(np.pad(dep, k // 2, mode="edge"), np.ones(k) / k, mode="valid"); ctx.cache[key] = (sq, dep)
    sq, dep = ctx.cache[key]; dz = np.interp(s, sq, dep); top = wb / 2 + dz / tsd
    cut = -dz * (1 - T.smoothstep(wb / 2, top, d)); bank = float(p["bank"]) * np.exp(-((d - top - 1.2) / 1.1) ** 2)
    D = (cut + bank) * (1 - T.smoothstep(0, 6, ext)); sg = sigma(p)
    if sg > 0: D = blur(D, sg / ctx.g.res)
    ctx.g.H[sl] = H + D


APPLY = {"works": apply_works, "mound": apply_mound, "fill": apply_fill, "lynchets": apply_lynchets, "furrows": apply_furrows, "pit": apply_pit,
         "heap": apply_heap, "dike": apply_dike, "dunes": apply_dunes, "holloway": apply_holloway}


def apply(f, ctx, seed):
    """A shape + its trace in ctx.works (where the ground is displaced by more than 0.15 m): by the trace, dressing and cover know what
    is man-made — no forest on a rampart, in a moat, on a mound, in a quarry. Dunes are not human work and leave no trace."""
    keep = ctx.g.H.copy() if ctx.near and f["kind"] != "dunes" else None
    APPLY[f["kind"]](f, ctx, seed)
    if keep is not None:
        if not hasattr(ctx, "works"): ctx.works = np.zeros(ctx.g.H.shape, np.float32)
        np.maximum(ctx.works, (np.abs(ctx.g.H - keep) > 0.15).astype(np.float32), out=ctx.works)


def influence(f):
    """The shape's box (for the belt: where the shape can change anything at all)."""
    k = f["kind"]
    if k == "works":
        if "contour" in f: c = f["contour"]["at"]; r = 2.6 * math.sqrt(float(f["contour"].get("area_ha", 2.0)) * 1e4 / math.pi) + 160; return T._bbox_pts([c], r)
        if "neck" in f: return T._bbox_pts([f["neck"]["at"]], 320)
        if "ring" in f: rg = f["ring"]; return T._bbox_pts([rg["at"]], max(rg.get("rx", rg.get("r", 50)), rg.get("ry", rg.get("r", 50))) + 120)
        return T._bbox_pts(f.get("poly") or f["path"], 120)
    if k in ("mound", "heap") and "at" in f: return T._bbox_pts([f["at"]], 400)
    if "path" in f: return T._bbox_pts(f["path"], 80)
    if "poly" in f or "at" in f: return T._bbox_pts(T.footprint(f), 60)
    return (-1e9, -1e9, 1e9, 1e9)


def path_of(f):
    if f["kind"] == "works" and f.get("_line"):
        P = np.asarray(f["_line"], float); return T.resample(np.concatenate([P, P[:1]]) if f.get("_closed") else P, 2.0)[0]
    if f["kind"] in ("holloway", "dike") or (f["kind"] == "heap" and f.get("shape") == "ridge"): return T.resample(T.chaikin(np.asarray(f["path"], float), 2), 2.0)[0]
    return None


def anchors_of(f, cache):
    k, i, A = f["kind"], f["id"], {}
    if k == "works" and i + ":line" in cache:
        P, closed, out_pt, G = cache[i + ":line"]; c = P.mean(0)
        if closed: A["yard"] = c; nrm = _orient_out(P)
        for j, (g, nv) in enumerate(G):
            nm = "gate" if j == 0 else f"gate{j + 1}"; A[nm] = g; A[nm + "_out"] = g + nv * 25; A[nm + "_in"] = g - nv * 12
        j = int(np.argmax(P[:, 1])) if closed else len(P) // 2; A["crest"] = P[j]; A["start"], A["end"] = P[0], P[-1]
        if not closed: A["mid"] = P[len(P) // 2]
    elif k == "mound":
        c = np.asarray(f["at"], float); A["top"] = c; M = cache.get(i + ":mound") or {}; rb = M.get("r_base", 60.0)
        for nm, a in (("n", 90), ("e", 0), ("s", 270), ("w", 180)): A["foot_" + nm] = c + rb * np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
    elif k in ("pit", "heap", "fill", "lynchets", "furrows"):
        if "at" in f or "poly" in f:
            c = np.asarray(f["at"], float) if "at" in f else np.asarray(f["poly"], float).mean(0); A["center"] = c
            if k == "pit": A["rim"] = c + np.array([float(f.get("r", 30)), 0.0])
    elif k in ("holloway", "dike"):
        P = T.resample(T.chaikin(np.asarray(f["path"], float), 2), 1.0)[0]; A["start"], A["end"], A["mid"] = P[0], P[-1], P[len(P) // 2]
    return {f"{i}.{n}": np.asarray(p, float) for n, p in A.items()}


# ---------------------------------------------------------------- archetypes: settlement sites and hillforts (nature + man-made, anchors)
def _frame(u):
    c = np.asarray(u.get("at", [0, 0]), float); a = T.ang(u.get("face", "N")); d = np.array([math.cos(a), math.sin(a)]); n = np.array([-d[1], d[0]])
    return c, d, n, (lambda x, y: (c + d * x + n * y).round(1).tolist()), math.degrees(a)


def _crescent(pt, x, y, r, hw, a0, a1, n=28):
    """Oxbow crescent: an arc of radius r, half-width hw, from angle a0 to a1 (°), in the archetype's axes."""
    ts = np.radians(np.linspace(a0, a1, n)); taper = np.sin(np.linspace(0.15, math.pi - 0.15, n))
    outer = [pt(x + (r + hw * k) * math.cos(t_), y + (r + hw * k) * math.sin(t_)) for t_, k in zip(ts, taper)]
    inner = [pt(x + (r - hw * k) * math.cos(t_), y + (r - hw * k) * math.sin(t_)) for t_, k in zip(ts[::-1], taper[::-1])]
    return outer + inner


def _big(pt, half=900.0):
    return [pt(-half, -half), pt(half, -half), pt(half, half), pt(-half, half)]


def a_spur(u):
    """A promontory between ravines above a river (the site of a promontory fort): a plateau, the river valley in front of the
    promontory, two ravines on the sides of the promontory — converging at the neck. x points to the river (face), the promontory
    runs from the neck (x = 0) to the brow (x = L)."""
    c, d, n, pt, fd = _frame(u); i = u["id"]; H, L, W = u.get("h", 24.0), u.get("spur_l", 150.0), u.get("spur_w", 90.0)
    fp, wr = u.get("fp", 70.0), u.get("w", 18.0); tw = math.tan(math.radians(28)); fw = fp / 2 + wr / 2; xv = L + H / tw + fw
    F = [{"kind": "plateau", "id": f"{i}_pl", "poly": _big(pt), "h": H, "edge": 40, "jag": 0.1},
         {"kind": "valley", "id": f"{i}_vl", "path": [pt(xv, -900), pt(xv + 15, 0), pt(xv, 900)], "floor": [0.8, 0.0], "fw": fw, "walls": 28, "w": fw + H / tw + 20},
         {"kind": "river", "id": f"{i}_river", "path": [pt(xv - fw + wr / 2 + 6, -900), pt(xv - fw + wr / 2 + 10, 0), pt(xv - fw + wr / 2 + 6, 900)], "w": wr, "depth": 1.6,
          "wig": [min(fp * 0.3, 2.0 * wr), 12.0 * wr]}]
    tr = math.tan(math.radians(34)); rh = u.get("ravine_head", -40.0)
    for nm, sg in (("l", 1), ("r", -1)):                             # ravines shaped like a drop: they converge at the heads (the neck), diverge toward the middle of the promontory, and converge again toward the river
        F.append({"kind": "valley", "id": f"{i}_rav_{nm}", "path": [pt(rh, sg * W * 0.22), pt(L * 0.15, sg * (W / 2 + 2)), pt(L * 0.55, sg * (W / 2 + 14)),
                  pt(L + H / tw * 0.5, sg * (W / 2 + 8)), pt(xv - fw + 4, sg * (W / 2 + 40))], "floor": [H - 3.0, H * 0.75, H * 0.45, 3.0, 0.6], "fw": 3.0, "walls": 34,
                  "w": 3 + H / tr + 12, "wig": [6.0, 70.0]})
    A = {"tip": pt(L * 0.85, 0), "yard": pt(L * 0.5, 0), "field": pt(-80, 0), "below": pt(xv - fw + 3, 0), "river": f"{i}_river.mid",
         "ravine_l": pt(L + 25, W / 2 + 45), "ravine_r": pt(L + 25, -(W / 2 + 45)), "far_bank": pt(xv + fw * 0.6, 0)}
    return F, A


def a_hillfort(u):
    """Hillfort: contour (a hill, a rampart along a contour around a top of the required area, gates by approach) | promontory (a
    promontory between ravines, a cut-off rampart across the neck) | ring (on flat ground)."""
    c, d, n, pt, fd = _frame(u); i = u["id"]; kind = u.get("kind", "contour"); F, A = [], {}
    wk = {k: u[k] for k in ("soil", "age", "lines", "spacing", "bank", "ditch", "water") if k in u}
    if kind == "contour":
        h, rx, ry = u.get("h", 38.0), u.get("rx", 220.0), u.get("ry", 165.0)
        F.append({"kind": "hill", "id": f"{i}_hill", "at": c.tolist(), "rx": rx, "ry": ry, "r": max(rx, ry), "rot": fd + 90, "h": h, "shape": "dome", "jag": 0.1})
        F.append({"kind": "works", "id": f"{i}_works", "type": u.get("works", "multivallate"), "contour": {"at": c.tolist(), "area_ha": u.get("area_ha", 4.0)},
                  "gates": u.get("gates", 2), **({"gate_to": u["gate_to"]} if "gate_to" in u else {}), **wk})
        A.update({"top": f"{i}_hill.top", "foot": pt(ry * 0.98, 0), "foot_back": pt(-ry * 0.98, 0)})
    elif kind == "spur":
        F, A = a_spur(u); L = u.get("spur_l", 150.0)
        F.append({"kind": "works", "id": f"{i}_works", "type": u.get("works", "cross_bank"), "neck": {"at": pt(L * 0.7, 0), "face": fd, "range": [8, L * 1.2], "drop": 4.0,
                  "area_ha": u.get("area_ha", 0.9)},
                  "gates": [u.get("gate_at", 0.35)], **wk})
    else:
        F.append({"kind": "works", "id": f"{i}_works", "type": u.get("works", "ring"), "ring": {"at": c.tolist(), "r": u.get("r", 55.0), "jag": 0.06},
                  "gates": u.get("gates", [fd]), **wk})
    A.update({"yard": f"{i}_works.yard" if kind != "spur" else A.get("yard"), "gate": f"{i}_works.gate", "gate_out": f"{i}_works.gate_out",
              "crest": f"{i}_works.crest", **({"gate2": f"{i}_works.gate2"} if kind == "contour" and u.get("gates", 2) != 1 else {}),
              **({"neck": f"{i}_works.mid"} if kind == "spur" else {})})
    return F, A


def a_confluence(u):
    """Confluence spit — a promontory at the junction of a river and a tributary (Nizhny Novgorod, Yaroslavl, Pskov, Moscow): a high
    bank is a plateau, in front of the promontory is the river valley (its own bank is steep — by Baer's law, the far one is a gentle
    floodplain), on the left the tributary valley, on the right a ravine; the kremlin on the promontory, the posad (trade suburb) on
    the plateau behind the neck, the podol (lower town) below at the tributary mouth."""
    c, d, n, pt, fd = _frame(u); i = u["id"]; H, L, W = u.get("h", 26.0), u.get("spur_l", 170.0), u.get("spur_w", 120.0)
    wm, wt = u.get("w", 40.0), u.get("w_trib", 16.0); fp = u.get("fp", 140.0); tb = math.tan(math.radians(30)); fw = wm / 2 + 25
    xv = L + H / tb + fw + fp                                       # the brow of its own bank is at the tip of the promontory; the valley floor — the river at the bank and the floodplain beyond it
    F = [{"kind": "plateau", "id": f"{i}_pl", "poly": _big(pt), "h": H, "edge": 40, "jag": 0.1},
         {"kind": "valley", "id": f"{i}_vm", "path": [pt(xv, -900), pt(xv + 10, 0), pt(xv, 900)], "floor": [0.5, 0.0], "fw": fw + fp, "walls": [30, 30], "w": fw + fp + H / tb + 30},
         {"kind": "river", "id": f"{i}_river", "path": [pt(xv - fw - fp + wm / 2 + 5, -900), pt(xv - fw - fp + wm / 2 + 8, 0), pt(xv - fw - fp + wm / 2 + 5, 900)], "w": wm, "depth": 2.2,
          "wig": [6.0, 14.0 * wm]}]
    yt = W / 2 + 30 + wt; ttw = math.tan(math.radians(26)); ftw = wt / 2 + 22; xr = xv - fw - fp + wm / 2 + 5
    F += [{"kind": "valley", "id": f"{i}_vt", "path": [pt(-900, yt + 40), pt(-200, yt + 10), pt(L * 0.6, yt), pt(xr, yt - 10)], "floor": [3.0, 1.2, 0.6, 0.2], "fw": ftw, "walls": 26, "w": ftw + H / ttw + 20},
          {"kind": "river", "id": f"{i}_trib", "path": [pt(-900, yt + 40), pt(-200, yt + 10), pt(L * 0.6, yt), pt(xr, yt - 10)], "w": wt, "depth": 1.3, "wig": [min(ftw * 0.8, 2.5 * wt), 11.0 * wt]}]
    tr = math.tan(math.radians(33))
    for k in range(int(u.get("ravines", 2))):                       # to the right of the promontory — a ravine (the right edge of the promontory), farther — more
        y0 = -(W / 2 + 10 + 150 * k)
        F.append({"kind": "valley", "id": f"{i}_rav{k}", "path": [pt(-40 - 60 * k, y0 + 6), pt(L * 0.3, y0 - 6), pt(L * 0.75, y0 + 4), pt(xv - fw - fp - 2, y0 - 40)],
                  "floor": [H - 4.0, H * 0.7, H * 0.4, 0.6], "fw": 3.0, "walls": 33, "w": 3 + H / tr + 12, "wig": [8.0, 80.0]})
    A = {"tip": pt(L * 0.85, 0), "kremlin": pt(L * 0.55, 0), "neck": pt(0, 0), "posad": pt(-110, 0), "podol": pt(L + 10, yt + wt / 2 + 12),
         "pier": pt(xr - wm / 2 - 4, 0), "far_bank": pt(xr + wm / 2 + fp * 0.5, 0), "trib_ford": f"{i}_trib@0.75", "river": f"{i}_river.mid", "ravine": pt(L * 0.8, -(W / 2 + 10))}
    return F, A


def a_meander(u):
    """Town in a meander (Durham, Bern, Besançon): the river is incised in a loop around the promontory, the arms come close at the neck
    (where the castle is); inside the loop the slope is gentle (the inner bank), outside — a steep undercut."""
    c, d, n, pt, fd = _frame(u); i = u["id"]; R, N, D, w = u.get("r", 150.0), u.get("neck", 55.0), u.get("h", 28.0), u.get("w", 24.0)
    dl = math.asin(min(0.95, (N / 2 + w / 2) / R)); a0 = math.atan2(d[1], d[0]); rng = np.random.default_rng(int(u.get("seed", 3)))
    ph = rng.uniform(0, 6.28); P = [pt(R + 420, N / 2 + w / 2 + 230), pt(R * math.cos(dl) + 70, R * math.sin(dl) + 25)]
    for t in np.linspace(a0 + dl, a0 + 2 * math.pi - dl, 30):
        rr = R * (1 + 0.1 * math.sin(2 * (t - a0) + ph) + 0.06 * math.sin(3 * (t - a0) + 2 * ph) + 0.04 * math.sin(5 * (t - a0) + 3 * ph)); P.append((c + rr * np.array([math.cos(t), math.sin(t)])).round(1).tolist())
    P += [pt(R * math.cos(dl) + 70, -(R * math.sin(dl) + 25)), pt(R + 420, -(N / 2 + w / 2 + 230))]
    fw = w / 2 + 8; ti, to = math.tan(math.radians(22)), math.tan(math.radians(40))
    F = [{"kind": "plateau", "id": f"{i}_pl", "poly": _big(pt), "h": D, "edge": 40, "jag": 0.1},
         {"kind": "valley", "id": f"{i}_v", "path": P, "floor": [1.0, 0.0], "fw": fw, "walls": [22, 40], "w": fw + D / ti + 15, "wiggle": 0},
         {"kind": "river", "id": f"{i}_river", "path": P, "w": w, "depth": 2.0, "wiggle": 0}]
    A = {"town": c.tolist(), "neck": pt(R * 0.92, 0), "tip": pt(-R * 0.5, 0), "bridge_up": pt(R * math.cos(dl) + 30, R * math.sin(dl) + 9),
         "bridge_down": pt(R * math.cos(dl) + 30, -(R * math.sin(dl) + 9)), "cliff": pt(-(R + fw + D / to + 6), 0), "river": f"{i}_river.mid"}
    return F, A


def a_river_terrace(u):
    """Town on an above-floodplain terrace at a ford: the terrace brow (a 5–10 m ledge), a floodplain with a river and an oxbow, a second terrace farther on;
    a road from the terrace to the ford (or bridge)."""
    c, d, n, pt, fd = _frame(u); i = u["id"]; Ht, H2, fp, w = u.get("h", 7.0), u.get("h2", 6.0), u.get("fp", 90.0), u.get("w", 20.0)
    e = Ht / math.tan(math.radians(20)); xr = e + fp
    F = [{"kind": "plateau", "id": f"{i}_t1", "poly": [pt(-900, -900), pt(0, -900), pt(0, 900), pt(-900, 900)], "h": Ht, "edge": e, "jag": 0.25},
         {"kind": "plateau", "id": f"{i}_t2", "poly": [pt(-900, -900), pt(-190, -900), pt(-190, 900), pt(-900, 900)], "h": H2, "edge": H2 / math.tan(math.radians(14)), "jag": 0.3},
         {"kind": "river", "id": f"{i}_river", "path": [pt(xr, -900), pt(xr - 10, -200), pt(xr + 15, 150), pt(xr, 900)], "w": w, "depth": 1.8, "wig": [min(0.45 * fp, 2.5 * w), 12.0 * w]},
         {"kind": "lake", "id": f"{i}_oxbow", "poly": _crescent(pt, xr + 85, 190, 55.0, 8.0, -60, 150), "depth": 1.2, "jag": 0.05},
         {"kind": "road", "id": f"{i}_road", "path": [pt(-70, 12), pt(-8, 8), pt(e + 20, 4), pt(xr, 0), pt(xr + 100, -8)], "w": 4.0, "grade": 0.12, "cross": u.get("cross", "ford")}]
    A = {"town": pt(-50, 0), "edge": pt(-2, 0), "floodplain": pt(e + 30, -40), "ford": pt(xr, 0), "far_bank": pt(xr + 60, 0), "terrace2": pt(-215, 0),
         "river": f"{i}_river.mid", "oxbow": f"{i}_oxbow.center"}
    return F, A


def a_acropolis(u):
    """Acropolis: a rock table above a plain (cliffs on three sides, an ascent on one — face), a satellite hill, a plain toward the sea."""
    c, d, n, pt, fd = _frame(u); i = u["id"]; h, rx, ry, Ls = u.get("h", 55.0), u.get("rx", 150.0), u.get("ry", 85.0), u.get("sea_at", 150.0)
    F = [{"kind": "hill", "id": f"{i}_hill", "at": c.tolist(), "rx": ry, "ry": rx, "r": rx, "rot": fd, "h": h, "shape": "mesa", "jag": 0.12},
         {"kind": "ridge", "id": f"{i}_ramp", "path": [pt(ry * 0.55, 0), pt(ry + 50, rx * 0.15), pt(ry + 110, rx * 0.25)], "h": [h * 0.7, h * 0.3, h * 0.05], "w": 45, "sharp": 0.15, "op": "max"},
         {"kind": "hill", "id": f"{i}_knoll", "at": pt(ry + 75, -rx * 0.75), "r": 32, "h": u.get("knoll_h", 15.0), "shape": "bell", "jag": 0.2}]
    if u.get("sea", True):
        F.append({"kind": "lake", "id": f"{i}_sea", "poly": [pt(-ry - Ls, -900), pt(-ry - Ls - 30, 0), pt(-ry - Ls, 900), pt(-1900, 900), pt(-1900, -900)], "level": -2.0, "depth": 8,
                  "beach": 6, "bank_slope": 8, "bluff": 30, "jag": 0.15})
    A = {"top": c.tolist(), "gate": pt(ry * 0.62, rx * 0.05), "approach": pt(ry + 110, rx * 0.25), "agora": pt(ry + 150, -40), "lower": pt(-ry - Ls * 0.45, 0),
         "harbour": pt(-ry - Ls - 8, 0), "knoll": f"{i}_knoll.top", "open_sea": pt(-ry - Ls - 400, 0) if u.get("sea", True) else None}
    return F, A


def a_bay(u):
    """Bay — a colony harbor: a rounded gulf with a throat, headland hills on both sides of the throat, a beach at the head of the bay, a plain behind it."""
    c, d, n, pt, fd = _frame(u); i = u["id"]; R, M, hh = u.get("r", 110.0), u.get("mouth", 70.0), u.get("cape_h", 22.0)
    Lc = math.sqrt(max(R * R - (M / 2) ** 2, 1.0)); t0 = math.atan2(-M / 2, Lc); ts = np.linspace(t0, t0 - (2 * math.pi - 2 * abs(t0)), 64)
    rs = int(R * 7 + M * 3 + hh) % 997; nz = lambda q, sc: float(T.fbm(np.array([q / sc]), np.array([0.37 + rs]), 3, rs)[0])
    rim = [pt((R + 0.12 * R * nz(k * 9.0, 40.0)) * math.cos(t), (R + 0.12 * R * nz(k * 9.0, 40.0)) * math.sin(t)) for k, t in enumerate(ts)]   # the bay is not drawn with a compass
    coast = lambda y: Lc + 14.0 * nz(y + 500.0, 90.0)
    poly = [pt(coast(y), y) for y in np.linspace(-900, -M / 2 - 10, 24)] + rim + [pt(coast(y), y) for y in np.linspace(M / 2 + 10, 900, 24)] + [pt(Lc + 1500, 900), pt(Lc + 1500, -900)]
    F = [{"kind": "lake", "id": f"{i}_sea", "poly": poly, "level": -1.5, "depth": 8, "beach": 5, "bank_slope": 10, "bluff": 40, "jag": 0.0}]
    for nm, sg in (("l", 1), ("r", -1)):
        F.append({"kind": "hill", "id": f"{i}_cape_{nm}", "at": pt(Lc - 45, sg * (M / 2 + 55)), "r": 62, "h": hh * (1.0 if sg > 0 else 0.8), "shape": "dome", "jag": 0.25,
                  "steep": [fd, 1.8]})
    A = {"harbour": pt(-R * 0.35, 0), "quay": pt(-R - 4, 0), "cape_l": f"{i}_cape_l.top", "cape_r": f"{i}_cape_r.top", "mouth": pt(Lc, 0), "plain": pt(-R - 90, 0),
         "open_sea": pt(Lc + 220, 0)}
    return F, A


def a_gap(u):
    """Town in a ridge gap: two ridges with a passage between them with a river and a road; a dry terrace in the passage is the town site."""
    c, d, n, pt, fd = _frame(u); i = u["id"]; H, G = u.get("h", 70.0), u.get("gap", 130.0)
    F = []
    for nm, sg in (("l", 1), ("r", -1)):
        F.append({"kind": "ridge", "id": f"{i}_ridge_{nm}", "path": [pt(-15, sg * (G / 2 + 95)), pt(25, sg * (G / 2 + 260)), pt(-20, sg * (G / 2 + 560))], "h": [H * 0.7, H, H * 0.9],
                  "w": 105, "sharp": 0.3, "jag": 0.1})
    F += [{"kind": "river", "id": f"{i}_river", "path": [pt(-500, 10), pt(0, -12), pt(500, 8)], "w": u.get("w", 14.0), "depth": 1.4, "wig": [min(G * 0.18, 30.0), 160.0]},
          {"kind": "plateau", "id": f"{i}_terrace", "poly": [pt(-150, 18), pt(40, 18), pt(40, 75), pt(-150, 75)], "h": 4.0, "edge": 10, "jag": 0.2},
          {"kind": "road", "id": f"{i}_road", "path": [pt(-480, 40), pt(-60, 45), pt(80, 40), pt(480, 30)], "w": 4.0, "grade": 0.1}]
    A = {"gap": pt(0, 0), "town": pt(-55, 46), "ridge_l": pt(15, G / 2 + 150), "ridge_r": pt(15, -(G / 2 + 150)), "river": f"{i}_river.mid", "road_in": pt(-230, 44),
         "road_out": pt(230, 36)}
    return F, A


def a_marsh(u):
    """Marsh with terps: a flat lowland, a dike along the sea, behind it a salt marsh with channels and the sea; a terp is a village mound."""
    c, d, n, pt, fd = _frame(u); i = u["id"]; Ld = u.get("dike_at", 170.0)
    F = [{"kind": "lake", "id": f"{i}_sea", "poly": [pt(Ld + 70, -900), pt(Ld + 80, 0), pt(Ld + 70, 900), pt(Ld + 1600, 900), pt(Ld + 1600, -900)], "level": -0.6, "depth": 4,
          "beach": 12, "bank_slope": 3, "bluff": 8, "jag": 0.2},
         {"kind": "dike", "id": f"{i}_dike", "path": [pt(Ld, -900), pt(Ld + 12, 0), pt(Ld, 900)], "h": u.get("dike_h", 4.5), "sea": "R"},
         {"kind": "river", "id": f"{i}_creek", "path": [pt(-600, 60), pt(-200, 30), pt(Ld - 8, 12)], "w": 6.0, "depth": 1.2, "wig": [22.0, 85.0]},
         {"kind": "river", "id": f"{i}_slenk", "path": [pt(Ld + 10, 120), pt(Ld + 40, 90), pt(Ld + 75, 60)], "w": 4.0, "depth": 1.0},
         {"kind": "mound", "id": f"{i}_terp", "type": "terp", "at": c.tolist(), "h": u.get("terp_h", 6.0), "top": u.get("terp_top", 35.0)}]
    A = {"terp": f"{i}_terp.top", "dike": pt(Ld + 12, 0), "sea": pt(Ld + 115, 0), "open_sea": pt(Ld + 400, 0), "kwelder": pt(Ld + 40, 0), "creek": f"{i}_creek@0.8", "terp_foot": f"{i}_terp.foot_e"}
    return F, A


def a_tell(u):
    """Tell on a plain: a settlement mound (flat top, steep side, gullies), the lower town around it, a stream, "hollow ways"
    (the hollow ways of Upper Mesopotamia — wide sunken tracks that radiate from the town; Wilkinson 1993)."""
    c, d, n, pt, fd = _frame(u); i = u["id"]; h, rx, ry = u.get("h", 18.0), u.get("rx", 75.0), u.get("ry", 58.0)
    F = [{"kind": "mound", "id": f"{i}_tell", "type": "tell", "at": c.tolist(), "h": h, "rx": rx, "ry": ry, "rot": fd + 90, **({"age": u["age"]} if "age" in u else {})},
         {"kind": "fill", "id": f"{i}_lower", "at": c.tolist(), "r": u.get("lower_r", 175.0), "t": u.get("lower_t", 1.6)},
         {"kind": "river", "id": f"{i}_stream", "path": [pt(-700, 220), pt(-100, 195), pt(150, 240), pt(700, 205)], "w": u.get("w", 9.0), "depth": 1.2, "wig": [30.0, 110.0]}]
    for k, a in enumerate(u.get("hollows", [200, 250, 320])):
        aa = math.radians(a); dv = np.array([math.cos(aa), math.sin(aa)]); r0 = u.get("lower_r", 175.0) * 0.9
        F.append({"kind": "holloway", "id": f"{i}_hw{k}", "path": [(c + dv * r0).round(1).tolist(), (c + dv * (r0 + 150) + np.array([-dv[1], dv[0]]) * 12).round(1).tolist(), (c + dv * (r0 + 400)).round(1).tolist()],
                  "depth": 1.1, "w": 22.0, "side": 9.0, "bank": 0.0})
    A = {"top": f"{i}_tell.top", "foot": pt(-(ry + h / math.tan(math.radians(32)) + 6), 0), "ramp": pt(ry + 22, 0), "lower": pt(-(u.get("lower_r", 175.0) * 0.6), 40),
         "stream": f"{i}_stream.mid", "hollow": f"{i}_hw0.start"}
    return F, A


def a_hive(u):
    """Hive wasteland: ash dunes to the horizon, the hive platform flat (crests cut off), a mine quarry with a poisoned lake on
    the floor, a spoil heap from the quarry excavation and a spoil tip, pipeline embankments leading to the hive."""
    c, d, n, pt, fd = _frame(u); i = u["id"]; Rh, Rp = u.get("r", 95.0), u.get("pit_at", 210.0)
    F = [{"kind": "dunes", "id": f"{i}_ash", "wind": fd + 90, "step": u.get("step", 55.0), "h": u.get("dune_h", 3.5), "soil": "ash"},
         {"kind": "fill", "id": f"{i}_pad", "at": c.tolist(), "r": Rh, "t": 0.3, "smooth": 60.0, "flat": True, "edge": 35},
         {"kind": "pit", "id": f"{i}_pit", "type": "mine", "at": pt(Rp, Rp * 0.35), "r": u.get("pit_r", 95.0), "depth": u.get("pit_d", 40.0),
          "water": 0.45 * float(u.get("pit_d", 40.0))},   # abandoned quarry: groundwater has risen onto the lower benches (pit lake)
         {"kind": "heap", "id": f"{i}_spoil", "type": "flat_top", "at": pt(Rp * 0.2, Rp * 1.05), "of": f"{i}_pit", "top": 42},
         {"kind": "heap", "id": f"{i}_cone", "type": "spoil_cone", "at": pt(-Rp * 0.8, Rp * 0.9)},
         {"kind": "dike", "id": f"{i}_pipe", "path": [pt(-Rh - 10, -20), pt(-260, -110), pt(-520, -150)], "h": 4.5, "crest": 6, "out": 33, "in": 33},
         {"kind": "dike", "id": f"{i}_rail", "path": [pt(10, -Rh - 10), pt(60, -280), pt(40, -560)], "h": 3.5, "crest": 8, "out": 30, "in": 30}]
    A = {"hive": c.tolist(), "pit": f"{i}_pit.center", "rim": f"{i}_pit.rim", "spoil": f"{i}_spoil.center", "cone": f"{i}_cone.center", "pipe": f"{i}_pipe@0.3",
         "rail": f"{i}_rail@0.3", "edge": pt(-Rh - 30, 0)}
    return F, A


ARCH = {"hillfort": a_hillfort, "spur": a_spur, "confluence": a_confluence, "meander": a_meander, "river_terrace": a_river_terrace, "acropolis": a_acropolis,
        "bay": a_bay, "gap": a_gap, "marsh": a_marsh, "tell": a_tell, "hive_wastes": a_hive}
SETTLE = {   # settlements of the archetypes — cultivation of the land around them (terrain.settle_of): type → kind ("" — any) → [(anchor, core radius, m)]
    "hillfort": {"contour": [("yard", 110)], "": [("yard", 55)]}, "spur": {"": [("yard", 55)]},
    "confluence": {"": [("kremlin", 70), ("posad", 140), ("podol", 90)]}, "meander": {"": [("town", 150)]}, "river_terrace": {"": [("town", 150)]},
    "acropolis": {"": [("top", 90), ("lower", 160)]}, "bay": {"": [("quay", 120)]}, "gap": {"": [("town", 130)]}, "marsh": {"": [("terp", 40)]},
    "tell": {"": [("top", 90), ("lower", 150)]}}            # a hive wasteland is not arable land: no cultivation


def archetype(u):
    F, A = ARCH[u["type"]](u)
    return F, {f"{u['id']}.{k}": v for k, v in A.items() if v is not None}
