"""Site from a plan: relief in meters — the near-plot height grid, the mountain belt to the horizon, masks, anchors, paths,
checks. Design — a tree of primitives (Génevaux et al. 2015): the plan shape = a profile by distance to a point, a line or a
contour + an operation (add, soft maximum, cut to a level, flatten). The shore profile is piecewise (a gentle beach,
then a steep bank), the soft minimum is I. Quilez's formula. numpy only: the module is read by both Blender
and .venv.
.venv/bin/python -m components.site.terrain build <plan.yaml> [folder]   → relief.npz, location.json, sheet.png; the __RESULT__ line
.venv/bin/python -m components.site.terrain scene <plan.yaml> <name>     → and a pipeline scene scenes/<name>/ (.blend — out/build/<name>/), cameras by anchors
.venv/bin/python -m components.site.terrain export <site_folder> [out]   → 16-bit PNG heights, masks, export.json — for simulations and third-party programs
.venv/bin/python -m components.site.terrain preset <name> [seed] [scene] [knob=value …] → a scene from a preset in library/terrain/presets/
The plan is the relief: key (or the whole file): size, res, center, seed, ground, base, detail, far, use (archetypes), features (shapes).
Axes: x — east, y — north, z — up; the far-ground level is far_z = 0. Queries to a finished site — core/surface/ground.py."""
import core.paths         # the code is here, the core — core/<part> through kernel
import json, math, os, sys, time, zlib
import numpy as np

from core.surface.noise import perlin, fbm, ridged, smoothstep, _perm  # noqa: E402,F401  (noise lives in the core; the names stay in relief)


# ---------------------------------------------------------------- profiles and operations


def smin(a, b, k):
    """Smooth minimum (I. Quilez): no seam where shapes meet; k is the seam width, m."""
    if k <= 0: return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def lift(H, T, fade):
    """Raise the relief up to the shape T (where it is lower), with a fade toward the edge of influence — no step at the window edge."""
    return H + np.maximum(0.0, T - H) * fade


def terrace(z, step, sharp=0.6):
    """Steps of height step: a flat shelf and a steep ledge (sharp is the ledge fraction within a step)."""
    q = z / step; f = np.floor(q); t = q - f
    return (f + smoothstep(1 - sharp, 1.0, t)) * step


def strata_layers(seed, mean=2.6, var=0.7, span=(-600.0, 900.0)):
    """A stack of strata by height: boundaries (m, thicknesses — log-normal around mean, spread var), hardness 0…1 (hard ones form a
    ledge), stratum tint (−1…1). One per site: it is used to carve the rock (cliff_detail) and to color the cliffs (terrain_tex)."""
    r = np.random.default_rng(int(seed) * 31 + 7); n = int((span[1] - span[0]) / (mean * 0.6)) + 8
    th = np.clip(np.exp(r.normal(np.log(mean), var, n)), 0.5, mean * 4.0); E = span[0] + np.r_[0, np.cumsum(th)]
    hard = np.clip(r.beta(0.8, 0.8, n), 0, 1); tint = np.clip(r.normal(0, 0.5, n), -1, 1)
    return E, hard, tint


def cliff_detail(H, X, Y, res, seed, st):
    """Rock on steep ground (on any base): strata of different thickness and hardness (hard ones form a ledge, soft ones recede; the
    stack is tilted and the strata break off in places), vertical division (buttresses and gullies along the slope's dip, 6–20 m
    spacing) and blocky facets (3–8 m blocks, each with its own offset, sharp edges — readable from afar). st: step (mean stratum
    thickness), var, dip (stack slope, m/m), ribs (division amplitude, m), blocks (facet amplitude, m), from/full — the slope from
    which rock starts. Returns an addition to the height."""
    gy, gx = np.gradient(H, res); g = np.hypot(gx, gy) + 1e-9
    Hb = H.copy()                                                   # slope on the smoothed (~1.5 m) ground: the rim and gentle steeps do not count
    for _ in range(max(1, int(1.5 / res))): Hb[1:-1, 1:-1] = (Hb[2:, 1:-1] + Hb[:-2, 1:-1] + Hb[1:-1, 2:] + Hb[1:-1, :-2] + 4 * Hb[1:-1, 1:-1]) / 8
    sy, sx = np.gradient(Hb, res); slp = np.degrees(np.arctan(np.hypot(sx, sy)))
    m = smoothstep(float(st.get("from", 46)), float(st.get("full", 60)), slp)   # rock only on walls (at 38° "teeth" appeared on the rim)
    if not (m > 0.01).any(): return np.zeros_like(H)
    ux, uy = -gy / g, gx / g                                        # horizontally along the slope (strike)
    U = X * ux + Y * uy; Z = H + float(st.get("dip", 0.08)) * (X * 0.6 + Y * 0.8) + 1.2 * fbm(X / 35.0, Y / 35.0, 2, seed + 41)
    E, hard, _ = strata_layers(seed, float(st.get("step", 2.6)), float(st.get("var", 0.7)))
    k = np.clip(np.searchsorted(E, Z) - 1, 0, len(hard) - 1); t = (Z - E[k]) / (E[k + 1] - E[k])
    sharp = 0.15 + 0.55 * hard[k]                                   # a hard stratum — a shelf and a short ledge; a soft one — an even slope
    terr = E[k] + (E[k + 1] - E[k]) * smoothstep(1 - sharp, 1.0, t)
    cont = smoothstep(-0.2, 0.3, fbm(U / 40.0, Z / 25.0, 2, seed + 43))   # ledges break off in places
    d_strata = (terr - Z) * cont * (0.4 + 0.6 * hard[k])
    rib = float(st.get("ribs", 0.9)) * (ridged(U / 13.0, Z / 40.0, 3, seed + 45) - 0.55) * 2   # buttresses and gullies along the dip
    cs = float(st.get("cell", 6.0)); qu, qz = U / cs, Z / (cs * 0.8); iu, iz = np.floor(qu), np.floor(qz)   # boulders: Voronoi cells
    best = np.full(H.shape, 9e9); off = np.zeros_like(H); tu = np.zeros_like(H); tz = np.zeros_like(H)
    hsh = lambda a, b, s_: (np.sin(a * 127.1 + b * 311.7 + s_ * 74.7) * 43758.5453) % 1.0
    for da in (-1, 0, 1):
        for db in (-1, 0, 1):
            ca, cb = iu + da, iz + db; pu, pz = ca + hsh(ca, cb, 1), cb + hsh(ca, cb, 2); d = (qu - pu) ** 2 + (qz - pz) ** 2
            w = d < best; best = np.where(w, d, best); off = np.where(w, hsh(ca, cb, 3) - 0.5, off)
            tu = np.where(w, hsh(ca, cb, 4) - 0.5, tu); tz = np.where(w, hsh(ca, cb, 5) - 0.5, tz)
    blk = float(st.get("blocks", 0.6)) * (off * 2 + tu * (qu - iu - 0.5) + tz * (qz - iz - 0.5))   # block offset and tilt — flat facets
    return (d_strata + rib + blk) * m


COMPASS = {"E": 0, "NE": 45, "N": 90, "NW": 135, "W": 180, "SW": 225, "S": 270, "SE": 315}


def ang(d):
    """Direction: N|NE|E…|degrees counterclockwise from east → radians."""
    return math.radians(COMPASS[d] if isinstance(d, str) else float(d))


# ---------------------------------------------------------------- geometry: polylines, contours
def chaikin(P, n=2, closed=False):
    P = np.asarray(P, float)
    for _ in range(n):
        if len(P) < 3: break
        Q = np.concatenate([P, P[:1]]) if closed else P
        a, b = Q[:-1], Q[1:]
        R = np.empty((2 * len(a), 2)); R[0::2] = 0.75 * a + 0.25 * b; R[1::2] = 0.25 * a + 0.75 * b
        P = R if closed else np.concatenate([P[:1], R[1:-1], P[-1:]])
    return P


WIGGLE = {"river": (1.2, 12.0), "road": (1.5, 50.0), "valley": (0.15, 6.0), "ridge": (0.2, 2.5), "cliff": (0.35, 2.0), "trench": (2.5, 18.0),
          "rampart": (1.5, 12.0), "moat": (1.0, 10.0)}   # (amplitude, wavelength) as fractions of width/drop


def wiggle(P, amp, lam, seed):
    """Polyline sinuosity: shift across the path by noise with wavelength lam m, endpoints stay in place."""
    P = np.asarray(P, float)
    if amp <= 0 or len(P) < 2: return P
    Q, s = resample(chaikin(P, 3), max(lam / 8, 1.0)); L = s[-1]   # smooth first, then add noise
    if L < lam * 0.5: return P
    t = np.gradient(Q, axis=0); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9); n = np.stack([-t[:, 1], t[:, 0]], 1)
    off = amp * fbm(s / lam, np.full_like(s, 0.37 + seed % 97), 3, seed) * smoothstep(0, lam * 0.5, s) * smoothstep(0, lam * 0.5, L - s)
    return Q + n * off[:, None]


def cumlen(P):
    d = np.hypot(*np.diff(P, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(d)])


def resample(P, step):
    c = cumlen(P); n = max(2, int(math.ceil(c[-1] / step)) + 1); s = np.linspace(0, c[-1], n)
    return np.stack([np.interp(s, c, P[:, 0]), np.interp(s, c, P[:, 1])], 1), s


def line_info(X, Y, P):
    """Nearest point on a polyline: distance d, path along s, side (+1 to the left along the direction of travel), overshoot past the ends ext (m)."""
    P = np.asarray(P, float); seg = np.diff(P, axis=0); L = np.hypot(seg[:, 0], seg[:, 1]); cum = np.concatenate([[0.0], np.cumsum(L)])
    d = np.full(X.shape, np.inf); s = np.zeros(X.shape); side = np.zeros(X.shape); ext = np.zeros(X.shape)
    last = len(seg) - 1
    for i, ((ax, ay), (dx, dy), l) in enumerate(zip(P[:-1], seg, L)):
        if l < 1e-9: continue
        tr = ((X - ax) * dx + (Y - ay) * dy) / (l * l); t = np.clip(tr, 0, 1)
        qx, qy = X - (ax + t * dx), Y - (ay + t * dy); di = np.hypot(qx, qy); m = di < d
        if not m.any(): continue
        d = np.where(m, di, d); s = np.where(m, cum[i] + t * l, s); side = np.where(m, np.sign(dx * qy - dy * qx + 1e-12), side)
        e = np.zeros(X.shape)
        if i == 0: e = np.maximum(e, -tr * l)
        if i == last: e = np.maximum(e, (tr - 1) * l)
        ext = np.where(m, e, ext)
    return d, s, side, ext, cum[-1]


def inside(X, Y, P):
    P = np.asarray(P, float); r = np.zeros(X.shape, bool)
    for (x1, y1), (x2, y2) in zip(P, np.roll(P, -1, 0)):
        c = ((y1 > Y) != (y2 > Y)) & (X < (x2 - x1) * (Y - y1) / np.where(y2 == y1, 1e-12, y2 - y1) + x1)
        r ^= c
    return r


def sdf_poly(X, Y, P):
    """Signed distance to a contour: negative inside."""
    P = np.asarray(P, float)
    d = line_info(X, Y, np.concatenate([P, P[:1]]))[0]
    return np.where(inside(X, Y, P), -d, d)


def ray_poly(P, c, a):
    """Contour point P along a ray from c at angle a (rad): the far intersection."""
    o, u = np.asarray(c, float), np.array([math.cos(a), math.sin(a)]); best = None
    for p1, p2 in zip(np.asarray(P, float), np.roll(np.asarray(P, float), -1, 0)):
        e = p2 - p1; den = u[0] * e[1] - u[1] * e[0]
        if abs(den) < 1e-12: continue
        w = p1 - o; t = (w[0] * e[1] - w[1] * e[0]) / den; s = (w[0] * u[1] - w[1] * u[0]) / den
        if t > 0 and 0 <= s <= 1 and (best is None or t > best): best = t
    return o + u * (best or 0.0)


def footprint(f):
    """Shape contour: poly | at+r (circle) | at+size [w, l]+rot (rectangle)."""
    if "poly" in f: return np.asarray(f["poly"], float)
    c = np.asarray(f["at"], float)
    if "size" in f:
        w, l = f["size"]; a = math.radians(f.get("rot", 0)); R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        return c + np.array([[-w / 2, -l / 2], [w / 2, -l / 2], [w / 2, l / 2], [-w / 2, l / 2]]) @ R.T
    t = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    return c + np.stack([np.cos(t), np.sin(t)], 1) * f.get("r", 10.0)


# ---------------------------------------------------------------- height grid
class Grid:
    """Height grid: H[iy, ix], node (x0 + ix·res, y0 + iy·res)."""

    def __init__(self, x0, y0, res, nx, ny, z=0.0):
        self.x0, self.y0, self.res, self.nx, self.ny = float(x0), float(y0), float(res), int(nx), int(ny)
        self.H = np.full((self.ny, self.nx), float(z))

    def bounds(self):
        return self.x0, self.y0, self.x0 + (self.nx - 1) * self.res, self.y0 + (self.ny - 1) * self.res

    def window(self, bb):
        """Grid window by the box (x0, y0, x1, y1): slices and node coordinates; None — outside the grid."""
        i0 = max(0, int(math.floor((bb[0] - self.x0) / self.res))); i1 = min(self.nx, int(math.ceil((bb[2] - self.x0) / self.res)) + 1)
        j0 = max(0, int(math.floor((bb[1] - self.y0) / self.res))); j1 = min(self.ny, int(math.ceil((bb[3] - self.y0) / self.res)) + 1)
        if i0 >= i1 or j0 >= j1: return None
        X, Y = np.meshgrid(self.x0 + np.arange(i0, i1) * self.res, self.y0 + np.arange(j0, j1) * self.res)
        return (slice(j0, j1), slice(i0, i1)), X, Y

    def sample(self, x, y, A=None):
        A = self.H if A is None else A
        gx = np.clip((np.asarray(x, float) - self.x0) / self.res, 0, self.nx - 1.001); gy = np.clip((np.asarray(y, float) - self.y0) / self.res, 0, self.ny - 1.001)
        ix, iy = gx.astype(int), gy.astype(int); fx, fy = gx - ix, gy - iy
        return (A[iy, ix] * (1 - fx) * (1 - fy) + A[iy, ix + 1] * fx * (1 - fy) + A[iy + 1, ix] * (1 - fx) * fy + A[iy + 1, ix + 1] * fx * fy)


# ---------------------------------------------------------------- archetypes: site presets with anchors
def _off(c, a_deg, r):
    a = math.radians(a_deg); return [c[0] + r * math.cos(a), c[1] + r * math.sin(a)]


def archetype(u):
    """Archetype → (shapes, anchor aliases). The defaults carry a margin for comics: the elevation drops are bigger than "real-life" ones."""
    t, i, c = u["type"], u["id"], list(map(float, u.get("at", [0, 0])))
    face = math.degrees(ang(u.get("face", "S")))                    # where the site's "entrance" faces
    F, A = [], {}
    if t == "fort_hill":                                            # hill fort: hill, yard, a serpentine road to the gate
        h, r = u.get("h", 26.0), u.get("r", 70.0); yard = u.get("yard", 18.0)
        F.append({"kind": "hill", "id": f"{i}_hill", "at": c, "r": r, "h": h, "shape": "dome", "jag": 0.12})   # the summit is rounded — the yard is not a raised fill
        F.append({"kind": "pad", "id": f"{i}_yard", "at": c, "r": yard, "h": "auto", "surface": "stone"})      # walls are library objects
        g = _off(c, face, yard)                                     # serpentine: the angle grows monotonically — the road goes around the hill, not across the yard
        road = [_off(c, face, r * 1.5), _off(c, face + 100, r * 1.05), _off(c, face + 200, r * 0.8), _off(c, face + 290, r * 0.55), _off(c, face + 360, yard + 2)]
        F.append({"kind": "road", "id": f"{i}_road", "path": road, "w": u.get("road_w", 4.0), "grade": 0.14})
        A.update({"yard": f"{i}_yard.center", "gate": g, "top": f"{i}_hill.top", "foot": f"{i}_road.start", "climb": f"{i}_road@0.5"})
    elif t == "gorge":                                              # gorge: a floor with a trail or a stream, walls, a pocket widening
        L, depth, w = u.get("length", 220.0), u.get("depth", 30.0), u.get("w", 14.0)
        d = np.array([math.cos(math.radians(face)), math.sin(math.radians(face))]); n = np.array([-d[1], d[0]])
        bend = u.get("bend", 0.15) * L
        P = [list(np.asarray(c) - d * L / 2), list(np.asarray(c) - d * L / 6 + n * bend), list(np.asarray(c) + d * L / 6 - n * bend), list(np.asarray(c) + d * L / 2)]
        M = [list(np.asarray(c) - d * L * 0.36 + n * bend * 0.5), P[1], P[2], list(np.asarray(c) + d * L * 0.36 - n * bend * 0.5)]
        if u.get("path"): P = M = [list(map(float, q)) for q in u["path"]]          # floor along the given path (found from the base)
        if u.get("mass", "ridge") == "plateau":                     # canyon in a plateau: a table with a cliffy edge, the gorge cut right through
            W2 = w + depth * 3.0; cc = np.asarray(c)
            poly = [list(cc - d * L * 0.42 - n * W2), list(cc + d * L * 0.42 - n * W2), list(cc + d * L * 0.42 + n * W2), list(cc - d * L * 0.42 + n * W2)]
            F.append({"kind": "plateau", "id": f"{i}_mass", "poly": poly, "h": depth * 1.05, "cliff": True, "jag": 0.25, "steps": u.get("steps", 0)})
        else: F.append({"kind": "ridge", "id": f"{i}_mass", "path": M, "h": depth * 1.2, "w": w * 2.0 + depth * 2.4, "sharp": 0.0, "jag": 0.08})
        F.append({"kind": "valley", "id": f"{i}_floor", "path": P, "floor": u.get("floor", 0.0), "fw": w / 2, "walls": u.get("walls", 72), "w": w / 2 + depth})
        if u.get("stream"): F.append({"kind": "river", "id": f"{i}_stream", "path": P, "w": 3.0, "depth": 0.6})
        A.update({"entry": f"{i}_floor.entry", "exit": f"{i}_floor.exit", "floor": f"{i}_floor.floor", "rim": f"{i}_floor.rim_l",
                  "rim_r": f"{i}_floor.rim_r", "pocket": f"{i}_floor@0.62"})
    elif t == "pass":                                               # pass: two ridges, a saddle, a road over the saddle
        h, L = u.get("h", 60.0), u.get("length", 320.0); a = math.radians(face); d = np.array([math.cos(a), math.sin(a)]); n = np.array([-d[1], d[0]])
        cc = np.asarray(c)
        crest = [list(cc - n * L / 2), list(cc - n * L / 8), list(cc + n * L / 8), list(cc + n * L / 2)]
        F.append({"kind": "ridge", "id": f"{i}_ridge", "path": crest, "h": [h, h * 0.45, h * 0.45, h], "w": u.get("w", 90.0), "sharp": 0.35, "jag": 0.12})
        F.append({"kind": "road", "id": f"{i}_road", "path": [list(cc - d * L * 0.45), list(cc - d * 20), list(cc + d * 20), list(cc + d * L * 0.45)],
                  "w": 4.0, "grade": 0.16})
        A.update({"saddle": f"{i}_road@0.5", "climb": f"{i}_road@0.3", "descent": f"{i}_road@0.7", "crest_l": f"{i}_ridge@0.15",
                  "crest_r": f"{i}_ridge@0.85", "below": f"{i}_road.start"})
    elif t == "crossing":                                           # crossing: a river with banks, a ford, a road to both banks
        b_ = float(u.get("bend", 0.0)) * 70; dd = np.array([math.cos(math.radians(face)), math.sin(math.radians(face))])   # bend — a meander at the ford
        P = u.get("river") or [_off(c, face + 90, 200), list(np.asarray(_off(c, face + 90, 85)) + dd * b_), c, list(np.asarray(_off(c, face - 90, 85)) - dd * b_ * 0.6),
                               _off(c, face - 90, 200)]
        Lp = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.asarray(P, float), axis=0), axis=1))]   # the ford is where the road is: with a meander the center is not at mid-length
        jf = int(np.argmin(np.linalg.norm(np.asarray(P, float) - np.asarray(u["ford_at"], float), axis=1))) if "ford_at" in u else (len(P) // 2 if "river" in u else 2)
        F.append({"kind": "river", "id": f"{i}_river", "path": P, "w": u.get("w", 14.0), "depth": u.get("depth", 1.6),
                  "ford": u.get("ford", float(Lp[jf] / Lp[-1]))})                 # ford_at — the ford point on the given river (found from the base)
        rc = P[jf] if "river" in u else c
        Lr = u.get("road_len", 90.0)
        F.append({"kind": "road", "id": f"{i}_road", "path": [_off(rc, face, Lr), _off(rc, face, 12), _off(rc, face + 180, 12), _off(rc, face + 180, Lr)],
                  "w": 4.0, "grade": 0.12, "cross": u.get("cross", "ford")})   # ford — a ford under water, bridge — a bridge (deck as a mesh)
        A.update({"ford": f"{i}_river.ford", "bank_near": f"{i}_road@0.4", "bank_far": f"{i}_road@0.6", "near": f"{i}_road.start",
                  "far": f"{i}_road.end"})
    elif t == "cliff_edge":                                         # plateau edge: a cliff toward face, flat top
        h, L = u.get("h", 40.0), u.get("length", 260.0); a = face; cc = np.asarray(c)
        n = np.array([math.cos(math.radians(a + 90)), math.sin(math.radians(a + 90))])
        F.append({"kind": "cliff", "id": f"{i}_cliff", "path": [list(cc - n * L / 2), list(cc + n * L / 2)], "drop": h, "high": a + 180,
                  "reach": u.get("reach", 160.0), "jag": 0.08})
        A.update({"edge": f"{i}_cliff.top", "below": f"{i}_cliff.bottom", "plateau": _off(c, a + 180, 40)})
    elif t == "arena":                                              # bowl: a ring higher than the center, a passage
        F.append({"kind": "bowl", "id": f"{i}_bowl", "at": c, "r": u.get("r", 45.0), "h": u.get("h", 14.0), "w": u.get("w", 22.0), "gap": face})
        F.append({"kind": "pad", "id": f"{i}_floor", "at": c, "r": u.get("r", 45.0) * 0.55, "h": "auto", "surface": "stone"})
        A.update({"center": f"{i}_floor.center", "rim": f"{i}_bowl.rim", "gap": f"{i}_bowl.gap", "outside": f"{i}_bowl.outside"})
    elif t == "crater_field":                                       # crater field: no man's land between the lines
        rng = np.random.default_rng(u.get("seed", 3)); R = u.get("r", 70.0)
        for k in range(u.get("n", 14)):
            rr = R * math.sqrt(rng.uniform(0, 1)); a = rng.uniform(0, 2 * math.pi); s = float(rng.choice([2.5, 3.5, 5.0, 8.0]))
            F.append({"kind": "crater", "id": f"{i}_c{k}", "at": [c[0] + rr * math.cos(a), c[1] + rr * math.sin(a)], "r": s, "depth": s * 0.45})
        F.append({"kind": "crater", "id": f"{i}_big", "at": c, "r": u.get("big", 14.0), "depth": u.get("big", 14.0) * 0.4})
        A.update({"big": f"{i}_big.center", "rim": f"{i}_big.rim"})
    elif t == "valley":                                             # valley: two ridges, a stream, a road alongside
        L, h = u.get("length", 400.0), u.get("h", 45.0); a = math.radians(face); d = np.array([math.cos(a), math.sin(a)]); n = np.array([-d[1], d[0]])
        cc = np.asarray(c); W = u.get("w", 120.0)
        for sgn, nm in ((1, "l"), (-1, "r")):
            F.append({"kind": "ridge", "id": f"{i}_ridge_{nm}", "path": [list(cc - d * L / 2 + sgn * n * W), list(cc + sgn * n * W * 1.1), list(cc + d * L / 2 + sgn * n * W)],
                      "h": h, "w": W * 0.8, "sharp": 0.3, "jag": 0.1})
        rp = u.get("river_path") or [list(cc - d * L / 2 - n * 8), list(cc + n * 10), list(cc + d * L / 2 - n * 6)]   # river — along the given path (found)
        F.append({"kind": "river", "id": f"{i}_river", "path": rp, "w": 8.0, "depth": 1.2})
        rq = [list(np.asarray(q, float) + n * 28) for q in (rp[0], rp[len(rp) // 2], rp[-1])] if u.get("river_path") else \
            [list(cc - d * L / 2 + n * 26), list(cc + n * 30), list(cc + d * L / 2 + n * 28)]
        F.append({"kind": "road", "id": f"{i}_road", "path": rq, "w": 4.0})
        A.update({"road": f"{i}_road@0.5", "river": f"{i}_river.mid", "slope_l": f"{i}_ridge_l.side_r", "slope_r": f"{i}_ridge_r.side_l"})
    else:
        from components.site import terrain_works as TW                                  # settlement sites and hillforts — terrain_works.ARCH
        if t in TW.ARCH: return TW.archetype(u)
        raise ValueError(f"archetype {t}? available: fort_hill, gorge, pass, crossing, cliff_edge, arena, crater_field, valley, {', '.join(TW.ARCH)}")
    return F, {f"{i}.{k}": v for k, v in A.items()}


# ---------------------------------------------------------------- shapes
WORKS = ("dunes", "mound", "fill", "lynchets", "furrows", "pit", "heap", "dike", "works", "holloway")   # man-made — terrain_works.py
ORDER = ["plateau", "hill", "ridge", "bowl", "cliff", "rampart", "noise", "@detail", "@erode", "valley", "crater", "moat", "trench", "river", "lake",
         "@strata", "@rock", *WORKS, "terraces", "pad", "road", "@pads", "foothold"]   # @ — assembly steps: fine noise, strata and rock on steep slopes,
                                                                   # platforms over roads; fighters' footholds — last, on the finished ground;
                                                                   # man-made — on the finished natural ground, before platforms and roads


def _bbox_pts(P, m):
    P = np.asarray(P, float); return (P[:, 0].min() - m, P[:, 1].min() - m, P[:, 0].max() + m, P[:, 1].max() + m)


def influence(f):
    """The box outside which the shape does not touch the relief."""
    k = f["kind"]
    if k in WORKS: from components.site import terrain_works as TW; return TW.influence(f)
    if k == "hill":
        r = max(f.get("rx", f.get("r", 30)), f.get("ry", f.get("r", 30))) * (1 + 1.4 * f.get("jag", 0)); return _bbox_pts([f["at"]], r * 1.05 + 1)   # noise shifts the edge
    if k in ("ridge", "rampart", "moat", "trench", "valley", "river", "road", "cliff"):
        m = {"ridge": f.get("w", 40) * (1 + 1.4 * f.get("vary", 0.25)) + 2, "rampart": f.get("base", f.get("w", 4) * 2.5) + 1, "moat": f.get("base", f.get("w", 6) * 2) + 1,
             "trench": f.get("w", 1.5) * 2 + 1, "valley": f.get("w", 30) * 1.5 + 5, "river": f.get("w", 10) / 2 + f.get("bank", 4) + 40,
             "road": f.get("w", 4) / 2 + 60, "cliff": f.get("reach", 4 * f.get("drop", 10)) + 2 * f.get("drop", 10) + f.get("w", 0.3 * f.get("drop", 10))}[k]
        return _bbox_pts(f["path"], m)
    if k in ("plateau", "pad", "lake", "terraces", "foothold"):
        if k == "plateau":
            h_ = float(f.get("h", 10)); m = f.get("edge", h_ * (0.5 if f.get("cliff") else 2.5)) + 2 + f.get("jag", 0) * 0.2 * np.ptp(footprint(f), 0).max()
        else: m = {"pad": 40, "lake": 60, "terraces": 2, "foothold": 3}[k]
        return _bbox_pts(footprint(f), m)
    if k == "crater": return _bbox_pts([f["at"]], f.get("r", 5) * 2.2)
    if k == "bowl": return _bbox_pts([f["at"]], f.get("r", 40) + f.get("w", 20) * 1.2)
    if k == "noise": return _bbox_pts(footprint(f), 2) if ("poly" in f or "at" in f) else (-1e9, -1e9, 1e9, 1e9)
    raise ValueError(f"shape {k}?")


class Ctx:
    """Build state on one grid: heights, masks (near plot only), a cache of derived quantities of the shapes."""

    def __init__(self, grid, near, cache, cx=0.0, cy=0.0, half=150.0, FS=3000.0, fres=5.0):
        self.g, self.near, self.cache, self.cx, self.cy, self.half, self.FS, self.fres = grid, near, cache, cx, cy, half, FS, fres
        if near:
            shp = grid.H.shape
            self.level = np.full(shp, np.nan); self.road = np.zeros(shp, np.float32); self.pad = np.zeros(shp, np.float32); self.pave = np.zeros(shp, np.float32)
            self.cliff = np.zeros(shp, np.float32)

    def fades(self, X, Y):
        """Multipliers: toward the edge of the near plot (fine detail fades — the belt does not hold it) and toward the edge of the belt
        (level far_z)."""
        q = np.maximum(np.abs(X - self.cx), np.abs(Y - self.cy)) / self.half; r = self.FS / (2 * self.half)
        return 1 - smoothstep(0.9, 0.99, q), 1 - smoothstep(0.86 * r, 0.99 * r, q)

    def put(self, name, sl, val, op=np.maximum):
        if self.near: A = getattr(self, name); A[sl] = op(A[sl], val)


def shore(r, lvl, f):
    """Shore from the waterline (r ≥ 0 — toward land, m): a gentle beach of beach m at bank_slope, then a steep bank at bluff — the
    terrain above is left untouched."""
    beach, b1, b2 = f.get("beach", 4.0), math.tan(math.radians(f.get("bank_slope", 12))), math.tan(math.radians(f.get("bluff", 55)))
    return lvl + np.minimum(r, beach) * b1 + np.maximum(r - beach, 0) * b2


def _profile_along(ctx, f, P, s_step, smooth_m, median_m=0.0):
    """Relief heights along a polyline (on the current grid), smoothed with a smooth_m window; median_m — first a window median:
    narrow dips across the path (a gully under a river) do not drag the profile down."""
    Q, s = resample(P, s_step); z = ctx.g.sample(Q[:, 0], Q[:, 1])
    km = max(1, int(median_m / s_step)) | 1
    if median_m and len(z) > km:
        zp = np.pad(z, km // 2, mode="edge"); z = np.median(np.lib.stride_tricks.sliding_window_view(zp, km), axis=1)
    k = max(1, int(smooth_m / s_step)) | 1
    if len(z) > k:
        zp = np.pad(z, k // 2, mode="edge"); z = np.convolve(zp, np.ones(k) / k, mode="valid")
    return Q, s, z


HILL_STEEP, HILL_BIG = 28.0, 150.0                                    # a free hill below 150 m: slope up to 28°, rounded summit
PEAK = {"dome": 1.54, "bell": 1.75, "cone": 1.4, "cinder": 1.54}     # the steepest slope of the shape profile, × h / r


def hill_law(f, biome):
    """A free-standing hill is not a small mountain. Mountains in nature are massifs; a low isolated rise on a plain is a moraine
    ridge, a drumlin, a soil-mantled hill: rounded and gentle. Below HILL_BIG m the summit is rounded (cone → bell; for volcanoes —
    a cinder cone: straight slope, top cut by a crater); the steepest slope, accounting for the ragged edge (jag) and the steep side
    (steep), is HILL_STEEP°: the height is clipped, the footprint on the ground stays the same. Flat-topped residual hills (mesa:
    table mountains, kekurs) are left as is. → a note | None."""
    h = float(f.get("h", 10.0)); shape0 = shape = f.get("shape", "dome")
    if shape == "mesa" or h >= HILL_BIG: return None
    if shape == "cone": shape = f["shape"] = "cinder" if biome == "volcanic" else "bell"
    r = min(float(f.get("rx", f.get("r", 30.0))), float(f.get("ry", f.get("r", 30.0))))
    k = PEAK.get(shape, 1.75) * (1 + float(f.get("jag", 0.0))) * (float(f["steep"][1]) if "steep" in f else 1.0)
    hmax = math.tan(math.radians(HILL_STEEP)) * r / k
    if h > hmax: f["h"] = round(hmax, 1)
    return f"{f.get('id')}: {h:.0f} → {float(f['h']):.0f} m, {shape}" if h > hmax or shape != shape0 else None


def gully_room(H, res, lo=20.0, hi=60.0, half_m=48.0):
    """Erosion fraction by the elevation drop in a ~100 m window: below lo m — 0 (a soil-mantled hill is smooth: creep smooths faster
    than incision; gullies start from the valley-spacing scale, Perron, Kirchner, Dietrich 2009: D/K — tens of meters), above
    hi — 1. A flat valley floor is not cut (water deposits there), a star of gullies does not form on a small mound — nor on a
    "small mountain". The drop is measured on a ~4 m grid. → a multiplier."""
    from core.surface.terrain_analysis import dilate
    from components.site import terrain_erosion as TE
    k = max(1, int(round(4.0 / res))); Hs = H[::k, ::k]; n = max(1, int(round(half_m / (res * k))))
    return smoothstep(lo, hi, TE.resize(dilate(Hs, n) - dilate(Hs, n, np.minimum), H.shape))


def apply(f, ctx, seed):
    if f.get("ghost"): return                                     # the base provides the shape: paths and anchors exist, the ground is left alone
    if f["kind"] in WORKS: from components.site import terrain_works as TW; return TW.apply(f, ctx, seed)   # man-made: its own profile and aging
    k, g = f["kind"], ctx.g
    w_ = g.window(influence(f))
    if w_ is None: return
    sl, X, Y = w_; H = g.H[sl]; rs = seed + (zlib.crc32(str(f.get("id", k)).encode()) & 0xFFFF)   # the shape's seed comes from its name, the same on every run
    if k == "hill":
        c = np.asarray(f["at"], float); rx, ry = f.get("rx", f.get("r", 30.0)), f.get("ry", f.get("r", 30.0)); rot = math.radians(f.get("rot", 0))
        dx, dy = X - c[0], Y - c[1]; jag = f.get("jag", 0.0)
        if jag:
            sc = 1.6 / max(rx, ry); dx = dx + jag * rx * fbm(X * sc, Y * sc, 3, rs); dy = dy + jag * ry * fbm(X * sc + 7.7, Y * sc - 3.3, 3, rs + 1)
        lx, ly = dx * math.cos(rot) + dy * math.sin(rot), -dx * math.sin(rot) + dy * math.cos(rot)
        t = np.hypot(lx / rx, ly / ry)
        if "steep" in f:                                            # steep side: [direction, how many times steeper]
            a, fac = ang(f["steep"][0]), float(f["steep"][1]); th = np.arctan2(dy, dx)
            t = t * (1 + (fac - 1) * np.maximum(0, np.cos(th - a)))
        t = np.clip(t, 0, 1); shape = f.get("shape", "dome")
        P = {"dome": (1 - t * t) ** 2, "cone": (1 - t) ** 1.4, "bell": (np.exp(-4.0 * t * t) - math.exp(-4.0)) / (1 - math.exp(-4.0)),
             "mesa": 1 - 0.85 * smoothstep(0.7, 0.82, t) - 0.15 * smoothstep(0.82, 1.0, t),
             "cinder": np.minimum((1 - t) / 0.65, 1.0) - 0.18 * np.clip(1 - (t / 0.3) ** 2, 0, 1)}[shape]   # cinder: top cut off, a crater
        z = f["h"] * P
        g.H[sl] = lift(H, f.get("base", 0.0) + z, 1 - smoothstep(0.85, 1.0, t)) if f.get("op") == "max" else H + z
    elif k == "ridge":                                              # capsule segments merged by a soft maximum: height along the path without breaks
        P = chaikin(f["path"], 2); hv = np.atleast_1d(np.asarray(f["h"], float)); w0 = f.get("w", 40.0)
        c0 = cumlen(np.asarray(f["path"], float)); cP = cumlen(P)
        hP = np.interp(cP / max(cP[-1], 1e-9), c0 / max(c0[-1], 1e-9) if len(hv) == len(c0) else np.linspace(0, 1, len(hv)), hv)
        w = w0 * (1 + f.get("vary", 0.25) * fbm(X / (1.5 * w0), Y / (1.5 * w0), 2, rs + 5))   # noise is by terrain point, not along the path
        sh = f.get("sharp", 0.3); z = None
        for (ax, ay), (bx, by), h0, h1 in zip(P[:-1], P[1:], hP[:-1], hP[1:]):
            dx, dy = bx - ax, by - ay; l2 = max(dx * dx + dy * dy, 1e-9)
            t_ = np.clip(((X - ax) * dx + (Y - ay) * dy) / l2, 0, 1); dd = np.hypot(X - ax - t_ * dx, Y - ay - t_ * dy)
            tt = np.clip(dd / w, 0, 1); zi = (h0 + (h1 - h0) * t_) * ((1 - sh) * (1 - tt * tt) ** 2 + sh * (1 - tt) ** 1.5)
            z = zi if z is None else np.maximum(z, zi)          # the soft maximum of zeros adds k/4 — a step would build up
        if f.get("jag"): z = z * (1 + f["jag"] * fbm(X / (0.6 * w0), Y / (0.6 * w0), 3, rs))
        g.H[sl] = lift(H, f.get("base", 0.0) + z, (z > 1e-6).astype(float)) if f.get("op") == "max" else H + z
    elif k == "plateau":
        Pp = footprint(f); jag = f.get("jag", 0.0)
        if jag:
            D = max(np.ptp(Pp, 0).max(), 1); sc = 2.0 / D; X2 = X + jag * D * 0.15 * fbm(X * sc, Y * sc, 3, rs); Y2 = Y + jag * D * 0.15 * fbm(X * sc + 4.4, Y * sc + 1.1, 3, rs + 1)
        else: X2, Y2 = X, Y
        sd = sdf_poly(X2, Y2, Pp); h = f["h"]; edge = f.get("edge", h * (0.5 if f.get("cliff") else 2.5)); u = np.clip(sd / edge, 0, 1)
        z = h * (1 - (0.82 * smoothstep(0, 0.28, u) + 0.18 * smoothstep(0.28, 1, u))) if f.get("cliff") else h * (1 - smoothstep(0, 1, u))
        if f.get("steps"): z = np.where(sd > 0, np.minimum(terrace(z, h / f["steps"], 0.5), h), z)
        g.H[sl] = lift(H, f.get("base", 0.0) + z, 1 - smoothstep(0.8, 1.0, u))
        if f.get("cliff"): ctx.put("cliff", sl, ((u > 0.02) & (u < 0.3)).astype(np.float32))
    elif k == "cliff":                                              # a ledge along a line: the high side is high (L|R|direction)
        P = chaikin(f["path"], 2); d, s, side, ext, L = line_info(X, Y, P)
        hi = f.get("high", "L")
        if hi in ("L", "R"): sgn = 1 if hi == "L" else -1
        else:
            a = ang(hi); dd = P[-1] - P[0]; nl = np.array([-dd[1], dd[0]]); sgn = 1 if nl @ [math.cos(a), math.sin(a)] > 0 else -1
        drop = f["drop"]; w = f.get("w", 0.3 * drop); reach = f.get("reach", 4 * drop); fade = f.get("fade", 1.5 * drop)
        jag = f.get("jag", 0.0)
        u = side * sgn * d
        if jag: u = u + jag * drop * fbm(X / max(drop, 1) * 0.8, Y / max(drop, 1) * 0.8, 4, rs)   # by terrain point: no breaks at bends
        up = 0.82 * smoothstep(-w / 2, w / 2, u) + 0.18 * smoothstep(-w / 2 - drop * 1.2, -w / 2, u)   # ledge + scree at the foot
        z = drop * up * (1 - smoothstep(reach, reach + 2 * drop, u)) * (1 - smoothstep(0, fade, ext))
        g.H[sl] = H + z
        ctx.put("cliff", sl, ((np.abs(u) < w * 0.6) & (ext < fade * 0.5)).astype(np.float32))
    elif k in ("rampart", "moat", "trench"):
        P = chaikin(f["path"], 1, f.get("closed", False)); P = np.concatenate([P, P[:1]]) if f.get("closed") else P
        d = line_info(X, Y, P)[0]
        if k == "rampart": top, base, h = f.get("w", 2.0), f.get("base", f.get("w", 2.0) * 2.5), f.get("h", 3.0)
        elif k == "moat": top, base, h = f.get("w", 6.0) * 0.6, f.get("base", f.get("w", 6.0)), -f.get("depth", 4.0)
        else: top, base, h = f.get("w", 1.4) / 2, f.get("w", 1.4) / 2 + 0.3, -f.get("depth", 1.8)
        g.H[sl] = H + h * (1 - smoothstep(top, base, d))
    elif k == "noise":
        sc = 1.0 / f.get("scale", 40.0); fn = ridged if f.get("type") == "ridged" else fbm
        m = 1.0 if not ("poly" in f or "at" in f) else 1 - smoothstep(-5, 5, sdf_poly(X, Y, footprint(f)))
        fn_, ff = ctx.fades(X, Y)
        m = m * ff * (fn_ if f.get("scale", 40.0) < 4 * ctx.fres else 1.0)
        g.H[sl] = H + f.get("amp", 2.0) * fn(X * sc, Y * sc, f.get("octaves", 5), rs) * m
    elif k == "valley":
        P = chaikin(f["path"], 2); d, s, side, ext, L = line_info(X, Y, P); w, fw = f.get("w", 30.0), f.get("fw", 4.0)
        if f.get("floor") is not None:                              # floor at a level (gorge, canyon): cuts away everything above
            key = f["id"] + ":floor"
            if key not in ctx.cache:
                Q, sq, zq = _profile_along(ctx, f, P, 2.0, 40.0)
                fl = f["floor"]
                if fl == "auto": zf = np.minimum.accumulate(zq.min() + 0 * zq) + 0.0       # flat floor at the lowest point along
                elif isinstance(fl, (list, tuple)): zf = np.interp(sq, np.linspace(0, sq[-1], len(fl)), fl)
                else: zf = np.full_like(zq, float(fl))
                ctx.cache[key] = (sq, zf)
            sq, zf = ctx.cache[key]; wl = f.get("walls", 60)        # walls: [left, right] along the direction of travel — an asymmetric valley (meander)
            tw = np.where(side > 0, math.tan(math.radians(wl[0])), math.tan(math.radians(wl[1]))) if isinstance(wl, (list, tuple)) else math.tan(math.radians(wl))
            T = np.interp(s, sq, zf) + np.maximum(0, d - fw) * tw
            g.H[sl] = smin(H, T, 1.0)
        else:
            dep = f.get("depth", 6.0)
            Pf = 1 - smoothstep(fw, w, d) if f.get("shape", "U") == "U" else np.clip(1 - d / w, 0, 1) ** 1.2
            g.H[sl] = H - dep * Pf
    elif k == "crater":
        c = np.asarray(f["at"], float); r, dep = f.get("r", 5.0), f.get("depth", 2.0); rim, rw = f.get("rim", dep * 0.3), f.get("rim_w", r * 0.35)
        d = np.hypot(X - c[0], Y - c[1])
        g.H[sl] = H - dep * np.clip(1 - (d / r) ** 2, 0, 1) + rim * np.exp(-((d - r) / rw) ** 2)
    elif k == "river":
        P = chaikin(f["path"], 2); d, s, side, ext, L = line_info(X, Y, P); hw, dep = f.get("w", 10.0) / 2, f.get("depth", 1.5)
        key = f["id"] + ":level"
        if key not in ctx.cache:                                    # water level: does not rise downstream
            Q, sq, zq = _profile_along(ctx, f, P, 2.0, 30.0, f.get("w", 10.0) * 2 + 40.0); lv = f.get("level", "auto")
            if lv == "auto": zl = np.minimum.accumulate(zq) - f.get("inset", 0.5)
            elif isinstance(lv, (list, tuple)): zl = np.interp(sq, [0, sq[-1]], lv)
            else: zl = np.full_like(zq, float(lv))
            ctx.cache[key] = (sq, zl, Q)
        sq, zl, Q = ctx.cache[key]; lvl = np.interp(s, sq, zl)
        T = np.where(d < hw, lvl - dep * np.sqrt(np.clip(1 - (d / hw) ** 2, 0, 1)), shore(d - hw, lvl, f))
        g.H[sl] = smin(H, T, 0.6)
        ctx.put("level", sl, np.where((d < hw + 2.0) & (ext <= 0), lvl, np.nan), np.fmax)
    elif k == "lake":
        Pp = footprint(f); jag = f.get("jag", 0.3); D = max(np.ptp(Pp, 0).max(), 1)   # uneven shore: a circle from the plan, but not drawn with a compass
        X2 = X + jag * D * 0.22 * fbm(X * 1.6 / D, Y * 1.6 / D, 3, rs) if jag else X; Y2 = Y + jag * D * 0.22 * fbm(X * 1.6 / D + 5.1, Y * 1.6 / D - 2.3, 3, rs + 1) if jag else Y
        sd = sdf_poly(X2, Y2, Pp); key = f["id"] + ":level"
        if key not in ctx.cache:
            lv = f.get("level", "auto"); Pb = resample(np.concatenate([Pp, Pp[:1]]), 2.0)[0]
            ctx.cache[key] = float(ctx.g.sample(Pb[:, 0], Pb[:, 1]).min() - 0.4) if lv == "auto" else float(lv)
        lvl = ctx.cache[key]; dep = f.get("depth", 3.0)
        if f.get("fill"):                                           # fill only: water at the level wherever the ground is lower (a sea on a real base)
            ctx.put("level", sl, np.where((sd < 0) & (H < lvl), lvl, np.nan), np.fmax); return
        T = np.where(sd < 0, lvl - dep * smoothstep(0, max(6.0, dep * 4), -sd), shore(sd, lvl, f))
        g.H[sl] = smin(H, T, 0.6)
        ctx.put("level", sl, np.where(sd < 2.0, lvl, np.nan), np.fmax)
    elif k == "terraces":
        m = 1 - smoothstep(-3, 3, sdf_poly(X, Y, footprint(f)))
        g.H[sl] = H + (terrace(H, f.get("step", 3.0), f.get("sharp", 0.4)) - H) * m
    elif k == "bowl":
        c = np.asarray(f["at"], float); r, w, h = f.get("r", 40.0), f.get("w", 20.0), f.get("h", 12.0)
        d = np.hypot(X - c[0], Y - c[1]); ring = np.clip(1 - ((d - r) / w) ** 2, 0, 1) ** 2
        if "gap" in f:
            a = ang(f["gap"]); th = np.arctan2(Y - c[1], X - c[0]); dth = np.abs((th - a + np.pi) % (2 * np.pi) - np.pi)
            ring = ring * smoothstep(math.radians(f.get("gap_w", 34)) / 2, math.radians(f.get("gap_w", 34)) / 2 + 0.25, dth)
        g.H[sl] = H + h * ring
    elif k in ("pad", "foothold"):                                  # a fighter's foothold is the same platform, but after roads and without recomputing the roads
        Pp = footprint(f); sd = sdf_poly(X, Y, Pp); key = f["id"] + ":z"
        if key not in ctx.cache:                                    # height "at the spot" — the median within radius h_r (for fighter footholds — wider than the foothold itself)
            m = sd <= (f.get("h_r", 0.0) - f.get("r", 0.0) if "h_r" in f else 0.0)
            base = float(np.median(H[m])) if m.any() else float(ctx.g.sample(*np.asarray(f.get("at", Pp.mean(0)))))
            hv = f.get("h", "auto")
            ctx.cache[key] = base if hv == "auto" else base + float(hv[1:]) if isinstance(hv, str) and hv[:1] in "+-" else float(hv)
        zt = ctx.cache[key]; mg = f.get("margin", 0.5)
        cut = math.tan(math.radians(82 if f.get("edge") == "wall" else f.get("slope_cut", 33)))
        need = np.minimum(np.abs(H - zt) / cut, f.get("need_max", 1e9))   # need_max — the edge is no wider (fighters' footholds do not grind down their neighbors)
        wgt = 1 - smoothstep(mg, mg + np.maximum(f.get("soft", 1.0), need), sd)
        g.H[sl] = H + (zt - H) * wgt
        if k == "pad":
            ctx.put("pad", sl, (sd <= 0).astype(np.float32))
            if f.get("surface") == "stone": ctx.put("pave", sl, (sd <= 0).astype(np.float32))   # paved (yard, arena); otherwise — trampled earth
    elif k == "road":
        P = chaikin(f["path"], 3); d, s, side, ext, L = line_info(X, Y, P); hw = f.get("w", 4.0) / 2; key = f["id"] + ":z"
        bridge = f.get("cross", "ford") == "bridge"
        if key not in ctx.cache:                                    # profile: smoothed, slope no steeper than grade; ford — under water with ramps
            Q, sq, zq = _profile_along(ctx, f, P, 1.0, f.get("smooth", 24.0)); gr = f.get("grade", 0.12); ds = np.diff(sq)
            pins = {}                                               # the section at the platform is at its height (the gate at the yard level)
            for pf in ctx.cache.get("@pads", []):
                if pf["id"] + ":z" in ctx.cache:
                    for j_ in np.nonzero(sdf_poly(Q[:, 0], Q[:, 1], footprint(pf)) <= 0.5)[0]: pins[int(j_)] = ctx.cache[pf["id"] + ":z"]
            free = np.array([j not in pins for j in range(len(zq))])
            for j_, z_ in pins.items(): zq[j_] = z_

            def fw():
                for i in range(1, len(zq)):
                    if free[i]: zq[i] = np.clip(zq[i], zq[i - 1] - gr * ds[i - 1], zq[i - 1] + gr * ds[i - 1])

            def bw():
                for i in range(len(zq) - 2, -1, -1):
                    if free[i]: zq[i] = np.clip(zq[i], zq[i + 1] - gr * ds[i], zq[i + 1] + gr * ds[i])
            for _ in range(3): fw(); bw()
            if pins and max(pins) >= len(zq) / 2: fw(); bw()       # last — the passage from the platform outward: the gate matters more than the foot
            elif pins: bw(); fw()
            if ctx.near:                                                # water — only where the point is entirely in water; the level at the rim — without admixture
                fin = np.isfinite(ctx.level)                                # "land" (−1e9 in the sample at the rim gave a level of −1e7 and a pit in the road)
                wet = ctx.g.sample(Q[:, 0], Q[:, 1], fin.astype(float)) > 0.5
                lvq = np.where(wet, ctx.g.sample(Q[:, 0], Q[:, 1], np.where(fin, ctx.level, ctx.g.H)), -1e9)
            else: lvq = np.full(len(Q), -1e9); wet = np.zeros(len(Q), bool)
            if wet.any() and bridge:                                # the bridge goes over the longest continuous stretch of water (the crossing itself);
                e_ = np.diff(np.r_[0, wet.astype(int), 0]); st_, en_ = np.nonzero(e_ == 1)[0], np.nonzero(e_ == -1)[0]   # other contacts are road
                k_ = int(np.argmax(en_ - st_)); wet = np.zeros_like(wet); wet[st_[k_]:en_[k_]] = True
            if wet.any() and not bridge:                            # ford: the roadbed ford m under water, with ramps to it no steeper than grade
                zq = np.where(wet, np.minimum(zq, lvq - f.get("ford", 0.35)), zq)
                for i in range(1, len(zq)): zq[i] = min(zq[i], zq[i - 1] + gr * ds[i - 1])
                for i in range(len(zq) - 2, -1, -1): zq[i] = min(zq[i], zq[i + 1] + gr * ds[i])
            if wet.any() and bridge:                                # bridge: a straight deck from a dry bank to a dry one, not lower than 1.8 m above the water;
                lv = float(lvq[wet].max()); j = np.nonzero(wet)[0]; i0, i1 = max(0, j[0] - 3), min(len(zq) - 1, j[-1] + 3)   # embankments to it no steeper than grade
                za, zb = np.maximum(np.maximum(ctx.g.sample(Q[[i0, i1], 0], Q[[i0, i1], 1]), zq[[i0, i1]]), lv + 1.8)
                zq[i0:i1 + 1] = za + (zb - za) * (sq[i0:i1 + 1] - sq[i0]) / max(sq[i1] - sq[i0], 1e-6)
                for i in range(1, len(zq)):
                    if free[i]: zq[i] = max(zq[i], zq[i - 1] - gr * ds[i - 1])
                for i in range(len(zq) - 2, -1, -1):
                    if free[i]: zq[i] = max(zq[i], zq[i + 1] - gr * ds[i])
                ib = np.unique(np.r_[np.arange(i0, i1 + 1, 2), i1])
                ctx.cache[f["id"] + ":bridge"] = {"path": np.round(np.column_stack([Q[ib, :2], zq[ib]]), 2).tolist(), "level": round(lv, 2), "w": f.get("w", 4.0) + 1.2}
            ctx.cache[key] = (sq, zq, wet)
        sq, zq, wet_s = ctx.cache[key]; zt = np.interp(s, sq, zq)
        cut = math.tan(math.radians(f.get("slope_cut", 33))); need = np.abs(H - zt) / cut
        wgt = (1 - smoothstep(hw, hw + np.maximum(f.get("shoulder", 1.5), need), d)) * (1 - smoothstep(0, np.maximum(4.0, need), ext))   # the end — by fading over the embankment width, with no cut
        if bridge and wet_s.any():                                  # bridge: the channel under the span is left alone (the span is dressing)
            wgt = wgt * (np.interp(s, sq, wet_s.astype(float)) < 0.5)
        g.H[sl] = H + (zt - H) * wgt
        ctx.put("road", sl, ((d <= hw) & (ext <= 0)).astype(np.float32))
    else:
        raise ValueError(f"shape {k}?")


# ---------------------------------------------------------------- anchors
def _dir_pts(c, P=None, r=None):
    out = {}
    for nm, a in (("n", 90), ("e", 0), ("s", 270), ("w", 180)):
        out[nm] = ray_poly(P, c, math.radians(a)) if P is not None else np.asarray(c, float) + r * np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
    return out


def anchors_of(f, cache):
    """Shape anchors (x, y): names of the form "<id>.<anchor>"; paths — for "<id>@fraction"."""
    if f["kind"] in WORKS: from components.site import terrain_works as TW; return TW.anchors_of(f, cache)
    k, i, A = f["kind"], f["id"], {}
    if k == "hill":
        c = np.asarray(f["at"], float); A["top"] = c
        for nm, p in _dir_pts(c, r=max(f.get("r", 30), f.get("rx", 0), f.get("ry", 0)) * 0.95).items(): A["foot_" + nm] = p
    elif k in ("plateau", "pad", "lake", "terraces"):
        Pp = footprint(f); c = np.asarray(f.get("at", Pp.mean(0)), float); A["center" if k != "plateau" else "top"] = c
        for nm, p in _dir_pts(c, Pp).items():
            A["edge_" + nm] = p
            if k == "plateau": A["foot_" + nm] = p + (p - c) / max(np.linalg.norm(p - c), 1e-6) * f.get("edge", f.get("h", 10) * 0.6) * 1.1
    elif k in ("ridge", "valley", "river", "road", "cliff", "rampart", "moat", "trench"):
        P = chaikin(f["path"], 2 if k != "road" else 3); Q, s = resample(P, 1.0); L = s[-1]
        at = lambda t: Q[min(len(Q) - 1, int(round(t * (len(Q) - 1))))]
        tg = lambda t: (lambda j: (Q[min(j + 1, len(Q) - 1)] - Q[max(j - 1, 0)]) / max(np.linalg.norm(Q[min(j + 1, len(Q) - 1)] - Q[max(j - 1, 0)]), 1e-9))(
            int(round(t * (len(Q) - 1))))
        nrm = lambda t: np.array([-tg(t)[1], tg(t)[0]])
        A["start"], A["end"], A["mid"] = Q[0], Q[-1], at(0.5)
        if k == "ridge":
            A["crest"] = at(0.5); w = f.get("w", 40) * 0.55; A["side_l"], A["side_r"] = at(0.5) + nrm(0.5) * w, at(0.5) - nrm(0.5) * w
            hv = np.atleast_1d(np.asarray(f["h"], float))
            if len(hv) > 2: A["saddle"] = at(float(np.argmin(hv[1:-1]) + 1) / (len(hv) - 1))
        elif k == "valley":
            A["floor"], A["entry"], A["exit"] = at(0.5), at(0.03), at(0.97)
            rim = f.get("fw", 4) + (f.get("w", 30) if f.get("floor") is None else 8.0)
            A["rim_l"], A["rim_r"] = at(0.5) + nrm(0.5) * rim, at(0.5) - nrm(0.5) * rim
        elif k == "river":
            hw = f.get("w", 10) / 2 + f.get("bank", 3.0); A["bank_l"], A["bank_r"] = at(0.5) + nrm(0.5) * hw, at(0.5) - nrm(0.5) * hw
            A["source"], A["mouth"], A["ford"] = Q[0], Q[-1], at(f.get("ford", 0.5))
        elif k == "cliff":
            hi = f.get("high", "L"); drop = f["drop"]; w = f.get("w", 0.3 * drop)
            if hi in ("L", "R"): sgn = 1 if hi == "L" else -1
            else:
                a = ang(hi); dd = P[-1] - P[0]; sgn = 1 if np.array([-dd[1], dd[0]]) @ [math.cos(a), math.sin(a)] > 0 else -1
            for t, nm in ((0.5, ""), (0.25, "_25"), (0.75, "_75")):
                A["top" + nm] = at(t) + nrm(t) * sgn * (w * 0.8 + 2.0); A["bottom" + nm] = at(t) - nrm(t) * sgn * (w * 0.8 + drop * 0.6 + 2.0)
    elif k == "crater":
        c = np.asarray(f["at"], float); A["center"] = c
        for nm, p in _dir_pts(c, r=f.get("r", 5)).items(): A["rim_" + nm] = p
        A["rim"] = A["rim_s"]
    elif k == "bowl":
        c = np.asarray(f["at"], float); A["center"] = c; r = f.get("r", 40)
        for nm, p in _dir_pts(c, r=r).items(): A["ring_" + nm] = p
        a = ang(f.get("gap", "S")) if "gap" in f else -math.pi / 2; u = np.array([math.cos(a), math.sin(a)])
        A["gap"], A["outside"], A["rim"] = c + u * r, c + u * (r + f.get("w", 20) * 1.3), c - u * r
    return {f"{i}.{n}": np.asarray(p, float) for n, p in A.items()}


def path_of(f):
    k = f["kind"]
    if k in WORKS: from components.site import terrain_works as TW; return TW.path_of(f)
    if k not in ("ridge", "valley", "river", "road", "cliff", "rampart", "moat", "trench"): return None
    return resample(chaikin(f["path"], 2 if k != "road" else 3), 2.0)[0]


# ---------------------------------------------------------------- site assembly
def _resolve(v, A, paths):
    """Plan point: [x, y] | "<id>.<anchor>" | "<id>@fraction" (a point on a path)."""
    if isinstance(v, str):
        if "@" in v:
            nm, t = v.split("@"); Q = paths[nm]; return Q[min(len(Q) - 1, int(round(float(t) * (len(Q) - 1))))][:2]
        return np.asarray(A[v][:2], float) if not isinstance(A[v], str) else _resolve(A[v], A, paths)
    return np.asarray(v, float)


# ---------------------------------------------------------------- site presets: knobs, scatters of shapes, seed → its own site
PRESETS = os.path.join(core.paths.ROOT, "library", "terrain", "presets")


def _knob(v, K):
    """A string "{knob}" or "{knob}*2" → the knob's value (simple arithmetic); anything else is left as is (lists and dicts — recursively)."""
    if isinstance(v, str) and "{" in v:
        e = v
        for k_, x in K.items(): e = e.replace("{" + k_ + "}", repr(x))
        try: return eval(e, {"__builtins__": {}}, {"min": min, "max": max, "round": round, "int": int})
        except Exception: return v
    if isinstance(v, list): return [_knob(x, K) for x in v]
    if isinstance(v, dict): return {k: _knob(x, K) for k, x in v.items()}
    return v


def generate(g, rng, K, taken):
    """Scatter of shapes: {gen: kind, n: [a, b], area: {ring: [r0, r1], at} | {box: [x0, y0, x1, y1]}, gap, keep: [x, y, r] (kept free),
    params: {name: [min, max] | value | [options, …] — for choice}} → a list of shapes with random places and sizes."""
    n = g["n"] if isinstance(g["n"], int) else int(rng.integers(g["n"][0], g["n"][1] + 1)); out = []; A = g.get("area", {"ring": [60, 150]})
    for i in range(400):
        if len(out) >= n: break
        if "ring" in A:
            c = np.asarray(A.get("at", [0, 0]), float); r = math.sqrt(rng.uniform(A["ring"][0] ** 2, A["ring"][1] ** 2)); a = rng.uniform(0, 2 * math.pi)
            p = c + r * np.array([math.cos(a), math.sin(a)])
        else: b = A["box"]; p = np.array([rng.uniform(b[0], b[2]), rng.uniform(b[1], b[3])])
        if any(np.hypot(*(p - np.asarray(q[:2]))) < q[2] for q in g.get("keep", [])): continue
        prm = {}
        num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)   # true/false in YAML are not range numbers
        for k, v in (g.get("params") or {}).items():
            if isinstance(v, list) and k.endswith("_choice"): prm[k[:-7]] = v[int(rng.integers(len(v)))]
            elif isinstance(v, list) and len(v) == 2 and all(num(x) for x in v): prm[k] = float(rng.uniform(min(v), max(v)))
            else: prm[k] = v
        size_ = float(prm.get("r", prm.get("w", 10)))
        if any(np.hypot(*(p - q[:2])) < (q[2] + size_) * g.get("gap", 0.8) for q in taken): continue
        f = {"kind": g["gen"], "id": f"{g.get('id', g['gen'])}{i}", "at": p.round(1).tolist(), "_gen": True, **prm}
        if g["gen"] in ("ridge", "cliff", "valley", "rampart", "moat", "trench", "river", "road"):   # linear ones — a segment with a random heading
            L = float(prm.get("length", 60)); a = float(prm.get("course", rng.uniform(0, 180))); d = np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
            f["path"] = [(p - d * L / 2).round(1).tolist(), (p + d * L / 2).round(1).tolist()]; f.pop("at"); f.pop("length", None); f.pop("course", None)
        out.append(f); taken.append((p[0], p[1], size_))
    return out


def preset(name, seed=1, **over):
    """Site preset (library/terrain/presets/<name>.yaml) → plan: knobs are drawn from ranges by the seed (or over), scatters are
    expanded, the biome and tiles come from the preset; the relief seed is the same. One preset, different seeds — different
    sites of one concept."""
    import yaml
    P = yaml.safe_load(open(name if name.endswith(".yaml") else os.path.join(PRESETS, name + ".yaml"))); rng = np.random.default_rng(seed)
    used = json.dumps({k: v for k, v in P.items() if k != "knobs"}, ensure_ascii=False)
    dead = [k for k in P.get("knobs") or {} if "{" + k + "}" not in used]   # a knob that nothing substitutes lies in the catalog
    if dead: raise ValueError(f"preset {P.get('name', name)}: knobs {dead} are not substituted anywhere")
    K = {}
    for k, v in (P.get("knobs") or {}).items():
        if k in over: K[k] = over[k]
        elif isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in v): K[k] = int(rng.integers(min(v), max(v) + 1))
        elif isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v): K[k] = round(float(rng.uniform(min(v), max(v))), 2)
        elif isinstance(v, list): K[k] = v[int(rng.integers(len(v)))]
        else: K[k] = v
    R = _knob(dict(P["relief"]), K); R["seed"] = int(seed) * 7 + 3; taken = []
    feats = list(R.get("features") or [])
    for g in R.pop("generate", []) or []: feats += generate(g, rng, K, taken)
    LMK = ("arch", "spire", "overhang", "megaliths")                 # landmarks from the scatter go into landmarks (meshes, not the height map)
    R["landmarks"] = list(R.get("landmarks") or []) + [f for f in feats if f["kind"] in LMK]; feats = [f for f in feats if f["kind"] not in LMK]
    R["features"] = feats; R["biome"] = P.get("biome", R.get("biome")); R["preset"] = {"name": P.get("name", name), "seed": int(seed), "knobs": K}
    if P.get("views") and not R.get("views"): R["views"] = _knob(P["views"], K)   # the preset's cameras — for its stories
    if over.get("base"):                                            # base: base=real (the type from the preset) | <type> | {real, pick, erosion}
        b_ = over["base"]; R["base"] = {"real": "auto" if b_ in ("real", "auto") else b_} if isinstance(b_, str) else dict(b_)
    return {"relief": R, "views": R.get("views"), "title": P.get("title"), "cases": P.get("cases")}


def build(plan, verbose=False, fast=False, _pre=False):
    """Plan (dict) → site: near-plot and belt grids, masks, anchors, paths, checks. Returns a dict for save().
    fast — without erosion (a pass only for the anchors and building places — the first pass in layout.py); _pre — a preliminary
    pass for laying roads (roads with route are ghosts)."""
    t0 = time.time(); R = plan.get("relief", plan)
    if R.get("preset") and isinstance(R["preset"], str):            # relief: {preset: name, seed, knobs…} — a preset from the library; shapes,
        P_ = preset(R["preset"], R.get("seed", 1), **{k: v for k, v in R.items() if k not in ("preset", "seed", "features", "size", "res", "views")})["relief"]
        P_["features"] = list(P_.get("features") or []) + list(R.get("features") or [])   # added by the plan (building platforms, footholds) go on top
        for k_ in ("size", "res", "views"):
            if k_ in R: P_[k_] = R[k_]
        R = P_
    if (R.get("base") or {}).get("real"):                          # real base: the window is chosen, archetypes — "find or make"
        from components.site import terrain_fit as TF
        R = TF.prepare(R)
    size, res, seed = float(R.get("size", 300)), float(R.get("res", 0.5)), int(R.get("seed", 7))
    cx, cy = map(float, R.get("center", [0, 0])); far = R.get("far", {}) or {}
    F, alias = [], {}
    for u in R.get("use", []) or []:
        f_, a_ = archetype(u); alias.update(a_)
        if u.get("found"): from components.site import terrain_fit as TF; f_ = TF.ghost(u, f_)
        F += f_
    F += [dict(f, _plan=True) for f in R.get("features", []) or []]
    routed = [f for f in F if f["kind"] == "road" and f.get("route")]
    if routed and _pre:
        for f in routed: f["ghost"] = True
    elif routed:                                                    # pathfinding roads — on the ground of the preliminary pass
        from components.site import terrain_fit as TF
        pre = build(R, fast=True, _pre=True); wet_ = np.isfinite(pre["level"]) & (np.nan_to_num(pre["level"], nan=-1e9) > pre["near"].H)
        for f in routed:
            f["path"], f["route_info"] = TF.route(pre["near"], f["path"], f.get("grade", 0.12), water=wet_); f["wiggle"] = 0
            f["grade"] = f["route_info"].get("limit", f.get("grade", 0.12))
    pads_ = [f for f in F if f["kind"] == "pad"]
    for f in F:                                                     # a road to a platform runs 4 m into it: the end lies under the platform, with no cut
        if f["kind"] == "road" and pads_:
            P_ = np.asarray(f["path"], float)
            for e, nb in ((-1, -2), (0, 1)):
                for pf in pads_:
                    sd_ = float(sdf_poly(P_[[e], 0], P_[[e], 1], footprint(pf))[0])
                    if sd_ <= 3.0:                                  # toward the platform center, 3 m past its edge
                        cc_ = np.asarray(pf.get("at", footprint(pf).mean(0)), float); d_ = cc_ - P_[e]
                        q_ = P_[e] + d_ / max(np.linalg.norm(d_), 1e-9) * (max(sd_, 0) + 3.0)
                        P_ = np.vstack([P_, q_]) if e else np.vstack([q_, P_])
                        break
            f["path"] = P_.tolist()
    for j, f in enumerate(F):
        f.setdefault("id", f"{f['kind']}{j}")
        if "path" in f and f["kind"] in WIGGLE and f.get("wiggle", 1) and not f.get("closed"):   # natural sinuosity; wiggle: 0 — ruler-straight
            a, lam = WIGGLE[f["kind"]]; ref = {"river": f.get("w", 10), "valley": f.get("w", 30), "ridge": f.get("w", 40),
                                             "cliff": f.get("drop", 10)}.get(f["kind"], 1.0)
            fixed = f["kind"] in ("road", "trench", "rampart", "moat")      # for these the amplitude and wavelength are in meters, not fractions
            amp = (a * ref if not fixed else a) * float(f.get("wiggle", 1)); L_ = (lam * ref if not fixed else lam)
            if f.get("wig"): amp, L_ = map(float, f["wig"])        # amplitude and wavelength — explicit, m (gullies, channels, floodplain rivers)
            f["path"] = wiggle(f["path"], amp + (2.0 if f["kind"] == "cliff" else 0), L_ + (20 if f["kind"] == "cliff" else 0),
                               seed + (zlib.crc32(f["id"].encode()) & 0xFFF)).tolist()
    hills_ = [n_ for n_ in (hill_law(f, R.get("biome") or "") for f in F if f["kind"] == "hill" and not f.get("ghost")) if n_]   # not small mountains
    # path ends given by anchors: from/to → path (straight through via)
    n = int(round(size / res)) + 1
    near = Grid(cx - size / 2, cy - size / 2, res, n, n)
    FS = float(far.get("size", max(3000.0, size * 6))) if far.get("kind", "ridged") != "none" else size * 1.6
    fres = float(far.get("res", max(4.0, FS / 640)))
    fn = int(round(FS / fres)) + 1
    farg = Grid(cx - FS / 2, cy - FS / 2, fres, fn, fn)
    base, det = R.get("base", {}) or {}, R.get("detail", {}) or {}
    cache, timing = {"@pads": [f for f in F if f["kind"] == "pad"]}, {}

    def ground_terms(g, near_):
        X, Y = np.meshgrid(g.x0 + np.arange(g.nx) * g.res, g.y0 + np.arange(g.ny) * g.res)
        q = np.maximum(np.abs(X - cx), np.abs(Y - cy)) / (size / 2)          # 1 — the edge of the near plot
        fade_far = 1 - smoothstep(0.86 * FS / size, 0.99 * FS / size, q)      # toward the belt edge — the far_z level
        bs = 1.0 / float(base.get("noise_scale", 90.0 if not base.get("map") else 60.0))
        zm, wm = map_term(X, Y, base, cx, cy) if base.get("map") or base.get("map_real") else (0.0, 0.0)   # the base is a game map or a real one
        z = float(base.get("amp", 1.2 if not base.get("map") else 0.0)) * fbm(X * bs, Y * bs, 4, seed) * fade_far + zm * fade_far
        if far.get("kind", "ridged") != "none":                     # mountain belt: from the edge of the near plot to the horizon
            ring = smoothstep(float(far.get("start", 1.4)), float(far.get("peak", 3.5)), q)   # in fractions of half the plot
            if far.get("kind") not in ("real", "mosaic"): ring = ring * fade_far   # real mountains do not fade out at the edge — the far terrain continues beyond (far_heights)
            sides = far.get("sides")
            if sides:
                th = np.arctan2(Y - cy, X - cx); wsec = np.zeros_like(th)
                for sd_ in sides: wsec = np.maximum(wsec, np.cos(np.clip((th - ang(sd_) + np.pi) % (2 * np.pi) - np.pi, -np.pi / 2, np.pi / 2)) ** 2)
                ring = ring * wsec
            op = far.get("open")                                    # open side (sea to the horizon): the belt parts toward the anchor
            p_ = alias.get(op, op) if isinstance(op, str) else op
            if op and not isinstance(p_, str):
                dth = np.abs((np.arctan2(Y - cy, X - cx) - math.atan2(float(p_[1]) - cy, float(p_[0]) - cx) + np.pi) % (2 * np.pi) - np.pi)
                w_ = math.radians(float(far.get("open_w", 50.0))); ring = ring * smoothstep(w_, w_ + math.radians(35.0), dth)
            fs = 1.0 / float(far.get("scale", 700.0)); kind = far.get("kind", "ridged"); h = float(far.get("h", 140.0))
            if kind == "real":                                      # real mountains of the window (terrain_ranges): meters above the valley floor, nature's own shape
                from components.site import terrain_ranges as TB
                return z + TB.heights(far, X - cx, Y - cy, seed, FS / 2) * ring * (1 - wm), q
            if kind == "mosaic":                                    # a mosaic of massifs from windows of different types (terrain_ranges.mosaic_grid)
                from components.site import terrain_ranges as TB
                return z + TB.mosaic(far, X - cx, Y - cy, int(far.get("seed", seed)), FS / 2, fres) * ring * (1 - wm), q
            m = ridged(X * fs, Y * fs, 6, seed + 101) if kind == "ridged" else (fbm(X * fs, Y * fs, 6, seed + 101) * 0.5 + 0.5)
            if kind == "mesa": m = terrace(m, 0.25, 0.25)
            z = z + h * m * ring * (1 - wm)                         # mountains — where there is no map
        return z, q

    def detail(g, ctx):                                             # fine noise — after the shapes, before the cuts and platforms; fades toward the plot edge
        if not det.get("amp", 0.25): return
        ds = 1.0 / float(det.get("scale", 7.0)); X, Y = np.meshgrid(g.x0 + np.arange(g.nx) * g.res, g.y0 + np.arange(g.ny) * g.res)
        g.H = g.H + float(det.get("amp", 0.25)) * fbm(X * ds, Y * ds, 3, seed + 7) * ctx.fades(X, Y)[0]

    er = R.get("erosion") or {}; amt = 0.0 if fast else float(er.get("amount", 1.0)); ED = None
    if amt > 0:                                                     # large-scale erosion — one for both grids: the belt before the cuts → droplets → the difference;
        from components.site import terrain_erosion as TE                                # the valleys match at the seam, toward the belt edge the difference fades
        t1 = time.time(); g2 = Grid(farg.x0, farg.y0, farg.res, farg.nx, farg.ny); z2_, q2_ = ground_terms(g2, False); g2.H = z2_.copy()
        c2 = Ctx(g2, False, {}, cx, cy, size / 2, FS, fres)
        for kk in ORDER[:ORDER.index("@erode")]:
            if kk == "@detail": detail(g2, c2)
            else:
                for f in F:
                    if f["kind"] == kk: apply(f, c2, seed)
        H_pre = g2.H.copy(); TE.erode_ms(g2.H, g2.res, amt, float(er.get("far_droplets", 0.6)), seed + 17, 2, 40.0, 20)
        X2, Y2 = np.meshgrid(g2.x0 + np.arange(g2.nx) * g2.res, g2.y0 + np.arange(g2.ny) * g2.res)
        ED = Grid(farg.x0, farg.y0, farg.res, farg.nx, farg.ny); timing["erosion_belt"] = round(time.time() - t1, 2)
        from core.surface.terrain_analysis import blur              # the belt grid step is 4.7 m — no steps on the near grid
        ED.H = blur((g2.H - H_pre) * c2.fades(X2, Y2)[1] * gully_room(H_pre, g2.res), 1.0)   # flat floors and small mounds are not cut: the water from the mountains
        # of the belt cut the plot floor with a network of gullies up to 25 m, with sharp crests between them
        if far.get("kind") in ("real", "mosaic"):                   # real mountains are already eroded by nature: erosion only near the site (otherwise scree
            ED.H = ED.H * (1 - smoothstep(float(far.get("start", 1.4)), float(far.get("peak", 3.5)), q2_)) # up to 40° eats away the walls)
    for g, near_ in ((near, True), (farg, False)):                  # near first: river levels, road profiles, platform heights go into the cache
        t1 = time.time()
        z, q = ground_terms(g, near_); g.H = z.copy(); ctx = Ctx(g, near_, cache, cx, cy, size / 2, FS, fres)
        for kk in ORDER:
            if kk == "@detail": detail(g, ctx)
            elif kk == "@erode" and ED is not None:                 # erosion: the large part is shared with the belt; the fine part — near plot only, fading toward the edge
                if near_:
                    X, Y = np.meshgrid(g.x0 + np.arange(g.nx) * g.res, g.y0 + np.arange(g.ny) * g.res)
                    g.H = g.H + ED.sample(X, Y); H0_ = g.H.copy()
                    TE.erode_ms(g.H, g.res, amt, float(er.get("droplets", 0.4)), seed + 13, int(er.get("macro", 4)), float(er.get("talus", 38)),
                                int(er.get("thermal", 30)))
                    g.H = H0_ + (g.H - H0_) * ctx.fades(X, Y)[0] * gully_room(H0_, g.res)   # a star of gullies on a small mound — a "small mountain"
                else: g.H = g.H + ED.H
            elif kk == "@strata" and near_ and (R.get("strata") or {}).get("step", 2.5):   # strata on steep ground: ledges, edges cut up by noise
                st = R.get("strata") or {}; step_ = float(st.get("step", 2.5))
                X, Y = np.meshgrid(g.x0 + np.arange(g.nx) * g.res, g.y0 + np.arange(g.ny) * g.res)
                gy_, gx_ = np.gradient(g.H, g.res); slp = np.degrees(np.arctan(np.hypot(gx_, gy_)))
                if st.get("model", "rock") == "rock":                   # rock: strata of different thickness, division, blocky facets
                    g.H = g.H + cliff_detail(g.H, X, Y, g.res, seed, st) * ctx.fades(X, Y)[0]
                else:                                                   # the previous even ledges (strata: {model: terrace})
                    m_ = smoothstep(float(st.get("from", 45)), float(st.get("full", 55)), slp) * (1 - smoothstep(62, 72, slp))
                    wob = 0.35 * step_ * fbm(X / 9.0, Y / 9.0, 3, seed + 21)
                    Hs = terrace(g.H + wob, step_, float(st.get("sharp", 0.45))) - wob
                    g.H = g.H + (Hs - g.H) * m_ * ctx.fades(X, Y)[0]
            elif kk == "@rock" and near_ and det.get("rock", 0.45):    # stone on steep ground — after the cuts (gorge walls), not by the water
                X, Y = np.meshgrid(g.x0 + np.arange(g.nx) * g.res, g.y0 + np.arange(g.ny) * g.res)
                gy_, gx_ = np.gradient(g.H, g.res); slp = np.degrees(np.arctan(np.hypot(gx_, gy_)))
                dry = 1 - np.where(np.isnan(ctx.level), 0.0, smoothstep(2.0, 0.5, g.H - np.nan_to_num(ctx.level)))
                g.H = g.H + float(det.get("rock", 0.45)) * smoothstep(30, 45, slp) * (ridged(X / 5.0, Y / 5.0, 3, seed + 9) - 0.5) * 2 * dry * ctx.fades(X, Y)[0]
            elif kk == "@pads":                                     # platforms — once more: a hard requirement, on top of roads
                for f in F:
                    if f["kind"] != "pad": continue
                    w_ = g.window(_bbox_pts(footprint(f), 1))
                    if w_:
                        sl, X, Y = w_; sd = sdf_poly(X, Y, footprint(f)); g.H[sl] = np.where(sd <= 0, cache[f["id"] + ":z"], g.H[sl])
            else:
                for f in F:
                    if f["kind"] == kk: apply(f, ctx, seed)
        timing["near" if near_ else "far"] = round(time.time() - t1, 2)
        if near_: nctx = ctx
    # anchors and paths (z from the finished grid)
    A = {}
    for f in F: A.update(anchors_of(f, cache))
    paths = {f["id"]: path_of(f) for f in F if path_of(f) is not None}
    for nm, v in alias.items():
        try: A[nm] = _resolve(v, A, paths)
        except Exception as e: print(f"anchor {nm}: {e}", file=sys.stderr)
    for nm, v in (R.get("anchors") or {}).items(): A[nm] = _resolve(v, A, paths)
    zat = lambda x, y: float(sample2(near, farg, np.array([x]), np.array([y]))[0])
    anchors = {nm: [round(float(p[0]), 2), round(float(p[1]), 2), round(zat(p[0], p[1]), 2)] for nm, p in A.items()}
    level = nctx.level
    P3 = {}
    for nm, Q in paths.items():
        f = next(x for x in F if x["id"] == nm); z = sample2(near, farg, Q[:, 0], Q[:, 1])
        if f["kind"] == "river" and nm + ":level" in cache: sq, zl, _ = cache[nm + ":level"]; z = np.interp(cumlen(Q), sq, zl)   # the river path follows the water level
        P3[nm] = {"kind": f["kind"], "w": f.get("w"), "pts": np.round(np.column_stack([Q, z]), 2).tolist()}
    loc = {"size": size, "res": res, "seed": seed, "center": [cx, cy], "far": {"size": FS, "res": fres, **{k: v for k, v in far.items() if k != "size"}},
           "far_z": 0.0, "features": [{"id": f["id"], "kind": f["kind"]} for f in F], "anchors": anchors, "paths": P3,
           "ground": R.get("ground", "grass"), "env": R.get("env"), "biome": R.get("biome"), "preset": R.get("preset") if isinstance(R.get("preset"), dict) else None,
           "public": sorted(a for a in (set(alias) | set(R.get("anchors") or {}) | {a for a in anchors if a.split(".")[0] in {f["id"] for f in F if f.get("_plan") and not f.get("_gen")}})
                            if a in anchors and near.bounds()[0] <= anchors[a][0] <= near.bounds()[2] and near.bounds()[1] <= anchors[a][1] <= near.bounds()[3])}
    if (R.get("base") or {}).get("found"): loc["base"] = R["base"]["found"]      # library window; what was found, what was made
    rt = {f["id"]: f["route_info"] for f in F if f.get("route_info")}
    if rt: loc["routes"] = rt                                         # pathfinding roads: length, ratio to the straight line, slope limit
    FH = None
    if far.get("kind") in ("real", "mosaic"):                       # height of the belt mountains (p99 beyond the mountain ring) — the default snow line
        Xf, Yf = np.meshgrid(farg.x0 + np.arange(farg.nx) * fres, farg.y0 + np.arange(farg.ny) * fres)
        loc["far_top"] = round(float(np.percentile(farg.H[np.maximum(np.abs(Xf - cx), np.abs(Yf - cy)) > float(far.get("peak", 3.5)) * size / 2], 99)), 1)
        from components.site import terrain_ranges as TB              # the far terrain beyond the belt comes from the same place (a wide window): the Light component takes it
        FH = TB.far_heights(far, seed, FS / 2) if far.get("kind") == "real" else TB.far_mosaic(far, int(far.get("seed", seed)), FS / 2)   # mosaic — far terrain as a mosaic
        if FH: FH["x0"] += cx; FH["y0"] += cy; loc["far_heights"] = {"file": "far_heights.npz", "res": round(FH["res"], 2), "half": round(-FH["x0"] + cx, 1)}
    loc["strata"] = dict(R.get("strata") or {})                     # the site's stack of strata — used to carve the rock and color the cliffs
    for k_ in ("snowline", "bright", "tiles", "tone_pull", "bright_k", "fluid", "forest", "stones"):   # site look: snow, brightness, tiles, fluid, forest, stones
        if k_ in R: loc[k_] = R[k_]
    LM = [dict(l_, id=l_.get("id", f"{l_['kind']}{j}")) for j, l_ in enumerate(R.get("landmarks") or [])]   # landmarks — as meshes in Blender
    if LM:
        loc["landmarks"] = LM
        for l_ in LM:
            x_, y_ = l_["at"]; loc["anchors"][l_["id"] + ".base"] = [round(x_, 2), round(y_, 2), round(zat(x_, y_), 2)]
            if l_["kind"] == "arch": loc["anchors"][l_["id"] + ".under"] = loc["anchors"][l_["id"] + ".base"]
        loc["public"] = sorted(set(loc["public"]) | {l_["id"] + ".base" for l_ in LM})
    BRG = [dict(cache[f["id"] + ":bridge"], id=f["id"] + "_bridge") for f in F if f["kind"] == "road" and f["id"] + ":bridge" in cache]
    if BRG:                                                         # bridges — as meshes in Blender (bl_relief.build_bridges); the anchor is the middle of the deck
        loc["bridges"] = BRG
        for b_ in BRG: loc["anchors"][b_["id"] + ".deck"] = b_["path"][len(b_["path"]) // 2]
        loc["public"] = sorted(set(loc["public"]) | {b_["id"] + ".deck" for b_ in BRG})
    ST = settle_of(R, anchors)
    if ST: loc["settle"] = ST                                       # settlement land: where people plow and graze (dressing, cover)
    out = {"near": near, "far": farg, "level": level.astype(np.float32), "road": nctx.road, "pad": nctx.pad, "pave": nctx.pave, "cliff": nctx.cliff, "loc": loc,
           "F": F, "cache": cache, "works": getattr(nctx, "works", None), "far_heights": FH}
    loc["checks"] = checks(out)
    if hills_: loc["checks"].append({"check": "free hills are not mountains (hill_law)", "ok": True, "fixed": hills_})
    loc["stats"] = stats(out)
    timing["total"] = round(time.time() - t0, 2); loc["s"] = timing
    return out


_MAPS = {}


def map_term(X, Y, b, cx, cy):
    """Site base — the height grid of a map (a real terrain window, map_real): the window center goes to the plot center,
    the center height is 0; at the window edge the base fades out over 40 m. Returns (z, map weight)."""
    if b.get("map_real"):                                           # real base — a library window (terrain_fit, terrain_library)
        from components.site import terrain_fit as TF
        return TF.real_term(X, Y, b, cx, cy)
    raise ValueError("site base: only a real terrain window (map_real) is supported")


def sample2(near, farg, x, y):
    """Height: the near grid inside the plot, the belt outside."""
    x, y = np.asarray(x, float), np.asarray(y, float); b = near.bounds()
    m = (x >= b[0]) & (x <= b[2]) & (y >= b[1]) & (y <= b[3])
    z = farg.sample(x, y)
    if m.any():                                                     # blend band at the edge — as in Ground.height (BAND = 12 m)
        d = np.minimum(np.minimum(x - b[0], b[2] - x), np.minimum(y - b[1], b[3] - y)); w = smoothstep(0, 12.0, d)
        z = np.where(m, near.sample(x, y) * w + z * (1 - w), z)
    return z


# ---------------------------------------------------------------- checks and summary
def checks(o):
    near, F, cache, loc = o["near"], o["F"], o["cache"], o["loc"]; out = []
    gy, gx = np.gradient(near.H, near.res); slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    for f in F:
        if f.get("ghost"): continue                                 # the base gave the shape — nothing to check
        if f["kind"] == "pad":
            w_ = near.window(_bbox_pts(footprint(f), 0))
            if not w_: continue
            sl, X, Y = w_; m = sdf_poly(X, Y, footprint(f)) <= -near.res          # without the rim
            if not m.any(): continue
            dev = float(np.abs(near.H[sl][m] - cache[f["id"] + ":z"]).max()); sm = float(slope[sl][m].max())
            Pp = footprint(f); c_ = Pp.mean(0); ring = c_ + (Pp - c_) * (1 + 2.0 / max(np.linalg.norm(Pp - c_, axis=1).mean(), 1))   # ring 2 m beyond the edge
            drop = float(np.abs(near.sample(ring[:, 0], ring[:, 1]) - cache[f["id"] + ":z"]).max())
            rr = float(np.linalg.norm(Pp - c_, axis=1).mean())        # a brow at a large platform on a dome is natural: the tolerance grows with the radius
            out.append({"check": f"platform {f['id']}", "ok": dev <= 0.05 and sm <= 2.0 and (drop <= max(3.0, 0.18 * rr) or f.get("edge") == "wall"), "deviation_m": round(dev, 3), "slope_max": round(sm, 1),
                        "edge_drop_m": round(drop, 1)})
        elif f["kind"] == "road":
            Q = resample(chaikin(f["path"], 3), 1.0)[0]; b = near.bounds()
            Q = Q[(Q[:, 0] > b[0] + 2) & (Q[:, 0] < b[2] - 2) & (Q[:, 1] > b[1] + 2) & (Q[:, 1] < b[3] - 2)]
            if len(Q) < 3: continue
            z = near.sample(Q[:, 0], Q[:, 1]); Bd = cache.get(f["id"] + ":bridge")
            if Bd:                                                  # bridge span — measured along the deck, not the channel under it
                D = np.asarray(Bd["path"], float); dd_, sd_, _, ex_, _ = line_info(Q[:, 0], Q[:, 1], D[:, :2])   # along the deck polyline (at a bend
                Ld = np.r_[0, np.cumsum(np.linalg.norm(np.diff(D[:, :2], axis=0), axis=1))]                    # the chord leads down to the channel bed under the bridge)
                on = (dd_ < 1.0) & (ex_ <= 0); z = np.where(on, np.interp(sd_, Ld, D[:, 2]), z)
            gr = np.abs(np.diff(z)) / np.maximum(np.hypot(*np.diff(Q, axis=0).T), 1e-6)
            lv = near.sample(Q[:, 0], Q[:, 1], np.nan_to_num(o["level"], nan=-1e9)); wet = float(np.mean(lv > z + 0.05))
            g90 = float(np.percentile(gr, 95))
            thru = sum(int((sdf_poly(Q[:, 0], Q[:, 1], footprint(pf)) < -1.0).sum()) for pf in F if pf["kind"] == "pad")
            out.append({"check": f"road {f['id']}", "ok": g90 <= f.get("grade", 0.12) + 0.03 and float(gr.max()) < 0.5, "slope_p95": round(g90, 3),
                        "slope_max": round(float(gr.max()), 3), "limit": f.get("grade", 0.12), "under_water": round(wet, 2), "through_pad_m": thru})
        elif f["kind"] == "river":
            sq, zl, Q = cache[f["id"] + ":level"]; mono = bool(np.all(np.diff(zl) <= 1e-6))
            P = resample(chaikin(f["path"], 2), 2.0)[0]; b = near.bounds()
            P = P[(P[:, 0] > b[0]) & (P[:, 0] < b[2]) & (P[:, 1] > b[1]) & (P[:, 1] < b[3])]
            dep = near.sample(P[:, 0], P[:, 1], np.nan_to_num(o["level"], nan=-1e9)) - near.sample(P[:, 0], P[:, 1]) if len(P) else np.array([0.0])
            out.append({"check": f"river {f['id']}", "ok": mono and float(np.median(dep)) > 0.3, "level_monotonic": mono,
                        "depth_median": round(float(np.median(dep)), 2), "level_drop": round(float(zl[0] - zl[-1]), 2)})
    b = near.bounds(); xs = np.linspace(b[0], b[2], 200)                   # seam between the near plot and the belt
    edge = np.concatenate([np.stack([xs, np.full_like(xs, b[1])], 1), np.stack([xs, np.full_like(xs, b[3])], 1),
                           np.stack([np.full_like(xs, b[0]), xs], 1), np.stack([np.full_like(xs, b[2]), xs], 1)])
    gap = np.abs(near.sample(edge[:, 0], edge[:, 1]) - o["far"].sample(edge[:, 0], edge[:, 1]))
    out.append({"check": "seam with the belt", "ok": float(np.percentile(gap, 95)) < 6.0, "gap_p95_m": round(float(np.percentile(gap, 95)), 2),
                "max_m": round(float(gap.max()), 2), "note": "smoothed by a 12 m band at the edge of the plot"})
    fb = o["far"].bounds(); fe = o["far"].H[[0, -1], :].ravel().tolist() + o["far"].H[:, [0, -1]].ravel().tolist()
    if loc.get("far_heights"):                                      # real mountains: the belt edge continues the far terrain from the same window
        out.append({"check": "belt edge continues the far terrain", "ok": bool(o.get("far_heights")), "max_m": round(float(np.max(np.abs(fe))), 1)})
    else: out.append({"check": "belt edge at far_z", "ok": float(np.max(np.abs(fe))) < 0.05, "max_m": round(float(np.max(np.abs(fe))), 3)})
    for nm, (x, y, z) in loc["anchors"].items():                   # plan anchors (not path ends beyond the plot) — inside the plot
        if nm in loc.get("public", []) and not nm.endswith((".start", ".end", ".source", ".mouth")) and not (b[0] <= x <= b[2] and b[1] <= y <= b[3]): out.append({"check": f"anchor {nm}", "ok": False, "outside": [x, y]})
    return out


def stats(o):
    near = o["near"]; H = near.H; gy, gx = np.gradient(H, near.res); sl = np.degrees(np.arctan(np.hypot(gx, gy)))
    dep = np.nan_to_num(o["level"], nan=-1e9) - H
    return {"z_min": round(float(H.min()), 1), "z_max": round(float(H.max()), 1), "steep_35": round(float((sl > 35).mean()), 3),
            "steep_15": round(float((sl > 15).mean()), 3), "water_share": round(float((dep > 0.05).mean()), 3), "road_share": round(float((o["road"] > 0).mean()), 3),
            "far_max": round(float(o["far"].H.max()), 1), "nodes": int(H.size + o["far"].H.size)}


# ---------------------------------------------------------------- site cameras (location shooting)
FMT = {"16:9": (1536, 864), "9:16": (864, 1536), "1:1": (1024, 1024), "2:1": (1536, 768), "4:3": (1280, 960)}


def cam_clear(cams, G, low=15.0):
    """Camera spot for dressing: for a low camera (below low m above the ground) — a 7 m patch at the camera and along the view
    direction at 9 and 18 m (r 5), so that a trunk or a bush does not end up in the lens. High cameras look over things — they do not need it."""
    out = []
    for c in cams or []:
        M = np.asarray(c["matrix_world"], float); e = M[:3, 3]; v = -M[:3, 2][:2]; n_ = np.linalg.norm(v)
        if e[2] - float(G.height(e[0], e[1])) > low: continue
        out.append((float(e[0]), float(e[1]), 7.0))
        if n_ > 1e-6: out += [(float(e[0] + v[0] / n_ * t), float(e[1] + v[1] / n_ * t), 5.0) for t in (9.0, 18.0)]
    return out


def look_at(eye, target, lens=35.0, fmt="16:9", name="cam"):
    """Write cameras.json: the camera at eye looks at target (Blender: view along −Z, up along +Y)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float); z = eye - target; z /= max(np.linalg.norm(z), 1e-9)
    x = np.cross([0.0, 0.0, 1.0], z); x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0, 0]); y = np.cross(z, x)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, eye
    return {"name": name, "lens_mm": float(lens), "sensor_width_mm": 36.0, "sensor_height_mm": 24.0, "sensor_fit": "AUTO", "shift_x": 0.0, "shift_y": 0.0,
            "resolution": list(FMT.get(fmt, FMT["16:9"])), "matrix_world": M.round(6).tolist()}


def shot_depths(G, e, t, lens=35, fmt="16:9", nx=15, ny=9, dmax=1500.0):
    """Frame depths by rays over the relief (support for frame scoring): an nx × ny grid of rays in the field of view; no hit — sky (inf)."""
    e, t = np.asarray(e, float), np.asarray(t, float); f = t - e; f /= np.linalg.norm(f) + 1e-9
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r) + 1e-9; u = np.cross(r, f)
    a, b = map(float, fmt.split(":")); hx = 36.0 / (2 * lens); hy = hx * b / a
    U, V = np.meshgrid(np.linspace(-hx, hx, nx), np.linspace(-hy, hy, ny)); D = f[None, None] + U[..., None] * r + V[..., None] * u
    D = (D / np.linalg.norm(D, axis=-1, keepdims=True)).reshape(-1, 3); dep = np.full(len(D), np.inf); s = np.r_[np.arange(1.0, 60, 1.0), np.geomspace(60, dmax, 120)]
    hx_ = np.zeros((len(D), 2))
    for d in s:                                                     # step: 1 m up close, then increasing further out
        P = e + D * d; hit = np.isinf(dep) & (P[:, 2] <= G.top(P[:, 0], P[:, 1]))
        dep[hit] = d; hx_[hit] = P[hit, :2]
        if not np.isinf(dep).any(): break
    nb = G.near_bounds(); ins = np.isfinite(dep) & (hx_[:, 0] >= nb[0]) & (hx_[:, 0] <= nb[2]) & (hx_[:, 1] >= nb[1]) & (hx_[:, 1] <= nb[3])
    return np.where(ins, dep, -dep)                                 # < 0 — a hit outside the plot (mountain belt)


def shot_score(dep):
    """Frame score: depth spread and near depth count in favor; a full-frame wall and sky alone count against."""
    hit = np.abs(dep[np.isfinite(dep)]); sky = 1 - len(hit) / len(dep); place = float((np.isfinite(dep) & (dep > 0)).mean())
    if not len(hit): return -100.0, sky, 0.0, 0.0
    med = float(np.median(hit)); sc = float(np.std(np.log(hit))) * 20 + 1.5 * min(float(hit.min()), 20.0) - 60 * max(0.0, 0.04 - sky) - 60 * max(0.0, sky - 0.6)
    sc -= 80 * max(0.0, 0.3 - place)                                # the frame shows the site: hits on the plot itself, not on the mountain belt
    return sc, sky, med, place


def views(G, plan):
    """Site cameras: plan views [{name, from, look, h, look_h, lens, fmt}] (points are anchors "x.ford", "path@0.3" or [x, y(, z)]);
    without them — wide and wide_2 (from outside, from above), from_below (from a low anchor to a high one), along (along the first
    path), overhead (at 55°). The camera is not in the ground (≥ 1 m above it) and the target is visible — otherwise the camera is raised."""
    R = plan.get("relief", plan); nb = G.near_bounds(); c = np.array([(nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2]); S = nb[2] - nb[0]
    pt = lambda v: G.anchor(v) if isinstance(v, str) else np.array([v[0], v[1], v[2] if len(v) > 2 else float(G.height(v[0], v[1]))], float)
    vs = R.get("views") or []
    if not vs:
        pub = [a for a in G.loc.get("public", []) if a in G.loc["anchors"]]
        zs = {a: G.loc["anchors"][a][2] for a in pub}
        A_ = np.array([G.loc["anchors"][a][:2] for a in pub]) if pub else np.array([c])
        cc = A_.mean(0) if len(A_) else c; ext = float(np.clip(np.percentile(np.hypot(*(A_ - cc).T), 90) if len(A_) > 1 else 0.25 * S, 35, 0.45 * S))
        D = 1.7 * ext + 30                                          # wide — by the site anchors: what the site is for, large in frame
        vs = [{"name": "wide", "from": [cc[0] - 0.55 * D, cc[1] - 0.83 * D], "h": 0.42 * D, "look": [cc[0], cc[1]], "look_h": 4, "lens": 28},
              {"name": "wide_2", "from": [cc[0] + 0.75 * D, cc[1] + 0.62 * D], "h": 0.2 * D, "look": [cc[0], cc[1]], "look_h": 4, "lens": 30}]
        if len(zs) >= 2:
            lo, hi = min(zs, key=zs.get), max(zs, key=zs.get)
            if zs[hi] - zs[lo] > 4: vs.append({"name": "from_below", "from": lo, "h": 1.6, "look": hi, "look_h": 2.0, "lens": 30, "back": 8})
        P = [n for n, p in G.loc["paths"].items() if p["kind"] in ("road", "valley", "river")]
        if P: vs.append({"name": "along", "from": f"{P[0]}@0.12", "h": 1.7, "look": f"{P[0]}@0.5", "look_h": 1.5, "lens": 30})
        vs.append({"name": "overhead", "from": [c[0], c[1] - 0.32 * S], "h": 0.55 * S, "look": [c[0], c[1]], "look_h": 0, "lens": 30})
    out = []
    for i, v in enumerate(vs):
        try: e, t = pt(v["from"]).copy(), pt(v["look"]).copy()
        except (KeyError, IndexError) as e_:                        # no anchor (the scatter did not produce one) — skip the camera
            print(f"camera {v.get('name')}: no point {e_}", file=sys.stderr); continue
        t[2] += v.get("look_h", 1.5)
        if v.get("back"):                                           # step back from the target: both the slope and the summit in the frame
            d = e[:2] - t[:2]; e[:2] += d / max(np.linalg.norm(d), 1e-9) * v["back"]; e[2] = float(G.top(e[0], e[1]))
        e[2] += v.get("h", 1.7)
        e[2] = max(e[2], float(G.top(e[0], e[1])) + 1.0)
        if v.get("walk") and not G.visible(e, t, clear=0.2):        # like a person: walk up to the brow (or step away) while the target is not visible — do not fly up
            d_ = t[:2] - e[:2]; L_ = float(np.linalg.norm(d_)); u_ = d_ / max(L_, 1e-9); h_ = float(v.get("h", 1.7))
            for s_ in [*np.arange(3.0, max(L_ - 5.0, 3.0), 3.0), *(-np.arange(3.0, 150.0, 3.0))]:
                q = e[:2] + u_ * s_; c2 = np.array([q[0], q[1], float(G.top(q[0], q[1])) + h_])
                if G.visible(c2, t, clear=0.2): e = c2; break
        for _ in range(30):                                         # is the target visible?
            if G.visible(e, t, clear=0.2): break
            e[2] += 1.0
        lens, fmt = v.get("lens", 35), v.get("fmt", "16:9"); sc, sky, med, place = shot_score(shot_depths(G, e, t, lens, fmt))
        if (med < 15 and sky < 0.03) or place < 0.25 or not G.visible(e, t, clear=0.2):   # wall, off the site, target occluded — alternatives
            best = (sc, e.copy()); d0 = e[:2] - t[:2]; L0 = max(float(np.linalg.norm(d0)), 1.0)
            for dh, back, rot in ((8, 0, 0), (20, 0, 0), (0, 30, 0), (6, 60, 0), (4, 0, 30), (4, 0, -30), (6, 0, 60), (6, 0, -60), (12, 40, 45), (12, 40, -45),
                                  (25, -40, 0), (35, -80, 0), (30, -60, 90), (30, -60, -90), (30, -60, 180)):   # closer and higher, from other sides
                a_ = math.atan2(d0[1], d0[0]) + math.radians(rot); q = t[:2] + np.array([math.cos(a_), math.sin(a_)]) * (L0 + back)
                c2 = np.array([q[0], q[1], max(e[2] - float(G.top(e[0], e[1])), 1.0) + float(G.top(q[0], q[1])) + dh])
                if not G.visible(c2, t, clear=0.2): continue
                s2 = shot_score(shot_depths(G, c2, t, lens, fmt))[0]
                if s2 > best[0]: best = (s2, c2)
            Rr = min(L0, 0.45 * S)                                  # orbit around a target inside the plot: 8 sides, height by distance
            for k_ in range(8):
                a_ = math.atan2(d0[1], d0[0]) + k_ * math.pi / 4; q = t[:2] + np.array([math.cos(a_), math.sin(a_)]) * Rr
                c2 = np.array([q[0], q[1], max(float(G.top(q[0], q[1])), t[2]) + max(2.0, 0.35 * Rr)])
                if not G.visible(c2, t, clear=0.2): continue
                s2 = shot_score(shot_depths(G, c2, t, lens, fmt))[0]
                if s2 > best[0]: best = (s2, c2)
            e = best[1]
        out.append(look_at(e, t, lens, fmt, v.get("name", f"cam{i + 1}")))
    return out


# ---------------------------------------------------------------- files
def data_maps(o):
    """Site data maps (analysis of the finished near grid, 1 m step): drainage, brows and feet of cliffs, flat spots, shapes,
    distances to water and roads — for asset placement, texture, frames; a summary of what was found — in location.json → analysis."""
    from core.surface import terrain_analysis as TA
    n = o["near"]; out, mp = TA.analyze(n.H, n.res, n.x0, n.y0); k = max(1, int(round(mp["step"] / n.res)))
    wet = np.nan_to_num(o["level"], nan=-1e9)[::k, ::k] > n.H[::k, ::k]; rd = o["road"][::k, ::k] > 0.5
    o["loc"]["analysis"] = {key: out[key] for key in ("flow", "saddles", "peaks", "bowls", "flats", "cliff_share", "brow_share", "forms", "slope_p50_p90")}
    A = {"an_step": np.array([mp["step"]]), "an_area": mp["area"].astype(np.float32), "an_brow": mp["brow"].astype(np.uint8),
         "an_foot": mp["foot"].astype(np.uint8), "an_flat": mp["flat"].astype(np.int16), "an_forms": mp["forms"].astype(np.int8),
         "an_dwater": np.round(TA.dist(wet, mp["step"])).astype(np.uint8), "an_droad": np.round(TA.dist(rd, mp["step"])).astype(np.uint8)}
    if o.get("works") is not None:                                  # traces of human works (terrain_works): rampart, moat, mound, quarry, spoil heap, dam
        A["an_works"] = (o["works"][::k, ::k][:A["an_area"].shape[0], :A["an_area"].shape[1]] > 0.5).astype(np.uint8)
    ST = o["loc"].get("settle")
    if ST:                                                          # cultivation 0–1: arable land and pasture around a settlement (Chisholm 1962)
        sh = A["an_area"].shape; st = float(mp["step"]); X, Y = np.meshgrid(n.x0 + np.arange(sh[1]) * st, n.y0 + np.arange(sh[0]) * st)
        A["an_use"] = np.round(255 * np.max([use_of(np.hypot(X - s["at"][0], Y - s["at"][1]) - s["r"], s.get("km", 1.0)) for s in ST], 0)).astype(np.uint8)
    return A


SETTLE_FALL = (0.6, 1.5)                                            # fractions of the "near kilometer": up to the first — fully cultivated, by the second — fades out


def use_of(d, km=1.0):
    """Land cultivation at distance d m from the edge of a settlement: arable land and pasture within the near kilometer, farther out
    the benefit falls (Chisholm 1962, "Rural Settlement and Land Use"; site catchment — Vita-Finzi, Higgs 1970). km is the
    cultivation radius in km (a city — larger, a hamlet — smaller)."""
    return 1 - smoothstep(SETTLE_FALL[0] * 1000 * km, SETTLE_FALL[1] * 1000 * km, np.maximum(d, 0))


def settle_of(R, anchors):
    """Settlements of the site: an explicit relief.settle list ([{at: anchor | [x, y], r: m, km}]) or the archetypes' settlements
    (terrain_works.SETTLE)."""
    from components.site import terrain_works as TW
    raw = list(R.get("settle") or [])
    if not raw:
        for u in R.get("use", []) or []:
            S_ = TW.SETTLE.get(u.get("type"), {})
            for nm, r in S_.get(u.get("kind", ""), S_.get("", [])): raw.append({"at": f"{u['id']}.{nm}", "r": r})
    out = []
    for s in raw:
        at = s["at"]; p = anchors.get(at) if isinstance(at, str) else at
        if p is not None: out.append({"at": [round(float(p[0]), 1), round(float(p[1]), 1)], "r": float(s.get("r", 60)), "km": float(s.get("km", 1.0)),
                                      **({"anchor": at} if isinstance(at, str) else {})})
    return out


def save(o, folder):
    os.makedirs(folder, exist_ok=True); n, f = o["near"], o["far"]
    try: an = data_maps(o)
    except Exception as e: an = {}; print("maps-data:", e, file=sys.stderr)
    np.savez_compressed(os.path.join(folder, "relief.npz"), near=n.H.astype(np.float32), near_o=np.array([n.x0, n.y0, n.res]),
                        far=f.H.astype(np.float32), far_o=np.array([f.x0, f.y0, f.res]), level=o["level"].astype(np.float32),
                        road=(o["road"] * 255).astype(np.uint8), pad=(o["pad"] * 255).astype(np.uint8), cliff=(o["cliff"] * 255).astype(np.uint8),
                        pave=(o.get("pave", o["pad"] * 0) * 255).astype(np.uint8), **an)
    if o.get("far_heights"):                                        # distance beyond the belt (real mountains): heights above the valley floor, NaN — no data
        FH = o["far_heights"]; np.savez_compressed(os.path.join(folder, "far_heights.npz"), z=FH["z"], o=np.array([FH["x0"], FH["y0"], FH["res"]]))
    json.dump(o["loc"], open(os.path.join(folder, "location.json"), "w"), ensure_ascii=False, indent=1)


def finish_scene(SD, folder):
    """Scene after the bridge: the kind of site parts (water, rocks — they have no graph role), the far-ground level far_z,
    relief.npz and location.json next to the scene — queries to the site by the scene folder (Ground(scenes/<name>))."""
    import shutil
    W = json.load(open(os.path.join(SD, "world.json")))
    for e in W["reg"]:
        if e["id"].startswith("water"): e["kind"] = "water"
        if e["id"].startswith("rocks"): e["kind"] = "rock"
        if e["id"].startswith("trees"): e["kind"] = "tree"
        if e["id"].startswith("landmarks"): e["kind"] = "rock"
        if e["id"].startswith("bridges"): e["kind"] = "wall"            # like buildings (role structure → wall)
    W["far_z"] = float(json.load(open(os.path.join(folder, "location.json"))).get("far_z", 0.0)); json.dump(W, open(os.path.join(SD, "world.json"), "w"), ensure_ascii=False)
    for f in ("relief.npz", "location.json", "sheet.png", "relief_w.npz", "surface.json"):   # layer maps and the ground recipe — for the lit frame
        if os.path.exists(os.path.join(folder, f)): shutil.copy(os.path.join(folder, f), os.path.join(SD, f))


def scene(plan_p, name, plan=None, label=None, bridge=True):
    """Plan (a file or a ready dict) → site → .blend (bl_relief.py, cameras by anchors, kinds) → bridge → scenes/<name>/ (water as
    the water kind, far_z, all cameras, relief.npz and location.json alongside — queries to the site by the scene folder).
    bridge=False — no bridge (catalog)."""
    import shutil, subprocess, yaml
    from components.site import terrain_sheet; from core.surface.ground import Ground
    root = core.paths.ROOT; plan = plan or yaml.safe_load(open(plan_p)); t0 = time.time()
    folder = os.path.join(root, "out", "relief", name); o = build(plan); o["loc"]["plan"] = label or os.path.relpath(plan_p, root); save(o, folder)
    G = Ground(folder)
    from components.site import terrain_tex; TX = None if (plan.get("relief", plan).get("texture") is False) else terrain_tex.bake(folder)   # game tiles by biome
    terrain_sheet.sheet(folder, os.path.join(folder, "sheet.png"))   # the sheet comes after the texture: colour as in 3D
    t1 = time.time()
    R_ = plan.get("relief", plan); extra, sq = [], []
    blend = os.path.join(root, "out", "build", name, name + ".blend"); os.makedirs(os.path.dirname(blend), exist_ok=True)
    cams = views(G, plan)
    json.dump(cams, open(os.path.join(folder, "cameras.json"), "w"), ensure_ascii=False, indent=1)
    try:                                                            # ground surfaces (Eevee shader: photoscans) — the biome recipe
        from components.site import terrain_clutter as TC
        Sv = yaml.safe_load(open(os.path.join(root, "library", "terrain", "surfaces.yaml"))); bio = G.loc.get("biome") or "temperate"
        sets = {k: {**v["default"], **(v.get("biomes", {}).get(bio) or {})} for k, v in Sv["sets"].items()}
        json.dump({"recipe": sets[Sv.get("use", "scan")], "sets": sets, "seed": int(R_.get("seed", 7)), "biome": bio,
                   "masks": {**Sv.get("masks", {}).get("default", {}), **(Sv.get("masks", {}).get(bio) or {})}, "clutter": TC.resolve(bio)},
                  open(os.path.join(folder, "surface.json"), "w"), ensure_ascii=False)
    except Exception as e: print("surfaces:", e, file=sys.stderr)
    from components.site import terrain_dress as TD                                      # dressing: site maps → density by ecotope → a pattern with a trace
    places, dst = TD.plan(G, seed=int(R_.get("seed", 7)) + 11, avoid=[(p_[0], p_[1], 2.5) for q in (sq or []) for p_ in q["pts"]] + cam_clear(cams, G))
    json.dump({"biome": G.loc.get("biome"), "by_layer": dst, "places": places}, open(os.path.join(folder, "dress.json"), "w"), ensure_ascii=False)
    cmd = ["blender", "-b", "--factory-startup"]
    r = subprocess.run(cmd + ["-P", core.paths.RUN, "--", os.path.join(root, "components/site/bl_relief.py"), "--", folder, blend,
                        "cams=" + os.path.join(folder, "cameras.json"), "shots=" + os.path.join(folder, "shots")] + extra, capture_output=True, text=True)
    rb = next((json.loads(l[10:]) for l in r.stdout.splitlines() if l.startswith("__RESULT__")), None)
    if rb is None: sys.exit("bl_relief: " + (r.stdout + r.stderr)[-1500:])
    t2 = time.time(); SD = core.paths.scene_dir(name, create=True)
    if bridge:
        r = subprocess.run(["blender", "-b", "--factory-startup", "-P", core.paths.RUN, "--", os.path.join(root, "core", "blender", "bl_export_scene.py"), "--", blend, SD], capture_output=True, text=True)
        if not os.path.exists(os.path.join(SD, "world.json")): sys.exit("bridge: " + (r.stdout + r.stderr)[-1500:])
        finish_scene(SD, folder); json.dump(cams, open(os.path.join(SD, "cameras.json"), "w"), ensure_ascii=False, indent=1)
    return {"scene": os.path.relpath(SD, root) if bridge else None, "folder": os.path.relpath(folder, root), "blend": os.path.relpath(blend, root),
            "cams": [c["name"] for c in cams], "shots": rb.get("shots"), "preset": o["loc"].get("preset"),
            "terrain": {k: rb[k] for k in ("verts", "lines", "tex", "tex_src", "s_mesh", "s_tex", "rocks", "s_dress", "water_verts") if k in rb},
            **({"texture": TX} if TX else {}),
            "checks_failed": [c for c in o["loc"]["checks"] if not c["ok"]],
            "s": {"site": round(t1 - t0, 1), "blender": round(t2 - t1, 1), "bridge": round(time.time() - t2, 1)}}


def export(folder, out=None):
    """Site → files for simulations and third-party programs (game engines, 3D software, your own computations): 16-bit PNG heights
    (near plot and belt; z = z_min + v / 65535 · (z_max − z_min)), 8-bit PNG masks (water — depth 0–2 m, road, platform, cliff,
    slope 0–90°, passable), export.json — origin, step, sizes, ranges, far_z, anchors and paths. PNG rows run from the north
    (the top of the image is north)."""
    from PIL import Image
    from core.surface.ground import Ground
    G = Ground(folder); out = out or os.path.join(folder, "export"); os.makedirs(out, exist_ok=True); meta = {"far_z": G.far_z, "grids": {}}
    for nm, H, x0, y0, r in (("near", G.Hn, G.nx0, G.ny0, G.nres), ("far", G.Hf, G.fx0, G.fy0, G.fres)):
        lo, hi = float(H.min()), float(H.max()); v = np.round((H - lo) / max(hi - lo, 1e-9) * 65535).astype(np.uint16)[::-1]
        Image.fromarray(v).save(os.path.join(out, f"height_{nm}.png"))
        meta["grids"][nm] = {"file": f"height_{nm}.png", "origin_xy": [float(x0), float(y0)], "res_m": float(r), "size_px": [H.shape[1], H.shape[0]],
                             "z_min": round(lo, 3), "z_max": round(hi, 3)}
    gy, gx = np.gradient(G.Hn, G.nres); slope = np.degrees(np.arctan(np.hypot(gx, gy))); dep = np.clip(np.nan_to_num(G.level, nan=-1e9) - G.Hn, 0, 2)
    masks = {"water_depth": dep / 2, "road": G.masks["road"], "pad": G.masks["pad"], "cliff": G.masks["cliff"], "slope": slope / 90,
             "walkable": ((slope <= 30) & (dep <= 0.5)).astype(float)}
    for nm, A in masks.items(): Image.fromarray((np.clip(A, 0, 1) * 255).astype(np.uint8)[::-1]).save(os.path.join(out, f"mask_{nm}.png"))
    meta["masks"] = {nm: f"mask_{nm}.png" for nm in masks}; meta["masks_note"] = "water_depth: ×2 m; slope: ×90°; the rest — fraction 0..1"
    meta["anchors"], meta["paths"] = G.loc["anchors"], G.loc["paths"]
    json.dump(meta, open(os.path.join(out, "export.json"), "w"), ensure_ascii=False, indent=1)
    return {"out": out, "files": sorted(os.listdir(out))}


def main():
    import yaml
    if len(sys.argv) >= 3 and sys.argv[1] == "export":
        print("__RESULT__" + json.dumps(export(os.path.abspath(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None), ensure_ascii=False)); return
    if len(sys.argv) >= 4 and sys.argv[1] == "scene":
        print("__RESULT__" + json.dumps(scene(os.path.abspath(sys.argv[2]), sys.argv[3]), ensure_ascii=False)); return
    if len(sys.argv) >= 3 and sys.argv[1] == "preset":                # preset → scene: preset <name> [seed] [scene] [knob=value …]
        nm = sys.argv[2]; sd = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 1
        rest = [a for a in sys.argv[3:] if not a.isdigit()]; out_ = next((a for a in rest if "=" not in a), f"{nm}_{sd}")
        kn = {k: yaml.safe_load(v) for k, v in (a.split("=", 1) for a in rest if "=" in a)}
        print("__RESULT__" + json.dumps(scene(None, out_, preset(nm, sd, **kn), f"preset {nm} #{sd}"), ensure_ascii=False)); return
    if len(sys.argv) < 3 or sys.argv[1] not in ("build",): sys.exit(__doc__)
    plan_p = os.path.abspath(sys.argv[2]); plan = yaml.safe_load(open(plan_p))
    root = core.paths.ROOT
    folder = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 else os.path.join(root, "out", "relief", os.path.splitext(os.path.basename(plan_p))[0])
    o = build(plan); o["loc"]["plan"] = os.path.relpath(plan_p, root); save(o, folder)
    from components.site import terrain_sheet
    t1 = time.time(); terrain_sheet.sheet(folder, os.path.join(folder, "sheet.png")); o["loc"]["s"]["sheet"] = round(time.time() - t1, 2)
    bad = [c for c in o["loc"]["checks"] if not c["ok"]]
    print("__RESULT__" + json.dumps({"folder": os.path.relpath(folder, root), "anchors": len(o["loc"]["anchors"]), "paths": list(o["loc"]["paths"]),
                                     "checks_failed": bad, "checks": len(o["loc"]["checks"]), "stats": o["loc"]["stats"], "s": o["loc"]["s"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
