"""Mountain belt from a real window (far.kind: real): a USGS 3DEP window of 6 × 6 km, ~4.7 m step (library/terrain/real/belt/<name>.npz:
z — heights, res — step). The plot sits in the window's valley (spot: a floor flat within 250 m, summits in 16 directions at
0.5–1.5 km), the mountains are real at scale k (horizontally and vertically: the same slopes). The mountain shape is nature's
(a valley network, crests, strata ribbons, cirques), not noise. Rotation is by 0/90/180/270° and a mirror: at any other angle the
belt corners fall outside the window.
far keys: window — the window name (none — chosen by seed from all), k — scale (0.55…1; smaller — the window is small), turn — 0…7
(rotation and mirror; none — by seed), at — [row, column] of the floor (none — spot). numpy only: the module is read by both
Blender and .venv.
Mosaic (far.kind: mosaic, mosaic_grid): a belt of massif patches ~1.6 km from different windows by mountain type (TYPES) — kinds
and their mix without a thousand mountains; far keys: types — {type: weight} (type patches ~2 km), seed.
.venv/bin/python -m components.site.terrain_ranges fetch <name> <lat> <lon>  → a window into the library (~5 MB, or ~20 MB for a wide one)
.venv/bin/python -m components.site.terrain_ranges catalog  → the missing windows of the REGIONS catalog"""
import core.paths
import json, os, sys, zlib

import numpy as np

LIB = os.path.join(core.paths.ROOT, "library", "terrain", "real", "belt")
_Z = {}


def names():
    return sorted(f[:-4] for f in os.listdir(LIB) if f.endswith(".npz") and not f.endswith("_wide.npz")) if os.path.isdir(LIB) else []


def load(name):
    if name not in _Z:
        d = np.load(os.path.join(LIB, name + ".npz")); _Z[name] = (d["z"].astype(np.float64), float(d["res"]))
    return _Z[name]


def spot(name, margin):
    """Valley floor for the plot: enclosure (the 25th percentile of the summit heights in 16 directions out to 1.5 km above the floor)
    minus the floor roughness (σ within 250 m); the point is no closer than margin m to the window edge. The cache is spots.json
    next to the windows. → (row, column)."""
    key = f"{name}@{int(margin)}"; cp = os.path.join(LIB, "spots.json")
    cache = json.load(open(cp)) if os.path.exists(cp) else {}
    if key in cache: return tuple(cache[key])
    from scipy.ndimage import uniform_filter
    Z, r = load(name); ny, nx = Z.shape; m = int(margin / r) + 2; st = max(1, int(80 / r))
    w = int(2 * 250 / r) + 1; mean = uniform_filter(Z, w); rough = np.sqrt(np.maximum(uniform_filter(Z * Z, w) - mean * mean, 0))
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False); dist = np.arange(500, 1500, 40.0)
    best, bs = None, -1e18
    for i in range(m, ny - m, st):
        for j in range(m, nx - m, st):
            ii = np.clip((i + np.outer(np.sin(ang), dist) / r).astype(int), 0, ny - 1); jj = np.clip((j + np.outer(np.cos(ang), dist) / r).astype(int), 0, nx - 1)
            s = float(np.percentile(Z[ii, jj].max(1) - Z[i, j], 25)) - 4.0 * float(rough[i, j]) - 0.05 * float(Z[i, j] - mean[i, j]) ** 2
            if s > bs: bs, best = s, (i, j)
    if best is None: raise ValueError(f"window {name}: a {margin:.0f} m margin from the edge does not fit — scale k is too small")
    cache[key] = list(best); json.dump(cache, open(cp, "w")); return best


def pick(seed):
    N = names()
    if not N: raise FileNotFoundError(f"no belt windows in {LIB}")
    return N[zlib.crc32(str(seed).encode()) % len(N)]


def heights(far, X, Y, seed, half):
    """Belt heights at points (X, Y — m from the plot middle), in meters above the window's valley floor; half — half the belt side, m."""
    name = far.get("window") or pick(seed); Z, r = load(name); k = float(far.get("k", 0.7))
    i0, j0 = far.get("at") or spot(name, half / k)
    t = int(far.get("turn", zlib.crc32(f"{seed}turn".encode()) % 8))   # 0…3 — rotation by 90°, 4…7 — the same with a mirror
    u, v = (X / k, Y / k) if t < 4 else (-X / k, Y / k)
    for _ in range(t % 4): u, v = -v, u
    ci, cj = i0 + v / r, j0 + u / r; ny, nx = Z.shape
    a, b = np.clip(np.floor(ci).astype(int), 0, ny - 2), np.clip(np.floor(cj).astype(int), 0, nx - 2)
    ti, tj = np.clip(ci - a, 0, 1), np.clip(cj - b, 0, 1)
    h = Z[a, b] * (1 - ti) * (1 - tj) + Z[a, b + 1] * (1 - ti) * tj + Z[a + 1, b] * ti * (1 - tj) + Z[a + 1, b + 1] * ti * tj
    return (h - Z[int(i0), int(j0)]) * k


def fetch(name, lat, lon, half_m=3000.0, n=1280):
    """A window into the library: 2·half_m × 2·half_m m, n × n (~4.7 m step) — USGS 3DEP public domain (terrain_library.fetch);
    and a wide window with the same middle (fetch_wide) — the far terrain to the horizon."""
    from components.site.terrain_library import fetch as F
    os.makedirs(LIB, exist_ok=True); Z = F(lat, lon, half_m, n).astype(np.float32)
    np.savez_compressed(os.path.join(LIB, name + ".npz"), z=Z, res=np.float32(2 * half_m / (n - 1)), lat=lat, lon=lon)
    return {"window": name, "relief": round(float(np.percentile(Z, 99) - np.percentile(Z, 1))), "shape": list(Z.shape), "wide": fetch_wide(name, lat, lon)}


def fetch_wide(name, lat, lon, half_m=45000.0, n=2048):
    """A wide window with the same middle (90 km, ~44 m step) → <name>_wide.npz; no data (across the border, over the sea) — NaN.
    Resampling to the coarse step is done on the server side without the interpolation key: with bilinear it answers 504."""
    import io, math, urllib.request
    from PIL import Image
    from components.site.terrain_library import URL
    dl, dn = half_m / 111320, half_m / (111320 * math.cos(math.radians(lat)))
    q = f"?bbox={lon - dn},{lat - dl},{lon + dn},{lat + dl}&bboxSR=4326&imageSR=4326&size={n},{n}&format=tiff&pixelType=F32&f=image"
    Z = np.array(Image.open(io.BytesIO(urllib.request.urlopen(URL + q, timeout=170).read())), np.float32)[::-1].copy()   # rows run from the south
    Z[~np.isfinite(Z) | (Z < -500)] = np.nan
    np.savez_compressed(os.path.join(LIB, name + "_wide.npz"), z=Z, res=np.float32(2 * half_m / (n - 1)), lat=lat, lon=lon)
    return {"shape": list(Z.shape), "no_data": round(float(np.isnan(Z).mean()), 3)}


def far_heights(far, seed, half, R=32000.0, step=40.0):
    """Far terrain beyond the belt from the same place: the wide window with the same middle in scene axes (the same scale k, rotation,
    valley floor) — a square ±min(R, window edge) with step step m; NaN — no data. → {z, x0, y0, res} (heights — m above the valley
    floor) or None (no wide window)."""
    name = far.get("window") or pick(seed); p = os.path.join(LIB, name + "_wide.npz")
    if not os.path.exists(p): return None
    W = np.load(p); Zw, rw = W["z"].astype(np.float64), float(W["res"]); Z, r = load(name); k = float(far.get("k", 0.7))
    i0, j0 = far.get("at") or spot(name, half / k)
    t = int(far.get("turn", zlib.crc32(f"{seed}turn".encode()) % 8))
    Rs = min(R, (Zw.shape[0] // 2 - 2) * rw * k); n = int(2 * Rs / step) + 1; g = np.linspace(-Rs, Rs, n); X, Y = np.meshgrid(g, g)
    u, v = (X / k, Y / k) if t < 4 else (-X / k, Y / k)
    for _ in range(t % 4): u, v = -v, u
    du, dv = (j0 - (Z.shape[1] - 1) / 2) * r, (i0 - (Z.shape[0] - 1) / 2) * r   # valley floor — measured from the window middle (the window middles are shared)
    ci, cj = (Zw.shape[0] - 1) / 2 + (v + dv) / rw, (Zw.shape[1] - 1) / 2 + (u + du) / rw
    a, b = np.clip(np.floor(ci).astype(int), 0, Zw.shape[0] - 2), np.clip(np.floor(cj).astype(int), 0, Zw.shape[1] - 2)
    ti, tj = np.clip(ci - a, 0, 1), np.clip(cj - b, 0, 1)
    h = Zw[a, b] * (1 - ti) * (1 - tj) + Zw[a, b + 1] * (1 - ti) * tj + Zw[a + 1, b] * ti * (1 - tj) + Zw[a + 1, b + 1] * ti * tj
    return {"z": ((h - Z[int(i0), int(j0)]) * k).astype(np.float32), "x0": -Rs, "y0": -Rs, "res": float(g[1] - g[0])}


REGIONS = {                                                          # belt windows: name → (mountain type, latitude, longitude of the window middle); USGS 3DEP
    "sawtooth": ("granite", 44.105, -114.99), "teton": ("granite", 43.741, -110.802), "wind_river": ("granite", 42.775, -109.215),
    "sierra_evolution": ("granite", 37.18, -118.70), "north_cascades": ("granite", 48.52, -121.07),
    "garden_wall": ("strata", 48.735, -113.76), "maroon_bells": ("strata", 39.07, -106.99), "chinese_wall": ("strata", 47.73, -112.85),
    "capitol_reef": ("mesa", 38.25, -111.22), "canyonlands": ("mesa", 38.40, -109.87), "zion": ("mesa", 37.25, -112.95),
    "rainier": ("volcanic", 46.85, -121.76), "shasta": ("volcanic", 41.41, -122.19),
    "smokies": ("hills", 35.61, -83.49)}
TYPES = {}                                                            # mountain type → windows (rock, shapes)
for _n, (_t, _la, _lo) in REGIONS.items(): TYPES.setdefault(_t, []).append(_n)
MP, MOV, MSC = 340, 120, (0.6, 0.7, 0.8, 0.9)                       # patch 1.6 km, overlap 560 m (belt grid 4.69 m), scales
_M = {}


def _turn(A, t):
    A = np.rot90(A, t % 4)
    return A[:, ::-1] if t >= 4 else A


def _cut(E, axis):
    """The seam of least discrepancy in the overlap strip (Efros, Freeman 2001). → mask: 0 — old, 1 — new."""
    A = E if axis == 1 else E.T; n, w = A.shape; C = A.copy(); B = np.zeros((n, w), np.int64)
    for i in range(1, n):
        pad = np.pad(C[i - 1], 1, constant_values=np.inf); st = np.stack([pad[:-2], pad[1:-1], pad[2:]]); k = st.argmin(0)
        C[i] += st[k, np.arange(w)]; B[i] = np.arange(w) + k - 1
    path = np.empty(n, np.int64); path[-1] = int(C[-1].argmin())
    for i in range(n - 1, 0, -1): path[i - 1] = B[i, path[i]]
    M = (np.arange(w)[None, :] >= path[:, None]).astype(np.float64)
    return M if axis == 1 else M.T


def _blend(A, B, M, sig=(2.0, 6.0, 16.0, 40.0)):
    """Frequency-band blending (Burt, Adelson 1983): coarse content with a wide feather, fine content with a narrow one."""
    from scipy.ndimage import gaussian_filter as gf
    out = np.zeros_like(A); la, lb = A, B
    for s_ in sig:
        ga, gb = gf(la, s_, mode="nearest"), gf(lb, s_, mode="nearest"); m = gf(M, s_ * 0.6, mode="nearest")
        out += (la - ga) * (1 - m) + (lb - gb) * m; la, lb = ga, gb
    m = gf(M, sig[-1], mode="nearest"); return out + la * (1 - m) + lb * m


def mosaic_grid(far, seed, half, res, K=150, D=1000.0):
    """A belt as a mosaic of massifs (a ±half grid with step res, the middle is the site; heights above the valley floor). The regions
    are Voronoi cells of seeds on an irregular grid with spacing D m, the border bent by noise (±180 m): no straight seams. The middle
    is the valley of the main type's window (spot), its region is wider by 400 m. Each region gets a 3.2 km patch of a real window of
    its own type (type patches ~2 km by the far.types weights): out of ~K candidates, the one that agrees best with the neighbors
    already placed within a 200 m strip at the border (after height fitting). Blending — all patches at once by frequency bands
    (Burt, Adelson 1983). → (n × n heights, summary); cached by key."""
    from scipy.ndimage import zoom, gaussian_filter as gf, binary_dilation
    from core.surface.noise import fbm
    types = far.get("types") or {"granite": 1.0}; key = json.dumps([types, seed, half, res], sort_keys=True)
    if key in _M: return _M[key]
    rng = np.random.default_rng(int(seed) * 7919 + 11); n = int(round(2 * half / res)) + 1; c0 = n // 2
    g = (np.arange(n) - c0) * res; X, Y = np.meshgrid(g, g); tt = [t for t in types if types[t] > 0 and t in TYPES]
    F = np.stack([float(types[t]) * np.exp(2.5 * fbm(X / 2000.0 + 7.1 * i, Y / 2000.0 - 3.3 * i, 3, int(seed) + 17 + i)) for i, t in enumerate(tt)])
    W = {t: F[i] / F.sum(0) for i, t in enumerate(tt)}               # type patches ~2 km (fractions by weights)
    src = {}
    for t in tt:
        for nm in TYPES[t]:
            if os.path.exists(os.path.join(LIB, nm + ".npz")) and nm not in src:
                Z, rw = load(nm); src[nm] = {k: zoom(Z, rw * k / res, order=1) * k for k in MSC}
    pts = [(0.0, 0.0)]                                               # seeds: the middle and an irregular grid
    for gx in np.arange(-half, half + 1, D):
        for gy in np.arange(-half, half + 1, D):
            p_ = (gx + rng.uniform(-0.3, 0.3) * D, gy + rng.uniform(-0.3, 0.3) * D)
            if np.hypot(*p_) > 0.75 * D: pts.append(p_)
    wx, wy = 180 * fbm(X / 700.0, Y / 700.0, 3, int(seed) + 31), 180 * fbm(X / 700.0 + 5.5, Y / 700.0 - 2.2, 3, int(seed) + 32)
    dist = np.stack([np.hypot(X + wx - px, Y + wy - py) - (400.0 if i == 0 else 0.0) for i, (px, py) in enumerate(pts)])
    lab = dist.argmin(0); PS = int(3200 / res) // 2 * 2; hp = PS // 2   # 3.2 km patch: the region with a bent edge and the blending feather lie inside
    flat = [t for t in tt if t != "volcanic"]                         # the site's valley is not on a volcano cone (volcano windows are centered on summits:
    main = max(flat, key=lambda t: float(W[t][c0, c0])) if flat else "granite"   # the site sits on the shoulder, the belt edge is 0.7 km lower)
    if main not in tt:
        for nm in TYPES[main]:
            if os.path.exists(os.path.join(LIB, nm + ".npz")) and nm not in src: Z, rw = load(nm); src[nm] = {k: zoom(Z, rw * k / res, order=1) * k for k in MSC}
    cname = next(nm for nm in TYPES[main] if nm in src)
    order = sorted(range(len(pts)), key=lambda i: np.hypot(*pts[i])); pieces, used, inside_ = {}, {}, {}
    canvas = np.zeros((n, n)); placed = np.zeros((n, n), bool); band = int(200 / res)
    gyP, gxP = (np.mgrid[0:PS, 0:PS] - hp) * res; gy16, gx16 = gyP[::16, ::16], gxP[::16, ::16]
    for i in order:
        reg = lab == i
        if not reg.any(): continue
        ci, cj = int(round(pts[i][1] / res)) + c0, int(round(pts[i][0] / res)) + c0
        y0, x0 = ci - hp, cj - hp; sl = (slice(max(y0, 0), min(y0 + PS, n)), slice(max(x0, 0), min(x0 + PS, n)))
        cr = (slice(sl[0].start - y0, sl[0].stop - y0), slice(sl[1].start - x0, sl[1].stop - x0))
        if i == 0:
            Yc, Xc = (np.mgrid[0:PS, 0:PS] - hp) * res + np.array([pts[0][1], pts[0][0]])[:, None, None]
            C, nm = heights({"window": cname, "k": 0.7, "turn": int(seed) % 8}, Xc, Yc, seed, 1500.0), cname
            q_ = C[::16, ::16]; A_ = np.c_[gx16.ravel(), gy16.ravel(), np.ones(gx16.size)]; cf = np.linalg.lstsq(A_, q_.ravel(), rcond=None)[0]
            rr = np.hypot(gxP, gyP); wr = np.clip((rr - 400.0) / 800.0, 0, 1); wr = wr * wr * (3 - 2 * wr)   # tilt removed outside; the site stays flat
            C = C - (cf[0] * gxP + cf[1] * gyP) * wr
        else:
            zone = binary_dilation(reg, iterations=band) & placed; zs = zone[sl]; Hs = canvas[sl]
            wt = np.array([float(W[t][reg].mean()) for t in tt]); t_ = tt[rng.choice(len(tt), p=wt / wt.sum())]
            cand = [m for m in TYPES[t_] if m in src]; best = None
            for _ in range(K):
                nm_ = cand[rng.integers(len(cand))]; Zk = src[nm_][MSC[rng.integers(len(MSC))]]
                a_, b_ = int(rng.integers(0, Zk.shape[0] - PS)), int(rng.integers(0, Zk.shape[1] - PS))
                C_ = _turn(Zk[a_:a_ + PS, b_:b_ + PS], int(rng.integers(8)))
                q_ = C_[::16, ::16]; A_ = np.c_[gx16.ravel(), gy16.ravel(), np.ones(gx16.size)]   # patch tilt (a cone slope over 3 km —
                cf = np.linalg.lstsq(A_, q_.ravel(), rcond=None)[0]; C_ = C_ - cf[0] * gxP - cf[1] * gyP   # up to 2 km) is removed: the shapes remain
                if zs.any(): v_ = C_[cr][zs]; off = float((Hs[zs] - v_).mean()); e = float(((v_ + off - Hs[zs]) ** 2).mean())
                else: off, e = -float(C_[hp, hp]), 0.0
                if best is None or e < best[0]: best = (e, C_ + off, nm_)
            _, C, nm = best
        full = np.pad(C[cr], ((sl[0].start, n - sl[0].stop), (sl[1].start, n - sl[1].stop)), mode="edge")   # outside the patch — its edge (the weight there is 0)
        inside = np.zeros((n, n), bool); inside[sl] = True; inside_[i] = inside
        pieces[i] = full; canvas = np.where(reg, full, canvas); placed |= reg; used[nm] = used.get(nm, 0) + 1
    sig = (2.0, 6.0, 16.0, 40.0, 100.0); out = np.zeros((n, n))   # band 100 (~470 m): the levels of neighboring massifs converge smoothly
    lv = {i: P_ for i, P_ in pieces.items()}
    for s_ in sig + (None,):                                          # bands: fine detail with a narrow feather, coarse detail with a wide one
        wts = {i: gf((lab == i).astype(np.float64), (s_ if s_ else sig[-1]) * 0.8, mode="nearest") * inside_[i] for i in pieces}; tot = sum(wts.values()) + 1e-9
        if s_ is None: out += sum(lv[i] * wts[i] for i in pieces) / tot; break
        nxt = {i: gf(lv[i], s_, mode="nearest") for i in pieces}
        out += sum((lv[i] - nxt[i]) * wts[i] for i in pieces) / tot; lv = nxt
    out = out - out[c0, c0]
    _M[key] = (out, {"pieces": used, "regions": len(pieces), "types": {t: round(float(W[t].mean()), 2) for t in tt}}); return _M[key]


def far_mosaic(far, seed, half, R=32000.0, step=40.0, D=12000.0, tau=1500.0):
    """The far terrain beyond the mosaic belt: Voronoi regions with spacing D m, each with a patch of a wide window of its own type
    (90 km, 44 m step; rotation, scale 0.6–0.9), aligned to the valley level (the 10th percentile of the region is 0); soft blending
    by distance to the seeds (tau m: haze in the distance hides small seams; memory does not grow). → {z, x0, y0, res} (heights —
    m above the site's valley floor, NaN means no data) or None."""
    from core.surface.noise import fbm
    types = far.get("types") or {"granite": 1.0}; tt = [t for t in types if types[t] > 0 and t in TYPES]
    wide = {nm: os.path.join(LIB, nm + "_wide.npz") for t in tt for nm in TYPES[t]}; wide = {k: v for k, v in wide.items() if os.path.exists(v)}
    if not wide: return None
    rng = np.random.default_rng(int(seed) * 104729 + 5); n = int(2 * R / step) + 1; g = np.linspace(-R, R, n); X, Y = np.meshgrid(g, g)
    Wd = {}
    for nm, p in wide.items():
        d = np.load(p); Z = d["z"].astype(np.float64); Wd[nm] = (np.nan_to_num(Z, nan=float(np.nanmin(Z))), float(d["res"]))
    pts = [(gx + rng.uniform(-0.35, 0.35) * D, gy + rng.uniform(-0.35, 0.35) * D) for gx in np.arange(-R, R + 1, D) for gy in np.arange(-R, R + 1, D)]
    wx, wy = 1200 * fbm(X / 6000.0, Y / 6000.0, 3, int(seed) + 61), 1200 * fbm(X / 6000.0 + 4.4, Y / 6000.0 - 1.7, 3, int(seed) + 62)
    Xw, Yw = X + wx, Y + wy; dmin = np.full((n, n), np.inf)
    for px, py in pts: np.minimum(dmin, np.hypot(Xw - px, Yw - py), out=dmin)   # two passes: distances computed on the fly (a list would take ~0.7 GB)
    num, den, used = np.zeros((n, n)), np.zeros((n, n)), {}
    for px, py in pts:
        di = np.hypot(Xw - px, Yw - py); w = np.exp(-(di - dmin) / tau); w[w < 1e-3] = 0.0
        if not w.any(): continue
        tw = np.array([float(types[t]) for t in tt]); t_ = tt[rng.choice(len(tt), p=tw / tw.sum())]
        cand = [m for m in TYPES[t_] if m in Wd]
        if not cand: continue
        nm = cand[rng.integers(len(cand))]; Z, rw = Wd[nm]; k = MSC[rng.integers(len(MSC))]; t = int(rng.integers(8))
        half_w = (Z.shape[0] // 2 - 2) * rw; cx_, cy_ = rng.uniform(-0.4, 0.4, 2) * half_w   # its own spot in the window
        u, v = (X - px) / k, (Y - py) / k
        if t >= 4: u = -u
        for _ in range(t % 4): u, v = -v, u
        ci, cj = (Z.shape[0] - 1) / 2 + (v + cy_) / rw, (Z.shape[1] - 1) / 2 + (u + cx_) / rw
        m = w > 0; a = np.clip(np.floor(ci[m]).astype(int), 0, Z.shape[0] - 2); b = np.clip(np.floor(cj[m]).astype(int), 0, Z.shape[1] - 2)
        ti, tj = np.clip(ci[m] - a, 0, 1), np.clip(cj[m] - b, 0, 1)
        h = (Z[a, b] * (1 - ti) * (1 - tj) + Z[a, b + 1] * (1 - ti) * tj + Z[a + 1, b] * ti * (1 - tj) + Z[a + 1, b + 1] * ti * tj) * k
        own = di[m] <= dmin[m] + 1.0; h = h - float(np.percentile(h[own], 10)) if own.any() else h - float(np.percentile(h, 10))   # valley level
        num[m] += h * w[m]; den[m] += w[m]; used[nm] = used.get(nm, 0) + 1
    return {"z": (num / np.maximum(den, 1e-9)).astype(np.float32), "x0": -R, "y0": -R, "res": float(g[1] - g[0]), "pieces": used}


def mosaic(far, X, Y, seed, half, res):
    """Mosaic heights at points (X, Y — m from the plot middle)."""
    H, _ = mosaic_grid(far, seed, half, res); n = H.shape[0]
    ci, cj = np.clip(Y / res + n // 2, 0, n - 1.001), np.clip(X / res + n // 2, 0, n - 1.001)
    a, b = np.floor(ci).astype(int), np.floor(cj).astype(int); ti, tj = ci - a, cj - b; a1, b1 = np.minimum(a + 1, n - 1), np.minimum(b + 1, n - 1)
    return H[a, b] * (1 - ti) * (1 - tj) + H[a, b1] * (1 - ti) * tj + H[a1, b] * ti * (1 - tj) + H[a1, b1] * ti * tj


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "fetch": print("__RESULT__" + json.dumps(fetch(a[1], float(a[2]), float(a[3])), ensure_ascii=False))
    elif a and a[0] == "wide": print("__RESULT__" + json.dumps(fetch_wide(a[1], float(a[2]), float(a[3])), ensure_ascii=False))
    elif a and a[0] == "catalog":                                    # the missing windows of the REGIONS catalog (small and wide; ~25 MB per window)
        import time
        for nm, (tp, la, lo) in REGIONS.items():
            if os.path.exists(os.path.join(LIB, nm + ".npz")) and os.path.exists(os.path.join(LIB, nm + "_wide.npz")): continue
            try: print(nm, tp, json.dumps(fetch(nm, la, lo), ensure_ascii=False), flush=True)
            except Exception as e: print(nm, "error", str(e)[:80], flush=True)
            time.sleep(3)
    else: print(__doc__)
