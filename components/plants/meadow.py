"""Meadow — where and what (numpy: the project's Python and Blender): tuft points near the scene's cameras, the species and variant, the detail tier, rotation and scale.
Built by bl_meadow.build_meadow (prototypes and instances). Cost: a frame with a meadow near 121 cameras was 112 thousand instances,
68 million triangles; hence the rules:
- a camera is a frame by its lens (the matrix and lens of the camera layer), not a ±43° sector on the ground: a high orbit camera does not sow grass under itself;
- distance is to the eye in 3D: density ρ(d) = ρ₀·min(1, (near/d)^p), beyond reach — the shader's ground colour;
- cameras are merged: a point's density follows the most demanding camera, not the sum (thinning by ρ_max / Σρ);
- tuft tiers by the distance to the nearest camera that sees it: 0 — as is, 1 — three times fewer, wider blades, 2 — ten times
  (in the distance the blades are wider, the density depends on the distance);
- the instance ceiling: beyond it the far ones are thinned, the near ones are not touched.
The site cover (Grass, cover.npz via Ground.cover) — if present, it decides the species: density, group, the group's species in clumps, height, dryness."""
import math

import numpy as np

LOD_D = (10.0, 28.0)                                                    # up to 10 m — tier 0, up to 28 m — 1, farther — 2
LOD_BLADES = (1.0, 0.33, 0.1)                                            # share of a grass's blades at the tier
LOD_WIDTH = (1.0, 1.6, 2.6)                                              # blade width (the same frame coverage)


def camera(c, default_lens=35.0):
    """A layer camera (a dict with matrix_world, lens_mm, sensor_width_mm, resolution) or (x, y, z, heading) — eye, axes, half-angle tangents."""
    if isinstance(c, dict):
        M = np.asarray(c["matrix_world"], float); R = M[:3, :3]; e = M[:3, 3]; lens = float(c.get("lens_mm", default_lens))
        sw = float(c.get("sensor_width_mm", 36.0)); rx, ry = (c.get("resolution") or [16, 9])[:2]
        tx = sw / 2 / lens; ty = tx * ry / rx
        return {"eye": e, "R": R, "tx": tx, "ty": ty, "s": 2 * tx / rx, "name": c.get("name")}   # s — a frame pixel, m per metre of distance
    x, y, z, yaw = (float(v) for v in c[:4]); f = np.array([math.cos(yaw), math.sin(yaw), 0.0]); up = np.array([0, 0, 1.0])
    rgt = np.cross(f, up); R = np.column_stack([rgt, up, -f])               # Blender camera: −z is forward, +y is up
    tx = 36.0 / 2 / default_lens * 1.6; return {"eye": np.array([x, y, z]), "R": R, "tx": tx, "ty": tx * 0.75, "s": 2 * tx / 1280, "name": None}   # the old input: wider


def visible(cam, P, margin=1.12):
    """Whether a point P (n, 3) is visible to the camera: in front of it and in the frame with a margin; → (mask, distance to the eye)."""
    v = (P - cam["eye"]) @ cam["R"]; z = -v[:, 2]; d = np.linalg.norm(P - cam["eye"], axis=1)
    ok = (z > 0.2) & (np.abs(v[:, 0]) <= z * cam["tx"] * margin) & (np.abs(v[:, 1]) <= z * cam["ty"] * margin + 0.6)
    return ok, d


def density(d, rho, near, reach, p=1.6):
    return np.where(d <= reach, rho * np.minimum(1.0, (near / np.maximum(d, 1e-3)) ** p), 0.0) * np.clip(1 - (d - reach * 0.8) / (reach * 0.2), 0, 1)


def points(G, cams, rho=40.0, near=6.0, reach=70.0, seed=1, budget=60000):
    """Meadow points near the cameras: (X, Y, Z, d — the distance to the nearest camera that sees it). A camera sows in its own frame by ρ(d); a point is kept by
    the thinning ρ_max/Σρ of the cameras that see it (density follows the most demanding one); the ceiling is enforced by thinning the far points."""
    r = np.random.default_rng(seed); C = [camera(c) for c in cams]; b = G.near_bounds(); allp = []
    for c in C:
        e = c["eye"]; zg = float(G.height(np.array([e[0]]), np.array([e[1]]))[0]); hgt = max(e[2] - zg, 0.0)
        if hgt >= reach: continue
        fwd = -c["R"][:, 2]; yaw = math.atan2(fwd[1], fwd[0]) if np.hypot(fwd[0], fwd[1]) > 1e-3 else 0.0
        half = min(math.atan(c["tx"] * 1.15) + 0.1, math.pi)                 # the horizontal sector follows the lens (+ a margin)
        rh = math.sqrt(max(reach ** 2 - hgt ** 2, 0.0))
        for d0, d1 in ((0.3, near), (near, near * 2.5), (near * 2.5, near * 6), (near * 6, rh)):
            if d1 <= d0: continue
            dm = math.hypot(d0, hgt); n = int(min(budget, density(np.array([max(dm, 1e-3)]), rho, near, reach)[0] * half * (d1 ** 2 - d0 ** 2)))   # sowing — by the near edge of the strip
            if n <= 0: continue
            u = r.random(n); dd = np.sqrt(d0 ** 2 + u * (d1 ** 2 - d0 ** 2)); a = yaw + r.uniform(-half, half, n)
            X, Y = e[0] + dd * np.cos(a), e[1] + dd * np.sin(a); k = (X > b[0] + 1) & (X < b[2] - 1) & (Y > b[1] + 1) & (Y < b[3] - 1)
            X, Y = X[k], Y[k]
            if not len(X): continue
            P = np.column_stack([X, Y, G.height(X, Y)]); ok, d = visible(c, P)
            rho_c = density(d, rho, near, reach) * ok; dm_ = np.full(len(X), dm)
            keep = r.random(len(X)) < rho_c / np.maximum(density(dm_, rho, near, reach), 1e-9)   # bring down to the point's ρ(d)
            allp.append(P[keep])
    if not allp: return np.zeros((0, 3)), np.zeros(0)
    P = np.vstack(allp); S = np.zeros(len(P)); Mx = np.zeros(len(P)); D = np.full(len(P), np.inf)
    for c in C:                                                         # merging cameras: keep with probability ρ_max / Σρ
        for i0 in range(0, len(P), 200000):
            sl = slice(i0, i0 + 200000); ok, d = visible(c, P[sl]); rc = density(d, rho, near, reach) * ok
            S[sl] += rc; Mx[sl] = np.maximum(Mx[sl], rc); D[sl] = np.where(ok, np.minimum(D[sl], d), D[sl])
    keep = r.random(len(P)) < Mx / np.maximum(S, 1e-9); P, D = P[keep], D[keep]
    if len(P) > budget:                                                 # ceiling: do not touch the near points (< 2·near), thin the far ones
        nearm = D < 2 * near; nf = int((~nearm).sum()); room = max(budget - int(nearm.sum()), 0)
        if nf > room: keep = nearm | (r.random(len(P)) < room / max(nf, 1)); P, D = P[keep], D[keep]
    return P, D


def ppm(cams, P):
    """Frame pixels per metre of height at a point P (n, 3) — for the most demanding camera that sees it (0 — not visible): grass of height h
    takes h · ppm pixels. The yardstick of the grass tiers."""
    out = np.zeros(len(P))
    for c in map(camera, cams):
        ok, d = visible(c, P); out = np.maximum(out, np.where(ok, 1.0 / (c["s"] * np.maximum(d, 1e-3)), 0.0))
    return out


def card_points(G, cams, px=(3.0, 60.0), w_px=24.0, overlap=6.0, w_min=0.4, cell=2.0, seed=1, budget=60000):
    """Card points (the cheap tier of the meadow): grass 3…60 px. Density follows the frame area, not the ground area (Papavasiliou 2015):
    ρ = overlap · cos ι / (width · height), the card width is at least w_px pixels (wider in the distance, the height is real), ι — the viewing angle
    to the ground normal; the number of cards grows as the logarithm of distance (beyond the near grid — on the far belt too, with cells twice as large; the belt's
    cover — Ground.cover). The zone edges dissolve: at 60 px cards give way to tufts
    (1.5·px → px), at 3 px they fade out (2·px → px). Cameras — by the most demanding one (the maximum ρ); where grass is larger than 60 px for even one
    camera — there are no cards (tufts there: a card for a far camera at the feet of a near one would be a huge leaf). The near grid cells are cell m,
    with Poisson sampling inside a cell. → (P (n, 3), D — to the nearest camera that sees it, SD — metres per pixel at the point)."""
    r = np.random.default_rng(seed + 13); b = G.near_bounds(); C = [camera(c) for c in cams]
    xs = np.arange(b[0] + cell / 2, b[2] - cell / 2, cell); ys = np.arange(b[1] + cell / 2, b[3] - cell / 2, cell)
    X, Y = (a.ravel() for a in np.meshgrid(xs, ys)); A = np.full(len(X), cell * cell)
    fb = G.far_bounds() if hasattr(G, "far_bounds") else b; c2 = cell * 2.0     # far belt: larger cells, within the zone's range
    if fb != b and C:                                                    # (1 m grass is taller than px[0] pixels) from the cameras
        reach = max(1.0 / (px[0] * c["s"]) for c in C); E = np.array([c["eye"][:2] for c in C])
        lo = np.maximum(E.min(0) - reach, fb[:2]); hi = np.minimum(E.max(0) + reach, fb[2:])
        if (hi > lo).all():
            Xb, Yb = (a.ravel() for a in np.meshgrid(np.arange(lo[0] + c2 / 2, hi[0], c2), np.arange(lo[1] + c2 / 2, hi[1], c2)))
            o = ~((Xb > b[0]) & (Xb < b[2]) & (Yb > b[1]) & (Yb < b[3]))
            X, Y, A = np.concatenate([X, Xb[o]]), np.concatenate([Y, Yb[o]]), np.concatenate([A, np.full(int(o.sum()), c2 * c2)])
    Z = G.height(X, Y); csz = np.sqrt(A)
    nx = G.height(X - 0.5, Y) - G.height(X + 0.5, Y); ny = G.height(X, Y - 0.5) - G.height(X, Y + 0.5); N_ = np.column_stack([nx, ny, np.ones(len(X))])
    N_ /= np.linalg.norm(N_, axis=1)[:, None]
    hg = np.clip(G.cover("height", X, Y), 0.08, 2.0) if (hasattr(G, "cover_meta") and G.cover_meta()) else np.full(len(X), 0.5)
    best = np.zeros(len(X)); SD = np.zeros(len(X)); D = np.full(len(X), np.inf); P = np.column_stack([X, Y, Z]); free = np.ones(len(X))
    for c in C:
        ok, d = visible(c, P); v = (P - c["eye"]) / np.maximum(d, 1e-3)[:, None]; cos_i = np.abs((v * N_).sum(1)); sd = c["s"] * d; ph = hg / sd
        w = np.clip((1.5 - ph / px[1]) / 0.5, 0, 1) * np.clip((ph - px[0]) / px[0], 0, 1)
        rho = overlap * cos_i / (np.maximum(w_min, w_px * sd) * hg) * w * ok
        up = rho > best; best[up] = rho[up]; SD[up] = sd[up]; D = np.where(ok, np.minimum(D, d), D)
        free = np.minimum(free, np.clip((1.5 - ph / px[1]) / 0.5, 0, 1))   # large for someone — tufts there, not cards (by distance, ignoring
        # the frame: the base is out of frame while the top of the card is in frame)
    n = r.poisson(best * free * A); i = np.repeat(np.arange(len(X)), n)
    if not len(i): return np.zeros((0, 3)), np.zeros(0), np.zeros(0)
    if len(i) > budget: i = i[r.random(len(i)) < budget / len(i)]
    h = csz[i] / 2; Xp = X[i] + r.uniform(-1, 1, len(i)) * h; Yp = Y[i] + r.uniform(-1, 1, len(i)) * h
    return np.column_stack([Xp, Yp, G.height(Xp, Yp)]), D[i], SD[i]


def lod_of(D):
    return np.where(D < LOD_D[0], 0, np.where(D < LOD_D[1], 1, 2))


def lod_spec(sp, lod):
    """A tier specification: a grass — fewer, wider blades (the same frame coverage), fewer spikelets; everything else — as is."""
    if lod == 0 or sp.get("habit") != "grass": return sp
    s = dict(sp); b0, b1 = sp.get("blades", [20, 50]); f = LOD_BLADES[lod]
    s["blades"] = [max(2, int(round(b0 * f))), max(3, int(round(b1 * f)) + 1)]; s["width"] = float(sp.get("width", 0.006)) * LOD_WIDTH[lod]
    s["heads"] = float(sp.get("heads", 0)) * (0.6 if lod == 1 else 0.3); return s


def choose(G, P, D, specs, K, near, reach, seed, canopy=None, ppm=None, px_tuft=None, px_herb=None):
    """Species, variant and tier per point; rotation and scale. → dict: X, Y, Z, sid (species index), k (variant), lod, rot (n, 3), scl (n, 3),
    idx (indices of the points of P that were kept). ppm and px_tuft (the card tier): a tuft smaller than px_tuft pixels gives way to cards — by dissolving
    from 1.5·px_tuft to px_tuft (each tuft has its own threshold); a non-grass — smaller than px_herb (the shape of a flower and a bush reads longer)."""
    from core.surface.noise import fbm
    r = np.random.default_rng(seed + 7); X, Y = P[:, 0], P[:, 1]; n = len(X); sids = list(specs); hab = {s: specs[s][0]["habit"] for s in sids}
    hnom = {s: float(np.mean([v["height"] for v in specs[s][:K]])) for s in sids}; tiers = {s: [v.get("dry_tier") for v in specs[s][:K]] for s in sids}
    CV = G.cover_meta() if hasattr(G, "cover_meta") else None; keep = np.ones(n, bool); s_i = np.zeros(n, int); k_i = r.integers(0, K, n); hz = np.ones(n)
    if CV:                                                              # cover: density → whether there is a tuft; the group by shares; the species — in clumps
        GR = CV["groups"]; keep = r.random(n) < G.cover("density", X, Y)
        SH = np.stack([G.cover(g, X, Y) for g in GR]); SH /= np.maximum(SH.sum(0), 1e-6)
        gi = (r.random(n)[None, :] > SH.cumsum(0)).sum(0).clip(0, len(GR) - 1)
        for g_ in np.unique(gi):
            m_ = gi == g_; tab = {sv: w for sv, w in (CV["species"].get(GR[g_]) or {}).items() if sv in sids} or {sids[0]: 1.0}; names_ = list(tab); Wg = []
            for sv in names_:
                f = fbm(X[m_] / 9.0, Y[m_] / 9.0, 2, seed + 31 * sids.index(sv)) * 0.5 + 0.5
                Wg.append(tab[sv] * (0.15 + 1.7 * np.clip(f - 0.35, 0, 1) ** 1.2 if hab[sv] in ("flower", "herb", "fern") else 0.7 + 0.6 * f) + 1e-6)   # clumps
            Wg = np.array(Wg); Wg /= Wg.sum(0); q = (r.random(int(m_.sum()))[None, :] > Wg.cumsum(0)).sum(0).clip(0, len(names_) - 1)
            s_i[m_] = [sids.index(names_[j]) for j in q]
        dry = G.cover("dryness", X, Y); hg = G.cover("height", X, Y)
        for q in range(n):                                              # grass: the variant by the dryness of the point (fresh, half, dry)
            tv = tiers[sids[s_i[q]]]
            if tv and tv[0] is not None:
                want = int(np.clip(round(dry[q] * 2 + r.normal(0, 0.35)), 0, 2)); k_i[q] = next((j for j, t_ in enumerate(tv) if t_ == want), k_i[q] % len(tv))
        hz = np.clip(hg / np.array([hnom[sids[q]] for q in s_i]), 0.35, 2.0) * r.uniform(0.85, 1.15, n)
    else:                                                               # without cover: the "base" share, moisture, roads, water, slope, canopy shade
        import os
        pw_ = os.path.join(G.folder, "relief_w.npz"); W = np.load(pw_) if os.path.exists(pw_) else None
        wet = G.wet(X, Y); road = G.mask("road", X, Y) + G.mask("pad", X, Y) + G.mask("pave", X, Y); sl = G.steep(X, Y)
        base, mo = np.ones(n), np.full(n, 0.5)
        if W is not None:
            xs, ys = W["xs"], W["ys"]; ii = np.clip(np.searchsorted(ys, Y), 0, len(ys) - 1); jj = np.clip(np.searchsorted(xs, X), 0, len(xs) - 1)
            base = W["base"][ii, jj].astype(float); base = base / (255.0 if base.max() > 1.5 else 1.0)
            if "a_moist" in W.files: mo = W["a_moist"][ii, jj].astype(float); mo = mo / (255.0 if mo.max() > 1.5 else 1.0)
        shade = np.zeros(n); T = np.array([[t["x"], t["y"], t["cr"]] for t in (canopy or []) if t.get("habit") != "shrub" and t.get("h", 0) > 5]) if canopy else np.zeros((0, 3))
        for cx, cy, cr in T:
            m = (np.abs(X - cx) < cr) & (np.abs(Y - cy) < cr)
            if m.any(): shade[m] = np.maximum(shade[m], np.clip(1 - np.hypot(X[m] - cx, Y[m] - cy) / cr, 0, 1))
        pw = np.clip((base - 0.2) / 0.5, 0, 1) * (wet < 0.01) * (road < 0.3) * np.clip((40 - sl) / 10, 0, 1) * np.clip(1.1 - 0.75 * shade, 0, 1)
        keep = r.random(n) < pw; Wk = []
        for n_, s in enumerate(sids):
            sp = specs[s][0]; m0, m1 = sp.get("moist", [0, 1]); w = np.clip((mo - (m0 - 0.15)) / 0.15, 0, 1) * np.clip(((m1 + 0.15) - mo) / 0.15, 0, 1)
            f = fbm(X / 9.0, Y / 9.0, 2, seed + 31 * n_) * 0.5 + 0.5; h = hab[s]
            if h == "flower": w = w * (0.02 + 0.9 * np.clip(f - 0.6, 0, 1) ** 1.3) * (1 - shade) * sp.get("abund", 1.0)   # in clumps, sparsely
            elif h in ("fern", "herb"): w = w * (0.01 + 0.7 * shade + 0.1 * np.clip(f - 0.7, 0, 1))   # in canopy shade; in the open — sparsely
            else: w = w * (0.7 + 0.6 * f) * sp.get("abund", 1.0)
            Wk.append(w + 1e-6)
        Wk = np.array(Wk); Wk /= Wk.sum(0); s_i = (r.random(n)[None, :] > Wk.cumsum(0)).sum(0).clip(0, len(sids) - 1)
        hz = (0.75 + 0.5 * mo) * r.uniform(0.8, 1.2, n)
    k_i = np.minimum(k_i, np.array([len(specs[sids[q]][:K]) - 1 for q in s_i]) if n else k_i)
    if ppm is not None and px_tuft and n:                               # card tier: a tuft that is small in the frame is not drawn as a tuft
        ph = np.array([hnom[sids[q]] for q in s_i]) * hz * ppm; pt = np.array([px_tuft if hab[sids[q]] == "grass" else (px_herb or px_tuft) for q in s_i])
        keep &= r.random(n) < np.clip((ph / pt - 1.0) / 0.5, 0, 1)
    idx = np.flatnonzero(keep)
    X, Y, Z, D, s_i, k_i, hz = X[keep], Y[keep], P[keep, 2], D[keep], s_i[keep], k_i[keep], hz[keep]; m = len(X)
    lod = lod_of(D); grow = np.clip(D / near, 1, 6) ** 0.6 / np.array(LOD_WIDTH)[lod] ** 0.5   # wider in the distance; a coarser tier already has wider blades
    rot = np.column_stack([r.normal(0, 0.06, m), r.normal(0, 0.06, m), r.uniform(0, 2 * math.pi, m)])
    wx = grow * (np.sqrt(hz) if CV else hz); scl = np.column_stack([wx, wx, hz])
    return {"X": X, "Y": Y, "Z": Z - 0.01, "D": D, "sid": s_i, "k": k_i, "lod": lod, "rot": rot, "scl": scl, "sids": sids, "idx": idx, "hnom": hnom}
