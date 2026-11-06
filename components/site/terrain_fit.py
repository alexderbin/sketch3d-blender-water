""""Find or make": a preset's site on a real base (library components/site/terrain_library.py) — the plan adapts to the base.
Window analysis (terrain_analysis) → the archetype takes what was found: crossing — the thalweg near the center, pass — the
saddle and heading, fort — a summit with a flat top, canyon — a valley, arena — a bowl; the shapes the base provides become
"ghosts" (paths and anchors exist, the ground is not sculpted); if nothing is found, the archetype makes the shape itself, as
without a base. Roads with route — by pathfinding (A*, 16 directions, segment slope ≤ the limit, cost — length × (1 + k·slope²));
if that fails, the limit is relaxed (×1.5, ×2), with a mark.
Plan: relief.base: {real: <type> | auto, pick: N} (auto — the type comes from the preset in regions.yaml); the seed picks the window."""
import copy, json, math, os, sys

import numpy as np

from core.surface import terrain_analysis as TA

LANDFORM = {"hill", "plateau", "ridge", "bowl", "cliff", "valley", "noise"}   # scatter shapes that are not needed on a real base
GHOST = {"fort_hill": {"hill"}, "pass": {"ridge"}, "gorge": {"plateau", "ridge", "valley"}, "arena": {"bowl"}, "valley": {"ridge"},
         "cliff_edge": {"cliff"}, "crossing": set()}
_WIN, _SM = {}, {}


def aug_of(seed, tp, size):
    """A window variant by seed: rotation by 90°·k, mirror, extra turn ±12°, shift of the crop (the plot does not leave the window), height scale."""
    import zlib
    r = np.random.default_rng(int(seed) * 7919 + zlib.crc32(str(tp).encode())); rot = float(r.uniform(-12, 12))
    reach = 0.5 * size * (math.cos(math.radians(abs(rot))) + math.sin(math.radians(abs(rot)))); om = max(0.0, 270.0 - reach)
    return {"k90": int(r.integers(4)), "mirror": int(r.integers(2)), "rot": round(rot, 1), "off": [round(float(r.uniform(-om, om)), 1),
            round(float(r.uniform(-om, om)), 1)], "zs": round(float(r.uniform(0.85, 1.2)), 3)}


def window(name, aug=None):
    """A library window (H, m; step; meta), with the variant aug — rotated, mirrored, shifted, with a height scale; meta['edge'] —
    distance to the edge of the original window, m (for fading toward the belt)."""
    key = (name, json.dumps(aug, sort_keys=True) if aug else None)
    if key not in _WIN:
        from components.site import terrain_library as L
        H, res, meta = L.load(name); n = H.shape[0]; c = (n - 1) / 2; meta = dict(meta)
        if aug:
            yy, xx = (np.mgrid[:n, :n] - c) * res; xx = xx - aug["off"][0]; yy = yy - aug["off"][1]
            a = -math.radians(90 * aug["k90"] + aug["rot"]); xs = xx * math.cos(a) - yy * math.sin(a); ys = xx * math.sin(a) + yy * math.cos(a)
            if aug["mirror"]: xs = -xs
            gx, gy = xs / res + c, ys / res + c
            ix, iy = np.clip(gx, 0, n - 1.001), np.clip(gy, 0, n - 1.001); i0, j0 = ix.astype(int), iy.astype(int); fx, fy = ix - i0, iy - j0
            H = (H[j0, i0] * (1 - fx) * (1 - fy) + H[j0, i0 + 1] * fx * (1 - fy) + H[j0 + 1, i0] * (1 - fx) * fy + H[j0 + 1, i0 + 1] * fx * fy) * aug["zs"]
            meta["edge"] = np.minimum(np.minimum(gx, n - 1 - gx), np.minimum(gy, n - 1 - gy)) * res
            meta["zmin"] = meta["zmin"] * aug["zs"]
        else:
            g = np.arange(n); meta["edge"] = np.minimum.outer(np.minimum(g, n - 1 - g), np.minimum(g, n - 1 - g)) * res
        _WIN[key] = (H, res, meta)
    return _WIN[key]


def _smooth(H, res, key, sigma_m=60.0):
    """A window smoothed with a Gaussian (σ, m) — the continuation of the base beyond the window edge."""
    if key not in _SM:
        import cv2
        _SM[key] = cv2.GaussianBlur(H.astype(np.float32), (0, 0), sigma_m / res, borderType=cv2.BORDER_REPLICATE)
    return _SM[key]


def real_term(X, Y, b, cx, cy):
    """Base — a library window (b['map_real']): the window center goes to the plot center, the center height is 0; within 40 m of the
    window edge it blends into the continuation — the smoothed window edge (σ 60 m), extended outward: the sea beyond the edge
    stays sea, a slope stays a slope."""
    H, res, meta = window(b["map_real"], b.get("aug")); n = H.shape[0]; c = (n - 1) / 2
    gx, gy = (X - cx) / res + c, (Y - cy) / res + c
    ix, iy = np.clip(gx, 0, n - 1.001), np.clip(gy, 0, n - 1.001); i0, j0 = ix.astype(int), iy.astype(int); fx, fy = ix - i0, iy - j0
    bil = lambda A: A[j0, i0] * (1 - fx) * (1 - fy) + A[j0, i0 + 1] * fx * (1 - fy) + A[j0 + 1, i0] * (1 - fx) * fy + A[j0 + 1, i0 + 1] * fx * fy
    z = bil(H); k = max(1, int(12 / res)); z0 = float(np.median(H[int(c) - k:int(c) + k + 1, int(c) - k:int(c) + k + 1]))
    inside = (gx >= 0) & (gx <= n - 1) & (gy >= 0) & (gy <= n - 1); edge = np.where(inside, bil(meta["edge"]), 0.0)   # to the edge of the original window
    w = np.clip(edge / 40.0, 0, 1); w = w * w * (3 - 2 * w)
    Hs = _smooth(H, res, (b["map_real"], json.dumps(b.get("aug"), sort_keys=True)))   # (after rotation beyond the window edge — the nearest edge)
    return (z * w + bil(Hs) * (1 - w)) - z0, w


def type_of(preset_name):
    from components.site import terrain_library as L
    for tp, cf in L.config().items():
        if preset_name in (cf.get("presets") or []): return tp
    return None


def prepare(R):
    """A plan with relief.base.real → the window is chosen, the archetypes are fitted to what was found (a copy of the plan). A
    repeated call changes nothing."""
    b = R.get("base") or {}
    if not b.get("real") or b.get("map_real"): return R
    from components.site import terrain_library as L
    R = copy.deepcopy(R); b = R["base"]; tp = b["real"] if b["real"] != "auto" else type_of((R.get("preset") or {}).get("name", ""))
    pick = int(b.get("pick", (R.get("preset") or {}).get("seed", R.get("seed", 1)))); name = L.find(tp, pick) if tp else None
    if not name: b["real_missing"] = tp; return R                      # no window — the site is made as without a base
    if b.get("aug", True) is not False: b["aug"] = b["aug"] if isinstance(b.get("aug"), dict) else aug_of(pick, tp, float(R.get("size", 400)))
    else: b["aug"] = None
    H, res, meta = window(name, b["aug"]); s = 2.0 / res; Hc = H[::int(s), ::int(s)]; rc = res * int(s); c = (np.array(Hc.shape) - 1) / 2
    cx, cy = map(float, R.get("center", [0, 0])); P = lambda ij: [cx + (ij[1] - c[1]) * rc, cy + (ij[0] - c[0]) * rc]
    b.update(map_real=name, amp=0.0); R["detail"] = {"amp": 0.0}; R.setdefault("erosion", {})["amount"] = float(b.get("erosion", 0.0))
    R["features"] = [f for f in R.get("features") or [] if not (f.get("_gen") and f.get("kind") in LANDFORM) and f.get("kind") != "noise"]
    zc = float(np.median(H[H.shape[0] // 2 - 12:H.shape[0] // 2 + 13, H.shape[1] // 2 - 12:H.shape[1] // 2 + 13])); sea = -meta["zmin"] - zc + 0.3
    for f in R["features"]:                                         # plan shapes: the base provides its own — "ghosts" (anchors and paths exist)
        if f.get("kind") in LANDFORM or (f.get("kind") == "river" and not f.get("keep")): f["ghost"] = True
        if f.get("kind") == "lake" and (f.get("sea") or f.get("id") == "sea"):
            if meta["zmin"] <= 1.0:                                 # a window at the sea: the sea at its real level, poured in, with no bowl
                S_ = float(R.get("size", 400)); f.update(fill=True, level=sea, poly=[[cx - S_, cy - S_], [cx + S_, cy - S_], [cx + S_, cy + S_], [cx - S_, cy + S_]], jag=0)
            else: f["ghost"] = True
    pads = [f for f in R["features"] if f.get("kind") == "pad" and f.get("_gen")]
    if pads:                                                        # scatter platforms (houses) go to the nearest flat spots of the base
        fl = TA.flats(Hc, rc, max_slope=7.0, min_area=60.0)[0]; used = []; keep = []
        for f in pads:
            r_ = float(f.get("r", 8)); at = np.asarray(f.get("at", [cx, cy]), float)
            cand = [P(d["ij"]) for d in fl if d["area_m2"] >= 1.3 * math.pi * r_ * r_]
            cand = [q for q in cand if all(math.dist(q, u_) > 2.2 * r_ for u_ in used) and abs(math.dist(q, (cx, cy)) - math.dist(at, (cx, cy))) < 60]
            if cand: q = min(cand, key=lambda q: math.dist(q, at)); f["at"] = q; used.append(q); keep.append(f)
        R["features"] = [f for f in R["features"] if not (f.get("kind") == "pad" and f.get("_gen")) or f in keep]
    found = {"window": name, "type": tp, "variant": b["aug"]}
    for u in R.get("use") or []:
        t = u.get("type"); hit = None
        if t in ("crossing", "gorge", "valley"):                        # a stream near the center: a thalweg from source to mouth
            F = TA.fill(Hc); rec = TA.receivers(F, rc); A = TA.accumulate(F, rec, rc); r = int(40 / rc)
            w = A[int(c[0]) - r:int(c[0]) + r + 1, int(c[1]) - r:int(c[1]) + r + 1]; i, j = np.unravel_index(np.argmax(w), w.shape)
            st = (int(c[0]) - r + i) * Hc.shape[1] + (int(c[1]) - r + j); tr = TA.trunk(A, rec, st, 500.0)[::3]
            Ec = meta["edge"][::int(s), ::int(s)]; ok_ = [Ec[divmod(q, Hc.shape[1])] >= 45.0 for q in tr]   # the path only where the window does not fade
            mid = int(np.argmin([math.dist(divmod(q, Hc.shape[1]), c) for q in tr])); a_, b_ = mid, mid    # (at the edge the base goes over to the belt —
            while a_ > 0 and ok_[a_ - 1]: a_ -= 1                                                       # the river would go underground)
            while b_ < len(tr) - 1 and ok_[b_ + 1]: b_ += 1
            path = [P(divmod(q, Hc.shape[1])) for q in tr[a_:b_ + 1]]
            if len(path) >= 8:
                m = int(np.argmin([math.hypot(p[0] - cx, p[1] - cy) for p in path])); tg = np.subtract(path[min(m + 2, len(path) - 1)], path[max(m - 2, 0)])
                hit = {"path": path, "dir": math.degrees(math.atan2(tg[1], tg[0]))}
                if t == "crossing": u["river"] = path; u["ford_at"] = path[m]; u["face"] = (hit["dir"] + 90) % 360
                elif t == "gorge": u["path"] = path; u["at"] = path[m]; u["face"] = hit["dir"]
                else: u["river_path"] = path; u["face"] = hit["dir"]
        elif t == "pass":
            sd = [d for sc_ in (20.0, 30.0, 50.0) for d in TA.saddles(Hc, rc, sc_, n=4) if math.dist(d["ij"], c) * rc <= 90]
            if sd:
                d = max(sd, key=lambda d: d["drop"]); hit = d; u["at"] = P(d["ij"]); u["face"] = d["road_deg"]
        elif t == "fort_hill":
            pk = [d for d in TA.peaks(Hc, rc, min_rel=8.0) if math.dist(d["ij"], c) * rc <= 90 and d.get("flat_top_m2", 0) >= 150]
            if pk: d = max(pk, key=lambda d: d["drop"] + 0.02 * d["flat_top_m2"]); hit = d; u["at"] = P(d["ij"])
        elif t == "cliff_edge":                                     # cliff edge — the brow nearest to the center, heading from its cells
            M_, brow, foot_ = TA.cliffs(Hc, rc); ys, xs = np.nonzero(brow)
            if len(ys) > 20:
                d = np.hypot(ys - c[0], xs - c[1]) * rc; k0 = int(np.argmin(d))
                if d[k0] <= 120:
                    sel = np.hypot(ys - ys[k0], xs - xs[k0]) * rc < 60; Q = np.column_stack([xs[sel], ys[sel]]).astype(float)
                    w_, v_ = np.linalg.eigh(np.cov((Q - Q.mean(0)).T)); tdir = v_[:, -1]; nrm = np.array([-tdir[1], tdir[0]])
                    i0, j0 = int(ys[k0]), int(xs[k0]); probe = lambda sg: Hc[int(np.clip(i0 + sg * nrm[1] * 15 / rc, 0, Hc.shape[0] - 1)), int(np.clip(j0 + sg * nrm[0] * 15 / rc, 0, Hc.shape[1] - 1))]
                    down = nrm if probe(1) < probe(-1) else -nrm             # where the cliff faces — toward the lower side
                    hit = {"ij": [i0, j0]}; u["at"] = P([i0, j0]); u["face"] = math.degrees(math.atan2(down[1], down[0])) % 360
        elif t == "arena":
            pk = [d for d in TA.peaks(Hc, rc, sign=-1, min_rel=4.0) if math.dist(d["ij"], c) * rc <= 90]
            if pk: d = max(pk, key=lambda d: d["drop"]); hit = d; u["at"] = P(d["ij"])
        if hit is not None: u["found"] = True
        found[u.get("id", t)] = "found" if hit is not None else "made"
    b["found"] = found
    return R


def ghost(u, F):
    """Archetype on what was found: the shapes provided by the base are "ghosts"; roads are found by pathfinding."""
    if not u.get("found"): return F
    for f in F:
        if f["kind"] in GHOST.get(u["type"], set()): f["ghost"] = True
        if f["kind"] == "road":
            f["route"] = True; f["wiggle"] = 0
            if u["type"] == "fort_hill": f["path"] = [f["path"][0], f["path"][-1]]   # foot → gate: pathfinding will find the serpentine
    return F


def route(g, pts, grade, step=3.0, k=25.0, relax=(1.0, 1.5, 2.0), margin=0.75, water=None, wcost=30.0):
    """Path over the grid g (the near plot's Grid) through the points pts: A* between neighboring points; segment slope ≤ grade·relax.
    Returns (path [[x, y]], info: length, how many times longer than the straight line, limit, limit relaxation)."""
    import heapq
    s_ = max(1, int(round(step / g.res))); G = g.H[::s_, ::s_]; st = g.res * s_; ny, nx = G.shape
    Wm = water[::s_, ::s_] if water is not None else np.zeros(G.shape, bool)   # water is expensive (a crossing only where needed), not the river bed as a road
    if Wm.any(): Wm = TA.dilate(Wm.astype(float), max(1, int(round(4.0 / st)))) > 0   # and a strip along the rim: the 4 m roadbed does not intrude into the water
    cell = lambda p: (int(np.clip(round((p[1] - g.y0) / st), 0, ny - 1)), int(np.clip(round((p[0] - g.x0) / st), 0, nx - 1)))
    D = [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if (dr, dc) != (0, 0) and math.gcd(abs(dr), abs(dc)) == 1]

    def leg(a, b, gmax):
        A, B = cell(a), cell(b); best = {A: 0.0}; prev = {}; pq = [(0.0, A)]; hB = lambda q: math.hypot(q[0] - B[0], q[1] - B[1]) * st
        while pq:
            _, q = heapq.heappop(pq)
            if q == B: break
            for dr, dc in D:
                r_, c_ = q[0] + dr, q[1] + dc
                if not (0 <= r_ < ny and 0 <= c_ < nx): continue
                L = math.hypot(dr, dc) * st; gr = abs(G[r_, c_] - G[q]) / L
                if gr > gmax: continue
                ng = best[q] + L * (1 + k * gr * gr) * (wcost if Wm[r_, c_] else 1.0)
                if ng < best.get((r_, c_), 1e18): best[(r_, c_)] = ng; prev[(r_, c_)] = q; heapq.heappush(pq, (ng + hB((r_, c_)), (r_, c_)))
        if B != A and B not in prev: return None
        P = [B]
        while P[-1] != A: P.append(prev[P[-1]])
        return [[g.x0 + c * st, g.y0 + r * st] for r, c in P[::-1]]
    for rl in relax:
        out = []
        for a, b in zip(pts[:-1], pts[1:]):
            q = leg(a, b, grade * rl * margin)                          # margin: smoothing cuts the serpentine's corners, neighboring legs
                                                                            # alter each other's ground (with a 3 m step the legs are longer and farther apart)
            if q is None: out = None; break
            out += q if not out else q[1:]
        if out:
            L = float(np.sum(np.hypot(*np.diff(np.asarray(out), axis=0).T))); L0 = float(np.sum(np.hypot(*np.diff(np.asarray(pts, float), axis=0).T)))
            return out, {"length_m": round(L), "detour": round(L / max(L0, 1e-6), 2), "limit": round(grade * rl, 3), "relax": rl}
    return [list(map(float, p)) for p in pts], {"not_routed": True, "limit": grade}
