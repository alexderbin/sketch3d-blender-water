"""The forest as a computation: from the site data — the suitability of each species (moisture, slope, landforms, water, roads, snow, biome) →
seeding → growth in steps (age, height by a Chapman–Richards curve) → competition for light (shade from taller neighbours, the species' shade
tolerance decides: grow slowly under the canopy or die) and for space (trunks in contact — the smaller one dies: self-thinning) → seeding from
mature trees (light-demanding species into the gaps, shade-tolerant ones under the canopy) → Kraft classes (dominant … suppressed) → the forest edge (crown
toward the open side, denser shrubs) and clearings → tilt (slope, light) and a shift of the crown toward the light. Like Deussen 1998.
Density is a factor (0 — no forest, 0.3 — open woodland, 1 — closed forest, > 1 — thicket).
.venv/bin/python -m components.plants.forest <site folder> [density=1] [seed=1] [box=x0,y0,x1,y1] [avoid=x,y,r;…] [out=…] → forest.json + __RESULT__"""
import json, math, os, sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
from components.plants import plants as PL

FORMS = ["flat", "peak", "ridge", "shoulder", "spur", "slope", "hollow", "footslope", "valley", "pit"]
FORM_W = {"valley": 1.3, "hollow": 1.35, "footslope": 1.2, "pit": 1.1, "peak": 0.6, "ridge": 0.55, "shoulder": 0.8}   # large trees — in the hollows
KRAFT = ["dominant", "codominant", "intermediate", "suppressed"]


class Site:
    """Site data at points: height, slope, water, roads, moisture (relief_w a_moist), landforms, distance to water/road, snow."""
    def __init__(self, folder):
        from core.surface.ground import Ground
        self.G = Ground(folder); self.loc = self.G.loc; p = os.path.join(folder, "relief_w.npz")
        self.W = np.load(p) if os.path.exists(p) else None
        sl = self.loc.get("snowline"); self.snow = float(sl) if sl not in (None, "none") else None
        self.biome = self.loc.get("biome") or "temperate"

    def _w(self, key, X, Y):
        if self.W is None or key not in self.W.files: return None
        xs, ys = self.W["xs"], self.W["ys"]; A = self.W[key]
        i = np.clip(np.searchsorted(ys, Y), 0, len(ys) - 1); j = np.clip(np.searchsorted(xs, X), 0, len(xs) - 1); return A[i, j].astype(float)

    def moist(self, X, Y):
        m = self._w("a_moist", X, Y)
        if m is None:                                                 # no landform analysis — by the distance to water
            dw = self.G.amap("dwater", X, Y).astype(float) if "dwater" in self.G.amaps else np.full(np.shape(X), 50.0); return np.clip(1 - dw / 60, 0, 1)
        return np.clip(m / (255.0 if m.max() > 1.5 else 1.0), 0, 1)

    def fields(self, X, Y):
        G = self.G; F = {"z": G.height(X, Y), "slope": G.steep(X, Y), "wet": G.wet(X, Y), "road": G.mask("road", X, Y) + G.mask("pad", X, Y),
                         "moist": self.moist(X, Y)}
        F["dwater"] = G.amap("dwater", X, Y).astype(float) if "dwater" in G.amaps else np.full(np.shape(X), 99.0)
        F["droad"] = G.amap("droad", X, Y).astype(float) if "droad" in G.amaps else np.full(np.shape(X), 99.0)
        F["form"] = G.amap("forms", X, Y) if "forms" in G.amaps else np.zeros(np.shape(X), int)
        return F


def _window(v, lo, hi, soft=0.15):
    return np.clip((v - (lo - soft)) / soft, 0, 1) * np.clip(((hi + soft) - v) / soft, 0, 1)


def suitability(S, sp, F):
    """Species suitability at points: moisture within the species' window, slope, water and shore (closer for moisture-loving ones), roads, snow, landforms."""
    m0, m1 = sp.get("moist", [0, 1]); s = _window(F["moist"], m0, m1)
    smax = 38.0 if sp["habit"] == "shrub" else 32.0; s *= np.clip((smax - F["slope"]) / 10.0, 0, 1)
    near = 0.6 if m1 >= 0.95 else 2.5; s *= (F["wet"] < 0.01) & (F["dwater"] >= near) & (F["road"] < 0.2) & (F["droad"] >= 3.0)
    if S.snow is not None: s *= F["z"] < S.snow - 4
    fw = np.ones(np.shape(s))
    if sp["habit"] in ("tree", "palm"):
        for nm, w in FORM_W.items(): fw = np.where(F["form"] == FORMS.index(nm), w, fw)
    return s * fw


def grow_h(sp, age):
    """Height by age (Chapman–Richards): H = Hmax·(1 − e^(−3·age/life))^1.3."""
    Hm = float(np.mean(sp["height"])); return Hm * (1 - np.exp(-3.0 * age / sp.get("life", 200))) ** 1.3


def plan(folder, density=1.0, seed=1, box=None, avoid=(), steps=10, dt=10.0, pool=None, clear_noise=0.35, limit=6000, blocked=None):
    """Computation of a plot's tree stand. box — (x0, y0, x1, y1), otherwise the site's near grid minus the edge; avoid — [(x, y, r)] (figures, cameras);
    blocked — a function (X, Y) → a bool array: ground occupied by buildings (the occupancy layer, core.layers.occupied_at) — no trees there.
    Returns {trees: […], stats}. trees[i]: species, habit, age (class), state, x, y, z, h, cr (crown radius), kraft, layer, lean, push,
    exposure, seed."""
    S = Site(folder); G = S.G; K = PL.knowledge(); rng = np.random.default_rng(seed)
    b = G.near_bounds(); box = box or (b[0] + 8, b[1] + 8, b[2] - 8, b[3] - 8); x0, y0, x1, y1 = box; area = (x1 - x0) * (y1 - y0)
    pool = pool or [k for k, s in K["species"].items() if S.biome in s.get("biomes", []) and s["habit"] in ("tree", "palm", "shrub")]
    SP = {k: PL.species(k) for k in pool}
    canopy = [k for k in pool if SP[k]["habit"] in ("tree", "palm")]; shrubs = [k for k in pool if SP[k]["habit"] == "shrub"]
    density = density * float((K.get("biomes") or {}).get(S.biome, {}).get("density", 1.0))   # biome density (desert — sparse, jungle — denser)
    empty = {"box": box, "trees": [], "stats": {"biome": S.biome, "density": density, "area_ha": round(area / 1e4, 2), "trees_n": 0, "shrubs_n": 0,
             "seedlings_below_1_5m": 0, "by_species": {}, "by_class": {}, "by_layer": {}, "seeded": 0, "stems_per_ha": 0, "mean_height": 0, "closure": 0}}
    if not canopy or density <= 0: return empty
    # plot fields on a 2 m grid: species suitability, gaps (clearings) — noise, story locations and avoid — are open
    gx = np.arange(x0, x1, 2.0); gy = np.arange(y0, y1, 2.0); GX, GY = np.meshgrid(gx, gy); F = S.fields(GX, GY)
    from core.surface.noise import fbm
    ab = {k: float((SP[k].get("abundance") or {}).get(S.biome, 1.0)) for k in pool}
    patch = {k: 0.25 + 1.5 * (fbm(GX / 85.0, GY / 85.0, 2, seed + 101 + 7 * n) * 0.5 + 0.5) ** 2 for n, k in enumerate(pool)}   # plantings in patches
    suit = {k: suitability(S, SP[k], F) * ab[k] * patch[k] for k in pool}
    forest = fbm(GX / 70.0, GY / 70.0, 3, seed + 3) * 0.5 + 0.5; cover = np.clip(0.25 + 0.55 * density, 0, 1)
    open_ = forest < np.quantile(forest, 1 - cover) if cover < 1 else np.zeros_like(forest, bool)
    for a_ in S.loc.get("public", []):
        if a_ in S.loc.get("anchors", {}): ax, ay, _ = S.loc["anchors"][a_]; open_ |= np.hypot(GX - ax, GY - ay) < 14
    for (ax, ay, r) in avoid: open_ |= np.hypot(GX - ax, GY - ay) < r
    if blocked is not None: open_ |= blocked(GX, GY)
    tree_ok = np.max([suit[k] for k in canopy], 0) * (~open_)

    def sample(n, w):
        p = w.ravel() / max(w.sum(), 1e-9); i = rng.choice(w.size, n, p=p)
        return GX.ravel()[i] + rng.uniform(-1, 1, n), GY.ravel()[i] + rng.uniform(-1, 1, n)

    def pick_species(X, Y, keys):
        Fp = S.fields(X, Y); ii = np.clip(((Y - y0) / 2).astype(int), 0, len(gy) - 1); jj = np.clip(((X - x0) / 2).astype(int), 0, len(gx) - 1)
        W = np.array([suitability(S, SP[k], Fp) * ab[k] * patch[k][ii, jj] for k in keys]) + 1e-6; W /= W.sum(0)
        return [keys[int(rng.choice(len(keys), p=W[:, i]))] for i in range(len(X))], Fp
    # seeding: seed density ~ canopy × density (3 times more than will survive — self-thinning)
    cr_mean = float(np.mean([0.5 * SP[k].get("width", 0.5) * np.mean(SP[k]["height"]) for k in canopy]))
    n0 = int(min(limit, 3.0 * density * area * max(tree_ok.mean(), 0.05) / (math.pi * cr_mean ** 2) * 2.2))
    X, Y = sample(max(n0, 10), tree_ok); spc, _ = pick_species(X, Y, canopy)
    life = np.array([SP[k].get("life", 200) for k in spc], float); age = rng.uniform(0, 0.6, len(X)) * life
    alive = np.ones(len(X), bool); sid = np.array(spc, object); tol = np.array([SP[k].get("tolerance", 0.3) for k in spc])
    for step in range(steps):
        H = np.array([grow_h(SP[s], a) for s, a in zip(sid, age)]); cr = np.array([0.5 * SP[s].get("width", 0.5) for s in sid]) * H
        idx = np.nonzero(alive)[0]; P = np.column_stack([X[idx], Y[idx]]); T = cKDTree(P); rmax = float(cr[idx].max()) if len(idx) else 1
        shade = np.zeros(len(X)); crowd = np.zeros(len(X), bool)
        for a_, i in enumerate(idx):
            for b_ in T.query_ball_point(P[a_], rmax):
                j = idx[b_]
                if j == i or H[j] <= H[i] * 1.05: continue
                d = math.hypot(X[i] - X[j], Y[i] - Y[j])
                if d < cr[j]: shade[i] += (1 - d / cr[j]) * min(1.0, (H[j] - H[i]) / max(H[j] * 0.3, 0.5))
                if d < 0.28 * (cr[i] + cr[j]) and H[j] > H[i]: crowd[i] = True   # trunks in contact: the smaller one yields
        light = np.exp(-1.6 * shade)
        age = np.where(alive, age + dt * light ** (1.0 - tol) * (0.7 + 0.3 * rng.random(len(X))), age)
        die = alive & ((light < 0.45 * (1 - tol)) & (rng.random(len(X)) < 0.55) | crowd & (rng.random(len(X)) < 0.6) | (age > life * rng.uniform(0.95, 1.3, len(X))))
        alive &= ~die
        mature = np.nonzero(alive & (age > 0.15 * life))[0]           # seeding: from each mature tree — 0…2 seeds nearby
        if len(mature) and alive.sum() < limit:
            k_ = rng.integers(0, 2, len(mature)); src = np.repeat(mature, k_)
            if len(src):
                a = rng.uniform(0, 2 * math.pi, len(src)); rr = cr[src] * rng.uniform(0.8, 3.0, len(src))
                nx, ny = X[src] + rr * np.cos(a), Y[src] + rr * np.sin(a); ok = (nx > x0) & (nx < x1) & (ny > y0) & (ny < y1)
                if ok.any():
                    nx, ny, src = nx[ok], ny[ok], src[ok]
                    ii = np.clip(((ny - y0) / 2).astype(int), 0, len(gy) - 1); jj = np.clip(((nx - x0) / 2).astype(int), 0, len(gx) - 1)
                    sh = np.zeros(len(nx))                              # a seedling in the shade of mature trees: the shade-tolerant one survives
                    Tm = cKDTree(np.column_stack([X[mature], Y[mature]]))
                    for q, (px, py) in enumerate(zip(nx, ny)):
                        for m in Tm.query_ball_point([px, py], rmax): sh[q] += max(0.0, 1 - math.hypot(px - X[mature[m]], py - Y[mature[m]]) / max(cr[mature[m]], 0.5))
                    tl = np.array([SP[s_].get("tolerance", 0.3) for s_ in sid[src]])
                    keep = rng.random(len(nx)) < np.array([suit[s][i, j] for s, i, j in zip(sid[src], ii, jj)]) * (~open_[ii, jj]) * np.exp(-sh * (1.6 - 1.4 * tl))
                    nx, ny, ns = nx[keep], ny[keep], sid[src][keep]
                    X = np.concatenate([X, nx]); Y = np.concatenate([Y, ny]); sid = np.concatenate([sid, ns]); age = np.concatenate([age, np.zeros(len(nx))])
                    life = np.concatenate([life, [SP[s].get("life", 200) for s in ns]]); tol = np.concatenate([tol, [SP[s].get("tolerance", 0.3) for s in ns]])
                    alive = np.concatenate([alive, np.ones(len(nx), bool)])
    Hf = np.array([grow_h(SP[s], a) for s, a in zip(sid, age)]); n_seedl = int((alive & (Hf < 1.5)).sum())   # seedlings below 1.5 m are counted (herb layer)
    keep = np.nonzero(alive & (Hf >= 1.5))[0]; X, Y, sid, age, life, tol = X[keep], Y[keep], sid[keep], age[keep], life[keep], tol[keep]
    H = np.array([grow_h(SP[s], a) for s, a in zip(sid, age)]); cr = np.array([0.5 * SP[s].get("width", 0.5) for s in sid]) * H
    out, P = [], np.column_stack([X, Y]); T = cKDTree(P) if len(P) else None
    Fp = S.fields(X, Y) if len(X) else None
    for i in range(len(X)):
        nb = [j for j in T.query_ball_point(P[i], max(cr[i] * 3.0, 6.0)) if j != i] if T else []
        top = max([H[j] for j in nb if math.hypot(X[j] - X[i], Y[j] - Y[i]) < cr[i] + cr[j]] + [H[i]])
        rel = H[i] / max(top, 1e-3); kraft = 0 if rel >= 0.98 else 1 if rel >= 0.82 else 2 if rel >= 0.6 else 3
        g = np.zeros(2); wsum = 0.0                                   # toward the light: away from neighbours, weighted by crown overlap
        for j in nb:
            d = math.hypot(X[j] - X[i], Y[j] - Y[i]); ov = (cr[i] + cr[j]) - d
            if ov > 0: g -= (P[j] - P[i]) / max(d, 0.1) * ov * (1.0 if H[j] >= H[i] * 0.8 else 0.4); wsum += ov
        exp_n = len(nb); openness = float(np.clip(1 - exp_n / 10.0, 0, 1))
        push = g / max(np.linalg.norm(g), 1e-6) * min(cr[i] * 0.35, 0.12 * wsum + openness * cr[i] * 0.25) if np.linalg.norm(g) > 1e-6 else np.zeros(2)
        sl = float(Fp["slope"][i]); e = 1.0
        gxv = float(G.height(np.array([X[i] + 1.0]), np.array([Y[i]]))[0] - G.height(np.array([X[i] - 1.0]), np.array([Y[i]]))[0]) / 2
        gyv = float(G.height(np.array([X[i]]), np.array([Y[i] + 1.0]))[0] - G.height(np.array([X[i]]), np.array([Y[i] - 1.0]))[0]) / 2
        down = math.degrees(math.atan2(-gyv, -gxv)); lean_deg = min(8.0, 0.18 * sl) + (3.0 * openness if np.any(push) else 0)
        lean_az = down if sl > 6 else (math.degrees(math.atan2(push[1], push[0])) if np.any(push) else float(rng.uniform(0, 360)))
        sp = SP[sid[i]]; a_rel = age[i] / life[i]; Hm = float(np.mean(sp["height"]))
        acls = "young" if H[i] < 0.45 * Hm else "old" if a_rel > 0.7 else "adult"
        if H[i] < 0.12 * Hm: acls = "seedling"
        if float(Fp["dwater"][i]) < 4: e *= 1.7                       # near water the roots are washed out
        state = "healthy"
        if acls == "old" and rng.random() < 0.12: state = rng.choice(["dead", "broken"])
        if kraft == 3 and rng.random() < 0.08: state = "dead"
        layer = "overstory" if kraft <= 1 else "saplings" if acls in ("young", "seedling") else "second_layer"
        if sp["habit"] == "shrub": layer = "understory"
        out.append({"species": str(sid[i]), "habit": sp["habit"], "age": acls, "state": str(state), "x": round(float(X[i]), 2), "y": round(float(Y[i]), 2),
                    "z": round(float(Fp["z"][i]), 2), "h": round(float(H[i]), 2), "cr": round(float(cr[i]), 2), "kraft": KRAFT[kraft], "layer": layer,
                    "lean": {"deg": round(lean_deg, 1), "az": round(lean_az, 1), "recover": 0.6}, "push": [round(float(push[0]), 2), round(float(push[1]), 2)],
                    "exposure": round(e * (1 + 0.02 * sl), 2), "openness": round(openness, 2), "seed": int(rng.integers(1, 1 << 30))})
    # understory: shrubs are denser at the forest edge (open on the side, light), under the canopy shade-tolerant ones are sparse
    if shrubs and out:
        P = np.array([[t["x"], t["y"]] for t in out]); T = cKDTree(P); crs = np.array([t["cr"] for t in out]); n_sh = int(density * area / 160)
        cx, cy = sample(n_sh * 3, np.max([suit[k] for k in shrubs], 0))
        spc_s, Fs = pick_species(cx, cy, shrubs); placed = 0
        for k, (x, y, s) in enumerate(zip(cx, cy, spc_s)):
            if placed >= n_sh: break
            d, j = T.query([x, y]); near_edge = (d > crs[j] * 0.8) and (d < crs[j] * 2.2)
            under = d < crs[j] * 0.8; ss = SP[s]; p = 0.9 if near_edge else (0.25 * ss.get("tolerance", 0.5) if under else 0.08)
            if rng.random() > p: continue
            H_ = float(rng.uniform(*ss["height"])) * (0.7 if under else 1.0)
            out.append({"species": s, "habit": "shrub", "age": "adult", "state": "healthy", "x": round(float(x), 2), "y": round(float(y), 2),
                        "z": round(float(Fs["z"][k]), 2), "h": round(H_, 2), "cr": round(0.5 * ss.get("width", 0.8) * H_, 2), "kraft": "understory",
                        "layer": "understory", "lean": {"deg": 0, "az": 0, "recover": 0.6}, "push": [0, 0], "exposure": 1.0, "openness": 1.0 if near_edge else 0.3,
                        "seed": int(rng.integers(1, 1 << 30)), "edge": bool(near_edge)})
            placed += 1
    st = {"biome": S.biome, "density": density, "area_ha": round(area / 1e4, 2), "trees_n": sum(t["habit"] != "shrub" for t in out),
          "shrubs_n": sum(t["habit"] == "shrub" for t in out), "seedlings_below_1_5m": n_seedl, "by_species": {}, "by_class": {}, "by_layer": {}, "seeded": int(n0)}
    for t in out:
        st["by_species"][t["species"]] = st["by_species"].get(t["species"], 0) + 1; st["by_class"][t["kraft"]] = st["by_class"].get(t["kraft"], 0) + 1
        st["by_layer"][t["layer"]] = st["by_layer"].get(t["layer"], 0) + 1
    tr = [t for t in out if t["habit"] != "shrub"]
    st["stems_per_ha"] = round(len(tr) / max(area / 1e4, 1e-6)); st["mean_height"] = round(float(np.mean([t["h"] for t in tr])), 1) if tr else 0
    st["closure"] = round(min(1.0, sum(math.pi * t["cr"] ** 2 for t in tr if t["kraft"] in KRAFT[:2]) / area), 2)
    return {"box": box, "trees": out, "stats": st}


def evergreen(sid):
    """Evergreen: needles, scale leaves, palm (species knowledge: foliage type, habit)."""
    S = PL.species(sid); return S["habit"] == "palm" or (S.get("leaves") or {}).get("type") in ("pine", "spruce", "fir", "scale")


def with_specs(R, season=None):
    """For each tree — a plant specification (plants.resolve): the height from the computation, tilt, crown toward the light, root exposure; season —
    winter (deciduous trees without leaves) or autumn (foliage tint); conifers and palms do not change."""
    PL.textures()
    for t in R["trees"]:
        st = t["state"]
        if st == "healthy" and season == "winter" and not evergreen(t["species"]): st = "winter"
        if st == "healthy" and season == "autumn" and not evergreen(t["species"]): st = "autumn"
        t["spec"] = PL.resolve(t["species"], t["age"], st, seed=t["seed"] % 100000, height=t["h"], lean=t["lean"], push=t["push"],
                               exposure=t.get("exposure", 1.0))
    R["season"] = season; return R


def plan_map(R, out, px=4.0):
    """A top-down map of the computation (PIL): crowns as circles — the species colour, the outline — the Kraft class; forest-edge shrubs — dots."""
    from PIL import Image, ImageDraw, ImageFont
    x0, y0, x1, y1 = R["box"]; W, H = int((x1 - x0) * px), int((y1 - y0) * px); im = Image.new("RGB", (W, H), (235, 232, 215)); d = ImageDraw.Draw(im)
    try: fnt = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except Exception: fnt = ImageFont.load_default()
    pal = {}; base = [(46, 110, 50), (120, 150, 40), (30, 80, 70), (160, 120, 40), (60, 140, 120), (100, 60, 120), (150, 60, 50), (80, 80, 160), (40, 60, 30)]
    kc = {"dominant": (0, 0, 0), "codominant": (60, 60, 60), "intermediate": (140, 140, 140), "suppressed": (210, 60, 60), "understory": (230, 160, 30)}
    for t in sorted(R["trees"], key=lambda t: t["h"]):
        c = pal.setdefault(t["species"], base[len(pal) % len(base)]); cx, cy, r = (t["x"] - x0) * px, (y1 - t["y"]) * px, max(t["cr"] * px, 2)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c, outline=kc.get(t["kraft"], (0, 0, 0)), width=2 if t["kraft"] != "suppressed" else 3)
    y = 6
    for k, c in pal.items(): d.rectangle([6, y, 22, y + 14], fill=c); d.text((28, y), k, fill=(0, 0, 0), font=fnt); y += 18
    for k, c in kc.items(): d.rectangle([6, y, 22, y + 14], outline=c, width=3); d.text((28, y), k, fill=(0, 0, 0), font=fnt); y += 18
    im.save(out); return out


def main():
    a = sys.argv[1:]; folder = a[0]; dens = float(a[1]) if len(a) > 1 else 1.0; seed = int(a[2]) if len(a) > 2 else 1
    box, out, avoid = None, os.path.join(folder, "forest.json"), []
    for x in a[3:]:
        if x.startswith("box="): box = tuple(map(float, x[4:].split(",")))
        if x.startswith("out="): out = x[4:]
        if x.startswith("avoid="): avoid = [tuple(map(float, q.split(","))) for q in x[6:].split(";") if q]   # x,y,r;… — cameras, buildings
    R = with_specs(plan(folder, dens, seed, box, avoid=avoid)); json.dump(R, open(out, "w"), ensure_ascii=False)
    plan_map(R, os.path.splitext(out)[0] + "_map.png")
    print("__RESULT__" + json.dumps(R["stats"], ensure_ascii=False))


if __name__ == "__main__":
    main()
