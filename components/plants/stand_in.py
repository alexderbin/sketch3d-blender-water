"""Forest stand-ins for the draft (the sketch stage; the look knob stand_in of the forest and
forest_clear steps): a plan tree (forest.json: species, age, state, h, tilt, crown toward the light) → a crown as a solid of revolution from the species' measured
silhouette, and a trunk. Pure Python without Blender — parameters and knowledge; the mesh is built by bl_stand_in.py from the core primitives (core/blender/stand_in.py).
The silhouette is a measurement of real trees (bl_plant, tier 0, no tilt) by species and states at three heights → stand_in.json (calibrate;
the measurement — bench/stand_in_bench.py --measure): the bottom and top of the crown, the box half-width (x and y on average) and the radii of the RINGS belts (q90
of the foliage distance from the axis; for bare trees — of the branches) — each quantity is a·H + b: the skeleton grows with the tree, while foliage and fronds are organs of
their own size (in young trees and shrubs the crown is relatively wider). The belt radii are fitted to the box half-width. The silhouette class (sphere, cone,
column, umbrella, weeping) comes out of the measurement by itself. A species without a measurement gets a canopy ellipsoid (cr, crown_base): the test bench names it.
The canopy box (cr) of real trees is different: the measurement writes cr_k — the crown half-width against cr (spruce, fir, pine are almost twice as wide)."""
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__)); CAL = os.path.join(HERE, "stand_in.json")
RINGS = 6                                                             # crown belts (rings of the solid of revolution)
VERSION = 2                                                           # format of stand_in.json: quantities are pairs (a, b), a·H + b
SHOOT = ("pine", "spruce", "fir", "scale")                            # needles and scales — the image as is (bl_plant.card_material)
FALLBACK = {"adult_leafy": (), "young_leafy": ("adult_leafy",), "broken": ("adult_leafy", "adult_dead", "adult_winter"),
            "adult_winter": ("adult_dead", "adult_leafy"), "adult_dead": ("adult_winter", "adult_leafy"),
            "young_winter": ("young_dead", "adult_winter", "young_leafy"), "young_dead": ("young_winter", "adult_dead", "young_leafy")}
TWIGS = {"winter": (0.045, 0.038, 0.032), "dead": (0.060, 0.058, 0.055)}   # a bare crown is a mass of thin branches with gaps: the share of branches
                                                                      # ~0.3 × bark — dark (the trunk bark colour of dead trees looked white in the draft)
_CAL = None


def calibration():
    global _CAL
    if _CAL is None:
        _CAL = json.load(open(CAL)) if os.path.exists(CAL) else {}
        if _CAL.get("version") != VERSION: _CAL = {"species": {}, "tex": _CAL.get("tex") or {}}   # a foreign format — treated as no measurement
    return _CAL


def cond(age, state, leafy):
    """The state for the silhouette: adult_leafy | young_leafy | broken | adult_winter | young_winter | adult_dead | young_dead (a seedling is
    treated as young, an old tree as adult; bare in a season other than winter — as dead)."""
    young = age in ("young", "seedling")
    if state == "broken" and not young: return "broken"
    return ("young_" if young else "adult_") + ("leafy" if leafy else "winter" if state == "winter" else "dead")


def cond_of(t):
    sp = t.get("spec") or {}; st = sp.get("state") or t.get("state") or "healthy"
    return cond(t.get("age", "adult"), st, bool(sp.get("leaves") or sp.get("frond")) and st not in ("dead", "winter"))


def pick(species, key, cal=None):
    """The species silhouette in state key (no measurement — the nearest by FALLBACK) or None."""
    S = (cal or calibration())["species"].get(species) or {}
    return next((S[k] for k in (key,) + FALLBACK.get(key, ()) if k in S), None)


def crown_rgb(sp, tex):
    """The crown colour (linear) from the specification: a leaf — the leaf-body reflectance Rd of the species' optics × tint; needles, scales, fronds without optics —
    the mean colour of the organ image × tint (as in bl_plant.card_material); a bare crown — None."""
    lf, fr, O = sp.get("leaves"), sp.get("frond"), sp.get("optics") or {}
    if lf: typ, hue, tv = lf["type"], lf.get("hue", [1, 1, 1]), sum(lf.get("tint", (0.8, 1.15))) / 2
    elif fr: typ, hue, tv = fr.get("type", "lanceolate"), fr.get("hue", [1, 1, 1]), 1.0
    else: return None
    base = O.get("Rd") if O and typ not in SHOOT and typ != "blade" else tex.get(typ)
    base = base or (0.06, 0.12, 0.04)
    return tuple(round(min(1.0, float(b) * float(h) * tv), 4) for b, h in zip(base, hue))


def bark_rgb(sp):
    b = sp.get("bark") or {}; c0, c1 = b.get("c0", (0.16, 0.12, 0.08)), b.get("c1", (0.30, 0.24, 0.18))
    return tuple(round((float(x) + float(y)) / 2, 4) for x, y in zip(c0, c1))


def shape(t, cal=None):
    """A plan tree → a stand-in: profile — [(r, z)…] of the crown from the bottom up, m from the base (the poles are the bottom and top of the crown), None — no crown
    (a dead palm); trunk — (r0, r1, height) or None (a crown down to the ground — the trunk is inside); rgb, bark — colours; lean — (°, azimuth °);
    shift — the crown shift toward the light (m); calibrated — from a measurement."""
    cal = cal or calibration(); sp = t.get("spec") or {}; h = float(t["h"]); c = pick(t["species"], cond_of(t), cal)
    lin = lambda p: p[0] * h + p[1]
    if c:
        z1 = max(lin(c["z1"]), 0.3); R = max(lin(c["half"]), 0.05); rr = None
        if c.get("r"):
            z0 = min(max(lin(c["z0"]), 0.0), 0.95 * z1); rr = [max(lin(p), 0.0) for p in c["r"]]; m = max(rr) or 1.0; rr = [x * R / m for x in rr]
    else:                                                             # without a measurement — a canopy ellipsoid
        z0, z1 = float(sp.get("crown_base", 0.35)) * h, h; cr = float(t.get("cr") or 0.25 * h)
        rr = [cr * math.sin(math.pi * (k + 0.5) / RINGS) for k in range(RINGS)]
    r0 = float(sp.get("bark_radius") or 0.012 * h)
    if rr is None: profile, trunk = None, (R, 0.7 * R, z1)            # a pole (a dead palm): the box is the trunk with its bend
    else:
        zs = [z0 + (k + 0.5) / RINGS * (z1 - z0) for k in range(RINGS)]
        profile = [(0.0, z0)] + list(zip(rr, zs)) + [(0.0, z1)]
        trunk = (r0, 0.6 * r0, z0 + 0.25 * (z1 - z0)) if z0 > 0.04 * h else None
    v = 0.9 + 0.2 * ((int(t.get("seed", 0)) * 7919) % 1000) / 1000.0   # tree tone spread ±10 %: the crowns of a species stay distinguishable within one object
    rgb = crown_rgb(sp, cal.get("tex") or {}); bark = bark_rgb(sp)
    if rgb is None: rgb = TWIGS["winter" if (sp.get("state") or t.get("state")) == "winter" else "dead"]
    L = t.get("lean") or {}; push = t.get("push") or (0.0, 0.0)
    return {"profile": profile, "trunk": trunk, "rgb": tuple(min(1.0, x * v) for x in rgb), "bark": bark,
            "lean": (float(L.get("deg", 0.0)), float(L.get("az", 0.0))), "shift": (0.5 * float(push[0]), 0.5 * float(push[1])), "calibrated": c is not None}


def _fit(P):
    """[(H, value)…] → (a, b): value ≈ a·H + b (least squares); with a single height — b = 0."""
    n = len(P); mh = sum(p[0] for p in P) / n; mv = sum(p[1] for p in P) / n; d = sum((p[0] - mh) ** 2 for p in P)
    if d < 1e-6: return (round(mv / max(mh, 1e-6), 5), 0.0)
    a = sum((p[0] - mh) * (p[1] - mv) for p in P) / d
    return (round(a, 5), round(mv - a * mh, 4))


def calibrate(trees, tex):
    """The measurement (bench/bl_stand_in_measure.py: tree records — H, box, z0, z1, rq per belt) → the stand-in knowledge {species: {species:
    {state: {z0, z1, half, r[RINGS] — pairs (a, b), cr_k, spread, heights, seeds}}}, tex}: a quantity in metres = a·H + b over all
    records of the state (seeds × heights); r — the mean q90 of adjacent measurement belts; spread — the scatter of the half-width of real trees
    around the fit (a fraction, RMS): how much trees of one species and one size differ among themselves (willow, acacia, the "alien tree" — 15–25 %)."""
    from components.plants import plants as PL
    acc = {}
    for r in trees:
        if "max" not in r: continue
        rq = r.get("rq"); n = len(rq) // RINGS if rq else 0
        acc.setdefault((r["species"], cond(r["age"], r["state"], r["foliage"])), []).append(
            (r["H"], r.get("z0"), r["z1"], ((r["max"][0] - r["min"][0]) + (r["max"][1] - r["min"][1])) / 4,
             [sum(rq[k * n:(k + 1) * n]) / n for k in range(RINGS)] if rq else None, r["seed"]))
    out = {}
    for (sid, key), L in sorted(acc.items()):
        Hs = sorted({x[0] for x in L}); q = {"z1": _fit([(x[0], x[2]) for x in L]), "half": _fit([(x[0], x[3]) for x in L])}
        if all(x[4] is not None for x in L):
            q["z0"] = _fit([(x[0], x[1]) for x in L]); q["r"] = [_fit([(x[0], x[4][k]) for x in L]) for k in range(RINGS)]
        Hc = Hs[len(Hs) // 2]; w = float(PL.species(sid).get("width", 0.5))
        res = [x[3] / max(q["half"][0] * x[0] + q["half"][1], 1e-6) - 1 for x in L]
        q.update(cr_k=round((q["half"][0] * Hc + q["half"][1]) / (0.5 * w * Hc), 3), spread=round(math.sqrt(sum(e * e for e in res) / len(res)), 3),
                 heights=[round(Hs[0], 2), round(Hs[-1], 2)], seeds=len({x[5] for x in L}))
        out.setdefault(sid, {})[key] = q
    return {"version": VERSION, "rings": RINGS, "species": out, "tex": tex}
