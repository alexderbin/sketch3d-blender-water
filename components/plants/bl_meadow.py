"""Herb layer (step 4): grasses in tufts and forbs with flowers — as instances of variants (bl_plant: ribbon blades, leaves,
flowers) at points near the cameras. Density comes from the site data: the share of the "base" (meadow) layer from relief_w, moisture (dry — dry grasses, wet — taller
and denser), paths and roads, water, canopy shade (forest.json: sparser under crowns, fern and nettle), species patches (flowers in clumps);
by distance: full up to near, beyond — density per area ∝ (near/d)², larger tufts (the same frame coverage), fading to nothing by reach,
beyond it — the ground colour of the site shader. Blades, tuft coherence (a shared tilt), colour from a field.
The site's cover (components/grass, cover.npz; read through the core's Ground.cover) — if present, it decides for the meadow: density (bare ground — no
grass), the group by shares (a shared budget), the group's species — in clumps (clones), the group's height, dryness — the grass variant.
Where and what — meadow.py (numpy): the camera frame by the lens, the 3D distance to the eye, cameras are merged (density — by the most
demanding one), tuft tiers by distance, a ceiling on instances (it once took 112 thousand tufts, 68 million triangles).
The function build_meadow is for the builders (the meadow sheet, the component step)."""
import time
import bpy
import numpy as np

from components.plants import bl_plant as BP
from components.plants import bl_forest as BF


def build_meadow(G, cams, specs, sc=None, forest=None, near=6.0, reach=70.0, rho=40.0, K=4, seed=1, fov=None, budget=40000, parent=None, canopy=None, cards=None):
    """cams — the layer's cameras (dict: matrix_world, lens_mm …) or [(x, y, z, heading)]; specs — {species: [variant specifications]} (meadow_specs);
    rho — tufts per m² near the camera; canopy — the canopy trees (the canopy.json layer; forest — the old input with the same trees list).
    cards — the card tier: {"source": the set folder, "px": [3, 60], "px_herb": 20, "width_px", "overlap",
    "budget", "gain"} — grass tufts smaller than px[1] pixels (forbs — px_herb) give way to cards (build_cards), the cards fade out toward px[0];
    None — tufts up to reach, as before.
    Where and what — meadow.points / meadow.choose (camera frame, 3D distance, merging cameras, tiers, ceiling); here — the prototypes
    (species × variant × tier — only the needed ones) and the instances. Returns: (collection, cost)."""
    from components.plants import meadow as MD
    sc = sc or bpy.context.scene; t0 = time.time()
    canopy = canopy if canopy is not None else (forest or {}).get("trees") if forest else None
    P, D = MD.points(G, cams, rho=rho, near=near, reach=reach, seed=seed, budget=budget)
    px_t = float(cards["px"][1]) if cards else None
    C = MD.choose(G, P, D, specs, K, near, reach, seed, canopy=canopy, ppm=MD.ppm(cams, P) if cards else None, px_tuft=px_t,
                  px_herb=float(cards.get("px_herb", 20)) if cards else None); sids = C["sids"]
    cc, ccost = build_cards(G, cams, specs, cards, sc, parent or sc.collection, K=K, near=near, seed=seed, canopy=canopy) if cards else (None, None)
    if not len(C["X"]) and ccost and ccost.get("cards"):             # cards only (the cameras are far from the ground)
        return cc, {"tufts": 0, "points": int(len(P)), "s": round(time.time() - t0, 1), "cards": ccost}
    if not len(C["X"]):                                                # the step does not stay silent: if it built nothing, it gives the reason; no empty objects are made
        hg = [float(c["eye"][2] - G.height(np.array([c["eye"][0]]), np.array([c["eye"][1]]))[0]) for c in map(MD.camera, cams)]
        why = (f"cameras are above to ({reach:g} m to the eye; the lowest is {min(hg):.0f} m above the ground) — no meadow in the frame" if hg and min(hg) >= reach else
               "no cameras" if not cams else "no ground for a meadow in the cameras' frames (water, buildings, cover without grass)" if len(P) else "no ground closer than to in the cameras' frames")
        return None, {"tufts": 0, "points": int(len(P)), "s": round(time.time() - t0, 1), "_skipped": why}
    parent = parent or sc.collection; coll = bpy.data.collections.new("meadow"); parent.children.link(coll)
    pc = bpy.data.collections.new("meadow_protos"); parent.children.link(pc)
    key = C["sid"] * 100 + C["k"] * 10 + np.where(np.array([specs[sids[i]][0]["habit"] == "grass" for i in C["sid"]], bool) if len(C["sid"]) else np.zeros(0, bool), C["lod"], 0)
    pidx, tris = {}, {}
    for kk in sorted(set(int(x) for x in key)):                       # a prototype for each needed combination of species × variant × tier
        si, k, lod = kk // 100, (kk // 10) % 10, kk % 10; sp = MD.lod_spec(specs[sids[si]][k], lod)
        objs, _ = BP.build(sp, (0.0, 0.0, 0.0), name=f"m_{sids[si]}_{k}_{lod}", coll=pc)
        ob = BF.join(objs, f"q{len(pidx):05d}") if len(objs) > 1 else objs[0]; ob.name = f"q{len(pidx):05d}"; ob["kind"] = "plant"
        pidx[kk] = len(pidx); tris[kk] = sum(len(f.vertices) - 2 for f in ob.data.polygons)
    BF.exclude(pc)
    idx = [pidx[int(x)] for x in key]
    inst = BF.instancer("meadow_instances", coll, list(zip(C["X"], C["Y"], C["Z"])), pc, idx, C["rot"], C["scl"]); inst["kind"] = "plant"
    lod = C["lod"]; tr = int(sum(tris[int(x)] for x in key))
    return coll, {"tufts": int(len(key)), "prototypes": len(pidx), "tris": tr, "points": int(len(P)), "s": round(time.time() - t0, 1),
                  "by_lod": [int((lod == i).sum()) for i in range(3)], "by_species": {s: int((C["sid"] == i).sum()) for i, s in enumerate(sids) if (C["sid"] == i).any()},
                  **({"cards": ccost} if ccost else {})}


def build_cards(G, cams, specs, cards, sc, parent, K=4, near=6.0, seed=1, canopy=None):
    """The card tier: points — meadow.card_points (density by frame, grass 3…60 px), species and height — meadow.choose
    (cover: density, group, species in clumps, height), the model — from the species' set (cards.yaml), the height — real, the width — at least
    width_px pixels (grasses; flowers and forbs keep their own width); the point colour "cc" of a grass — the site's grass colour (bl_terrain_surf.grass_colour:
    the base scan × a fit, dry/wet by moisture) × the cover tint ×
    (the variant colour with dryness / fresh grass): the same colour as the distant canopy, — a seam without a stripe; otherwise — the model's own colour.
    → (collection, cost)."""
    import os
    import core.paths
    from components.plants import meadow as MD
    from components.plants import bl_cards as BC
    t0 = time.time(); px = tuple(float(v) for v in cards.get("px", (3, 60))); wpx = float(cards.get("width_px", 24)); px_h = float(cards.get("px_herb", 20))
    P, D, SD = MD.card_points(G, cams, px=px, w_px=wpx, overlap=float(cards.get("overlap", 6)), seed=seed, budget=int(cards.get("budget", 60000)))
    if not len(P): return None, {"cards": 0, "s": round(time.time() - t0, 1)}
    C = MD.choose(G, P, D, specs, K, near, 1e9, seed + 5, canopy=canopy); sids = C["sids"]; sd = SD[C["idx"]]
    src = cards.get("source") or "library/plants/cards/low_poly"; lib = BC.library(src if os.path.isabs(src) else os.path.join(core.paths.ROOT, src))
    want = sorted({w for i in set(int(x) for x in C["sid"]) for w in lib["species"].get(sids[i], [])})
    coll = bpy.data.collections.new("meadow_cards"); parent.children.link(coll); pc = bpy.data.collections.new("meadow_card_protos"); parent.children.link(pc)
    W = BC.load(lib["dir"], want, pc); names = sorted(W); widx = {w: i for i, w in enumerate(names)}
    for i, w in enumerate(names): W[w].ob.name = f"c{i:05d}"                # name order = prototype index
    BF.exclude(pc); r = np.random.default_rng(seed + 29); n = len(C["X"])
    from core.surface import bl_terrain_surf as TS                         # the colour is the same as for distant grass (the canopy): the base scan
    gc = TS.grass_colour(G.folder); base = None; tn = None                 # of the site × a fit (dry/wet — by the cover moisture) × tint
    if gc is not None:
        if G.cover_meta():
            t_ = np.clip((G.cover("W", C["X"], C["Y"]) - 0.44) / 0.14, 0, 1); t_ = (t_ * t_ * (3 - 2 * t_))[:, None]
            base = gc[0][None, :] * (1 - t_) + gc[1][None, :] * t_; tn = G.cover("tint", C["X"], C["Y"])
        else: base = np.repeat(gc[0][None, :], len(C["X"]), 0)
    ref = next((np.asarray(v[0]["color"], float) for k_, v in specs.items() if v[0].get("color") is not None and k_ == "meadow_grass"),
               next((np.asarray(v[0]["color"], float) for v in specs.values() if v[0].get("color") is not None), None))
    keep = np.zeros(n, bool); idx = np.zeros(n, int); scl = np.zeros((n, 3)); col = np.zeros((n, 3)); tris = 0
    for q in range(n):
        s = sids[C["sid"][q]]; cand = [w for w in lib["species"].get(s, []) if w in W]
        if not cand: continue
        w = cand[r.integers(len(cand))]; cd = W[w]; ht = C["hnom"][s] * C["scl"][q, 2]; sz = ht / max(cd.h, 1e-3)
        sp = specs[s][min(int(C["k"][q]), len(specs[s]) - 1)]           # wider in the distance — grasses; a flower and forbs keep their own width
        if sp.get("habit") != "grass" and ht / sd[q] > px_h: continue      # and larger than px_herb — as a tuft (the shape reads)
        sxy = max(sz, wpx * sd[q] / cd.w) if sp.get("habit") == "grass" else sz   # (a stretched flower would be a rusty field instead of a spark)
        if sp.get("color") is not None:                                     # grass: the species colour with the variant's dryness (as for a tuft) —
            col[q] = (1 - float(sp.get("dry", 0.0))) * np.array(sp["color"]) + float(sp.get("dry", 0.0)) * np.array(sp.get("dry_color", sp["color"]))
            if base is not None and ref is not None:                        # relative to a fresh grass, on the site base tone × tint
                col[q] = base[q] * (tn[q] if tn is not None else 1.0) * np.clip(col[q] / np.maximum(ref, 1e-3), 0.3, 3.0)
        else: col[q] = cd.mean
        keep[q] = True; idx[q] = widx[w]; scl[q] = (sxy, sxy, sz); tris += cd.tris
    if not keep.any(): return None, {"cards": 0, "s": round(time.time() - t0, 1)}
    rot = np.column_stack([r.normal(0, 0.05, n), r.normal(0, 0.05, n), r.uniform(0, 2 * np.pi, n)])[keep]
    inst = BF.instancer("meadow_cards_instances", coll, list(zip(C["X"][keep], C["Y"][keep], C["Z"][keep])), pc, idx[keep], rot, scl[keep]); inst["kind"] = "plant"
    a = inst.data.attributes.new("cc", "FLOAT_VECTOR", "POINT"); a.data.foreach_set("vector", (col[keep] * float(cards.get("gain", 1.0))).astype(np.float32).ravel())
    return coll, {"cards": int(keep.sum()), "models": len(W), "tris": int(tris), "candidates": int(len(P)), "s": round(time.time() - t0, 1)}
