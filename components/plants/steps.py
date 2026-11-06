"""Entry of the "Plants and details" component for the step-by-step build (core/blender/build.py): the "plants" step — trees and boulders on the terrain
of an open scene (the old path — bl_tree.py and bl_boulder.py, commands that append to the scene.obj of a grey copy).
plan (project Python): keys (trees N, boulders N, seed …) + the site (the "site" section of the run scene: relief.npz, location.json —
after the "building" step this is the edited terrain) + the building (the plan folder from "gives": the castle footprint — plants do not stand on the pad) →
plan.json: places (Ground.spots: slope, dry, not on a road or pad, spacing from each other, a ring around the building), species and seed.
build (Blender): a tree — bl_tree.build_parts (tree skeleton, bark and crown), a boulder — bl_boulder.boulder (A.N.T., exact size);
seating by the scene terrain (a ray onto the "terrain" object, fallback — the height from the grid), one object per tree and boulder in the section collection."""
import json, math, os

BIOME_TREES = {"temperate": {"oak": 0.6, "pine": 0.25, "shrub": 0.15}, "jungle": {"oak": 0.8, "shrub": 0.2},
               "alpine": {"pine": 0.8, "shrub": 0.2}, "desert": {"dead": 0.5, "shrub": 0.5}, "arid": {"dead": 0.5, "shrub": 0.5},
               "snow": {"pine": 0.6, "oak_winter": 0.3, "shrub": 0.1}, "arctic": {"pine": 0.6, "oak_winter": 0.3, "shrub": 0.1},
               "ash": {"dead": 0.8, "shrub": 0.2}, "wasteland": {"dead": 0.8, "shrub": 0.2}}
BARK = (0.30, 0.22, 0.15)
CROWN = {"oak": (0.17, 0.30, 0.09), "pine": (0.07, 0.19, 0.10), "shrub": (0.20, 0.30, 0.10)}
ROCK = (0.36, 0.35, 0.33)


def _footprint(scene):
    """The footprint of buildings from the occupancy layer (core.layers): the centre — the anchor castle.centre | town.centre | temple.centre | edifice.centre,
    otherwise the middle of the occupied area; the radius — to the farthest point of the pad, walls, towers and buildings. No layer — (None, 0)."""
    from core import layers as L
    items = L.occupancy(scene, ("site", "wall", "tower", "building", "yard"))
    if not items: return None, 0.0
    pts = [p for it in items for p in (it.get("poly") or it.get("line") or [])]
    a = L.anchors(scene); c = next((a[k][:2] for k in ("castle.centre", "town.centre", "temple.centre", "edifice.centre") if k in a), None)
    if c is None: c = [sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)]
    return [float(c[0]), float(c[1])], max(math.hypot(p[0] - c[0], p[1] - c[1]) for p in pts)


def plan_plants(M, keys, ctx):
    import numpy as np
    from core.surface.ground import Ground
    G = Ground(ctx["scene"]); rng = np.random.default_rng(int(keys.get("seed", M.get("seed") or 1)))
    n_tree, n_rock = int(keys.get("trees", 8)), int(keys.get("boulders", 10))
    centre, r_site = _footprint(ctx["scene"])
    nb = G.near_bounds(); centre = centre or [(nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2]
    r_in = float(keys.get("from", r_site + 12.0)); r_out = float(keys.get("to", r_in + 90.0))
    biome = G.loc.get("biome") or "temperate"
    mix = BIOME_TREES.get(biome, BIOME_TREES["temperate"])
    if keys.get("species"):
        names = [s.strip() for s in str(keys["species"]).split(",") if s.strip()]; mix = {k: 1.0 / len(names) for k in names}
    kinds = list(mix); wk = np.array([mix[k] for k in kinds], float); wk /= wk.sum()
    sl = float(keys.get("slope", 22.0))

    def ring(n, gap, slope):
        pts = G.spots(n * 4, center=centre, r=r_out, max_slope=slope, gap=gap, seed=int(rng.integers(1 << 30)), avoid=("road", "pad", "cliff"))
        pts = [p for p in pts if math.hypot(p[0] - centre[0], p[1] - centre[1]) >= r_in]
        if pts:                                                             # not on ground occupied by buildings (the occupancy layer, 2 m margin)
            from core import layers as L
            busy = L.occupied_at(ctx["scene"], np.array([p[0] for p in pts]), np.array([p[1] for p in pts]), pad=2.0)
            pts = [p for p, b in zip(pts, busy) if not b]
        return pts[:n]
    trees = [dict(x=round(float(x), 2), y=round(float(y), 2), z=round(float(z), 2), species=str(kinds[int(rng.choice(len(kinds), p=wk))]),
                  seed=int(rng.integers(1, 10 ** 6))) for x, y, z in ring(n_tree, float(keys.get("gap", 12.0)), sl)]
    k0, k1 = [float(v) for v in str(keys.get("size", "1.2,3.0")).split(",")]
    rocks = [dict(x=round(float(x), 2), y=round(float(y), 2), z=round(float(z), 2), kind=str(keys.get("boulder_kind", "rock")),
                  scale=round(float(rng.uniform(k0, k1)), 2), rotation=round(float(rng.uniform(0, 2 * math.pi)), 3),
                  seed=int(rng.integers(1, 10 ** 6))) for x, y, z in ring(n_rock, float(keys.get("boulder_gap", 6.0)), sl + 8)]
    return {"trees": trees, "boulders": rocks, "biome": biome, "centre": centre, "ring_m": [round(r_in, 1), round(r_out, 1)],
            "requested": [n_tree, n_rock]}


def _material(name, rgb, rough=0.9):
    import bpy
    m = bpy.data.materials.get(name)
    if m is None:
        m = bpy.data.materials.new(name); m.diffuse_color = (*rgb, 1.0)
        bs = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bs: bs.inputs["Base Color"].default_value = (*rgb, 1.0); bs.inputs["Roughness"].default_value = rough
    return m


def _lander(sc, G):
    """The ground z under a point: a ray from above onto the "terrain" object of the open scene; no object or a miss — the height from the terrain grid."""
    from mathutils import Vector
    ter = __import__("bpy").data.objects.get("terrain")
    inv = ter.matrix_world.inverted() if ter is not None else None

    def z_at(x, y):
        if ter is not None:
            o, d = inv @ Vector((x, y, 1e4)), inv.to_3x3() @ Vector((0, 0, -1))
            hit, loc, *_ = ter.ray_cast(o, d)
            if hit: return float((ter.matrix_world @ loc).z)
        return float(G.height(x, y))
    return z_at


def build_plants(ctx):
    import bpy
    import numpy as np
    from core.surface.ground import Ground
    from components.plants import bl_tree as T
    from components.plants import bl_boulder as BB
    sc, P, coll = ctx["sc"], ctx["plan"], ctx["collection"]
    G = Ground(ctx["scene"]); z_at = _lander(sc, G)
    stats = {"trees": 0, "boulders": 0, "vertices": 0, "failures": 0}; tall = []   # log statistics; tree heights
    for i, t in enumerate(P.get("trees") or []):
        try:
            z = z_at(t["x"], t["y"]); V_, F_, mats, off = [], [], [], 0
            for part, V, F in T.build_parts((t["x"], t["y"], z - 0.1), t["seed"], t["species"]):
                V_ += [tuple(v) for v in V]; F_ += [[off + j for j in f] for f in F]; off += len(V)
                mats += [0 if part == "trunk" else 1] * len(F)
            me = bpy.data.meshes.new(f"tree_{i}"); me.from_pydata(V_, [], F_); me.update(); me.shade_smooth()
            me.materials.append(_material("plants_bark", BARK)); me.materials.append(_material(f"plants_crown_{t['species']}", CROWN.get(t["species"], CROWN["oak"])))
            me.polygons.foreach_set("material_index", mats); me.update()
            ob = bpy.data.objects.new(f"tree_{i}", me); sc.collection.objects.link(ob); ob["kind"] = "tree"; ob["sk_kind"] = t["species"]
            stats["trees"] += 1; stats["vertices"] += len(V_); tall.append(round(max(v[2] for v in V_) - min(v[2] for v in V_), 1))
        except Exception as e:
            stats["failures"] += 1; print(f"tree {i} ({t['species']}): {type(e).__name__}: {str(e)[:160]}", flush=True)
    for i, r in enumerate(P.get("boulders") or []):
        try:
            dims = [d * r["scale"] for d in BB.dims_of(r["kind"])]
            o = BB.boulder(dims, r["seed"], r["kind"])
            z = z_at(r["x"], r["y"]); ca, sa = math.cos(r["rotation"]), math.sin(r["rotation"])
            V = np.array([v.co[:] for v in o.data.vertices]); F = [list(p.vertices) for p in o.data.polygons]
            V = np.column_stack([V[:, 0] * ca - V[:, 1] * sa + r["x"], V[:, 0] * sa + V[:, 1] * ca + r["y"], V[:, 2] + z - 0.03 * dims[2]])
            me0 = o.data; bpy.data.objects.remove(o); bpy.data.meshes.remove(me0)
            me = bpy.data.meshes.new(f"boulder_{i}"); me.from_pydata(V.tolist(), [], F); me.update(); me.shade_smooth()
            me.materials.append(_material("plants_rock", ROCK, 0.95))
            ob = bpy.data.objects.new(f"boulder_{i}", me); sc.collection.objects.link(ob); ob["kind"] = "rock"
            stats["boulders"] += 1; stats["vertices"] += len(V)
        except Exception as e:
            stats["failures"] += 1; print(f"boulder {i}: {type(e).__name__}: {str(e)[:160]}", flush=True)
    return {**stats, "tree_height_m": [min(tall), max(tall)] if tall else None, "biome": P.get("biome"), "ring_m": P.get("ring_m")}


# ---------------------------------------------------------------- forest and herb layer (vegetation)
def plan_forest(M, keys, ctx):
    """Step "forest", plan (project Python): forest.plan from the site data (density — a factor, seed, plot) + tree specifications
    (plants.resolve); ground occupied by buildings — the occupancy layer (the forest does not stand on yards, streets, the pad); canopy — the canopy layer."""
    from components.plants import forest as FO
    seed = int(keys.get("seed", M.get("seed") or 1)); dens = float(keys.get("density", 1.0))
    from core import layers as L
    box = tuple(float(v) for v in str(keys["area"]).split(",")) if keys.get("area") else None
    pool = [s.strip() for s in str(keys["kinds"]).split(",") if s.strip()] if keys.get("kinds") else None
    blocked = lambda X, Y: L.occupied_at(ctx["scene"], X, Y, L.BUILT, pad=3.0)   # ground occupied by buildings — no trees (the occupancy layer)
    R = FO.with_specs(FO.plan(ctx["scene"], dens, seed, box, pool=pool, blocked=blocked), season=keys.get("season"))
    if R["trees"]:                                                       # the final result — checked against occupancy once more: the spread and the growth reseeding
        import numpy as np                                               # move trees from the checked grid into yards
        busy = L.occupied_at(ctx["scene"], np.array([t["x"] for t in R["trees"]]), np.array([t["y"] for t in R["trees"]]), L.BUILT, pad=1.0)
        R["stats"]["culled_by_occupancy"] = int(busy.sum()); R["trees"] = [t for t, b in zip(R["trees"], busy) if not b]
    json.dump(R, open(os.path.join(ctx["scene"], "forest.json"), "w"), ensure_ascii=False)   # own plan: the meadow reads the canopy shade
    L.write(ctx["scene"], "canopy", {"trees": [{k: t.get(k) for k in ("x", "y", "cr", "h", "species", "habit")} for t in R["trees"]]})   # the canopy layer — for Grass
    return {"forest": R, "density": dens, "seed": seed, **{k: R["stats"].get(k) for k in ("trees_n", "shrubs_n", "stems_per_ha", "closure", "by_species", "culled_by_occupancy")}}


def build_forest(ctx):
    """Step "forest", build (Blender): bl_forest.build_forest — unique trees near the cameras, farther out instances of variants by tier
    (the tree's height in the frame by the layer's cameras; no cameras yet — from the middle of the plot: then the "forest at the cameras" step sets the tiers by the cameras).
    look: stand_in (draft) — stand-ins by the same plan: a crown by the species silhouette and a trunk, one object per species (bl_stand_in)."""
    from core.surface.ground import Ground
    G = Ground(ctx["scene"]); F = ctx["plan"].get("forest") or {"trees": []}; k = ctx["keys"]
    opts = {"near": float(k.get("unique_to", 22)), "max_unique": int(k.get("unique", 14))}
    ctx["sc"]["sk_forest_opts"] = json.dumps(opts)                         # "forest at the cameras" rebuilds with the same knobs
    if k.get("look") == "stand_in":
        from components.plants import bl_stand_in as BSI
        coll, cost = BSI.build(F, G, ctx["sc"], parent=ctx["collection"])
        return {"forest": cost, "trees": len(F["trees"])}
    from components.plants import bl_forest as BF
    coll, cost = BF.build_forest(F, G, BF.cams_of(ctx["sc"], G), ctx["sc"], parent=ctx["collection"], **opts)
    return {"forest": cost, "trees": len(F["trees"])}


def plan_forest_clear(M, keys, ctx):
    """Step "forest at the cameras", plan: by the camera layer (core.layers — written by Camera and take, including after the forest) — trees
    whose crown is at the camera (closer than the crown radius + margin) or in the near zone in front of it (up to front m, within the view cone, with the top above the camera)
    are removed; the canopy is rewritten without them. The order of steps does not matter: the forest is placed before the cameras, the clearing after them."""
    import numpy as np
    from core import layers as L
    fj = os.path.join(ctx["scene"], "forest.json"); F = json.load(open(fj)) if os.path.exists(fj) else {"trees": []}
    T = F.get("trees") or []; cams = L.cameras(ctx["scene"]); margin, front = float(keys.get("margin", 2.0)), float(keys.get("front", 12.0))
    if not T or not cams: return {"remove": [], "removed": 0, "cameras": len(cams), "_skipped": "no forest (forest.json) or camera layer"}
    X = np.array([t["x"] for t in T]); Y = np.array([t["y"] for t in T]); CR = np.array([t.get("cr", 2.0) for t in T])
    Z = np.array([t.get("z", 0.0) for t in T]); H = np.array([t.get("h", 5.0) for t in T]); hit = np.zeros(len(T), bool)
    for c in cams:
        Mw = np.asarray(c["matrix_world"], float); p = Mw[:3, 3]; f = -Mw[:3, 2]; f2 = f[:2] / max(np.linalg.norm(f[:2]), 1e-6)
        tan_h = 18.0 / float(c.get("lens_mm", 35.0)); dx, dy = X - p[0], Y - p[1]
        near = np.hypot(dx, dy) < CR + margin                            # the camera is in the crown or right next to it
        along = dx * f2[0] + dy * f2[1]; side = np.abs(dx * f2[1] - dy * f2[0])
        ahead = (along > 0) & (along < front) & (side < CR + along * tan_h + margin) & (Z + H > p[2])   # a screen in the near zone of the frame
        hit |= near | ahead
    rem = [int(i) for i in np.flatnonzero(hit)]
    keep = [t for i, t in enumerate(T) if not hit[i]]
    L.write(ctx["scene"], "canopy", {"trees": [{k: t.get(k) for k in ("x", "y", "cr", "h", "species", "habit")} for t in keep]})
    return {"remove": rem, "at": [[T[i]["x"], T[i]["y"]] for i in rem], "removed": len(rem), "cameras": len(cams)}


def build_forest_clear(ctx):
    """Step "forest at the cameras", build: the forest again by the layer's cameras (bl_forest.rebuild_forest: tiers — by the tree's height in the frame of these
    cameras; the orbit and take cameras do not exist yet at the forest step) without the plan's removed trees. look: stand_in (draft) — the same
    plan decision, the forest is rebuilt with stand-ins (bl_stand_in.rebuild)."""
    import bpy
    from core.surface.ground import Ground
    P = ctx["plan"]; fj = os.path.join(ctx["scene"], "forest.json")
    if P.get("_skipped") or bpy.data.collections.get("forest") is None or not os.path.exists(fj):
        return {"rebuilt": False, "_skipped": P.get("_skipped") or "no forest in the scene"}
    G = Ground(ctx["scene"]); F = json.load(open(fj)); sc = ctx["sc"]
    if ctx["keys"].get("look") == "stand_in":
        from components.plants import bl_stand_in as BSI
        coll, cost = BSI.rebuild(F, G, sc, skip=P.get("remove") or [])
        return {"rebuilt": True, "trees_removed": len(P.get("remove") or []), "forest": cost}
    from components.plants import bl_forest as BF
    opts = json.loads(sc.get("sk_forest_opts", "{}"))
    coll, cost = BF.rebuild_forest(F, G, BF.cams_of(sc, G), sc, skip=P.get("remove") or [], **opts)
    return {"rebuilt": True, "trees_removed": len(P.get("remove") or []), "forest": cost}


def plan_meadow(M, keys, ctx):
    """Step "meadow", plan: the species of the site biome's herb layer (plants.meadow_specs) — grasses, forbs, flowers, ferns; the forest (if the
    "forest" step ran) — canopy shade."""
    from core.surface.ground import Ground
    from components.plants import plants as PL
    G = Ground(ctx["scene"]); biome = G.loc.get("biome") or "temperate"; seed = int(keys.get("seed", M.get("seed") or 1))
    cov = os.path.exists(os.path.join(ctx["scene"], "cover.npz"))   # cover (Grass): grasses — variants by dryness, species — by the cover groups
    PL.textures(); S = PL.meadow_specs(biome, K=max(int(keys.get("variants", 3)), 3 if cov else 1), seed=seed, season=keys.get("season"), dry_tiers=cov)
    cs = keys.get("cards", "low_poly"); cards = None                  # card tier: the set exists on disk — enabled
    if cs and cs != "off":
        import glob, core.paths
        src = os.path.join("library", "plants", "cards", str(cs))
        if glob.glob(os.path.join(core.paths.ROOT, src, "*.glb")):
            cards = {"source": src, "px": [3.0, float(keys.get("card_px", 60))], "px_herb": float(keys.get("herb_px", 20)),
                     "overlap": float(keys.get("card_overlap", 6)), "width_px": 24.0, "budget": 60000}
    return {"species": S, "biome": biome, "seed": seed, "tufts_m2": float(keys.get("density", 40)), "reach_m": float(keys.get("to", 70)),
            "budget": int(keys.get("budget", 40000)), "cards": cards}


def build_meadow(ctx):
    """Step "meadow", build: bl_meadow.build_meadow near the layer's cameras (the cameras of the scene and take — core.layers.cameras; none — the .blend cameras):
    in the camera frame, by 3D distance, in tiers; shade — the canopy layer (canopy.json)."""
    from core import layers
    from core.surface.ground import Ground
    from components.plants import bl_meadow as BM
    G = Ground(ctx["scene"]); P = ctx["plan"]; sc = ctx["sc"]; cams = layers.cameras(ctx["scene"])
    if not cams:
        for o in [o for o in sc.objects if o.type == "CAMERA"]:
            cams.append({"name": o.name, "matrix_world": [list(r) for r in o.matrix_world], "lens_mm": o.data.lens, "sensor_width_mm": o.data.sensor_width,
                         "resolution": [sc.render.resolution_x, sc.render.resolution_y]})
    cn = layers.canopy(ctx["scene"]); trees = cn.get("trees") if isinstance(cn, dict) else cn
    coll, cost = BM.build_meadow(G, cams, P.get("species") or {}, sc, canopy=trees or [], rho=float(P.get("tufts_m2", 40)), reach=float(P.get("reach_m", 70)),
                                 K=max(len(v) for v in (P.get("species") or {"x": [0]}).values()), seed=int(P.get("seed", 1)),
                                 budget=int(P.get("budget", 40000)), parent=ctx["collection"], cards=P.get("cards"))
    return {"meadow": cost, **({"_skipped": cost["_skipped"]} if cost.get("_skipped") else {})}   # if empty, the step says why
