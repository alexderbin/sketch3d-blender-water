"""Entry of the "Site" component for the step-by-step build (core/blender/build.py, core/manager/plan_run.py): the "relief" step.
plan (project Python): a catalogue preset + seed + knobs → relief.npz, location.json into the run scene (the "site" section),
the ground texture relief_tex.png (terrain_tex.bake). build (Blender): ground, water, landmarks, bridges into the "site" collection."""
import json, os


def plan_relief(M, keys, ctx):
    from components.site import terrain
    from components.site import terrain_tex
    name = keys.get("preset", "alpine_pass"); seed = int(keys.get("seed", M.get("seed") or 1))
    knobs = {k: v for k, v in keys.items() if k not in ("preset", "seed", "ground", "far", "size", "res")}
    plan = terrain.preset(name, seed, **knobs)
    for k in ("size", "res"):                                        # the near grid: size (m) and step (m) on top of the preset
        if keys.get(k) is not None: plan["relief"][k] = float(keys[k])
    if isinstance(keys.get("far"), dict):                            # belt mountains on top of the preset (real ones, a mosaic of kinds)
        R = plan["relief"]; R["far"] = {**(R.get("far") or {}), **keys["far"]}
        if R["far"].get("kind") in ("real", "mosaic") and "snowline" not in keys: R.pop("snowline", None)   # snow — from the mountain height
    o = terrain.build(plan)
    terrain.save(o, ctx["scene"])                                    # relief.npz, location.json — the model's "site" section
    note = None
    try:
        terrain_tex.bake(ctx["scene"])                               # relief_tex.png/json — the ground colour by biome
    except Exception as e:                                           # without a texture the ground takes the site colour (bl_relief.material)
        note = f"the ground texture was not baked: {type(e).__name__}: {str(e)[:120]}"
    L = json.load(open(os.path.join(ctx["scene"], "location.json")))
    from components.site import refresh                              # the biome's surface and small-item recipe (surface.json) — for the ground and Grass
    refresh.surface_recipe(ctx["scene"], L.get("biome") or plan.get("biome") or "temperate", seed)
    return {"preset": name, "seed": seed, "knobs": knobs, "ground": keys.get("ground", "scans"), "biome": L.get("biome") or plan.get("biome"),
            "far_z": L.get("far_z", 0.0), "anchors": sorted(L.get("anchors") or {})[:40], "note": note}


SURFACE = {"scans": True, "vista": "vista", "baked": "baked"}       # the ground knob → bl_relief.build_terrain(surface=…)


def build_relief(ctx):
    from components.site import bl_relief
    from core.surface.ground import Ground
    G = Ground(ctx["scene"]); sc = ctx["sc"]
    mode = ctx["plan"].get("ground", "scans"); sc["sk_ground"] = mode   # ground: scans up close | far-view shader everywhere | picture (the rebuild api does the same)
    ter, info = bl_relief.build_terrain(G, sc=sc, surface=SURFACE.get(mode, True))
    made = {"terrain": ter.name}
    for fn, name in ((bl_relief.build_water, "water"), (bl_relief.build_landmarks, "landmarks"), (bl_relief.build_bridges, "bridges")):
        try:
            r = fn(G, sc=sc)
            if r is not None: made[name] = getattr(r, "name", str(type(r).__name__))
        except Exception as e:
            made[name] = f"no: {type(e).__name__}: {str(e)[:80]}"
    return {"biome": ctx["plan"].get("biome"), "far_z": G.far_z, "site": ctx["scene"], "objects": made,
            "vertices": len(ter.data.vertices) if ter is not None else 0}


def plan_rocks(M, keys, ctx):
    """Step "rocks", plan (Python): boulders, cliff faces, outcrops, scree — from the relief data of the near plot and the mountain
    belt (terrain_rocks); not inside buildings (the Buildings occupancy layer, if present) → rocks.json in the run's scene."""
    from components.site import terrain_rocks
    from core import layers
    seed = int(keys.get("seed", M.get("seed") or 11))
    R = terrain_rocks.plan(ctx["scene"], seed, float(keys.get("density", 1.0)), str(keys.get("belt", "0")) != "0")
    near = [p for p in R["places"] if p["region"] == "near"]
    if near:                                                        # occupancy — near plot only (the buildings are there; do not measure the 3 km belt with a mask)
        busy = layers.occupied_at(ctx["scene"], [p["x"] for p in near], [p["y"] for p in near], pad=2.0)
        drop = {id(p) for p, b in zip(near, busy) if b}
        R["places"] = [p for p in R["places"] if id(p) not in drop]; R["stats"]["within_buildings"] = len(drop)
    json.dump(R, open(os.path.join(ctx["scene"], "rocks.json"), "w"), ensure_ascii=False)
    by = {}
    for p in R["places"]: by[p["region"]] = by.get(p["region"], 0) + 1
    return {"seed": seed, "places": by, "stats": R["stats"], "crags": str(keys.get("crags", "1")) != "0"}


def build_rocks(ctx):
    """Step "rocks", build (Blender): Poly Haven scans as instances (bl_rocks) from rocks.json; belt rocks as a wall shape
    (bl_crags: bedding ledges, couloirs, buttresses on the belt grid near the frame camera; key crags)."""
    from components.site import bl_rocks
    from core.surface.ground import Ground
    rj = os.path.join(ctx["scene"], "rocks.json")
    if not os.path.exists(rj): return {"skipped": "no rocks.json — the step plan did not run"}
    G = Ground(ctx["scene"]); pl = ctx.get("plan") or {}
    res = bl_rocks.build(G, json.load(open(rj)), sc=ctx["sc"], seed=int(pl.get("seed", 11)), parent=ctx.get("collection"))
    if pl.get("crags", True):
        import bpy
        from components.site import bl_crags
        md = bl_crags.build(bpy.data.objects.get("terrain_belt"), os.path.join(ctx["scene"], "relief_far.png"), G.loc.get("strata"), int(pl.get("seed", 11)), hole=G.near_bounds())
        res = dict(res or {}, crags=md is not None)
    return res
