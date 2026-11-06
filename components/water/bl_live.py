"""The scene's live water. Assembly (step water): water for each scene camera — a mesh from the camera — by the step plan (water
bodies, wind, the substance in the water) and the Site relief (level by points: rivers downhill, lakes, the sea to the horizon);
it hides the Site's water from the render (it remains for the sketch). The engine numbers are in the object property
water_params. When a scene is opened by the frames output (the light hook, components/water/api.py), the engine is rebuilt from
these numbers (nothing goes to disk): on a frame change — the time, before a render — the water of the current camera is
updated to the frame's moment and only it is visible."""
import json
import math

import bpy
import numpy as np

from components.water import optics as O
from components.water import waves as W
from components.water.bl_water import Water, near_for

LIVE = {"waters": [], "t": 0.0, "handlers": []}


def ground_of(folder):
    """The Site relief of the scene folder → (ground height f(x, y), water level f(x, y) → z | NaN, the reference level, the sea level
    | None). The level is extended past the waterline by 3 nodes (the water edge goes under the shore, as in the Site); beyond that
    — NaN (there is no water)."""
    from scipy import ndimage
    from core.surface.ground import Ground
    G = Ground(folder)
    L = np.asarray(G.level, float); ok = np.isfinite(L)
    if ok.any():
        idx = ndimage.distance_transform_edt(~ok, return_distances=False, return_indices=True)
        near = ndimage.distance_transform_edt(~ok) <= 3
        L = np.where(near, L[tuple(idx)], np.nan)
    feats = G.loc.get("features") or []
    sea = any(f.get("id") == "sea" or str(f.get("id", "")).endswith("_sea") or f.get("sea") for f in feats)
    ref = float(np.nanmedian(G.level)) if ok.any() else 0.0
    sea_lv = ref if sea else None
    ny, nx = L.shape; x0, y0, r = float(G.nx0), float(G.ny0), float(G.nres); x1, y1 = x0 + (nx - 1) * r, y0 + (ny - 1) * r

    def level_fn(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        i = np.clip(np.round((y - y0) / r).astype(int), 0, ny - 1); j = np.clip(np.round((x - x0) / r).astype(int), 0, nx - 1)
        out = np.where(inside, L[i, j], np.nan)
        return np.where(~inside, sea_lv, out) if sea_lv is not None else out
    return (lambda x, y: G.height(np.asarray(x, float), np.asarray(y, float))), level_fn, ref, sea_lv


def params(body, wind, cam, folder, kinds, seed=1, grid=480, resolution=None):
    """The camera's water numbers for a plan's water body (water.json) — everything needed to rebuild the engine."""
    k = kinds[body["kind"]]
    lvl = body.get("level")
    return {"camera": cam, "folder": folder, "seed": int(seed), "grid": int(grid), "kind": body["kind"], "resolution": resolution,
            "sea": {"u10": max(float(wind["u10"]), 0.3), "wind_to": list(wind["to"]), "fetch": body["fetch"],
                    "depth": None if body["kind"] == "sea" else max(float(body["depth_mean"]), 0.2), "chop": float(k.get("chop", 1.0))},
            "optics": body["optics"], "gust": k.get("gust"), "level": lvl}


def make(sc, prm, sun=None, coll=None):
    """The camera's water from the numbers prm: engine, mesh, material; → Water (object water_<camera>)."""
    cam = bpy.data.objects.get(prm["camera"])
    if cam is None or cam.type != "CAMERA": return None
    ground, level_fn, ref, sea_lv = ground_of(prm["folder"])
    keep = (sc.render.resolution_x, sc.render.resolution_y)
    if prm.get("resolution"): sc.render.resolution_x, sc.render.resolution_y = prm["resolution"]   # the frame point — by the camera
    s = prm["sea"]
    sea = W.Sea(u10=s["u10"], wind_to=tuple(s["wind_to"]), fetch=s["fetch"], depth=s["depth"], chop=s["chop"],
                near=near_for(cam, sc), seed=prm["seed"])
    look = O.preset(prm["optics"])
    lv0 = ref if prm.get("level") is None else float(prm["level"])
    w = Water(sc, sea, cam=cam, level=lv0, name=f"water_{prm['camera']}", grid=(prm["grid"], int(prm["grid"] * 0.75)),
              body=look["body"], K=look["K"], ground=ground, level_fn=level_fn, gust=prm.get("gust"), sun=sun, coll=coll)
    sc.render.resolution_x, sc.render.resolution_y = keep
    w.obj["water_params"] = json.dumps(prm, ensure_ascii=False); w.obj["sk_id"] = f"water_{prm['camera']}"; w.obj["sk_kind"] = "water"
    w.cam_name = prm["camera"]
    return w


def sun_of(P):
    """The sun of the light plan → (toward the sun, strength) for the light through the crests and the caustic."""
    if not P or not P.get("sun"): return None
    d = P["sun"].get("dir"); rgb = P["sun"].get("lamp_rgb") or [0.0]
    return (tuple(float(c) for c in d), float(max(rgb))) if d and max(rgb) > 0 else None


def build(sc, plan, cams, coll, kinds, sun=None):
    """Step assembly: water for each camera (cams — cameras.json entries: name, resolution); the Site's water is removed from the
    render. → a summary."""
    bodies = plan.get("bodies") or []
    if not bodies: return {"water": "no water"}
    hidden = 0
    for o in sc.objects:                                                # the Site's water is a sketch placeholder, in the render ours is used
        if o.get("kind") == "water" and not o.get("water_params"): o.hide_render = True; hidden += 1
    main = bodies[0]; made = []
    for c in cams:
        w = make(sc, params(main, plan["wind"], c["name"], plan["folder"], kinds, resolution=c.get("resolution")), sun=sun, coll=coll)
        if w is None: continue
        w.update(0.0); w.obj.hide_render = c["name"] != (sc.camera.name if sc.camera else None); made.append(w.obj.name)
    return {"water_objects": len(made), "site_water_hidden": hidden, "kind": main["kind"], "hs": main.get("hs")}


def _on_frame(scene, depsgraph=None):
    LIVE["t"] = scene.frame_current / max(scene.render.fps / scene.render.fps_base, 1e-3)


def _on_render(scene, depsgraph=None):
    cam = scene.camera.name if scene.camera else None
    for w in LIVE["waters"]:
        on = w.cam_name == cam; w.obj.hide_render = not on
        if on and getattr(w, "t_done", None) != LIVE["t"]:
            w.update(LIVE["t"]); w.t_done = LIVE["t"]


def attach(sc, P=None):
    """The light hook of the frames output: the scene's water rebuilt from the object numbers (the engine writes nothing to disk),
    frame handlers. → the number of water bodies."""
    objs = [o for o in sc.objects if o.get("water_params")]
    if not objs: return 0
    prms = [json.loads(o["water_params"]) for o in objs]; coll = objs[0].users_collection[0] if objs[0].users_collection else sc.collection
    for o in objs:
        me = o.data; bpy.data.objects.remove(o)
        if me is not None and me.users == 0: bpy.data.meshes.remove(me)
    sun = sun_of(P)
    LIVE["waters"] = [w for w in (make(sc, p, sun=sun, coll=coll) for p in prms) if w is not None]
    for h in LIVE["handlers"]:
        for lst in (bpy.app.handlers.frame_change_pre, bpy.app.handlers.render_pre):
            if h in lst: lst.remove(h)
    bpy.app.handlers.frame_change_pre.append(_on_frame); bpy.app.handlers.render_pre.append(_on_render)
    LIVE["handlers"] = [_on_frame, _on_render]
    _on_frame(sc); _on_render(sc)
    return len(LIVE["waters"])
