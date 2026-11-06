"""The forest as stand-ins in Blender (draft; silhouettes and colours — stand_in.py): for each plan tree — a crown as a solid of revolution from the species' measurement and a trunk
(core primitives core/blender/stand_in.py: lathe, cone, place, build), the tilt — by rotation about the base, as for bl_forest instances;
everything is one object per species (forest_stand_in_<species>: stand-in labels sk_stand_in, sk_kind tree; no sk_id — a composite object
of the class): Workbench pays for objects, not for faces. build — the "forest" step, rebuild — "forest at the cameras" (the previous forest is removed, as in
bl_forest.rebuild_forest). parts — vertices without Blender: the test bench (bench/stand_in_bench.py) measures the box from them."""
import math, time

import numpy as np

from components.plants import stand_in as PS
from core.blender import stand_in as SI

N_CROWN, N_TRUNK = 12, 6                                             # sides of the crown (a multiple of 4: the box is exactly 2r) and of the trunk
SINK = 0.08                                                          # sunk into the ground — like bl_forest instances


def _tilt(V, base, deg, az):
    """Points V — rotated about the base: the tree axis leans from the vertical by deg toward the azimuth az (a bl_forest instance does the same)."""
    if not deg: return V
    th, a = math.radians(deg), math.radians(az); k = (-math.sin(a), math.cos(a), 0.0)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]]); R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
    return (V - base) @ R.T + base


def parts(t, z, cal=None):
    """A stand-in of plan tree t on the ground z: the SI.place parts (the crown by the species silhouette; the trunk — below the crown, or whole if there is no crown)
    in the world and the "measured" flag."""
    s = PS.shape(t, cal); x, y = float(t["x"]), float(t["y"]); base = np.array([x, y, z]); P = []
    if s["profile"]:
        P.append(SI.place(SI.lathe(s["profile"], N_CROWN), at=(x + s["shift"][0], y + s["shift"][1], z), rgb=s["rgb"], smooth=True))
    if s["trunk"]:
        r0, r1, ht = s["trunk"]; P.append(SI.place(SI.cone(r0, ht, N_TRUNK, r_top=r1), at=(x, y, z), rgb=s["bark"], kind="trunk"))
    for p in P: p["V"] = _tilt(p["V"], base, *s["lean"])
    return P, s["calibrated"]


def build(F, G, sc=None, parent=None, skip=()):
    """A forest of stand-ins from forest.json F on the ground G (core.surface.ground.Ground); the skip trees (plan numbers) are not placed (clearing
    near the cameras); parent — the section collection. → (the forest collection, cost: trees, objects, faces, triangles, per tree, without measurement, s)."""
    import bpy
    sc = sc or bpy.context.scene; t0 = time.time(); T = F.get("trees") or []; parent = parent or sc.collection; skip = set(int(i) for i in skip)
    coll = bpy.data.collections.new("forest"); parent.children.link(coll)
    ii = [i for i in range(len(T)) if i not in skip]; cal = PS.calibration(); by, unc = {}, set()
    Z = G.height(np.array([T[i]["x"] for i in ii], float), np.array([T[i]["y"] for i in ii], float)) - SINK if ii else []
    for i, z in zip(ii, Z):
        P, ok = parts(T[i], float(z), cal); by.setdefault(T[i]["species"], []).extend(P)
        if not ok: unc.add(T[i]["species"])
    faces = tris = 0
    for sid, P in sorted(by.items()):
        ob = SI.build(f"forest_stand_in_{sid}", P, coll, kind="tree"); ob["plant_species"] = sid
        faces += sum(len(p["F"]) for p in P); tris += sum(len(f) - 2 for p in P for f in p["F"])
    n = len(ii)
    return coll, {"look": "stand_in", "trees": n, "objects": len(by), "faces": faces, "tris": tris, "faces_per_tree": round(faces / n) if n else 0,
                  "uncalibrated": sorted(unc), "s": round(time.time() - t0, 1)}


def rebuild(F, G, sc=None, parent=None, skip=()):
    """Rebuilding with stand-ins (the "forest at the cameras" step): the previous collections forest and forest_protos (stand-ins or a tiered forest) are removed
    with their objects, meshes and scatter nodes (as in bl_forest.rebuild_forest); the forest is built again by build without the skip trees. → (collection, cost)."""
    import bpy
    sc = sc or bpy.context.scene
    for name in ("forest", "forest_protos"):
        c = bpy.data.collections.get(name)
        if c is None: continue
        if parent is None and name == "forest": parent = next((p for p in bpy.data.collections if c.name in p.children), None)
        for o in list(c.all_objects):
            ng = next((m.node_group for m in o.modifiers if m.type == "NODES" and m.node_group), None)
            me = o.data if o.type == "MESH" else None
            bpy.data.objects.remove(o, do_unlink=True)
            if ng is not None and ng.users == 0: bpy.data.node_groups.remove(ng)
            if me is not None and me.users == 0: bpy.data.meshes.remove(me)
        bpy.data.collections.remove(c)
    return build(F, G, sc, parent=parent, skip=skip)
