"""Trees for a pipeline scene: the tree skeleton package as is (components/plants/ext/tree_skeletons: a genotype of 15 knobs, season, pine, shrub;
one seed — always the same tree), the skinning is our own, a curve with the package's thickness formula r = clamp((reverse depth × 0.02)^1.5),
the crown — spheres at the ends of the branches (a silhouette and depth for the maps).
blender -b --factory-startup -P core/blender/run.py -- components/plants/bl_tree.py -- <bench/scenes/X> <oak|oak_winter|dead|pine|shrub> <x,y[,z][;x,y…]> [seed]
Without z the tree stands on the scene ground (a ray from top to bottom, figures do not count): scenes need not lie at zero (urban_squad — 255 m).
The tree is appended to bench/scenes/X/scene.obj as the part tree_<n> of the kind "tree" (world.json); then make_frame / frames.py."""
import core.paths         # the code is here, the core — core/<part> through kernel
import json, math, os, sys
core.paths.component_ext("plants", "tree_skeletons")
import bpy
import numpy as np
import loader as L

SCALE = 0.35                       # their gotcha: "trees in strange units", 0.35 converts to metres (GenericTreeFactory.scale)
KEYS = ["size", "trunk_warp", "n_trunks", "branch_start", "branch_angle", "multi_branch", "branch density", "branch_len",
        "branch_warp", "pull_dir_vt", "pull_dir_hz", "outgrowth", "branch_thickness", "twig_density", "twig_scale"]


def genome(**kw):
    g = np.random.RandomState(7).rand(32)
    for k, v in kw.items(): g[KEYS.index(k)] = v
    return g


OAK = genome(size=0.75, trunk_warp=0.45, n_trunks=0.1, branch_start=0.3, branch_angle=0.45, multi_branch=0.6, **{"branch density": 0.85},
             branch_len=0.75, branch_warp=0.45, outgrowth=0.9, branch_thickness=0.6)
DEAD = genome(size=0.6, trunk_warp=0.9, n_trunks=0.1, branch_start=0.35, branch_angle=0.7, **{"branch density": 0.6}, branch_len=0.6,
              branch_warp=0.9, outgrowth=0.6, branch_thickness=0.5)
KINDS = {"oak": (lambda: L.configs.random_tree(OAK, season="summer")[0], 0.5, 1.6), "oak_winter": (lambda: L.configs.random_tree(OAK, season="winter")[0], 0, 1.6),
         "dead": (lambda: L.configs.random_tree(DEAD, season="winter")[0], 0, 2.0), "pine": (lambda: L.configs.pine_tree()[0], 0.45, 1.0),
         "shrub": (lambda: L.configs.shrub()[0], 0.25, 1.0)}


def norm(p):
    if hasattr(p, "skeleton"): return p.skeleton, p.trunk_spacecol, p.roots_spacecol, dict(p.skinning or {})
    return p["config"], p.get("space_kargs"), p.get("root_kargs"), dict(p.get("radii_kargs") or {})


def build_parts(pos, seed, kind="oak"):
    """Skeleton → a curve with a per-point radius → a mesh; the crown — spheres at the branch ends → a mesh. Parts in the world: [(name, vertices, faces)] —
    "trunk" (bark) and "crown" (the crown, if the species has one). Works in an open scene and cleans up only its own objects (the step-by-step build)."""
    make, leaf, rscale = KINDS[kind]
    sc = bpy.context.scene
    had = {k: set(getattr(bpy.data, k)) for k in ("objects", "meshes", "curves")}
    np.random.seed(seed); sk, tr, rt, skin = norm(make())
    obj = L.tree.tree_skeleton(sk, tr, rt, init_pos=(0, 0, 0), scale=SCALE)
    me = obj.data; V = np.array([v.co[:] for v in me.vertices]); E = np.array([e.vertices[:] for e in me.edges]) if len(me.edges) else np.zeros((0, 2), int)
    rev = np.zeros(len(V)); at = me.attributes.get("rev_depth")
    if at is not None and at.data_type == "INT": at.data.foreach_get("value", rev)
    r = np.clip((rev * skin.get("Scaling", 0.2) * 0.1) ** skin.get("Exponent", 1.5), skin.get("Min radius", 0.02), skin.get("Max radius", 5.0)) * rscale
    adj = [[] for _ in range(len(V))]
    for x, y in E: adj[x].append(y); adj[y].append(x)
    seen = np.zeros(len(V), bool); cd = bpy.data.curves.new("trunk", "CURVE"); cd.dimensions = "3D"; cd.bevel_depth, cd.bevel_resolution, cd.use_fill_caps = 1.0, 2, True
    for s in range(len(V)):                                         # polylines from the ends: after they are parsed, a point has ≤ 1 child
        if seen[s] or len(adj[s]) != 1: continue
        path, prev, cur = [s], -1, s; seen[s] = True
        while True:
            nxt = [q for q in adj[cur] if q != prev and not seen[q]]
            if not nxt:                                             # extend to the fork: otherwise the branch hangs a step away from its parent
                path += [q for q in adj[cur] if q != prev and seen[q]][:1]; break
            prev, cur = cur, nxt[0]; seen[cur] = True; path.append(cur)
        sp = cd.splines.new("POLY"); sp.points.add(len(path) - 1)
        for k, vi in enumerate(path): sp.points[k].co = (*V[vi], 1); sp.points[k].radius = float(r[vi])
    trunk = bpy.data.objects.new("trunk", cd); sc.collection.objects.link(trunk); parts = [trunk]
    if leaf:
        tips = V[rev == 0]; mt = bpy.data.meshes.new("tips"); mt.from_pydata([tuple(p) for p in tips], [], [])
        holder = bpy.data.objects.new("tips", mt); sc.collection.objects.link(holder)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=leaf); ball = bpy.context.active_object; ball.parent = holder
        holder.instance_type = "VERTS"; parts.append(holder)
    bpy.data.objects.remove(obj); bpy.context.view_layer.update(); dg = bpy.context.evaluated_depsgraph_get()
    out = []
    for o in parts:
        VV, FF = [], []
        ev = o.evaluated_get(dg)
        if o.instance_type == "VERTS":                              # crown copies — from the depsgraph instance list
            for inst in dg.object_instances:
                if inst.is_instance and inst.parent and inst.parent.original == holder:
                    m = inst.object.to_mesh(); off = len(VV)
                    VV += [tuple(inst.matrix_world @ v.co) for v in m.vertices]; FF += [[off + i for i in p.vertices] for p in m.polygons]
                    inst.object.to_mesh_clear()
            out.append(("crown", VV, FF)); continue
        m = ev.to_mesh(); off = len(VV); VV += [tuple(ev.matrix_world @ v.co) for v in m.vertices]; FF += [[off + i for i in p.vertices] for p in m.polygons]
        ev.to_mesh_clear(); out.append(("trunk", VV, FF))
    for o in [o for o in bpy.data.objects if o not in had["objects"]]: bpy.data.objects.remove(o)   # only its own: other objects of the scene stay intact
    for k in ("meshes", "curves"):
        for d in [d for d in getattr(bpy.data, k) if d not in had[k] and d.users == 0]: getattr(bpy.data, k).remove(d)
    return [(n, np.array(VV) + np.array(pos), FF) for n, VV, FF in out]


def build(pos, seed, kind="oak"):
    """A tree as one mesh (bark and crown): vertices and faces in the world — as main appends it to scene.obj."""
    VV, FF, off = [], [], 0
    for _n, V, F in build_parts(pos, seed, kind):
        VV += [tuple(v) for v in V]; FF += [[off + i for i in f] for f in F]; off += len(V)
    return np.array(VV), FF


def main():
    a = sys.argv[sys.argv.index("--") + 1:]
    SD, KIND, POS = os.path.abspath(a[0]), a[1], [tuple(map(float, p.split(","))) for p in a[2].split(";")]
    SEED = int(a[3]) if len(a) > 3 else 1
    sc = bpy.context.scene
    for o in list(sc.objects): bpy.data.objects.remove(o)
    if any(len(p) == 2 for p in POS):                               # ground height under the point: a ray from above over the scene without figures
        bpy.ops.wm.obj_import(filepath=os.path.join(SD, "scene.obj"), forward_axis="Y", up_axis="Z")
        for o in list(sc.objects):
            if o.name.startswith("figure_"): bpy.data.objects.remove(o)
        dg = bpy.context.evaluated_depsgraph_get(); top = max((o.matrix_world @ __import__("mathutils").Vector(c))[2] for o in sc.objects for c in o.bound_box) + 10
        ground = []
        for p in POS:
            if len(p) == 3: ground.append(p); continue
            hit, loc, *_ = sc.ray_cast(dg, (p[0], p[1], top), (0, 0, -1))
            ground.append((p[0], p[1], loc[2] if hit else 0.0))
        POS = ground
        for o in list(sc.objects): bpy.data.objects.remove(o)
    from core.blender import bl_scene as B
    stats, parts = [], []
    for i, pos in enumerate(POS):
        VV, FF = build(pos, SEED + i, KIND); parts.append(("tree", "tree", VV, FF))
        stats.append({"verts": len(VV), "height_m": round(float(np.ptp(VV[:, 2])), 1)})
    for st, pid in zip(stats, B.append_parts(SD, parts)): st["id"] = pid
    print("__RESULT__" + json.dumps({"kind": KIND, "trees": stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
