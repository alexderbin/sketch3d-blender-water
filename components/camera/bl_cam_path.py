"""Collision-free camera path — checks (the camera no closer than the gap to buildings, the target visible in every key) on the keys
of the movement in components/camera/cam_move.py. No path search (RRT*) is needed: the movement is given (push-in, orbit,
crane), and only a bad key is repaired — with the first of the lifts and pulls toward the target that passes (the smallest shift), so that
the camera looks at the target again; then the path is smoothed (the mean of the neighbours), and a smoothed key stays only if it passes the checks.
blender -b --factory-startup -P core/blender/run.py -- components/camera/bl_cam_path.py -- scenes/<name> <a-b> <center x,y,z> [hero=m1] [base=K] [gap=0.3]
hero — visibility by the points of its grid (without it — the centre and ±0.45 m); base — the original movement camera: a key is repaired
only if it sees worse than that camera does (the hero may already be hidden in the original frame — that is not a path error).
Writes the scene's cameras.json; the answer is the __RESULT__ line: what was repaired, what failed, the smallest gap."""
import json, math, os, sys
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from core.blender import bl_scene as B

a = sys.argv[sys.argv.index("--") + 1:]
SD = os.path.abspath(a[0]); k0, k1 = (int(x) for x in a[1].split("-")); C = Vector([float(x) for x in a[2].split(",")])
opt = dict(kv.split("=", 1) for kv in a[3:]); GAP = float(opt.get("gap", 0.3))
cp = os.path.join(SD, "cameras.json"); cams = json.load(open(cp))
sc, statics, members, objs, Wd, _ = B.load_scene(SD); dg = bpy.context.evaluated_depsgraph_get()
V, P = [], []                                                       # one BVH over buildings and ground (no figures): the camera gap
for ob in statics:
    if ob.type != "MESH" or ob.name.startswith("figure_"): continue
    me = ob.evaluated_get(dg).to_mesh(); base = len(V); M = ob.matrix_world
    V += [M @ v.co for v in me.vertices]; P += [[base + i for i in p.vertices] for p in me.polygons]; ob.evaluated_get(dg).to_mesh_clear()
bvh = BVHTree.FromPolygons(V, P)
hero = bpy.data.objects.get(f"figure_{opt['hero']}") if opt.get("hero") else None
if hero:                                                            # target — the hero's grid points (for a crouching hero the "chest" lies above the head)
    hv = [hero.matrix_world @ v.co for v in hero.evaluated_get(dg).data.vertices]; TGT = hv[::max(1, len(hv) // 40)]
else:
    TGT = [C, C + Vector((0, 0, 0.45)), C - Vector((0, 0, 0.45))]   # target: the centre and points above and below it — visibility is measured on them


def clearance(p):
    hit = bvh.find_nearest(p); return (hit[3] if hit[0] is not None else 1e9)


def seen(p):
    """Fraction of target points that the ray reaches: the first hit is at the target itself (0.8 m) or a figure (a wall in the way — no)."""
    ok = 0
    for t in TGT:
        d = t - p; L = d.length
        hit, loc, nrm, idx, ob, _ = sc.ray_cast(dg, p, d / L, distance=L)
        ok += (not hit) or (loc - p).length >= L - (0.3 if hero else 0.8) or (ob == hero if hero else ob.name.startswith("figure_"))
    return ok / len(TGT)


base_seen = seen(Matrix(cams[int(opt["base"]) - 1]["matrix_world"]).translation) if opt.get("base") else 1.0
NEED = min(0.66, 0.85 * base_seen)                                   # no stricter than the original movement camera


def good(p): return clearance(p) >= GAP and seen(p) >= NEED


def look(p, aim):
    """Camera matrix at p looking at point aim, with no horizon roll."""
    f = (aim - p).normalized(); r = f.cross(Vector((0, 0, 1))); r = r.normalized() if r.length > 1e-6 else Vector((1, 0, 0)); u = r.cross(f)
    return Matrix(((r.x, u.x, -f.x, p.x), (r.y, u.y, -f.y, p.y), (r.z, u.z, -f.z, p.z), (0, 0, 0, 1)))


keys = list(range(k0, k1 + 1)); pos = {k: Matrix(cams[k - 1]["matrix_world"]).translation.copy() for k in keys}
aim = {}                                                             # where the key's camera was looking: the point on its axis at the centre's distance
for k in keys:
    M_ = Matrix(cams[k - 1]["matrix_world"]); aim[k] = pos[k] - M_.col[2].to_3d().normalized() * (C - pos[k]).length
fixed, failed, moved = [], [], {}
for k in keys:
    p = pos[k]
    if good(p): continue
    best = None
    for dz in (0, 0.5, 1.0, 2.0, 3.0, 5.0):                      # smallest shift: a lift, a pull toward the target, and both together
        for t in (0, 0.15, 0.3, 0.45):
            if dz == 0 and t == 0: continue
            q = p + (C - p) * t + Vector((0, 0, dz))
            if good(q) and (best is None or (q - p).length < (best - p).length): best = q
    if best is None: failed.append(k); continue
    pos[k] = best; moved[k] = round((best - p).length, 2); fixed.append(k)
near = {j for k in fixed for j in (k - 1, k, k + 1) if k0 < j < k1 and j not in failed}
for k in sorted(near):                                               # smoothing of repaired keys: the mean with the neighbours, if it passes
    q = (pos[k - 1] + pos[k] * 2 + pos[k + 1]) / 4
    if good(q): pos[k] = q
for k in sorted(near | set(fixed)):                                  # a shifted key looks where it looked before (the same framing)
    c = cams[k - 1]; c["matrix_world"] = [list(r) for r in look(pos[k], aim[k])]; c["location"] = list(pos[k])
if fixed:
    if os.path.islink(cp): import shutil; real = os.path.realpath(cp); os.remove(cp); shutil.copy(real, cp)
    json.dump(cams, open(cp, "w"), ensure_ascii=False, indent=1)
print("__RESULT__" + json.dumps({"keys": len(keys), "fixed": fixed, "moved_m": moved, "failed": failed,
                                 "min_clearance": round(min(clearance(pos[k]) for k in keys), 2),
                                 "visible_all": all(seen(pos[k]) >= NEED for k in keys), "base_seen": round(base_seen, 2), "need": round(NEED, 2)}, ensure_ascii=False))
