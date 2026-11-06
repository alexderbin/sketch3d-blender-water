"""View coverage (with a ray occlusion check): what
fraction of the surfaces of a site or an object the cameras show, and which views to add — greedily, up to a fraction or a count.
Points are taken on faces (by area); a point counts as seen if the ray from the camera reaches it first, the angle to the face is under 75°, and the point is in the frame.
blender -b --factory-startup -P core/blender/run.py -- components/camera/bl_cover.py -- scenes/<name> [cameras=1-5] [add=N] [frac=0.9] [obj=<id>] [write=1]
Without 'obj' — the site: everything except the ground, the sky and figures, under 60 m. Candidates are 48 directions of the upper hemisphere,
the distance from the target's box (lens 24 mm, 16:9). write=1 — append the chosen views to cameras.json. The answer is the __RESULT__ line."""
import json, math, os, random, sys
import bpy
from mathutils import Matrix, Vector
from core.blender import bl_scene as B

a = sys.argv[sys.argv.index("--") + 1:]; SD = os.path.abspath(a[0]); opt = dict(kv.split("=", 1) for kv in a[1:])
cp = os.path.join(SD, "cameras.json"); cams = json.load(open(cp)) if os.path.exists(cp) else []
sc, statics, members, objs, Wd, _ = B.load_scene(SD); dg = bpy.context.evaluated_depsgraph_get(); rng = random.Random(5)
BIG = ("ground", "terrain", "floor", "skyline", "sky", "water", "ridge")
if opt.get("obj"):
    tgt_obs = [o for o in statics if o.name.split(".")[0] == opt["obj"]]
else:
    tgt_obs = [o for o in statics if o.type == "MESH" and not o.name.startswith(("figure_", "ground")) and not o.name.split(".")[0].startswith(BIG)
               and max(o.dimensions) < 60]
tri = []                                                            # points on faces by area, with the normal
for o in tgt_obs:
    me = o.evaluated_get(dg).to_mesh(); M = o.matrix_world; N3 = M.to_3x3().inverted().transposed()
    for p in me.polygons:
        if p.area > 1e-6: tri.append((p.area * max(abs(M.determinant()), 1e-9) ** (2 / 3), M @ p.center, (N3 @ p.normal).normalized()))
    o.evaluated_get(dg).to_mesh_clear()
tot = sum(t[0] for t in tri); NP = 2500
pts = [(c, n) for w, c, n in rng.choices(tri, weights=[t[0] for t in tri], k=NP)] if tri else []
lo = Vector([min(c[i] for c, _ in pts) for i in range(3)]); hi = Vector([max(c[i] for c, _ in pts) for i in range(3)])
ctr = (lo + hi) / 2; rad = (hi - lo).length / 2
COS75 = math.cos(math.radians(75))


def seen_by(M, lens, res):
    """Indices of the points visible to camera M: in the frame, the ray reaches them, the angle to the face < 75°."""
    p0 = M.translation; Mi = M.inverted(); w, h = res; sw = 36.0; tx = sw / 2 / lens; ty = tx * h / w if w >= h else tx; tx = tx if w >= h else tx * w / h
    out = set()
    for i, (c, n) in enumerate(pts):
        q = Mi @ c
        if q.z >= -0.05 or abs(q.x / -q.z) > tx or abs(q.y / -q.z) > ty: continue
        d = c - p0; L = d.length; d /= L
        if n.dot(-d) < COS75: continue
        hit, loc, _, _, ob, _ = sc.ray_cast(dg, p0, d, distance=L + 0.05)
        if hit and (loc - c).length < max(0.05, 0.01 * L): out.add(i)
    return out


def cam_on(dir_):
    """Camera by direction: the distance is such that the whole target fits in the frame with a 10 % angle margin."""
    lens = 24.0; half = math.atan(36.0 * 720 / 1280 / 2 / lens) * 0.9; dist = rad / math.sin(half)
    p = ctr + dir_ * dist; f = (ctr - p).normalized(); r = f.cross(Vector((0, 0, 1))).normalized(); u = r.cross(f)
    return Matrix(((r.x, u.x, -f.x, p.x), (r.y, u.y, -f.y, p.y), (r.z, u.z, -f.z, p.z), (0, 0, 0, 1))), lens


have = set(); per = {}
if opt.get("cameras"):
    k0, k1 = (int(x) for x in opt["cameras"].split("-")) if "-" in opt["cameras"] else (int(opt["cameras"]),) * 2
    for k in range(k0, k1 + 1):
        c = cams[k - 1]; s_ = seen_by(Matrix(c["matrix_world"]), c.get("lens_mm", 50), c.get("resolution", [1280, 720])); per[k] = round(len(s_) / max(NP, 1), 3); have |= s_
base_cov = len(have) / max(NP, 1)
cand = []
for i in range(48):                                                 # Fibonacci points on the upper hemisphere (elevation 8–75°)
    z = math.sin(math.radians(8)) + (math.sin(math.radians(75)) - math.sin(math.radians(8))) * (i + 0.5) / 48
    az = i * math.pi * (3 - math.sqrt(5)); r_ = math.sqrt(1 - z * z); d_ = Vector((r_ * math.cos(az), r_ * math.sin(az), z))
    M, lens = cam_on(d_); cand.append((M, lens, seen_by(M, lens, [1280, 720]), round(math.degrees(math.atan2(d_.y, d_.x))), round(math.degrees(math.asin(z)))))
added, cov = [], base_cov
for _ in range(int(opt.get("add", 0))):                      # greedy: the view that adds the most unseen area
    best = max(cand, key=lambda c: len(c[2] - have))
    gain = len(best[2] - have) / max(NP, 1)
    if gain < 0.01 or cov >= float(opt.get("frac", 0.9)): break
    have |= best[2]; cov = len(have) / max(NP, 1); added.append({"azimuth": best[3], "elevation": best[4], "gain": round(gain, 3), "coverage": round(cov, 3), "M": best[0], "lens": best[1]})
if opt.get("write") == "1" and added:
    for j, a_ in enumerate(added, 1):
        p = a_["M"].translation
        cams.append({"name": f"coverage {j}: azimuth {a_['azimuth']}°, elevation {a_['elevation']}° (+{a_['gain']:.0%})", "location": list(p), "lens_mm": a_["lens"],
                     "sensor_width_mm": 36.0, "sensor_fit": "AUTO", "shift_x": 0.0, "shift_y": 0.0, "resolution": [1280, 720],
                     "matrix_world": [list(r) for r in a_["M"]]})
    if os.path.islink(cp): import shutil; real = os.path.realpath(cp); os.remove(cp); shutil.copy(real, cp)
    json.dump(cams, open(cp, "w"), ensure_ascii=False, indent=1)
print("__RESULT__" + json.dumps({"points": NP, "objects": len(tgt_obs), "coverage_given": round(base_cov, 3), "per_camera": per,
                                 "added": [{k: v for k, v in x.items() if k not in ("M", "lens")} for x in added], "coverage": round(cov, 3)}, ensure_ascii=False))
