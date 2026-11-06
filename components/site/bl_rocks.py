"""Site rocks in Blender by the rocks.json plan (terrain_rocks.py): Poly Haven scans (library/terrain/rocks.yaml → library/rocks/polyhaven)
as Geometry Nodes instances — one object per area (rocks_near, rocks_belt). A plan layer → scan roles (rocks.yaml → layers);
a prototype is a scan variant scaled to a 1 m diameter with the bottom at zero; the Poly Haven LOD level by layer, simpler still in the distance
(Decimate down to far faces). An instance's tint is the tone of the site's rock under it (relief_w: tone_rock/tone_cliff × dilution; like the small items
near the camera and the ground scans), the tilt — toward the ground normal by the fraction tilt, sunk by the fraction sink of its height.
Building blocks: the Site's small items (bl_terrain_clutter: LOD variants, tint, rotation) and the core (core/blender/bl_kit)."""
import core.paths         # the code is here, the core — core/<part> through kernel
import json, math, os, time

import bpy
import numpy as np
from mathutils import Matrix

ROOT = core.paths.ROOT
IDX = os.path.join(ROOT, "library", "rocks", "polyhaven", "index.json")
CFG = os.path.join(ROOT, "library", "terrain", "rocks.yaml")


def _normalized(me, name):
    """A copy of the mesh: its face (the mean normal of the faces, if the scan is open at the back — a wall, an outcrop) toward +X, a horizontal diameter of 1 m,
    centred on the axis, with the bottom at zero. → (mesh, height per 1 m of diameter, whether it has a face)."""
    nr = np.empty(len(me.polygons) * 3); me.polygons.foreach_get("normal", nr); ar = np.empty(len(me.polygons)); me.polygons.foreach_get("area", ar)
    nm = (nr.reshape(-1, 3) * ar[:, None]).sum(0) / max(ar.sum(), 1e-9); face = bool(np.hypot(nm[0], nm[1]) > 0.15)
    m2 = me.copy(); m2.name = name
    if face: m2.transform(Matrix.Rotation(-math.atan2(nm[1], nm[0]), 4, "Z"))   # the face toward +X
    co = np.empty(len(m2.vertices) * 3); m2.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    lo, hi = co.min(0), co.max(0); k = 1.0 / max(hi[0] - lo[0], hi[1] - lo[1], 1e-6)
    m2.transform(Matrix.Scale(k, 4) @ Matrix.Translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2])))
    return m2, float((hi[2] - lo[2]) * k), face


def _cover_material(m):
    """The rock grows into the slope (like the blending of rocks with the ground in engines — a world-aligned blend): faces that look up take the tone
    of the ground under the instance (attribute gtone) by the fraction cover (attribute); the sides keep their own scan. Once per material."""
    if m is None or not m.use_nodes or m.get("sk_cover"): return
    nt = m.node_tree; bs = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bs is None or not bs.inputs["Base Color"].is_linked: return
    src = bs.inputs["Base Color"].links[0].from_socket; N = nt.nodes.new; L = nt.links
    geo = N("ShaderNodeNewGeometry"); sp = N("ShaderNodeSeparateXYZ"); L.new(geo.outputs["Normal"], sp.inputs[0])
    up = N("ShaderNodeMapRange"); up.interpolation_type = "SMOOTHSTEP"; up.inputs["From Min"].default_value, up.inputs["From Max"].default_value = 0.55, 0.9
    L.new(sp.outputs["Z"], up.inputs["Value"])
    cv = N("ShaderNodeAttribute"); cv.attribute_type = "INSTANCER"; cv.attribute_name = "cover"
    f = N("ShaderNodeMath"); f.operation = "MULTIPLY"; L.new(up.outputs[0], f.inputs[0]); L.new(cv.outputs["Fac"], f.inputs[1])
    gt = N("ShaderNodeAttribute"); gt.attribute_type = "INSTANCER"; gt.attribute_name = "gtone"
    mx = N("ShaderNodeMix"); mx.data_type = "RGBA"; ins = {s_.identifier: s_ for s_ in mx.inputs}
    L.new(f.outputs[0], ins["Factor_Float"]); L.new(src, ins["A_Color"]); L.new(gt.outputs["Color"], ins["B_Color"])
    L.new(next(s_ for s_ in mx.outputs if s_.identifier == "Result_Color"), bs.inputs["Base Color"])
    m["sk_cover"] = 1


def _decimated(me, faces, name):
    """A simpler mesh: Decimate down to ~faces faces (in the distance a rock is tens of pixels)."""
    if len(me.polygons) <= faces: return me
    ob = bpy.data.objects.new("_dec", me); bpy.context.scene.collection.objects.link(ob)
    md = ob.modifiers.new("dec", "DECIMATE"); md.ratio = max(faces / len(me.polygons), 0.002)
    dg = bpy.context.evaluated_depsgraph_get(); m2 = bpy.data.meshes.new_from_object(ob.evaluated_get(dg)); m2.name = name
    for m in me.materials: m2.materials.append(m)
    bpy.data.objects.remove(ob); return m2


def build(G, R, sc=None, seed=11, parent=None):
    """R — the rocks.json plan; G — the site's core.surface.ground.Ground; parent — the step's collection (the manager: the scene section), otherwise the scene
    root. → a summary."""
    import yaml
    from core.blender import bl_kit as BC
    from core.surface import bl_terrain_surf as TS
    from components.site import bl_terrain_clutter as TC
    t0 = time.time(); sc = sc or bpy.context.scene; rng = np.random.default_rng(seed)
    if not R.get("places"): return {"rocks": 0}
    if not os.path.exists(IDX): return {"rocks": "no rock scans: .venv/bin/python -m components.site.terrain_clutter fetch --cfg library/terrain/rocks.yaml --out library/rocks/polyhaven"}
    cfg = yaml.safe_load(open(CFG)); idx = json.load(open(IDX)); W = np.load(os.path.join(G.folder, "relief_w.npz"))
    xs, ys = W["xs"], W["ys"]
    coll = bpy.data.collections.get("rocks") or bpy.data.collections.new("rocks")
    host = parent or sc.collection                                  # under the step's collection: the frame cost and the step's contribution are reported per component
    if coll.name not in host.children: host.children.link(coll)
    cache, out = {}, {}

    def protos(role, lod, far):                                     # scan variants of a role → [(mesh, height, mean)]
        key = (role, lod, far)
        if key in cache: return cache[key]
        A = cfg["assets"][role]; keys, meshes, _ = TC._protos(A, idx)
        res = []
        for v in range(len(keys)):
            me, h, face = _normalized(meshes[v * 4 + min(lod, 3)], f"rock_{role}_{v}_l{lod}")
            if far: me = _decimated(me, far, me.name + "_far")
            for m in me.materials: TC._tint_material(m); _cover_material(m)
            res.append((me, h, idx[A["id"]].get("mean") or [0.15, 0.13, 0.11], face))
        cache[key] = res; return res

    by = {}
    for p in R["places"]: by.setdefault(p["region"], []).append(p)
    for region, P in by.items():
        meshes, H, means, faces, rows, LS = [], [], [], [], [], {}
        for lay in sorted({p["layer"] for p in P}):
            L = (cfg.get("layers") or {}).get(lay)
            if not L: continue
            far = L.get("far") if region == "belt" else None
            pro = [q for role in L["roles"] if role in cfg["assets"] and cfg["assets"][role]["id"] in idx for q in protos(role, int(L.get("lod", 2)), far)]
            if not pro: continue
            base = len(meshes); meshes += [q[0] for q in pro]; H += [q[1] for q in pro]; means += [q[2] for q in pro]; faces += [q[3] for q in pro]
            LS[lay] = L
            for p in (q for q in P if q["layer"] == lay): rows.append((p, base + int(rng.integers(len(pro)))))
        if not rows: continue
        x = np.array([p["x"] for p, _ in rows]); y = np.array([p["y"] for p, _ in rows]); s = np.array([p["s"] for p, _ in rows])
        pidx = np.array([i for _, i in rows]); flat = np.array([float(LS[p["layer"]].get("flatten", 1.0)) for p, _ in rows])
        h = np.array(H)[pidx] * s * flat
        z = np.array([p["z"] for p, _ in rows]) - h * np.array([float(LS[p["layer"]].get("sink", p.get("sink", 0.3))) for p, _ in rows])
        n = np.asarray(G.normal(x, y), float).reshape(-1, 3)
        yaw = np.radians([p.get("rot", 0.0) for p, _ in rows])
        out_ = np.array([bool(LS[p["layer"]].get("face_out")) and faces[i] for p, i in rows])   # a wall, an outcrop: facing along the dip of the slope
        if out_.any():                                              # (cliff rocks as modules facing outward)
            jit = rng.normal(0, 0.25, len(x)); yaw = np.where(out_, np.arctan2(-n[:, 1], -n[:, 0]) + np.pi + jit, yaw)
        eul = TC._euler(n, np.array([float(LS[p["layer"]].get("tilt", p.get("tilt", 0.5))) for p, _ in rows]), yaw)
        tone = np.zeros((len(x), 3)); wsum = np.zeros(len(x))       # the site rock tone under the instance (like the small items: the layer tone × dilution)
        fx = np.clip(np.interp(x, xs, np.arange(len(xs))), 0, len(xs) - 1).astype(int); fy = np.clip(np.interp(y, ys, np.arange(len(ys))), 0, len(ys) - 1).astype(int)
        for l in ("rock", "cliff", "talus"):
            if f"tone_{l}" in W.files and l in W.files:
                c = W[l][fy, fx].astype(np.float32) + 1e-3; tone += c[:, None] * W[f"tone_{l}"][None, :]; wsum += c
        tone = tone / np.maximum(wsum, 1e-6)[:, None]
        V = W["var"][fy, fx].astype(np.float32) if "var" in W.files else np.ones((len(x), 3))
        bright = np.array([float(LS[p["layer"]].get("bright", 1.0)) for p, _ in rows])   # weathered rock is lighter than grass (albedo 0.25–0.4)
        tint = np.vstack([TC._gains(tone[i:i + 1], means[pidx[i]], TS.TINT_L, TS.TINT_C) for i in range(len(x))]) * V * bright[:, None]
        gt = np.zeros((len(x), 3)); gw = np.zeros(len(x))             # the ground tone under the rock (base, dry earth, scree) — for the top of the rock
        for l in ("base", "dirt", "talus"):
            if f"tone_{l}" in W.files and l in W.files:
                c = W[l][fy, fx].astype(np.float32) + 1e-3; gt += c[:, None] * W[f"tone_{l}"][None, :]; gw += c
        gt = gt / np.maximum(gw, 1e-6)[:, None] * V
        cover = np.array([float(LS[p["layer"]].get("cover", 0.3)) for p, _ in rows])
        proto = BC.proto_collection(f"rocks_{region}_p", meshes, coll); BC.exclude(bpy.context.view_layer, proto)
        ob = BC.points_object(f"rocks_{region}", coll, np.column_stack([x, y, z]),
                              {"pidx": ("INT", pidx), "rot": ("FLOAT_VECTOR", eul), "sclv": ("FLOAT_VECTOR", np.column_stack([s, s, s * flat])),
                               "tint": ("FLOAT_VECTOR", np.clip(tint, 0.05, 10.0)), "gtone": ("FLOAT_VECTOR", gt), "cover": ("FLOAT", cover)})
        g = BC.Graph(f"rocks_{region}_gn")
        ci = g.new("GeometryNodeCollectionInfo", transform_space="ORIGINAL"); ci.inputs["Collection"].default_value = proto
        ci.inputs["Separate Children"].default_value = True; ci.inputs["Reset Children"].default_value = True
        ipn = g.new("GeometryNodeInstanceOnPoints"); g.put(ipn.inputs["Points"], g.gin.outputs[0]); g.put(ipn.inputs["Instance"], ci.outputs["Instances"])
        ipn.inputs["Pick Instance"].default_value = True; g.put(ipn.inputs["Instance Index"], g.attr("pidx", "INT"))
        er = g.new("FunctionNodeEulerToRotation"); g.put(er.inputs["Euler"], g.attr("rot", "FLOAT_VECTOR")); g.put(ipn.inputs["Rotation"], er.outputs["Rotation"])
        g.put(ipn.inputs["Scale"], g.attr("sclv", "FLOAT_VECTOR")); g.put(g.gout.inputs[0], ipn.outputs["Instances"])
        md = ob.modifiers.new("rocks", "NODES"); md.node_group = g.ng; ob["kind"] = "rock"; ob["sk_layer"] = "site"
        out[region] = {"n_instances": int(len(x)), "n_prototypes": len(meshes), "n_faces_within_prototypes": int(sum(len(m.polygons) for m in meshes))}
    return {"rocks": out, "s_rocks": round(time.time() - t0, 1)}
