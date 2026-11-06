"""Small items near the camera: grass tufts, weeds, pebbles, rocks, twigs, moss — Poly Haven scans (library/terrain/clutter.yaml, downloaded with
components/site/terrain_clutter.py fetch), as Geometry Nodes instances (bl_crowd: points → prototypes by index). Where — only in the field of
view of the cameras up to reach m; how many — the biome layer's density × the layer's share (the same rules as in the ground shader: terrain_tex.layer_weights)
× moisture (grass is denser when wet, pebbles when dry), full up close, zero toward reach; the variant — dry or wet by the site's moisture;
the detail level — by the distance to the nearest camera; the tilt — toward the ground normal; the instance's tint — the ground colour under it
(the layer tone × dilution V; vegetation keeps its own tint). Called from bl_relief (clutter=1) after the patch near the camera."""
import core.paths         # the code is here, the core — core/<part> through kernel
import json, math, os, re, sys, time

import bpy
import numpy as np

ROOT = core.paths.ROOT
VEG = ("grass", "grass_short", "weeds", "moss")


def _grid(W, k, xs, ys, x, y):
    """A value of the relief_w.npz grid (nodes xs, ys are non-uniform) at points: bilinear by indices."""
    A = W[k].astype(np.float32); fx = np.interp(x, xs, np.arange(len(xs))); fy = np.interp(y, ys, np.arange(len(ys)))
    i0, j0 = np.clip(fx.astype(int), 0, len(xs) - 2), np.clip(fy.astype(int), 0, len(ys) - 2); tx, ty = fx - i0, fy - j0
    if A.ndim == 3: tx, ty = tx[:, None], ty[:, None]
    return A[j0, i0] * (1 - tx) * (1 - ty) + A[j0, i0 + 1] * tx * (1 - ty) + A[j0 + 1, i0] * (1 - tx) * ty + A[j0 + 1, i0 + 1] * tx * ty


def _visible(P, cams, reach):
    """A point in the field of view of at least one camera closer than reach: (mask, distance to the nearest camera that sees it)."""
    dmin = np.full(len(P), np.inf)
    for c in cams:
        M = np.array(c.matrix_world); e = M[:3, 3]; R = M[:3, :3]; q = (P - e) @ R          # in camera axes: x is right, y is up, −z is forward
        z = -q[:, 2]; hx = 36.0 / (2 * c.data.lens); hy = hx * bpy.context.scene.render.resolution_y / max(bpy.context.scene.render.resolution_x, 1)
        ok = (z > 0.3) & (np.abs(q[:, 0]) < z * hx * 1.08) & (np.abs(q[:, 1]) < z * hy * 1.15)
        d = np.linalg.norm(P - e, axis=1); dmin = np.where(ok & (d < dmin), d, dmin)
    return dmin < reach, dmin


def _euler(n, tilt, yaw):
    """Rotation: the up axis toward the normal n by the fraction tilt, then a rotation about it by yaw → XYZ Euler angles (like Euler to Rotation)."""
    up = np.array([0.0, 0.0, 1.0]); t = up + (n - up) * tilt[:, None]; t /= np.linalg.norm(t, axis=1, keepdims=True)
    ax = np.cross(np.broadcast_to(up, t.shape), t); s = np.linalg.norm(ax, axis=1); c = t[:, 2]
    ax = np.where(s[:, None] > 1e-9, ax / np.maximum(s, 1e-9)[:, None], np.array([1.0, 0, 0]))
    Kx = np.zeros((len(t), 3, 3)); Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 2] = -ax[:, 2], ax[:, 1], -ax[:, 0]; Kx -= Kx.transpose(0, 2, 1)
    Rt = np.eye(3)[None] + s[:, None, None] * Kx + (1 - c)[:, None, None] * (Kx @ Kx)              # Rodrigues: up → t
    cz, sz = np.cos(yaw), np.sin(yaw); Rz = np.zeros((len(t), 3, 3)); Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = cz, -sz, sz, cz, 1
    R = Rt @ Rz
    return np.stack([np.arctan2(R[:, 2, 1], R[:, 2, 2]), np.arcsin(np.clip(-R[:, 2, 0], -1, 1)), np.arctan2(R[:, 1, 0], R[:, 0, 0])], 1)


def _gains(target, mean, kl, kc):
    """Fitting to the ground tone, as for ground scans (bl_terrain_surf._gain), for all instances at once."""
    lum = np.array([0.2126, 0.7152, 0.0722]); m = np.maximum(np.asarray(mean, float), 1e-4); lm = max(float(m @ lum), 1e-4)
    lt = np.maximum(target @ lum, 1e-4)[:, None]
    return np.clip(np.power(lt / lm, kl) * np.power(np.clip((target / lt) / (m / lm), 0.2, 5.0), kc), 0.25, 10.0)   # (the colour of Poly Haven grass
    # is dark, 0.025: a ×4 limit left the tufts as black blotches)


def _protos(A, idx):
    """Meshes of a Poly Haven asset: variants (group, letter) × LOD0…3 levels (a missing level — the nearest one); the group is part of the object name."""
    path = os.path.join(ROOT, idx[A["id"]]["blend"])
    with bpy.data.libraries.load(path, link=False, relative=False) as (src, dst):
        dst.meshes = []; dst.objects = [n for n in src.objects if not re.search(r"geonodes|geometry", n)]
    var = {}
    for o in dst.objects:
        if o is None or o.type != "MESH": continue
        rest = o.name[len(A["id"]) + 1:] if o.name.startswith(A["id"] + "_") else o.name
        m = re.search(r"_?LOD(\d)$", rest); lod = int(m.group(1)) if m else 0; key = rest[:m.start()] if m else rest
        var.setdefault(key or "_", {})[lod] = o.data
    for o in list(dst.objects):
        if o is not None: bpy.data.objects.remove(o)
    keys = sorted(var); meshes = []
    for k in keys:
        L = var[k]
        for lod in range(4): meshes.append(L.get(lod) or L[min(L, key=lambda q: abs(q - lod))])
    groups = {g: [i for i, k in enumerate(keys) if k.split("_")[0] == g or k.startswith(g)] for g in sum((A.get("groups") or {}).values(), [])}
    return keys, meshes, groups


def _tint_material(m):
    """Colour × the instance's tint (the tint attribute on a GN instance; Attribute — type INSTANCER)."""
    if m is None or not m.use_nodes or m.get("sk_tint"): return
    nt = m.node_tree; bs = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    sock = bs.inputs["Base Color"] if bs is not None else next((i for n in nt.nodes if n.type == "GROUP" for i in n.inputs
                                                              if i.name in ("Color", "Base Color", "Albedo", "Diffuse")), None)
    if sock is None: return                                         # (Poly Haven grass has a shader that is a node group with a Color input)
    src = sock.links[0].from_socket if sock.is_linked else None
    at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_type = "INSTANCER"; at.attribute_name = "tint"
    mx = nt.nodes.new("ShaderNodeMix"); mx.data_type = "RGBA"; mx.blend_type = "MULTIPLY"
    ins = {s.identifier: s for s in mx.inputs}; ins["Factor_Float"].default_value = 1.0
    if src is not None: nt.links.new(src, ins["A_Color"])
    else: ins["A_Color"].default_value = sock.default_value
    nt.links.new(at.outputs["Color"], ins["B_Color"]); nt.links.new(next(s for s in mx.outputs if s.identifier == "Result_Color"), sock)
    grp = next((n for n in nt.nodes if n.type == "GROUP" and "Dead Amount" in n.inputs), None)
    if grp is not None:                                             # grass: light follows the ground normal under the tuft (as for undergrowth:
        g = grp.node_tree.copy(); grp.node_tree = g                 # the lighting uses the normal of the heightfield to look the same as
        g.interface.new_socket("GroundN", in_out="INPUT", socket_type="NodeSocketVector")   # the terrain): otherwise upright cards
        gi = next(n for n in g.nodes if n.type == "GROUP_INPUT")    # are darker than the flat ground — dots on a carpet
        for n in g.nodes:
            if n.type in ("BSDF_PRINCIPLED", "BSDF_TRANSLUCENT"): g.links.new(gi.outputs["GroundN"], n.inputs["Normal"])
        an = nt.nodes.new("ShaderNodeAttribute"); an.attribute_type = "INSTANCER"; an.attribute_name = "gn"; nt.links.new(an.outputs["Vector"], grp.inputs["GroundN"])
    dead = next((i for n in nt.nodes if n.type == "GROUP" for i in n.inputs if i.name == "Dead Amount"), None)
    if dead is not None:                                            # Poly Haven grass: "dry" is a ready input of their shader, the share is the site's dryness
        ad = nt.nodes.new("ShaderNodeAttribute"); ad.attribute_type = "INSTANCER"; ad.attribute_name = "dead"; nt.links.new(ad.outputs["Fac"], dead)
    if bs is not None and "Specular IOR Level" in bs.inputs: bs.inputs["Specular IOR Level"].default_value = min(bs.inputs["Specular IOR Level"].default_value, 0.35)
    m["sk_tint"] = 1


def build(G, cams, cfg, seed=11, masks=None, snowline=None, sc=None, cap=160000):
    """Small items near the site G's cameras by the biome config (terrain_clutter.resolve → surface.json): returns a summary (instances per asset, time)."""
    from core.blender import bl_kit as BC
    from core.surface import bl_terrain_surf as TS
    from components.site import terrain_tex as TT
    t0 = time.time(); sc = sc or bpy.context.scene
    ip = os.path.join(ROOT, "library", "clutter", "polyhaven", "index.json")
    if not cfg or not os.path.exists(ip): return {"clutter": "no config (surface.json → clutter) or scans (components/site/terrain_clutter.py fetch)"}
    idx = json.load(open(ip)); dens = cfg["density"]
    reach, full, lods = float(cfg["reach"]), float(cfg["full"]), cfg["lod"]
    W = np.load(os.path.join(G.folder, "relief_w.npz")); xs, ys = W["xs"].astype(np.float64), W["ys"].astype(np.float64)
    from core.surface.bl_terrain_near import footprint
    boxes = [(P.min(0) - 2, P.max(0) + 2) for P in (footprint(G, c, reach) for c in cams) if len(P) >= 4]
    if not boxes: return {"clutter": 0}
    SC = TS.scans(); rng = np.random.default_rng(seed); assets = {a for v in dens.values() for a in v}
    out, coll = {}, bpy.data.collections.new("clutter"); sc.collection.children.link(coll)
    for name in sorted(assets):
        A = cfg["assets"][name]
        if A["id"] not in idx: continue
        dmax = max(v.get(name, 0.0) for v in dens.values()); cell = float(np.clip(0.75 / math.sqrt(max(dmax, 1e-3)), 0.12, 3.0))
        pts = []
        for lo, hi in boxes:                                        # candidates — a jittered grid within the camera footprint box
            gx, gy = np.meshgrid(np.arange(lo[0], hi[0], cell), np.arange(lo[1], hi[1], cell))
            pts.append(np.column_stack([gx.ravel(), gy.ravel()]) + rng.uniform(0, cell, (gx.size, 2)))
        Q = np.unique(np.round(np.concatenate(pts) / cell * 0.999).astype(np.int64), axis=0, return_index=True)[1]
        XY = np.concatenate(pts)[Q]
        if len(XY) == 0: continue
        Z = np.asarray(G.top(XY[:, 0], XY[:, 1]), float); P3 = np.column_stack([XY, Z])
        vis, d = _visible(P3, cams, reach); XY, Z, d = XY[vis], Z[vis], d[vis]
        if len(XY) == 0: continue
        x, y = XY[:, 0], XY[:, 1]; S = np.asarray(G.slope(x, y), float)
        g_ = lambda k, dflt=0.0: _grid(W, k, xs, ys, x, y) if k in W.files else np.full(len(x), dflt, np.float32)
        M = g_("a_moist", 0.5); plan = {k: g_(f"a_{k}") for k in ("bank", "bed", "pad", "pave", "road", "cliff")}
        _, cov = TT.layer_weights(x, y, S, M, g_("a_expo"), g_("a_foot"), g_("a_gully"), plan, masks, Z, snowline, seed)
        p = sum(cov[l] * v.get(name, 0.0) for l, v in dens.items() if l in cov)
        p = p * ((0.4 + 1.2 * M) if name in VEG else (1.4 - 0.8 * M))   # grass is denser when wet, pebbles when dry
        p = p * (1 - TT.smoothstep(full, reach, d)) * cell * cell
        wet = np.asarray(G.water(x, y), float) if hasattr(G, "water") else np.zeros(len(x))
        p = np.where((wet > 0.02) | (S > 55), 0.0, p)
        keep = rng.uniform(size=len(p)) < np.clip(p, 0, 1)
        if os.environ.get("CLUTTER_DEBUG"): print("CLUTTER", name, "n_candidates", len(np.concatenate(pts)), "visible", len(x), "p", float(np.mean(p)), "cov", {k: round(float(v.mean()), 3) for k, v in cov.items()}, "gully", float(g_("a_gully").mean()), "expo", float(g_("a_expo").mean()), "M", float(M.mean()), "d", float(np.median(d)), flush=True)
        if not keep.any(): continue
        x, y, Z, d, M, S = x[keep], y[keep], Z[keep], d[keep], M[keep], S[keep]
        keys, meshes, groups = _protos(A, idx); nv = len(keys)
        if nv == 0: continue
        if groups:                                                  # variant: dry or wet by the site's moisture
            gd = sum((groups.get(g, []) for g in A["groups"].get("dry", [])), []); gw = sum((groups.get(g, []) for g in A["groups"].get("wet", [])), [])
            wetp = TT.smoothstep(0.38, 0.66, M); pickw = rng.uniform(size=len(x)) < wetp
            v = np.where(pickw & bool(gw), np.array(gw or [0])[rng.integers(0, max(len(gw), 1), len(x))],
                         np.array(gd or list(range(nv)))[rng.integers(0, max(len(gd), 1) if gd else nv, len(x))])
        else: v = rng.integers(0, nv, len(x))
        lod = np.searchsorted(np.asarray(lods, float), d); pidx = v * 4 + np.clip(lod, 0, 3)
        gx_, gy_ = G.grad(x, y); n = np.column_stack([-np.asarray(gx_), -np.asarray(gy_), np.ones(len(x))]); n /= np.linalg.norm(n, axis=1, keepdims=True)
        eul = _euler(n, np.full(len(x), float(A.get("tilt", 0.5))), rng.uniform(0, 2 * math.pi, len(x)))
        if A.get("flat"): eul[:, 0] += math.pi / 2                   # twigs lie flat (the scans stand upright)
        s0, s1 = A.get("scale", [0.8, 1.2]); scl = rng.uniform(s0, s1, len(x))
        if name in VEG: scl = scl * (0.75 + 0.5 * M)                # taller when wet
        scl = scl * (1 + 0.6 * TT.smoothstep(full, reach, d))      # sparser but larger in the distance — the coverage holds
        lay_cov = {l: c[keep] for l, c in cov.items()}
        tone = np.zeros((len(x), 3)); wsum = np.zeros(len(x))       # the ground colour under the instance: the layer tone × dilution
        for l, c in lay_cov.items():
            if f"tone_{l}" in W.files: tone += c[:, None] * W[f"tone_{l}"][None, :]; wsum += c
        tone = tone / np.maximum(wsum, 1e-6)[:, None]
        V = _grid(W, "var", xs, ys, x, y) if "var" in W.files else np.ones((len(x), 3))
        mean = idx[A["id"]].get("mean") or [0.2, 0.2, 0.2]
        tint = _gains(tone, mean, TS.TINT_L, TS.TINT_VEG if name in VEG else TS.TINT_C) * V * (1.15 if name in VEG else 1.0)   # a tuft shades
        # itself: slightly lighter than the ground tone
        dead = np.clip((0.55 - M) * 2.2 + rng.normal(0, 0.12, len(x)), 0, 0.85) if name in VEG else np.zeros(len(x))   # dry — withered grass
        if len(x) > cap:
            sel = rng.choice(len(x), cap, replace=False); x, y, Z, pidx, eul, scl, tint, dead, n = x[sel], y[sel], Z[sel], pidx[sel], eul[sel], scl[sel], tint[sel], dead[sel], n[sel]
        for me in set(meshes):
            for m in me.materials: _tint_material(m)
        proto = BC.proto_collection(f"clutter_{name}_p", meshes, coll); BC.exclude(bpy.context.view_layer, proto)
        if name in VEG:                                             # grass casts no shadow (like small vegetation in engines): shadows
            for o in proto.objects: o.visible_shadow = False        # of tufts are dark dots on the meadow
        ob = BC.points_object(f"clutter_{name}", coll, np.column_stack([x, y, Z - 0.02]),
                              {"pidx": ("INT", pidx), "rot": ("FLOAT_VECTOR", eul), "sclv": ("FLOAT_VECTOR", np.repeat(scl[:, None], 3, 1)),
                               "tint": ("FLOAT_VECTOR", np.clip(tint, 0.05, 10.0)), "dead": ("FLOAT", dead), "gn": ("FLOAT_VECTOR", n)})
        g = BC.Graph(f"clutter_{name}_gn")
        ci = g.new("GeometryNodeCollectionInfo", transform_space="ORIGINAL"); ci.inputs["Collection"].default_value = proto
        ci.inputs["Separate Children"].default_value = True; ci.inputs["Reset Children"].default_value = True
        ipn = g.new("GeometryNodeInstanceOnPoints"); g.put(ipn.inputs["Points"], g.gin.outputs[0]); g.put(ipn.inputs["Instance"], ci.outputs["Instances"])
        ipn.inputs["Pick Instance"].default_value = True; g.put(ipn.inputs["Instance Index"], g.attr("pidx", "INT"))
        er = g.new("FunctionNodeEulerToRotation"); g.put(er.inputs["Euler"], g.attr("rot", "FLOAT_VECTOR")); g.put(ipn.inputs["Rotation"], er.outputs["Rotation"])
        g.put(ipn.inputs["Scale"], g.attr("sclv", "FLOAT_VECTOR")); g.put(g.gout.inputs[0], ipn.outputs["Instances"])
        md = ob.modifiers.new("clutter", "NODES"); md.node_group = g.ng; ob["kind"] = "clutter"
        out[name] = int(len(x))
    return {"clutter": out, "s_clutter": round(time.time() - t0, 1)}


def frame_near(sc, scene, G, cams):
    """Core hook frame_near (core/hooks.py; a lit scene frame from the .blend — core/outputs/bl_color): small items near the cameras by the site's surface
    recipe (surface.json: clutter, masks; biome → snow line). → the build summary."""
    import json, os
    from core.surface.ground import snowline_of
    sj = json.load(open(os.path.join(scene, "surface.json")))
    return build(G, cams, sj.get("clutter"), masks=sj.get("masks"), snowline=snowline_of(G.loc, sj.get("biome") or "temperate"), sc=sc)
