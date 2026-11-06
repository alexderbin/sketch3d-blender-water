"""Plants in ControlNet maps (mask, id, depth, normals, outline): the core's map pass (core/outputs/bl_parts.collect, bl_pack) takes
the object's mesh without Geometry Nodes instances and replaces the materials with the class tone — the cards' alpha is lost, a leaf comes out as a rectangle,
and forests and meadows made of instances vanish. Here — proxies for the maps: organ cards → "cut-out" meshes along the alpha contour
(polygons — leaves.cutout, <type>.cut.json), instances → real meshes (coarser for far ones). Alpha is no longer needed.
proxies(sc) → a list of proxy objects (kind=plant_map); hide the original plants while the map pass runs."""
import core.paths
import json, os
import bpy
import numpy as np

TEX = os.path.join(core.paths.ROOT, "library", "plants", "tex"); _CUT = {}


def polys_of(tex):
    if tex not in _CUT:
        p = os.path.join(TEX, tex + ".cut.json"); _CUT[tex] = json.load(open(p)) if os.path.exists(p) else [[[0, 0], [1, 0], [1, 1], [0, 1]]]
    return _CUT[tex]


def tex_of_mat(m):
    """The organ type of a card from the material image (card_material: the image <type>.png); not a card — None."""
    if m and m.node_tree:
        for n in m.node_tree.nodes:
            if n.type == "TEX_IMAGE" and n.image: return os.path.splitext(os.path.basename(n.image.filepath))[0]
    return None


def cut_cards(V, polys, uv6=None):
    """Cards (6 vertices each: bottom left, fold, bottom right, top right, fold, top left — bl_plant.cards) → a cut-out mesh:
    each polygon of the cut-out is a fan of triangles in the card's plane. uv6 — the UVs of the 6 vertices, if the card is trimmed to the foliage
    (bl_forest.trim_cards, material: sk_trim_uv): a point's position comes from its own half of the card, built from its vertices. Returns (V, F)."""
    Q = V.reshape(-1, 6, 3); P = (Q[:, 0] + Q[:, 2]) / 2; Sw = (Q[:, 2] - Q[:, 0]) / 2; Ls = Q[:, 5] - Q[:, 0]
    if uv6 is not None:                                                   # a half of the card is a flat parallelogram: X = O + M·(uv − o)
        w = np.asarray(uv6, float).reshape(6, 2); half = []
        for o, b, c in ((0, 1, 5), (1, 2, 4)):
            Mi = np.linalg.inv(np.stack([w[b] - w[o], w[c] - w[o]], 1)); E = np.stack([Q[:, b] - Q[:, o], Q[:, c] - Q[:, o]], 2) @ Mi
            half.append((Q[:, o], E, w[o]))
    VV, FF, off = [], [], 0
    for poly in polys:
        uv = np.asarray(poly, float); k = len(uv)
        if k < 3: continue
        if uv6 is None: pts = P[:, None, :] + Sw[:, None, :] * (2 * uv[None, :, 0:1] - 1) + Ls[:, None, :] * uv[None, :, 1:2]     # (n, k, 3)
        else: pts = np.stack([O + E @ (p - o) for p in uv for (O, E, o) in [half[0 if p[0] <= 0.5 else 1]]], 1)
        VV.append(pts.reshape(-1, 3)); tri = np.array([[0, j, j + 1] for j in range(1, k - 1)])
        F = (tri[None] + (np.arange(len(Q)) * k)[:, None, None] + off).reshape(-1, 3); FF.append(F); off += len(Q) * k
    return (np.vstack(VV), np.vstack(FF)) if VV else (np.zeros((0, 3)), np.zeros((0, 3), int))


def proxy_of(ob):
    """A proxy of one plant object in its own coordinate system: by materials — cards (the organ image, 6 consecutive vertices each) → a cut-out;
    bark, ribbons and grass blades — as is (triangles). Works on merged prototypes too (bark + foliage + roots in one object)."""
    me = ob.data; n = len(me.vertices); V = np.empty(n * 3); me.vertices.foreach_get("co", V); V = V.reshape(-1, 3); npg = len(me.polygons)
    mi = np.empty(npg, np.int32); me.polygons.foreach_get("material_index", mi); ls = np.empty(npg, np.int32); me.polygons.foreach_get("loop_start", ls)
    lt = np.empty(npg, np.int32); me.polygons.foreach_get("loop_total", lt); lv = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", lv)
    VV, FF, off = [V], [], n
    for k in range(max(len(me.materials), 1)):
        sel = np.nonzero(mi == k)[0]
        if not len(sel): continue
        tex = tex_of_mat(me.materials[k]) if len(me.materials) else None
        if tex not in (None, "blade"):
            vs = np.unique(lv[np.add.outer(ls[sel], np.arange(4)).ravel()]) if (lt[sel] == 4).all() else np.unique(np.concatenate([lv[ls[i]:ls[i] + lt[i]] for i in sel]))
            if len(vs) % 6 == 0 and vs[-1] - vs[0] + 1 == len(vs):
                uv6 = me.materials[k].get("sk_trim_uv")
                Vc, Fc = cut_cards(V[vs[0]:vs[-1] + 1], polys_of(tex), list(uv6) if uv6 else None); VV.append(Vc); FF.append(Fc + off); off += len(Vc); continue
        for m_ in (3, 4):
            q = sel[lt[sel] == m_]
            if not len(q): continue
            L = lv[np.add.outer(ls[q], np.arange(m_))]; FF.append(L[:, [0, 1, 2]])
            if m_ == 4: FF.append(L[:, [0, 2, 3]])
        for i in sel[(lt[sel] != 3) & (lt[sel] != 4)]:
            pv = lv[ls[i]:ls[i] + lt[i]]; FF.append(np.array([[pv[0], pv[j], pv[j + 1]] for j in range(1, len(pv) - 1)]))
    F = np.vstack(FF) if FF else np.zeros((0, 3), int); return np.vstack(VV), F


def _mesh(name, V, F, coll):
    me = bpy.data.meshes.new(name); me.from_pydata([tuple(v) for v in np.asarray(V, float)], [], [list(map(int, f)) for f in F]); me.update()
    ob = bpy.data.objects.new(name, me); coll.objects.link(ob); ob["kind"] = "plant_map"; return ob


def proxies(sc, coll_name="plant_maps", max_inst=20000):
    """Proxies of all the scene's plants for the map pass: objects of kind=plant (unique ones, parts) — by cut-out; instances (forest_instances,
    meadow_instances) — realized at points (prototype index, rotation, scale), up to max_inst per object. Returns (collection, objects)."""
    from mathutils import Euler, Matrix
    coll = bpy.data.collections.get(coll_name) or bpy.data.collections.new(coll_name)
    if coll.name not in sc.collection.children: sc.collection.children.link(coll)
    out = []; cache = {}
    for ob in [o for o in sc.objects if o.get("kind") == "plant" and o.type == "MESH"]:
        if ob.modifiers and any(m.type == "NODES" for m in ob.modifiers):  # instances: points + prototypes
            md = next(m for m in ob.modifiers if m.type == "NODES"); ci = next((n for n in md.node_group.nodes if n.type == "COLLECTION_INFO"), None)
            if ci is None: continue
            protos = sorted(ci.inputs["Collection"].default_value.objects, key=lambda o: o.name); me = ob.data; n = min(len(me.vertices), max_inst)
            co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)[:n]
            g = lambda k, w: (lambda a: (me.attributes[k].data.foreach_get("value" if w == 1 else "vector", a), a)[1])(np.empty(len(me.vertices) * w))
            pidx = g("pidx", 1).astype(int)[:n]; rot = g("rot", 3).reshape(-1, 3)[:n]
            vec = me.attributes["scl"].data_type == "FLOAT_VECTOR"; scl = g("scl", 3).reshape(-1, 3)[:n] if vec else np.repeat(g("scl", 1)[:n, None], 3, 1)
            VV, FF, off = [], [], 0
            for i in range(n):
                p = protos[min(pidx[i], len(protos) - 1)]
                if p.name not in cache: cache[p.name] = proxy_of(p)
                V, F = cache[p.name]
                if not len(F): continue
                M = np.array(Matrix.Translation(tuple(co[i])) @ Euler(tuple(rot[i]), "XYZ").to_matrix().to_4x4() @ Matrix.Diagonal((*map(float, scl[i]), 1.0)))
                W = np.asarray(V) @ M[:3, :3].T + M[:3, 3]; W = W @ np.array(ob.matrix_world)[:3, :3].T + np.array(ob.matrix_world)[:3, 3]
                VV.append(W); FF.append(np.asarray(F, int) + off); off += len(W)
            if VV: out.append(_mesh(ob.name + "_map", np.vstack(VV), np.vstack(FF), coll))
        else:
            V, F = proxy_of(ob)
            if len(F):
                W = np.asarray(V) @ np.array(ob.matrix_world)[:3, :3].T + np.array(ob.matrix_world)[:3, 3]; out.append(_mesh(ob.name + "_map", W, F, coll))
    return coll, out


# ---------------------------------------------------------------- tiers for the core's map pass (light proxies)
def _kmeans(P, k, iters=8, seed=1):
    r = np.random.default_rng(seed); C = P[r.choice(len(P), min(k, len(P)), replace=False)]
    for _ in range(iters):
        lab = np.argmin(((P[:, None, :] - C[None, :, :]) ** 2).sum(-1), 1)
        C = np.array([P[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(len(C))])
    return lab


def _hulls(P, k):
    """A crown as several convex hulls (k-means over the foliage cloud): lumps instead of one sphere → (V, F)."""
    import bmesh
    if len(P) < 8: return np.zeros((0, 3)), np.zeros((0, 3), int)
    if len(P) > 6000: P = P[np.random.default_rng(3).choice(len(P), 6000, replace=False)]
    lab = _kmeans(P, k); VV, FF, off = [], [], 0
    for j in range(lab.max() + 1):
        Q = P[lab == j]
        if len(Q) < 4: continue
        bm = bmesh.new()
        for q in Q: bm.verts.new(q)
        try: bmesh.ops.convex_hull(bm, input=bm.verts[:])
        except Exception: bm.free(); continue
        for v in [v for v in bm.verts if not v.link_faces]: bm.verts.remove(v)
        bm.verts.index_update(); V = np.array([v.co[:] for v in bm.verts]); F = [[v.index for v in f.verts] for f in bm.faces]; bm.free()
        T = [[f[0], f[i], f[i + 1]] for f in F for i in range(1, len(f) - 1)]
        if len(T): VV.append(V); FF.append(np.array(T) + off); off += len(V)
    return (np.vstack(VV), np.vstack(FF)) if VV else (np.zeros((0, 3)), np.zeros((0, 3), int))


def _cylinder(z0, z1, r, n=8):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False); ring = np.column_stack([np.cos(a) * r, np.sin(a) * r])
    V = np.vstack([np.column_stack([ring, np.full(n, z0)]), np.column_stack([ring, np.full(n, z1)])])
    F = [[i, (i + 1) % n, n + (i + 1) % n] for i in range(n)] + [[i, n + (i + 1) % n, n + i] for i in range(n)]
    return V, np.array(F)


def map_proxy_of(ob, k=6):
    """A light plant proxy for the maps (in its own coordinate system): the crown — k hulls over the foliage card points, the trunk — an octagonal prism
    from the ground to the bottom of the crown with the base radius; grasses and flowers (no bark) — one hull for the whole plant."""
    me = ob.data; n = len(me.vertices); V = np.empty(n * 3); me.vertices.foreach_get("co", V); V = V.reshape(-1, 3)
    npg = len(me.polygons); mi = np.empty(npg, np.int32); me.polygons.foreach_get("material_index", mi)
    ls = np.empty(npg, np.int32); me.polygons.foreach_get("loop_start", ls); lv = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", lv)
    leaf = np.zeros(n, bool); bark = np.zeros(n, bool)
    for j, m in enumerate(me.materials):
        sel = np.nonzero(mi == j)[0]
        if not len(sel): continue
        vs = np.unique(lv[ls[sel]]); (leaf if tex_of_mat(m) else bark)[vs] = True
    rad = np.zeros(n)
    if "radius" in me.attributes: me.attributes["radius"].data.foreach_get("value", rad)
    if leaf.any() and bark.any():
        Vh, Fh = _hulls(V[leaf], k); zc = float(np.quantile(V[leaf][:, 2], 0.03))
        r0 = float(rad[bark & (V[:, 2] < 1.0)].max()) if (bark & (V[:, 2] < 1.0)).any() else max(0.05, 0.01 * float(V[:, 2].max()))
        Vt, Ft = _cylinder(-0.2, max(zc, 0.5) + 0.3, max(r0, 0.03))
        return np.vstack([Vh, Vt]), np.vstack([Fh, Ft + len(Vh)])
    return _hulls(V, k)


def for_maps(sc, near=25.0, k_far=6, k_near=12):
    """The scene's plants → proxies for the core's map pass (called by core/outputs/bl_parts.collect): forest instances — a crown hull and a trunk
    per prototype (hundreds of vertices per tree); unique trees near the camera — their bark as is + 12 crown hulls; the herb layer is not taken (the
    ground covers it). The original plants are marked sketch_fx — the map pass skips them. Returns: a summary."""
    from mathutils import Euler, Matrix
    if bpy.data.collections.get("plant_maps") is not None: undo_maps(sc)  # a repeated call starts over, without duplicates
    coll = bpy.data.collections.new("plant_maps"); sc.collection.children.link(coll)
    plants = [o for o in sc.objects if o.get("kind") == "plant" and o.type == "MESH"]; cache = {}; made, verts = 0, 0
    def add(name, V, F):
        nonlocal made, verts
        if not len(F): return
        o = _mesh(name, V, F, coll); o["sketch_kind"] = "tree"; made += 1; verts += len(V)
    for ob in plants:
        md = next((m for m in ob.modifiers if m.type == "NODES"), None)
        if md is not None:
            ci = next((nd for nd in md.node_group.nodes if nd.type == "COLLECTION_INFO"), None)
            if ci is None or "meadow" in ob.name: continue                 # the meadow is not in the maps (the ground covers it)
            protos = sorted(ci.inputs["Collection"].default_value.objects, key=lambda o: o.name); me = ob.data; n = len(me.vertices)
            co = np.empty(n * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
            g = lambda kk, w: (lambda a: (me.attributes[kk].data.foreach_get("value" if w == 1 else "vector", a), a)[1])(np.empty(n * w))
            pidx = g("pidx", 1).astype(int); rot = g("rot", 3).reshape(-1, 3)
            vec = me.attributes["scl"].data_type == "FLOAT_VECTOR"; scl = g("scl", 3).reshape(-1, 3) if vec else np.repeat(g("scl", 1)[:, None], 3, 1)
            VV, FF, off = [], [], 0; Mw = np.array(ob.matrix_world)
            for i in range(n):
                p = protos[min(pidx[i], len(protos) - 1)]
                if p.name not in cache: cache[p.name] = map_proxy_of(p, k_far)
                V, F = cache[p.name]
                if not len(F): continue
                M = np.array(Matrix.Translation(tuple(co[i])) @ Euler(tuple(rot[i]), "XYZ").to_matrix().to_4x4() @ Matrix.Diagonal((*map(float, scl[i]), 1.0)))
                W = V @ M[:3, :3].T + M[:3, 3]; W = W @ Mw[:3, :3].T + Mw[:3, 3]; VV.append(W); FF.append(np.asarray(F) + off); off += len(W)
            if VV: add(ob.name + "_map", np.vstack(VV), np.vstack(FF))
        else:
            nm = ob.name
            if nm.endswith(("_leaves", "_fronds")):                         # unique trees near the camera: the crown as hulls
                V, F = _hulls(np.asarray([ob.matrix_world @ v.co for v in ob.data.vertices]), k_near); add(nm + "_map", V, F)
            elif nm.endswith(("_bark", "_roots", "_rachis")):              # bark and roots as is (there are few of them)
                V, F = proxy_of(ob); W = np.asarray(V) @ np.array(ob.matrix_world)[:3, :3].T + np.array(ob.matrix_world)[:3, 3]; add(nm + "_map", W, F)
            else:                                                          # grasses and flowers as separate objects — as a hull
                V, F = map_proxy_of(ob, 3); W = np.asarray(V) @ np.array(ob.matrix_world)[:3, :3].T + np.array(ob.matrix_world)[:3, 3]; add(nm + "_map", W, F)
        if not ob.get("sketch_fx"): ob["sketch_fx"] = True; ob["_veg_fx"] = True   # the original bypasses the map pass (undo_maps will restore it)
    return {"proxies": made, "vertices": verts, "prototypes": len(cache)}


def undo_maps(sc):
    """After the scene is split into parts: remove the map proxies and restore the original plants (the sketch_fx flag set by for_maps)."""
    coll = bpy.data.collections.get("plant_maps")
    if coll is not None:
        for o in list(coll.objects): bpy.data.objects.remove(o)
        bpy.data.collections.remove(coll)
    for o in sc.objects:
        if o.get("_veg_fx"): del o["sketch_fx"]; del o["_veg_fx"]
