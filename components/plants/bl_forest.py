"""The forest in Blender from the forest.py computation (forest.json): near the camera — unique trees (tilt, crown toward the light, roots seated on the terrain),
farther out — instances of variants (species × age × state: K variants, one mesh per variant — bark, foliage, roots merged),
detail tiers by the tree's height in the frame (TIER_PX: the largest over the cameras whose frame the tree falls into; not by distance to the eye, since at the
forest step, which runs before the cameras, the eye is the middle of the plot); an instance has its own scale by height, tilt by rotation,
crown toward the light — by choosing the variant's rotation (its crown offset points toward the gap).
The function build_forest is for the builders (the forest sheet, the component step); rebuild_forest rebuilds by the cameras of the layer (the "forest at the cameras" step)."""
import core.paths
import json, math, os, sys, time
import bpy
import numpy as np
from mathutils import Vector

from components.plants import bl_plant as BP


def join(objs, name):
    """The parts of a plant → one object (the materials are stacked), for instances."""
    bpy.context.view_layer.update()                                   # the layer list is stale after deletions and linking
    for o in list(bpy.context.selected_objects or []):
        if o is not None: o.select_set(False)
    for o in objs: o.select_set(True)
    with bpy.context.temp_override(active_object=objs[0], object=objs[0], selected_objects=objs, selected_editable_objects=objs):
        bpy.ops.object.join()
    objs[0].name = name; objs[0].data.name = name; return objs[0]


def points_object(name, coll, co, attrs):
    me = bpy.data.meshes.new(name); me.vertices.add(len(co)); me.vertices.foreach_set("co", np.asarray(co, np.float32).ravel())
    for k, (typ, v) in attrs.items():
        a = me.attributes.new(k, typ, "POINT"); key = "value" if typ in ("FLOAT", "INT") else "vector"; a.data.foreach_set(key, np.asarray(v).ravel())
    me.update(); ob = bpy.data.objects.new(name, me); coll.objects.link(ob); return ob


def instancer(name, coll, co, proto, idx, rot, scl):
    """Instances at points: the prototype index idx (the proto collection — children by name), Euler rotation rot (n,3), scale scl (n)."""
    scl = np.asarray(scl, np.float32); vec = scl.ndim == 2                # scale: a number or (x, y, z) — meadow: wider, not taller
    ob = points_object(name, coll, co, {"pidx": ("INT", np.asarray(idx, np.int32)), "rot": ("FLOAT_VECTOR", np.asarray(rot, np.float32)),
                                        "scl": ("FLOAT_VECTOR" if vec else "FLOAT", scl)})
    ng = bpy.data.node_groups.new(name + "_gn", "GeometryNodeTree"); ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"); N_, L_ = ng.nodes, ng.links
    gi, go = N_.new("NodeGroupInput"), N_.new("NodeGroupOutput")
    ci = N_.new("GeometryNodeCollectionInfo"); ci.transform_space = "ORIGINAL"; ci.inputs["Collection"].default_value = proto
    ci.inputs["Separate Children"].default_value = True; ci.inputs["Reset Children"].default_value = True
    ip = N_.new("GeometryNodeInstanceOnPoints"); ip.inputs["Pick Instance"].default_value = True
    def attr(nm, typ):
        a = N_.new("GeometryNodeInputNamedAttribute"); a.data_type = typ; a.inputs["Name"].default_value = nm; return a.outputs["Attribute"]
    er = N_.new("FunctionNodeEulerToRotation")
    L_.new(gi.outputs[0], ip.inputs["Points"]); L_.new(ci.outputs["Instances"], ip.inputs["Instance"]); L_.new(attr("pidx", "INT"), ip.inputs["Instance Index"])
    L_.new(attr("rot", "FLOAT_VECTOR"), er.inputs["Euler"]); L_.new(er.outputs["Rotation"], ip.inputs["Rotation"]); L_.new(attr("scl", "FLOAT_VECTOR" if vec else "FLOAT"), ip.inputs["Scale"])
    L_.new(ip.outputs["Instances"], go.inputs[0]); md = ob.modifiers.new("scatter", "NODES"); md.node_group = ng; return ob


def exclude(coll):
    """The prototype collection — from the layer (only instances are in the frame); searches at any depth."""
    def walk(lc):
        if lc.collection == coll: lc.exclude = True; return True
        return any(walk(ch) for ch in lc.children)
    walk(bpy.context.view_layer.layer_collection)


def eyes_of(sc, G):
    """Eyes for the detail tiers: the scene's cameras; none — the site's cameras.json; none — the middle of the plot."""
    E = [tuple(o.matrix_world.translation) for o in sc.objects if o.type == "CAMERA"]
    p = os.path.join(G.folder, "cameras.json")
    if not E and os.path.exists(p): E = [tuple(np.array(c["matrix_world"])[:3, 3]) for c in json.load(open(p))]
    if not E: b = G.near_bounds(); E = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 0.0)]
    return E


def leaf_rgb(ob):
    """Mean foliage colour of a prototype: the organ image (alpha > 0.5) × the card tint (attribute tint)."""
    from components.plants import bl_maps as BMp
    for m in ob.data.materials:
        if m is None or m.name.startswith("bark"): continue           # bark with a scan is an image too (otherwise the hulls came out brown)
        tex = BMp.tex_of_mat(m)
        if tex and tex != "blade":
            im = next(n.image for n in m.node_tree.nodes if n.type == "TEX_IMAGE" and n.image)
            px = np.empty(im.size[0] * im.size[1] * 4, np.float32); im.pixels.foreach_get(px); px = px.reshape(-1, 4); px = px[px[:, 3] > 0.5]
            rgb = px[:, :3].mean(0) if len(px) else np.array([0.05, 0.1, 0.03])
            if not im.is_float: rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)   # an 8-bit PNG is sRGB
            tn = ob.data.attributes.get("tint")
            if tn is not None:
                tv = np.empty(len(tn.data) * 4, np.float32); tn.data.foreach_get("color", tv); tv = tv.reshape(-1, 4)[:, :3]; tv = tv[tv.sum(1) > 0.3]
                if len(tv): rgb = rgb * tv.mean(0)
            return tuple(float(x) for x in rgb)
    return (0.05, 0.1, 0.03)


def far_material(rgb, bark):
    """Far crown hull: foliage colour with blotches, a lumpy relief (Voronoi), a fluffy edge — by the viewing angle (Layer Weight
    Facing) with noise-driven see-through (Dithered); the shadow has no transparency (cheap)."""
    key = ("far", tuple(round(x, 3) for x in rgb))
    if key in BP._MATS: return BP._MATS[key]
    m = bpy.data.materials.new("far_crown"); m.surface_render_method = "DITHERED"; m.use_transparent_shadow = False; m.use_backface_culling = False
    nt = m.node_tree; N_, L_ = nt.nodes, nt.links; bs = N_.get("Principled BSDF"); out = N_.get("Material Output"); sid = lambda ss, i: next(x for x in ss if x.identifier == i)
    tc = N_.new("ShaderNodeTexCoord"); nz = N_.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 0.6; nz.inputs["Detail"].default_value = 4
    vo = N_.new("ShaderNodeTexVoronoi"); vo.inputs["Scale"].default_value = 3.0
    L_.new(tc.outputs["Object"], nz.inputs["Vector"]); L_.new(tc.outputs["Object"], vo.inputs["Vector"])
    mx = N_.new("ShaderNodeMix"); mx.data_type = "RGBA"; sid(mx.inputs, "A_Color").default_value = (*[c * 0.55 for c in rgb], 1); sid(mx.inputs, "B_Color").default_value = (*[c * 1.3 for c in rgb], 1)
    L_.new(nz.outputs["Fac"], mx.inputs["Factor"]); L_.new(sid(mx.outputs, "Result_Color"), bs.inputs["Base Color"]); bs.inputs["Roughness"].default_value = 0.8
    bp = N_.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = 0.8; L_.new(vo.outputs["Distance"], bp.inputs["Height"]); L_.new(bp.outputs["Normal"], bs.inputs["Normal"])
    lw = N_.new("ShaderNodeLayerWeight"); lw.inputs["Blend"].default_value = 0.35
    edge = N_.new("ShaderNodeMapRange"); edge.inputs["From Min"].default_value = 0.55; edge.inputs["From Max"].default_value = 0.95; L_.new(lw.outputs["Facing"], edge.inputs["Value"])
    hole = N_.new("ShaderNodeMath"); hole.operation = "GREATER_THAN"; hole.inputs[1].default_value = 0.5; L_.new(vo.outputs["Distance"], hole.inputs[0])
    cut = N_.new("ShaderNodeMath"); cut.operation = "MULTIPLY"; L_.new(edge.outputs["Result"], cut.inputs[0]); L_.new(hole.outputs[0], cut.inputs[1])
    tp = N_.new("ShaderNodeBsdfTransparent"); ms = N_.new("ShaderNodeMixShader"); L_.new(cut.outputs[0], ms.inputs["Fac"]); L_.new(bs.outputs[0], ms.inputs[1]); L_.new(tp.outputs[0], ms.inputs[2])
    L_.new(ms.outputs[0], out.inputs["Surface"]); BP._MATS[key] = m; return m


def far_hull(ob, name, coll):
    """Far prototype made of hulls: the crown — 6 convex lumps over the foliage cloud (bl_maps.map_proxy_of), the trunk — an octagonal prism; hundreds of
    vertices instead of tens of thousands of cards."""
    from components.plants import bl_maps as BMp
    V, F = BMp.map_proxy_of(ob, 6)
    if not len(F): return None
    me = bpy.data.meshes.new(name); me.from_pydata([tuple(v) for v in V], [], [list(map(int, f)) for f in F]); me.update(); me.shade_smooth()
    bark = next((m for m in ob.data.materials if m and m.name.startswith("bark")), None)
    me.materials.append(far_material(leaf_rgb(ob), bark)); me.materials.append(bark or bpy.data.materials.new("bark_far"))
    nt = len(F) - 16                                                   # the last 16 faces are the trunk (bl_maps._cylinder): bark
    if nt > 0:
        mi = np.zeros(len(me.polygons), np.int32); mi[nt:] = 1; me.polygons.foreach_set("material_index", mi); me.update()
    o = bpy.data.objects.new(name, me); coll.objects.link(o); return o


def tree_spec(t, lod=0, unique=True):
    """A forest record → a plant specification (plants.resolve has already been run on the project's Python: t['spec'])."""
    sp = dict(t["spec"]); sp["lod"] = lod
    if not unique: sp["lean"] = {"deg": 0, "az": 0, "recover": 0.6}; sp.pop("push", None)
    return sp


# Tiers: the lower bound of the tree's height in the frame, px, for tiers 0–3; below that and invisible — tier 4 (hull). On
# tiers 1–4, branches that are thinner than CUT_PX pixels at the upper bound of the tier are dropped: they are not visible under the foliage, and they are most of the bark
# (beech: bark of 124 thousand triangles at tier 1 → 36 thousand without branches thinner than 1.5 cm).
TIER_PX = (450.0, 200.0, 60.0, 25.0)
CUT_PX = (0.0, 0.75, 0.75, 0.75, 0.75)                                # per tier: a branch thinner than this fraction of a pixel is dropped
PLANT_LOD = (0, 1, 1, 2, 2)                                           # forest tier → bl_plant tier (its tier 2 has only the first order of branches:
                                                                      # at 200 px the crown is full of holes (as seen on a sheet); below 60 px it is fine)
LEAF = {1: (0.5, 1.41), 2: (0.15, 2.6), 3: (0.6, 1.3)}               # tier → (card count ×, size ×) on top of the bl_plant tier
HULL = len(TIER_PX)                                                   # the last tier is the hull (at 60 px the hull is lumps, not a tree)
SHADOW_THIN = (1, 3, 2, 2, 1)                                         # tier → how many times sparser the cards of the shadow double are (bl_veg_light:
                                                                      # every Nth one, larger by √N — the shadow's places and area stay the same; the shadow
                                                                      # of a coarse tier — different card places — darkened crowns by 8–14 %)
FPX_EYE = 960 * 35 / 36.0                                             # an eye without a camera description: a 960 px frame, a 35 mm lens
SHADOW_H = 2.0                                                        # a tree's shadow falls into the frame if the tree is closer to the frame than 2 heights


def tree_px(T, cams, z=None, shadow=SHADOW_H):
    """The tree's height in the frame, px, over the cameras → (px, dist, px_shadow, px_any): px — the largest over the cameras whose frame
    contains the crown sphere (centre at half the height, radius — max(crown, half the height)), dist — to the nearest such camera; px_shadow — the same for the
    frame extended by shadow tree heights (its shadow may fall into the frame); px_any — by distance to the nearest camera, ignoring the frame.
    cams — descriptions of the camera layer (matrix_world, lens_mm, sensor_width_mm, resolution) or eye points (x, y, z): a point has no frame —
    by distance with a 35 mm lens. If none sees it — 0 and inf."""
    n = len(T); px = np.zeros(n); dist = np.full(n, np.inf); psh = np.zeros(n); pany = np.zeros(n)
    if not n: return px, dist, psh, pany
    H = np.array([t["h"] for t in T], float); R = np.maximum(np.array([t.get("cr", 2.0) for t in T], float), H / 2)
    Z = np.array([t.get("z", 0.0) for t in T], float) if z is None else np.asarray(z, float)
    C = np.stack([[t["x"] for t in T], [t["y"] for t in T], Z + H / 2], 1)
    for c in cams:
        if isinstance(c, dict):
            M = np.asarray(c["matrix_world"], float); rx, ry = (c.get("resolution") or [960, 540])[:2]
            sw = float(c.get("sensor_width_mm", 36.0)); lens = float(c.get("lens_mm", 35.0)); fpx = rx * lens / sw
            loc = (C - M[:3, 3]) @ M[:3, :3]; depth = -loc[:, 2]; tw = sw / 2 / lens; th = tw * ry / rx
            dd = np.maximum(depth, 0.0); d = np.linalg.norm(C - M[:3, 3], axis=1); p = H * fpx / np.maximum(depth, 1.0)
            inside = lambda r: (depth > -r) & (np.abs(loc[:, 0]) < dd * tw + r) & (np.abs(loc[:, 1]) < dd * th + r)
            vis = inside(R); sh = inside(R + shadow * H)
            px = np.maximum(px, np.where(vis, p, 0.0)); dist = np.minimum(dist, np.where(vis, d, np.inf))
            psh = np.maximum(psh, np.where(sh, p, 0.0)); pany = np.maximum(pany, H * fpx / np.maximum(d, 1.0))
        else:
            e = np.asarray(c, float)[:3]; d = np.linalg.norm(C - e, axis=1); p = H * FPX_EYE / np.maximum(d, 1.0)
            px = np.maximum(px, p); dist = np.minimum(dist, d); psh = np.maximum(psh, p); pany = np.maximum(pany, p)
    return px, dist, psh, pany


def tiers(px, psh, pany):
    """A tree's tier — by its height in the frame; the hull — only if the tree is not visible, its shadow does not fall into the frame (the frame extended by
    SHADOW_H heights) and by distance it is smaller than TIER_PX[-1]; otherwise an invisible tree gets tier 3: the shadow of a hull is a solid polygon
    (a forest in the evening: dark patches with straight edges on the ground), the shadow of sparse cards is blotchy, like a tree's."""
    t = tier_of(px)
    return np.where((t == HULL) & ((psh > 0) | (pany >= TIER_PX[-1])), HULL - 1, t)


def tier_of(px):
    return np.select([px >= v for v in TIER_PX], list(range(len(TIER_PX))), HULL)


def cut_twigs(ob, r_cut):
    """Branches thinner than r_cut (m; the largest vertex radius of a face, the bark's radius attribute) are removed from the bark. Returns: how many faces were removed."""
    me = ob.data; ra = me.attributes.get("radius")
    if ra is None or r_cut <= 0 or not len(me.polygons): return 0
    import bmesh
    r = np.empty(len(me.vertices), np.float32); ra.data.foreach_get("value", r)
    ls = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get("loop_start", ls)
    lv = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", lv)
    dead = np.flatnonzero(np.maximum.reduceat(r[lv], ls) < r_cut)
    if not len(dead): return 0
    bm = bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.faces[int(i)] for i in dead], context="FACES")
    bm.to_mesh(me); bm.free(); me.update(); return len(dead)


def tris_of(ob):
    lt = np.empty(len(ob.data.polygons), np.int32); ob.data.polygons.foreach_get("loop_total", lt); return int((lt - 2).sum())


# Trimming the cards to the foliage (frames of a take under the canopy: the forest is pixels, not triangles — a frame takes 10 s without the forest against 26, and tiers
# with the same card area do not change the cost). A leaf occupies 15–27 % of a branch image, and the empty margins are still drawn (depth with
# an alpha cut-out — for each layer of cards). The card stays itself (6 vertices, 2 quads, the fold — bl_plant.cards):
# its vertices move toward the foliage hull — the edges to the extreme points, the bottom and top along supporting lines to the hull that meet at
# the fold; a branch card is ~0.53 of its former area, a spruce's 0.35; render time of a take frame −16…19 %, of an orbit −11 %, the same look (mean pixel
# difference 0.05–1.6 of 255 — cut-out noise). Shadow doubles shrink the card to the leaf share (sk_cov) — the share is recomputed for
# the trimmed area; the maps (bl_maps.cut_cards) place the leaves by the unwrap of the trimmed card (sk_trim_uv).
_TRIM = {}


def _chains(P):
    """Convex hull of points (monotone chain) → the lower and upper chains, left to right."""
    P = sorted(set(map(tuple, P)))
    def half(pts):
        h = []
        for p in pts:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= 0: h.pop()
            h.append(p)
        return np.array(h)
    return half(P), half(P[::-1])[::-1]


def _at(ch, u):
    """The chain at u: the value and the slopes on the left and right (at a chain vertex each side has its own)."""
    x, y = ch[:, 0], ch[:, 1]; i = int(np.clip(np.searchsorted(x, u), 1, len(x) - 1))
    s = (y[i] - y[i - 1]) / max(x[i] - x[i - 1], 1e-9); sl = s
    if abs(x[i - 1] - u) < 1e-9 and i >= 2: sl = (y[i - 1] - y[i - 2]) / max(x[i - 1] - x[i - 2], 1e-9)
    return y[i - 1] + s * (u - x[i - 1]), sl, s


def card_trim(m, thr=8 / 255.0, pad=0.015):
    """A card hexagon from the foliage of the material m's image: the UVs of vertices 0–5 of bl_plant.cards (0 — bottom left, 1 — the fold at the bottom,
    2 — bottom right, 3 — top right, 4 — the fold at the top, 5 — top left) and its share of the area. Not a card, or the leaf is not on both sides of
    the fold — None."""
    from components.plants import bl_maps as BMp
    key = m.name if m else None
    if key in _TRIM: return _TRIM[key]
    tex = BMp.tex_of_mat(m) if m else None; out = None
    im = next((n.image for n in m.node_tree.nodes if n.type == "TEX_IMAGE" and n.image), None) if tex else None
    if im is not None and im.size[0] > 0:
        W, H = im.size; px = np.empty(W * H * 4, np.float32); im.pixels.foreach_get(px); a = px[3::4].reshape(H, W)
        st = max(1, W // 256); a = a[::st, ::st]; ys, xs = np.nonzero(a > thr)                  # Blender pixels run bottom to top
        if len(xs) > 8:
            P = np.round(np.stack([xs / (a.shape[1] - 1), ys / (a.shape[0] - 1)], 1), 4); lo, up = _chains(P)
            uL, uR = P[:, 0].min() - pad, P[:, 0].max() + pad
            if uL < 0.5 < uR:
                vb, gl, gr = _at(lo, 0.5); ve, hl, hr = _at(up, 0.5); vb -= pad; ve += pad
                hx = np.clip(np.array([(uL, vb + gl * (uL - 0.5)), (0.5, vb), (uR, vb + gr * (uR - 0.5)),
                                       (uR, ve + hr * (uR - 0.5)), (0.5, ve), (uL, ve + hl * (uL - 0.5))]), 0.0, 1.0)
                x, y = hx[:, 0], hx[:, 1]; area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                if area < 0.9: out = (hx, float(area))
    _TRIM[key] = out; return out


def trim_cards(ob):
    """Foliage cards of an object (a merged prototype or a plant part) — trimmed to the foliage (card_trim): the vertices and the UVs, in the same
    order; the material's leaf share (sk_cov, for shadow doubles) — for the trimmed area. Returns: how many cards were trimmed."""
    me = ob.data; mats = list(me.materials); npg = len(me.polygons)
    if not npg or not any(m and m.name.startswith("card_") for m in mats): return 0
    V = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get("co", V); V = V.reshape(-1, 3)
    mi = np.empty(npg, np.int32); me.polygons.foreach_get("material_index", mi); ls = np.empty(npg, np.int32); me.polygons.foreach_get("loop_start", ls)
    lt = np.empty(npg, np.int32); me.polygons.foreach_get("loop_total", lt); lv = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", lv)
    uvl = me.uv_layers.get("UVMap"); UV = np.empty(len(me.loops) * 2, np.float32)
    if uvl is None: return 0
    uvl.data.foreach_get("uv", UV); UV = UV.reshape(-1, 2); n_done = 0
    for i, m in enumerate(mats):
        if not (m and m.name.startswith("card_")) or not (mi == i).any(): continue
        tr = card_trim(m)
        if tr is None: continue
        hx, area = tr; sel = np.flatnonzero(mi == i)
        if not (lt[sel] == 4).all(): continue
        Lp = np.add.outer(ls[sel], np.arange(4)).ravel(); vs = np.unique(lv[Lp])
        if len(vs) % 6 or vs[-1] - vs[0] + 1 != len(vs): continue           # cards come in blocks of 6 consecutive vertices
        Q = V[vs].reshape(-1, 6, 3); Q0, Q1, Q2, Q4, Q5 = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 4], Q[:, 5]
        new = np.empty_like(Q)
        for j, (u, v) in enumerate(hx):
            if u <= 0.5: new[:, j] = Q0 + (u / 0.5) * (Q1 - Q0) + v * (Q5 - Q0)
            else: new[:, j] = Q1 + ((u - 0.5) / 0.5) * (Q2 - Q1) + v * (Q4 - Q1)
        V[vs] = new.reshape(-1, 3); UV[Lp] = hx[(lv[Lp] - vs[0]) % 6]
        if not m.get("sk_trim"): m["sk_cov_card"] = float(m.get("sk_cov", 0.3)); m["sk_trim"] = area; m["sk_trim_uv"] = hx.ravel().tolist()
        m["sk_cov"] = min(1.0, m["sk_cov_card"] / m["sk_trim"]); n_done += len(Q)
    if n_done:
        me.vertices.foreach_set("co", V.ravel()); uvl.data.foreach_set("uv", UV.ravel()); me.update()
    return n_done


def variant(t, lod, height, seed, name, coll, far="hull"):
    """A prototype of tier lod (0–4) for tree t: height height, seed seed → one object name in the collection coll. Tiers 1–2 —
    the skeleton and cards of the bl_plant tier PLANT_LOD, with cards sparser and larger by LEAF (the same crown mass: count × size² ≈ 1), without
    branches thinner than CUT_PX; tier 4 (HULL) — the hull (far="hull"; palm, dead and winter trees — the cards of bl_plant tier 2)."""
    hull = lod == HULL and far == "hull" and t["species"] not in ("palm",) and t["state"] not in ("dead", "winter")
    spec = tree_spec(t, PLANT_LOD[lod], unique=False); spec["height"] = float(height); spec["seed"] = int(seed)
    a, b = LEAF.get(lod, (1.0, 1.0))
    if spec.get("leaves") and (a, b) != (1.0, 1.0): spec["leaves"] = dict(spec["leaves"], per_m=spec["leaves"].get("per_m", 6) * a, size=spec["leaves"]["size"] * b)
    if lod >= 2: spec.pop("roots", None)                              # below 200 px the roots are not visible
    objs, c = BP.build(spec, (0.0, 0.0, 0.0), ground=None, name=f"v{name}", coll=coll)
    if lod >= 1 and not hull:                                         # branches thinner than CUT_PX at the upper bound of the tier are dropped
        r_cut = min(CUT_PX[lod] * spec["height"] / (2.0 * TIER_PX[lod - 1]), 0.1)   # thicker than 10 cm — limbs: the tree's silhouette
        for o in objs:
            if o.name.endswith("_bark"): cut_twigs(o, r_cut)
    if not hull:                                                      # the hull is built from the points of whole cards
        for o in objs: trim_cards(o)
    ob = join(objs, name) if len(objs) > 1 else objs[0]; ob.name = name; ob.location = (0, 0, 0)
    if hull:                                                          # the smallest in the frame and invisible ones (shadow only) — as hulls
        h = far_hull(ob, "h" + name, coll)
        if h is not None: bpy.data.objects.remove(ob); h.name = name; ob = h
    ob["sk_tier"], ob["sk_h"], ob["sk_shadow_thin"] = int(lod), float(height), int(SHADOW_THIN[lod])
    return ob


def cams_of(sc, G):
    """Cameras for the tiers: the camera layer (the run scene's cameras.json); none — the .blend cameras as descriptions; none — the middle of the plot (a point)."""
    from core import layers as L
    C = [c for c in L.cameras(G.folder) if c.get("matrix_world")]
    if C: return C
    r = sc.render; res = [int(r.resolution_x * r.resolution_percentage / 100), int(r.resolution_y * r.resolution_percentage / 100)]
    C = [{"matrix_world": [list(v) for v in o.matrix_world], "lens_mm": o.data.lens, "sensor_width_mm": o.data.sensor_width, "resolution": res}
         for o in sc.objects if o.type == "CAMERA"]
    return C or eyes_of(sc, G)


def build_forest(F, G, cams, sc=None, near=22.0, max_unique=14, K=(2, 2, 1, 1, 1), parent=None, far="hull", skip=()):
    """The forest from forest.json (+ a spec for each tree — plants.resolve, done in the project's Python). cams — descriptions of the layer's cameras or
    eye positions [(x, y, z)…] (tiers — tree_px); parent — the section collection (otherwise the scene root); skip — tree numbers not to place
    (clearing near the cameras); far — the farthest tier: hull (hulls) | cards (cards of bl_plant tier 2). Returns: (collection, cost)."""
    sc = sc or bpy.context.scene; t0 = time.time(); T = F["trees"]; parent = parent or sc.collection; skip = set(int(i) for i in skip)
    coll = bpy.data.collections.new("forest"); parent.children.link(coll)
    gfun = lambda X, Y: G.height(np.asarray(X, float), np.asarray(Y, float))
    cams = list(cams) or [(0.0, 0.0, 0.0)]
    px, d, psh, pany = tree_px(T, cams, gfun([t["x"] for t in T], [t["y"] for t in T]) if T else None); tier = tiers(px, psh, pany)
    order = np.argsort(-px); uniq = set(int(i) for i in order if d[i] < near and tier[i] == 0 and int(i) not in skip)
    uniq = set(sorted(uniq, key=lambda i: -px[i])[:max_unique])
    cost = {"unique": len(uniq), "unique_vertices": 0, "unique_tris": 0, "cameras": len(cams)}
    for i in sorted(uniq):                                            # near the camera — each one is unique
        t = T[i]; objs, c = BP.build(tree_spec(t, 0), (t["x"], t["y"], None), ground=gfun, name=f"tree_{i:05d}", coll=coll)
        for o in objs: trim_cards(o)
        cost["unique_vertices"] += c["verts"]; cost["unique_tris"] += sum(tris_of(o) for o in objs if o.type == "MESH")
    groups = {}                                                       # variants: (species, age, state, tier)
    for i, t in enumerate(T):
        if i in uniq or i in skip: continue
        lod = int(tier[i])                                            # beyond the near tier — age is not distinguished: the height is applied by scale
        groups.setdefault((t["species"], t["age"] if lod == 0 else "any", t["state"], lod), []).append(i)
    pc = bpy.data.collections.new("forest_protos"); parent.children.link(pc); protos, info = [], []
    co, idx, rot, scl = [], [], [], []
    for (sp_, age, st, lod), ii in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] == "any", kv[0][1], kv[0][2], kv[0][3])):   # "any" (far variants) — after the ages, as before
        hs = np.array([T[i]["h"] for i in ii]); base = []
        for k in range(min(K[lod], len(ii))):
            t = T[ii[k]]; hv = float(np.median(hs))
            ob = variant(t, lod, hv, int(t["seed"]) % 100000 + k, f"p{len(protos):05d}", pc, far)
            V = np.empty(len(ob.data.vertices) * 3); ob.data.vertices.foreach_get("co", V); V = V.reshape(-1, 3)
            top = V[V[:, 2] > hv * 0.5]; off = top[:, :2].mean(0) if len(top) else np.zeros(2)   # crown offset of the variant
            base.append((len(protos), hv, off)); protos.append(ob)
            info.append({"species": sp_, "age": age, "state": st, "lod": lod, "vertices": len(V), "tris": tris_of(ob)})
        for j, i in enumerate(ii):
            t = T[i]; pid, hv, off = base[j % len(base)]; push = np.asarray(t.get("push", [0, 0]), float)
            if np.linalg.norm(push) > 0.3 and np.linalg.norm(off) > 0.05: yaw = math.atan2(push[1], push[0]) - math.atan2(off[1], off[0])
            else: yaw = (t["seed"] % 6283) / 1000.0
            L = t.get("lean", {}); th = math.radians(L.get("deg", 0)); az = math.radians(L.get("az", 0)) - yaw
            co.append((t["x"], t["y"], float(gfun([t["x"]], [t["y"]])[0]) - 0.08)); idx.append(pid)
            rot.append((-th * math.sin(az), th * math.cos(az), yaw)); scl.append(t["h"] / hv)
    for o in pc.objects: o["kind"] = "plant"
    if protos:
        exclude(pc); inst = instancer("forest_instances", coll, co, pc, idx, rot, scl); inst["kind"] = "plant"
    pt = np.array([x["tris"] for x in info] or [0]); pl = np.array([x["lod"] for x in info] or [0]); ii = np.asarray(idx, int)
    cost.update({"instances": len(co), "variants": len(protos), "variant_vertices": int(sum(x["vertices"] for x in info)),
                 "tiers": {str(k): int((pl[ii] == k).sum()) if len(ii) else 0 for k in range(HULL + 1)},
                 "tiers_tris_m": {str(k): round(float(pt[ii][pl[ii] == k].sum()) / 1e6, 2) if len(ii) else 0.0 for k in range(HULL + 1)},
                 "tris_m": round((float(pt[ii].sum()) if len(ii) else 0.0) / 1e6 + cost["unique_tris"] / 1e6, 2), "s": round(time.time() - t0, 1)})
    return coll, cost


def rebuild_forest(F, G, cams, sc=None, parent=None, skip=(), **kw):
    """Rebuilding the forest by the cameras (the "forest at the cameras" step: the orbit and take cameras appear after the forest step): the previous collections forest and
    forest_protos are removed together with the objects and scatter nodes, the forest is built again by build_forest without the skip trees. Returns: (collection, cost)."""
    sc = sc or bpy.context.scene
    for name in ("forest_instances__shadow",):                        # the double of the previous forest (the light step places a new one)
        o = bpy.data.objects.get(name)
        if o is not None: bpy.data.objects.remove(o, do_unlink=True)
    for name in ("forest", "forest_protos", "forest_protos__shadow"):
        c = bpy.data.collections.get(name)
        if c is None: continue
        if parent is None:
            parent = next((p for p in bpy.data.collections if c.name in p.children), None) if name == "forest" else parent
        for o in list(c.all_objects):
            ng = next((m.node_group for m in o.modifiers if m.type == "NODES" and m.node_group), None)
            bpy.data.objects.remove(o, do_unlink=True)
            if ng is not None and ng.users == 0: bpy.data.node_groups.remove(ng)
        bpy.data.collections.remove(c)
    return build_forest(F, G, cams, sc, parent=parent, skip=skip, **kw)


def dress_place(G, cams, sc=None, density=0.45, seed=1, meadow=True, parent=None, clear_cam=15.0):
    """Dressing the site with vegetation (instead of the stand-ins of components/site/bl_relief.dress_from and scan grass): the forest — the site's forest.json
    (none — computed by the project's Python: forest.py, density 0.45 — like the former dressing tree layer — × the biome density; near the site's cameras
    a clearing of clear_cam m), the meadow — near the cameras (plants.py meadow). cams — [(x, y, z, heading)].
    Returns: a summary."""
    import subprocess
    py = os.path.join(core.paths.ROOT, ".venv", "bin", "python"); here = os.path.dirname(os.path.abspath(__file__)); res = {}
    fj = os.path.join(G.folder, "forest.json")
    if not os.path.exists(fj):
        av = ";".join(f"{c[0]:.1f},{c[1]:.1f},{clear_cam}" for c in cams or [])   # a clearing near the site's cameras (the site composition is built around them)
        r = subprocess.run([py, "-m", "components.plants.forest", G.folder, str(density), str(seed), f"out={fj}"] + ([f"avoid={av}"] if av else []), capture_output=True, text=True)
        if r.returncode or not os.path.exists(fj): return {"plants": "forest.py: " + (r.stderr or r.stdout)[-300:]}
    F = json.load(open(fj)); eyes = [tuple(c[:3]) for c in cams] if cams else eyes_of(sc or bpy.context.scene, G)
    _, res["forest"] = build_forest(F, G, eyes, sc, parent=parent)
    if meadow and cams:
        mj = os.path.join(G.folder, "meadow.json"); biome = G.loc.get("biome") or "temperate"
        if not os.path.exists(mj): subprocess.run([py, "-m", "components.plants.plants", "meadow", biome, mj], capture_output=True, text=True)
        if os.path.exists(mj):
            from components.plants import bl_meadow as BM
            _, res["meadow"] = BM.build_meadow(G, cams, json.load(open(mj)), sc, forest=F, parent=parent)
    return res
