"""Light through vegetation in Eevee: foliage shadow via a double offset from the sun/moon.
A thin leaf under a grazing sun shades itself: the shadow map cannot tell a leaf from its own shadow, and at a sun-to-leaf angle above
45° the translucency and highlight lose 12–60 %, while Cycles does not. A non-zero material thickness shifts the shadow, but switches Eevee's translucency to a
"thick body" model (3–5 times darker at any thickness) — therefore the leaf thickness is 0, and the shadow offset is done by a double: the same mesh (or the same
instances) is invisible to the camera and only casts a shadow, while the leaf itself casts none. The offset from the sun/moon grows with the distance to the active
camera (a cell of Eevee's shadow map is about one frame pixel): d = min(d0 + k · distance, dmax); the leaf still shades its neighbours.
The doubles live in the collection sk_shadow_twins and are enabled only for the Eevee render: Workbench ignores the "invisible to camera" flag
(measured), and Cycles needs no offset; in the saved file they are disabled (the foliage casts its own shadow, as before).
Entry: bl_far.attach (the node's light) calls aim(scene, plan) — in all frame paths with light."""
import bpy
from bpy.app.handlers import persistent

GROUP, COLL = "sk_shadow_twin", "sk_shadow_twins"


def _twin_group():
    ng = bpy.data.node_groups.get(GROUP)
    if ng: return ng
    ng = bpy.data.node_groups.new(GROUP, "GeometryNodeTree"); I = ng.interface
    I.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"); I.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    s = I.new_socket("Sun", in_out="INPUT", socket_type="NodeSocketVector"); s.default_value = (0.0, 0.0, 1.0)
    s = I.new_socket("d0", in_out="INPUT", socket_type="NodeSocketFloat"); s.default_value = 0.02
    s = I.new_socket("k", in_out="INPUT", socket_type="NodeSocketFloat"); s.default_value = 0.02
    s = I.new_socket("dmax", in_out="INPUT", socket_type="NodeSocketFloat"); s.default_value = 1.0
    N_, L_ = ng.nodes, ng.links; gi, go = N_.new("NodeGroupInput"), N_.new("NodeGroupOutput")
    cam = N_.new("GeometryNodeInputActiveCamera"); oi = N_.new("GeometryNodeObjectInfo"); oi.transform_space = "ORIGINAL"
    L_.new(cam.outputs[0], oi.inputs["Object"])
    pos = N_.new("GeometryNodeInputPosition"); dist = N_.new("ShaderNodeVectorMath"); dist.operation = "DISTANCE"
    L_.new(pos.outputs[0], dist.inputs[0]); L_.new(oi.outputs["Location"], dist.inputs[1])
    mul = N_.new("ShaderNodeMath"); mul.operation = "MULTIPLY_ADD"                     # d0 + k · distance
    L_.new(dist.outputs["Value"], mul.inputs[0]); L_.new(gi.outputs["k"], mul.inputs[1]); L_.new(gi.outputs["d0"], mul.inputs[2])
    cap = N_.new("ShaderNodeMath"); cap.operation = "MINIMUM"; L_.new(mul.outputs[0], cap.inputs[0]); L_.new(gi.outputs["dmax"], cap.inputs[1])
    nrm = N_.new("ShaderNodeVectorMath"); nrm.operation = "NORMALIZE"; L_.new(gi.outputs["Sun"], nrm.inputs[0])
    off = N_.new("ShaderNodeVectorMath"); off.operation = "SCALE"; L_.new(nrm.outputs[0], off.inputs[0])
    neg = N_.new("ShaderNodeMath"); neg.operation = "MULTIPLY"; neg.inputs[1].default_value = -1.0; L_.new(cap.outputs[0], neg.inputs[0])
    L_.new(neg.outputs[0], off.inputs["Scale"])
    sp = N_.new("GeometryNodeSetPosition"); L_.new(gi.outputs["Geometry"], sp.inputs["Geometry"]); L_.new(off.outputs[0], sp.inputs["Offset"])
    L_.new(sp.outputs[0], go.inputs[0]); return ng                   # Set Position moves both vertices and instances (measured: Translate on top shifts it twice)


def _shadow_only(o):
    for a in ("visible_camera", "visible_diffuse", "visible_glossy", "visible_transmission", "visible_volume_scatter"): setattr(o, a, False)
    o.visible_shadow = True; o["sk_shadow_twin"] = 1; o["kind"] = "plant_shadow"; return o


def _set(md, name, val):
    for it in md.node_group.interface.items_tree:
        if getattr(it, "in_out", None) == "INPUT" and it.name == name: getattr(md.properties.inputs, it.identifier).value = val if not hasattr(val, "__len__") else tuple(val)


def _layer_colls(lc):
    yield lc
    for c in lc.children: yield from _layer_colls(c)


def _exclude(sc, coll, on):
    for vl in sc.view_layers:
        for lc in _layer_colls(vl.layer_collection):
            if lc.collection == coll: lc.exclude = on


def _coll(sc):
    c = bpy.data.collections.get(COLL)
    if c is None: c = bpy.data.collections.new(COLL); sc.collection.children.link(c)
    return c


def _src(sc):
    return [bpy.data.objects.get(n) for n in sc.get("sk_twin_src", []) if bpy.data.objects.get(n)]


def _excluded(sc):
    """Collections excluded from the view layer (instance prototypes, shadow sources): their objects do not go into the frame — they need no double."""
    return {lc.collection.name for vl in sc.view_layers for lc in _layer_colls(vl.layer_collection) if lc.exclude}


def leaf_objects(sc):
    """Foliage and grasses of the scene: the plants' own meshes with organ maps, and instance nodes (forest, meadow). Objects only in excluded
    collections (prototypes) are not included: their shadow comes from the double of the instance node (otherwise a prototype at the origin would cast a shadow)."""
    out = []; ex = _excluded(sc)
    for o in sc.objects:
        if o.get("sk_shadow_twin") or o.type != "MESH" or o.get("kind") != "plant": continue
        if o.users_collection and all(c.name in ex for c in o.users_collection): continue
        if any(m.type == "NODES" and m.node_group and m.node_group.name.endswith("_gn") for m in o.modifiers): out.append(o)
        elif any(s.material and s.material.name.startswith("card_") for s in o.material_slots): out.append(o)
    return out


def thin_cards(o, N, seed=1):
    """A double gets every Nth foliage card (randomly, by seed), larger by √N: the shadow's places and area stay the same, N times
    fewer triangles (the forest shadow time scales with the doubles' triangles; the shadow of a coarse tier — different card places — darkened crowns by
    8–14 %, while a thinned one — by < 1 %). Its own mesh; cards are 6 consecutive vertices each (bl_plant.cards)."""
    import bmesh
    import numpy as np
    if N <= 1: return 0
    o.data = o.data.copy(); me = o.data; rng = np.random.default_rng(seed)
    V = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get("co", V); V = V.reshape(-1, 3)
    npg = len(me.polygons); mi = np.empty(npg, np.int32); me.polygons.foreach_get("material_index", mi)
    ls = np.empty(npg, np.int32); me.polygons.foreach_get("loop_start", ls); lv = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", lv)
    lt = np.empty(npg, np.int32); me.polygons.foreach_get("loop_total", lt); dead = []
    for i, sl in enumerate(o.material_slots):
        m = sl.material
        if not (m and m.name.startswith("card_")) or not (mi == i).any(): continue
        sel = np.flatnonzero(mi == i)
        if not (lt[sel] == 4).all(): continue
        vs = np.unique(lv[np.add.outer(ls[sel], np.arange(4)).ravel()])
        if len(vs) % 6 or vs[-1] - vs[0] + 1 != len(vs): continue
        Q = vs.reshape(-1, 6); keep = rng.random(len(Q)) < 1.0 / N
        C = V[Q[keep]].mean(1, keepdims=True); V[Q[keep]] = C + (V[Q[keep]] - C) * np.sqrt(N)
        dead += list(sel[~keep[(lv[ls[sel]] - vs[0]) // 6]])
    if not dead: return 0
    me.vertices.foreach_set("co", V.ravel()); me.update()
    bm = bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.faces[int(f)] for f in dead], context="FACES"); bm.to_mesh(me); bm.free(); me.update()
    return len(dead) // 2


def _proto_twin(sc, ng, src):
    """A copy of the instance node tree with a collection of double prototypes (shadow only): copies of the prototypes, with cards sparser by a factor of
    sk_shadow_thin (bl_forest.SHADOW_THIN by tier; the forest shadow costs 3–5 s of a frame, its time scales with triangles)."""
    ci = next(n for n in ng.nodes if n.type == "COLLECTION_INFO"); pc = ci.inputs["Collection"].default_value
    tc = bpy.data.collections.get(pc.name + "__shadow")
    if tc is None:
        tc = bpy.data.collections.new(pc.name + "__shadow"); _coll(sc).children.link(tc)
        for o in sorted(pc.objects, key=lambda o: o.name):                 # children are ordered by name: the prototype indices stay the same
            t = o.copy(); t.name = o.name + "_s"; thin_cards(t, int(o.get("sk_shadow_thin", 1)))
            tc.objects.link(t); _shadow_only(t); src.append(o.name); shrink_cards(t); _own_materials(t)
        _exclude(sc, tc, True)                                          # prototypes do not go into the frame (Collection Info takes an excluded collection too)
    ng2 = bpy.data.node_groups.get(ng.name + "__shadow") or ng.copy(); ng2.name = ng.name + "__shadow"
    next(n for n in ng2.nodes if n.type == "COLLECTION_INFO").inputs["Collection"].default_value = tc
    return ng2


def shadow_material(m):
    """Leaf shadow material: alpha × (1 − leaf transmittance sk_leak) — as much light passes through a leaf's shadow as through the leaf
    (second-order scattering in the crown: in Cycles a backlit crown glows from inside); transparent shadow is enabled. Without leakage — the same material."""
    leak = float(m.get("sk_leak", 0.0)) if m else 0.0
    if not m or leak <= 0.0 or not m.use_transparent_shadow: return m
    nm = m.name + "__shadow"; s_ = bpy.data.materials.get(nm)
    if s_: return s_
    s_ = m.copy(); s_.name = nm; nt = s_.node_tree
    ms = next(n for n in nt.nodes if n.type == "MIX_SHADER"); lk = ms.inputs["Fac"].links[0]; src = lk.from_socket
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul.inputs[1].default_value = 1.0 - leak
    nt.links.new(src, mul.inputs[0]); nt.links.new(mul.outputs[0], ms.inputs["Fac"]); return s_


def shrink_cards(o, far_only=True):
    """The far tier casts a shadow without transparency (frame cost −42 %), but with the whole card square — the crown in shadow is darker than
    in Cycles. A double gets its own mesh: the foliage cards are shrunk toward the centre to the leaf area × (1 − transmittance)
    (√(leaf share · (1 − leakage))): the mean shadow density is right, no transparent shadows. Cards are 6 consecutive vertices each (bl_plant.cards)."""
    mats = [sl.material for sl in o.material_slots]
    far = [i for i, m in enumerate(mats) if m and m.name.startswith("card_") and (not far_only or not m.use_transparent_shadow) and m.get("sk_cov")]
    if not far: return False
    me = o.data.copy(); me.name = o.data.name + "__shadow"; import numpy as np
    V = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get("co", V); V = V.reshape(-1, 3)
    mi = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get("material_index", mi)
    ls = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get("loop_start", ls); lt = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get("loop_total", lt)
    lv = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", lv)
    for i in far:
        m = mats[i]; k = float(np.sqrt(max(0.02, m["sk_cov"] * (1.0 - float(m.get("sk_leak", 0.0))))))
        P = np.unique(np.concatenate([lv[a:a + b] for a, b in zip(ls[mi == i], lt[mi == i])])) if (mi == i).any() else np.zeros(0, int)
        if len(P) < 6: continue
        P = P[: len(P) // 6 * 6].reshape(-1, 6); C = V[P].mean(1, keepdims=True); V[P] = C + (V[P] - C) * k
    me.vertices.foreach_set("co", V.ravel()); me.update(); o.data = me; return True


def _own_materials(o):
    """Shadow materials for a double (own on the object, the mesh is shared)."""
    for sl in o.material_slots:
        m = sl.material; sm = shadow_material(m)
        if sm is not m: sl.link = "OBJECT"; sl.material = sm


def twins(sc, objs, d0=0.02, k=0.02, dmax=1.0):
    """Shadow doubles of foliage objects (mesh) and of instance nodes (forest, meadow); a repeated call makes no new ones. Returns the doubles."""
    tc = _coll(sc); src = list(sc.get("sk_twin_src", [])); out = []
    for ob in objs:
        tw = bpy.data.objects.get(ob.name + "__shadow")
        if tw is None:
            tw = ob.copy(); tw.name = ob.name + "__shadow"; tc.objects.link(tw); _shadow_only(tw); src.append(ob.name); _own_materials(tw)
            pr = next((m for m in ob.modifiers if m.type == "NODES" and m.node_group and m.node_group.name.endswith("_gn")), None)
            if pr is not None: tw.modifiers[pr.name].node_group = _proto_twin(sc, pr.node_group, src)
            md = tw.modifiers.new("shadow_twin", "NODES"); md.node_group = _twin_group()
        md = tw.modifiers["shadow_twin"]; _set(md, "d0", d0); _set(md, "k", k); _set(md, "dmax", dmax); out.append(tw)
    sc["sk_twin_src"] = sorted(set(src)); return out


def sky_k(eevee):
    """The sky share above a leaf (sk_sky_k, if enabled in the species optics) — Eevee only: Cycles occludes the sky with the crown by itself."""
    for m in bpy.data.materials:
        n = m.node_tree.nodes.get("sk_sky_k") if m.node_tree else None
        if n: n.outputs[0].default_value = float(m.get("sk_sky_k", 1.0)) if eevee else 0.0


def enable(sc, on):
    """Doubles for the Eevee render: they cast the shadow, the foliage does not; off — as before (the foliage casts its own shadow). sc['sk_twins_off'] —
    always switch off (a test)."""
    on = bool(on) and not sc.get("sk_twins_off")
    c = bpy.data.collections.get(COLL)
    if c is None: return
    _exclude(sc, c, not on)
    for c2 in c.children: _exclude(sc, c2, True)                       # double prototypes are always outside the layer
    for o in _src(sc): o.visible_shadow = not on
    sc["sk_twins_on"] = int(on)


def light_dir(P):
    """The direction to the node plan's sun/moon: the sun if above the horizon, otherwise the moon (if it shines)."""
    if P["sun"]["el"] > -1.0: return P["sun"]["dir"]
    m = P.get("moon") or {}
    if m.get("on") and m.get("dir"): return m["dir"]
    import math
    if m.get("on"): a, e = math.radians(m["az"]), math.radians(m["el"]); return [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]
    return P["sun"]["dir"]


def aim(sc, P, d0=0.02, k=0.02, dmax=1.0):
    """The node's light is set: shadow doubles along the plan's sun/moon, enabled for the Eevee render (handlers — for this process)."""
    objs = leaf_objects(sc)
    if not objs: return 0
    tw = twins(sc, objs, d0, k, dmax)
    for t in tw: _set(t.modifiers["shadow_twin"], "Sun", list(light_dir(P)))
    enable(sc, sc.render.engine == "BLENDER_EEVEE"); _handlers(); return len(tw)


@persistent
def _pre_render(sc, *_):
    ee = sc.render.engine == "BLENDER_EEVEE"; sky_k(ee)
    if sc.get("sk_twin_src"): enable(sc, ee)


@persistent
def _pre_save(*_):
    for sc in bpy.data.scenes:
        if sc.get("sk_twin_src"): sc["_twins_was"] = sc.get("sk_twins_on", 0); enable(sc, False)


@persistent
def _post_save(*_):
    for sc in bpy.data.scenes:
        if sc.get("_twins_was"): enable(sc, sc.render.engine == "BLENDER_EEVEE")


def _handlers():
    H = bpy.app.handlers
    for lst, fn in ((H.render_pre, _pre_render), (H.save_pre, _pre_save), (H.save_post, _post_save)):
        if not any(getattr(f, "__name__", "") == fn.__name__ and getattr(f, "__module__", "") == fn.__module__ for f in lst): lst.append(fn)
