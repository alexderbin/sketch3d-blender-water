"""Belt rocks as the shape of the wall (Geometry Nodes on the terrain_belt mesh) — instead of a scatter of separate rocks in the distance (separate small rocks look bad;
the logic of rocky outcrops on the walls was missing; an even carpet of blocks gave uniform noise and squares of snow). The structure comes from the mountain itself:
- walls steeper than 42° closer than near to the active camera get finer (4.7 m → 2.3 m at level 1);
- ledges follow the site's strata package (the same step, tilt and ragged edge as the far shader's bands core/surface/bl_terrain_vista), and
  each stratum has its own overhang: hard ones protrude as a cornice, soft ones are almost smooth;
- couloirs are cut along the runoff (the thalwegs of the belt map relief_far.png), buttresses sit on ridges and spurs;
- blocks and ripples are of small amplitude, blocks only in hard strata.
The displacement weight is set at vertices before the mesh is split (the edge of the selection is 0): no cracks. The normal before displacement is kept in the attribute sk_n0: the far shader's layers
(snow, forest, rock) follow it as if there were no rocks. Cost (Garden Wall ×0.7, three 960×540 frames): belt faces 0.4 → 0.9 million, the frame +47 %."""
import os

import bpy


class _N:
    def __init__(self, ng):
        self.ng, self.L = ng, ng.links

    def n(self, kind, **kw):
        x = self.ng.nodes.new(kind)
        for k, v in kw.items(): setattr(x, k, v)
        return x

    def put(self, sock, v):
        if isinstance(v, (int, float)): sock.default_value = float(v)
        elif isinstance(v, tuple): sock.default_value = v
        else: self.L.new(v, sock)

    def m(self, op, a, b=None):
        x = self.n("ShaderNodeMath", operation=op); self.put(x.inputs[0], a)
        if b is not None: self.put(x.inputs[1], b)
        return x.outputs[0]

    def v(self, op, a, b=None, s=None):
        x = self.n("ShaderNodeVectorMath", operation=op); self.put(x.inputs[0], a)
        if b is not None: self.put(x.inputs[1], b)
        if s is not None: self.put(x.inputs["Scale"], s)
        return x.outputs["Value"] if op in ("DISTANCE", "LENGTH", "DOT_PRODUCT") else x.outputs["Vector"]

    def ramp(self, x, lo, hi):
        r = self.n("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP"); self.put(r.inputs["Value"], x)
        self.put(r.inputs["From Min"], lo); self.put(r.inputs["From Max"], hi); return r.outputs["Result"]


def build(ob, data, strata=None, seed=1, amp=1.4, near=2000.0, level=1, hole=None):
    """ob — the belt mesh (terrain_belt), data — the belt map relief_far.png (R ridges, G thalwegs), strata — the site's strata package (location.json).
    → a modifier."""
    if ob is None or not os.path.exists(data): return None
    old = ob.modifiers.get("crags")
    if old is not None: ob.modifiers.remove(old)
    ng = bpy.data.node_groups.new("crags", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    g = _N(ng); gi, go = g.n("NodeGroupInput"), g.n("NodeGroupOutput")
    P = g.n("GeometryNodeInputPosition").outputs[0]; Nrm = g.n("GeometryNodeInputNormal").outputs[0]
    nz = g.n("ShaderNodeSeparateXYZ"); g.put(nz.inputs[0], Nrm)
    slope = g.m("DEGREES", g.m("ARCCOSINE", g.m("MINIMUM", nz.outputs["Z"], 1.0)))
    cam = g.n("GeometryNodeInputActiveCamera"); oi = g.n("GeometryNodeObjectInfo", transform_space="ORIGINAL"); g.put(oi.inputs["Object"], cam.outputs[0])
    dist = g.v("DISTANCE", P, oi.outputs["Location"])                # to the frame's camera (recomputed for each camera)
    sel = g.m("MULTIPLY", g.m("GREATER_THAN", slope, 42.0), g.m("LESS_THAN", dist, near))   # face: a wall and close
    if hole:                                                         # not near the plot (+40 m): there the belt goes under the near grid — steep and crumpled
        x0, y0, x1, y1 = (float(v) for v in hole); hp = g.n("ShaderNodeSeparateXYZ"); g.put(hp.inputs[0], P)
        out_ = g.m("MAXIMUM", g.m("MAXIMUM", g.m("SUBTRACT", x0 - 40.0, hp.outputs["X"]), g.m("SUBTRACT", hp.outputs["X"], x1 + 40.0)),
                   g.m("MAXIMUM", g.m("SUBTRACT", y0 - 40.0, hp.outputs["Y"]), g.m("SUBTRACT", hp.outputs["Y"], y1 + 40.0)))
        sel = g.m("MULTIPLY", sel, g.m("GREATER_THAN", out_, 0.0))
    sf = g.n("GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="FACE"); sf.inputs["Name"].default_value = "crag_sel"
    g.put(sf.inputs["Geometry"], gi.outputs[0]); g.put(sf.inputs["Value"], sel)   # the face selection goes as a face attribute: a vertex reads the mean
    ev = g.n("GeometryNodeInputNamedAttribute", data_type="FLOAT"); ev.inputs["Name"].default_value = "crag_sel"   # over its own faces (a field on
    # points took the vertex slope — the selection edge shifted and the mesh tore)
    fade = g.m("MULTIPLY", g.m("SUBTRACT", 1.0, g.ramp(dist, near * 0.5, near)), g.ramp(dist, 80.0, 250.0))   # near the camera (< 80 m) — none:
    w = g.m("MULTIPLY", g.m("GREATER_THAN", ev.outputs["Attribute"], 0.999), g.m("MULTIPLY", g.ramp(slope, 44.0, 54.0), fade))   # a 2.3 m grid folds the faces
    st = g.n("GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="POINT"); st.inputs["Name"].default_value = "crag_w"
    g.put(st.inputs["Geometry"], sf.outputs[0]); g.put(st.inputs["Value"], w)
    im = bpy.data.images.load(data, check_existing=True); im.colorspace_settings.name = "Non-Color"; im.alpha_mode = "CHANNEL_PACKED"
    uv = g.n("GeometryNodeInputNamedAttribute", data_type="FLOAT_VECTOR"); uv.inputs["Name"].default_value = "ground"
    tx = g.n("GeometryNodeImageTexture", interpolation="Linear", extension="EXTEND"); tx.inputs["Image"].default_value = im; g.put(tx.inputs["Vector"], uv.outputs["Attribute"])
    dd = g.n("FunctionNodeSeparateColor"); g.put(dd.inputs[0], tx.outputs["Color"]); ridge, thal = dd.outputs[0], dd.outputs[1]
    wc = g.m("MULTIPLY", g.m("GREATER_THAN", ev.outputs["Attribute"], 0.999), g.m("MULTIPLY", g.ramp(ridge, 0.45, 0.85), fade))
    sc_ = g.n("GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="POINT"); sc_.inputs["Name"].default_value = "crest_w"   # ridge: the weight comes from the map
    g.put(sc_.inputs["Geometry"], st.outputs[0]); g.put(sc_.inputs["Value"], wc)                               # of ridges (the slope is small right on the ridge)
    sn = g.n("GeometryNodeStoreNamedAttribute", data_type="FLOAT_VECTOR", domain="POINT"); sn.inputs["Name"].default_value = "sk_n0"
    g.put(sn.inputs["Geometry"], sc_.outputs[0]); g.put(sn.inputs["Value"], Nrm)   # the normal before displacement — for the shader layers
    sep = g.n("GeometryNodeSeparateGeometry", domain="FACE"); g.put(sep.inputs["Geometry"], sn.outputs[0]); g.put(sep.inputs["Selection"], g.m("GREATER_THAN", sel, 0.5))
    sub = g.n("GeometryNodeSubdivideMesh"); g.put(sub.inputs["Mesh"], sep.outputs["Selection"]); sub.inputs["Level"].default_value = level
    St = strata or {}; S_ = 2 * 3.0 * float(St.get("step", 2.6)); dip = float(St.get("dip", 0.08)); sd = int(seed)
    pz = g.n("ShaderNodeSeparateXYZ"); g.put(pz.inputs[0], P)                   # the strata package — like the far shader's bands (zp and an edge noise of 60 m)
    en = g.n("ShaderNodeTexNoise", noise_dimensions="3D"); en.inputs["Scale"].default_value = 1 / 60.0; en.inputs["Detail"].default_value = 3.0
    g.put(en.inputs["Vector"], g.v("ADD", P, ((sd + 41) * 13.7, (sd + 41) * 7.3, (sd + 41) * 3.1)))
    zp = g.m("ADD", g.m("ADD", pz.outputs["Z"], g.m("MULTIPLY", g.m("ADD", g.m("MULTIPLY", pz.outputs["X"], 0.6), g.m("MULTIPLY", pz.outputs["Y"], 0.8)), dip)),
             g.m("MULTIPLY", g.m("SUBTRACT", en.outputs["Fac"], 0.5), 9.0))
    q = g.m("DIVIDE", zp, S_); li = g.m("FLOOR", q); f = g.m("FRACT", q)
    ln = g.n("ShaderNodeTexNoise", noise_dimensions="1D"); ln.inputs["Scale"].default_value = 1.0; ln.inputs["Detail"].default_value = 0.0
    g.put(ln.inputs["W"], g.m("ADD", g.m("MULTIPLY", li, 0.37), float(sd)))
    hard = g.ramp(ln.outputs["Fac"], 0.45, 0.7)                         # a hard stratum — a cornice, a soft one — almost smooth
    prof = g.m("SUBTRACT", g.m("SUBTRACT", g.ramp(f, 0.0, 0.8), g.ramp(f, 0.8, 1.0)), 0.5)   # overhang toward the top of the stratum, a shelf above it
    ledge = g.m("MULTIPLY", prof, g.m("ADD", 0.3, g.m("MULTIPLY", hard, 2.7)))
    gully = g.m("MULTIPLY", g.ramp(thal, 0.35, 0.85), -1.5)              # a couloir along the runoff — shallow: the real couloirs are already in the height map;
    # a 6 m cut (×1.4) gave a smooth trench along the thalweg — gradient "streams"
    butt = g.m("MULTIPLY", g.m("MULTIPLY", g.ramp(ridge, 0.3, 0.8), g.m("SUBTRACT", 1.0, g.ramp(thal, 0.35, 0.85))), 3.0)   # buttress
    vor = g.n("ShaderNodeTexVoronoi", voronoi_dimensions="3D", feature="F1"); g.put(vor.inputs["Vector"], P); vor.inputs["Scale"].default_value = 1 / 5.0
    cell = g.n("FunctionNodeSeparateColor"); g.put(cell.inputs[0], vor.outputs["Color"])
    ved = g.n("ShaderNodeTexVoronoi", voronoi_dimensions="3D", feature="DISTANCE_TO_EDGE"); g.put(ved.inputs["Vector"], P); ved.inputs["Scale"].default_value = 1 / 5.0
    blocks = g.m("MULTIPLY", g.m("ADD", g.m("SUBTRACT", cell.outputs[0], 0.5), g.m("MULTIPLY", g.m("SUBTRACT", 1.0, g.ramp(ved.outputs["Distance"], 0.0, 0.08)), -0.5)), hard)
    rn = g.n("ShaderNodeTexNoise", noise_dimensions="3D"); rn.inputs["Scale"].default_value = 1 / 1.5; rn.inputs["Detail"].default_value = 2.0; g.put(rn.inputs["Vector"], P)
    rough = g.m("MULTIPLY", g.m("SUBTRACT", rn.outputs["Fac"], 0.5), 0.5)
    h = g.m("ADD", g.m("ADD", ledge, gully), g.m("ADD", g.m("ADD", butt, blocks), rough))
    wa = g.n("GeometryNodeInputNamedAttribute", data_type="FLOAT"); wa.inputs["Name"].default_value = "crag_w"
    off = g.v("SCALE", Nrm, s=g.m("MULTIPLY", g.m("MULTIPLY", h, wa.outputs["Attribute"]), amp))
    tv = g.n("ShaderNodeTexVoronoi", voronoi_dimensions="3D", feature="F1"); tv.inputs["Scale"].default_value = 1 / 9.0
    sxy = g.n("ShaderNodeCombineXYZ"); g.put(sxy.inputs[0], pz.outputs["X"]); g.put(sxy.inputs[1], pz.outputs["Y"]); g.put(tv.inputs["Vector"], sxy.outputs[0])   # a tooth — a pillar in plan view
    tc = g.n("FunctionNodeSeparateColor"); g.put(tc.inputs[0], tv.outputs["Color"])
    cw = g.n("GeometryNodeInputNamedAttribute", data_type="FLOAT"); cw.inputs["Name"].default_value = "crest_w"
    teeth = g.m("MULTIPLY", g.m("MULTIPLY", g.m("SUBTRACT", tc.outputs[0], 0.45), 7.0), g.m("MULTIPLY", cw.outputs["Attribute"], amp))   # gendarmes ±3 m
    cz = g.n("ShaderNodeCombineXYZ"); g.put(cz.inputs[2], teeth); off = g.v("ADD", off, cz.outputs[0])
    shelf = g.m("MULTIPLY", g.m("MULTIPLY", g.ramp(f, 0.8, 0.9), g.m("SUBTRACT", 1.0, g.ramp(f, 0.96, 1.0))), g.m("MULTIPLY", hard, wa.outputs["Attribute"]))
    sl_ = g.n("GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="POINT"); sl_.inputs["Name"].default_value = "sk_ledge"   # the stratum's shelf —
    g.put(sl_.inputs["Geometry"], sub.outputs[0]); g.put(sl_.inputs["Value"], shelf)                                                # for the shader's snow and grass
    sp = g.n("GeometryNodeSetPosition"); g.put(sp.inputs["Geometry"], sl_.outputs[0]); g.put(sp.inputs["Offset"], off)
    join = g.n("GeometryNodeJoinGeometry"); g.put(join.inputs[0], sp.outputs[0]); g.put(join.inputs[0], sep.outputs["Inverted"])
    g.put(go.inputs[0], join.outputs[0])
    md = ob.modifiers.new("crags", "NODES"); md.node_group = ng
    return md
