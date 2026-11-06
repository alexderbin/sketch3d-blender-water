"""Waterfall in Blender (the engine is fall.Fall): the jets are meshes (across × along the path), per frame — vertices from
Fall.sheet(t) (the fluttering of a free jet). Material: at the lip — smooth water (reflects the sky, the rock is visible through
it), with breakup — white (bubbles: scattering and sun translucency); strands run at the water speed — the pattern lives in
"detachment time" coordinates w = t − τ (where the water accelerates, the strands stretch); the edges and the bottom break up
(transparency by noise). Water mist — a volume at the foot: density from the impact line downward-upward and sideways, it rises
and drifts; an Eevee volume is not lit by the sky — an emission was added.
  FallMesh(sc, fall).update(t); mist(sc, fall) → a volume object (update_mist(o, t))."""
import math

import bpy
import numpy as np

from components.water.bl_water import N


class FallMesh:
    """The jets of the waterfall fall (fall.Fall) — objects fall_<k> with vertex attributes: tau (s from the lip), b (breakup),
    u (0…1 across), um (m along the lip), c (contact with the rock)."""

    def __init__(self, sc, fall, coll=None, name="fall", layers=2):
        self.fall, self.sc, self.name, self.layers = fall, sc, name, int(layers)
        coll = coll or sc.collection; self.mat = self.material(); self.objs = []
        for k, sh in enumerate(self._sheets(0.0)):
            nu, nt = sh["P"].shape[:2]
            ii = np.arange(nu * nt).reshape(nu, nt)
            q = np.stack([ii[:-1, :-1], ii[1:, :-1], ii[1:, 1:], ii[:-1, 1:]], -1).reshape(-1, 4)
            me = bpy.data.meshes.new(f"{name}_{k}"); me.vertices.add(nu * nt); me.loops.add(q.size); me.polygons.add(len(q))
            me.vertices.foreach_set("co", sh["P"].reshape(-1, 3).astype(np.float32).ravel())
            me.loops.foreach_set("vertex_index", q.ravel().astype(np.int32))
            me.polygons.foreach_set("loop_start", np.arange(0, q.size, 4, dtype=np.int32)); me.polygons.foreach_set("loop_total", np.full(len(q), 4, np.int32))
            for nm in ("tau", "b", "u", "um", "c", "lay"):
                a = me.attributes.new(nm, "FLOAT", "POINT"); a.data.foreach_set("value", np.broadcast_to(np.asarray(sh[nm], np.float32), sh["tau"].shape).ravel())
            me.update(); me.shade_smooth()
            ob = bpy.data.objects.new(f"{name}_{k}", me); coll.objects.link(ob); ob.visible_shadow = False
            ob["kind"] = "waterfall"; me.materials.append(self.mat); self.objs.append(ob)

    def _sheets(self, t):
        """Jets by layers: layer 1 — the same jet deeper (by 0.35 m × breakup, from the rock toward the niche) with its own pattern: the thickness of the jet."""
        out = []
        for sh in self.fall.sheet(t):
            for L in range(self.layers):
                q = dict(sh); q["lay"] = float(L)
                if L:
                    P = sh["P"].copy(); P[..., 1] -= 0.35 * L * (0.3 + sh["b"]) * (1 - sh["c"]); q["P"] = P
                out.append(q)
        return out

    def update(self, t):
        for ob, sh in zip(self.objs, self._sheets(t)):
            ob.data.vertices.foreach_set("co", sh["P"].reshape(-1, 3).astype(np.float32).ravel()); ob.data.update()
        nd = self.mat.node_tree.nodes.get("fall_time")
        if nd: nd.outputs[0].default_value = float(t)

    def material(self):
        f = self.fall
        mat = bpy.data.materials.new(self.name); nt = mat.node_tree; nt.nodes.clear(); g = N(nt)
        tm = g.n("ShaderNodeValue"); tm.name = "fall_time"; tm.outputs[0].default_value = 0.0
        A = lambda nm: g.n("ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name=nm).outputs["Fac"]
        tau, b, u, um, c, lay = A("tau"), A("b"), A("u"), A("um"), A("c"), A("lay")
        w = g.m("SUBTRACT", tm.outputs[0], tau)                          # detachment time: the pattern drifts with the water
        zl = g.m("MULTIPLY", lay, 11.0)                                  # each layer has its own pattern
        def nz(sx, sw, z_, det):
            n_ = g.n("ShaderNodeTexNoise", noise_dimensions="3D"); g.put(n_.inputs["Vector"], g.xyz(g.m("MULTIPLY", um, sx), g.m("MULTIPLY", w, sw), g.m("ADD", zl, z_)))
            n_.inputs["Scale"].default_value = 1.0; n_.inputs["Detail"].default_value = det; n_.inputs["Roughness"].default_value = 0.55
            return n_.outputs["Fac"]
        n1, n2, n3 = nz(5.0, 1.0, 0.0, 4.0), nz(1.1, 0.35, 3.0, 2.0), nz(14.0, 3.0, 7.0, 2.0)   # strands ~0.2 m (along — v·1 s), clumps, fibers
        mixn = g.m("ADD", g.m("ADD", g.m("MULTIPLY", n1, 0.55), g.m("MULTIPLY", n2, 0.3)), g.m("MULTIPLY", n3, 0.15))
        white = g.ss(g.m("ADD", g.m("ADD", g.m("SUBTRACT", g.m("MULTIPLY", b, 1.25), 0.15), g.m("MULTIPLY", g.m("SUBTRACT", n1, 0.5), 0.5)),
                     g.m("MULTIPLY", c, 0.4)), 0.1, 0.55)                # at the lip — smooth water, with breakup — white
        segw = max(b_ - a for a, b_ in f.segs)
        edge = g.m("MULTIPLY", g.m("MINIMUM", u, g.m("SUBTRACT", 1.0, u)), segw)   # distance to the jet edge, m
        ex = g.m("ADD", edge, g.m("MULTIPLY", g.m("SUBTRACT", g.m("ADD", g.m("MULTIPLY", n2, 0.6), g.m("MULTIPLY", n1, 0.4)), 0.5),
                                  g.m("ADD", 0.3, g.m("MULTIPLY", b, 1.6))))
        fray = g.m("MINIMUM", g.m("MAXIMUM", g.m("DIVIDE", ex, g.m("ADD", 0.1, g.m("MULTIPLY", b, 0.8))), 0.0), 1.0)   # the edges break up (the width grows with breakup)
        cover = g.ss(g.m("SUBTRACT", g.m("ADD", mixn, 0.62), g.m("MULTIPLY", b, 0.62)), 0.47, 0.55)   # breakup — gaps between the strands
        cover = g.m("MULTIPLY", cover, g.m("SUBTRACT", 1.0, g.m("MULTIPLY", lay, 0.35)))
        cover = g.m("MAXIMUM", cover, g.m("MULTIPLY", c, 0.75))           # on the rock — a film, it does not break up
        alpha = g.m("MULTIPLY", cover, fray)
        alpha = g.m("MAXIMUM", alpha, g.m("MULTIPLY", g.ss(g.m("MULTIPLY", tau, -1.0), 0.0, 0.05), fray))   # approaching the lip — solid water
        glass = g.n("ShaderNodeBsdfPrincipled"); glass.inputs["Base Color"].default_value = (0.03, 0.05, 0.05, 1.0)
        glass.inputs["Roughness"].default_value = 0.05; glass.inputs["IOR"].default_value = 1.333
        foam = g.n("ShaderNodeBsdfPrincipled"); g.put(foam.inputs["Base Color"], g.v("SCALE", g.xyz(0.84, 0.87, 0.9), s=g.m("ADD", 0.72, g.m("MULTIPLY", n2, 0.4))))
        foam.inputs["Roughness"].default_value = 0.45
        tr = g.n("ShaderNodeBsdfTranslucent"); tr.inputs["Color"].default_value = (0.78, 0.84, 0.88, 1.0)
        fm = g.n("ShaderNodeMixShader"); fm.inputs[0].default_value = 0.35; g.put(fm.inputs[1], foam.outputs[0]); g.put(fm.inputs[2], tr.outputs[0])
        sh = g.n("ShaderNodeMixShader"); g.put(sh.inputs[0], white); g.put(sh.inputs[1], glass.outputs[0]); g.put(sh.inputs[2], fm.outputs[0])
        gl_a = g.m("ADD", 0.55, g.m("MULTIPLY", white, 0.45))             # smooth water is semi-transparent (the rock shows through)
        tp = g.n("ShaderNodeBsdfTransparent")
        out_ = g.n("ShaderNodeMixShader"); g.put(out_.inputs[0], g.m("MULTIPLY", alpha, gl_a)); g.put(out_.inputs[1], tp.outputs[0]); g.put(out_.inputs[2], sh.outputs[0])
        o = g.n("ShaderNodeOutputMaterial"); g.put(o.inputs["Surface"], out_.outputs[0])
        mat.surface_render_method = "DITHERED"; mat.use_backface_culling = False
        return mat


def mist(sc, fall, strength=1.0, amb=(0.55, 0.6, 0.66)):
    """Water mist at the foot: a box volume; density β = β₀·e^{−|y − y_imp|/R}·e^{−(z − z_pool)/H_p}·noise (it rises and drifts);
    β₀ ~ 0.05 m⁻¹ per unit of impact strength; R ~ 0.15·height + 1.5 m, H_p ~ 0.18·height + 1 m."""
    imp = fall.impact; cx, cy = float(imp[:, 0].mean()), float(imp[:, 1].mean()); W_ = float(np.ptp(imp[:, 0])) + 1.0
    R, Hm = 0.15 * fall.H + 1.5, 0.18 * fall.H + 1.0                   # the cloud lies at the foot: sideways ~R, upward ~H_p
    sx, sy, sz = W_ + 5 * R, 5 * R, 3.5 * Hm
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(cx, cy + 0.5 * R, fall.z_pool + 0.5 * sz - 0.5))
    box = bpy.context.object; box.name = "fall_mist"; box.scale = (sx, sy, sz); box.visible_shadow = False
    m = bpy.data.materials.new("fall_mist"); nt = m.node_tree; nt.nodes.clear(); g = N(nt)
    tm = g.n("ShaderNodeValue"); tm.name = "mist_time"
    tc = g.n("ShaderNodeTexCoord"); p = g.sep(tc.outputs["Object"])
    X, Y, Z = g.m("MULTIPLY", p[0], sx), g.m("ADD", g.m("MULTIPLY", p[1], sy), 0.5 * R), g.m("ADD", g.m("MULTIPLY", p[2], sz), 0.5 * sz - 0.5)
    ex = g.m("MAXIMUM", g.m("SUBTRACT", g.m("ABSOLUTE", X), 0.5 * W_), 0.0)  # sideways — beyond the edge of the jets
    dxy = g.m("SQRT", g.m("ADD", g.m("MULTIPLY", ex, ex), g.m("MULTIPLY", Y, Y)))
    nzv = g.n("ShaderNodeTexNoise", noise_dimensions="4D"); g.put(nzv.inputs["Vector"], g.xyz(g.m("MULTIPLY", X, 0.3), g.m("MULTIPLY", Y, 0.3),
                                                                                             g.m("SUBTRACT", g.m("MULTIPLY", Z, 0.3), g.m("MULTIPLY", tm.outputs[0], 0.35))))
    g.put(nzv.inputs["W"], g.m("MULTIPLY", tm.outputs[0], 0.15)); nzv.inputs["Scale"].default_value = 1.0; nzv.inputs["Detail"].default_value = 3.0
    beta0 = 0.05 * strength * min(1.0, fall.q * fall.W / 2.5 + 0.3)  # β₀ 0.05 m⁻¹: through the cloud at the impact one sees ~80 m, higher up — haze
    dens = g.m("MULTIPLY", g.m("MULTIPLY", g.m("EXPONENT", g.m("DIVIDE", dxy, -R)), g.m("EXPONENT", g.m("DIVIDE", g.m("MAXIMUM", Z, 0.0), -Hm))),
               g.m("MULTIPLY", g.ss(nzv.outputs["Fac"], 0.3, 0.75), beta0))
    win = g.m("MULTIPLY", g.m("MULTIPLY", g.ss(g.m("SUBTRACT", 0.5, g.m("ABSOLUTE", p[0])), 0.0, 0.15), g.ss(g.m("SUBTRACT", 0.5, g.m("ABSOLUTE", p[1])), 0.0, 0.15)),
              g.ss(g.m("SUBTRACT", 0.5, g.m("ABSOLUTE", p[2])), 0.0, 0.2))   # to zero at the box faces: otherwise a hard shadow lands on the rock
    dens = g.m("MULTIPLY", dens, win)
    vol = g.n("ShaderNodeVolumePrincipled"); vol.inputs["Color"].default_value = (0.92, 0.94, 0.96, 1.0); vol.inputs["Anisotropy"].default_value = 0.35
    g.put(vol.inputs["Density"], dens)
    vol.inputs["Emission Color"].default_value = (*amb, 1.0); g.put(vol.inputs["Emission Strength"], g.m("MULTIPLY", dens, 0.9))
    o = g.n("ShaderNodeOutputMaterial"); g.put(o.inputs["Volume"], vol.outputs[0])
    box.data.materials.append(m)
    e = sc.eevee; e.volumetric_start = 0.5; e.volumetric_end = max(e.volumetric_end, 400.0); e.volumetric_tile_size = "4"
    return box


def update_mist(box, t):
    nd = box.data.materials[0].node_tree.nodes.get("mist_time")
    if nd: nd.outputs[0].default_value = float(t)
