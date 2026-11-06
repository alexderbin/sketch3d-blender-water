"""Storm spray: from force 8 the wind tears spray off the crests (spindrift), from 9 the spray haze obstructs the view. The layer
is a second mesh of the same camera above the water: where a crest has fresh foam, the mesh rises by a fraction of the wave
height and is carried downwind; the material is a white veil with strands along the wind (alpha, no shadow). Stateless between
frames, like the water."""
import math

import bpy
import numpy as np

from components.water.bl_water import N, ss_np


class Spray:
    """Spray above the water w (bl_water.Water): amount — strength (0…1, by force: 8 — 0.3, 10 — 0.7, 12 — 1); lift — rise above the
    crest as a fraction of Hs; carry — downwind drift, m."""

    def __init__(self, w, amount=None, lift=0.22, carry=None):
        self.w = w; sea = w.sea; st = sea.stats(); self.hs = st["hs"]
        self.amount = float(np.clip((sea.u10 - 17.2) / (32.7 - 17.2), 0, 1) * 0.8 + 0.2) if amount is None else float(amount)
        self.lift, self.carry = float(lift), float(carry if carry is not None else 0.25 * sea.u10)
        me = w.obj.data.copy(); me.name = w.name + "_spray"; me.materials.clear()
        self.obj = bpy.data.objects.new(w.name + "_spray", me); (w.obj.users_collection[0] if w.obj.users_collection else w.sc.collection).objects.link(self.obj)
        self.obj.visible_shadow = False; self.obj["kind"] = "spray"
        me.materials.append(self.material())

    def update(self, t):
        """After w.update(t): where a crest has fresh foam — the veil goes up and downwind; the rest sits under the water (invisible)."""
        w = self.w; me = w.obj.data; n = len(me.vertices)
        co = np.empty(n * 3, np.float32); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
        fo = np.zeros(n, np.float32)
        a = me.attributes.get("foam_c")
        if a is not None: a.data.foreach_get("value", fo)
        crest = ss_np(co[:, 2] - w.base[:, 2], 0.15 * self.hs, 0.4 * self.hs)    # crest: the top of the wave
        sp = np.clip(ss_np(fo, 0.45, 0.85) * crest * self.amount * 1.4, 0, 1)   # fresh foam at the crest
        wx, wy = w.sea.wind_to
        up = self.lift * self.hs * sp
        P = co.copy(); P[:, 0] += wx * self.carry * sp; P[:, 1] += wy * self.carry * sp; P[:, 2] += up
        P[:, 2] = np.where(sp > 0.02, P[:, 2], co[:, 2] - 0.5)
        sm = self.obj.data
        sm.vertices.foreach_set("co", P.ravel()); sm.update()
        at = sm.attributes.get("spray") or sm.attributes.new("spray", "FLOAT", "POINT"); at.data.foreach_set("value", sp.astype(np.float32))
        nd = sm.materials[0].node_tree.nodes.get("spray_time")
        if nd: nd.outputs[0].default_value = float(t)

    def material(self):
        mat = bpy.data.materials.new(self.w.name + "_spray"); nt = mat.node_tree; nt.nodes.clear(); g = N(nt)
        tm = g.n("ShaderNodeValue"); tm.name = "spray_time"
        uv = g.n("ShaderNodeUVMap", uv_map="wxy").outputs[0]
        wx, wy = self.w.sea.wind_to; v = 0.6 * self.w.sea.u10
        along = g.m("SUBTRACT", g.m("ADD", g.m("MULTIPLY", g.sep(uv)[0], wx), g.m("MULTIPLY", g.sep(uv)[1], wy)), g.m("MULTIPLY", tm.outputs[0], v))
        cross = g.m("ADD", g.m("MULTIPLY", g.sep(uv)[0], -wy), g.m("MULTIPLY", g.sep(uv)[1], wx))
        nz = g.n("ShaderNodeTexNoise", noise_dimensions="3D"); g.put(nz.inputs["Vector"], g.xyz(g.m("MULTIPLY", along, 0.12), g.m("MULTIPLY", cross, 0.5), g.m("MULTIPLY", tm.outputs[0], 0.4)))
        nz.inputs["Scale"].default_value = 1.0; nz.inputs["Detail"].default_value = 5.0; nz.inputs["Roughness"].default_value = 0.6
        spa = g.n("ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name="spray").outputs["Fac"]
        alpha = g.m("MINIMUM", g.m("MULTIPLY", g.ss(g.m("ADD", nz.outputs["Fac"], g.m("MULTIPLY", spa, 0.4)), 0.4, 0.8), g.m("MULTIPLY", spa, 1.2)), 0.85)
        bs = g.n("ShaderNodeBsdfPrincipled"); g.put(bs.inputs["Base Color"], (0.78, 0.8, 0.82)); bs.inputs["Roughness"].default_value = 1.0
        g.put(bs.inputs["Alpha"], alpha); bs.inputs["Subsurface Weight"].default_value = 0.0
        o = g.n("ShaderNodeOutputMaterial"); g.put(o.inputs["Surface"], bs.outputs[0])
        mat.surface_render_method = "DITHERED"
        return mat
