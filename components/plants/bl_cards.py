"""Grass cards — the cheap tier of the meadow: models of 4–8 triangles (files in
library/plants/cards/<set>/: .glb + cards.yaml — meadow kind → models). Light is as for the ground: the shader normal is a constant "up" in
the world (Eevee flips a mesh's given normal on the back face — instances with the back face toward the camera come out black).
Colour — the texture image / its mean colour × the point colour (the instance attribute "cc", linear): the card gives detail, the mean colour comes
from the kind and dryness at that point, as for tufts (the tiers join without a ring). Alpha — a threshold of 0.5 (the models' own alpha key; the export does not carry it).
load(...) → {name: Card}; the instances are built by bl_meadow.build_cards."""
import os
from collections import namedtuple

import bpy
import numpy as np
import yaml

Card = namedtuple("Card", "ob h w tris mean")                            # object, height and width (m), triangles, mean colour (linear)


def library(folder):
    """Card set: the folder's cards.yaml → {"dir": folder, "species": {species: [models]}}."""
    y = yaml.safe_load(open(os.path.join(folder, "cards.yaml"))) or {}
    return {"dir": folder, "species": {k: list(v) for k, v in (y.get("species") or {}).items()}}


def _lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _mean_colour(ob, img, n=48, seed=1):
    """Mean colour of a model (linear): points over its triangles in the UV layout (weighted by area), opaque ones only."""
    me = ob.data; me.calc_loop_triangles(); uv = me.uv_layers[0].data; W, H = img.size
    px = np.empty(W * H * 4, np.float32); img.pixels.foreach_get(px); px = px.reshape(H, W, 4); r = np.random.default_rng(seed); acc, wsum = np.zeros(3), 0.0
    for t in me.loop_triangles:
        U = np.array([uv[l].uv[:] for l in t.loops]); a = float(t.area)
        b = r.dirichlet((1, 1, 1), n); q = b @ U; i = np.clip((q[:, 1] % 1.0) * H, 0, H - 1).astype(int); j = np.clip((q[:, 0] % 1.0) * W, 0, W - 1).astype(int)
        s = px[i, j]; ok = s[:, 3] > 0.5
        if ok.any(): acc += _lin(s[ok, :3]).sum(0) * a / n; wsum += ok.sum() * a / n
    return acc / wsum if wsum > 0 else np.array([0.1, 0.18, 0.05])


def load(folder, names, coll, tinted=True):
    """The set's models → objects in coll (meshes merged, origin at the middle of the base). tinted — the colour comes from the instance attribute "cc"."""
    out = {}
    for nm in names:
        p = os.path.join(folder, nm + ".glb")
        if not os.path.exists(p): continue
        before = set(bpy.data.objects); bpy.ops.import_scene.gltf(filepath=p)
        obs = [o for o in bpy.data.objects if o not in before]                 # Icosphere — the shape of a glTF import bone, not a model
        ms = [o for o in obs if o.type == "MESH" and any(m and m.node_tree and any(n.type == "TEX_IMAGE" for n in m.node_tree.nodes) for m in o.data.materials)]
        for o in ms:
            mw = o.matrix_world.copy(); o.parent = None; o.modifiers.clear(); o.data.transform(mw); o.matrix_world.identity()
            for c in list(o.users_collection): c.objects.unlink(o)
            coll.objects.link(o)
        for o in obs:
            if o not in ms: bpy.data.objects.remove(o)
        if not ms: continue
        if len(ms) > 1:
            with bpy.context.temp_override(active_object=ms[0], object=ms[0], selected_objects=ms, selected_editable_objects=ms):
                bpy.ops.object.join()
        ob = ms[0]; ob.name = ob.data.name = "card_" + nm; ob["kind"] = "plant"
        V = np.empty(len(ob.data.vertices) * 3, np.float32); ob.data.vertices.foreach_get("co", V); V = V.reshape(-1, 3)
        V[:, :2] -= (V[:, :2].min(0) + V[:, :2].max(0)) / 2; V[:, 2] -= V[:, 2].min(); ob.data.vertices.foreach_set("co", V.ravel()); ob.data.update()
        mean = None
        for m in ob.data.materials:
            if m is None: continue
            m.use_backface_culling = False; m.surface_render_method = "DITHERED"
            N_, L_ = m.node_tree.nodes, m.node_tree.links; tx = next(n for n in N_ if n.type == "TEX_IMAGE"); bs = next(n for n in N_ if n.type == "BSDF_PRINCIPLED")
            tx.image.alpha_mode = "STRAIGHT"; bs.inputs["Roughness"].default_value = 0.85
            if mean is None: mean = _mean_colour(ob, tx.image)
            th = N_.new("ShaderNodeMath"); th.operation = "GREATER_THAN"; th.inputs[1].default_value = 0.5
            L_.new(tx.outputs["Alpha"], th.inputs[0]); L_.new(th.outputs[0], bs.inputs["Alpha"])
            up = N_.new("ShaderNodeCombineXYZ"); up.inputs["Z"].default_value = 1.0; L_.new(up.outputs[0], bs.inputs["Normal"])
            if tinted:                                                            # colour = image / mean × point colour
                at = N_.new("ShaderNodeAttribute"); at.attribute_type = "INSTANCER"; at.attribute_name = "cc"
                inv = N_.new("ShaderNodeCombineXYZ")
                for k, a in zip("XYZ", 1.0 / np.maximum(mean, 1e-3)): inv.inputs[k].default_value = float(a)
                m1 = N_.new("ShaderNodeVectorMath"); m1.operation = "MULTIPLY"; L_.new(tx.outputs["Color"], m1.inputs[0]); L_.new(inv.outputs[0], m1.inputs[1])
                m2 = N_.new("ShaderNodeVectorMath"); m2.operation = "MULTIPLY"; L_.new(m1.outputs[0], m2.inputs[0]); L_.new(at.outputs["Vector"], m2.inputs[1])
                L_.new(m2.outputs[0], bs.inputs["Base Color"])
        ext = V.max(0) - V.min(0)
        out[nm] = Card(ob, float(ext[2]), float(max(ext[0], ext[1], 1e-3)), sum(len(f.vertices) - 2 for f in ob.data.polygons), mean)
    return out
