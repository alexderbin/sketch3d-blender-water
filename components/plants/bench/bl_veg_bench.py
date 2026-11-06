"""Building the reference plants in Blender: each one at the origin without ground; the answer — vertices, foliage cards, bounding size (to 1 cm) and
a fingerprint of the vertices (rounded to 1 mm) by parts. blender -b --factory-startup -P core/blender/run.py -- components/plants/bench/bl_veg_bench.py -- <specs.json>"""
import hashlib, json, sys
import bpy
import numpy as np

from components.plants import bl_plant as BP

P = json.load(open(sys.argv[sys.argv.index("--") + 1]))["plants"]; out = {}
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
for sp in P:
    objs, cost = BP.build(sp, (0.0, 0.0, 0.0), name=sp["id"]); rec = {"verts": cost["verts"], "leaves": cost.get("leaves", 0)}
    for o in objs:
        V = np.empty(len(o.data.vertices) * 3); o.data.vertices.foreach_get("co", V); V = V.reshape(-1, 3)
        part = o.name[len(sp["id"]) + 1:]
        rec[part] = [len(V), hashlib.md5(np.round(V, 3).astype(np.float32).tobytes()).hexdigest()[:10]] + [round(float(x), 2) for x in np.concatenate([V.min(0), V.max(0)])]
    for o in objs:                                                    # light through a leaf: thickness 0, share of sky above the leaf
        if o.name.endswith("_leaves") and "sk_sky" in o.data.attributes:
            v = np.empty(len(o.data.vertices), np.float32); o.data.attributes["sk_sky"].data.foreach_get("value", v)
            th = [n for m in o.data.materials for n in m.node_tree.nodes if n.name == "sk_thick"]
            rec["light"] = {"sky": round(float(v.mean()), 2), "thickness0": bool(th) and all(n.outputs[0].default_value == 0.0 and n.outputs[0].is_linked for n in th)}
    out[sp["id"]] = rec
    for o in objs: bpy.data.objects.remove(o)
# shadow doubles (bl_veg_light): along the plan's sun/moon, for the Eevee render only; Workbench and saving go without them; a leaf's shadow lets light through
from components.plants import bl_veg_light as VL
sc = bpy.context.scene; sp = next(x for x in P if x["species"] == "oak" and x["state"] == "healthy")
objs, _ = BP.build(sp, (0.0, 0.0, 0.0), name="tw"); [o.__setitem__("kind", "plant") for o in objs]
sc.render.engine = "BLENDER_EEVEE"; n = VL.aim(sc, {"sun": {"el": 20.0, "dir": [0.0, 0.94, 0.34]}, "moon": {"on": False}})
tw = [o for o in sc.objects if o.get("sk_shadow_twin")]; lv = next(o for o in objs if o.name.endswith("_leaves"))
ee = lv.visible_shadow is False
sc.render.engine = "BLENDER_WORKBENCH"; VL._pre_render(sc); wb = lv.visible_shadow is True and sc.get("sk_twins_on") == 0
sc.render.engine = "BLENDER_EEVEE"; VL._pre_render(sc); VL._pre_save(); sv = lv.visible_shadow is True and sc.get("sk_twins_on") == 0; VL._post_save()
sm = [s_.material for o in tw for s_ in o.material_slots if s_.material and s_.material.name.endswith("__shadow")]
out["twins"] = {"twins_n": n, "eevee": bool(ee and sc.get("sk_twins_on") == 1), "workbench_without": bool(wb), "save_without": bool(sv),
                   "shadow_transmits": round(1.0 - next((nd.inputs[1].default_value for m in sm for nd in m.node_tree.nodes if nd.type == "MATH" and nd.operation == "MULTIPLY" and nd.inputs[0].is_linked), 1.0), 3)}
print("__RESULT__" + json.dumps(out, ensure_ascii=False))
