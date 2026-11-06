"""Measurement of real trees for the draft stand-ins (components/plants/stand_in.py): each one is bl_plant.build at the origin without ground,
tier 0, no tilt; the bounding box of bark and foliage (roots excluded: they are underground), the bottom and top of the crown, crown radii in BANDS belts from the axis
(q90 and the maximum) — by foliage, for bare trees — by branches (bark thinner than 0.3 of the base radius); the mean colour of the foliage images (bl_plant._tex_mean).
blender -b --factory-startup --python-exit-code 1 -P core/blender/run.py -- components/plants/bench/bl_stand_in_measure.py -- <specs.json> <out.json>"""
import json, sys, time

import bpy
import numpy as np

from components.plants import bl_plant as BP

BANDS = 12
FOLIAGE = ("leaves", "fronds", "rachis")


def verts(o):
    V = np.empty(len(o.data.vertices) * 3); o.data.vertices.foreach_get("co", V); return V.reshape(-1, 3)


a = sys.argv[sys.argv.index("--") + 1:]; P = json.load(open(a[0]))["plants"]; out = {"trees": [], "tex": {}}
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
t_all = time.time()
for i, sp in enumerate(P):
    t0 = time.time(); nm = f"m{i}"; objs, cost = BP.build(sp, (0.0, 0.0, 0.0), name=nm); H = float(sp["height"])
    fol, bark, rad = [], np.zeros((0, 3)), np.zeros(0)
    for o in objs:
        part = o.name[len(nm) + 1:]
        if part == "bark":
            bark = verts(o); ra = o.data.attributes.get("radius"); rad = np.zeros(len(bark))
            if ra is not None: ra.data.foreach_get("value", rad)
        elif part in FOLIAGE:
            fol.append(verts(o))
            for m in o.data.materials:
                im = next((n.image for n in m.node_tree.nodes if n.type == "TEX_IMAGE" and n.image), None) if m and m.node_tree else None
                if im is not None: out["tex"].setdefault(im.name.rsplit(".", 1)[0], [round(float(x), 5) for x in BP._tex_mean(im)])
    F = np.vstack(fol) if fol else np.zeros((0, 3)); ALL = np.vstack([F, bark])
    r_base = float(rad[bark[:, 2] < 0.05 * H].max()) if len(bark) and (bark[:, 2] < 0.05 * H).any() else 0.0
    C = F if len(F) else bark[rad < 0.3 * max(r_base, 1e-3)]                     # crown: the foliage; for bare trees — the branches (a palm trunk has no crown)
    rec = {"id": sp["id"], "species": sp["species"], "age": sp["age"], "state": sp["state"], "seed": sp["seed"], "H": H,
           "foliage": bool(len(F)), "n_crown": int(len(C)), "r_base": round(r_base, 3)}
    if len(ALL):
        z1 = float(ALL[:, 2].max()); rec.update(min=[round(float(x), 3) for x in ALL.min(0)], max=[round(float(x), 3) for x in ALL.max(0)], z1=round(z1, 3))
    if len(ALL) and len(C) > 8:
        z0 = float(np.quantile(C[:, 2], 0.02)); e = np.linspace(z0, z1, BANDS + 1)
        r = np.hypot(C[:, 0], C[:, 1]); b = np.clip(np.searchsorted(e, C[:, 2], side="right") - 1, 0, BANDS - 1)
        rq = [float(np.quantile(r[b == k], 0.9)) if (b == k).any() else 0.0 for k in range(BANDS)]
        rm = [float(r[b == k].max()) if (b == k).any() else 0.0 for k in range(BANDS)]
        rec.update(z0=round(z0, 3), rq=[round(x, 3) for x in rq], rm=[round(x, 3) for x in rm])
    rec["sec"] = round(time.time() - t0, 2); out["trees"].append(rec)
    for o in objs:
        me = o.data; bpy.data.objects.remove(o); bpy.data.meshes.remove(me)
    print(f"{i + 1}/{len(P)} {sp['id']} H={H:.1f} {rec['sec']} s", flush=True)
json.dump(out, open(a[1], "w"), ensure_ascii=False)
print("__RESULT__" + json.dumps({"trees": len(out["trees"]), "tex": len(out["tex"]), "sec": round(time.time() - t_all, 1)}, ensure_ascii=False))
