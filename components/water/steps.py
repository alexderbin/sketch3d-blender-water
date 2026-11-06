"""The Water entry point for step-by-step assembly: the "water" step — the plan: the water bodies of the Site relief and the scene
wind → the water state of each water body (fetch, depth, Hs, peak period and length, whitecaps, the substance in the water and
its Forel–Ule index, for a river — speed and regime), water.json in the step folder. The Blender assembly (a mesh from the
camera, the frame on the fly) is the next step of the plan; showcases — bench/stand.py."""
import json
import math
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def plan_water(M, keys, ctx):
    from core.light import wind as WD
    from core.surface.ground import Ground
    from components.water import bodies as B, optics as O, waves as W
    Gr = Ground(ctx["scene"]); wd = WD.of_scene(ctx["scene"])
    u10 = float(keys["wind"]) if keys.get("wind") is not None else wd.u10
    to = wd.dir_to if u10 > 0 else (1.0, 0.0)
    kinds = yaml.safe_load(open(os.path.join(HERE, "kinds.yaml"), encoding="utf-8"))["kinds"]
    biome = Gr.loc.get("biome") or "temperate"
    out = []
    for b in B.bodies(Gr, to):
        k = kinds[b["kind"]]
        preset = keys.get("optics") or k["optics"].get(biome, k["optics"]["default"])
        sea = W.Sea(u10=max(u10, 0.3), wind_to=to, fetch=b["fetch"], depth=max(b["depth_mean"], 0.2) if b["kind"] != "sea" else None,
                    near=2.0, foam={"thr": -10.0})
        st = sea.stats(); L = O.preset(preset)
        b.update({"wind": {"u10": round(u10, 2), "to": [round(to[0], 3), round(to[1], 3)], "beaufort": WD.beaufort_of(u10)},
                  "hs": round(st["hs"], 3), "tp": round(st["tp"], 2), "lp": round(st["lp"], 2), "whitecaps": round(st["whitecaps"], 5),
                  "optics": preset, "fu": L["fu"], "body": [round(c, 4) for c in L["body"]], "K": [round(c, 3) for c in L["K"]],
                  "chop": k["chop"], "gust": k["gust"]})
        out.append(b)
    plan = {"biome": biome, "folder": os.path.abspath(ctx["scene"]),
            "wind": {"u10": round(u10, 2), "to": list(to), "describe": WD.BEAUFORT[WD.beaufort_of(u10)][2]}, "bodies": out}
    p = os.path.join(ctx.get("out") or ctx["scene"], "water.json")
    json.dump(plan, open(p, "w"), ensure_ascii=False, indent=1)
    return {"water_plan": p, "bodies": len(out), "kinds": ",".join(sorted({b["kind"] for b in out})) or "no water"}


def build_water(ctx):
    """Assembly: water for each scene camera (bl_live.build) by the step plan; the Site's water is removed from the render (it remains for the sketch)."""
    from components.water import bl_live
    plan = ctx["plan"]; sc = ctx["sc"]
    if not plan.get("bodies"): return {"water": "no water"}
    P = json.load(open(plan["water_plan"])) if plan.get("water_plan") else plan
    cams = [c for c in json.load(open(os.path.join(ctx["scene"], "cameras.json"))) if c.get("name")]
    kinds = yaml.safe_load(open(os.path.join(HERE, "kinds.yaml"), encoding="utf-8"))["kinds"]
    return bl_live.build(sc, P, cams, ctx["collection"], kinds)
