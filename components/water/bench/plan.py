"""Check of the "water" step without Blender: the Site's preset places (through its api) → water bodies and the water state
(steps.plan_water).
  .venv/bin/python -m components.water.bench.plan [presets, comma-separated] [wind=m/s]
The places live in out/dev/water/places/<preset> (the relief is built once)."""
import json
import os
import re
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def place(name, seed=2):
    """Preset place: knobs at the middles of the catalog ranges, seed; → a folder with relief.npz, location.json."""
    d = os.path.join(ROOT, "out", "dev", "water", "places", name)
    if os.path.exists(os.path.join(d, "relief.npz")): return d
    from components.site import api
    P = yaml.safe_load(open(os.path.join(ROOT, "library", "terrain", "presets", name + ".yaml"), encoding="utf-8"))
    kn = {k: (v[0] if isinstance(v[0], str) else (v[0] + v[1]) / 2) if isinstance(v, list) else v for k, v in (P.get("knobs") or {}).items()}
    txt = yaml.safe_dump(P["relief"], allow_unicode=True)
    for k, v in kn.items():                                             # "{r}" → a number (integers stay integers: n, ridges)
        val = int(round(v)) if isinstance(v, float) and k in ("ridges", "hills", "stacks", "sinks") else v
        txt = txt.replace("'{" + k + "}'", str(val)).replace("{" + k + "}", str(val))
    spec = {"relief": yaml.safe_load(txt), "seed": seed, "biome": P.get("biome")}
    spec["relief"].setdefault("biome", P.get("biome"))
    o = api.place_build(spec); os.makedirs(d, exist_ok=True); api.place_save(o, d)
    return d


def main(names, wind):
    from components.water.steps import plan_water
    ok = True
    for nm in names:
        d = place(nm)
        r = plan_water({}, {"wind": wind}, {"scene": d, "out": d})
        P = json.load(open(r["water_plan"]))
        print(f"{nm}: biome {P['biome']}, wind {P['wind']['u10']} m/s ({P['wind']['describe']}), water bodies {len(P['bodies'])}")
        for b in P["bodies"][:4]:
            extra = f", river: {b['river']['v']} m/s, Fr {b['river']['froude']} — {b['river']['regime']}" if "river" in b else ""
            print(f"  {b['id']} {b['kind']:5s} {b['area']:>9.0f} m², depth {b['depth_mean']}/{b['depth_max']} m, fetch {b['fetch']}, "
                  f"Hs {b['hs']} m, Tp {b['tp']} s, λp {b['lp']} m, water {b['optics']} (FU {b['fu']}){extra}")
        ok &= bool(P["bodies"])
    return ok


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if "=" not in x]; kv = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    names = a[0].split(",") if a else ["mountain_lake", "harbour_bay", "river_crossing"]
    sys.exit(0 if main(names, float(kv["wind"]) if "wind" in kv else None) else 1)
