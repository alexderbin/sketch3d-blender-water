"""Site rocks — the plan (the Site step rocks): where boulders, cliff faces, rock outcrops, scree and stones stand — from the relief
data, by the same dressing mechanism as the near plot (terrain_dress: density from the site maps, a footprint pattern, large before
small). Near plot — the rock layers of the biome's ecotope; mountain belt — the large ones (dress.yaml → belt)
from the belt grid analysis. Assembly — bl_rocks.py (Poly Haven scans as instances). Output — rocks.json in the scene folder:
{places: [{layer, region, x, y, z, s, rot, sink, tilt}], stats}.
.venv/bin/python -m components.site.terrain_rocks <site_folder> [seed] [density=1] [belt=0]
By default the belt has no scatter (isolated rocks in the distance read as litter; belt rocks come from the relief shape)."""
import json, os, sys, time

import numpy as np

from components.site import terrain_dress as TD


def plan(folder, seed=11, density=1.0, belt=False, avoid=(), limit=9000):
    """→ {places, stats}; avoid — [(x, y, r)] (buildings, squads)."""
    import yaml
    from core.surface.ground import Ground
    t0 = time.time(); G = Ground(folder); L = G.loc; biome = L.get("biome") or "temperate"; k = float(density)
    if L.get("stones") is False: return {"places": [], "stats": {"no stones": "site without stones (stones: false)"}}
    near = [dict(l, base=float(l.get("base", 0.1)) * k) for l in TD.ecotope(biome) if l["kind"] == "rock"]
    P, st = TD.plan(G, seed=seed, biome=biome, avoid=avoid, limit=limit, layers=near)
    for p in P: p["region"] = "near"
    stats = {"near": st}
    if belt and (G.far_bounds()[2] - G.far_bounds()[0]) > (G.near_bounds()[2] - G.near_bounds()[0]) * 1.5:
        lays = [dict(l, base=float(l.get("base", 0.1)) * k) for l in (yaml.safe_load(open(TD.RULES)).get("belt") or {}).get("layers", [])]
        fb, nb = G.far_bounds(), G.near_bounds(); pad = 0.06 * (fb[2] - fb[0])   # the belt edge — not all the way to the grid edge
        Pb, sb = TD.plan(TD.BeltView(G), seed=seed + 1, biome=biome, avoid=avoid, limit=limit, layers=lays,
                         box=(fb[0] + pad, fb[1] + pad, fb[2] - pad, fb[3] - pad), hole=nb)
        for p in Pb: p["region"] = "belt"
        P += Pb; stats["belt"] = sb
    stats["s"] = round(time.time() - t0, 1)
    return {"places": P, "stats": stats}


def main():
    a = sys.argv[1:]; folder = os.path.abspath(a[0]); kv = dict(x.split("=", 1) for x in a[1:] if "=" in x)
    seed = int(a[1]) if len(a) > 1 and a[1].isdigit() else 11
    R = plan(folder, seed, float(kv.get("density", 1.0)), kv.get("belt", "0") != "0")
    json.dump(R, open(os.path.join(folder, "rocks.json"), "w"), ensure_ascii=False)
    print("__RESULT__" + json.dumps({"n_places": len(R["places"]), **R["stats"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
