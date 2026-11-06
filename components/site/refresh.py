"""The ground again from the edited terrain — after the "building" step: Buildings edit the site's relief.npz (moat, terraces,
yard, road, streets, squares, house pads, the hive pad; masks road / pave / pad), but copy the old ground texture — the moat,
terraces and streets are not visible in the ground, and the Eevee surface and the small items know nothing about the edits. Here — one line for the caller:
the site texture from the edited terrain and masks (terrain_tex.bake: road, paving, pad — with their own tiles; layer shares,
moisture and masks for the ground shader — relief_w.npz), the surface recipe and the biome's small items (surface.json). It does not touch the editing logic.
.venv/bin/python -m components.site.refresh <folder with relief.npz and location.json>   → line __RESULT__
import refresh; refresh.run(folder)"""
import core.paths         # the code is here, the core — core/<part> through kernel
import json, os, sys, time

import yaml

from components.site import terrain_clutter as TC
from components.site import terrain_tex as TT


FOREST = {"temperate": (0.85, [0.040, 0.068, 0.028]), "jungle": (0.95, [0.030, 0.062, 0.024])}   # tree line (a fraction of the belt mountains' height), canopy tone


def surface_recipe(folder, biome, seed=7):
    """The surface and small-item recipe of a site (as in terrain.scene): the biome's scan set, layer thresholds, small items — into surface.json."""
    Sv = yaml.safe_load(open(os.path.join(core.paths.ROOT, "library", "terrain", "surfaces.yaml")))
    sets = {k: {**v["default"], **(v.get("biomes", {}).get(biome) or {})} for k, v in Sv["sets"].items()}
    S = {"recipe": sets[Sv.get("use", "scan")], "sets": sets, "seed": int(seed), "biome": biome,
         "masks": {**Sv.get("masks", {}).get("default", {}), **(Sv.get("masks", {}).get(biome) or {})}, "clutter": TC.resolve(biome)}
    lp = os.path.join(folder, "location.json"); L = json.load(open(lp)) if os.path.exists(lp) else {}
    fk = (L.get("far") or {}).get("kind")
    if fk in ("real", "mosaic") and L.get("far_top") and biome in FOREST:   # mountains in a forest biome without snow: the height bands of the far shader —
        k_, canopy = FOREST[biome]; S["masks"].setdefault("treeline", round(k_ * float(L["far_top"]), 1))   # forest almost to the tops, rock
        S["masks"].setdefault("canopy", canopy)                       # on the steeps (core/surface/bl_terrain_vista: the tree line without snow)
    json.dump(S, open(os.path.join(folder, "surface.json"), "w"), ensure_ascii=False)
    return S


def run(folder):
    t0 = time.time(); folder = os.path.abspath(folder)
    L = json.load(open(os.path.join(folder, "location.json")))
    biome = L.get("biome") or TT.GROUND_BIOME.get(L.get("ground", "grass"), "temperate")
    tx = TT.bake(folder)                                            # relief_tex.png + relief_tex.json + relief_w.npz from the edited terrain
    surface_recipe(folder, biome, L.get("seed", 7))
    return {"folder": folder, "biome": biome, "tex": tx if isinstance(tx, (str, dict)) else str(tx), "s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    if len(sys.argv) < 2: raise SystemExit(__doc__)
    print("__RESULT__" + json.dumps(run(sys.argv[1]), ensure_ascii=False, default=str))
