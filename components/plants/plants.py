"""A plant as one model: species (species.yaml) + age + state + seed → a specification for Blender (bl_plant.build).
The project's Python (YAML is here; Blender receives JSON). Organ images — leaves.write into library/plants/tex (cached).
.venv/bin/python -m components.plants.plants sheet <out.json> [species[:age[:state]] …] — plants for a species sheet
.venv/bin/python -m components.plants.plants list — species by biome
.venv/bin/python -m components.plants.plants meadow <biome> <out.json> [K] [season] — the biome's herb layer (specifications for Blender)"""
import core.paths
import copy, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); KNOW = os.path.join(HERE, "species.yaml")
TEX = os.path.join(core.paths.ROOT, "library", "plants", "tex")
_K = None


def knowledge():
    global _K
    if _K is None:
        import yaml
        _K = yaml.safe_load(open(KNOW))
    return _K


def species(sid):
    K = knowledge(); s = K["species"][sid]; base = copy.deepcopy(K["habits"].get(s["habit"], {}))
    out = {**base, **copy.deepcopy(s)}
    for k in ("roots", "woody"):
        if k in base and k in s: out[k] = {**base[k], **s[k]}
    return out


def resolve(sid, age="adult", state="healthy", seed=1, height=None, lean=None, push=None, exposure=None):
    """The specification of one plant. height — set the height (m), otherwise from the species range × age; lean — {deg, az, recover};
    push — the crown shift toward the light [dx, dy] (m); exposure — a multiplier of root exposure (slope, water, old age)."""
    K = knowledge(); S = species(sid); r = np.random.default_rng(seed)
    if state == "healthy" and S.get("state"): state = S["state"]          # the species is always in its own state (dead standing wood)
    A = K["ages"].get(age, {}); St = K["states"].get(state, {})
    H_ad = float(r.uniform(*S["height"])); H = float(height) if height else H_ad * A.get("h", 1.0)
    sp = {"id": f"{sid}_{age}_{state}_{seed}", "species": sid, "age": age, "state": state, "seed": int(seed), "habit": S["habit"],
          "height": round(H, 3), "width": round(H * S.get("width", 0.5), 2), "sink": S.get("sink", 0.03)}
    if S["habit"] in ("tree", "shrub", "palm"):
        sp.update(trunk=dict(S["trunk"]), levels=[dict(l) for l in S.get("levels", [])], woody=S.get("woody", {}), bark=dict(S.get("bark", {})),
                  crown_base=S.get("crown_base", 0.35) * (A.get("crown_base", 1.0) if "crown_base" in A else 1.0))
        if "dbh_h" in S: sp["bark_radius"] = round(0.5 * S["dbh_h"] * H_ad * A.get("dbh", 1.0), 4)
        bc = max(A.get("break_chance", 0), St.get("break_chance", 0))
        for lv in sp["levels"]:
            if bc: lv["break_chance"] = max(lv.get("break_chance", 0), bc)
        if age in ("seedling", "young") and sp["levels"]:              # young: the crown lower on the trunk, thinner and shorter branches
            sp["levels"][0]["start"] = round(sp["levels"][0].get("start", 0.2) * 0.6, 3)
        if St.get("trunk_cut"):                                       # broken trunk: shorter, a thick broken end, branches below the break
            sp["trunk"]["length"] *= St["trunk_cut"]; sp["trunk"]["end_radius"] = sp["trunk"].get("start_radius", 0.3) * 0.45
        if "bark" in St: sp["bark"].update(St["bark"]); sp["bark"]["name"] = sp["bark"].get("name", sid) + "_" + state
        from components.plants import bark_scans as BSC                                           # close-up bark is a Poly Haven scan, if downloaded (bark_scans.py fetch)
        if sid in BSC.BARK and state != "dead":
            P = BSC.paths(BSC.BARK[sid])
            if all(os.path.exists(f) for f in P.values()): sp["bark"]["scan"] = BSC.BARK[sid]   # the scan name (paths are from the project root)
        lf = S.get("leaves")
        if lf and St.get("leaves", 1) and A.get("leaves", 1):
            lf = dict(lf); lf["per_m"] = lf.get("per_m", 6) * St.get("leaves", 1) * A.get("leaves", 1)
            if "hue" in St: lf["hue"] = list(np.asarray(lf.get("hue", [1, 1, 1])) * np.asarray(St["hue"]))
            sp["leaves"] = lf
        if S.get("frond") and St.get("leaves", 1):
            sp["frond"] = dict(S["frond"], z=0.0)
        rs = dict(S.get("roots") or {"type": "none"})
        if rs.get("type") != "none" and A.get("roots", 1) > 0:
            rs["spread"] = round(rs.get("spread", 0.4) * 0.5 * S.get("width", 0.5) * H, 2)
            rs["exposure"] = float(np.clip(rs.get("exposure", 0.3) * A.get("exposure", 1.0) * (exposure or 1.0), 0, 0.95))
            if A.get("roots", 1) < 1: rs["n"] = max(3, int(rs.get("n", 8) * 0.7)); rs["exposure"] *= A["roots"]
            sp["roots"] = rs
        sp["lean"] = lean if lean is not None else {"deg": float(r.uniform(0, 4)), "az": float(r.uniform(0, 360)), "recover": 0.6}
        if push is not None: sp["push"] = list(map(float, push))
        sp["state"] = state
    else:
        for k in ("stems", "stem_angle", "bend", "stem_w", "stem_color", "leaf", "flower", "frond", "blades", "width", "spread", "clump_r",
                  "dry", "color", "dry_color", "heads", "head_color", "clump_lean"):
            if k in S: sp[k] = copy.deepcopy(S[k])
        if S["habit"] == "grass": sp["width"] = S.get("width", 0.006)
        if state == "dead" or state == "winter":
            sp["dry"] = 1.0 if S["habit"] == "grass" else sp.get("dry", 0)
        sp["layout_w"] = round(H * 0.8, 2)
    oo = organ_optics(S, state)
    if oo: sp["optics"] = oo
    return sp


def organ_optics(S, state):
    """Optics of a species' organs: a PROSPECT-D leaf pigment set (species.yaml optics, otherwise the habit's optics_default; autumn —
    the species' own set, grasses as dead stalks and in winter — dry_grass) → Rd (leaf body), T (translucency), R (total reflectance), linear rgb, + cuticle
    gloss (rough) and the underside (under). The set none means no optics (the image and tint as is: "alien")."""
    K = knowledge(); o = dict((K.get("optics_default") or {}).get(S["habit"], {})); o.update(S.get("optics") or {})
    if o.get("set", "none") == "none": return None
    from components.plants import leaf_optics as LO
    name = o.get("autumn", "autumn_yellow") if state == "autumn" else o["set"]
    q = LO.optics(name, **(o.get("over") or {}))
    g = LO.optics("dry_grass" if state in ("dead", "winter") else o.get("blade", "grass"))
    return {"set": name, "Rd": q["Rd"], "T": q["T"], "R": q["R"], "rough": float(o.get("gloss", 0.42)), "under": float(o.get("under", 1.3)),
            "grass": {"Rd": g["Rd"], "T": g["T"]}}


def meadow_specs(biome, K=4, seed=1, kinds=("grass", "herb", "flower", "fern"), season=None, dry_tiers=False):
    """A biome's herb layer: {species: [K variant specifications]} + the species' abundance (abund) and moisture window (moist) — for bl_meadow. Winter — grasses
    are dry, no flowers or forbs; autumn — grasses half dry, no flowers. dry_tiers — variants of grasses by dryness (fresh,
    half, dry): the cover-based meadow (components/grass) takes the variant by the point's dryness."""
    out = {}; winter = season == "winter"; autumn = season == "autumn"
    for sid, s in knowledge()["species"].items():
        if s["habit"] in kinds and biome in s.get("biomes", []):
            if (winter or autumn) and s["habit"] in ("flower", "herb"): continue
            if winter and s["habit"] == "fern": continue                    # bracken turns brown and lodges toward winter
            ab = float((s.get("abundance") or {}).get(biome, 1.0)); st = "dead" if winter and s["habit"] == "grass" else "healthy"
            out[sid] = [dict(resolve(sid, "adult", st, seed=seed + 17 * k), abund=ab, moist=s.get("moist", [0, 1])) for k in range(K)]
            if autumn and s["habit"] == "grass":
                for x in out[sid]: x["dry"] = max(x.get("dry", 0.1), 0.5)
            if dry_tiers and s["habit"] == "grass":
                for k, x in enumerate(out[sid]): x["dry"] = max(x.get("dry", 0.1), (0.0, 0.4, 0.8)[k % 3]); x["dry_tier"] = k % 3
    return out


def textures(kinds=None):
    """Organ images in library/plants/tex (drawn once)."""
    from components.plants import leaves
    return leaves.write(TEX, kinds)


def by_biome():
    K = knowledge(); out = {}
    for sid, s in K["species"].items():
        for b in s.get("biomes", []): out.setdefault(b, []).append(sid)
    return out


def main():
    a = sys.argv[1:]
    if a and a[0] == "list":
        print(json.dumps(by_biome(), ensure_ascii=False, indent=1)); return
    if a and a[0] == "meadow":                                       # the biome's herb layer for Blender: plants.py meadow <biome> <out.json> [K] [season]
        textures(); M = meadow_specs(a[1], K=int(a[3]) if len(a) > 3 else 3, season=a[4] if len(a) > 4 else None)
        json.dump(M, open(a[2], "w"), ensure_ascii=False); print("__RESULT__" + json.dumps({"species_n": len(M)}, ensure_ascii=False)); return
    if a and a[0] == "sheet":
        out = a[1]; ids = a[2:] or list(knowledge()["species"]); textures(); P = []
        for i, x in enumerate(ids):
            sid, age, st = (x.split(":") + ["adult", "healthy"])[:3]
            P.append(resolve(sid, age or "adult", st or "healthy", seed=3 + i))
        json.dump({"plants": P}, open(out, "w"), ensure_ascii=False)
        print("__RESULT__" + json.dumps({"plants": len(P), "out": out}, ensure_ascii=False)); return
    print(__doc__)


if __name__ == "__main__":
    main()
