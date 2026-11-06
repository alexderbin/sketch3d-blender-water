"""Site dressing — one mechanism for any ground: the site data maps (relief.npz: an_* — terrain.data_maps) → layer density by the
ecotope rules (library/terrain/dress.yaml) → a pattern with the asset's footprint (cell = footprint, one point per cell with its
own threshold: placed if the density is above the threshold — deterministic and stable under local edits, as in Horizon Zero
Dawn) → large before small, small avoids large → children around parents → planting (z on the ground, sink, tilt by the normal).
Output — dress.json in the site folder: [{layer, kind, sub, x, y, z, s, rot, sink, tilt}]; built by bl_relief.dress_from.
.venv/bin/python -m components.site.terrain_dress <site_folder> [seed] → dress.json + a summary (__RESULT__)."""
import core.paths         # the code is here, the core — core/<part> through kernel
import json, math, os, sys

import numpy as np

RULES = os.path.join(core.paths.ROOT, "library", "terrain", "dress.yaml")
FORMS = ["flat", "peak", "ridge", "shoulder", "spur", "slope", "hollow", "footslope", "valley", "pit"]


def ecotope(biome):
    """Biome layers: the common ones (base) with biome overrides by id."""
    import yaml
    R = yaml.safe_load(open(RULES)); L = {l["id"]: dict(l) for l in R["base"]["layers"]}
    for l in ((R.get("biomes") or {}).get(biome) or {}).get("layers") or []:
        L[l["id"]] = {**L.get(l["id"], {}), **l}
    return [L[k] for k in L]


def curve(v, pts):
    p = np.asarray(pts, float); return np.interp(v, p[:, 0], p[:, 1])


def density(G, X, Y, lay, ctx):
    """Layer density at points (X, Y): base × multipliers from the site maps."""
    from components.site.terrain import fbm
    d = np.full(X.shape, float(lay.get("base", 0.1)))
    if "slope" in lay: d *= curve(G.steep(X, Y), lay["slope"])
    am = lambda k: G.amap(k, X, Y)
    if "brow" in lay: d *= np.where(ctx["brow"](X, Y), lay["brow"], 1.0)
    if "foot_of_cliff" in lay: d *= np.where(ctx["foot"](X, Y), lay["foot_of_cliff"], 1.0)
    if lay.get("foot_only"): d *= ctx["foot"](X, Y)                 # only at the foot of a cliff (scree)
    if lay.get("brow_only"): d *= ctx["brow"](X, Y)                 # only at the brow
    if "cliff" in lay: d *= np.where(G.steep(X, Y) > 40, lay["cliff"], 1.0)
    if "forms" in lay:
        f = am("forms"); m = np.ones(X.shape)
        for nm, w in lay["forms"].items(): m = np.where(f == FORMS.index(nm), w, m)
        d *= m
    if "flow" in lay: d *= curve(np.log10(np.maximum(am("area"), 1.0)), lay["flow"])
    if "dwater" in lay: d *= curve(am("dwater").astype(float), lay["dwater"])
    if "droad" in lay: d *= curve(am("droad").astype(float), lay["droad"])
    if "flat" in lay: d *= np.where(am("flat") > 0, lay["flat"], 1.0)
    if lay.get("noise"):
        n = lay["noise"]; d *= np.clip(1 + n.get("amp", 0.5) * fbm(X / n["scale"], Y / n["scale"], 3, ctx["seed"] + 17 * ctx["k"]), 0, None)
    if lay.get("below_snow") and ctx.get("snow") is not None: d *= (G.height(X, Y) < ctx["snow"] - 4)
    d *= G.wet(X, Y) < 0.01                                          # not in water
    d *= (G.mask("road", X, Y) < 0.2) & (G.mask("pad", X, Y) < 0.2)  # not on the roadbed or a platform
    if lay.get("clear"):
        for a in ctx["public"]: d *= np.hypot(X - a[0], Y - a[1]) > lay["clear"]
    for (ax, ay, r) in ctx["avoid"]: d *= np.hypot(X - ax, Y - ay) > r
    S_ = ctx.get("settle")
    if S_ and "use" in G.amaps:                                     # settlement land: forest only where it is not plowed, stones are cleared from the fields
        u_ = am("use").astype(float) / 255.0
        if lay["kind"] == "tree":
            kp = S_["keep"]; keep = np.maximum(curve(G.steep(X, Y), kp["slope"]), curve(am("dwater").astype(float), kp["dwater"]))
            d *= 1 - u_ * (1 - np.maximum(float(S_["open"]), keep))
        elif lay["id"] in S_["rocks"]["ids"]: d *= 1 - u_ * (1 - float(S_["rocks"]["k"])) * (G.steep(X, Y) < float(S_["rocks"]["flat_below"]))
    if S_ and "works" in G.amaps and lay["kind"] in S_["works"]:  # on man-made features
        d *= np.where(am("works") > 0, float(S_["works"][lay["kind"]]), 1.0)
    return d


class BeltView:
    """The mountain belt as seen by dressing: the data maps are an analysis of the belt grid (components/site/terrain_tex.far_analysis:
    catchment, shapes, brows, feet), the rest comes from the site's Ground. Belt dressing uses the same mechanism and the same
    rules as the near plot."""

    def __init__(self, G, fa=None):
        from components.site import terrain_tex as TT
        self.G, self.fa = G, fa or TT.far_analysis(G)
        self.amaps = {k: None for k in ("area", "forms", "brow", "foot")}   # the keys exist — for the checks `"x" in G.amaps` in density

    def __getattr__(self, k):
        return getattr(self.G, k)

    def amap(self, k, X, Y):
        if k in self.amaps: return self.fa["at"](k, X, Y)
        return np.full(np.shape(X), 255.0)                          # belt water and roads are far away (the Ground masks take care of them)


def plan(G, seed=11, biome=None, avoid=(), limit=6000, layers=None, box=None, hole=None):
    """Placement by the ecotope of the site's biome. avoid — [(x, y, r)] (fighters, buildings). layers — own layers instead of the
    ecotope (belt rocks); box — where to place (x0, y0, x1, y1; the near plot by default), hole — where not to place (the near
    plot, for the belt). Returns (places, a summary per layer)."""
    L = G.loc; biome = biome or L.get("biome") or "temperate"; layers = layers if layers is not None else ecotope(biome)
    if L.get("forest") is False: layers = [l for l in layers if l["kind"] != "tree"]
    if L.get("stones") is False: layers = [l for l in layers if l["kind"] != "rock"]   # marine clay, marsh silt — no stones
    nb = box or G.near_bounds(); m = 6.0; rng = np.random.default_rng(seed)
    out_hole = (lambda X, Y: np.ones(np.shape(X), bool)) if hole is None else         (lambda X, Y: ~((X > hole[0]) & (X < hole[2]) & (Y > hole[1]) & (Y < hole[3])))
    pub = [L["anchors"][a][:2] for a in L.get("public", []) if a in L["anchors"]]
    brow = lambda X, Y: G.amap("brow", X, Y) > 0; foot = lambda X, Y: G.amap("foot", X, Y) > 0
    if isinstance(G, BeltView):                                     # belt: brows and feet come from the analysis of its grid (4.7 m cell — no dilation)
        brow = lambda X, Y: G.amap("brow", X, Y) > 0.5; foot = lambda X, Y: G.amap("foot", X, Y) > 0.3
    elif "brow" in G.amaps:                                         # "near the brow/foot" means within ~3 m: dilate the masks
        from core.surface import terrain_analysis as TA
        bw, ft = TA.dilate(G.amaps["brow"].astype(float), 3) > 0, TA.dilate(G.amaps["foot"].astype(float), 3) > 0
        st = G.astep; idx = lambda X, Y, A: A[np.clip(np.round((Y - G.ny0) / st).astype(int), 0, A.shape[0] - 1), np.clip(np.round((X - G.nx0) / st).astype(int), 0, A.shape[1] - 1)]
        brow = lambda X, Y: idx(X, Y, bw); foot = lambda X, Y: idx(X, Y, ft)
    snow = L.get("snowline"); ctx = {"seed": seed, "public": pub, "avoid": list(avoid), "brow": brow, "foot": foot,
                                     "snow": float(snow) if snow not in (None, "none") else None}
    import yaml
    ctx["settle"] = yaml.safe_load(open(RULES)).get("settle") if (L.get("settle") or "works" in G.amaps) else None
    out, placed, stats = [], [], {}                                 # placed: (x, y, half-footprint) — for collisions
    cell = 4.0; grid = {}

    def free(x, y, r):
        i, j = int(x // cell), int(y // cell); k = int(math.ceil((r + 4.0) / cell))
        for a in range(i - k, i + k + 1):
            for b in range(j - k, j + k + 1):
                for (px, py, pr) in grid.get((a, b), ()):
                    if (px - x) ** 2 + (py - y) ** 2 < (pr + r) ** 2: return False
        return True

    def put(lay, x, y, k_):
        r = lay["foot"] * 0.5
        if not free(x, y, r): return False
        grid.setdefault((int(x // cell), int(y // cell)), []).append((x, y, r)); sz = lay.get("size", [1, 1]); s = float(rng.uniform(*sz))
        if lay["kind"] == "tree": s = s / (0.5 * (sz[0] + sz[1]))     # trees: a fraction of the species height (bl_relief H0), stones — diameter, m
        sub = None
        if lay.get("mix"):
            ks = list(lay["mix"]); w = np.array([lay["mix"][q] for q in ks], float); sub = ks[int(rng.choice(len(ks), p=w / w.sum()))]
        out.append({"layer": lay["id"], "kind": lay["kind"], "sub": sub, "x": round(float(x), 2), "y": round(float(y), 2),
                    "z": round(float(G.top(x, y)), 2), "s": round(s, 2), "rot": round(float(rng.uniform(0, 360)), 1),
                    "sink": lay.get("sink", 0.0), "tilt": lay.get("tilt", 0.0), **({"facet": True} if lay.get("facet") else {})}); return True
    for k_, lay in enumerate(layers):
        ctx["k"] = k_; n0 = len(out); f = float(lay["foot"])
        if lay.get("base", 0) > 0 and not lay.get("children"):      # pattern: cell = footprint, a point in the cell with its own threshold
            r2 = np.random.default_rng(seed * 1000 + k_)
            xs, ys = np.arange(nb[0] + m, nb[2] - m, f), np.arange(nb[1] + m, nb[3] - m, f); X, Y = np.meshgrid(xs, ys)
            X = X + r2.uniform(0, f, X.shape); Y = Y + r2.uniform(0, f, Y.shape); thr = r2.uniform(0, 1, X.shape)
            D = density(G, X, Y, lay, ctx) * out_hole(X, Y); hit = D > thr
            for x, y in zip(X[hit], Y[hit]):
                if len(out) >= limit: break
                put(lay, x, y, k_)
        ch = lay.get("children")
        if ch:                                                      # children around the parents of the layer "of"
            par = [o for o in out if o["layer"] == ch["of"]]; r2 = np.random.default_rng(seed * 2000 + k_)
            pf = next((l["foot"] for l in layers if l["id"] == ch["of"]), 4.0)
            for o in par:
                for _ in range(int(r2.integers(ch["n"][0], ch["n"][1] + 1))):
                    a = r2.uniform(0, 2 * math.pi); rr = r2.uniform(*ch["r"]) * pf
                    x, y = o["x"] + rr * math.cos(a), o["y"] + rr * math.sin(a)
                    if not (nb[0] + m < x < nb[2] - m and nb[1] + m < y < nb[3] - m): continue
                    if density(G, np.array([x]), np.array([y]), dict(lay, base=1.0, noise=None) if "noise" in lay else dict(lay, base=1.0), ctx)[0] > 0.2:
                        put(lay, x, y, k_)
        stats[lay["id"]] = len(out) - n0
    return out, stats


def main():
    folder = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    from core.surface.ground import Ground
    G = Ground(folder); P, st = plan(G, seed)
    json.dump({"seed": seed, "biome": G.loc.get("biome"), "places": P}, open(os.path.join(folder, "dress.json"), "w"), ensure_ascii=False)
    print("__RESULT__" + json.dumps({"total": len(P), "by_layer": st}, ensure_ascii=False))


if __name__ == "__main__":
    main()
