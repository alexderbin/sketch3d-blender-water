"""Site texture from ground tiles: the biome is a set of tiles per layer (base with patches, ground, stone, cliff, scree, snow, bank,
bed, road, platform), the layer weights come from the site (slope, height, water, masks, concavity), the tiles are placed by
world coordinates (on steep ground "by height": a wall does not stretch into stripes), in the distance — reduced copies of the
tile and its mean color. The grid and UV are the same as in components/site/bl_relief.py (ground.axes): the texture lives in
the space of grid-line indices. .venv (PIL).
.venv/bin/python -m components.site.terrain_tex <site_folder> [px=8] [biome=…]  → relief_tex.png + relief_tex.json; the __RESULT__ line"""
import core.paths         # the code is here, the core — core/<part> through kernel
import json, math, os, sys, time
import numpy as np
ROOT = core.paths.ROOT
from core.surface.ground import Ground, axes, snowline_of  # noqa: F401  (snow line — a read of the relief layer, the core)
from components.site.terrain import fbm, smoothstep, strata_layers

TILES = os.path.join(ROOT, "library/textures/ground_tiles")
# layer → (tiles, period in m); base — up to three tiles in patches; steep — layers on steep ground (by height)
BIOMES = {
    "temperate": {"base": (["dirt/grass_mixed_01", "dirt/grass_01", "dirt/grass_tall_02"], 6), "dirt": (["dirt/dirt_01", "dirt/grassdirt_01"], 5),
                  "rock": (["stone/jungle_stone_02"], 7), "cliff": (["stone/typhon_viney_cliff_01", "stone/jungle_stone_01"], 9),
                  "talus": (["dirt/dirt_rock_rubble_01"], 5), "snow": (["snow/snow_rock_mix_01"], 8), "bank": (["dirt/mud_wet_02"], 5),
                  "bed": (["dirt/clay_01"], 5), "road": (["dirt/dirt_light_01"], 5), "pad": (["sand/d_ground_cobblestone_02"], 4)},
    "alpine": {"base": (["snow/snow_02", "snow/snow_01", "snow/snow_03"], 8), "dirt": (["snow/slushy_snow_01"], 6), "rock": (["snow/snow_rock_mix_01"], 7),
               "cliff": (["snow/high_ice_cliff_03", "snow/snow_stone_01"], 9), "talus": (["stone/snow_stone_rubble_01"], 5),
               "snow": (["snow/snow_smooth_01"], 8), "bank": (["snow/ice_floor_02"], 6), "bed": (["snow/glacier_01"], 8),
               "road": (["snow/slush_snow_01"], 5), "pad": (["stone/snow_urban_stone_01"], 4)},
    "desert": {"base": (["sand/d_smooth_01", "sand/d_rough_01", "sand/d_ripples_01"], 7), "dirt": (["sand/d_ground_cracked_01"], 6),
               "rock": (["red_stone/desert_ground_flatrock", "red_stone/d_red_smooth_01"], 7), "cliff": (["red_stone/d_cliff_rough_01", "red_stone/desert_sandstone"], 10),
               "talus": (["red_stone/desert_strewn_rocks_02"], 5), "snow": (["red_stone/d_red_smooth_02"], 8), "bank": (["dirt/clay_02"], 5),
               "bed": (["dirt/clay_01"], 5), "road": (["sand/d_ground_stonemixed_01"], 5), "pad": (["sand/d_ground_cobblestone_02"], 4)},
    "volcanic": {"base": (["volcanic/rock_basalt_04", "volcanic/rock_basalt_09", "volcanic/basalt_crack"], 7), "dirt": (["volcanic/burned_roots_01"], 6),
                 "rock": (["volcanic/rock_basalt_06"], 7), "cliff": (["volcanic/rock_basalt_14", "volcanic/rock_basalt_17"], 9),
                 "talus": (["volcanic/rock_basalt_11"], 5), "snow": (["volcanic/sulfur_05"], 8), "bank": (["volcanic/rock_basalt_16"], 5),
                 "bed": (["red_stone/lava_01"], 6), "road": (["volcanic/building_debris_01"], 5), "pad": (["stone/cement_01"], 4)},
    "jungle": {"base": (["dirt/jungle_fern_grass", "dirt/grass_jungle_01", "dirt/jungle_foilage_01"], 6), "dirt": (["roots/jungle_roots_top_01"], 6),
               "rock": (["dirt/jungle_rocky_moss_02"], 7), "cliff": (["stone/jungle_volcanic_cliff_01", "stone/typhon_viney_cliff_01"], 9),
               "talus": (["dirt/jungle_rocky_01"], 5), "snow": (["stone/jungle_stone_04"], 8), "bank": (["dirt/mud_wet_02"], 5),
               "bed": (["dirt/clay_01"], 5), "road": (["stone/jungle_cobbles_01"], 4), "pad": (["stone/jungle_ancient_stone_floor"], 4)},
    "wasteland": {"base": (["dirt/mud_field_01", "dirt/mission_e_dirt_01", "dirt/clay_02"], 6), "dirt": (["dirt/u_rubble_02"], 5),
                  "rock": (["dirt/u_rubble_01"], 6), "cliff": (["red_stone/dark_rocky_01", "stone/u_concrete_09"], 8),
                  "talus": (["dirt/u_garbage_debris_01"], 5), "snow": (["dirt/u_rubble_03"], 8), "bank": (["dirt/mud_wet_02"], 5),
                  "bed": (["dirt/clay_01"], 5), "road": (["stone/cement_01"], 5), "pad": (["stone/u_concrete_05"], 4)},
    "alien": {"base": (["tyrannoformation/pleuge01_ground", "tyrannoformation/tyr_coral_01", "tyrannoformation/tyr_sandy_01"], 7),
              "dirt": (["tyrannoformation/tyr_roots_grey"], 6), "rock": (["stone/tyr_coral_stone_01"], 7), "cliff": (["stone/tyra_stone_03", "stone/tyra_stone_rubble_01"], 9),
              "talus": (["stone/tyra_stone_rubble_01"], 5), "snow": (["tyrannoformation/tyr_coral_02"], 8), "bank": (["dirt/mud_wet_02"], 5),
              "bed": (["dirt/clay_01"], 5), "road": (["stone/tyra_stone_01"], 5), "pad": (["stone/tyr_coral_stone_02"], 4)},
}
# layer brightness (mean sRGB): the game tiles are dark — each tile goes to its own target (a common
# multiplier overexposed the light ones: road, snow)
LUM = {"base": 0.42, "dirt": 0.40, "rock": 0.48, "cliff": 0.44, "talus": 0.45, "snow": 0.84, "bank": 0.40, "bed": 0.30, "road": 0.50, "pad": 0.45, "pave": 0.50}
LUM_BIOME = {"alpine": {"base": 0.80, "dirt": 0.70, "rock": 0.58, "cliff": 0.46, "talus": 0.55, "road": 0.66, "pad": 0.55, "bank": 0.62, "bed": 0.50},
             "desert": {"base": 0.62, "dirt": 0.55, "rock": 0.55, "cliff": 0.52, "talus": 0.52, "snow": 0.6, "road": 0.56},
             "volcanic": {"base": 0.34, "dirt": 0.30, "rock": 0.36, "cliff": 0.33, "talus": 0.34, "snow": 0.55, "bed": 0.45},
             "jungle": {"base": 0.38, "dirt": 0.34, "rock": 0.40, "cliff": 0.38}, "alien": {"base": 0.45}}
GROUND_BIOME = {"grass": "temperate", "soil": "temperate", "snow": "alpine", "ice": "alpine", "sand": "desert", "ash": "volcanic", "mud": "wasteland",
                "rock": "alpine"}
STEEP = {"rock", "cliff"}
_TILE = {}


# tile tone without a file — by a word in its name: mean tones of the tile set by group (the tile set is not distributed)
_PLAIN = (("snow ice glacier", (0.58, 0.66, 0.70)), ("sand beach desert dune", (0.63, 0.50, 0.35)),
          ("water bed river shore", (0.24, 0.24, 0.21)), ("ash lava volcan burnt", (0.23, 0.14, 0.16)),
          ("pave cobble brick floor street", (0.43, 0.40, 0.33)), ("rock stone cliff gravel scree rubble", (0.31, 0.30, 0.28)),
          ("mud dirt soil earth path road", (0.32, 0.28, 0.22)), ("grass moss leaf forest meadow", (0.22, 0.23, 0.13)))


def _plain_tile(name, n=512):
    """No tile file: a flat tone chosen by a word in the file name, with soft blotches — the bake, the layer
    weights (relief_w.npz) and the far maps are written as with tiles; the ground is just simpler."""
    from PIL import Image
    b = name.rsplit("/", 1)[-1].lower()
    c = next((np.array(col, np.float32) for words, col in _PLAIN if any(w in b for w in words.split())),
             np.array((0.44, 0.39, 0.32), np.float32))
    rng = np.random.default_rng(sum(ord(ch) * 31 ** i for i, ch in enumerate(name)) % 2 ** 32)
    nz = np.asarray(Image.fromarray((rng.random((32, 32)) * 255).astype(np.uint8)).resize((n, n), Image.BICUBIC), np.float32) / 255
    return np.clip(c[None, None, :] * (0.85 + 0.3 * nz[..., None]), 0, 1).astype(np.float32)


def tile(name):
    """Tile → mip levels (512, 128, 32 px and the mean color), linear sRGB 0..1."""
    if name not in _TILE:
        from PIL import Image                                       # PIL is only in .venv: the biome tables are read by Blender too
        p = os.path.join(TILES, name + "_dif.jpg")
        im = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255 if os.path.exists(p) else _plain_tile(name)
        L = [im]
        for s in (128, 32):
            L.append(np.asarray(Image.fromarray((im * 255).astype(np.uint8)).resize((s, s), Image.LANCZOS), np.float32) / 255)
        L.append(im.reshape(-1, 3).mean(0)[None, None, :]); _TILE[name] = L
    return _TILE[name]


def sample(name, U, V, period, foot, tone=None, pull=0.0, lum=None):
    """Tile color at world coordinates (U, V, m) with period period; foot — texel size (m): the larger it is, the smaller the mip;
    tone, pull — the tile's mean color is shifted toward tone by the fraction pull (tiles of one biome — in one color scheme); lum — mean brightness."""
    L = tile(name); r = foot / (period / 512.0); out = np.empty(U.shape + (3,), np.float32)
    lev = np.where(r < 3, 0, np.where(r < 12, 1, np.where(r < 48, 2, 3)))
    for k in range(4):
        m = lev == k
        if not m.any(): continue
        T = L[k]; n = T.shape[0]
        if n == 1: out[m] = T[0, 0]; continue
        i = (np.floor(U[m] / period * n).astype(np.int64)) % n; j = (np.floor(V[m] / period * n).astype(np.int64)) % n
        out[m] = T[n - 1 - j, i]
    if tone is not None and pull: out += (np.asarray(tone, np.float32) - L[3][0, 0]) * pull
    if lum is not None: out *= lum / max(float(L[3][0, 0].mean()) + (float(np.mean(tone) - L[3][0, 0].mean()) * pull if tone is not None and pull else 0.0), 1e-3)
    return out


def _hash2(vx, vy, k):
    """Random numbers [0, 1) at integer grid vertices (deterministic, stateless)."""
    h = (vx.astype(np.int64) * 73856093) ^ (vy.astype(np.int64) * 19349663) ^ (k * 83492791)
    h = (h ^ (h >> 13)) * 1274126177; return ((h ^ (h >> 16)) & 0xFFFFFF).astype(np.float64) / float(0x1000000)


def sample_hex(name, U, V, period, foot, tone=None, pull=0.0, lum=None, cell=0.55, rot=True, gamma=6.0, r=0.75, beta=0.6):
    """Non-repeating tile — hex tiling (Mikkelsen, "Practical Real-Time Hex-Tiling", JCGT 11(2), 2022, listings 1–2): a triangular
    grid in tile fractions (cell cell), each vertex has its own tile shift and rotation; three samples, the weight is the barycentric
    coordinate raised to the power gamma with a brightness correction (β), an S-curve (r > 0.5 — sharper borders, less "mush").
    Cost — three samples instead of one."""
    q = np.stack([U / period, V / period], -1) / cell                 # in grid cells
    sx = q[..., 0] - 0.57735027 * q[..., 1]; sy = 1.15470054 * q[..., 1]   # into the skewed triangular grid
    bx, by = np.floor(sx), np.floor(sy); fx, fy = sx - bx, sy - by; z = 1.0 - fx - fy; sg = (z < 0).astype(np.float64); s2 = 2 * sg - 1
    W = [-z * s2, sg - fy * s2, sg - fx * s2]; Vt = [(bx + sg, by + sg), (bx + sg, by + 1 - sg), (bx + 1 - sg, by + sg)]
    C, Wt = [], []
    for (vx, vy), w in zip(Vt, W):
        cx_, cy_ = (vx + 0.5 * vy) * cell, (0.8660254 * vy) * cell     # cell center — in tile fractions
        a = _hash2(vx, vy, 1) * 2 * np.pi if rot else 0.0; ox, oy = _hash2(vx, vy, 2), _hash2(vx, vy, 3)
        dx, dy = U / period - cx_, V / period - cy_; ca, sa = np.cos(a), np.sin(a)
        uu = (ca * dx - sa * dy + cx_ + ox) * period; vv = (sa * dx + ca * dy + cy_ + oy) * period
        c = sample(name, uu, vv, period, foot, tone, pull, lum); C.append(c)
        Wt.append(np.power(np.clip(w, 0, 1), gamma) * ((1 - beta) + beta * (c @ np.array([0.299, 0.587, 0.114], np.float32))))
    Wt = np.stack(Wt, -1); Wt /= Wt.sum(-1, keepdims=True) + 1e-9
    k = np.log(0.5) / np.log(1 - r) if r != 0.5 else 1.0                 # Perlin S-curve: at r = 0.5 — unchanged
    Wt = np.where(Wt < 0.5, 0.5 * np.power(2 * Wt, k), 1 - 0.5 * np.power(2 - 2 * Wt, k)); Wt /= Wt.sum(-1, keepdims=True) + 1e-9
    return sum(C[i] * Wt[..., i:i + 1].astype(np.float32) for i in range(3))


VAR = None
MSK = None


def masks_of(biome):
    """Biome layer thresholds (library/terrain/surfaces.yaml → masks): common + own."""
    global MSK
    if MSK is None:
        import yaml
        MSK = yaml.safe_load(open(os.path.join(ROOT, "library", "terrain", "surfaces.yaml"))).get("masks", {})
    return {**MSK.get("default", {}), **(MSK.get(biome) or {})}


def far_analysis(G):
    """Analysis of the belt grid (4.7 m) by the same core as the near one (core/surface/terrain_analysis): catchment, shapes, brows,
    feet. The belt gets relief data just like the near plot, so the dilution (variation) and the far maps (far_maps) do not break off
    at the edge of the near grid. → {area, forms, brow, foot, at(k, X, Y)}."""
    from core.surface import terrain_analysis as TA
    H, r = G.Hf, float(G.fres); F = TA.fill(H); _, brow, foot = TA.cliffs(H, r)
    D = {"area": TA.accumulate(F, TA.receivers(F, r), r), "forms": TA.geomorphons(H, r, L=40.0), "brow": brow.astype(np.float32), "foot": foot.astype(np.float32)}

    def at(k, X, Y):                                                # shapes — by the nearest cell (a category), the rest — bilinear
        if k == "forms":
            i = np.clip(np.round((Y - G.fy0) / r).astype(int), 0, H.shape[0] - 1); j = np.clip(np.round((X - G.fx0) / r).astype(int), 0, H.shape[1] - 1)
            return D[k][i, j]
        return G._bil(np.asarray(D[k], np.float64), G.fx0, G.fy0, r, X, Y)
    D["at"] = at
    return D


def variation(G, biome, X, Y, Z, S, gx, gy, W, seed=7, fa=None):
    """Site color dilution by relief data: a low-frequency color map + a detail tile (the color map is the layer's tint),
    colored by data maps (slope, curvature, flow, deposits) instead of bare noise.
    Moisture M 0…1 (0.5 — average): + near water, along watercourses (catchment), in hollows, on shaded (north) slopes;
    − on ridges and brows, on sunny (south) and steep slopes, higher up the site; + large patches of 60–250 m (soils, grass cover) —
    so that even a flat field is not one color. Color V (an RGB multiplier around 1): dry ↔ wet by biome (library/terrain/surfaces.yaml
    → variation), ridges and brows lighter (exposure), feet — scree, tint patches, cavities darker. Returns M (ny, nx), V (ny, nx, 3),
    exposure (ridges, brows) and feet (ny, nx) — the shader layer masks."""
    import cv2, yaml
    global VAR
    if VAR is None:
        VAR = yaml.safe_load(open(os.path.join(ROOT, "library", "terrain", "surfaces.yaml"))).get("variation", {})
    P = {**VAR.get("default", {}), **(VAR.get(biome) or {})}
    blur = lambda A, s_: cv2.GaussianBlur(A.astype(np.float32), (0, 0), s_)
    inb = G._in(G.near_bounds(), X, Y).astype(np.float32)
    out_ = (lambda k, d: fa["at"](k, X, Y) if fa is not None and k in fa else d)   # outside the near grid — the belt analysis (far_analysis)
    am = lambda k, d=0.0: np.where(inb > 0, G.amap(k, X, Y), out_(k, d)).astype(np.float32) if k in G.amaps else np.full(X.shape, d, np.float32)
    area = am("area", 1.0)
    if fa is not None:                                              # catchment — from the belt mountains too: the near analysis sees runoff only within its own
        area = np.maximum(area, fa["at"]("area", X, Y)).astype(np.float32)   # 360 m — the plain at the edge would be drier than the belt (a moisture seam)
    wet = np.exp(-am("dwater", 999.0) / 18.0) * inb + np.clip(W, 0, 1)                       # near water
    drain = blur(np.clip((np.log10(np.maximum(area, 1.0)) - 2.6) / 1.8, 0, 1), 3.0)   # along watercourses (thin: experience
    fm = am("forms", 5); concave = np.isin(fm, (6, 7, 8, 9)).astype(np.float32); convex = np.isin(fm, (1, 2, 3, 4)).astype(np.float32)   # advises not to overdo it)
    cell = np.minimum.outer(np.gradient(Y[:, 0]), np.gradient(X[0, :])).astype(np.float32)   # grid step, m (sparser toward the belt)
    Zs = blur(Z, 5.0); lap = cv2.Laplacian(Zs, cv2.CV_32F) / (cell * cell)              # curvature in meters — no banding at the junction of steps
    lap = np.clip(lap / (3 * float(np.std(lap[inb > 0] if inb.any() else lap)) + 1e-6), -1, 1)   # form at 3–10 m
    cav, ridge = blur(np.maximum(concave, np.clip(lap, 0, 1)), 2.0), blur(np.maximum(convex * 0.8, np.clip(-lap, 0, 1)), 2.0)
    brow, foot = blur(am("brow").clip(0, 1), 2.0), blur(am("foot").clip(0, 1), 3.0)
    gn = np.hypot(gx, gy) + 1e-6; north = (-gy / gn) * smoothstep(4.0, 25.0, S)               # a slope facing north (+y) is in shade
    zq = np.percentile(Z[inb > 0] if inb.any() else Z, [5, 95]); hr = np.clip((Z - zq[0]) / max(zq[1] - zq[0], 1.0), 0, 1)
    patch = fbm(X / 170.0 + 5.1, Y / 170.0 - 2.7, 3, seed + 61) * 0.6 + fbm(X / 61.0 - 3.3, Y / 61.0 + 8.8, 2, seed + 62) * 0.4
    M = 0.5 + 0.30 * np.clip(wet, 0, 1) + 0.22 * drain + 0.16 * cav - 0.18 * ridge - 0.12 * brow + 0.12 * north \
        - 0.12 * smoothstep(18.0, 40.0, S) - 0.08 * (hr - 0.5) + float(P.get("patch_m", 0.22)) * patch + float(P.get("moist_bias", 0.0))
    M = np.clip(M, 0, 1).astype(np.float32)
    dry_, wet_ = np.array(P.get("dry", [1.12, 1.07, 0.9]), np.float32), np.array(P.get("wet", [0.82, 0.92, 0.78]), np.float32)
    t = np.clip((M - 0.5) * 2.6, -1, 1)[..., None]; V = np.where(t > 0, 1 + t * (wet_ - 1), 1 - t * (dry_ - 1))   # moisture to color — with a margin
    # (moisture rarely leaves 0.3…0.7: without a margin the field is almost uniform)
    pa, pb = (np.array(c, np.float32) for c in P.get("patches", [[1.05, 1.0, 0.92], [0.94, 1.0, 1.03]]))
    q = smoothstep(-0.35, 0.35, fbm(X / 113.0 - 7.7, Y / 113.0 + 1.9, 2, seed + 63))[..., None]; V = V * (pa * q + pb * (1 - q))   # tint patches
    V = V * (1 + 0.08 * ridge + 0.06 * brow)[..., None]                                       # exposure: ridges and brows are lighter
    V = V * (1 + (np.array(P.get("foot", [1.05, 1.03, 1.0]), np.float32) - 1) * foot[..., None])   # feet — scree
    V = V * (1 + 0.10 * fbm(X / 140.0 + 1.7, Y / 140.0 - 4.2, 2, 21) + 0.06 * fbm(X / 41.0, Y / 41.0, 2, 22))[..., None]   # brightness in patches
    expo = np.clip(ridge * 0.7 + brow, 0, 1).astype(np.float32)                               # exposure: ridges and brows (rock sticking out)
    return M, np.clip(V, 0.55, 1.5).astype(np.float32), expo, np.clip(foot, 0, 1).astype(np.float32)


MASKS = {"rock_from": 27, "rock_band": 11, "cliff_from": 43, "dirt_dry": 0.36, "talus_from": 16, "talus_to": 30, "jit": 4}
ORDER = ["dirt", "talus", "rock", "cliff", "snow", "bank", "bed", "pad", "pave", "road"]   # on top of the base — one by one (alpha)


def layer_weights(X, Y, S, M, expo, foot, gully, plan, K=None, Z=None, snowline=None, seed=7):
    """Layer fractions from the relief data — the same rules as in the ground shader (bl_terrain_surf: a layer comes from data,
    noise only roughens the edge): rock — from the slope rock_from (exposed ground — 10° earlier), cliff — from cliff_from or by the
    plan, ground — dry exposed spots and hollows, scree — feet and occasionally slopes talus_from…talus_to, plan — bank, bed,
    platform, paving, road. plan — {bank, bed, pad, pave, road, cliff}: 0…1 arrays. Returns (layer alphas, coverage fractions with base)."""
    K = {**MASKS, **(K or {})}; J = float(K["jit"]); R0, C0 = float(K["rock_from"]), float(K["cliff_from"])
    jit = lambda k: fbm(X / 9.0 + 3.1 * k, Y / 9.0 - 1.7 * k, 2, seed + 70 + k) * J
    z0 = np.zeros_like(S); P = {k: plan.get(k, z0) for k in ("bank", "bed", "pad", "pave", "road", "cliff")}
    w = {"rock": smoothstep(R0, R0 + float(K["rock_band"]), S + jit(4) + 10 * expo) * (1 - smoothstep(C0 + 1, C0 + 9, S + jit(5))),
         "cliff": np.maximum(smoothstep(C0, C0 + 9, S + jit(6)), P["cliff"]),
         "dirt": np.maximum(smoothstep(1 - float(K["dirt_dry"]), 1 - float(K["dirt_dry"]) + 0.18, (1 - M) + 0.012 * jit(1)), gully)
                 * (1 - smoothstep(R0 - 2, R0 + 6, S)),
         "talus": np.maximum(foot * smoothstep(6, 14, S), smoothstep(K["talus_from"], K["talus_from"] + 8, S + jit(2))
                             * smoothstep(0.45, 0.7, fbm(X / 13.0, Y / 13.0, 3, seed + 3) * 0.5 + 0.5)) * (1 - smoothstep(K["talus_to"], K["talus_to"] + 8, S)),
         "snow": smoothstep(snowline - 6, snowline + 6, Z + jit(7) * 2) * (1 - smoothstep(45, 60, S)) if snowline is not None and Z is not None else z0,
         "bank": P["bank"], "bed": P["bed"], "pad": P["pad"] * 0.6, "pave": P["pave"] * 0.9, "road": P["road"]}
    cov, rest = {}, np.ones_like(S)
    for lay in ORDER[::-1]: a_ = np.clip(w[lay], 0, 1); cov[lay] = a_ * rest; rest = rest * (1 - a_)
    cov["base"] = rest
    return w, cov


def bake(folder, px=8, biome=None, bright=None, out=None, hexs="base"):
    t0 = time.time(); G = Ground(folder); L = G.loc
    biome = biome or L.get("biome") or GROUND_BIOME.get(L.get("ground", "grass"), "temperate"); B = dict(BIOMES[biome])
    B.setdefault("pave", B["pad"]); B["pad"] = B["dirt"]               # platform — trampled earth, paved — the biome's cobblestones
    for lay, v in (L.get("tiles") or {}).items(): B[lay] = (v if isinstance(v, list) else [v], B.get(lay, B["pad"])[1])   # the preset's tiles on top of the biome
    xs, ys = axes(G); X, Y = np.meshgrid(xs, ys)                    # bl_relief grid nodes
    Z = G.height(X, Y); S = G.slope(X, Y); gx, gy = G.grad(X, Y)
    W = G.water(X, Y); lv = G._bil(np.nan_to_num(G.level, nan=-1e9), G.nx0, G.ny0, G.nres, X, Y)
    shore = ((lv > -1e8) & G._in(G.near_bounds(), X, Y) & (Z - lv < 1.2) & (W <= 0.02)).astype(np.float32)
    road, pad, cliffm = G.mask("road", X, Y), G.mask("pad", X, Y), G.mask("cliff", X, Y)
    lap = np.zeros_like(Z); lap[1:-1, 1:-1] = (Z[2:, 1:-1] + Z[:-2, 1:-1] + Z[1:-1, 2:] + Z[1:-1, :-2] - 4 * Z[1:-1, 1:-1])   # concavity (> 0 — a hollow)
    cell = np.minimum.outer(np.gradient(ys), np.gradient(xs))       # grid cell size, m
    curv = np.clip(lap / np.maximum(cell, 0.5), -2, 2)
    snowline = snowline_of(L, biome)
    nz = fbm(X / 37.0, Y / 37.0, 3, 5) * 0.5 + 0.5; nz2 = fbm(X / 13.0 + 7.1, Y / 13.0 - 2.3, 3, 6) * 0.5 + 0.5   # base and ground patches
    gully = np.clip(curv * 0.8, 0, 1) * (1 - smoothstep(30, 40, S))
    pave = G.mask("pave", X, Y)
    fa = far_analysis(G)                                            # the belt analysis — for the dilution and the far maps
    Mo, Vr, Ex, Ft = variation(G, biome, X, Y, Z, S, gx, gy, W, int(L.get("seed", 7)), fa)     # dilution by relief data
    plan = {"bank": shore, "bed": np.clip(W / 0.6, 0, 1), "road": road, "pad": np.clip(pad - pave, 0, 1), "pave": pave, "cliff": cliffm}
    w, cov = layer_weights(X, Y, S, Mo, Ex, Ft, gully, plan, masks_of(biome), Z, snowline, int(L.get("seed", 7)))   # the same rules,
    order = ORDER                                                   # as in the ground shader: the previz and the layer tones — in the same places
    raw = {f"a_{k}": np.clip(plan[k], 0, 1).astype(np.float32) for k in ("bank", "bed", "pad", "pave", "road", "cliff")}   # plan masks (the platform
    raw["a_gully"] = np.clip(gully, 0, 1)                           # and paving fractions are taken by the shader itself: ×0.6 and ×0.9)
    raw["a_moist"], raw["a_expo"], raw["a_foot"] = Mo, Ex, Ft
    wsave = {k: v.astype(np.float16) for k, v in {**cov, **raw}.items()}; wsave["var"] = Vr.astype(np.float16)
    ny, nx = Z.shape; H_, W_ = ny * px, nx * px
    U = np.interp((np.arange(W_) + 0.5) / px - 0.5, np.arange(nx), xs); V = np.interp((np.arange(H_) + 0.5) / px - 0.5, np.arange(ny), ys)
    fu = np.gradient(U); fv = np.gradient(V)                        # texel size, m
    tex = np.empty((H_, W_, 3), np.uint8); CH = 256
    base_t, base_p = B["base"]; tone = tile(base_t[0])[3][0, 0]; pull = float(L.get("tone_pull", 0.5))
    lum = lambda lay: {**LUM, **LUM_BIOME.get(biome, {})}[lay] * float(L.get("bright_k", 1.0))
    nzc = fbm(X / 23.0 + 3.3, Z / 9.0, 2, 9) * 0.5 + 0.5                # the second cliff tile — in patches by height (at the nodes)
    st_ = L.get("strata") or {}; seed_ = int(L.get("seed", 7)); E_, hard_, tint_ = strata_layers(seed_, float(st_.get("step", 2.6)), float(st_.get("var", 0.7)))
    zp = Z + float(st_.get("dip", 0.08)) * (X * 0.6 + Y * 0.8) + 1.2 * fbm(X / 35.0, Y / 35.0, 2, seed_ + 41)   # height "within the stack" — as for the rock
    for r0 in range(0, H_, CH):
        r1 = min(H_, r0 + CH); rows = slice(r0, r1)
        UU, VV = np.meshgrid(U, V[rows]); foot = np.maximum.outer(fv[rows], fu)
        vv = np.clip((np.arange(r0, r1) + 0.5) / px - 0.5, 0, ny - 1.0001); uu = np.clip((np.arange(W_) + 0.5) / px - 0.5, 0, nx - 1.0001)
        j0, i0 = vv.astype(int), uu.astype(int); fy, fx = (vv - j0)[:, None], (uu - i0)[None, :]
        up_rows = lambda A: A[j0][:, i0] * (1 - fy) * (1 - fx) + A[j0][:, i0 + 1] * (1 - fy) * fx + A[j0 + 1][:, i0] * fy * (1 - fx) + A[j0 + 1][:, i0 + 1] * fy * fx
        smp = sample_hex if hexs is True else sample                 # no repetition: hex tiling (True — all natural layers,
        n_ = up_rows(nz); C = (sample_hex if hexs else sample)(base_t[0], UU, VV, base_p, foot, lum=lum("base"))   # "base" — the main tile of the base)
        if len(base_t) > 1:
            a1 = (smoothstep(0.35, 0.75, n_) * 0.8)[..., None]; C = C * (1 - a1) + smp(base_t[1], UU + 31.7, VV - 11.3, base_p, foot, tone, pull, lum("base")) * a1
        if len(base_t) > 2:
            a2 = (smoothstep(0.7, 0.92, n_) * 0.6)[..., None]; C = C * (1 - a2) + smp(base_t[2], UU - 7.9, VV + 23.1, base_p, foot, tone, pull, lum("base")) * a2
        Zt, gxt, gyt = up_rows(Z), up_rows(gx), up_rows(gy); gn = np.hypot(gxt, gyt) + 1e-9
        Us = UU * np.abs(gyt / gn) + VV * np.abs(gxt / gn)          # on steep ground — along the slope and by height
        nc = up_rows(nzc)
        for lay in order:
            a = np.clip(up_rows(w[lay]), 0, 1); m = a > 0.01
            if not m.any(): continue
            names, per = B[lay]; am = a[m][:, None]; pl = pull * (0.5 if lay in STEEP else 1.0)
            if lay in STEEP:
                sh = (lambda *a_, **k_: sample_hex(*a_, rot=False, **k_)) if hexs is True else sample   # strata — without rotation
                kl = np.clip(np.searchsorted(E_, up_rows(zp)[m]) - 1, 0, len(tint_) - 1)   # stratum: its own tint and pattern shift — the bands
                Vz = Zt[m] + 23.7 * tint_[kl] + 11.3 * hard_[kl]                          # coincide with the ledges and are not periodic
                T_ = sh(names[0], Us[m], Vz, per, foot[m], tone, pl, lum(lay))
                if len(names) > 1:
                    b_ = smoothstep(0.4, 0.6, nc[m])[:, None]; T_ = T_ * (1 - b_) + sh(names[1], Us[m] + 13.1, Vz, per, foot[m], tone, pl, lum(lay)) * b_
                T_ = T_ * (1 + 0.16 * tint_[kl] + 0.06 * (hard_[kl] - 0.5))[:, None] * np.where(tint_[kl][:, None] > 0, np.array([1.0, 0.96, 0.92], np.float32), 1.0)   # warm and cool strata
            else:
                f_ = sample if lay in ("road", "pad", "pave") else smp
                T_ = f_(names[0], UU[m], VV[m], per, foot[m], tone, pl if lay not in ("road", "pad", "pave", "bed", "bank") else 0.0, lum(lay))
            C[m] = C[m] * (1 - am) + T_ * am
        occ = 1 - 0.18 * np.clip(up_rows(curv), 0, 1)[..., None]     # hollows are darker (like cavities)
        occ = occ * np.stack([up_rows(Vr[..., c]) for c in range(3)], -1)   # data-driven dilution (variation): moisture, exposure, patches
        tex[rows] = np.clip(C * occ * 255, 0, 255).astype(np.uint8)
    # brightness: the base mean goes to the target (the game tiles are dark; the previz is lighter, but darker
    # than the figures)
    target = float(bright) if bright else None; k = 1.0                    # brightness per layer (LUM); bright= — a common manual multiplier
    if target: k = target / max(tex.reshape(-1, 3).mean() / 255, 1e-3); tex = np.clip(tex.astype(np.float32) * k, 0, 255).astype(np.uint8)
    lin = np.power(tex.reshape(ny, px, nx, px, 3).mean((1, 3)) / 255.0, 2.2)   # layer tone (linear) — the mean color beneath it: the shader
    for lay, c_ in cov.items():                                      # fits the ground scans to the site palette (we do not lose the previz color)
        if float(c_.sum()) > 1.0: wsave[f"tone_{lay}"] = (lin * c_[..., None]).sum((0, 1)) / float(c_.sum())
    np.savez_compressed(os.path.join(folder, "relief_w.npz"), xs=xs, ys=ys, **wsave)
    out = out or os.path.join(folder, "relief_tex.png"); from PIL import Image
    Image.fromarray(tex[::-1]).save(out)                            # PNG runs top to bottom; the texture rows run from the south (as in Blender)
    meta = {"biome": biome, "px": px, "xs": [round(float(x), 4) for x in xs], "ys": [round(float(y), 4) for y in ys], "size": [W_, H_],
            "mean": [round(float(c), 4) for c in tex[::7, ::7].reshape(-1, 3).mean(0) / 255],
            "gain": round(float(k), 3), "s": round(time.time() - t0, 1)}
    json.dump(meta, open(os.path.splitext(out)[0] + ".json", "w"))
    nrm = normals(G, xs, ys, os.path.join(os.path.dirname(out), "relief_nrm.png"))
    fm = far_maps(G, xs, ys, os.path.join(os.path.dirname(out), "relief_far.png"), fa=fa)
    return {"tex": out, "biome": biome, "size": [W_, H_], "gain": round(float(k), 2), "nrm": nrm, "far_maps": fm, "s": round(time.time() - t0, 1)}


def _uv_grid(xs, ys, px):
    """Texel centers of the image in the bl_relief grid UV (u = i/(nx − 1)) → world x (by columns), y (by rows)."""
    nx, ny = len(xs), len(ys); W_, H_ = nx * px, ny * px
    return (np.interp((np.arange(W_) + 0.5) / W_ * (nx - 1), np.arange(nx), xs), np.interp((np.arange(H_) + 0.5) / H_ * (ny - 1), np.arange(ny), ys))


def normals(G, xs, ys, out, px=4):
    """Ground normal map in the bl_relief grid UV (u = i/(nx − 1)): object-space normals from the site heights (near grid 0.5 m, belt —
    4.7 m) — the far shader (core/surface/bl_terrain_vista) lays the light and the layer slopes by them on top of the coarse belt
    mesh (cells up to 25 m). px — texels per grid cell: in the belt 25 / 4 ≈ 6 m is the step of its grid. → summary."""
    t0 = time.time(); from PIL import Image
    gx, gy = _uv_grid(xs, ys, px); img = np.empty((len(gy), len(gx), 3), np.uint8)
    for r0 in range(0, len(gy), 256):
        X, Y = np.meshgrid(gx, gy[r0:r0 + 256]); n = G.normal(X, Y)
        img[r0:r0 + 256] = np.clip((n * 0.5 + 0.5) * 255 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(img[::-1]).save(out)                            # PNG rows run top to bottom, Blender UV bottom to top
    return {"path": out, "size": [len(gx), len(gy)], "s": round(time.time() - t0, 1)}


def far_maps(G, xs, ys, out, px=4, fa=None):
    """Belt data maps for the far shader (core/surface/bl_terrain_vista) — an analysis of its grid (4.7 m) by the same core as the near
    one (core/surface/terrain_analysis): R — ridges (convexity and the shapes "summit, ridge, shoulder, spur": rock sticks out there),
    G — thalwegs (catchment and hollows: scree, wet grass), B — feet (the bottoms of cliffs and scree trains: scree), A — height
    above the valley floor 0…1 (floor within a 600 m window → crest: like HAND, Rennó et al. 2008 — the shader drives the mountain
    belts by it: thinner soil, more frequent rock). Masks come from data: slope + curvature + flow instead of noise. In
    the bl_relief grid UV. → summary."""
    import cv2
    from scipy.ndimage import minimum_filter, maximum_filter
    from core.surface import terrain_analysis as TA
    t0 = time.time(); from PIL import Image
    H, r = G.Hf, float(G.fres); blur = lambda A, s_: cv2.GaussianBlur(np.asarray(A, np.float32), (0, 0), s_)
    fa = fa or far_analysis(G); A, C, foot = fa["area"], fa["forms"], fa["foot"]
    lap = cv2.Laplacian(blur(H, 1.5), cv2.CV_32F) / (r * r); conv = np.clip(-lap / (2.5 * float(np.std(lap)) + 1e-6), 0, 1)
    R = np.clip(0.6 * conv + 0.6 * np.isin(C, (1, 2, 3, 4)), 0, 1)
    Gd = np.clip(np.maximum((np.log10(np.maximum(A, 1.0)) - 2.3) / 1.6, 0.5 * np.isin(C, (6, 8, 9))), 0, 1)
    B = np.clip(np.maximum(blur(foot.astype(np.float32), 2.0) * 2.5, 0.6 * np.isin(C, (7,))), 0, 1)
    w_ = int(600 / r) | 1; lo, hi = blur(minimum_filter(H, size=w_), 20 / r), blur(maximum_filter(H, size=w_), 20 / r)
    Hr = np.clip((H - lo) / np.maximum(hi - lo, 30.0), 0, 1)          # height above the floor: 0 — valley floor, 1 — crest (a drop < 30 m — a plain)
    D = np.stack([blur(R, 1.0), blur(Gd, 0.8), blur(B, 1.0), blur(Hr, 1.0)], -1)
    gx, gy = _uv_grid(xs, ys, px); img = np.empty((len(gy), len(gx), 4), np.uint8)
    for r0 in range(0, len(gy), 256):
        X, Y = np.meshgrid(gx, gy[r0:r0 + 256])
        img[r0:r0 + 256] = np.stack([np.clip(G._bil(D[..., c], G.fx0, G.fy0, r, X, Y), 0, 1) * 255 + 0.5 for c in range(4)], -1).astype(np.uint8)
    Image.fromarray(img[::-1], "RGBA").save(out)                       # alpha is data (the shader loads it as "packed channels")
    return {"path": out, "size": [len(gx), len(gy)], "ridge": round(float((R > 0.5).mean()), 3), "gully": round(float((Gd > 0.5).mean()), 3),
            "foot": round(float((B > 0.5).mean()), 3), "high": round(float((Hr > 0.5).mean()), 3), "s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    a = sys.argv[1:]; opt = dict(kv.split("=", 1) for kv in a[1:])
    print("__RESULT__" + json.dumps(bake(os.path.abspath(a[0]), int(opt.get("px", 8)), opt.get("biome")), ensure_ascii=False))
