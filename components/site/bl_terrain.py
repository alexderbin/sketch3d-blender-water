"""Far terrain around the pad: recipes for A.N.T. in metres (lengths × size, heights × height).
The Blender extension A.N.T. Landscape (bl_ext.blender_org.antlandscape) is used as is. The terrain is laid as a ring: over the pad (the box of the
ground* parts) a hole, from its edge the height rises smoothly and never drops below the scene ground — so the far-ground ring (bl_scene) does not conflict.
blender -b --factory-startup -P core/blender/run.py -- components/site/bl_terrain.py -- <bench/scenes/X> <hills|canyon|cliff|mesa|riverbed|map> [size=600] [height=80]
        [res=3] [ramp=0.15] [seed=7] [map=<height map.png>]   (map — brightness 0..1 → 0..height, the top of the image is north)
The part ridge_<n> of the kind "ridge" (class hill) is appended to bench/scenes/X/scene.obj; then make_frame / frames.py."""
import json, math, os, sys, time
import addon_utils, bpy
import numpy as np
from core.blender import bl_scene as B
from core.model import scene_info as SI

a = sys.argv[sys.argv.index("--") + 1:]
SD, RECIPE = os.path.abspath(a[0]), a[1]                            # a recipe or "map" (then map=<png> is required)
raw = dict(kv.split("=", 1) for kv in a[2:]); HMAP = raw.pop("map", None)                  # map=<png> — a height map instead of a recipe
opt = {k: float(v) for k, v in raw.items()}
SIZE, HEIGHT, RES, RAMP, SEED = opt.get("size", 600), opt.get("height", 80), opt.get("res", 3), opt.get("ramp", 0.15), int(opt.get("seed", 7))
RECIPES = {
    "hills": {},
    "canyon": dict(noise_offset_y=-0.25, noise_size_y=1.25, noise_size=1.5, noise_type="marble_noise", distortion=2.0, hard_noise="1", noise_depth=12,
                   marble_shape="4", height=0.6, fx_mix_mode="8", fx_type="20", fx_depth=3, fx_frequency=1.65, fx_size=1.5, fx_loc_x=3, fx_loc_y=2, fx_height=0.25,
                   fx_offset=0.05, edge_falloff="2", edge_level=0.15, maximum=0.5, minimum=-0.2, strata_type="2", strata=9),
    "cliff": dict(noise_offset_y=-0.88, noise_size_x=2, noise_size_y=2, noise_type="marble_noise", basis_type="VORONOI_F2F1", distortion=0.5, noise_depth=7,
                  marble_shape="6", height=1.8, height_offset=-0.15, fx_height=0.5, edge_falloff="0", falloff_x=25, falloff_y=25, maximum=1.25, minimum=0, strata=11),
    "mesa": dict(noise_size=0.75, noise_type="shattered_hterrain", basis_type="VORONOI_F1", vl_basis_type="VORONOI_F2F1", distortion=1.15, hard_noise="1",
                 amplitude=0.4, gain=4, height_offset=0.2, fx_frequency=1.5, fx_height=0.5, edge_falloff="3", falloff_x=3, falloff_y=3, maximum=0.25, minimum=0,
                 strata=2.25, strata_type="2"),
    "riverbed": dict(noise_type="marble_noise", marble_bias="2", marble_shape="7", height=0.2, fx_frequency=1.5, fx_height=0.5, edge_falloff="0", falloff_x=40,
                     falloff_y=40, maximum=0.5, minimum=0, strata=1.25, strata_type="1"),
}
kind, lo, hi, gz = SI.parts_of(SD)
G = [p for p in lo if p.startswith("ground")]
x0, y0 = min(lo[p][0] for p in G), min(lo[p][1] for p in G); x1, y1 = max(hi[p][0] for p in G), max(hi[p][1] for p in G)
cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
bpy.ops.wm.read_factory_settings(use_empty=True); addon_utils.enable("bl_ext.blender_org.antlandscape", default_set=False)
kw = dict(land_material="", water_material="", texture_block="", at_cursor=False, smooth_mesh=False, tri_face=False, sphere_mesh=False, water_plane=False,
          remove_double=False, show_main_settings=False, show_noise_settings=False, show_displace_settings=False, auto_refresh=False,
          noise_offset_x=11.0, noise_offset_y=-23.0, noise_offset_z=5.0, refresh=True)
kw.update(RECIPES.get(RECIPE, {}))
sxy, sz = SIZE / 2.0, HEIGHT / 2.0                                   # their _scale_absolute_params: (x + y) / 4 and z / 2
for k, d in (("noise_size", 1.0), ("fx_size", 1.0)): kw[k] = kw.get(k, d) * sxy
for k, d in (("height", 0.5), ("height_offset", 0.0), ("edge_level", 0.0), ("maximum", 1.0), ("minimum", -1.0)): kw[k] = kw.get(k, d) * sz
n = min(2000, math.ceil(SIZE / RES))
kw.update(random_seed=SEED, mesh_size_x=SIZE, mesh_size_y=SIZE, subdivision_x=n, subdivision_y=n, ant_terrain_name="ridge", vert_group="")
t0 = time.time()
if HMAP:                                                            # height map: brightness 0..1 → 0..height, a size×size grid
    im = bpy.data.images.load(os.path.abspath(HMAP)); w, h_ = im.size
    px = np.array(im.pixels[:], np.float32).reshape(h_, w, im.channels)[..., :3].mean(-1)       # rows run bottom to top, like the y axis
    iy = np.linspace(0, h_ - 1, n + 1).astype(int); ix = np.linspace(0, w - 1, n + 1).astype(int)
    Z = px[np.ix_(iy, ix)] * HEIGHT; g = np.linspace(-SIZE / 2, SIZE / 2, n + 1); X, Y = np.meshgrid(g, g)
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], -1)
    F = [(i * (n + 1) + j, i * (n + 1) + j + 1, (i + 1) * (n + 1) + j + 1, (i + 1) * (n + 1) + j) for i in range(n) for j in range(n)]
    RECIPE = "map:" + os.path.basename(HMAP)
else:
    bpy.ops.mesh.landscape_add(**kw); ob = bpy.context.active_object; me = ob.data
    V = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", V); V = V.reshape(-1, 3)
    F = [tuple(p.vertices) for p in me.polygons]
# the ring: distance from the pad's box, a smooth rise over RAMP·SIZE, never below the scene ground
V[:, 0] += cx; V[:, 1] += cy
dx = np.maximum(np.maximum(x0 - V[:, 0], V[:, 0] - x1), 0); dy = np.maximum(np.maximum(y0 - V[:, 1], V[:, 1] - y1), 0)
d = np.hypot(dx, dy); inside = (V[:, 0] > x0) & (V[:, 0] < x1) & (V[:, 1] > y0) & (V[:, 1] < y1)
t = np.clip(d / max(RAMP * SIZE, 1e-6), 0, 1); w = t * t * (3 - 2 * t)
h = np.maximum(V[:, 2] - V[:, 2].min(), 0) * w                       # from the bottom of the recipe: all of the terrain is above the ground
V[:, 2] = gz + h
keep = [f for f in F if not any(inside[i] for i in f) and max(h[i] for i in f) > 0.05]
used = sorted({i for f in keep for i in f}); remap = {i: k for k, i in enumerate(used)}
pid = B.append_parts(SD, [("ridge", "ridge", V[used], [[remap[i] for i in f] for f in keep])])[0]
print("__RESULT__" + json.dumps({"id": pid, "recipe": RECIPE, "faces": len(keep), "verts": len(used), "height_max_m": round(float(h.max()), 1),
                                 "hole": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)], "s": round(time.time() - t0, 2)}, ensure_ascii=False))
