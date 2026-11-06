"""Boulders of exact size for a pipeline scene: A.N.T. in sphere mode with the species' parameters, a grid step in metres,
fitting to w × d × h and seating by the bottom on the scene ground (a ray from above, figures do not count). Size error 0 mm.
blender -b --factory-startup -P core/blender/run.py -- components/plants/bl_boulder.py -- <bench/scenes/X> <rock|crystalline|abstract|flatstones|ridged|cauliflower>
    <x,y[;x,y…]> [size=w,d,h] [n=1] [spread=0] [seed=1] [step=0.08]
size — metres (by default 1.5 m × the species profile); n and spread — a scatter around each point (size ±20 %).
The boulder is appended to scene.obj as the part boulder_<n> of the kind "rock" (the rock class in the maps, the stone colour in the frame colour)."""
import json, math, os, random, sys
import addon_utils
import bpy
import numpy as np
from mathutils import Matrix, Vector
from core.blender import bl_scene as B

BASE = dict(noise_size=1.0, noise_type="hetero_terrain", basis_type="PERLIN_ORIGINAL", vl_basis_type="PERLIN_ORIGINAL",
            distortion=1.0, noise_depth=8, amplitude=0.5, frequency=2.0, dimension=1.0, lacunarity=2.0, offset=1.0, gain=1.0,
            height=0.5, height_invert=False, height_offset=0.0, maximum=1.0, minimum=-1.0, strata=5.0, strata_type="0")
TYPES = {   # size profile + noise parameters (the midpoints of the ranges per kind)
    "rock": ((1.2, 0.9, 0.65), dict(noise_size=2.0, noise_type="slick_rock", basis_type="VORONOI_F1", vl_basis_type="BLENDER",
                                    noise_depth=6, gain=3.0, height=2.5, maximum=3.0, minimum=-1.0, strata=15.0)),
    "crystalline": ((0.8, 0.75, 1.2), dict(noise_size=1.0, noise_type="turbulence_vector", basis_type="VORONOI_F4",
                                           vl_basis_type="BLENDER", noise_depth=1, distortion=1.0, height=1.0, maximum=2.0, minimum=-1.0)),
    "abstract": ((1.35, 0.75, 0.9), dict(noise_type="planet_noise", basis_type="VORONOI_F1", vl_basis_type="VORONOI_F4",
                                         distortion=1.0, noise_depth=1, amplitude=0.41, frequency=1.8, height=1.0, height_invert=True,
                                         maximum=1.0, minimum=-1.0, strata=5.0, strata_type="3")),
    "flatstones": ((1.4, 1.0, 0.3), dict(noise_size=0.5, noise_type="slick_rock", basis_type="VORONOI_CRACKLE", vl_basis_type="BLENDER",
                                         distortion=1.2, noise_depth=1, frequency=1.5, offset=1.1, gain=2.0, height=0.05,
                                         maximum=1.0, minimum=0.0)),
    "ridged": ((1.2, 0.85, 0.65), dict(noise_type="ridged_multi_fractal", basis_type="BLENDER", vl_basis_type="VORONOI_F1",
                                       noise_depth=8, frequency=1.75, dimension=0.92, lacunarity=2.3, offset=0.9, gain=2.1,
                                       height=0.3, maximum=0.5, minimum=-1.0, strata=11.0)),
    "cauliflower": ((1.0, 1.0, 0.75), dict(noise_size=0.5, noise_type="hybrid_multi_fractal", basis_type="VORONOI_F1",
                                           vl_basis_type="BLENDER", noise_depth=6, lacunarity=5.0, gain=2.0, height=0.25,
                                           height_invert=True, maximum=1.0, minimum=-1.0)),
}
def dims_of(kind, size=None):
    """The default boulder size for a kind (1.5 m × the kind's profile) or the given w, d, h."""
    return list(size) if size else [1.5 * p for p in TYPES[kind][0]]


def boulder(dims, seed, kind="rock", step=0.08):
    """Their boulder(): the sphere resolution by the step in metres, the A.N.T. operator, exactly one new mesh, fitting and seating (_fit_and_ground).
    The mesh is one object in the current scene (the caller takes it and removes it)."""
    params, STEP = TYPES[kind][1], step
    addon_utils.enable("bl_ext.blender_org.antlandscape", default_set=False)
    horizontal = max(dims[0], dims[1]); lon = max(8, math.ceil(math.pi * horizontal / STEP)); lat = max(4, math.ceil(math.pi * (horizontal + dims[2]) / 4 / STEP))
    before = set(bpy.data.objects)
    r = bpy.ops.mesh.landscape_add("EXEC_DEFAULT", ant_terrain_name="boulder", at_cursor=False, smooth_mesh=False, tri_face=False, sphere_mesh=True,
                                   subdivision_x=lon, subdivision_y=lat, mesh_size=2.0, random_seed=seed, water_plane=False, remove_double=True,
                                   refresh=True, **dict(BASE, **params))
    created = set(bpy.data.objects) - before
    if "FINISHED" not in r or len(created) != 1:
        for o in created: bpy.data.objects.remove(o)
        raise RuntimeError(f"landscape_add: {r}, created {len(created)}")
    o = created.pop(); cur = np.array(o.dimensions, float)
    o.data.transform(Matrix.Diagonal((*(np.array(dims) / cur), 1.0)))
    co = np.array([v.co[:] for v in o.data.vertices]); lo, hi = co.min(0), co.max(0); c = (lo + hi) / 2
    o.data.transform(Matrix.Translation((-c[0], -c[1], -lo[2]))); o.data.update(); return o


def main():
    a = sys.argv[sys.argv.index("--") + 1:]; SD, KIND = os.path.abspath(a[0]), a[1]
    POS = [[float(x) for x in p.split(",")] for p in a[2].split(";") if p]; opt = dict(kv.split("=", 1) for kv in a[3:])
    rng = random.Random(int(opt.get("seed", 1))); N, SPREAD, STEP = int(opt.get("n", 1)), float(opt.get("spread", 0)), float(opt.get("step", 0.08))
    dims0 = dims_of(KIND, [float(x) for x in opt["size"].split(",")] if "size" in opt else None)
    bpy.ops.wm.read_factory_settings(use_empty=True); addon_utils.enable("bl_ext.blender_org.antlandscape", default_set=False); sc = bpy.context.scene
    bpy.ops.wm.obj_import(filepath=os.path.join(SD, "scene.obj"), forward_axis="Y", up_axis="Z")   # the ground under the points: a ray from above, without figures
    for o in list(sc.objects):
        if o.name.startswith("figure_"): bpy.data.objects.remove(o)
    dg = bpy.context.evaluated_depsgraph_get(); top = max((o.matrix_world @ Vector(c))[2] for o in sc.objects for c in o.bound_box) + 10
    spots = []
    for p in POS:
        for j in range(N):
            q = (p[0] + (rng.uniform(-SPREAD, SPREAD) if j else 0), p[1] + (rng.uniform(-SPREAD, SPREAD) if j else 0))
            hit, loc, *_ = sc.ray_cast(dg, (q[0], q[1], top), (0, 0, -1)); spots.append((q[0], q[1], loc[2] if hit else 0.0))
    for o in list(sc.objects): bpy.data.objects.remove(o)
    parts, stats = [], []
    for i, (x, y, z) in enumerate(spots):
        k_ = rng.uniform(0.8, 1.2) if N > 1 else 1.0; dims = [d * k_ for d in dims0]
        o = boulder(dims, int(opt.get("seed", 1)) + i, KIND, STEP)
        R = Matrix.Rotation(rng.uniform(0, 2 * math.pi), 4, "Z"); M = Matrix.Translation((x, y, z - 0.03 * dims[2])) @ R   # slightly into the ground: no gap
        V = np.array([(M @ v.co)[:] for v in o.data.vertices]); F = [list(p.vertices) for p in o.data.polygons]
        parts.append(("boulder", "rock", V, F)); stats.append({"at": [round(x, 2), round(y, 2), round(z, 2)], "dims": [round(d, 2) for d in dims], "verts": len(V)})
        bpy.data.objects.remove(o)
    for st, pid in zip(stats, B.append_parts(SD, parts)): st["id"] = pid
    print("__RESULT__" + json.dumps({"kind": KIND, "boulders": stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
