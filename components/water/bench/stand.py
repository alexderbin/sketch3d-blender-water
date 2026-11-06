"""Water showcase (Blender): a scene by name, a frame or a clip — for viewing by eye.
  .venv/bin/python -m core.services.guard heavy --what water -- blender -b --factory-startup --python-exit-code 1 \
    -P core/blender/run.py -- components/water/bench/stand.py -- out/dev/water/sea scene=sea [t=6] [frames=0] [fps=12] [samples=48]
Scenes: sea — a sea at sunset (HDR belfast sky, low sun, 7 m/s wind with swell: a sun glitter path, backlit crests, whitecaps).
Knobs: wind= (m/s), swell= (Hs, m), t= (moment, s), frames= (0 — a single frame), fps=, samples=, width= (from the height at 16:9), grid= (vertices along x)."""
import json
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Euler, Vector

from components.water import waves as W
from components.water import optics
from components.water.bl_water import Water, near_for

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SKY = os.path.join(ROOT, "library", "sky")


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = os.path.abspath(a[0]) if a and "=" not in a[0] else os.path.join(ROOT, "out", "dev", "water", "stand")
    kv = dict(x.split("=", 1) for x in a if "=" in x)
    return out, kv


def scene_base(sc, width, samples):
    sc.render.engine = "BLENDER_EEVEE"
    sc.render.resolution_x, sc.render.resolution_y = width, int(round(width * 9 / 16))
    sc.render.resolution_percentage = 100
    e = sc.eevee; e.taa_render_samples = samples; e.use_raytracing = True; e.ray_tracing_method = "SCREEN"
    o = e.ray_tracing_options; o.resolution_scale = "1"; o.trace_max_roughness = 0.6; o.screen_trace_quality = 0.5
    e.use_shadows = True; e.use_fast_gi = True
    sc.view_settings.view_transform = "AgX"; sc.view_settings.look = "None"; sc.view_settings.exposure = -1.0
    sc.render.image_settings.file_format = "PNG"


def world_hdr(sc, file, rot_deg=0.0, strength=1.0, sun_threshold=10.0, cap_mirror=None):
    w = bpy.data.worlds.new("sky"); sc.world = w
    nt = w.node_tree; nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping"); mp.inputs["Rotation"].default_value = (0, 0, math.radians(rot_deg))
    env = nt.nodes.new("ShaderNodeTexEnvironment"); env.image = bpy.data.images.load(os.path.join(SKY, file))
    bg = nt.nodes.new("ShaderNodeBackground"); bg.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputWorld")
    sp = nt.nodes.new("ShaderNodeSeparateXYZ"); ab = nt.nodes.new("ShaderNodeMath"); ab.operation = "ABSOLUTE"   # the lower half — a mirror of the sky:
    cb = nt.nodes.new("ShaderNodeCombineXYZ")                           # rough water near the horizon catches the sky, not a black underside
    nt.links.new(tc.outputs["Generated"], sp.inputs[0]); nt.links.new(sp.outputs[2], ab.inputs[0])
    nt.links.new(sp.outputs[0], cb.inputs[0]); nt.links.new(sp.outputs[1], cb.inputs[1]); nt.links.new(ab.outputs[0], cb.inputs[2])
    nt.links.new(cb.outputs[0], mp.inputs["Vector"]); nt.links.new(mp.outputs[0], env.inputs[0])
    col = env.outputs[0]
    if cap_mirror is None and sun_threshold < 1e4: cap_mirror = 0.3 * sun_threshold   # by default — a cutoff (without it the sun
    if cap_mirror is not None:                                                         # of the lake, pond and river showcases — at the horizon)   # lower mirror — no second sun: Eevee extracts the sun from
        lt = nt.nodes.new("ShaderNodeMath"); lt.operation = "LESS_THAN"; lt.inputs[1].default_value = 0.0   # everything bright, and two suns
        nt.links.new(sp.outputs[2], lt.inputs[0])                       # (±height) merge into one at the horizon
        mn = nt.nodes.new("ShaderNodeVectorMath"); mn.operation = "MINIMUM"; mn.inputs[1].default_value = (cap_mirror,) * 3
        nt.links.new(col, mn.inputs[0])
        mx = nt.nodes.new("ShaderNodeMix"); mx.data_type = "RGBA"; nt.links.new(lt.outputs[0], mx.inputs["Factor"])
        nt.links.new(col, mx.inputs[6]); nt.links.new(mn.outputs[0], mx.inputs[7]); col = mx.outputs[2]
    nt.links.new(col, bg.inputs[0]); nt.links.new(bg.outputs[0], out.inputs[0])
    w.sun_threshold = sun_threshold; w.use_sun_shadow = True; w.probe_resolution = "2048"
    return w


def sun_vec(az, el):
    a, e = math.radians(az), math.radians(el)
    return Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))


def camera(sc, pos, az, pitch, lens=35.0, clip=30000.0):
    cd = bpy.data.cameras.new("cam"); cd.lens = lens; cd.sensor_width = 36.0; cd.clip_start = 0.1; cd.clip_end = clip
    cam = bpy.data.objects.new("cam", cd); sc.collection.objects.link(cam); sc.camera = cam
    cam.location = pos
    cam.rotation_euler = Euler((math.radians(90 + pitch), 0.0, math.radians(az - 90)), "XYZ")   # view along the azimuth az (from +x)
    bpy.context.view_layer.update()
    return cam


def sun_lamp(sc, az, el, rgb, strength, angle=0.53):
    ld = bpy.data.lights.new("sun", "SUN"); ld.energy = strength; ld.color = rgb; ld.angle = math.radians(angle)
    ob = bpy.data.objects.new("sun", ld); sc.collection.objects.link(ob)
    ob.rotation_euler = (Vector((0, 0, 1)).rotation_difference(sun_vec(az, el))).to_euler()
    return ob


def add_caustics(obj, wat, level=0.0, strength=0.9):
    """Water caustics on the showcase bed: the underwater illumination × (1 + strength·(C − 1)·depth attenuation); the map comes from
    the water frame (Water.caustic), the bed coordinates are world meters on the tile of the fine band."""
    m = obj.data.materials[0]; nt = m.node_tree; L_ = nt.links.new
    bs = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"); src = bs.inputs["Base Color"].links[0].from_socket
    im, Lc = wat.caustic
    tc = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping"); mp.inputs["Scale"].default_value = (1 / Lc, 1 / Lc, 1.0)
    L_(tc.outputs["Object"], mp.inputs[0])
    ti = nt.nodes.new("ShaderNodeTexImage"); ti.image = im; ti.interpolation = "Linear"; ti.extension = "REPEAT"; L_(mp.outputs[0], ti.inputs[0])
    sz = nt.nodes.new("ShaderNodeSeparateXYZ"); L_(tc.outputs["Object"], sz.inputs[0])
    dep = nt.nodes.new("ShaderNodeMath"); dep.operation = "SUBTRACT"; dep.inputs[0].default_value = level; L_(sz.outputs[2], dep.inputs[1])
    att = nt.nodes.new("ShaderNodeMath"); att.operation = "MULTIPLY"; att.inputs[1].default_value = -0.9; L_(dep.outputs[0], att.inputs[0])
    ex = nt.nodes.new("ShaderNodeMath"); ex.operation = "EXPONENT"; L_(att.outputs[0], ex.inputs[0])
    wet = nt.nodes.new("ShaderNodeMapRange"); wet.inputs["From Min"].default_value = 0.0; wet.inputs["From Max"].default_value = 0.03
    L_(dep.outputs[0], wet.inputs["Value"])
    fac = nt.nodes.new("ShaderNodeMath"); fac.operation = "MULTIPLY"; L_(ex.outputs[0], fac.inputs[0]); L_(wet.outputs[0], fac.inputs[1])
    sep = nt.nodes.new("ShaderNodeSeparateColor"); L_(ti.outputs["Color"], sep.inputs[0])
    cm1 = nt.nodes.new("ShaderNodeMath"); cm1.operation = "SUBTRACT"; L_(sep.outputs[0], cm1.inputs[0]); cm1.inputs[1].default_value = 1.0
    k = nt.nodes.new("ShaderNodeMath"); k.operation = "MULTIPLY"; L_(cm1.outputs[0], k.inputs[0]); L_(fac.outputs[0], k.inputs[1])
    k2 = nt.nodes.new("ShaderNodeMath"); k2.operation = "MULTIPLY_ADD"; L_(k.outputs[0], k2.inputs[0]); k2.inputs[1].default_value = strength; k2.inputs[2].default_value = 1.0
    mul = nt.nodes.new("ShaderNodeMixRGB"); mul.blend_type = "MULTIPLY"; mul.inputs[0].default_value = 1.0
    L_(src, mul.inputs[1]); cc = nt.nodes.new("ShaderNodeCombineColor"); [L_(k2.outputs[0], cc.inputs[i]) for i in range(3)]
    L_(cc.outputs[0], mul.inputs[2]); L_(mul.outputs[0], bs.inputs["Base Color"])


SKIES = {   # library/sky/skies.yaml; lamp — the luminary as a lamp (a sun in haze with no disk: color, strength), otherwise Eevee extracts the disk itself
    "qwantani_sunset": dict(file="qwantani_sunset_puresky_4k.hdr", sun=(324.0, 6.0)),
    "industrial_sunset": dict(file="industrial_sunset_02_puresky_4k.hdr", sun=(324.1, 2.2)),
    "evening_road": dict(file="evening_road_01_puresky_4k.hdr", sun=(325.2, 6.0), lamp=((1.0, 0.62, 0.32), 4.0)),
    "belfast": dict(file="belfast_sunset_puresky_4k.hdr", sun=(322.3, 2.1), lamp=((1.0, 0.45, 0.2), 4.0)),
    "kloofendal_day": dict(file="kloofendal_48d_partly_cloudy_puresky_4k.hdr", sun=(325.8, 47.9)),
    "misty_morning": dict(file="kloofendal_misty_morning_puresky_4k.hdr", sun=(317.1, 43.6)),
    "alps_field": dict(file="alps_field_4k.hdr", sun=(323.8, 42.1)),           # mountains on the horizon — the far shore of the lake
    "chinese_garden": dict(file="chinese_garden_4k.hdr", sun=(323.7, 30.4)),
    "overcast": dict(file="kloofendal_overcast_puresky_4k.hdr", sun=(0.0, 60.0)),
}
TEX = os.path.join(ROOT, "library", "textures", "terrain")


def ground_mesh(sc, f, x0, x1, y0, y1, step, name="shore", tex="Gravel043", tile=1.6, dark=0.55, tint=(1.0, 1.0, 1.0), under=None, wet_h=0.06,
                spec=0.5, rough_min=0.0, grid=None):
    """Bed and shore — a height grid f(x, y) with step step (or a ready grid of points grid: ny × nx × 3 — rock), the material is an
    ambientCG scan (color, roughness, height → relief; box projection — suits a sheer face too); below the waterline and at the
    water — wet."""
    if grid is not None:
        ny, nx = grid.shape[:2]; X, Y, Z = grid[..., 0], grid[..., 1], grid[..., 2]
    else:
        nx, ny = int((x1 - x0) / step) + 1, int((y1 - y0) / step) + 1
        X, Y = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)); Z = f(X, Y)
    ii = np.arange(nx * ny).reshape(ny, nx)
    q = np.stack([ii[:-1, :-1], ii[:-1, 1:], ii[1:, 1:], ii[1:, :-1]], -1).reshape(-1, 4)
    me = bpy.data.meshes.new(name); me.vertices.add(nx * ny); me.loops.add(q.size); me.polygons.add(len(q))
    me.vertices.foreach_set("co", np.stack([X, Y, Z], -1).astype(np.float32).ravel())
    me.loops.foreach_set("vertex_index", q.ravel().astype(np.int32))
    me.polygons.foreach_set("loop_start", np.arange(0, q.size, 4, dtype=np.int32)); me.polygons.foreach_set("loop_total", np.full(len(q), 4, np.int32))
    me.update(); me.shade_smooth()
    ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob)
    m = bpy.data.materials.new(name); nt = m.node_tree; nt.nodes.clear(); L = nt.links.new
    tc = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping"); mp.inputs["Scale"].default_value = (1 / tile, 1 / tile, 1 / tile)
    L(tc.outputs["Object"], mp.inputs[0])
    def img(kind, data=False):
        n = nt.nodes.new("ShaderNodeTexImage"); n.image = bpy.data.images.load(os.path.join(TEX, tex, f"{tex}_2K-JPG_{kind}.jpg"))
        if data: n.image.colorspace_settings.name = "Non-Color"
        n.projection = "BOX"; n.projection_blend = 0.2; L(mp.outputs[0], n.inputs[0]); return n
    col, rou, dis = img("Color"), img("Roughness", True), img("Displacement", True)
    if under:                                                           # the bed with another scan: below z0 — under (name, tile)
        tex_, tile_ = tex, tile; tex, tile = under[0], under[1]
        mp2 = nt.nodes.new("ShaderNodeMapping"); mp2.inputs["Scale"].default_value = (1 / tile, 1 / tile, 1 / tile); L(tc.outputs["Object"], mp2.inputs[0])
        def img2(kind, data=False):
            n = nt.nodes.new("ShaderNodeTexImage"); n.image = bpy.data.images.load(os.path.join(TEX, tex, f"{tex}_2K-JPG_{kind}.jpg"))
            if data: n.image.colorspace_settings.name = "Non-Color"
            n.projection = "BOX"; n.projection_blend = 0.2; L(mp2.outputs[0], n.inputs[0]); return n
        c2, r2, d2 = img2("Color"), img2("Roughness", True), img2("Displacement", True)
        sz = nt.nodes.new("ShaderNodeSeparateXYZ"); L(tc.outputs["Object"], sz.inputs[0])
        fu = nt.nodes.new("ShaderNodeMapRange"); fu.inputs["From Min"].default_value = under[2] + 0.04; fu.inputs["From Max"].default_value = under[2] - 0.04
        L(sz.outputs[2], fu.inputs["Value"])
        def mixf(a, b, rgb=True):
            n = nt.nodes.new("ShaderNodeMix"); n.data_type = "RGBA" if rgb else "FLOAT"; L(fu.outputs[0], n.inputs["Factor"])
            L(a, n.inputs[6 if rgb else 2]); L(b, n.inputs[7 if rgb else 3]); return n.outputs[2 if rgb else 0]
        class _W:
            def __init__(self, o): self.outputs = [o]
        col, rou, dis = _W(mixf(col.outputs[0], c2.outputs[0])), _W(mixf(rou.outputs[0], r2.outputs[0], False)), _W(mixf(dis.outputs[0], d2.outputs[0], False))
    sep = nt.nodes.new("ShaderNodeSeparateXYZ"); L(tc.outputs["Object"], sep.inputs[0])
    wet = nt.nodes.new("ShaderNodeMapRange"); wet.inputs["From Min"].default_value = wet_h; wet.inputs["From Max"].default_value = 0.0
    L(sep.outputs[2], wet.inputs["Value"])
    tn = nt.nodes.new("ShaderNodeMixRGB"); tn.blend_type = "MULTIPLY"; tn.inputs[0].default_value = 1.0; L(col.outputs[0], tn.inputs[1])
    tn.inputs[2].default_value = (*tint, 1.0); col = tn
    mc = nt.nodes.new("ShaderNodeMix"); mc.data_type = "RGBA"; L(wet.outputs[0], mc.inputs["Factor"]); L(col.outputs[0], mc.inputs[6])
    dk = nt.nodes.new("ShaderNodeMixRGB"); dk.blend_type = "MULTIPLY"; dk.inputs[0].default_value = 1.0; L(col.outputs[0], dk.inputs[1])
    dk.inputs[2].default_value = (dark, dark * 0.97, dark * 0.92, 1); L(dk.outputs[0], mc.inputs[7])
    mr = nt.nodes.new("ShaderNodeMath"); mr.operation = "MULTIPLY"; L(rou.outputs[0], mr.inputs[0])
    mr2 = nt.nodes.new("ShaderNodeMapRange"); L(wet.outputs[0], mr2.inputs["Value"]); mr2.inputs["To Min"].default_value = 1.0; mr2.inputs["To Max"].default_value = 0.35
    L(mr2.outputs[0], mr.inputs[1])
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.8; bump.inputs["Distance"].default_value = 0.03; L(dis.outputs[0], bump.inputs["Height"])
    if rough_min > 0:                                                   # grass is matte: otherwise it glints with the sky at grazing angles, like water
        rm = nt.nodes.new("ShaderNodeMath"); rm.operation = "MAXIMUM"; L(mr.outputs[0], rm.inputs[0]); rm.inputs[1].default_value = rough_min; mr = rm
    bs = nt.nodes.new("ShaderNodeBsdfPrincipled"); L(mc.outputs[2], bs.inputs["Base Color"]); L(mr.outputs[0], bs.inputs["Roughness"]); L(bump.outputs[0], bs.inputs["Normal"])
    bs.inputs["Specular IOR Level"].default_value = spec
    out = nt.nodes.new("ShaderNodeOutputMaterial"); L(bs.outputs[0], out.inputs[0])
    me.materials.append(m)
    return ob


def scene_lake(sc, kv):
    """Mountain lake at noon: the camera on a pebble shore (1.5 m above the water) looks at a ridge with a snowy summit (sun from the
    side); at one's feet — a clear shallow with visible pebbles; farther out — deep water and the reflection of the mountains; a
    light wind of 3 m/s: patches of ripples from gusts and calm water between."""
    sky = SKIES[kv.get("sky_kind", "alps_field")]; sun_az, sun_el = sky["sun"]
    world_hdr(sc, sky["file"], strength=float(kv.get("sky", 1.0)))
    az = float(kv.get("azimuth", 250.0))
    cam = camera(sc, (0.0, 0.0, float(kv.get("height", 1.2))), az, float(kv.get("tilt", -5.0)), lens=float(kv.get("lens", 32)))
    fx, fy = math.cos(math.radians(az)), math.sin(math.radians(az))
    shore = float(kv.get("shore", 3.2)); slope = float(kv.get("slope", 0.15))
    def ground(x, y):
        d = x * fx + y * fy; c = -x * fy + y * fx
        z = 0.25 - slope * (d - shore) - 0.012 * c * c / (1 + 0.02 * c * c) * 0.0
        z = z + 0.08 * np.sin(c * 0.45 + 1.3) * np.sin(d * 0.3) - 0.004 * np.maximum(d - 12.0, 0.0) ** 2
        return np.maximum(z, -25.0)
    c0, c1 = 14.0 * fx, 14.0 * fy                                       # the near bed — fine at the camera, the far one — coarse out to 400 m
    tint = (1.45, 1.3, 1.08)                                            # pebbles are lighter and warmer than the scan (gray gravel)
    shore_near = ground_mesh(sc, ground, c0 - 16.0, c0 + 16.0, c1 - 16.0, c1 + 16.0, float(kv.get("step_bed", 0.06)), tint=tint)
    ground_mesh(sc, lambda x, y: ground(x, y) - 0.03, -400.0, 400.0, -400.0, 400.0, 1.0, name="shore_far", tint=tint)
    L = sun_vec(sun_az, sun_el)
    u = float(kv.get("wind", 1.2)); wa = math.radians(float(kv.get("wind_to", az - 30)))
    sea = W.Sea(u10=u, wind_to=(math.cos(wa), math.sin(wa)), fetch=float(kv.get("accel", 1500)), depth=20.0,
                near=float(kv.get("closer", near_for(cam, sc))), seed=int(kv.get("seed", 5)))
    look = optics.preset(kv.get("water", "mountain_lake"))
    wat = Water(sc, sea, cam=cam, grid=(int(kv.get("grid", 560)), int(int(kv.get("grid", 560)) * 0.75)), body=look["body"], K=look["K"],
                ground=ground, scatter=(0.02, 0.12, 0.1), sun=(tuple(L), 4.0), debug=kv.get("kind"),
                bottom=(0.16, 0.14, 0.11),
                gust=dict(scale=float(kv.get("spots", 35.0)), floor=float(kv.get("lull", 0.03)), slick=0.6, drift=0.8,
                          edge=(float(kv.get("threshold", 0.56)), float(kv.get("threshold", 0.56)) + 0.1)))
    wat.update(0.0); add_caustics(shore_near, wat, strength=float(kv.get("caustics", 0.9)))
    return wat, sea




def scene_sea(sc, kv):
    """Sea at sunset: the camera 3.5 m above the water looks 8° to the left of the sun, the horizon above the middle; the wind blows
    from the sun toward the camera, the swell comes from the south-west (wind-sea Hs ~1 m, swell 1.2 m / 10 s). The sky is golden
    hour (the sun in haze → a lamp in its direction: a sun glitter path on the water)."""
    sky = SKIES[kv.get("sky_kind", "qwantani_sunset")]; sun_az, sun_el = sky["sun"]
    world_hdr(sc, sky["file"], strength=float(kv.get("sky", 1.0)), sun_threshold=10.0 if not sky.get("lamp") else 1e5)
    if sky.get("lamp"): sun_lamp(sc, sun_az, sun_el, sky["lamp"][0], float(kv.get("strength", sky["lamp"][1])))
    cam = camera(sc, (0.0, 0.0, float(kv.get("height", 3.5))), sun_az + float(kv.get("off_sun", 6)), float(kv.get("tilt", -5.0)),
                 lens=float(kv.get("lens", 35)))
    L = sun_vec(sun_az, sun_el)
    u = float(kv.get("wind", 8.0)); to = (-L.x, -L.y)
    sw = None if float(kv.get("swell", 1.2)) <= 0 else dict(hs=float(kv.get("swell", 1.2)), tp=float(kv.get("period", 10.0)),
                                                          to=(math.cos(math.radians(sun_az + 200)), math.sin(math.radians(sun_az + 200))))
    sea = W.Sea(u10=u, wind_to=to, swell=sw, near=float(kv.get("closer", near_for(cam, sc))), seed=int(kv.get("seed", 3)))
    wat = Water(sc, sea, cam=cam, grid=(int(kv.get("grid", 520)), int(int(kv.get("grid", 520)) * 0.75)),
                body=optics.preset(kv.get("water", "ocean"))["body"], scatter=(0.03, 0.22, 0.16),
                sun=(tuple(L), float(kv.get("strength", 4.0)) * float(kv.get("clearance", 1.0))), foam_rgb=(0.8, 0.8, 0.78), debug=kv.get("kind"))
    return wat, sea




def lily_pad(name, r, seed):
    """Water lily pad: a disk with a notch (a ~25° wedge), the edge slightly raised; rings for the vein pattern (UV: angle, radius)."""
    rng = np.random.default_rng(seed); n, rings = 56, 6; cut = math.radians(rng.uniform(18, 30)); a0 = rng.uniform(0, 2 * math.pi)
    V, F, UV = [(0.0, 0.0, 0.0045)], [], [(0.5, 0.0)]
    for j in range(1, rings + 1):
        rr = r * j / rings
        for i in range(n):
            a = a0 + cut / 2 + (2 * math.pi - cut) * i / (n - 1)
            V.append((rr * math.cos(a), rr * math.sin(a), 0.0045 + 0.012 * (j / rings) ** 4 + rng.normal(0, 0.0006) * (j > 1))); UV.append((i / (n - 1), j / rings))
    for i in range(n - 1): F.append((0, 1 + i, 2 + i))
    for j in range(rings - 1):
        b0, b1 = 1 + j * n, 1 + (j + 1) * n
        for i in range(n - 1): F.append((b0 + i, b0 + i + 1, b1 + i + 1, b1 + i))
    me = bpy.data.meshes.new(name); me.from_pydata(V, [], F); me.update(); me.shade_smooth()
    uv = me.uv_layers.new(name="uv")
    for poly in me.polygons:
        for li in poly.loop_indices: uv.data[li].uv = UV[me.loops[li].vertex_index]
    return me


def pad_material():
    m = bpy.data.materials.new("lily_pad"); nt = m.node_tree; nt.nodes.clear(); L = nt.links.new
    uv = nt.nodes.new("ShaderNodeUVMap"); uv.uv_map = "uv"; sp = nt.nodes.new("ShaderNodeSeparateXYZ"); L(uv.outputs[0], sp.inputs[0])
    w = nt.nodes.new("ShaderNodeMath"); w.operation = "SINE"; mu = nt.nodes.new("ShaderNodeMath"); mu.operation = "MULTIPLY"; mu.inputs[1].default_value = 140.0
    L(sp.outputs[0], mu.inputs[0]); L(mu.outputs[0], w.inputs[0])                    # veins — rays from the center
    vein = nt.nodes.new("ShaderNodeMapRange"); vein.inputs["From Min"].default_value = 0.85; vein.inputs["From Max"].default_value = 1.0; L(w.outputs[0], vein.inputs["Value"])
    noi = nt.nodes.new("ShaderNodeTexNoise"); noi.inputs["Scale"].default_value = 18.0
    base = nt.nodes.new("ShaderNodeMix"); base.data_type = "RGBA"; L(noi.outputs["Fac"], base.inputs["Factor"])
    base.inputs[6].default_value = (0.035, 0.085, 0.018, 1); base.inputs[7].default_value = (0.07, 0.13, 0.03, 1)
    edge = nt.nodes.new("ShaderNodeMapRange"); edge.inputs["From Min"].default_value = 0.88; edge.inputs["From Max"].default_value = 1.0; L(sp.outputs[1], edge.inputs["Value"])
    c2 = nt.nodes.new("ShaderNodeMix"); c2.data_type = "RGBA"; L(edge.outputs[0], c2.inputs["Factor"]); L(base.outputs[2], c2.inputs[6]); c2.inputs[7].default_value = (0.16, 0.07, 0.03, 1)
    c3 = nt.nodes.new("ShaderNodeMix"); c3.data_type = "RGBA"; L(vein.outputs[0], c3.inputs["Factor"]); L(c2.outputs[2], c3.inputs[6]); c3.inputs[7].default_value = (0.1, 0.16, 0.05, 1)
    bs = nt.nodes.new("ShaderNodeBsdfPrincipled"); L(c3.outputs[2], bs.inputs["Base Color"]); bs.inputs["Roughness"].default_value = 0.28
    bs.inputs["Subsurface Weight"].default_value = 0.0
    out = nt.nodes.new("ShaderNodeOutputMaterial"); L(bs.outputs[0], out.inputs[0])
    m.surface_render_method = "BLENDED"                                 # floating objects — a layer after the water: the screen-space refraction ray
    return m                                                            # otherwise it passes "behind" the pad and picks up the sky (a blue hole)


def rocks(sc, pts, seed):
    """Mossy rocks at the rim (Poly Haven rock_moss_set_01, CC0) — at points: (x, y, z, size)."""
    f = os.path.join(ROOT, "library", "rocks", "polyhaven", "rock_moss_set_01", "rock_moss_set_01_2k.blend")
    with bpy.data.libraries.load(f) as (src, dst): dst.objects = [o for o in src.objects]
    protos = [o for o in dst.objects if o is not None]
    rng = np.random.default_rng(seed); out = []
    for i, (x, y, z, sz) in enumerate(pts):
        p = protos[rng.integers(len(protos))]
        co = np.empty(len(p.data.vertices) * 3); p.data.vertices.foreach_get("co", co); co = co.reshape(-1, 3) * np.array(p.scale)
        dims = float(np.ptp(co, axis=0).max()); k = sz / max(dims, 1e-3)     # the extent from the vertices: an object outside the scene has dimensions 0
        o = bpy.data.objects.new(f"rock_{i}", p.data); sc.collection.objects.link(o)
        o.scale = (k, k, k * rng.uniform(0.7, 1.0)); o.rotation_euler = (0, 0, rng.uniform(0, 2 * math.pi))
        zc = float(co[:, 2].min()) * k                                  # the rock's bottom on the ground, a quarter of it sunk in
        o.location = (x, y, z - zc - sz * 0.25); out.append(o)
    return out


def scene_pond(sc, kv):
    """Pond in a Chinese garden: the camera above the water (1.6 m) looks down at ~38° toward the big tree of the panorama (it and the
    sky are in the reflection), an irregular basin ~0.9 m deep (fallen leaves on the bottom), the rim of mossy rocks, paving around;
    water lilies sway on the ripples; a breeze of 0.8 m/s — patches of ripples and calm water."""
    sky = SKIES[kv.get("sky_kind", "chinese_garden")]; sun_az, sun_el = sky["sun"]
    world_hdr(sc, sky["file"], strength=float(kv.get("sky", 1.0)), rot_deg=float(kv.get("turn", 0.0)))
    az = float(kv.get("azimuth", 125.0)); h = float(kv.get("height", 1.7))
    cam = camera(sc, (0.0, 0.0, h), az, float(kv.get("tilt", -30.0)), lens=float(kv.get("lens", 26)))
    fx, fy = math.cos(math.radians(az)), math.sin(math.radians(az)); cx, cy = 3.0 * fx, 3.0 * fy
    def shape(x, y):
        dx, dy = x - cx, y - cy; d = dx * fx + dy * fy; c = -dx * fy + dy * fx
        th = np.arctan2(c / 1.35, d); R = 2.3 + 0.3 * np.sin(2 * th + 1.0) + 0.2 * np.sin(3 * th - 0.5)
        return np.hypot(d, c / 1.35) / R, th, R
    def ground(x, y):
        s, th, R = shape(x, y)
        bowl = -0.08 - 0.85 * np.sqrt(np.clip(1 - s * s, 0, 1))
        return bowl * (1 - ss(s, 0.95, 1.04)) + 0.14 * ss(s, 0.95, 1.04)
    pond_bed = ground_mesh(sc, ground, cx - 7.0, cx + 7.0, cy - 7.0, cy + 7.0, float(kv.get("step_bed", 0.04)), tex="PavingStones046", tile=2.0,
                           dark=0.6, under=("ScatteredLeaves009", 1.2, 0.02))
    rng = np.random.default_rng(int(kv.get("seed", 4)))
    pts = []
    for th in np.arange(0, 2 * math.pi, 0.26):
        R = 2.3 + 0.3 * math.sin(2 * th + 1.0) + 0.2 * math.sin(3 * th - 0.5); rr = R * 1.0
        d, c = rr * math.cos(th), rr * math.sin(th) * 1.35
        x, y = cx + d * fx - c * fy, cy + d * fy + c * fx
        pts.append((x, y, 0.05, rng.uniform(0.45, 0.95) * (1.6 if rng.random() < 0.15 else 1.0)))
    rocks(sc, pts, 9)
    L = sun_vec(sun_az, sun_el)
    u = float(kv.get("wind", 0.8))
    sea = W.Sea(u10=u, wind_to=(fy, -fx), fetch=6.0, depth=0.9, near=float(kv.get("closer", near_for(cam, sc))), seed=int(kv.get("seed", 4)))
    look = optics.preset(kv.get("water", "pond_clear"))
    wat = Water(sc, sea, cam=cam, grid=(int(kv.get("grid", 480)), int(int(kv.get("grid", 480)) * 0.75)), body=look["body"], K=look["K"],
                ground=ground, scatter=(0.02, 0.1, 0.06), sun=(tuple(L), 3.0), debug=kv.get("kind"), bottom=(0.16, 0.12, 0.07),
                gust=dict(scale=float(kv.get("spots", 2.5)), floor=float(kv.get("lull", 0.15)), slick=0.3, drift=0.6, edge=(0.45, 0.6)),
                extra=[W.Rain(rate=float(kv["rain"]), L=2.0, N=512, seed=3, fps=float(kv.get("fps", 12)), amp=1.2e-3 * float(kv.get("drops", 1.0)),
                              s=0.004 * float(kv.get("width_drops", 1.0))),
                       W.Rain(rate=float(kv["rain"]) * 0.5, L=2.71, N=512, seed=8, fps=float(kv.get("fps", 12)), amp=1.2e-3 * float(kv.get("drops", 1.0)),
                              s=0.004 * float(kv.get("width_drops", 1.0)))]
                if float(kv.get("rain", 0)) > 0 else None)
    pm = pad_material(); pads = []
    for k in range(int(kv.get("lilies", 16))):
        while True:
            x, y = cx + rng.uniform(-2.2, 2.2), cy + rng.uniform(-2.6, 2.6)
            if shape(np.array(x), np.array(y))[0] < 0.8: break
        me = lily_pad(f"pad_{k}", rng.uniform(0.12, 0.26), k); me.materials.append(pm)
        o = bpy.data.objects.new(f"pad_{k}", me); sc.collection.objects.link(o); o.rotation_euler = (0, 0, rng.uniform(0, 2 * math.pi))
        o.location = (x, y, 0.0); pads.append(o)
    def on_frame(t, F):
        if not pads: return
        xy = np.array([[o.location.x, o.location.y] for o in pads]); off = np.array([[0, 0], [0.08, 0], [-0.08, 0], [0, 0.08], [0, -0.08]])
        zs, ns = zip(*[W.surface_at(sea, F, xy + o_) for o_ in off])       # the pad lies at the mean ripple height beneath it
        z, n = np.mean(zs, 0), np.mean(ns, 0)
        for o, zz, nn in zip(pads, z, n):
            o.location.z = float(max(zz, 0.0)) + 0.006
            o.rotation_euler.x = float(math.atan2(nn[1], nn[2])) * -1; o.rotation_euler.y = float(math.atan2(nn[0], nn[2]))
    wat.on_frame = on_frame
    wat.update(0.0); add_caustics(pond_bed, wat, strength=float(kv.get("caustics", 0.7)))
    return wat, sea


def ss(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _ss(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def scene_beach(sc, kv):
    """Beach by day: the camera on dry sand (1.6 m) looks out to sea at an angle to the surf lines, the sun behind and to the side (the
    turquoise of the shallows over pale sand); sand 1:30 — the waves spill (ξ < 0.5), a foamy bore, a swash tongue runs up the sand;
    wind 5 m/s onshore, swell 0.9 m / 9 s."""
    sky = SKIES[kv.get("sky_kind", "kloofendal_day")]; sun_az, sun_el = sky["sun"]
    world_hdr(sc, sky["file"], strength=float(kv.get("sky", 1.0)), rot_deg=float(kv.get("turn", 0.0)))
    az = float(kv.get("azimuth", sun_az + 150.0)); off = math.radians(az + float(kv.get("oblique", 42.0)))   # "out to sea" — at an angle
    ox, oy = math.cos(off), math.sin(off)
    tanb = float(kv.get("slope", 0.033)); s_cam = float(kv.get("off_water", 34.0)); dune = float(kv.get("dune", 6.0))
    def ground(x, y):
        s_ = x * ox + y * oy - s_cam; c = -x * oy + y * ox                 # s_ — from the waterline out to sea, m; c — along the shore
        z = 0.35 - tanb * s_ + 0.12 * np.sin(c / 9.0) * np.exp(-np.maximum(s_, 0) / 40.0) - 0.0006 * np.maximum(s_ - 120.0, 0) ** 2
        z = z + dune * _ss(-s_, 16.0, 30.0) * (1 + 0.15 * np.sin(c / 17.0))   # a dune behind the beach — the camera stands on it
        return np.maximum(z, -18.0)
    zc = float(ground(np.array(0.0), np.array(0.0)))
    cam = camera(sc, (0.0, 0.0, zc + float(kv.get("height", 1.6))), az, float(kv.get("tilt", -11.0)), lens=float(kv.get("lens", 30)))
    tint = (1.55, 1.38, 1.12)                                           # dry sand is light and warm
    wh = 0.3                                                            # wet sand — in the swash tongue
    near = ground_mesh(sc, ground, -26.0, 26.0, -26.0, 26.0, float(kv.get("step_bed", 0.08)), tex="Ground054", tile=3.5, dark=0.62, tint=tint, wet_h=wh)
    ground_mesh(sc, lambda x, y: ground(x, y) - 0.03, -320.0, 320.0, -320.0, 320.0, 1.0, name="shore_far", tex="Ground054", tile=3.5, dark=0.62, tint=tint, wet_h=wh)
    L = sun_vec(sun_az, sun_el)
    to = (-ox, -oy)                                                     # toward the shore
    sea = W.Sea(u10=float(kv.get("wind", 5.0)), wind_to=to, swell=dict(hs=float(kv.get("swell", 0.9)), tp=float(kv.get("period", 9.0)), to=to),
                near=float(kv.get("closer", near_for(cam, sc))), seed=int(kv.get("seed", 6)))
    from components.water.shore import Shore
    b0 = (-260.0, -260.0, 260.0, 260.0)
    shore = Shore(ground, 0.0, sea, b0, res=float(kv.get("step_surf", 0.5)), seed=3)
    look = optics.preset(kv.get("water", "lagoon"))
    wat = Water(sc, sea, cam=cam, grid=(int(kv.get("grid", 560)), int(int(kv.get("grid", 560)) * 0.75)), body=look["body"], K=look["K"],
                ground=ground, scatter=(0.03, 0.2, 0.16), sun=(tuple(L), 4.0), debug=kv.get("kind"), bottom=(0.42, 0.36, 0.26), shore=shore)
    wat.update(0.0); add_caustics(near, wat, strength=float(kv.get("caustics", 0.8)))
    print("surf:", shore.summary(), flush=True)
    return wat, sea


def scene_river(sc, kv):
    """Mountain river in a valley: the camera on a grassy bank (1.6 m) looks downstream at an angle; channel 12 m wide, depth ~1.1 m,
    pebbles on the bed, midstream speed 1.3 m/s (the current along the axis, slower near the banks), stones in the flow — foam
    trails, the current carries the ripples."""
    sky = SKIES[kv.get("sky_kind", "alps_field")]; sun_az, sun_el = sky["sun"]
    world_hdr(sc, sky["file"], strength=float(kv.get("sky", 1.0)))
    az = float(kv.get("azimuth", 250.0)); fa = math.radians(az + float(kv.get("oblique", 11.0)))   # the river axis is on the right, converging toward the center in the distance
    fx, fy = math.cos(fa), math.sin(fa); nx_, ny_ = fy, -fx                # (nx_, ny_) — to the right of the axis
    c0 = float(kv.get("from_side", 8.0))                                    # the river axis is 8 m to the right of the camera
    path = [[c0 * nx_ + fx * q + nx_ * 3.0 * math.sin(q / 23.0), c0 * ny_ + fy * q + ny_ * 3.0 * math.sin(q / 23.0)] for q in np.arange(-80, 160, 4.0)]
    P = np.array(path); b = float(kv.get("width_river", 12.0)) / 2; D = float(kv.get("depth", 1.1))
    from scipy.spatial import cKDTree
    from components.water.flow import _dense
    tree = cKDTree(_dense(P, 0.3))                                      # to the axis line, not to the points (otherwise the bank is "scalloped")
    def ground(x, y):
        d, _ = tree.query(np.stack([np.ravel(x), np.ravel(y)], 1)); d = d.reshape(np.shape(x))
        q = d / b
        bed = -0.12 - D * np.clip(1 - q * q, 0, 1)                       # the channel is a bowl, the waterline at the edge
        ramp = -0.12 + 0.42 * _ss(d, b, b + 2.5)                          # a gentle pebble slope up to the grass
        bank = ramp + 0.05 * np.maximum(d - b - 2.5, 0) + 0.12 * np.sin(np.asarray(x) / 7.0) * np.cos(np.asarray(y) / 9.0) * _ss(d, b + 2, b + 5)
        return np.where(q < 1, bed, bank)
    zc = float(ground(np.array(0.0), np.array(0.0)))
    cam = camera(sc, (0.0, 0.0, zc + float(kv.get("height", 1.6))), az, float(kv.get("tilt", -10.0)), lens=float(kv.get("lens", 28)))
    lk = (math.cos(math.radians(az)), math.sin(math.radians(az))); cx_, cy_ = 18 * lk[0], 18 * lk[1]   # the near part — in front of the camera
    near = ground_mesh(sc, ground, cx_ - 24.0, cx_ + 24.0, cy_ - 24.0, cy_ + 24.0, float(kv.get("step_bed", 0.08)), tex="Grass004", tile=1.4,
                       dark=0.7, under=("Gravel043", 1.6, 0.18), spec=0.15, rough_min=0.8)
    ground_mesh(sc, lambda x, y: ground(x, y) - 0.03, -200.0, 200.0, -200.0, 200.0, 0.5, name="shore_far", tex="Grass004", tile=1.4, dark=0.7,
                under=("Gravel043", 1.6, 0.18), spec=0.15, rough_min=0.8)
    rng = np.random.default_rng(int(kv.get("seed", 5)))
    rk = []
    for q in (14.0, 22.0, 31.0, 38.0, 47.0):                            # stones in the flow: along the axis with a lateral offset
        i = int(np.argmin(np.abs(np.arange(-80, 160, 4.0) - q))); cx_, cy_ = P[i]; lat = rng.uniform(-0.55, 0.55) * b
        rk.append((cx_ + nx_ * lat, cy_ + ny_ * lat, rng.uniform(0.5, 0.95)))
    rocks(sc, [(x, y, -0.05, 2.3 * r) for x, y, r in rk], 13)
    from components.water.flow import River
    flow = River(path, 2 * b, float(kv.get("speed", 1.3)), (-40.0, -60.0, 120.0, 80.0), res=0.25, rocks=rk)
    L = sun_vec(sun_az, sun_el)
    sea = W.Sea(u10=float(kv.get("ripple", 4.0)), wind_to=(fx, fy), fetch=25.0, depth=D, near=float(kv.get("closer", near_for(cam, sc))),
                seed=int(kv.get("seed", 5)))                             # river ripples come from the current: short and steep (fetch ~25 m)
    look = optics.preset(kv.get("water", "mountain_lake"))
    wat = Water(sc, sea, cam=cam, grid=(int(kv.get("grid", 560)), int(int(kv.get("grid", 560)) * 0.75)), body=look["body"], K=look["K"],
                ground=ground, scatter=(0.02, 0.12, 0.1), sun=(tuple(L), 4.0), debug=kv.get("kind"), bottom=(0.16, 0.14, 0.11), flow=flow)
    wat.update(0.0); add_caustics(near, wat, strength=float(kv.get("caustics", 0.7)))
    return wat, sea


def scene_rapids(sc, kv):
    """Torrent: a mountain river, slope 3.5 %, two steps (0.9 and 0.6 m), channel 9 m, depth 0.8 m, boulders; speed by Manning
    (n = 0.045 — boulder bed) ~3.6 m/s, Froude > 1: standing waves, hydraulic jumps below the steps, pillows and V-wakes of the stones.
    The camera is on the bank downstream (1.6 m above the water) and looks upstream, at the flow."""
    from scipy.spatial import cKDTree
    from components.water.flow import River, _dense, profile, manning
    sky = SKIES[kv.get("sky_kind", "alps_field")]; sun_az, sun_el = sky["sun"]
    world_hdr(sc, sky["file"], strength=float(kv.get("sky", 1.0)))
    az = float(kv.get("azimuth", 250.0)); lk = np.array([math.cos(math.radians(az)), math.sin(math.radians(az))]); rt = np.array([lk[1], -lk[0]])
    c0 = float(kv.get("from_side", 5.5)); S0 = 95.0                         # the axis is 5.5 m to the right; the axis origin (s = 0) is 95 m ahead
    slope, D, W_ = float(kv.get("slope", 0.035)), float(kv.get("depth", 0.8)), float(kv.get("width_river", 9.0))
    drops = [(S0 - 52.0, 0.9), (S0 - 30.0, 0.6)]                        # the steps are 52 and 30 m from the camera
    v0 = manning(D, slope, 0.045)
    ss_ = np.arange(0.0, S0 + 30.0, 2.0)
    path = [list(rt * c0 + lk * (S0 - q) + rt * 2.2 * math.sin(q / 19.0)) for q in ss_]
    Pd = _dense(np.array(path), 0.25); sP = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Pd, axis=0), axis=1))]; tree = cKDTree(Pd)
    z0 = float(profile(np.array(S0), slope, drops, 0.0, v0)) * -1.0        # the level at the camera is 0
    b = W_ / 2
    def ground(x, y):
        d, j = tree.query(np.stack([np.ravel(x), np.ravel(y)], 1)); d = d.reshape(np.shape(x)); s_ = sP[j].reshape(np.shape(x))
        lv = profile(s_, slope, drops, z0, v0); q = d / b
        bed = lv - 0.12 - D * np.clip(1 - q * q, 0, 1)
        bank = lv - 0.12 + 0.55 * _ss(d, b, b + 2.0) + 0.06 * np.maximum(d - b - 2.0, 0) \
            + 0.15 * np.sin(np.asarray(x) / 5.0) * np.cos(np.asarray(y) / 6.0) * _ss(d, b + 1.5, b + 4.0)
        return np.where(q < 1, bed, bank)
    cam = camera(sc, (0.0, 0.0, float(kv.get("height", 1.6))), az, float(kv.get("tilt", -3.0)), lens=float(kv.get("lens", 30)))
    cx_, cy_ = 34 * lk[0], 34 * lk[1]
    near = ground_mesh(sc, ground, cx_ - 36.0, cx_ + 36.0, cy_ - 36.0, cy_ + 36.0, float(kv.get("step_bed", 0.1)), tex="Gravel043", tile=1.6,
                       dark=0.6, spec=0.3, rough_min=0.5)                # mountain river: pebble banks
    ground_mesh(sc, lambda x, y: ground(x, y) - 0.03, -250.0, 250.0, -250.0, 250.0, 0.6, name="shore_far", tex="Grass004", tile=1.4, dark=0.7,
                under=("Gravel043", 1.6, 0.18), spec=0.15, rough_min=0.8)
    rng = np.random.default_rng(int(kv.get("seed", 3)))
    rk, rk3 = [], []
    for dist_ in (12.0, 18.0, 24.0, 35.0, 41.0, 47.0, 58.0, 66.0):      # boulders in the channel (m from the camera), lateral offset
        q = S0 - dist_; i = int(np.argmin(np.abs(ss_ - q))); cx_, cy_ = path[i]; lat = rng.uniform(-0.6, 0.6) * b
        x_, y_ = cx_ + rt[0] * lat, cy_ + rt[1] * lat; sz = rng.uniform(2.0, 2.8)          # they stick out of the water (depth 0.8 m)
        rk.append((x_, y_, 0.33 * sz)); rk3.append((x_, y_, float(ground(np.array(x_), np.array(y_))), sz))
    for k in range(14):                                                 # boulders on the banks
        q = rng.uniform(5.0, S0); i = int(np.argmin(np.abs(ss_ - q))); cx_, cy_ = path[i]; lat = (b + rng.uniform(0.3, 2.5)) * rng.choice([-1, 1])
        x_, y_ = cx_ + rt[0] * lat, cy_ + rt[1] * lat; rk3.append((x_, y_, float(ground(np.array(x_), np.array(y_))), rng.uniform(0.8, 2.0)))
    bb = (min(p_[0] for p_ in path) - 8, min(p_[1] for p_ in path) - 8, max(p_[0] for p_ in path) + 8, max(p_[1] for p_ in path) + 8)
    flow = River(path, W_, v0, bb, res=0.2, rocks=rk, slope=slope, drops=drops, depth=D, z0=z0, n=0.045)
    objs = rocks(sc, rk3, 11)
    for o, (x_, y_, r_) in zip(objs, rk):                               # boulders in the channel — 0.3–0.6 m above the water, the bottom sunk in the bed
        co = np.empty(len(o.data.vertices) * 3); o.data.vertices.foreach_get("co", co); zz = co.reshape(-1, 3)[:, 2]
        lv = float(flow.level_at(np.array(x_), np.array(y_))); bed = float(ground(np.array(x_), np.array(y_)))
        lv = lv if np.isfinite(lv) else bed + D
        top = lv + float(rng.uniform(0.3, 0.6)); bot = bed - 0.3
        o.scale.z = (top - bot) / max(float(np.ptp(zz)), 1e-3); o.location.z = bot - float(zz.min()) * o.scale.z
    print("flow:", flow.summary(), flush=True)
    L = sun_vec(sun_az, sun_el)
    fdir = (-lk[0], -lk[1])
    sea = W.Sea(u10=float(kv.get("ripple", 6.0)), wind_to=fdir, fetch=25.0, depth=D, near=float(kv.get("closer", near_for(cam, sc))),
                seed=int(kv.get("seed", 3)))
    look = optics.preset(kv.get("water", "mountain_lake"))
    wat = Water(sc, sea, cam=cam, grid=(int(kv.get("grid", 640)), int(int(kv.get("grid", 640)) * 0.75)), body=look["body"], K=look["K"],
                ground=ground, scatter=(0.02, 0.12, 0.1), sun=(tuple(L), 4.0), debug=kv.get("kind"), bottom=(0.16, 0.14, 0.11), flow=flow,
                level_fn=flow.level_at)
    wat.update(0.0)
    return wat, sea


def scene_fall(sc, kv):
    """Waterfall of the kind type= (components/water/falls.yaml): the rock follows the kind's profile (a niche — the jet flies, a convex
    slope — it slides), a plateau on top with the channel in a cut, a plunge pool below (impact — foam, rings, boiling) and water
    mist; the camera on the pool bank looks at the waterfall."""
    from components.water.fall import Fall
    from components.water.bl_fall import FallMesh, mist, update_mist
    from components.water.flow import Pool
    from components.water.fall import FALLS
    kind = kv.get("type", "plunge")
    sky = SKIES[kv.get("sky_kind", "kloofendal_day")]; sun_az, sun_el = sky["sun"]
    want = float(kv.get("sun", 70.0))                                # the sun is in front of the rock (+y), slightly to the side: Mapping rotates the sky before
    rot = float(kv.get("turn", sun_az - want)); sun_az = sun_az - rot  # the sampling → in the world, the azimuth of the shot − the rotation
    world_hdr(sc, sky["file"], strength=float(kv.get("sky", 1.0)), rot_deg=rot)
    seed = int(kv.get("seed", 2)); pk = FALLS[kind]
    if pk["face"] == "steps":                                           # stepped: the jet lands on a shelf, white water runs along the shelf to the next lip
        Fs, z_, y_ = [], float(pk["height"]), 0.0
        for i, (h_, shelf) in enumerate(pk["steps"]):
            Fs.append(Fall("plunge", z_lip=z_, y0=y_, seed=seed + i, approach=(pk["steps"][i - 1][1] if i else 3.0), white_in=i > 0,
                           height=h_, width=pk["width"], q=pk["q"], recess=0.08, spread=pk.get("spread", 0.1), rough=pk.get("rough", 0.4)))
            z_ -= h_; y_ += shelf
    else:
        Fs = [Fall(kind, seed=seed)]
    F = Fs[0]; Fl = Fs[-1]
    H, Wd = float(pk["height"]), F.W; Rp, Pd = pk.get("pool", [10, 2.0])
    for F_ in Fs: print("waterfall:", F_.summary(), flush=True)
    def bowl_y(x):                                                      # amphitheater: the side walls come forward toward the camera
        return 0.03 * np.maximum(np.abs(np.asarray(x, float) - Fs[0].x0) - Fs[0].W / 2 - 2.0, 0.0) ** 2
    def face_all(x, z):                                                 # rock: each height has its own step (the shelves at the joints)
        x, z = np.asarray(x, float), np.asarray(z, float); out = Fs[-1].face(x, z)
        for F_ in Fs[-2::-1]: out = np.where(z > F_.z_pool, F_.face(x, z), out)
        return out + bowl_y(x)
    ix, iy = float(Fl.impact[:, 0].mean()), float(Fl.impact[:, 1].mean())
    Wc = max(Wd * 1.2 + 14.0, 40.0); bank = 1.2 + 0.04 * H
    def zplat(x):                                                       # plateau: the channel in a cut as wide as the lip, the banks higher
        return F.z_lip + bank * _ss(np.abs(x - F.x0) - Wd / 2, 0.3, 2.5) + 0.25 * np.sin(np.asarray(x) / 3.1)
    zb = -Pd - 1.5
    xs = np.arange(-Wc, Wc + 1e-6, 0.25); us = np.linspace(0.0, 1.0, int((F.z_lip + bank - zb) / 0.25) + 1)
    Xg, Ug = np.meshgrid(xs, us); Zt = zplat(Xg); Zg = zb + (Zt - zb) * Ug
    Yg = np.where(Zg <= F.z_lip, face_all(Xg, Zg), F.y0 + bowl_y(Xg))
    ground_mesh(sc, None, 0, 0, 0, 0, 0, name="cliff", tex=kv.get("rock", "Rock030"), tile=5.0, dark=0.45, grid=np.stack([Xg, Yg, Zg], -1))
    xp = np.arange(-Wc, Wc + 1e-6, 0.3); vp = np.linspace(0.0, 1.0, 140); Xp, Vp = np.meshgrid(xp, vp)
    Yp = -40.0 + (F.y0 + bowl_y(Xp) - 0.05 + 40.0) * Vp                 # the plateau extends to the curved rock lip
    ground_mesh(sc, None, 0, 0, 0, 0, 0, name="plateau", tex="Ground037", tile=2.0, dark=0.7, grid=np.stack([Xp, Yp, zplat(Xp)], -1))
    ymin = float(face_all(xs, np.full_like(xs, zb)).min()) - 1.0
    def basin(x, y):
        r = np.hypot(np.asarray(x) - ix, np.asarray(y) - iy)
        bowl = -0.15 - Pd * np.clip(1 - (r / Rp) ** 2, 0, 1)
        return np.where(r < Rp, bowl, -0.15 + 0.45 * _ss(r, Rp, Rp + 3.0) + 0.05 * np.maximum(r - Rp - 3.0, 0.0)
                        + 0.12 * np.sin(np.asarray(x) / 2.7) * np.cos(np.asarray(y) / 3.3))
    D = float(kv.get("far", 1.4 * H + 12.0))
    ground_mesh(sc, basin, -Wc - 30, Wc + 30, ymin, D + 60.0, 0.2, name="basin", tex="Gravel043", tile=1.6, dark=0.8, spec=0.3, rough_min=0.5)
    rng = np.random.default_rng(int(kv.get("seed", 2))); pts = []
    for k in range(16):                                                 # boulders along the pool edge
        a_ = rng.uniform(0.15, 0.85) * math.pi; r_ = Rp + rng.uniform(-0.5, 2.5)
        x_, y_ = ix + r_ * math.cos(a_ + math.pi / 2) * 1.0, iy + r_ * math.sin(a_)
        pts.append((x_, y_, float(basin(np.array(x_), np.array(y_))), rng.uniform(0.8, 2.2)))
    rocks(sc, pts, 9)
    zc = float(basin(np.array(F.x0), np.array(iy + D))) + float(kv.get("height", 1.7))
    pitch = math.degrees(math.atan2(0.45 * H - zc, D)); cam = camera(sc, (F.x0, iy + D, zc), -90.0, float(kv.get("tilt", pitch)),
                                                                       lens=float(kv.get("lens", 28)))
    L = sun_vec(sun_az, sun_el)
    pool = Pool(Fl.impact, float(Fl.summary()["v_impact"]), (ix - Rp - 8, ymin, ix + Rp + 8, iy + Rp + 12), res=0.2, radius=Rp,
                out=(0.0, 0.25))
    sea = W.Sea(u10=float(kv.get("ripple", 3.0)), wind_to=(0.0, 1.0), fetch=20.0, depth=Pd, near=float(kv.get("closer", near_for(cam, sc))),
                seed=int(kv.get("seed", 2)))
    look = optics.preset(kv.get("water", "mountain_lake"))
    wat = Water(sc, sea, cam=cam, grid=(int(kv.get("grid", 560)), int(int(kv.get("grid", 560)) * 0.75)), body=look["body"], K=look["K"],
                ground=basin, scatter=(0.02, 0.12, 0.1), sun=(tuple(L), 4.0), debug=kv.get("kind"), bottom=(0.16, 0.14, 0.11), flow=pool)
    fms = [FallMesh(sc, F_, name=f"fall{i}") for i, F_ in enumerate(Fs)]
    mss = [mist(sc, F_, strength=float(kv.get("dust", 1.0)) * (1.0 if F_ is Fl else 0.4)) for F_ in Fs]
    old = wat.update
    def upd(t, _o=old):
        D_ = _o(t)
        for fm in fms: fm.update(t)
        for ms in mss: update_mist(ms, t)
        return D_
    wat.update = upd
    return wat, sea


def peaks(window="maroon_bells_wide", r0=66.0, r1=40000.0, nr=900, nth=1440, lake=1100.0, depth=25.0, at=None, center=(0.0, 0.0)):
    """Mountains for the far-terrain test: a real-relief window (library/terrain/real/belt) around the camera as a polar grid — an
    equal number of points per degree (sparser far out, as the angular size of a frame point demands); the middle is the lowest
    place within 6 km of the window middle, with a lake basin of radius lake (floor −depth) there; slope detail — ridged noise at
    30–150 m on steep ground. → (grid (nth, nr, 3), f(x, y) → z above the lake)."""
    from scipy import ndimage
    d = np.load(os.path.join(ROOT, "library", "terrain", "real", "belt", window + ".npz")); Z = d["z"].astype(float); res = float(d["res"]); n = Z.shape[0]
    if at is None:
        yy, xx = np.mgrid[0:n, 0:n]; m = np.hypot(yy - n // 2, xx - n // 2) * res < 6000
        iy, ix = divmod(int(np.argmin(np.where(m, Z, 1e9))), n)
    else: iy, ix = at
    z0 = Z[iy, ix]
    def h(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        z = ndimage.map_coordinates(Z, [np.clip(iy + y / res, 0, n - 1).ravel(), np.clip(ix + x / res, 0, n - 1).ravel()], order=3, mode="nearest").reshape(x.shape) - z0
        r = np.hypot(x, y); bowl = -depth * np.clip(1 - (r / lake) ** 2, 0, 1)
        return np.where(r < lake * 1.6, np.minimum(z, bowl + (z - bowl) * _ss(r, lake * 0.85, lake * 1.6)) if True else z, z)
    rr = r0 * (r1 / r0) ** (np.arange(nr) / (nr - 1)); th = np.linspace(0, 2 * math.pi, nth + 1)   # +1: the ring is closed (no crack)
    T, Rr = np.meshgrid(th, rr, indexing="ij"); X, Y = center[0] + Rr * np.cos(T), center[1] + Rr * np.sin(T)
    Zm = h(X, Y)
    gy, gx = np.gradient(Zm, axis=(0, 1)); steep = _ss(np.hypot(gx, gy) / np.maximum(Rr * (rr[1] / rr[0] - 1), 1.0), 0.35, 0.9)
    det = 0.0                                                           # fine detail only on steep ground and not far away (farther out there are fewer points)
    for k, L_ in enumerate((140.0, 60.0, 28.0)):
        ph = 17.0 * k
        det = det + L_ * 0.12 * (1 - np.abs(np.sin(X / L_ + ph) * np.cos(Y / (L_ * 0.83) + 2 * ph) + 0.5 * np.sin((X + Y) / (L_ * 0.61) + ph)))
    Zm = Zm + det * steep * (1 - _ss(Rr, 6000.0, 15000.0))
    return np.stack([X, Y, Zm], -1), h, rr


def peaks_material(name="peaks", snow=900.0, haze_rgb=(0.62, 0.7, 0.82), vis=60000.0, fade=None):
    """Cheap mountain shader: rock / scree / grass / snow by slope and height above the lake, lighting — Principled; aerial perspective —
    toward the horizon color by e^(−d/visibility) (Koschmieder; no volume)."""
    m = bpy.data.materials.new(name); nt = m.node_tree; nt.nodes.clear()
    from components.water.bl_water import N
    g = N(nt); geo = g.n("ShaderNodeNewGeometry"); pos = g.sep(geo.outputs["Position"]); nz = g.sep(geo.outputs["Normal"])[2]
    nn = g.n("ShaderNodeTexNoise", noise_dimensions="3D"); g.put(nn.inputs["Vector"], g.v("SCALE", geo.outputs["Position"], s=1 / 90.0))
    nn.inputs["Detail"].default_value = 6.0; noise = nn.outputs["Fac"]
    rock, scree, grass, snowc = (0.16, 0.14, 0.12), (0.3, 0.27, 0.23), (0.07, 0.1, 0.035), (0.86, 0.88, 0.92)
    flat = g.ss(g.m("ADD", nz, g.m("MULTIPLY", g.m("SUBTRACT", noise, 0.5), 0.3)), 0.72, 0.9)
    low = g.m("SUBTRACT", 1.0, g.ss(g.m("ADD", pos[2], g.m("MULTIPLY", noise, 160.0)), 350.0, 650.0))
    col = g.mix(flat, rock, scree)
    col = g.mix(g.m("MULTIPLY", flat, low), col, grass)
    sn = g.m("MULTIPLY", g.ss(g.m("ADD", pos[2], g.m("MULTIPLY", g.m("SUBTRACT", noise, 0.5), 300.0)), snow - 150.0, snow + 150.0), g.ss(nz, 0.45, 0.7))
    col = g.mix(sn, col, snowc)
    bs = g.n("ShaderNodeBsdfPrincipled"); g.put(bs.inputs["Base Color"], col); bs.inputs["Roughness"].default_value = 0.85
    cd = g.n("ShaderNodeCameraData"); fog = g.m("SUBTRACT", 1.0, g.m("EXPONENT", g.m("DIVIDE", cd.outputs["View Distance"], -vis)))
    em = g.n("ShaderNodeEmission"); em.inputs["Color"].default_value = (*haze_rgb, 1.0); em.inputs["Strength"].default_value = 1.0
    mx = g.n("ShaderNodeMixShader"); g.put(mx.inputs[0], fog); g.put(mx.inputs[1], bs.outputs[0]); g.put(mx.inputs[2], em.outputs[0])
    out_ = mx.outputs[0]
    if fade:                                                            # the ring edge at the far panorama — to the panorama's own color by direction
        bp = g.n("ShaderNodeCombineXYZ"); bp.name = "bake_point"         # from the bake point (no transparency: Dithered is three times as expensive)
        d_ = g.v("NORMALIZE", g.v("SUBTRACT", geo.outputs["Position"], bp.outputs[0]))
        tx = g.n("ShaderNodeTexEnvironment", interpolation="Linear"); tx.name = "pano_lookup"; g.put(tx.inputs[0], d_)   # the 2:1 seam — as in the world
        e2 = g.n("ShaderNodeEmission"); g.put(e2.inputs["Color"], tx.outputs["Color"])
        fm = g.n("ShaderNodeMixShader"); g.put(fm.inputs[0], g.ss(cd.outputs["View Distance"], fade[0], fade[1])); g.put(fm.inputs[1], out_)
        g.put(fm.inputs[2], e2.outputs[0]); out_ = fm.outputs[0]
    o = g.n("ShaderNodeOutputMaterial"); g.put(o.inputs["Surface"], out_)
    return m


def bake_far(sc, at, clip=500.0, face=1024, samples=16, out=None, hide=()):
    """The far terrain as an image of its own scene: 6 Eevee cube faces (90°, linear EXR) from the point at, anything closer than clip
    is cut off (only the far terrain and the sky); stitched into an equirectangular 2:1 (like Environment Texture). → a Blender image
    (float) and seconds."""
    from scipy import ndimage
    t0 = time.time(); keep_cam, keep_res = sc.camera, (sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage)
    keep_fmt = (sc.render.image_settings.file_format, sc.render.image_settings.color_depth); keep_s = sc.eevee.taa_render_samples
    keep_h = [(o, o.hide_render) for o in hide]
    for o in hide: o.hide_render = True                                 # hide the near part: the Eevee clip works by depth, not by distance
    cd = bpy.data.cameras.new("bake_cam"); cd.angle = math.pi / 2; cd.clip_start, cd.clip_end = (1.0 if hide else clip), 1.2e5; cd.sensor_fit = "AUTO"
    co = bpy.data.objects.new("bake_cam", cd); sc.collection.objects.link(co); co.location = at
    sc.render.resolution_x = sc.render.resolution_y = face; sc.render.resolution_percentage = 100; sc.eevee.taa_render_samples = samples
    sc.render.image_settings.file_format = "OPEN_EXR"; sc.render.image_settings.color_depth = "32"   # the sun is brighter than the half-float limit
    faces = [((1, 0, 0), (0, 0, 1)), ((-1, 0, 0), (0, 0, 1)), ((0, 1, 0), (0, 0, 1)), ((0, -1, 0), (0, 0, 1)), ((0, 0, 1), (0, -1, 0)), ((0, 0, -1), (0, 1, 0))]
    out = out or bpy.app.tempdir; Rs, imgs = [], []
    for k, (f, u) in enumerate(faces):
        Zc = -np.array(f, float); Yc = np.array(u, float); Xc = np.cross(Yc, Zc); R = np.stack([Xc, Yc, Zc], 1)
        from mathutils import Matrix
        co.matrix_world = Matrix.Translation(at) @ Matrix([list(R[0]) + [0], list(R[1]) + [0], list(R[2]) + [0], [0, 0, 0, 1]])
        sc.camera = co; fp = os.path.join(out, f"far_face_{k}.exr"); sc.render.filepath = fp
        bpy.ops.render.render(write_still=True)
        im = bpy.data.images.load(fp); a = np.empty(face * face * 4, np.float32); im.pixels.foreach_get(a); bpy.data.images.remove(im)
        Rs.append(R); imgs.append(a.reshape(face, face, 4))
    H, W_ = 2 * face, 4 * face
    v = (np.arange(H) + 0.5) / H; u_ = (np.arange(W_) + 0.5) / W_
    lat = (v - 0.5) * math.pi; lon = (0.5 - u_) * 2 * math.pi
    LA, LO = np.meshgrid(lat, lon, indexing="ij"); D = np.stack([np.cos(LA) * np.cos(LO), np.cos(LA) * np.sin(LO), np.sin(LA)], -1)
    pano = np.zeros((H, W_, 4), np.float32); best = np.full((H, W_), -1.0)
    for R, a in zip(Rs, imgs):
        c = D @ R; dep = -c[..., 2]; uu, vv = c[..., 0] / np.maximum(dep, 1e-6), c[..., 1] / np.maximum(dep, 1e-6)
        ok = (dep > best) & (np.abs(uu) <= 1.0001) & (np.abs(vv) <= 1.0001)
        if not ok.any(): continue
        col, row = (uu[ok] + 1) / 2 * face - 0.5, (vv[ok] + 1) / 2 * face - 0.5
        for ch in range(4): pano[..., ch][ok] = ndimage.map_coordinates(a[..., ch], [row, col], order=1, mode="nearest")
        best[ok] = dep[ok]
    pano[..., 3] = 1.0
    img = bpy.data.images.new("far_pano", W_, H, alpha=True, float_buffer=True); img.pixels.foreach_set(pano.ravel())
    img.pack()
    if hasattr(img, "use_half_precision"): img.use_half_precision = True   # into video memory — half precision (8k × 4k: 256 MB, not 512)
    sc.camera = keep_cam; sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = keep_res
    sc.render.image_settings.file_format, sc.render.image_settings.color_depth = keep_fmt; sc.eevee.taa_render_samples = keep_s
    bpy.data.objects.remove(co)
    for o, h_ in keep_h: o.hide_render = h_
    return img, time.time() - t0


def world_image(sc, img, strength=1.0, sun_threshold=10.0):
    """World — the panorama from bake_far: background, reflections and ambient light; Eevee extracts the sun from the bright part (as with a photo)."""
    w = sc.world; nt = w.node_tree; nt.nodes.clear()
    en = nt.nodes.new("ShaderNodeTexEnvironment"); en.image = img; en.interpolation = "Linear"
    bg = nt.nodes.new("ShaderNodeBackground"); bg.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputWorld"); nt.links.new(en.outputs[0], bg.inputs[0]); nt.links.new(bg.outputs[0], out.inputs[0])
    w.sun_threshold = sun_threshold


def scene_lake_peaks(sc, kv):
    """The "far terrain" test: a mountain lake in the Maroon Bells valley (3DEP window 90 km, 44 m) — mountains as their own polar grid
    around the camera (an equal number of points per degree), the sky is a pure sky panorama. far=geo — the mountains as a grid in
    the frame; far=pano — the mountains beyond the cutoff (2.5 km: past the far shore) and the sky are baked once (6 faces) into the
    frame's world, the far grid is hidden, the near ring and the water are live."""
    sky = SKIES[kv.get("sky_kind", "kloofendal_day")]; sun_az, sun_el = sky["sun"]
    world_hdr(sc, sky["file"], strength=float(kv.get("sky", 1.0)), rot_deg=float(kv.get("turn", 0.0)))
    az = float(kv.get("azimuth", 209.0)); fx, fy = math.cos(math.radians(az)), math.sin(math.radians(az))
    off = float(kv.get("shore", 650.0)); cx, cy = -off * fx, -off * fy  # the camera is on the near shore of the basin (650 m from the middle)
    t0 = time.time(); P, h, rr = peaks(center=(cx, cy)); print(f"mountains: grid {P.shape[0]}×{P.shape[1]}, {time.time() - t0:.1f} s", flush=True)
    hz = lambda x, y: h(np.asarray(x) + cx, np.asarray(y) + cy)
    def ground(x, y):                                                   # near shore: pebbles toward the water, by 60–70 m it becomes the window relief
        x, y = np.asarray(x, float), np.asarray(y, float); d = x * fx + y * fy; r = np.hypot(x, y)
        near = np.maximum(0.25 - 0.15 * (d - 3.0) + 0.06 * np.sin(x * 0.45) * np.sin(y * 0.3), -25.0)
        far_ = hz(x, y); k = _ss(r, 55.0, 66.0)
        return near * (1 - k) + far_ * k
    cam = camera(sc, (0.0, 0.0, float(kv.get("height", 1.6))), az, float(kv.get("tilt", 2.0)), lens=float(kv.get("lens", 30)))
    cam.data.clip_end = 1.0e5
    ground_mesh(sc, lambda x, y: np.where(np.hypot(x, y) < 68.0, ground(x, y), ground(x, y) - 40.0), -70.0, 70.0, -70.0, 70.0, 0.08,
                tex="Gravel043", tile=1.6, dark=0.55)                   # circle of 68 m (the corners lie under the ring)
    Pw = P.copy(); Pw[..., 0] -= cx; Pw[..., 1] -= cy
    clip = float(kv.get("cutoff", 2500.0)); j = int(np.searchsorted(rr, clip))
    pm = peaks_material(snow=float(kv.get("snow", 1000.0)))
    pano = kv.get("far", "geo") == "pano"
    pmid = peaks_material(snow=float(kv.get("snow", 1000.0)), fade=(0.86 * clip, clip) if pano else None)
    rings = []
    jf = int(np.searchsorted(rr, 0.86 * clip)) if pano else j           # far terrain — from where the ring starts to dissolve (overlapping: the far terrain lies under it)
    for nm, sl, m_ in (("peaks_mid", slice(0, j + 2), pmid), ("peaks_far", slice(jf, None), pm)):
        o = ground_mesh(sc, None, 0, 0, 0, 0, 0, name=nm, grid=Pw[:, sl], tex="Gravel043"); o.data.materials.clear(); o.data.materials.append(m_)
        rings.append(o)
    rc = (np.hypot(cx, cy))
    def lvl(x, y):                                                       # water only in the lake basin (not in other lowlands of the window)
        x, y = np.asarray(x, float), np.asarray(y, float)
        return np.where((ground(x, y) < -0.02) & (np.hypot(x + cx, y + cy) < 1900.0), 0.0, np.nan)
    L = sun_vec(sun_az, sun_el)
    sea = W.Sea(u10=float(kv.get("wind", 2.5)), wind_to=(fx, fy), fetch=1500.0, depth=25.0, near=float(kv.get("closer", near_for(cam, sc))),
                seed=int(kv.get("seed", 4)))
    look = optics.preset(kv.get("water", "mountain_lake"))
    gust = {"scale": 45.0, "floor": 0.06, "slick": 0.5, "edge": (0.5, 0.64)}
    wat = Water(sc, sea, cam=cam, grid=(int(kv.get("grid", 600)), int(int(kv.get("grid", 600)) * 0.75)), body=look["body"], K=look["K"],
                ground=ground, scatter=(0.02, 0.12, 0.1), sun=(tuple(L), 4.0), debug=kv.get("kind"), gust=gust, level_fn=lvl, far=4000.0)
    if pano:                                                            # the mountains beyond the cutoff and the sky go into the frame's world once
        near_objs = [o for o in sc.objects if o.type == "MESH" and o is not rings[1]]
        img, sec = bake_far(sc, tuple(cam.location), clip=clip, face=int(kv.get("face", 2048)), hide=near_objs)
        print(f"far terrain baked: {sec:.1f} s", flush=True)
        rings[1].hide_render = True; world_image(sc, img, strength=1.0)
        nd = pmid.node_tree.nodes                                       # the ring edge — to the panorama color from the bake point
        nd["pano_lookup"].image = img; bp_ = nd["bake_point"]
        for k_, v_ in zip(("X", "Y", "Z"), tuple(cam.location)): bp_.inputs[k_].default_value = float(v_)
        if kv.get("pano_file"): img.filepath_raw = kv["pano_file"]; img.file_format = "OPEN_EXR"; img.save()
    return wat, sea


def haze(sc, visibility, rgb=(0.62, 0.66, 0.7), level=0.18, H=120.0, size=6000.0, center=(0.0, 0.0)):
    """Spray and drizzle haze — a layer at the water: a box volume, density β·e^(−z/H), β = 3.912 / visibility (Koschmieder);
    absorption + emission β·L (an Eevee volume is not lit by the sky — without emission the haze is black): far away at the water
    the brightness tends to L, and the sky above the layer is visible."""
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(center[0], center[1], 2.5 * H))
    box = bpy.context.object; box.name = "haze"; box.scale = (size, size, 5.0 * H); box.visible_shadow = False
    m = bpy.data.materials.new("haze"); nt = m.node_tree; nt.nodes.clear(); L = nt.links.new
    tc = nt.nodes.new("ShaderNodeTexCoord"); sp = nt.nodes.new("ShaderNodeSeparateXYZ"); L(tc.outputs["Object"], sp.inputs[0])
    z = nt.nodes.new("ShaderNodeMath"); z.operation = "MULTIPLY_ADD"; z.inputs[1].default_value = 5.0 * H; z.inputs[2].default_value = 2.5 * H
    L(sp.outputs[2], z.inputs[0])                                       # object-space −0.5…0.5 → meters above the water
    fall = nt.nodes.new("ShaderNodeMath"); fall.operation = "MULTIPLY"; fall.inputs[1].default_value = -1.0 / H; L(z.outputs[0], fall.inputs[0])
    ex = nt.nodes.new("ShaderNodeMath"); ex.operation = "EXPONENT"; L(fall.outputs[0], ex.inputs[0])
    beta = 3.912 / max(visibility, 50.0)
    dn = nt.nodes.new("ShaderNodeMath"); dn.operation = "MULTIPLY"; dn.inputs[1].default_value = beta; L(ex.outputs[0], dn.inputs[0])
    de = nt.nodes.new("ShaderNodeMath"); de.operation = "MULTIPLY"; de.inputs[1].default_value = level; L(dn.outputs[0], de.inputs[0])
    ab = nt.nodes.new("ShaderNodeVolumeAbsorption"); ab.inputs["Color"].default_value = (0, 0, 0, 1); L(dn.outputs[0], ab.inputs["Density"])
    em = nt.nodes.new("ShaderNodeEmission"); em.inputs["Color"].default_value = (*rgb, 1.0); L(de.outputs[0], em.inputs["Strength"])
    ad = nt.nodes.new("ShaderNodeAddShader"); L(ab.outputs[0], ad.inputs[0]); L(em.outputs[0], ad.inputs[1])
    out = nt.nodes.new("ShaderNodeOutputMaterial"); L(ad.outputs[0], out.inputs["Volume"])
    box.data.materials.append(m)
    e = sc.eevee; e.volumetric_start, e.volumetric_end = 1.0, 5000.0; e.volumetric_samples = 48; e.volumetric_sample_distribution = 0.9
    return box


def scene_storm(sc, kv):
    """Storm: the camera on the bridge (15 m, looking down 12°, like the NOAA Beaufort scale photos) facing the waves; the force → wind
    and height (a table), steeper crests, foam of three ages and streaks along the wind (7+), spindrift torn from the crests (8+),
    spray haze (9+); the sky is solid overcast."""
    sky = SKIES[kv.get("sky_kind", "overcast")]
    world_hdr(sc, sky["file"], strength=float(kv.get("sky", 0.55)), sun_threshold=1e5)
    b = float(kv.get("force", 10))
    u = float(np.interp(b + 0.5, range(len(W.BEAUFORT_U)), W.BEAUFORT_U)); hs = W.beaufort_hs(u)
    az = float(kv.get("azimuth", 90.0))
    cam = camera(sc, (0.0, 0.0, float(kv.get("height", 15.0))), az, float(kv.get("tilt", -12.0)), lens=float(kv.get("lens", 28)))
    to = (-math.cos(math.radians(az + 15)), -math.sin(math.radians(az + 15)))   # the wind and waves come toward the camera, slightly at an angle
    sea = W.Sea(u10=u, wind_to=to, hs=hs, chop=float(kv.get("sharpness", 1.25)), near=float(kv.get("closer", near_for(cam, sc))),
                seed=int(kv.get("seed", 7)))
    print(f"storm: force {b:.0f}, wind {u:.1f} m/s, Hs {sea.stats()['hs']:.1f} m, Tp {sea.stats()['tp']:.1f} s, whitecaps {sea.stats()['whitecaps'] * 100:.1f} %", flush=True)
    look = optics.preset(kv.get("water", "ocean_blue"))
    wat = Water(sc, sea, cam=cam, grid=(int(kv.get("grid", 620)), int(int(kv.get("grid", 620)) * 0.75)), body=look["body"],
                scatter=(0.03, 0.18, 0.15), sun=None, foam_rgb=(0.84, 0.86, 0.87), debug=kv.get("kind"))
    if b >= 8:
        from components.water.bl_spray import Spray
        sp = Spray(wat); old = wat.update
        def upd(t, _o=old, _s=sp):
            D = _o(t); _s.update(t); return D
        wat.update = upd
    if b >= 9: haze(sc, float(kv.get("visibility", 2400.0 - 300.0 * (b - 9))))
    return wat, sea


DECK = {   # sea states for the deck view: wind (force → m/s), swell {hs, tp, angle to the wind waves °}, sky, the rogue wave focus
    "swell":    dict(b=3, swell=(3.5, 14.0, 0.0), sky="qwantani_sunset", exp=-1.4, note="long swell from a distant storm under a light wind"),
    "wind6":  dict(b=6, sky="kloofendal_day", exp=-1.0, note="wind sea at force 6: whitecaps everywhere"),
    "cross_sea":  dict(b=7, swell=(3.5, 12.0, 75.0), sky="misty_morning", exp=-0.8, note="cross sea: swell across the wind waves"),
    "storm10": dict(b=10, sky="overcast", exp=0.3, note="storm, force 10"),
    "storm12": dict(b=12, sky="overcast", exp=0.4, note="hurricane, force 12"),
    "rogue":  dict(b=9, focus=1.7, sky="overcast", exp=0.3, note="rogue wave at force 9: the group converges 200 m ahead of the ship"),
}


def scene_deck(sc, kv):
    """View from the deck: a ship ~70 m long heading into the waves, the bridge 9 m above the water; the pitch and roll follow the water
    under the hull (heave — the mean height under the hull points, trim — bow and stern, roll — the sides; the hull averages out
    short waves). The sea state — waves= from DECK; in "rogue" the crest converges at a point 200 m ahead of the bow at the moment
    t_f (a frame — at t_f, a clip — before and after)."""
    st = DECK[kv.get("waves", "storm10")]
    if "exposure" not in kv: sc.view_settings.exposure = st["exp"]
    sky = SKIES[kv.get("sky_kind", st["sky"])]; sun_az, sun_el = sky["sun"]
    storm = st["b"] >= 8
    world_hdr(sc, sky["file"], strength=float(kv.get("sky", 0.55 if storm else 1.0)), sun_threshold=1e5 if storm else 10.0)
    b_ = float(kv.get("force", st["b"])); u = float(np.interp(b_ + 0.5, range(len(W.BEAUFORT_U)), W.BEAUFORT_U))
    hs = W.beaufort_hs(u) if b_ >= 7 else None
    az = float(kv.get("azimuth", sun_az + 180.0 if st.get("sky") == "qwantani_sunset" else 90.0))   # the swell comes toward the sunset: the glitter path is in the frame
    lk = (math.cos(math.radians(az)), math.sin(math.radians(az))); to = (-lk[0], -lk[1])             # the waves head toward the ship
    deck = float(kv.get("height", 9.0)); view = kv.get("view", "deck")
    rt0 = (lk[1], -lk[0]); fx_, fy_ = 200.0 * lk[0], 200.0 * lk[1]       # the rogue wave focus point — 200 m ahead of the bridge
    if view == "deck":
        cam = camera(sc, (0.0, 0.0, deck), az, float(kv.get("tilt", -3.0)), lens=float(kv.get("lens", 45)))
    else:                                                               # three_quarters — front-side, low; helicopter — from above and the side
        back, side, hz_ = {"three_quarters": (250.0, 250.0, 15.0), "helicopter": (350.0, 200.0, 80.0)}[view]
        px, py = fx_ - back * lk[0] + side * rt0[0], fy_ - back * lk[1] + side * rt0[1]
        hz_ = float(kv.get("height", hz_)); dx_, dy_ = fx_ - px, fy_ - py
        cam = camera(sc, (px, py, hz_), math.degrees(math.atan2(dy_, dx_)), float(kv.get("tilt", math.degrees(math.atan2(4.0 - hz_, math.hypot(dx_, dy_))))),
                     lens=float(kv.get("lens", 40 if view == "helicopter" else 50)))
    sw = None
    if st.get("swell"):
        h_, tp_, ang = st["swell"]; a_ = math.atan2(to[1], to[0]) + math.radians(ang)
        sw = {"hs": h_, "tp": tp_, "to": (math.cos(a_), math.sin(a_)), "xi": 1.4}
    t_f = float(kv.get("t_f", kv.get("t", 8.0)))
    foc = {"xy": (fx_, fy_), "t": t_f, "k": float(st["focus"]), "spread": float(kv.get("focus_angle", 28.0))} if st.get("focus") else None
    sea = W.Sea(u10=u, wind_to=to, hs=hs, fetch=None if hs else float(kv.get("accel", 2e5)), swell=sw, chop=float(kv.get("sharpness", 1.6 if b_ >= 6 else 1.1)),
                near=float(kv.get("closer", near_for(cam, sc))), seed=int(kv.get("seed", 7)), focus=foc,
                mix=float(kv.get("scattered", 0.92)), smul=float(kv.get("taper", 1.3)))   # crests in rows, but overlapping
    s_ = sea.stats(); print(f"deck: {kv.get('waves', 'storm10')} — {st['note']}; wind {u:.1f} m/s, Hs {s_['hs']:.1f} m, Tp {s_['tp']:.1f} s"
                            + (f"; focus {sea.focus}" if sea.focus else ""), flush=True)
    look = optics.preset(kv.get("water", "ocean_blue" if not storm else "ocean"))
    L = sun_vec(sun_az, sun_el)
    Lb = Vector((lk[0], lk[1], 0.25)).normalized()                      # overcast: the light through the crests comes from the bright sky ahead
    wat = Water(sc, sea, cam=cam, grid=(int(kv.get("grid", 720)), int(int(kv.get("grid", 720)) * 0.75)), body=look["body"],
                scatter=(0.03, 0.2, 0.16), sun=(tuple(Lb), float(kv.get("clearance", 0.9))) if storm else (tuple(L), 4.0),
                foam_rgb=(0.86, 0.88, 0.89), debug=kv.get("kind"))
    wat.margin = max(wat.margin, 0.35)
    if u >= 17.2:
        from components.water.bl_spray import Spray
        sp = Spray(wat); old = wat.update
        def upd(t, _o=old, _s=sp):
            D = _o(t); _s.update(t); return D
        wat.update = upd
    if u >= 24.5: haze(sc, float(kv.get("visibility", 2400.0 - 300.0 * (b_ - 9))))
    Lh, Bh = float(kv.get("length_ship", 70.0)), 14.0                   # hull: points along the length and width; the bridge is at 0.3 of the length toward the stern
    al = np.linspace(-0.5, 0.5, 7) * Lh; ac = np.linspace(-0.5, 0.5, 3) * Bh
    A_, C_ = np.meshgrid(al, ac); hull = np.stack([A_.ravel(), C_.ravel()], 1)
    rt = (lk[1], -lk[0]); ox, oy = 0.3 * Lh * lk[0], 0.3 * Lh * lk[1]  # the hull middle is ahead of the bridge
    pts = np.stack([ox + hull[:, 0] * lk[0] + hull[:, 1] * rt[0], oy + hull[:, 0] * lk[1] + hull[:, 1] * rt[1]], 1)
    base = cam.rotation_euler.copy()
    def rig(t, F, _cam=cam):
        z, _ = W.surface_at(sea, F, pts, iters=2); z = z.reshape(A_.shape)
        heave = float(z.mean()); pitch = math.atan2(float(z[:, -1].mean() - z[:, 0].mean()), Lh) * 0.8
        roll = math.atan2(float(z[0].mean() - z[-1].mean()), Bh) * 0.5
        _cam.location.z = deck + 0.9 * heave
        _cam.rotation_euler = (base.x + pitch, base.y + roll, base.z)
    if view == "deck": wat.on_frame = rig                            # from the deck — pitching; other angles — the camera stays in place
    return wat, sea


SCENES = {"sea": scene_sea, "lake": scene_lake, "pond": scene_pond, "beach": scene_beach, "river": scene_river, "storm": scene_storm,
          "rapids": scene_rapids, "fall": scene_fall, "lake_peaks": scene_lake_peaks, "deck": scene_deck}


def main():
    out, kv = args(); os.makedirs(out, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    scene_base(sc, int(kv.get("width", 1600)), int(kv.get("samples", 48)))
    sc.view_settings.exposure = float(kv.get("exposure", -1.6))
    t0 = time.time()
    wat, sea = SCENES[kv.get("scene", "sea")](sc, kv)
    if kv.get("without_water") == "1": wat.obj.hide_render = True               # debug: a scene without water
    if kv.get("objects") == "1":                                        # debug: the scene meshes — vertices and heights
        for o in sc.objects:
            if o.type == "MESH" and len(o.data.vertices):
                co = np.empty(len(o.data.vertices) * 3); o.data.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
                print(f"OBJECT {o.name}: {len(co)} vertices, z {co[:, 2].min():.2f}…{co[:, 2].max():.2f}, x {co[:, 0].min():.0f}…{co[:, 0].max():.0f}, "
                      f"y {co[:, 1].min():.0f}…{co[:, 1].max():.0f}, render {not o.hide_render}", flush=True)
    info = {"stats": sea.stats(), "foam_thr": sea.fp["thr"], "verts": len(wat.base), "s_build": round(time.time() - t0, 2)}
    n, fps, t_ = int(kv.get("frames", 0)), float(kv.get("fps", 12)), float(kv.get("t", 6.0))
    times = [t_] if n <= 0 else [t_ + i / fps for i in range(n)]
    cost = []
    for i, t in enumerate(times):
        a = time.time(); wat.update(t)
        if getattr(wat, "on_frame", None): wat.on_frame(t, wat.last)
        b = time.time()
        sc.render.filepath = os.path.join(out, "still.png" if n <= 0 else f"f_{i:04d}.png")
        bpy.ops.render.render(write_still=True); c = time.time()
        cost.append((round(b - a, 3), round(c - b, 2)))
        print(f"frame {i}: water {b - a:.2f} s, render {c - b:.1f} s", flush=True)
    info["cost"] = cost
    json.dump(info, open(os.path.join(out, "stand.json"), "w"), ensure_ascii=False, indent=1, default=float)
    print("__RESULT__" + json.dumps({"out": out, "frames": len(times), "s_build": info["s_build"], "hs": round(info["stats"]["hs"], 2)}))


main()
