"""Weather as a layer (class V4, loops and effects): rain or snow with the same camera, a mask per frame — the scene is black and occludes the particles,
the particles are white. The emitter follows the camera (COPY_LOCATION); the rain and snow parameters are ranges (the presets are below).
blender -b --factory-startup -P core/blender/run.py -- components/light/bl_weather.py -- <bench/scenes/X> <camera K> <rain|snow> <output folder> [n_frames=8] [strength=0.5]
Overlaying on a picture needs no separate layer step: the mask is laid on with "screen" (lighten) blending, or goes to the model as a hint."""
import json, math, os, sys
import bpy
from mathutils import Vector
from core.blender import bl_scene as B

a = sys.argv[sys.argv.index("--") + 1:]
SD, K, KIND, OUT = os.path.abspath(a[0]), int(a[1]), a[2], os.path.abspath(a[3])
N = int(a[4]) if len(a) > 4 else 8; POWER = float(a[5]) if len(a) > 5 else 0.5     # strength 0..1 — the position within the range of their density
os.makedirs(OUT, exist_ok=True)
sc, statics, members, objs, W, kit = B.load_scene(SD)
cam = B.make_camera(B.load_cameras(SD)[K - 1], f"cam{K}", sc)
fps = 24
# Presets: rain — 0.05–1 drops/s/m², speed 9–20 m/s, lifetime 100 frames, drop 1–1.5 cm; snow — 2–26 per m², lifetime 2000.
# Size is what is visible in the shot: a drop with exposure is a 3 cm × 0.8 m streak (a 1–1.5 cm drop is under a pixel at 10 m and would not be drawn)
P = {"rain": dict(per_m2=(0.05 + 0.95 * POWER) * 100, speed=9 + 11 * POWER, life=100, size=0.03, warm=100, grav=0.0, streak=0.04),
     "snow": dict(per_m2=2 + 24 * POWER, speed=1.0, life=2000, size=0.04, warm=300, grav=0.05, streak=1.0)}[KIND]
side = 60.0; height = 25.0
bpy.ops.mesh.primitive_plane_add(size=side, location=cam.matrix_world.translation + Vector((0, 0, height)))
em = bpy.context.active_object; em.name = "weather_emitter"; em.rotation_euler = (math.pi, 0, 0)       # normal pointing down — the particles fly down
con = em.constraints.new("COPY_LOCATION"); con.target = cam; con.use_offset = True
em.location = Vector((0, 0, height))
bpy.ops.mesh.primitive_uv_sphere_add(segments=6, ring_count=4, radius=1.0); drop = bpy.context.active_object; drop.name = "weather_drop"
drop.scale = (1, 1, 1 / P["streak"]) if KIND == "rain" else (1, 1, 1);   # rain falls straight down: a streak along the world vertical, with no particle rotation
drop.hide_render = False; drop.location = (0, 0, -1000)
ps = em.modifiers.new("weather", "PARTICLE_SYSTEM").particle_system.settings
ps.count = int(P["per_m2"] * side * side * (0.08 if KIND == "rain" else 0.5));   # a 0.75 m streak is more visible than a 1 cm drop — lower density
ps.frame_start = -P["warm"]; ps.frame_end = N + 1; ps.lifetime = P["life"]
ps.emit_from = "FACE"; ps.normal_factor = P["speed"]; ps.effector_weights.gravity = P["grav"]
ps.render_type = "OBJECT"; ps.instance_object = drop; ps.particle_size = P["size"]; ps.size_random = 0.5
ps.use_rotations = False;   # "along velocity" in 5.2 gives streaks radiating from the centre of the frame
ps.mass = 0.001
em.show_instancer_for_render = False
# mask: Workbench with the object's flat colour — the scene is black (occludes the particles), the particles are white
for o in objs: o.color = (0, 0, 0, 1)
drop.color = (1, 1, 1, 1)
B.flat_workbench(sc)
sc.display.shading.background_type = "VIEWPORT"; sc.display.shading.background_color = (0, 0, 0)
sc.frame_start, sc.frame_end = -P["warm"], N
pc = em.particle_systems[0].point_cache; pc.frame_start, pc.frame_end = -P["warm"], N
with bpy.context.temp_override(scene=sc, point_cache=pc):           # particle cache: without baking, the render simulates them afresh and comes out empty
    bpy.ops.ptcache.bake(bake=True)
files = []
for f in range(1, N + 1):
    sc.frame_set(f); p = os.path.join(OUT, f"{KIND}_{f:04d}.png"); B.render_to(sc, p); files.append(p)
json.dump({"kind": KIND, "camera": K, "frames": N, "count": ps.count, "preset": P}, open(os.path.join(OUT, "weather.json"), "w"), ensure_ascii=False)
print("__RESULT__" + json.dumps({"files": len(files), "count": ps.count, "out": OUT}, ensure_ascii=False))
