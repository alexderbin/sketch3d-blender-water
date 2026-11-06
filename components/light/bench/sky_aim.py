"""Sky-image probe for the reference (bench.py, check 17): the plan's world (bl_far.build_world), a camera at the origin
aimed at the node's sun — by how many degrees from the frame centre the brightest spot of the picture lies (the equirectangular convention
and the picture's rotation: the node's lamp and the sun in the picture must coincide).
blender -b --factory-startup -P core/blender/run.py -- components/light/bench/sky_aim.py -- <plan.json> <folder>   → line __RESULT__ {"displacement": °}"""
import json, math, os, sys
import bpy
from core.light import bl_far
import numpy as np
from mathutils import Vector

PL, OUT = sys.argv[sys.argv.index("--") + 1:][:2]
P = json.load(open(PL)); P["path"] = PL
sc = bpy.context.scene
for o in list(sc.objects): bpy.data.objects.remove(o)               # the default scene has a cube at the origin — the camera would be inside it
sc.world = bl_far.build_world(P); sc.render.engine = "BLENDER_EEVEE"
N = 128; sc.render.resolution_x = sc.render.resolution_y = N; sc.render.resolution_percentage = 100
sc.render.image_settings.file_format = "OPEN_EXR"; sc.render.image_settings.color_depth = "32"
cd = bpy.data.cameras.new("aim"); cd.lens = 24.0; cd.sensor_width = 36.0; cam = bpy.data.objects.new("aim", cd)
sc.collection.objects.link(cam); sc.camera = cam
cam.rotation_euler = Vector(P["sun"]["dir"]).to_track_quat("-Z", "Y").to_euler()
p = os.path.join(OUT, "sky_aim.exr"); sc.render.filepath = p; bpy.ops.render.render(write_still=True)
im = bpy.data.images.load(p); a = np.array(im.pixels[:], np.float32).reshape(N, N, 4)[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
y, x = np.unravel_index(int(np.argmax(a)), a.shape); f = N * 24.0 / 36.0
print("__RESULT__" + json.dumps({"displacement": round(float(math.degrees(math.atan(math.hypot(x - (N - 1) / 2, y - (N - 1) / 2) / f))), 2)}))
