"""Camera check: a sketch25 scene → Blender, render an id map (Workbench, flat colour) and depth (Eevee, EXR Z pass).
Run: blender -b -P core/blender/run.py -- components/camera/bl_cam_check.py -- <scene folder> <camera 1..> <output folder> [ss_scale]"""
import json
import os
import sys

import bpy
from core.blender import bl_scene as B

scene_dir, k, out = sys.argv[sys.argv.index("--") + 1:][:3]
k = int(k)
os.makedirs(out, exist_ok=True)

sc = B.reset()
objs = B.import_obj(os.path.join(scene_dir, "scene.obj"))
objs.append(B.add_far_ground(objs))
cams = B.load_cameras(scene_dir)
B.make_camera(cams[k - 1], f"cam{k}", sc)

# 1. id map
table = B.paint_ids(objs)
B.flat_workbench(sc)
B.render_to(sc, os.path.join(out, f"cam{k}-id.png"))
json.dump(table, open(os.path.join(out, f"cam{k}-idmap.json"), "w"), ensure_ascii=False)

# 2. depth and normals: Eevee, multilayer EXR
sc.render.engine = "BLENDER_EEVEE"
vl = sc.view_layers[0]
vl.use_pass_z = True
vl.use_pass_normal = True
sc.render.image_settings.media_type = "MULTI_LAYER_IMAGE"
sc.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
sc.render.image_settings.color_depth = "32"
sc.render.filepath = os.path.join(out, f"cam{k}-passes.exr")
bpy.ops.render.render(write_still=True)
print("DONE", out)
