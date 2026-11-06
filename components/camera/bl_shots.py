"""Shot film cameras rebuilt — without rebuilding the scene (components/battle/bl_battle.py builds them the same way): the old shot cameras,
their look-at points and markers are deleted; the new ones come from the plan's anim_cfg.shots or from the camera solver's shots file
(components/camera/battle_cam.py: edit <out>/cams/edit_<edit>_<fmt>.json — shots in a row, format 9:16 | 16:9) through tools/bl_crowd.shot_camera.
For shots with a format the scene resolution follows it (9:16 → 1080×1920; the trailer render takes its own res=). The file is saved in place
(file=<path> — as a copy there, the original stays intact); the markers in it are not bound to cameras (other renders of the file use their own
camera). look=1 — one more copy <file>_view.blend with bound markers: in Blender the camera switches by itself along the timeline (for viewing).
Comparison: the camera at the shot's keys is where the task path is (< 1 cm), a marker at the start of each shot — the "comparison" answer.
blender -b <scene.blend> -P core/blender/run.py -- components/camera/bl_shots.py -- <battle.yaml plan | shots.json> [file=<copy.blend>] [look=1]"""
import json, os, sys
import bpy
from mathutils import Vector
from bpy_extras import anim_utils
from core.blender import bl_kit as C

a = sys.argv[sys.argv.index("--") + 1:]; plan = a[0]; opt = dict(x.split("=", 1) for x in a[1:]); sc = bpy.context.scene
if plan.endswith(".json"): shots = json.load(open(plan))               # the solver's shots (the edit) or an export of the plan
else:
    try:
        import yaml; shots = yaml.safe_load(open(plan).read())["anim_cfg"]["shots"]
    except ImportError:                                                # Blender's Python has no PyYAML — the plan is read by the launching side
        shots = json.load(open(plan + ".shots.json"))
names = ({sh["name"] for sh in shots} | {m.name for m in sc.timeline_markers if m.camera}
         | {o.name.split(".")[0] for o in bpy.data.objects if o.type == "CAMERA" and o.get("shot")})
for m in list(sc.timeline_markers):
    if m.camera or m.name in names: sc.timeline_markers.remove(m)
for o in [o for o in bpy.data.objects if o.name.split(".")[0] in names or o.name.split(".")[0] in {n + "_aim" for n in names}]: bpy.data.objects.remove(o)
coll = bpy.data.collections.get("anim") or sc.collection
cams = [C.shot_camera(sc, coll, sh, phase=11.0 * i) for i, sh in enumerate(shots)]
fm = {sh.get("fmt") for sh in shots}
if len(fm) == 1 and None not in fm:                                    # the resolution follows the edit's format
    sc.render.resolution_x, sc.render.resolution_y = {"9:16": (1080, 1920), "16:9": (1920, 1080)}[fm.pop()]


def fc(idb):
    ad = idb.animation_data
    cb = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
    return {(f.data_path, f.array_index): f for f in cb.fcurves}


dev, lens_dev = 0.0, 0.0                                               # comparison: the camera at the keys is a point of the task path
for cam, sh in zip(cams, shots):
    c_ = fc(cam); a_, b_ = sh["frames"]; n = len(sh["path"])
    for i, p in enumerate(sh["path"]):
        f = a_ + (b_ - a_) * i / (n - 1); dev = max(dev, (Vector([c_[("location", j)].evaluate(f) for j in range(3)]) - Vector(p)).length)
    if sh.get("lenses") and cam.data.animation_data:
        lc = fc(cam.data)
        if ("lens", 0) in lc: lens_dev = max(lens_dev, abs(lc[("lens", 0)].evaluate(a_) - sh["lenses"][0]))
starts = sorted(m.frame for m in sc.timeline_markers if m.name in {sh["name"] for sh in shots})
over = [(x["name"], y["name"]) for x, y in zip(sorted(shots, key=lambda s: s["frames"][0]), sorted(shots, key=lambda s: s["frames"][0])[1:]) if y["frames"][0] <= x["frames"][1]]
res = {"shots": [sh["name"] for sh in shots], "fmt": sorted({sh.get("fmt") or "plan" for sh in shots}), "frames": [min(s["frames"][0] for s in shots), max(s["frames"][1] for s in shots)],
       "comparison": {"camera_m": round(dev, 4), "lens_mm": round(lens_dev, 3), "n_markers": len(starts), "markers_on_starts": starts == sorted(sh["frames"][0] for sh in shots),
                  "overlays": over}}
out = opt.get("file")
if out: bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out), copy=True); res["file"] = os.path.abspath(out)
else: bpy.ops.wm.save_mainfile(); res["file"] = bpy.data.filepath
if opt.get("look") == "1":
    C.bind_shot_markers(sc); view = (out and os.path.abspath(out) or bpy.data.filepath)[:-6] + "_view.blend"
    bpy.ops.wm.save_as_mainfile(filepath=view, copy=True); res["view"] = view
print("__RESULT__" + json.dumps(res, ensure_ascii=False))
