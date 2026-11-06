"""Camera from someone else's picture, without a neural network ("fit the camera to someone else's picture"): the field of view, the horizon row and the pixels
of the hero's head and feet in the picture → a scene camera in which our hero stands in the same place and has the same height.
The tilt comes from the horizon (the vanishing points of the ground), the camera's distance and height from the hero ruler (d = f·H/h; more
accurate than a ray to the floor), the heading is the side from which the camera sees the hero (--az from the hero's gaze; a pose from the
picture gives view_yaw), the shift along the width comes from the column of the feet. Field of view: --fov (default 60°; a MoGe-2 estimate
only on explicit request).
.venv/bin/python -m components.camera.cam_from_picture scenes/<name> <hero> <picture> --horizon <y> --feet <u,v> --head <u,v> [--fov 60]
        [--az 0] [--name …] [--dry]
  pixels are in picture coordinates (y down); head — the top of the head, feet — the soles; hero — a figure id from figkit.json (the height comes from there).
  --ref-height <m> — a ruler from an object of known height instead of the hero (--head and --feet are its top and bottom, it stands where the
  hero stands): on storyboard (sb) pictures the heroes are 1.35–2.55 times larger than in the plan — a ruler based on them puts the camera that many times closer.
The camera is appended to cameras.json (field from_picture); the answer is the __RESULT__ line with the camera height, distance and tilt."""
import argparse, json, math, os
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser(); ap.add_argument("scene"); ap.add_argument("hero"); ap.add_argument("image")
ap.add_argument("--horizon", type=float, required=True); ap.add_argument("--feet", required=True); ap.add_argument("--head", required=True)
ap.add_argument("--fov", type=float, default=60.0); ap.add_argument("--az", type=float, default=0.0); ap.add_argument("--name")
ap.add_argument("--dry", action="store_true"); ap.add_argument("--ref-height", type=float)
A = ap.parse_args()
W, H = Image.open(A.image).size; cx, cy = W / 2, H / 2
f = (max(W, H) / 2) / math.tan(math.radians(A.fov) / 2)            # the field of view is along the long side (sensor_fit AUTO)
ub, vb = map(float, A.feet.split(",")); ut, vt = map(float, A.head.split(","))
kit = json.load(open(os.path.join(A.scene, "figkit.json"))); fig = kit["named"][A.hero]
Hm, P = float(A.ref_height or fig.get("stature") or fig["height"]), np.array(fig["pos"], float)    # the ruler — the hero's height or an object of known height
pitch = math.atan((A.horizon - cy) / f)                             # horizon below the centre — the camera looks up
el = lambda v: pitch - math.atan((v - cy) / f)                      # angle of the ray of row v to the horizontal
tb, tt = math.tan(el(vb)), math.tan(el(vt))
if tt <= tb: raise SystemExit("head is below the feet in the picture — check --head and --feet")
D = Hm / (tt - tb)                                                  # hero ruler: the height between the head and feet rays
c = -D * tb                                                         # camera height above the hero's soles
yaw_h = math.radians(fig.get("heading") or 0.0) + math.radians(A.az)   # a camera on this heading from the hero looks at the hero
back = np.array([math.cos(yaw_h), math.sin(yaw_h), 0.0])
du = math.atan((ub - cx) / f)                                       # hero right of centre — the camera axis is left of the hero
pos = P + back * D + np.array([0, 0, c])
look = -back; ca, sa = math.cos(du), math.sin(du)
look = np.array([look[0] * ca - look[1] * sa, look[0] * sa + look[1] * ca, 0.0])   # counter-clockwise: the axis is left of the hero
fwd = np.array([look[0] * math.cos(pitch), look[1] * math.cos(pitch), math.sin(pitch)])
right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = right, up, -fwd, pos
s = 1536 / max(W, H); res = [int(round(W * s / 8)) * 8, int(round(H * s / 8)) * 8]
spec = {"name": A.name or f"from picture {os.path.basename(A.image)}", "lens_mm": round(f / max(W, H) * 36.0, 3), "sensor_width_mm": 36.0,
        "sensor_fit": "AUTO", "shift_x": 0.0, "shift_y": 0.0, "resolution": res, "matrix_world": M.round(6).tolist(),
        "from_picture": {"image": os.path.abspath(A.image), "hero": A.hero, "fov": A.fov, "horizon": A.horizon, "feet": [ub, vb], "head": [ut, vt],
                         "az": A.az}}
k = None
if not A.dry:
    cams_p = os.path.join(A.scene, "cameras.json"); cams = json.load(open(cams_p)) if os.path.exists(cams_p) else []
    if os.path.islink(cams_p):                                      # a link to another scene's cameras — make our own copy
        import shutil; real = os.path.realpath(cams_p); os.remove(cams_p); shutil.copy(real, cams_p)
    cams.append(spec); json.dump(cams, open(cams_p, "w"), ensure_ascii=False); k = len(cams)
print("__RESULT__" + json.dumps({"camera": k, "height_m": round(c, 2), "distance_m": round(D, 2), "pitch_deg": round(math.degrees(pitch), 1),
                                 "lens_mm": spec["lens_mm"], "pos": pos.round(2).tolist(), "resolution": res}, ensure_ascii=False))
