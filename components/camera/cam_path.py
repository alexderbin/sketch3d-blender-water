"""Fly-through along control points: the camera path and the look-at point are Catmull–Rom splines, the lens changes smoothly; N cameras are appended to
scenes/<name>/cameras.json (then the usual core/outputs/frames.py <a-b> and core/outputs/video.py). cam_move.py moves from one camera
(push-in, circular orbit, crane) — this path is free: "a fly-over a horde to the gates and a rise to the palace".
.venv/bin/python -m components.camera.cam_path scenes/<name> <path.json> <N> [--res 1280x720] [--name fly]
path.json: [{"pos": [x, y, z], "look": [x, y, z], "lens": mm}, …] — from 2 points; uniform along the length of the path (not per point).
The answer is the __RESULT__ line {"cams": "a-b"}."""
import argparse, json, math, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("scene"); ap.add_argument("path"); ap.add_argument("n", type=int)
ap.add_argument("--res", default="1280x720"); ap.add_argument("--name", default="fly"); A = ap.parse_args()
P = json.load(open(A.path)); W, H = map(int, A.res.split("x"))


def cr(pts, t):
    """Catmull–Rom through points pts (k, d) at parameter t ∈ [0, k−1]."""
    k = len(pts); i = min(int(t), k - 2); u = t - i
    p0, p1, p2, p3 = pts[max(i - 1, 0)], pts[i], pts[i + 1], pts[min(i + 2, k - 1)]
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)


pos = np.array([p["pos"] for p in P], float); look = np.array([p["look"] for p in P], float); lens = np.array([[p.get("lens", 28)] for p in P], float)
ts = np.linspace(0, len(P) - 1, 2000); curve = np.array([cr(pos, t) for t in ts])
s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))])   # uniform along the length
tt = np.interp(np.linspace(0, s[-1], A.n), s, ts)
cams = json.load(open(os.path.join(A.scene, "cameras.json"))); a = len(cams) + 1
for j, t in enumerate(tt):
    p, l, f = cr(pos, t), cr(look, t), float(cr(lens, t)[0])
    fwd = l - p; fwd /= np.linalg.norm(fwd); right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
    m = np.eye(4); m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, up, -fwd, p          # Blender camera: looks along −Z, up is +Y
    cams.append({"name": f"{A.name}_{j:03d}", "lens_mm": round(f, 3), "sensor_width_mm": 36.0, "sensor_height_mm": 24.0, "sensor_fit": "HORIZONTAL",
                 "shift_x": 0.0, "shift_y": 0.0, "resolution": [W, H], "matrix_world": m.tolist()})
json.dump(cams, open(os.path.join(A.scene, "cameras.json"), "w"), ensure_ascii=False, indent=1)
print("__RESULT__" + json.dumps({"cams": f"{a}-{len(cams)}", "n": A.n, "length_m": round(float(s[-1]), 1)}, ensure_ascii=False))
