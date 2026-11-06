"""Camera movement from a base camera: push-in, orbit, lift — N new cameras in the scene's cameras.json.
.venv/bin/python -m components.camera.cam_move scenes/<name> <camera K> <dolly=0.25 | orbit=10 | crane=2.0 | truck=3 | pan=15 | zoom=1.5> <frames> [center=x,y,z] [hero=m1] [detour=0]
  dolly — fraction of the distance to the centre (push-in > 0, pull-back < 0); orbit — degrees around the centre; crane — lift in metres.
  The frame is held: an orbit rotates the whole camera, a push-in and a crane do not change its direction.
  The centre by default is the point on the camera axis at the depth of the frame centre (frame.json is not needed: from the raw depth if present, otherwise 10 m).
The path is checked and repaired by itself (components/camera/bl_cam_path.py, checks: a 0.3 m gap to buildings, the hero visible no worse than from
the original camera): a bad key is lifted or pulled toward the target, the frame stays the same; impassable keys are reported in the "path.failed" answer
(shorten the movement there). The hero comes from the intent of the original camera or the hero= key; detour=0 — no check.
The answer is the numbers of the new cameras; then core/outputs/frames.py."""
import core.paths         # the code is here, the core — core/<part> through kernel
import json, math, os, subprocess, sys
import numpy as np
from components.camera import api as CAMAPI
a = sys.argv[1:]
SD, K, move, N = a[0], int(a[1]), a[2], int(a[3]); opt = dict(x.split("=", 1) for x in a[4:])
cams = json.load(open(os.path.join(SD, "cameras.json"))); base = cams[K - 1]
M = np.array(base["matrix_world"], float); pos, fwd = M[:3, 3].copy(), -M[:3, 2]
center = np.array(list(map(float, opt["center"].split(","))), float) if "center" in opt else pos + fwd * float(opt.get("dist", 10.0))
kind, val = move.split("=")
new = []
for i in range(1, N + 1):                        # the frame is held: an orbit rotates the whole camera around the centre, a push-in and a crane
    t = float(val) * i / N                         # the frame is held: an orbit rotates the whole camera, a push-in and a crane do not rotate it
    Mn, _ = CAMAPI.move(M, center, kind, t, 1.0, base.get("lens_mm", 35), aim=False); Mn = np.asarray(Mn, float)   # one mechanism with the take
    P = Mn[:3, 3]
    spec = dict(base, name=f"{base['name'][:40]} · {kind} {t:.3g}", location=P.tolist(), matrix_world=Mn.round(7).tolist())
    spec.pop("intent", None); spec.pop("from_picture", None)   # the base's shot size and angle are wrong after the orbit and push-in — the prompt measures them itself
    cams.append(spec); new.append(len(cams))
cp = os.path.join(SD, "cameras.json")
if os.path.islink(cp):                                              # a link to another scene's cameras — make our own copy, the other one stays unchanged
    import shutil; real = os.path.realpath(cp); os.remove(cp); shutil.copy(real, cp)
json.dump(cams, open(cp, "w"), ensure_ascii=False, indent=1)
path = None
if opt.get("detour", "1") != "0" and os.path.exists(os.path.join(SD, "scene.obj")):   # collision-free path
    hero = opt.get("hero") or (base.get("intent") or {}).get("hero")
    r = subprocess.run(["blender", "-b", "--factory-startup", "-P", core.paths.RUN, "--", os.path.join(core.paths.ROOT, "components/camera/bl_cam_path.py"), "--",
                        SD, f"{new[0]}-{new[-1]}", ",".join(f"{c:.4f}" for c in center), f"base={K}"] + ([f"hero={hero}"] if hero else []),
                       capture_output=True, text=True)
    path = next((json.loads(l.split("__RESULT__", 1)[1]) for l in r.stdout.splitlines() if "__RESULT__" in l), {"error": r.stderr[-300:]})
print("__RESULT__" + json.dumps({"cameras": new, "center": [round(c, 2) for c in center], **({"path": path} if path else {})}, ensure_ascii=False))
