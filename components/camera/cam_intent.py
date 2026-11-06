"""Camera by intent: intent → candidates around the hero → rejection of blocked ones (rays from the hero's points to the camera; the camera
not inside a wall) → the frame by the framing solver (height fraction, position on a third, headroom) → the 180° axis from a pair of heroes → the frame caption.
Synthesis of the "camera" node: the framing solver (armature_core) as it is (components/camera/ext/armature), the candidate search, the rejection by rays,
the line side by a perpendicular from the pair in a constant order, the vocabulary of intents below.
blender -b --factory-startup -P core/blender/run.py -- components/camera/cam_intent.py -- <bench/scenes/X> <hero> <intent> [key=value …]
  intent: wide | medium | closeup | heroic | overhead (the whole squad) | two_shot (pair=) | over_shoulder (pair= — whose shoulder;
          the base point: 0.55 m sideways, 1.6 m back from the head, 32 mm) | from_cover (the nearest cover — at the bottom of the frame) | pov (through the hero's eyes)
  keys: az=<azimuth shift from the hero's gaze, °> el=<°> lens=<mm> x=<fraction of width> hf=<fraction of height> fmt=16:9|9:16|1:1|2:1…
        pair=<second hero> (the camera on the same side of the line as camera 1 of the scene) name=<camera name> dry=1 (do not write)
The camera is appended to bench/scenes/X/cameras.json; the answer is the __RESULT__ line with the camera number, the caption and the checks."""
import core.paths         # the code is here, the core — core/<part> through kernel
import json, math, os, sys
core.paths.component_ext("camera", "armature")
import bpy
from mathutils import Vector, Matrix
from armature_core import framing
from core.blender import bl_scene as B
from components.camera import framing_fast
framing_fast.install(framing)      # the same framing solver ~50 times faster (wrappers _bisect, _extent; the solver's own code is untouched)

a = sys.argv[sys.argv.index("--") + 1:]
SD, HERO, INTENT = os.path.abspath(a[0]), a[1], a[2]
opt = dict(kv.split("=", 1) for kv in a[3:])
# intent → cloud, height fraction of the frame, position in x, position in y, azimuth shift from the gaze,
# tilt, focal length
PRESETS = {"wide": ("full", 0.30, 0.33, 0.55, 35, 3, 35), "medium": ("upper", 0.88, 0.62, 0.56, -35, 0, 50),
           "closeup": ("head", 0.75, 0.50, 0.45, 20, -10, 85), "heroic": ("full", 0.70, 0.66, 0.50, 50, -18, 24),
           "overhead": ("group", 0.50, 0.50, 0.50, 150, 55, 28), "two_shot": ("pair", 0.62, 0.50, 0.52, 0, 4, 35),
           "over_shoulder": ("upper", 0.55, 0.50, 0.50, 0, 0, 32), "from_cover": ("upper", 0.55, 0.50, 0.50, 0, 0, 35),
           "pov": ("full", 0.50, 0.50, 0.50, 0, 0, 28)}
SPECIAL = {"over_shoulder", "from_cover", "pov"}                     # the camera position is computed from the heroes and cover, not searched
if INTENT in ("two_shot", "over_shoulder") and "pair" not in opt: sys.exit(f"intent '{INTENT}' needs pair=<second hero>")
cloud_kind, hf, ex, ty, daz, el, lens = PRESETS[INTENT]
hf, ex, el, lens = float(opt.get("hf", hf)), float(opt.get("x", ex)), float(opt.get("el", el)), float(opt.get("lens", lens))
daz = float(opt.get("az", daz))
fa, fb = (float(x) for x in opt.get("fmt", "16:9").split(":"))          # any panel format: the long side is 1536
W, H = (1536, int(round(1536 * fb / fa / 8)) * 8) if fa >= fb else (int(round(1536 * fa / fb / 8)) * 8, 1536)

kit = json.load(open(os.path.join(SD, "figkit.json")))
cams_path = os.path.join(SD, "cameras.json"); cams = json.load(open(cams_path)) if os.path.exists(cams_path) else []


def figure(fid):
    """Joints, gaze, height. The gaze is the heading from the export (from the feet; core/blender/bl_export_scene.py), otherwise from the hip line."""
    f = kit["named"][fid]; J = {k: Vector(v) for k, v in f["joints"].items()}
    if "rig" in f:
        h = math.radians(f["heading"]); fwd = Vector((math.cos(h), math.sin(h), 0))
    else:
        left = J["l_hip"] - J["r_hip"]; left.z = 0; fwd = left.normalized().cross(Vector((0, 0, 1))).normalized()
    return J, fwd, float(f.get("stature") or f["height"])   # height from the skeleton (Actors), without it — the posed mesh


J, fwd, Hm = figure(HERO)
heading = math.degrees(math.atan2(fwd.y, fwd.x))
feet = min(v.z for v in J.values()) - 0.05 * Hm / 1.8
top = max(J.values(), key=lambda v: v.z) + Vector((0, 0, 0.12 * Hm))
full = [list(v) for v in J.values()] + [list(top)] + [[J[s].x, J[s].y, feet] for s in ("r_ankle", "l_ankle")]
upper = [list(J[k]) for k in J if k not in ("r_knee", "l_knee", "r_ankle", "l_ankle")] + [list(top)]
head = [list(J[k]) for k in ("nose", "neck", "r_eye", "l_eye", "r_ear", "l_ear", "r_shoulder", "l_shoulder")] + [list(top)]
cloud = {"full": full, "upper": upper, "head": head}.get(cloud_kind, full)

sc, statics, members, objs, Wd, _ = B.load_scene(SD)
hero_ob = bpy.data.objects.get(f"figure_{HERO}")
dg = bpy.context.evaluated_depsgraph_get()
hv = [hero_ob.matrix_world @ v.co for v in hero_ob.evaluated_get(dg).data.vertices] if hero_ob else []
step = max(1, len(hv) // 160); samples = hv[::step] or [Vector(p) for p in cloud]
if hv:   # the framing cloud is mesh vertices, not joints (a lesson from armature: shoulder pads, backpack and barrel stick far beyond the joints)
    # "medium" — waist up: the cloud starts at the waist (hips + 8 % of height), the bottom of the cloud at the
    # lower edge (ty 0.56 at fraction 0.88); "close" — head and neck within 0.22 of the height from the neck:
    # shoulder pads and the barrel squeezed the frame in width (head 43 % instead of 75 %)
    cut = {"full": -1e9, "upper": max(J["r_hip"].z, J["l_hip"].z) + 0.08 * Hm, "head": J["neck"].z - 0.04 * Hm}.get(cloud_kind, -1e9)
    near = (lambda v: (Vector((v.x - J["neck"].x, v.y - J["neck"].y)).length <= 0.22 * Hm)) if cloud_kind == "head" else (lambda v: True)
    pts = [v for v in hv if v.z >= cut and near(v)] or [v for v in hv if v.z >= cut]
    ext = {id(max(pts, key=lambda v: v[i] * sgn)) : max(pts, key=lambda v: v[i] * sgn) for i in range(3) for sgn in (1, -1)}
    cloud = [list(v) for v in pts[::max(1, len(pts) // 160)]] + [list(v) for v in ext.values()]   # 160 points + the extreme ones: the solver works per point
    samples = pts[::max(1, len(pts) // 160)]       # visibility is measured on the part that is in the frame: legs behind cover do not hurt the medium shot
pair_ob, samples2 = None, []
if cloud_kind == "pair":                           # two: the cloud is both figures, the visibility is the worse of the two
    pair_ob = bpy.data.objects.get(f"figure_{opt['pair']}")
    hv2 = [pair_ob.matrix_world @ v.co for v in pair_ob.evaluated_get(dg).data.vertices] if pair_ob else []
    samples2 = hv2[::max(1, len(hv2) // 160)] or [Vector(p) for p in figure(opt["pair"])[0].values()]
    cloud = cloud + [list(v) for v in samples2] + [list(v) for v in (max(hv2 or samples2, key=lambda q: q.z),)]
if cloud_kind == "group":                          # from above: the whole squad in the frame, not a hero at the edge of an empty ground
    cloud = [list(v) for f in kit["named"].values() for v in map(Vector, f["joints"].values())]


def share_of(pos, smp, who):
    ok = 0
    for p in smp:
        d = p - pos; L = d.length
        hit, loc, nrm, idx, ob, _ = sc.ray_cast(dg, pos, d / L, distance=L - 0.02)
        ok += (not hit) or ob == who
    return ok / max(len(smp), 1)


def visible_share(pos):
    """Fraction of the hero's points that the ray from the camera reaches unobstructed (the first hit is the hero); for "two" — the worse of the two."""
    share = share_of(pos, samples, hero_ob)
    if samples2: share = min(share, share_of(pos, samples2, pair_ob))
    if cloud_kind == "head":       # close: the face is not covered even by its own shoulder pad — the ray to the nose and eyes reaches the face itself
        face = [J[k] for k in ("nose", "r_eye", "l_eye") if k in J]; fo = 0
        for p in face:
            d = p - pos; L = d.length
            hit, loc, nrm, idx, ob, _ = sc.ray_cast(dg, pos, d / L, distance=L + 0.2)
            fo += (not hit) or (ob == hero_ob and (loc - p).length < 0.06 * Hm)
        share = min(share, fo / max(len(face), 1) / 0.66 * 0.85)            # 2 of 3 face points — threshold 0.85
    return share


def inside_wall(pos, tgt):
    """Camera inside a wall: the ray from the target to the camera hits foreign geometry before it reaches the camera."""
    d = pos - tgt; L = d.length
    hit, loc, nrm, idx, ob, _ = sc.ray_cast(dg, tgt + d / L * 0.3, d / L, distance=L - 0.3)
    return hit and ob != hero_ob


side_ref = None
if "pair" in opt:                                # 180° axis: the camera's side of the pair's line is the same as for camera 1 of the scene
    J2, _, _ = figure(opt["pair"]); A_, B_ = (J["neck"], J2["neck"]) if HERO < opt["pair"] else (J2["neck"], J["neck"])
    axis = (B_ - A_); axis.z = 0
    side = lambda p: math.copysign(1, axis.cross(Vector((p[0] - A_.x, p[1] - A_.y, 0))).z)
    side_ref = side(Matrix(cams[0]["matrix_world"]).translation) if cams else None
def solve(az, e):
    """The framing solver holds the height fraction; if the point cloud sticks out in width (a 9:16 portrait, broad shoulders), the height fraction
    is reduced until the width fits in 92 % of the frame."""
    h, x_, y_, r = hf, ex, ty, None
    for _ in range(8):
        try:
            r = framing.solve_camera(cloud, cloud, height_frac=h, end_x_frac=x_, target_y_frac=y_, azimuth_deg=az, elevation_deg=e,
                                     lens_mm=lens, sensor_mm=36.0, width=W, height=H, radius_bounds=(0.5, 400.0), passes=3)
        except framing.FramingError:                                # the position on a third is unreachable (a wide 2:1 frame) — move closer to the centre
            x_, y_ = 0.5 + (x_ - 0.5) * 0.8, 0.5 + (y_ - 0.5) * 0.8; continue
        ux = r["achieved"]["union_x"]; wf = ux[1] - ux[0]
        if wf <= 0.92 and ux[0] >= 0 and ux[1] <= 1: break
        h *= min(0.92 / max(wf, 1e-6), 0.9)
    if r is None: raise RuntimeError(f"frame cannot be solved: azimuth {az:.0f}°, pitch {e:.0f}°, format {W}×{H}")
    return r


tries, best, prev = [], None, None
base = heading
if INTENT == "two_shot":                         # across the pair's line, from the side of camera 1 (otherwise — toward the hero's face)
    ax_ = math.degrees(math.atan2(axis.y, axis.x))
    base = next((b for b in (ax_ + 90, ax_ - 90) if side_ref is not None and side(framing.camera_position(list(J["neck"]), 10, 0, b)) == side_ref),
                max((ax_ + 90, ax_ - 90), key=lambda b: math.cos(math.radians(b - heading))))
feet_of = lambda Jx, Hx: min(v.z for v in Jx.values()) - 0.05 * Hx / 1.8
sens_h = 36.0 * (H / W if W >= H else 1.0)                       # the 36 mm sensor along the long side (AUTO)
special = None
if INTENT == "over_shoulder":                    # the camera in the second one's axes (−0.55; −1.6; +0.05 m from the head at 0.9 of the height)
    JA, _, HA = figure(opt["pair"]); s_ = HA / 1.8
    headA = Vector((JA["neck"].x, JA["neck"].y, feet_of(JA, HA) + 0.9 * HA)); tgt0 = Vector((J["neck"].x, J["neck"].y, feet + 0.86 * Hm))
    f_ = tgt0 - headA; f_.z = 0; f_.normalize(); rg = f_.cross(Vector((0, 0, 1)))
    for back, wide, up in ((1.6, 0.55, 0.05), (1.2, 0.5, 0.05), (0.9, 0.45, 0.05), (1.2, 0.5, 0.7), (0.9, 0.45, 1.3)):   # the base point;
        for sg in (1, -1):                                       # with a wall behind — closer to the shoulder, then higher (over the cover)
            pos0 = headA + rg * (wide * s_ * sg) - f_ * (back * s_) + Vector((0, 0, up * s_))
            if side_ref is not None and side(pos0) != side_ref: tries.append((sg, back, "beyond the axis")); continue
            if inside_wall(pos0, tgt0): tries.append((sg, back, "inside a wall")); continue
            vs0 = visible_share(pos0); tries.append((sg, back, round(vs0, 2)))
            if vs0 >= 0.5 and (special is None or vs0 > special[2]): special = (pos0, tgt0, vs0, lens, "over the shoulder", "seen over the shoulder of another figure in the foreground")
        if special: break
elif INTENT == "from_cover":                     # the nearest cover lower than 1.5 heights — between the camera and the hero, at the bottom of the frame
    kinds = {e["id"]: e["kind"] for e in json.load(open(os.path.join(SD, "world.json")))["reg"]}
    chest = Vector((J["neck"].x, J["neck"].y, feet + 0.62 * Hm)); hc = Vector((J["neck"].x, J["neck"].y, 0))
    for ob in statics:
        if ob.type != "MESH" or kinds.get(ob.name.split(".")[0]) not in ("box", "wall", "rock", "rocks", "siege_tower"): continue
        bb = [ob.matrix_world @ Vector(c) for c in ob.bound_box]; lo_ = Vector([min(q[i] for q in bb) for i in range(3)]); hi_ = Vector([max(q[i] for q in bb) for i in range(3)])
        c_ = (lo_ + hi_) / 2; d_ = Vector((c_.x - hc.x, c_.y - hc.y, 0)); L_ = d_.length
        if not 1.2 < L_ < 10 or hi_.z > feet + 1.5 * Hm or hi_.z < feet + 0.3 * Hm: continue
        rr = max(hi_.x - lo_.x, hi_.y - lo_.y) / 2; pos0 = Vector((c_.x, c_.y, 0)) + d_.normalized() * (rr + 0.8); pos0.z = max(hi_.z + 0.35, feet + 0.5 * Hm)
        if inside_wall(pos0, chest): continue
        vs0 = visible_share(pos0); low = chest - Vector((0, 0, 0.5 * Hm)); dl = low - pos0
        hit, loc, nrm, idx, hob, _ = sc.ray_cast(dg, pos0, dl.normalized(), distance=dl.length)
        front = max(0.0, fwd.dot((pos0 - hc).normalized()))       # the cover is in front of the hero — facing the camera, not with the back to it
        sc_ = vs0 + 0.3 * (hit and hob == ob) + 0.4 * front; tries.append((ob.name[:16], round(L_, 1), round(vs0, 2)))
        if vs0 >= 0.55 and (special is None or sc_ > special[6]):
            lens0 = max(24.0, min(70.0, sens_h * (chest - pos0).length / (1.1 * Hm)))
            special = (pos0, chest, vs0, lens0, f"from behind cover ({ob.name.split('.')[0]})", None, sc_)
elif INTENT == "pov":                            # through the hero's eyes: just in front of the eyes, looking along the heading
    eye = (J["r_eye"] + J["l_eye"]) / 2 if "r_eye" in J and "l_eye" in J else J["neck"] + Vector((0, 0, 0.1 * Hm))
    pos0 = eye + fwd * 0.08; tgt0 = eye + fwd * 10
    special = (pos0, tgt0, 1.0, lens, "hero's view", "out of frame: first-person view through the hero's eyes")
    if inside_wall(tgt0, pos0): tries.append((0, 0, "a wall in front of the eyes"))
if INTENT in SPECIAL:
    if special is None:
        print("__RESULT__" + json.dumps({"ok": False, "why": f"no position for '{INTENT}': blocked, beyond the axis, or no cover nearby", "tries": tries[:20]}, ensure_ascii=False)); sys.exit(1)
    pos, tgt, vs, lens = special[0], special[1], special[2], special[3]; dv = pos - tgt
    best = (math.degrees(math.atan2(dv.y, dv.x)), math.degrees(math.asin(max(-1, min(1, dv.z / dv.length)))), {}, tgt, pos, vs)
NEAR, FAR = (0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90), (120, -120, 150, -150, 180)
order = [(dd, de) for de in (0, 10, 20) for dd in NEAR] + [(dd, de) for de in (0, 10, 20) for dd in FAR]
for dd, de in ([] if INTENT in SPECIAL else order):   # azimuth first, then elevation: a low angle must stay low; the far
                                                  # azimuths (a pair beyond the 180° axis) — after all elevations
    az, e = base + daz + dd, el + de
    if prev is not None:                      # estimate of the position from the previous solution: the framing solver (0.01 s with wrappers, 0.6 s without)
        pe = Vector(framing.camera_position(prev["target"], prev["radius"], e, az))   # called only for candidates that passed the rejection
        if side_ref is not None and side(pe) != side_ref: tries.append((dd, de, "beyond the axis")); continue
        if inside_wall(pe, Vector(prev["target"])): tries.append((dd, de, "inside a wall")); continue
        if visible_share(pe) < 0.6: tries.append((dd, de, "closed")); continue
    try:
        r = solve(az, e); prev = r
    except RuntimeError:
        tries.append((dd, de, "cannot be solved")); continue
    tgt = Vector(r["target"]); pos = Vector(framing.camera_position(r["target"], r["radius"], e, az))
    if side_ref is not None and side(pos) != side_ref:
        tries.append((dd, de, "beyond the axis")); continue
    if inside_wall(pos, tgt): tries.append((dd, de, "inside a wall")); continue
    vs = visible_share(pos)
    tries.append((dd, de, round(vs, 2)))
    if vs >= 0.85: best = (az, e, r, tgt, pos, vs); break
if best is None:
    print("__RESULT__" + json.dumps({"ok": False, "why": "no angle: the hero is blocked or the camera is inside a wall", "tries": tries[:20]}, ensure_ascii=False)); sys.exit(1)
az, e, r, tgt, pos, vs = best
fwd_c = (tgt - pos).normalized(); right = fwd_c.cross(Vector((0, 0, 1))).normalized(); up = right.cross(fwd_c)
M = Matrix(((right.x, up.x, -fwd_c.x, pos.x), (right.y, up.y, -fwd_c.y, pos.y), (right.z, up.z, -fwd_c.z, pos.z), (0, 0, 0, 1)))
rel = (az - heading + 180) % 360 - 180
angle = "frontal" if abs(rel) < 20 else ("¾" if abs(rel) < 65 else ("profile" if abs(rel) < 115 else ("¾ from behind" if abs(rel) < 160 else "from behind")))
level = "from_below" if e < -8 else ("from_above" if e > 15 else "at eye level")
size = {"full": "full length" if hf > 0.5 else "wide", "upper": "medium (waist up)", "head": "closeup", "pair": f"two ({HERO} and {opt.get('pair')})",
        "group": "the whole squad"}[cloud_kind]
if INTENT in SPECIAL: size = special[4]
label = f"{INTENT}: {size}, {angle}, {level}, {lens:.0f} mm" if INTENT != "pov" else f"view: through the eyes of {HERO}, {lens:.0f} mm"
spec = {"name": opt.get("name", label), "location": list(pos), "lens_mm": lens, "sensor_width_mm": 36.0, "sensor_fit": "AUTO",
        "shift_x": 0.0, "shift_y": 0.0, "resolution": [W, H], "matrix_world": [list(row) for row in M],
        "intent": {"hero": HERO, "intent": INTENT, "azimuth": round(az, 1), "elevation": e, "height_frac": hf, "x_frac": ex, "view": angle,
                   "view_en": (special[5] if INTENT in SPECIAL and special[5] else {"frontal": "facing the viewer", "¾": "in three-quarter view", "profile": "in profile",
                               "¾ from behind": "in three-quarter back view, face turned away", "from behind": "seen from behind, back to the viewer"}[angle]
                               + (", seen from behind cover in the foreground" if INTENT == "from_cover" else "") + (", together with a second figure" if INTENT == "two_shot" else "")),
                   **({"pair": opt["pair"]} if "pair" in opt else {}),
                   "visible": round(vs, 2), "solver": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in (r.get("achieved") or {}).items()
                                                        if isinstance(v, (int, float))}}}
if not opt.get("dry"):
    if os.path.islink(cams_path):                  # a link to another scene's cameras — make our own copy, the other one stays unchanged
        import shutil; real = os.path.realpath(cams_path); os.remove(cams_path); shutil.copy(real, cams_path)
    cams.append(spec); json.dump(cams, open(cams_path, "w"), ensure_ascii=False, indent=1)
hint = (f"hero has the back to the camera: turn it — .venv/bin/python -m core.model.scene_edit {os.path.relpath(SD)} turn {HERO} face={pos.x:.2f},{pos.y:.2f}"
        if "from behind" in angle and INTENT not in ("overhead", "pov", "over_shoulder") else None)
print("__RESULT__" + json.dumps({"ok": True, "camera": len(cams), "label": label, "visible": round(vs, 2), "tries": len(tries), **({"hint": hint} if hint else {}),
                                 "rejected": [t for t in tries if not isinstance(t[2], float) or t[2] < 0.85][:8]}, ensure_ascii=False))
