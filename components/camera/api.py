"""Public entry of the Camera (ARCHITECTURE.md: the mechanism stays with its owner, neighbours call api): shot movement from a camera.
Used by the take (components/time/bl_take.py: camera.move) and the cam_move.py command; neighbours should not create their own
camera movement. Pure numpy: works both in the project's Python and in Blender."""
import math

import numpy as np

KINDS = ("orbit", "dolly", "crane", "truck", "pan", "zoom")


def _rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def look_at(pos, look):
    """Camera matrix 4×4 at point pos, looking at look (up is world Z): columns r, u, −f, pos."""
    pos, look = np.asarray(pos, float), np.asarray(look, float)
    f = look - pos; f /= max(np.linalg.norm(f), 1e-9)
    r = np.cross(f, [0.0, 0.0, 1.0]); r /= max(np.linalg.norm(r), 1e-9); u = np.cross(r, f)
    return [[r[0], u[0], -f[0], pos[0]], [r[1], u[1], -f[1], pos[1]], [r[2], u[2], -f[2], pos[2]], [0.0, 0.0, 0.0, 1.0]]


def move(M0, center, kind, val, e=1.0, lens=35.0, aim=True):
    """Shot movement from camera M0 (4×4) toward the centre or around it: orbit — ° about Z, dolly — fraction of the way to the centre (< 0 — pull back),
    crane — m up, truck — m sideways (the target moves along), pan — ° of gaze rotation, zoom — lens multiplier; e — fraction of the path 0…1.
    aim=True — the camera looks at the centre (take); False — the frame is held: orbit and pan rotate the whole camera, dolly, crane
    and truck do not rotate it (the cam_move command). → (4×4 matrix as a list, lens)."""
    if kind not in KINDS: raise ValueError(f"movement '{kind}': available are {', '.join(KINDS)}")
    M0 = np.asarray(M0, float); p0, R0 = M0[:3, 3].copy(), M0[:3, :3].copy(); c = np.asarray(center, float)
    r0 = R0[:, 0] / max(np.linalg.norm(R0[:, 0]), 1e-9); t = float(val) * float(e)
    pos, look, R, lens2 = p0, c, R0, float(lens)
    if kind == "orbit": pos = c + _rz(t) @ (p0 - c); R = _rz(t) @ R0
    elif kind == "dolly": pos = p0 + (c - p0) * t
    elif kind == "crane": pos = p0 + np.array([0.0, 0.0, t])
    elif kind == "truck": pos = p0 + r0 * t; look = c + r0 * t
    elif kind == "pan": look = p0 + _rz(t) @ (c - p0); R = _rz(t) @ R0
    elif kind == "zoom": lens2 = float(lens) * (1 + (float(val) - 1) * float(e))
    if aim: return look_at(pos, look), lens2
    Mn = np.eye(4); Mn[:3, :3] = R; Mn[:3, 3] = pos
    return Mn.tolist(), lens2


# ── shot framing: camera position and gaze from the participants (positions smoothed over the take's window) ──
def _spec(pos, look, lens, size, name, look_key=True):
    W, H = size
    d = {"name": name, "location": [float(v) for v in pos], "lens_mm": float(lens), "sensor_width_mm": 36.0, "sensor_fit": "AUTO"}
    if look_key: d["_look"] = [float(v) for v in look]
    d.update({"shift_x": 0.0, "shift_y": 0.0, "resolution": [W, H], "matrix_world": look_at(pos, look)})
    return d


def frame_group(pts, hz, side_n, lens=28.0, el=24.0, size=(1280, 720), name="frame"):
    """Wide shot of a group: the camera across the axis (side_n — the axis normal toward the camera side), coverage — the group along the axis
    with a 2 m margin, tilt el°; pts — the participants' positions (x, y), hz — the mean height of their ground."""
    P = np.asarray(pts, float)[:, :2]; cm = P.mean(0); n = np.asarray(side_n, float)[:2]; th = 18.0 / lens
    ad = np.array([-n[1], n[0]]); half = float(np.max(np.abs((P - cm) @ ad))) + 2.0; dep = float(np.max(np.abs((P - cm) @ n)))
    dist = max(6.0, half / th + dep); e = math.radians(el)
    pos = [cm[0] + n[0] * dist * math.cos(e), cm[1] + n[1] * dist * math.cos(e), hz + 1.0 + dist * math.sin(e)]
    return _spec(pos, [cm[0], cm[1], hz + 0.8], lens, size, name)


def frame_subject(xy, yaw, hz, h, side_n, plan="medium", lens=None, size=(1280, 720), name="frame"):
    """Medium (the full figure with a margin) or close (chest and head) shot of the hero: from the front-side, but from the camera side of the axis
    (the 180° rule); xy, yaw — the hero's position and heading, hz — ground, h — height."""
    W, H = size; lens = float(lens or (35.0 if plan == "medium" else 50.0)); tv = 18.0 / lens * H / W
    n = np.asarray(side_n, float)[:2]; fw = np.array([math.cos(yaw), math.sin(yaw)])
    d = (fw + n) / np.linalg.norm(fw + n) if np.linalg.norm(fw + n) > 0.3 else n.copy()
    if d @ n < 0.3: d = (d + n) / np.linalg.norm(d + n)
    fh = 1.5 if plan == "medium" else 0.75; dist = max(1.8, h * fh / 2 / tv); c = np.asarray(xy, float)[:2]
    pos = [c[0] + d[0] * dist, c[1] + d[1] * dist, hz + h * (0.6 if plan == "medium" else 0.8)]
    return _spec(pos, [c[0], c[1], hz + h * (0.5 if plan == "medium" else 0.75)], lens, size, name)


def frame_two(P1, P2, Q1, Q2, now1, now2, hz, h1, h2, side=1, lens=32.0, size=(1280, 720), name="frame"):
    """Two in frame: to the side of the line between them, at a distance where both fit with a margin; one side per shot — chosen by the positions
    at the start of the piece (Q1, Q2), never jumping the line; P1, P2 — positions smoothed over the window, now1, now2 — positions in this frame (coverage)."""
    P1, P2, Q1, Q2 = (np.asarray(v, float)[:2] for v in (P1, P2, Q1, Q2)); d2 = P2 - P1; L = max(float(np.linalg.norm(d2)), 0.5)
    n2 = np.array([-d2[1], d2[0]]) / L * side; q = Q2 - Q1; n0 = np.array([-q[1], q[0]]) * side
    if n2 @ n0 < 0: n2 = -n2
    cm, dh = (P1 + P2) / 2, d2 / L
    half = max(abs((np.asarray(p, float)[:2] - cm) @ dh) for p in (now1, now2)) + 1.2
    dist = max(3.5, half / math.tan(math.atan(18.0 / lens)))
    pos = [cm[0] + n2[0] * dist, cm[1] + n2[1] * dist, hz + 1.6 + 0.1 * dist]
    return _spec(pos, [cm[0], cm[1], hz + 0.55 * (h1 + h2) / 2], lens, size, name)


def frame_over(shooter_xy, head_z, target_xy, target_z, offset=(0.6, 1.8, 0.4), lens=32.0, size=(1280, 720), name="frame"):
    """Over the shoulder of a shooter at a target: sideways offset[0], back offset[1], above the head offset[2], m."""
    pa, pb = np.asarray(shooter_xy, float)[:2], np.asarray(target_xy, float)[:2]; f2 = (pb - pa) / max(np.linalg.norm(pb - pa), 1e-9)
    r2 = np.array([f2[1], -f2[0]]); o = offset; p = pa + r2 * o[0] - f2 * o[1]
    return _spec([p[0], p[1], head_z + o[2]], [pb[0], pb[1], target_z], lens, size, name, look_key=False)


def frame_point(pos, look, lens=32.0, size=(1280, 720), name="frame"):
    """Camera from scene points: where it stands and where it looks."""
    return _spec(pos, look, lens, size, name, look_key=False)


def follow_pos(xy, hz, yaw, offset=(2.5, 5.0, 1.8)):
    """Position of a follow camera: from the hero (xy — the position smoothed over a window; yaw — heading) sideways offset[0], forward offset[1], up offset[2]."""
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0]); lef = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    return np.array([xy[0], xy[1], hz]) + lef * offset[0] + fwd * offset[1] + np.array([0.0, 0.0, offset[2]])


def follow_look(pos, hero, lead, lens=35.0):
    """Where a follow camera looks: at the hero's point through the lead offset, but not beyond the third-of-frame line — the hero sits on a third,
    with open space ahead (aiming straight at the lead point, a close camera pushed the hero off the edge of the frame)."""
    pos, H, Lp = (np.asarray(v, float) for v in (pos, hero, lead))
    dH = (H - pos) / max(np.linalg.norm(H - pos), 1e-9); dL = (Lp - pos) / max(np.linalg.norm(Lp - pos), 1e-9)
    a = math.acos(max(-1.0, min(1.0, float(dH @ dL)))); th = math.atan(6.0 / lens)
    if a > th and a > 1e-9:                                              # spherical interpolation of directions (like Vector.slerp)
        t = th / a; v = (math.sin((1 - t) * a) * dH + math.sin(t * a) * dL) / math.sin(a)
    else: v = dL
    return pos + v * float(np.linalg.norm(H - pos))


def shot(spec, scene=None, track=None):
    """Place-aware move (components/camera/move): recipe → frames A and B from the place field → rig → timing → checks; track — the hero's
    track (pos N×3, yaw N, h) for following. → {cams (records per frame: frame, fps), checks, bad, plan, skipped}. The old path is
    move() (a move from a camera, without a place)."""
    from components.camera import move as MV
    return MV.shot(spec, scene, track)


def panel(spec, scene=None):
    """A comic-panel frame by the place — a move of zero length (the same choice of position and the same checks)."""
    from components.camera import move as MV
    return MV.panel(spec, scene)


def plan_check(shots, fmt="fiction"):
    """Episode plan against a format profile (data/profiles.yaml): shots — [{sec, recipe}]; share of static shots, shot lengths. → (numbers, "what is wrong")."""
    from components.camera import move as MV
    from components.camera.move import checks as K
    return K.sequence(shots, MV.profile({"format": fmt})[1])


def cut_check(a, b, subject, axis=None, comic=False):
    """Cut between two shots of one subject (the last frame of A, the first frame of B; {pos, look, lens}): the 30° and 20 % rule, one height per
    scene, the scene axis, the position of the main subject on screen. → (numbers, "what is wrong")."""
    from components.camera.move import checks as K
    return K.cut(a, b, subject, axis, comic)


def frame_follow(xy, hz, yaw, hero, lead, offset=(2.5, 5.0, 1.8), lens=35.0, size=(1280, 720), name="frame"):
    """Following the hero: position — follow_pos, gaze — follow_look (the hero on a third, with open space ahead)."""
    pos = follow_pos(xy, hz, yaw, offset)
    return _spec(pos, follow_look(pos, hero, lead, lens), lens, size, name, look_key=False)
