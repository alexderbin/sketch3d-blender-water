"""Frame values relative to a subject ↔ camera (the toric space: Lino and Christie 2015; Bonatti 2020).

Subject: pos — a point on the ground (x, y, z), h — height, m, yaw — heading, ° (0 — along the world +X, counter-clockwise).
Frame: size — the subject's height as a fraction of the frame height; yaw — the camera's angle around the subject from its heading, ° (0 — from
the front, 90 — on the hero's left, 180 — from behind); pitch — the gaze tilt, ° (+ — from above); sx, sy — where on the screen the aim point is
(0…1, left to right and top to bottom); aim — the aim point on the subject, a fraction of height; lens — the lens, mm (a 36 mm sensor by width,
as in api.look_at).
The conversion is exact both ways: place(measure(cam)) = cam to 1e-6 (test stand). Pure numpy."""
import math

import numpy as np

from components.camera.api import look_at

DEFAULT = {"size": 0.6, "yaw": 40.0, "pitch": 8.0, "sx": 0.5, "sy": 0.55, "aim": 0.6, "lens": None}
# Shot sizes in words (Brown, "Character shots": full — head to toe, "cowboy" — to mid-thigh, medium — waist
# up, medium close — to mid-chest, close — head and shoulders, close-up — from the chin, extreme close-up —
# the eyes) → size: the hero's height as a fraction of the frame height (the visible share of the body ≈ 0.9 /
# size)
SIZES = {"extreme_wide": 0.12, "wide": 0.35, "full": 0.85, "cowboy": 1.4, "medium": 2.0, "medium_close": 3.0, "close": 4.5,
         "big_close": 7.5, "extreme_close": 18.0}


def lens_for(size, h=1.8):
    """Lens by shot size (wide 24, medium-wide 35, medium 50, close 85 mm); a landmark taller than 6 m — 35."""
    if h > 6.0:
        return 35.0
    return 24.0 if size < 0.6 else 35.0 if size < 1.2 else 50.0 if size < 2.5 else 85.0


def full(frame, h=1.8):
    """A frame with all values: missing ones take defaults, a size given as a word is looked up in SIZES, the lens comes from the size."""
    given = {k: v for k, v in (frame or {}).items() if v is not None}; f = dict(DEFAULT); f.update(given)
    if isinstance(f["size"], str):
        if f["size"] not in SIZES: raise ValueError(f"size '{f['size']}': available are {', '.join(SIZES)} or a number")
        f["size"] = SIZES[f["size"]]
    f["size"] = float(f["size"]); f["lens"] = float(f["lens"] or lens_for(f["size"], h))
    if "aim" not in given and f["size"] >= 0.6:                      # from full length and closer: the top of the head is 8 % below the top of the frame
        f["aim"] = min(0.95, max(0.5, 1.0 - 0.42 / f["size"])); f["sy"] = float(given.get("sy", 0.5))
    return f


def tans(lens, aspect):
    """Tangents of the half field of view: by width and by height."""
    th = 18.0 / float(lens)
    return th, th / float(aspect)


def aim_point(subject, aim):
    p = np.asarray(subject["pos"], float)
    return p + np.array([0.0, 0.0, float(subject.get("h", 1.8)) * float(aim)])


def basis(pos, look):
    """Right, up, view direction of the camera (world Z up), as in api.look_at."""
    f = np.asarray(look, float) - np.asarray(pos, float); f /= max(np.linalg.norm(f), 1e-12)
    r = np.cross(f, [0.0, 0.0, 1.0]); r /= max(np.linalg.norm(r), 1e-12)
    return r, np.cross(r, f), f


def project(pos, look, lens, aspect, p):
    """World point → screen (x, y) 0…1 (left to right, top to bottom) and depth; behind the camera — depth ≤ 0."""
    r, u, f = basis(pos, look); th, tv = tans(lens, aspect); d = np.asarray(p, float) - np.asarray(pos, float)
    z = d @ f
    with np.errstate(divide="ignore", invalid="ignore"):
        return 0.5 + 0.5 * (d @ r) / z / th, 0.5 - 0.5 * (d @ u) / z / tv, z


def place(subject, frame, aspect=16 / 9):
    """Frame → camera: position, look-at point, lens. The distance comes from size and the vertical field of view; the position from yaw (from the subject's
    heading) and pitch; the gaze is shifted so that the aim lands at (sx, sy) — three refinement steps (a shift changes the basis)."""
    h = float(subject.get("h", 1.8)); f = full(frame, h)
    P = aim_point(subject, f["aim"]); th, tv = tans(f["lens"], aspect)
    d = h / (float(f["size"]) * 2.0 * tv)
    a = math.radians(float(subject.get("yaw", 0.0)) + float(f["yaw"])); e = math.radians(float(f["pitch"]))
    C = P + d * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    return {"pos": C, "look": look_for(C, P, f["sx"], f["sy"], f["lens"], aspect), "lens": f["lens"]}


def look_for(C, P, sx, sy, lens, aspect=16 / 9):
    """The look-at point from position C at which point P lands at screen (sx, sy): refined to 1e-10 of the frame (shifting the gaze changes the basis;
    converges in 3–6 steps)."""
    C, P = np.asarray(C, float), np.asarray(P, float); th, tv = tans(lens, aspect); L = P.copy()
    for _ in range(12):
        r, u, fw = basis(C, L); z = float((P - C) @ fw)
        x, y, _ = project(C, L, lens, aspect, P)
        if abs(x - float(sx)) < 1e-10 and abs(y - float(sy)) < 1e-10:
            break
        L = L + r * (x - float(sx)) * 2.0 * th * z - u * (y - float(sy)) * 2.0 * tv * z
    return L


def measure(pos, look, lens, subject, aim=0.6, aspect=16 / 9):
    """Camera → frame relative to a subject (the inverse of place for the same aim)."""
    h = float(subject.get("h", 1.8)); P = aim_point(subject, aim); C = np.asarray(pos, float)
    _, tv = tans(lens, aspect); r, u, fw = basis(C, look)
    v = C - P; dist = float(np.linalg.norm(v))
    # place puts the camera at distance d = h / (size · 2 tv) from the aim along the ray (yaw, pitch) — the
    # inverse calculation uses the same length
    size = h / (2.0 * tv * dist)
    yaw = (math.degrees(math.atan2(v[1], v[0])) - float(subject.get("yaw", 0.0)) + 180.0) % 360.0 - 180.0
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, v[2] / max(dist, 1e-12)))))
    x, y, _ = project(C, look, lens, aspect, P)
    return {"size": size, "yaw": yaw, "pitch": pitch, "sx": float(x), "sy": float(y), "aim": float(aim), "lens": float(lens)}


def lerp(fa, fb, u, wrap=True):
    """A frame between A and B: size and lens by logarithm (an even feel of push-in), yaw by the shortest path (wrap=False — as given: an arc over 180°),
    the rest linearly."""
    out = {}
    for k in ("size", "lens"):
        out[k] = math.exp((1 - u) * math.log(fa[k]) + u * math.log(fb[k]))
    dy = fb["yaw"] - fa["yaw"]
    out["yaw"] = fa["yaw"] + u * ((dy + 180.0) % 360.0 - 180.0 if wrap else dy)
    for k in ("pitch", "sx", "sy", "aim"):
        out[k] = (1 - u) * fa[k] + u * fb[k]
    return out


def camera(pos, look, lens, size=(1280, 720), name="frame", **extra):
    """A camera record for cameras.json (as in api: a 36 mm sensor by width, the look_at matrix)."""
    d = {"name": name, "location": [float(v) for v in pos], "lens_mm": float(lens), "sensor_width_mm": 36.0, "sensor_fit": "AUTO",
         "shift_x": 0.0, "shift_y": 0.0, "resolution": [int(size[0]), int(size[1])], "matrix_world": look_at(pos, look),
         "_look": [float(v) for v in look]}
    d.update(extra)
    return d
