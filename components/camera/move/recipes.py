"""Move recipes are primitives: what changes in the frame, how to choose frames A and B by the place, how to lay the path.

A recipe → a pose function pose(u) for the path fraction u ∈ [0, 1] (position, gaze, lens; "by frame" also the frame values for the aim
after the rig is set) and the selection details. Following (follow) goes along the subject's track, frame by frame. The candidates are yaw every 10°
around the subject: the camera by frame, the height by rig, whether it is free (the place field), whether the aim is visible, what is behind the subject (the background).
There is no selection logic by meaning: what to shoot is set by the recipe knobs (background A and B, side, shot size); the code only finds where it is possible."""
import math

import numpy as np

from components.camera.move import frame as F

RECIPES = ("static", "pan", "push", "pull", "truck", "arc", "crane", "follow", "zoom")
STEP = 10.0                                                          # candidate step around the subject, °


# ── subject ─────────────────────────────────────────────────────────────────────────────
def subject(spec, field, track=None):
    """Subject of the move: an anchor of the place or of the buildings (anchor: name; h — height above the ground, by default the top of the anchor or of
    the field above the ground), a point (point: x, y; h) or the hero's track (track: pos N×3, yaw N, h) — the first frame. yaw — heading, °."""
    s = spec.get("subject") or {}
    if isinstance(s, str):
        s = {"anchor": s}
    if track is not None:
        p = np.asarray(track["pos"], float)[0]
        return {"pos": p, "h": float(track.get("h", s.get("h", 1.8))), "yaw": float(np.asarray(track.get("yaw", [0.0]), float)[0]),
                "kind": "hero"}
    if s.get("anchor"):
        if s["anchor"] not in field.anchors:
            raise ValueError(f"anchor '{s['anchor']}' not found: available are {', '.join(sorted(field.anchors)) or 'no anchors'}")
        a = field.anchors[s["anchor"]]; x, y = float(a[0]), float(a[1]); g = float(field.ground(x, y))
        top = max(float(a[2]) if len(a) > 2 else g, float(field.top(x, y)))   # top: the anchor or the field (masonry, cornice) — whichever is higher
        h = float(s.get("h") or max(2.0, top - g))
    elif s.get("point") is not None:
        x, y = (float(v) for v in list(s["point"])[:2]); g = float(field.ground(x, y)); h = float(s.get("h", 1.8))
    else:
        raise ValueError("move subject: subject {anchor | point} or the hero's track (track)")
    kind = "landmark" if h > 6 else "hero"
    return {"pos": np.array([x, y, g]), "h": h, "yaw": float(s.get("yaw", 0.0)), "kind": kind,
            "r": float(s.get("r") or (footprint(field, x, y, h) if kind == "landmark" else 0.4)),
            "vis": float(s.get("vis", 0.9 if kind == "landmark" else 0.0)) or None}   # a landmark reads by its silhouette: visibility is measured at the top


def vis_point(subj, aim):
    """Visibility check point: for a landmark — the top (vis, a fraction of height), for a hero — the frame's aim."""
    return F.aim_point(subj, subj.get("vis") or aim)


def footprint(field, x, y, h):
    """Radius of a landmark (a tower, a keep, a house): over 16 directions — where the field top falls below a third of the height above the ground;
    the median. A landmark's visibility is measured up to its edge, not to the middle (the aim is inside the building)."""
    a = np.linspace(0, 2 * math.pi, 16, endpoint=False); d = np.arange(0.0, 40.0, 0.5)
    X = x + np.cos(a)[:, None] * d[None]; Y = y + np.sin(a)[:, None] * d[None]
    low = field.top(X, Y) < field.ground(X, Y) + h / 3.0
    r = np.where(low.any(1), d[np.argmax(low, 1)], 40.0)
    return float(np.median(r))


# ── candidates ──────────────────────────────────────────────────────────────────────────
def _height(P, rig, field):
    """Height of a candidate by rig (without path smoothing — that is in rig.project)."""
    g = field.ground(P[:, 0], P[:, 1])
    if rig in ("steadicam", "handheld"):
        P[:, 2] = g + np.clip(P[:, 2] - g, 0.6, 2.0)
    elif rig == "dolly":
        P[:, 2] = g + np.clip(P[:, 2] - g, 0.5, 2.5)
    elif rig == "tripod":
        P[:, 2] = np.maximum(P[:, 2], g + 0.46)
    elif rig == "drone":
        P[:, 2] = np.maximum(P[:, 2], field.envelope(P, r=5.0, window=1) + 3.0)
    return P


def evaluate(subj, fr, field, rig, yaws, aspect=16 / 9, dist=None):
    """Candidates by yaw (° from the subject's heading): camera positions, whether valid (free and aim visible), the background behind the subject."""
    fr = F.full(fr, subj["h"]); dist = np.ones(len(yaws)) if dist is None else np.asarray(dist, float)
    P = np.array([F.place(subj, dict(fr, yaw=float(y), size=fr["size"] / q), aspect)["pos"] for y, q in zip(yaws, dist)])
    P = _height(P, rig, field)
    aim = F.aim_point(subj, fr["aim"]); A = np.repeat(vis_point(subj, fr["aim"])[None], len(P), 0)
    free = field.clear(P, r=0.7) > 0.3
    if rig in ("steadicam", "handheld", "dolly", "tripod"):           # ground rig: not inside an obstacle and not on a steep slope (a person cannot stand there)
        free &= field.kind_at(P[:, 0], P[:, 1]) == 0
        if field.G is not None:
            free &= field.G.slope(P[:, 0], P[:, 1]) <= 30.0
    ok = free & field.sees(P, A, skip_end=subj.get("r", 0.4) + 1.5)
    bg = field.behind(P, A, skip=subj.get("r", 0.4) + 2.0)
    return P, ok, [b + (seen, sh) for b, seen, sh in zip(bg, _in_frame(subj, fr, field, P, aim, aspect), _shares(subj, fr, field, P, aim, aspect))]


def _shares(subj, fr, field, P, aim, aspect, nx=12, ny=7):
    """What is in the frame behind the subject: label shares over a grid of rays across the whole frame (behind the subject means farther than it by
    a radius): sky, far, ground, wall, tower, building, tree. The frame's background over the whole field of view, not along a single ray
    (a hill hides the base of a castle while the towers show above it)."""
    from components.camera.move.field import KINDS, FAR
    u = (np.arange(nx) + 0.5) / nx; v = (np.arange(ny) + 0.5) / ny; U, V = (a.ravel() for a in np.meshgrid(u, v))
    out = []
    for c in P:
        look = F.look_for(c, aim, fr["sx"], fr["sy"], fr["lens"], aspect); r, up, f = F.basis(c, look); th, tv = F.tans(fr["lens"], aspect)
        D = f[None] + r[None] * ((U - 0.5) * 2 * th)[:, None] - up[None] * ((V - 0.5) * 2 * tv)[:, None]
        t, _, lab = field.cast(np.repeat(c[None], len(D), 0), D)
        dist = t * np.linalg.norm(D, axis=1); behind = dist > float(np.linalg.norm(aim - c)) + subj.get("r", 0.4)
        names = np.where(lab > 0, np.array(KINDS, dtype=object)[np.clip(lab, 0, len(KINDS) - 1)], np.where(np.isfinite(t), np.where(dist > FAR, "far", "ground"), "sky"))
        sh, us = {}, {}
        for nme, b, uu in zip(names, behind | ~np.isfinite(t), U):
            if b: sh[nme] = sh.get(nme, 0.0) + 1.0 / len(D); us.setdefault(nme, []).append(float(uu))
        out_k = {k: round(x, 3) for k, x in sh.items()}
        for k, xs in us.items():                                           # where the background is in the frame: the middle and the cut by the side edge
            out_k[f"{k}@x"] = round(float(np.mean(xs)), 2); out_k[f"{k}@cut"] = bool(min(xs) < u[1] or max(xs) > u[-2])
        out.append(out_k)
    return out


def _placed(b, want):
    """How the desired background sits in the frame: +1 — not cut by a side edge, +0.5 — the middle is in the central part (0.2–0.8). For labels only."""
    if want is None or len(b) < 5 or b[4].get(want, 0.0) < SHARE:
        return 0.0
    return (0.0 if b[4].get(f"{want}@cut") else 1.0) + (0.5 if 0.2 <= b[4].get(f"{want}@x", 0.5) <= 0.8 else 0.0)


def _in_frame(subj, fr, field, P, aim, aspect):
    """Anchors in the candidate's frame that are behind the subject (farther than it by a radius) and not occluded — the frame's background over the whole field of view."""
    names = [n for n in field.anchors if np.hypot(field.anchors[n][0] - subj["pos"][0], field.anchors[n][1] - subj["pos"][1]) > subj.get("r", 0.4) + 3]
    if not names:
        return [()] * len(P)
    Q = np.array([field.anchors[n][:3] if len(field.anchors[n]) > 2 else [*field.anchors[n][:2], float(field.ground(*field.anchors[n][:2]))]
                  for n in names], float)
    out = []
    for c in P:
        look = F.look_for(c, aim, fr["sx"], fr["sy"], fr["lens"], aspect); x, y, z = F.project(c, look, fr["lens"], aspect, Q)
        far = z > float(np.linalg.norm(aim - c)) + subj.get("r", 0.4)
        inside = (x > 0.02) & (x < 0.98) & (y > 0.02) & (y < 0.98) & far
        vis = field.sees(np.repeat(c[None], len(Q), 0), Q, skip_end=3.0) if inside.any() else np.zeros(len(Q), bool)
        out.append(tuple(n for n, i, v in zip(names, inside, vis) if i and v))
    return out


SHARE = 0.03                                                         # the background is "in the frame": the label takes ≥ 3 % of the frame behind the subject


def _match(b, want):
    """The candidate's background matches the wish: a label (sky, far, ground, wall, tower, building, tree) is on the ray behind the subject or
    covers ≥ 3 % of the frame behind it; an anchor is behind the subject on the ray or in the frame beyond the subject."""
    return (want is None or want == b[0] or want == b[2] or (len(b) > 3 and want in b[3])
            or (len(b) > 4 and b[4].get(want, 0.0) >= SHARE))


def choose_yaw(subj, fr, field, rig, aspect=16 / 9, want=None, prefer=40.0, ends=()):
    """The best yaw for a frame at the place: valid, the background as wished, closest to prefer° from the face (the action reads best at
    30–120° from the face; 3/4 by default); ends — frame overrides for other points of the move (end, middle): all must be valid. → (yaw, background)
    or (None, None)."""
    yaws = np.arange(-180.0, 180.0, STEP)
    P, ok, bg = evaluate(subj, fr, field, rig, yaws, aspect)
    for e in ends:                                                     # the whole move must be valid, not only frame A
        ok = ok & evaluate(subj, dict(fr, **e), field, rig, yaws, aspect)[1]
    best = None
    for y, o, b in zip(yaws, ok, bg):
        if not o: continue
        s = 3.0 * _match(b, want) + _placed(b, want) - abs(abs(y) - prefer) / 90.0 - 0.5 * (b[0] == "sky")
        if best is None or s > best[0]: best = (s, float(y), b)
    return (best[1], best[2]) if best else (None, None)


# ── recipes ─────────────────────────────────────────────────────────────────────────────
def _base(spec, subj):
    keys = ("size", "yaw", "pitch", "sx", "sy", "aim", "lens")
    return F.full({k: spec.get(k) for k in keys}, subj["h"])


def _yaw(subj, fr, field, rig, spec, meta, ends=()):
    """Frame yaw by the place, if not set by a knob: the best valid one (choose_yaw; ends — other points of the move); if none is valid — the default,
    with the reason in the details."""
    if spec.get("yaw") is not None:
        return
    y, b = choose_yaw(subj, fr, field, rig, want=spec.get("bg_a"), prefer=float(spec.get("prefer_yaw", 40.0)), ends=ends); meta["bg"] = b
    if y is None:
        meta["fallback"] = f"no valid frame position for {rig} (free, not steep, aim visible) — default yaw; change the rig or the place"
    else:
        fr["yaw"] = y


def _framed(subj, fa, fb, wrap=True):
    """Pose by frame: the frame values between A and B (frame.lerp), the camera — frame.place."""
    def pose(u):
        f = F.lerp(fa, fb, u, wrap); c = F.place(subj, f); c["frame"] = f
        return c
    return pose


def static(subj, field, spec, prof, rig, k=1.0):
    fr = _base(spec, subj); meta = {}
    _yaw(subj, fr, field, rig, spec, meta)
    fb = dict(fr, size=fr["size"] * (1.0 + float(spec.get("drift", 0.0)) * k))   # drift — a "live" hold (< 1 %/s)
    return _framed(subj, fr, fb), meta


def push(subj, field, spec, prof, rig, k=1.0, sign=1):
    fr = _base(spec, subj); meta = {}
    m = 1.0 + float(spec.get("amount", 0.3)) * k; e = m if sign > 0 else 1.0 / m
    _yaw(subj, fr, field, rig, spec, meta, ends=({"size": fr["size"] * e}, {"size": fr["size"] * math.sqrt(e)}))
    fb = dict(fr, size=fr["size"] * (m if sign > 0 else 1.0 / m))
    if spec.get("pitch_b") is not None: fb["pitch"] = float(spec["pitch_b"])
    return _framed(subj, fr, fb), meta


def zoom(subj, field, spec, prof, rig, k=1.0):
    """Zoom: the position stays, the lens ×factor; hold_size — the "Hitchcock effect": a push-in with a counter-zoom, the shot size stays the same."""
    fr = _base(spec, subj); meta = {}; f = 1.0 + (float(spec.get("factor", 1.5)) - 1.0) * k
    _yaw(subj, fr, field, rig, spec, meta, ends=({"lens": fr["lens"] * f},) if spec.get("hold_size") else ())
    c0 = F.place(subj, fr); P = F.aim_point(subj, fr["aim"])
    d0 = float(np.linalg.norm(c0["pos"] - P)); hold = bool(spec.get("hold_size"))
    def pose(u):
        lens = fr["lens"] * f ** u
        C = P + (c0["pos"] - P) * (lens / fr["lens"] if hold else 1.0)
        return {"pos": C, "look": F.look_for(C, P, fr["sx"], fr["sy"], lens), "lens": lens, "frame": dict(fr, lens=lens)}
    meta["dist"] = [d0, d0 * f if hold else d0]
    return pose, meta


def truck(subj, field, spec, prof, rig, k=1.0):
    """A sideways shift, the gaze stays parallel (parallax): dist — m (by default 15 % of the distance to the aim), side +1 — to the right."""
    fr = _base(spec, subj); meta = {}
    _yaw(subj, fr, field, rig, spec, meta)
    c0 = F.place(subj, fr); r, _, _ = F.basis(c0["pos"], c0["look"])
    D = float(spec.get("dist") or 0.15 * np.linalg.norm(c0["pos"] - F.aim_point(subj, fr["aim"]))) * float(spec.get("side", 1)) * k
    def pose(u):
        return {"pos": c0["pos"] + r * D * u, "look": c0["look"] + r * D * u, "lens": fr["lens"]}
    meta["dist_m"] = D
    return pose, meta


def pan(subj, field, spec, prof, rig, k=1.0):
    """Pan from the spot: to a second subject (to: anchor | x,y[,z]) or by deg° (+ — to the left); the camera stands still (tripod)."""
    fr = _base(spec, subj); meta = {}
    _yaw(subj, fr, field, rig, spec, meta)
    c0 = F.place(subj, fr); C = c0["pos"]; L0 = c0["look"]
    if spec.get("to") is not None:
        t = spec["to"]
        tp = np.asarray(field.anchors[t], float) if isinstance(t, str) else np.asarray([float(v) for v in t], float)
        if len(tp) < 3: tp = np.array([tp[0], tp[1], float(field.ground(tp[0], tp[1])) + 1.5])
        L1 = C + (tp - C) / np.linalg.norm(tp - C) * np.linalg.norm(L0 - C)
    else:
        a = math.radians(float(spec.get("deg", 30.0))); v = L0 - C
        L1 = C + np.array([v[0] * math.cos(a) - v[1] * math.sin(a), v[0] * math.sin(a) + v[1] * math.cos(a), v[2]])
    v0, v1 = (L0 - C) / np.linalg.norm(L0 - C), (L1 - C) / np.linalg.norm(L1 - C); dd = float(np.linalg.norm(L0 - C))
    ang = math.acos(max(-1.0, min(1.0, float(v0 @ v1)))) * k
    def pose(u):
        if ang < 1e-9: return {"pos": C, "look": L0, "lens": fr["lens"]}
        w = math.acos(max(-1.0, min(1.0, float(v0 @ v1))))
        t = ang * u / w; v = (math.sin((1 - t) * w) * v0 + math.sin(t * w) * v1) / math.sin(w)   # spherical interpolation
        return {"pos": C, "look": C + v * dd, "lens": fr["lens"]}
    meta["pan_deg"] = round(math.degrees(ang), 1)
    return pose, meta


def crane(subj, field, spec, prof, rig, k=1.0):
    """Crane: dir down — from above, from a wide shot down to the subject (the start of a scene, Brown); up — from below upward with a pull-away
    (the end of a scene, the reveal). The tilt and shot size of the ends are the knobs pitch_a, pitch_b, size_a, size_b."""
    fr = _base(spec, subj); down = spec.get("dir", "down") == "down"
    fa = dict(fr, pitch=float(spec.get("pitch_a", 35.0 if down else 2.0)), size=float(spec.get("size_a", fr["size"] * (0.6 if down else 1.0))))
    fb = dict(fr, pitch=float(spec.get("pitch_b", 6.0 if down else 30.0)), size=float(spec.get("size_b", fr["size"] * (1.0 if down else 0.6))))
    meta = {}
    if spec.get("yaw") is None:                                       # both ends must be valid
        yaws = np.arange(-180.0, 180.0, STEP)
        _, oka, bga = evaluate(subj, fa, field, rig, yaws); _, okb, _ = evaluate(subj, fb, field, rig, yaws)
        good = [(abs(abs(y) - 40.0), float(y), b) for y, a, b_, b in zip(yaws, oka, okb, bga) if a and b_]
        if good:
            _, y, b = min(good); fa["yaw"] = fb["yaw"] = y; meta["bg"] = b
    fb = F.lerp(fa, fb, k)
    return _framed(subj, fa, fb), meta


def arc(subj, field, spec, prof, rig, k=1.0):
    """Orbit is an arc, not a circle (measured: the median arc is 30°, > 180° in 1 % of moves; Brown: circling without a reason looks unnatural).
    Sectors around the subject every 10°: whether valid (free, aim visible), what is behind the subject. The arc goes from a sector with background bg_a to
    a sector with background bg_b by the shortest valid path (sweep — no longer than 180°); without backgrounds — an arc of length sweep (from the profile), where the background
    changes most. The radius stays in a ± 30 % corridor (free and visible), smoothed; rise — the tilt rise over the arc (a spiral), °."""
    fr = _base(spec, subj); lo, hi = (float(v) for v in (spec.get("sweep") or prof.get("arc") or (20, 60)))
    wa, wb = spec.get("bg_a"), spec.get("bg_b"); start = spec.get("yaw")
    yaws = np.arange(-180.0, 180.0, STEP); n = len(yaws)
    P, ok, bg = evaluate(subj, fr, field, rig, yaws)
    pref = lo if wb is not None else (lo + hi) / 2
    best = None
    for i in range(n):
        if not ok[i] or not _match(bg[i], wa): continue
        if start is not None and abs((yaws[i] - float(start) + 180.0) % 360.0 - 180.0) > STEP / 2: continue
        for d in (1, -1):
            for m in range(1, n):
                j = (i + d * m) % n
                if not ok[j]: break                                    # the path is blocked
                sw = m * STEP
                if sw > (180.0 if wb is not None else hi): break
                if sw < lo or not _match(bg[j], wb): continue
                change = (bg[i][0] != bg[j][0] or bg[i][2] != bg[j][2] or set(bg[i][3]) != set(bg[j][3])
                          or sum(abs(bg[i][4].get(k, 0) - bg[j][4].get(k, 0)) for k in set(bg[i][4]) | set(bg[j][4])) > 0.3)
                s = (2.0 * change - abs(sw - pref) / 100.0 + (3.0 if wb is not None else 0.0)
                     + _placed(bg[i], wa) + _placed(bg[j], wb))                 # the background is not cut by an edge, its middle is not at the edge
                if best is None or s > best[0]: best = (s, i, d, m)
    meta = {"candidates_ok": int(ok.sum()), "of": n}
    if best is None:                                                  # no valid arc — a hold from the best position
        meta["fallback"] = (f"no valid arc for {rig}: {int(ok.sum())} of {n} positions are valid (steepness, walls, occlusion, background) — a hold; "
                            "change the rig (drone, crane), the background or the place")
        return static(subj, field, dict(spec, yaw=None), prof, rig)[0], meta
    _, i, d, m = best; sweep = d * m * STEP * k
    ys = yaws[i] + np.linspace(0.0, sweep, max(2, int(abs(sweep) / STEP) + 1))
    q = np.ones(len(ys))                                              # the radius in the corridor: closer to 1 is better
    for t, y in enumerate(ys):
        for f in (1.0, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15, 0.8, 1.2, 0.75, 1.25, 0.7, 1.3):
            _, o, _ = evaluate(subj, fr, field, rig, np.array([y]), dist=np.array([f]))
            if o[0]: q[t] = f; break
    if len(q) >= 3:
        q = np.convolve(np.pad(q, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    rise = float(spec.get("rise", 0.0)) * k
    def pose(u):
        x = u * (len(ys) - 1); a = int(min(len(ys) - 2, math.floor(x))); w = x - a
        qq = q[a] * (1 - w) + q[a + 1] * w
        f = dict(fr, yaw=float(ys[0] + (ys[-1] - ys[0]) * u), size=fr["size"] / qq, pitch=fr["pitch"] + rise * u)
        c = F.place(subj, f); c["frame"] = f
        return c
    meta.update({"from": float(yaws[i]), "to": float(yaws[i] + sweep), "sweep": round(float(sweep), 1), "bg": [bg[i], bg[(i + d * m) % n]],
                 "radius": [round(float(q.min()), 2), round(float(q.max()), 2)]})
    return pose, meta


def follow(track, field, spec, prof, rig, fps=24.0, aspect=16 / 9):
    """Following along the subject's track: the frame is kept relative to the smoothed heading (1.5 s window), the side comes from the knob side
    (° from the heading: 90 — left, −90 — right, 0 — front, 180 — behind) or is the one with fewer blocked frames (from sides); in a long follow the side changes
    by 6 s windows if it is cramped on its own — smoothly over 2 s, the nearest or a neutral one first; the gaze has a dead zone of ± dz of the frame
    and damping tau s. → positions, gazes, lens, aims per frame, and the details."""
    Pt = np.asarray(track["pos"], float); n = len(Pt); h = float(track.get("h", 1.8)); fr = _base(spec, {"h": h})
    w = max(1, int(round(1.5 * fps))) | 1
    pad = np.pad(Pt, ((w // 2, w // 2), (0, 0)), mode="edge"); Ps = np.array([pad[i:i + w].mean(0) for i in range(n)])
    v = np.gradient(Ps[:, :2], axis=0); sp = np.linalg.norm(v, axis=1)
    yaw0 = np.asarray(track.get("yaw", np.zeros(n)), float)
    head = np.where(sp > 0.02 / fps, np.degrees(np.arctan2(v[:, 1], v[:, 0])), yaw0)
    head = np.degrees(np.unwrap(np.radians(head)))
    sides = ([float(spec["side"])] if spec.get("side") is not None else [float(x) for x in spec["sides"]] if spec.get("sides")
             else [90.0, -90.0, 30.0, -30.0, 150.0, -150.0, 0.0, 180.0])   # sides — the allowed sides (following from the side — ±30…90)
    A = np.array([F.aim_point({"pos": Pt[t], "h": h}, fr["aim"]) for t in range(n)])
    def places(yaws):                                                  # camera positions by the frames' yaw (° from the heading) and the validity of the positions
        C = _height(np.array([F.place({"pos": Ps[t], "h": h, "yaw": head[t]}, dict(fr, yaw=float(yaws[t])), aspect)["pos"] for t in range(n)]), rig, field)
        good = (field.clear(C, r=0.7) > 0.3) & field.sees(C, A)
        if rig in ("steadicam", "handheld", "dolly", "tripod") and field.G is not None:
            good &= field.G.slope(C[:, 0], C[:, 1]) <= 30.0               # the operator does not stand on a steep slope
        return C, good
    G_ = {s: places(np.full(n, s)) for s in sides}
    win = int(round(6.0 * fps)); switch = spec.get("side") is None and n >= 2 * win
    first = (lambda s: G_[s][1][:win].mean()) if switch else (lambda s: 0.0)   # a long follow — the start is taken from the first window
    side = max(sides, key=lambda s: (first(s), G_[s][1].mean(), -abs(abs(s) - 90.0))); yaw = np.full(n, side); plan = [(0.0, side)]
    best1 = max(sides, key=lambda s: (G_[s][1].mean(), -abs(abs(s) - 90.0)))   # one best side for the whole follow (on a tie — from the side)
    if switch:                                                         # a long follow: the side is chosen by 6 s windows and changes only if
        cur = side                                                     # it became cramped on its own side (Bonatti: a new angle once per ~12 s, not abruptly)
        for w0 in range(win, n, win):
            sl = slice(w0, min(n, w0 + win))
            if G_[cur][1][sl].mean() >= 0.9: continue
            alt = [s for s in sides if s != cur and G_[s][1][sl].mean() >= 0.98]
            if not alt: continue
            rank = lambda s: (0 if s * cur > 0 else 1 if s in (0.0, 180.0) or cur in (0.0, 180.0) else 2, abs(s - cur))
            new = min(alt, key=rank)                                   # the nearest side, then a neutral one (front, back), the opposite one last
            yaw[w0:] = new; plan.append((round(w0 / fps, 2), new)); cur = new
        if len(plan) > 1:                                              # the transition is smooth over 2 s (Brown: via a neutral angle)
            ramp = int(round(2.0 * fps)); k = np.convolve(np.pad(yaw, ramp // 2, mode="edge"), np.ones(ramp + 1) / (ramp + 1), mode="valid")
            yaw = k[:n] if len(k) >= n else np.pad(k, (0, n - len(k)), mode="edge")
    C, good = places(yaw)
    if good.mean() < G_[best1][1].mean():                              # switching sides is worse than one best side — keep one side
        side = best1; yaw = np.full(n, side); C, good = G_[side]; plan = [(0.0, side)]
    share = float(good.mean())
    dz, tau = float(spec.get("dead_zone", 0.05)), float(spec.get("tau", 0.35)); alpha = 1.0 - math.exp(-1.0 / (fps * tau))
    L = [F.look_for(C[0], A[0], fr["sx"], fr["sy"], fr["lens"], aspect)]
    for t in range(1, n):
        x, y, _ = F.project(C[t], L[-1], fr["lens"], aspect, A[t])
        if abs(float(x) - fr["sx"]) <= dz and abs(float(y) - fr["sy"]) <= dz:
            L.append(L[-1] + (C[t] - C[t - 1]))                         # inside the dead zone the gaze does not turn
        else:
            Lt = F.look_for(C[t], A[t], fr["sx"], fr["sy"], fr["lens"], aspect)
            L.append(L[-1] + (C[t] - C[t - 1]) + alpha * (Lt - (L[-1] + (C[t] - C[t - 1]))))
    return C, np.array(L), fr["lens"], A, {"side": side, "sides": plan, "free_share": round(share, 3)}
