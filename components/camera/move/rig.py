"""Rig: the camera path is laid onto the rig and the relief with limits; operator overlays (a jolt, a hand from a recording of a live camera).

tripod — the position of frame A, the lens no lower than 0.46 m above the ground (Brown: a hi-hat with a head); dolly — the rail is levelled:
the height is constant (± the boom, linearly), a ground drop under the rail > 0.6 m — "will not fit" [hyp]; crane — a boom arc around a pivot
on the ground (1.5 m), a boom up to 9 m, a telescope up to 15 m, beyond that — "cannot reach" (Brown pp. 315–316); steadicam, handheld — the smoothed
ground + 0.6…2.0 m, slope ≤ 30°; drone — above the envelope of the tops + 3 m [hyp], smoothed; virtual — as is. Everywhere — no closer than 0.3 m to
the top of an obstacle. Pure numpy."""
import math

import numpy as np

RIGS = ("tripod", "dolly", "crane", "steadicam", "handheld", "drone", "virtual")
TRIPOD_MIN, DOLLY_LEVEL, CRANE_ARM, CRANE_TELE = 0.46, 0.6, 9.0, 15.0
OPERATOR_H, SLOPE, DRONE_CLEAR, MIN_CLEAR = (0.6, 2.0), 30.0, 3.0, 0.3


def _smooth(v, w):
    """Moving average over a window of w points (the edges are padded with the edge value)."""
    v = np.asarray(v, float)
    if w < 2 or len(v) < 3:
        return v
    h = int(w) // 2; pad = np.pad(v, h, mode="edge")
    return np.convolve(pad, np.ones(2 * h + 1) / (2 * h + 1), mode="valid")


def _window(P, metres):
    """Smoothing window in path points for the given move length."""
    L = float(np.sum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))) if len(P) > 1 else 0.0
    return max(1, int(round(metres / max(L / max(len(P) - 1, 1), 1e-6)))) if L > 0 else 1


def _arclen(P):
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return s / s[-1] if s[-1] > 1e-9 else np.linspace(0, 1, len(P))


def project(pos, rig, field, look=None):
    """Desired camera positions (N×3) → positions on the rig and "what is wrong" notes (strings)."""
    if rig not in RIGS:
        raise ValueError(f"rig '{rig}': available are {', '.join(RIGS)}")
    P = np.array(pos, float); notes = []
    g = field.ground(P[:, 0], P[:, 1])
    if rig == "tripod":
        P[:] = P[0]; P[:, 2] = max(P[0, 2], float(g[0]) + TRIPOD_MIN)
    elif rig == "dolly":
        rail = float(g.max()); spread = float(g.max() - g.min())
        if spread > DOLLY_LEVEL:
            notes.append(f"rail will not fit: ground drop under it {spread:.1f} m > {DOLLY_LEVEL} m — use a steadicam or a crane")
        za, zb = (min(2.5, max(0.5, P[i, 2] - rail)) for i in (0, -1))
        P[:, 2] = rail + za + (zb - za) * _arclen(P)
    elif rig == "crane":
        P = _crane(P, field, notes)
    elif rig in ("steadicam", "handheld"):
        w = _window(P, 1.5); gs = _smooth(g, w)
        P[:, 2] = gs + _smooth(np.clip(P[:, 2] - gs, *OPERATOR_H), w)
        if field.G is not None:
            steep = field.G.slope(P[:, 0], P[:, 1]) > SLOPE
            if steep.any():
                notes.append(f"{rig}: slope > {SLOPE:.0f}° under {int(steep.sum())} of {len(P)} positions — use a drone or a crane")
        inside = field.kind_at(P[:, 0], P[:, 1]) > 0
        if inside.any():
            notes.append(f"{rig}: the operator is inside an obstacle at {int(inside.sum())} of {len(P)} positions")
    elif rig == "drone":
        w = _window(P, 6.0); need = field.envelope(P, r=5.0, window=max(3, w)) + DRONE_CLEAR
        Lxy = float(np.sum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))) if len(P) > 1 else 0.0
        if Lxy < 150.0:                                               # orbit, push-in, lift: the pilot holds the altitude — a single lift above
            lift = max(0.0, float((need - P[:, 2]).max()))            # the highest obstacle of the path, the intended height profile of the move stays intact
            P[:, 2] += lift
            if lift > 0.05: notes.append(f"drone raised by {lift:.1f} m above the path obstacles")
        else:                                                         # fly-through: above the envelope of the tops, smoothed
            z = np.maximum(P[:, 2], need)
            for _ in range(2):
                z = np.maximum(_smooth(z, w), need)
            P[:, 2] = z
    c = field.clear(P, r=0.5); need = np.where(c < MIN_CLEAR, MIN_CLEAR - c, 0.0)   # common limit: not inside an obstacle
    if need.any():
        n = int((need > 0).sum())
        if rig in ("tripod", "dolly", "steadicam", "handheld"):       # a ground rig does not pass through a wall and does not rise 17 m
            P[:, 2] += np.minimum(need, 1.0)
            if (need > 1.0).any():
                notes.append(f"{rig}: the path goes through an obstacle at {int((need > 1.0).sum())} of {len(P)} positions — choose another place or rig")
            else:
                notes.append(f"{rig}: the camera is raised above an obstacle at {n} of {len(P)} positions (up to {float(need.max()):.1f} m)")
        else:                                                         # from the air the lift is smooth (a moving maximum and mean)
            h = max(1, _window(P, 6.0) // 2)
            lift = np.array([need[max(0, i - h):i + h + 1].max() for i in range(len(need))])
            lift = np.maximum(_smooth(lift, 2 * h + 1), need); P[:, 2] += lift
            notes.append(f"{rig}: raised smoothly above an obstacle at {n} of {len(P)} positions (up to {float(lift.max()):.1f} m)")
    return P, notes


def _crane(P, field, notes):
    """Boom arc: a pivot on the ground in the vertical plane of the move, equidistant from the ends (the chord midpoint → a perpendicular down to
    the pivot height); a constant radius is a boom, otherwise a telescope (the radius changes linearly from end to end)."""
    A, B = P[0], P[-1]; dxy = B[:2] - A[:2]; L = float(np.linalg.norm(dxy))
    gz = float(field.ground(*((A[:2] + B[:2]) / 2))) + 1.5
    e = dxy / L if L > 1.0 else np.array([1.0, 0.0])
    piv = None
    if L > 1.0:                                                       # constant boom length: the pivot lies on the perpendicular to chord AB
        M = (A + B) / 2; dz = B[2] - A[2]; s = -(dz * (gz - M[2])) / L
        if abs(s) >= L / 2 + 0.5:                                     # the pivot is behind the start or beyond the end — an arc without a crest
            piv = np.array([M[0] + e[0] * s, M[1] + e[1] * s, gz])
    if piv is None:                                                   # otherwise the boom would swing over its crest — the pivot behind the start, a telescope
        piv = np.array([A[0] - e[0] * max(3.0, L / 2), A[1] - e[1] * max(3.0, L / 2), gz])
    ra, rb = np.linalg.norm(A - piv), np.linalg.norm(B - piv)
    arm = max(ra, rb)
    if arm > CRANE_TELE:
        notes.append(f"crane cannot reach: boom {arm:.1f} m > {CRANE_TELE:.0f} m — use a drone")
    elif arm > CRANE_ARM or (arm > 4.0 and abs(ra - rb) > 0.15 * arm):
        notes.append(f"telescoping crane: boom {min(ra, rb):.1f}…{arm:.1f} m")
    s = _arclen(P); va, vb = A - piv, B - piv
    ang = math.atan2(np.linalg.norm(np.cross(va, vb)), float(va @ vb))
    axis = np.cross(va, vb); axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-9 else np.array([0.0, 0.0, 1.0])
    out = []
    for si in s:                                                      # rotation of va about the axis by a fraction of the angle (Rodrigues), the radius linearly
        a = ang * si; v = va / ra
        v = v * math.cos(a) + np.cross(axis, v) * math.sin(a) + axis * (axis @ v) * (1 - math.cos(a))
        out.append(piv + v * (ra + (rb - ra) * si))
    return np.array(out)


def shake(looks, pos, at, fps=24.0, amp_deg=0.4, tau=0.1, dur=0.35, seed=0):
    """A jolt (a hit, an explosion) at frames at: the gaze jerks by amp_deg with a decay exp(−t/tau) over dur s (the default preset: 9 px at 1280,
    0.35 s). → new look-at points."""
    L = np.array(looks, float); P = np.asarray(pos, float); rng = np.random.default_rng(seed)
    for k0 in at:
        ph = rng.uniform(0, 2 * math.pi, 2)
        for k in range(int(k0), min(len(L), int(k0) + int(dur * fps) + 1)):
            t = (k - k0) / fps; a = math.radians(amp_deg) * math.exp(-t / tau)
            L[k] = _turn(P[k], L[k], a * math.sin(2 * math.pi * 14.0 * t + ph[0]), a * math.sin(2 * math.pi * 11.0 * t + ph[1]))
    return L


_SHAKE = None


def handheld(looks, pos, fps=24.0, style="the_closeup", amp=1.0, seed=0):
    """A hand from a recording of a live handheld camera (CC0 data — data/shake.json): the_closeup (quiet, 1.3° swing),
    the_wedding, walk_to_the_store, investigation (with drift), handycam_run (running). Rotation — tilt (rot_x) and sideways (rot_y) about the
    camera axes, roll is not taken (a camera without roll); displacement — along the camera axes; amp — the strength relative to the recording;
    the start of the recording comes from the seed, the recording repeats in a loop. → (positions, look-at points)."""
    global _SHAKE
    if _SHAKE is None:
        import json, os
        _SHAKE = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shake.json")))["shakes"]
    if style not in _SHAKE:
        raise ValueError(f"hand '{style}': available are {', '.join(_SHAKE)}")
    from components.camera.move.frame import basis
    rec = _SHAKE[style]; m = len(rec["rot_x"]); rfps = float(rec["fps"])
    cur = {k: np.asarray(rec[k], float) - np.mean(rec[k]) for k in ("rot_x", "rot_y", "loc_x", "loc_y", "loc_z") if k in rec}
    t0 = np.random.default_rng(seed).uniform(0, m)
    L = np.array(looks, float); P = np.array(pos, float)
    for k in range(len(L)):
        x = (t0 + k * rfps / fps) % m; i = int(x); w = x - i; j = (i + 1) % m
        g = lambda c: float(cur[c][i] * (1 - w) + cur[c][j] * w) * amp if c in cur else 0.0
        r, u, f = basis(P[k], L[k]); d = r * g("loc_x") + u * g("loc_y") - f * g("loc_z")
        P[k] = P[k] + d; L[k] = _turn(P[k], L[k] + d, g("rot_y"), g("rot_x"))
    return P, L


def _turn(pos, look, yaw, pitch):
    """The gaze turned by small angles (rad): sideways and up-down; the distance to the look-at point stays the same."""
    from components.camera.move.frame import basis
    r, u, f = basis(pos, look); d = float(np.linalg.norm(np.asarray(look) - np.asarray(pos)))
    return np.asarray(pos) + d * (f + r * math.tan(yaw) + u * math.tan(pitch)) / math.sqrt(1 + math.tan(yaw) ** 2 + math.tan(pitch) ** 2)
