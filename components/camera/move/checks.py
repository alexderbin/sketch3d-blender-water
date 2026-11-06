"""Checks of a move by numbers, without rendering: aim visibility, clearance, edge speed against the format ceiling, jerk, going around a subject,
the hold at the end, "the move changes nothing". → (numbers, a list of what is wrong)."""
import math

import numpy as np

from components.camera.move import timing as T

JUMP_M, JUMP_DEG = 36.0, 600.0                                       # jerk, per second: 1.5 m and 25° per frame at 24 fps (take checks)
CUT_DEG, CUT_SIZE, CUT_H = 30.0, 1.2, 0.5                           # cut: the point ≥ 30° or the shot size ≥ ×1.2 (Brown: 20 % and 30°);
                                                                     # one camera height per scene (± 0.5 m)


def cut(a, b, subject, axis=None, comic=False, aspect=16 / 9):
    """Cut between two shots of one subject: a — the last frame of shot A, b — the first frame of shot B ({pos, look, lens}); subject —
    the subject ({pos, h, yaw}); axis — the scene axis, ° (a line through the subject; the camera side is the sign, crossing only by a visible move);
    comic — the panels of a page (alternate the subject's screen position). → (numbers, "what is wrong")."""
    from components.camera.move import frame as F
    P = F.aim_point(subject, 0.6); bad = []
    ma, mb = (F.measure(c["pos"], c["look"], c["lens"], subject, 0.6, aspect) for c in (a, b))
    dyaw = abs((mb["yaw"] - ma["yaw"] + 180.0) % 360.0 - 180.0); ratio = max(ma["size"], mb["size"]) / max(min(ma["size"], mb["size"]), 1e-9)
    g = lambda c: float(c["pos"][2]) - float(subject["pos"][2])
    out = {"yaw_diff": round(dyaw, 1), "size_ratio": round(ratio, 2), "height_diff_m": round(abs(g(a) - g(b)), 2),
           "screen_x": [round(ma["sx"], 2), round(mb["sx"], 2)]}
    if dyaw < CUT_DEG and ratio < CUT_SIZE:
        bad.append(f"jump: the point changed by {dyaw:.0f}° (< {CUT_DEG:.0f}°), size ×{ratio:.2f} (< ×{CUT_SIZE})")
    if out["height_diff_m"] > CUT_H:
        bad.append(f"camera height changed by {out['height_diff_m']} m — one height per scene, a change only as an event")
    if axis is not None:
        n = np.array([-math.sin(math.radians(axis)), math.cos(math.radians(axis))])
        sa, sb = (float(np.sign((np.asarray(c["pos"], float)[:2] - np.asarray(subject["pos"], float)[:2]) @ n)) for c in (a, b))
        out["side"] = [sa, sb]
        if sa * sb < 0:
            bad.append("scene axis crossed by the cut — only by a visible move or via a neutral shot")
    if comic and abs(ma["sx"] - mb["sx"]) < 0.08 and ratio < CUT_SIZE:
        bad.append("the same screen position twice in a row at the same shot size — alternate")
    if not comic and abs(ma["sx"] - mb["sx"]) > 0.4:
        bad.append(f"the main subject jumped across the screen by {abs(ma['sx'] - mb['sx']):.2f} of the width — after the cut the eye looks for it for ⅓ s")
    return out, bad


def run(cams, aims, field, fps=24.0, aspect=16 / 9, prof=None, centre=None, skip=1.5):
    """cams — [{pos, look, lens}] per frame; aims — aim points per frame (N×3); prof — the format profile; centre — the subject of the circuit
    (x, y) for counting the arc; skip — the margin at the aim, m (for a landmark — its radius: visibility up to the edge, not to the middle)."""
    prof = prof or {}; P = np.array([c["pos"] for c in cams], float); L = np.array([c["look"] for c in cams], float)
    A = np.atleast_2d(np.asarray(aims, float)); A = np.repeat(A, len(P), 0) if len(A) == 1 else A
    vis = field.sees(P, A, skip_end=skip); clr = field.clear(P, r=0.5)
    out = {"frames": len(P), "visible": round(float(vis.mean()), 3), "clear_min": round(float(clr.min()), 2)}
    bad = []
    if out["visible"] < 0.95:
        bad.append(f"aim is blocked in {100 * (1 - out['visible']):.0f} % of frames")
    if out["clear_min"] < 0.25:
        bad.append(f"camera right at the top of an obstacle or inside it (clearance {out['clear_min']} m)")
    if len(P) > 1:
        v = T.speeds(cams, field, fps, aspect); out.update({k: round(x, 2) for k, x in v.items()})
        cs, cz = prof.get("edge_ceiling"), prof.get("scale_ceiling")
        if cs and v["shift_p90"] > cs:
            bad.append(f"frame edge p90 {v['shift_p90']:.1f} %/s > the ceiling {cs} %/s — shorten the move or lengthen the shot")
        if cz and v["scale_p90"] > cz:
            bad.append(f"scale p90 {v['scale_p90']:.1f} %/s > the ceiling {cz} %/s")
        step = np.linalg.norm(np.diff(P, axis=0), axis=1)
        F = (L - P) / np.linalg.norm(L - P, axis=1, keepdims=True)
        turn = np.degrees(np.arccos(np.clip(np.sum(F[1:] * F[:-1], 1), -1, 1)))
        out.update({"step_max_m": round(float(step.max()), 3), "turn_max_deg": round(float(turn.max()), 2)})
        if step.max() * fps > JUMP_M or turn.max() * fps > JUMP_DEG:          # the threshold is per second: at 6 fps the step per frame is four times larger
            bad.append(f"jerk: {step.max():.2f} m or {turn.max():.1f}° per frame at {fps:g} fps")
        still = (step < 1e-4) & (turn < 1e-3)
        k = 0
        for s in still[::-1]:
            if not s: break
            k += 1
        out["hold_end_s"] = round(k / fps, 2)
        moved = float(np.linalg.norm(P[-1] - P[0])); turned = float(turn.sum())
        if moved < 0.05 and turned < 0.5 and abs(float(cams[-1]["lens"]) - float(cams[0]["lens"])) < 0.1:
            out["still"] = True
        if centre is not None:
            d = P[:, :2] - np.asarray(centre, float)[:2]; az = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
            out["arc_deg"] = round(float(np.degrees(np.ptp(az))), 1)
            out["radius_change"] = round(float(np.ptp(np.linalg.norm(d, axis=1)) / max(np.median(np.linalg.norm(d, axis=1)), 1e-6)), 3)
    return out, bad


def sequence(shots, prof):
    """Episode plan against a format profile: shots — [{sec, recipe}] (recipe static — a static shot); prof — the profile (shot_s,
    static_share). → (numbers, "what is wrong"): the share of static shots (by shot count), shots outside the format's length."""
    secs = [float(s["sec"]) for s in shots]; still = sum(1 for s in shots if s.get("recipe", "static") == "static")
    lo, hi = prof.get("shot_s") or (0.0, 1e9); share = still / max(len(shots), 1)
    out = {"shots": len(shots), "static_share": round(share, 2), "sec_median": round(float(np.median(secs)), 2) if secs else 0.0}
    bad = []
    want = prof.get("static_share")
    if want is not None and share < float(want) - 0.15:
        bad.append(f"static shots {share:.0%} against {float(want):.0%} for the format — the camera drifts all the time (in strong videos the camera stands still more often)")
    out_of = [i for i, x in enumerate(secs) if x < lo * 0.75 or x > hi * 1.5]
    if out_of:
        bad.append(f"shots outside the format length {lo:g}–{hi:g} s: {', '.join(str(i + 1) for i in out_of[:8])}")
    return out, bad
