"""Camera move — the mechanics: recipe → frames A and B by the place → path → rig → timing → operator → checks. Pure numpy: the place is
a 2.5D field from the layers (field.py), Blender is not needed.

shot(spec, scene, track) — a move: cameras per frame (cameras.json records with the fields frame, fps) and checks; panel(spec, scene) — a panel
frame (a move of zero length) with the same checks. spec is a dict of knobs: recipe (static | pan | push | pull | truck | arc | crane |
follow | zoom), subject {anchor | point, h, yaw}, frame values (size, yaw, pitch, sx, sy, aim, lens), rig (tripod | dolly |
crane | steadicam | handheld | drone | virtual), format (a profile of components/camera/data/profiles.yaml), role (defaults for the
frame and the rig), sec, fps, res "WxH", time and ramp, hold_end, recipe knobs (bg_a, bg_b, sweep, rise, amount, dist, side,
deg, to, dir, factor, hold_size, drift), shake_at (the frames of a jolt), hand, hand_amp, seed (the hand is a recording of a live camera), name (a prefix of the record names, default move)."""
import os

import numpy as np

from components.camera.move import checks as K
from components.camera.move import frame as F
from components.camera.move import recipes as R
from components.camera.move import rig as G
from components.camera.move import timing as T
from components.camera.move.field import Field

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALE_MOVES = ("push", "pull", "zoom", "crane", "static")
BROWN_STROBE = 20.0                                                   # % of width/s: a subject crosses the frame no faster than in 5 s (Brown, p. 303)
_PROF = None


def profiles():
    global _PROF
    if _PROF is None:
        import yaml
        _PROF = yaml.safe_load(open(os.path.join(HERE, "data", "profiles.yaml")))
    return _PROF


def profile(spec):
    """Format profile with the knobs' overrides (edge_ceiling, scale_ceiling, time, ramp, hold_end, arc, move_share)."""
    P = profiles(); name = spec.get("format") or "fiction"
    if name not in P or name == "roles":
        raise ValueError(f"format '{name}': available are {', '.join(k for k in P if k != 'roles')}")
    p = dict(P[name])
    for k in ("edge_ceiling", "scale_ceiling", "time", "ramp", "hold_end", "arc", "move_share"):
        if spec.get(k) is not None: p[k] = spec[k]
    return name, p


def _res(spec):
    r = spec.get("res") or "1280x720"
    W, H = (int(v) for v in str(r).lower().split("x")) if isinstance(r, str) else (int(r[0]), int(r[1]))
    return W, H


def shot(spec, scene=None, track=None, field=None):
    """Camera move from the spec knobs at the place of scene (a run folder with layers) → {cams, checks, bad, plan, skipped}."""
    spec = dict(spec or {})
    if spec.get("role"):                                               # the camera role gives defaults (explicit knobs win)
        role = (profiles().get("roles") or {}).get(spec["role"])
        if role is None: raise ValueError(f"role '{spec['role']}': available are {', '.join(profiles().get('roles') or {})}")
        for k, v in role.items():
            if k == "yaw":                                              # the role's angle is a preference for choosing the position, not a hard constraint
                spec.setdefault("prefer_yaw", v)
            else:
                spec.setdefault(k, v)
    fname, prof = profile(spec); field = field or Field(scene)
    W, H = _res(spec); aspect = W / H; fps = float(spec.get("fps", 24))
    lo, hi = prof.get("shot_s") or (2.0, 6.0)
    n = 1 if spec.get("panel") else max(2, int(round(float(spec.get("sec", (lo + hi) / 2)) * fps)))
    recipe = spec.get("recipe", "static")
    if recipe not in R.RECIPES: raise ValueError(f"recipe '{recipe}': available are {', '.join(R.RECIPES)}")
    subj = R.subject(spec, field, track)
    rig = spec.get("rig") or (prof.get("rig") or {}).get(subj["kind"], "steadicam")
    us = T.progress(n, prof.get("time", "ease"), prof.get("ramp", 0.12), prof.get("hold_end", 0.0), prof.get("move_share", 0.6), fps)
    notes = []
    if recipe == "follow":
        if track is None: raise ValueError("follow: the hero's track (track) is required")
        C, L, lens, A, meta = R.follow(track, field, spec, prof, rig, fps, aspect)
        C2, nn = G.project(C, rig, field); notes += nn
        L = L + (C2 - C); cams = [{"pos": c, "look": l, "lens": lens} for c, l in zip(C2, L)]
        aims = A; k_fit = 1.0
    else:
        build_r = {"static": R.static, "pan": R.pan, "push": R.push, "truck": R.truck, "arc": R.arc, "crane": R.crane,
                   "zoom": R.zoom, "pull": lambda *a, **kw: R.push(*a, sign=-1, **kw)}[recipe]
        last = {}
        def build(k):
            pose, meta = build_r(subj, field, spec, prof, rig, k=k); last["meta"] = meta
            ps = [pose(float(u)) for u in us]
            C = np.array([p["pos"] for p in ps]); C2, nn = G.project(C, rig, field); last["notes"] = nn
            out = []
            for p, c in zip(ps, C2):
                if "frame" in p:                                       # by frame: the aim goes to its own screen position after the rig is set
                    f = p["frame"]; look = F.look_for(c, F.aim_point(subj, f["aim"]), f["sx"], f["sy"], p["lens"], aspect)
                else:
                    look = p["look"] + (c - p["pos"])
                out.append({"pos": c, "look": look, "lens": p["lens"]})
            return out
        adjustable = (recipe in ("push", "pull", "truck", "zoom", "crane") or (recipe == "arc" and spec.get("bg_b") is None)
                      or (recipe == "pan" and spec.get("to") is None))   # with a goal at the end (background B, a second subject) the move is not cut
        if recipe not in SCALE_MOVES:                                   # the edge scale is the measure for push-in, zoom and crane; for an arc and a pass-by
            prof = dict(prof, scale_ceiling=None)                       # it is a parallax trace, the measure is the shift
        if adjustable and n > 1 and not spec.get("no_fit"):
            k_fit, cams, _ = T.fit(build, field, float(prof.get("edge_ceiling") or 99), float(prof.get("scale_ceiling") or 99), fps, aspect)
        else:
            k_fit, cams = 1.0, build(1.0)
        meta = last.get("meta", {}); notes += last.get("notes", [])
        fr = R._base(spec, subj); aims = R.vis_point(subj, fr["aim"])[None]
    if spec.get("shake_at"):
        Ls = G.shake([c["look"] for c in cams], [c["pos"] for c in cams], spec["shake_at"], fps)
        for c, l in zip(cams, Ls): c["look"] = l
    if rig == "handheld":                                              # the hand is a recording of a live camera (data/shake.json)
        Ps, Ls = G.handheld([c["look"] for c in cams], [c["pos"] for c in cams], fps, spec.get("hand", "the_closeup"),
                            float(spec.get("hand_amp", 1.0)), int(spec.get("seed", 0)))
        for c, p, l in zip(cams, Ps, Ls): c["pos"], c["look"] = p, l
    if recipe == "follow":                                              # the background flows behind the followed hero — normal (the eye is on the hero):
        prof = dict(prof, edge_ceiling=max(float(prof.get("edge_ceiling", 0)), BROWN_STROBE))   # the ceiling — Brown's strobing
    nums, bad = K.run(cams, aims, field, fps, aspect, prof, centre=subj["pos"] if recipe == "arc" else None, skip=subj.get("r", 0.4) + 1.5)
    if k_fit < 1.0:
        notes.append(f"move shortened to {k_fit:.2f} of its amplitude to fit the format ceiling {fname}")
    cs = float(prof.get("edge_ceiling") or 0)
    if recipe != "follow" and n > 1 and cs and nums.get("shift_p90", 0) > cs and k_fit >= 1.0:
        notes.append(f"longer shot: ≈ {n / fps * nums['shift_p90'] / cs:.1f} s (the move ends on a target — it is not cut)")
    if nums.get("still") and recipe not in ("static",):
        bad.append("the move changes nothing — use static")
    name = spec.get("name") or "move"
    recs = [F.camera(c["pos"], c["look"], c["lens"], (W, H), name=f"{name}_{i:03d}", frame=int(spec.get("start", 1)) + i, fps=fps) for i, c in enumerate(cams)]   # start — the scene's first frame of the shot (the moment of action)
    plan = {"recipe": recipe, "format": fname, "rig": rig, "frames": len(recs), "fps": fps, "fit": round(k_fit, 3),
            "subject": {"pos": [round(float(v), 2) for v in subj["pos"]], "h": round(subj["h"], 2), "kind": subj["kind"]}}
    plan.update({k: v for k, v in (meta or {}).items()})
    return {"cams": recs, "checks": nums, "bad": bad + notes, "plan": plan, "skipped": list(field.skipped)}


def panel(spec, scene=None, field=None):
    """A comic-panel frame — a move of zero length: the same choice of position (background, visibility, free space) and the same checks."""
    return shot(dict(spec, panel=True, recipe=spec.get("recipe", "static")), scene, field=field)
