"""Entry of the Camera component for the step-by-step build: the "camera" step — a camera by target, and an orbit.
plan (project Python): the target (x,y, a place or building anchor, a building centre — the anchor layer core.layers), azimuth, elevation (°), distance,
lens, frame; orbit: orbit N cameras around the target along an arc of arc° (360 — a full circle) with a rise of rise° and a push-in close (a fraction of the distance)
→ records in the run scene's cameras.json (names cam_<view>, orbit_<i>). The place relief (relief.npz) is optional: if present — the target height
follows the ground and the camera stays at least 2 m above it; if not — the ground height is 0, and the step line says "skipped" (the ground was not checked).
Pinned records (the pinned field, core.services.stage pin): the step does not recompute a record of its own name — it leaves it as it is; pinned records
of its own names outside the step (a shorter orbit, another view) it removes from the run scene.
build (Blender): camera objects from cameras.json into the "cameras" collection — the step's own (pinned ones from the record) and other writers' plan cameras; take_NNN —
none (the take builds them); an object with the same name already present is left alone. New objects get the contract tags: sk_id — the record name,
sk_kind — camera."""
import json, math, os

OWN = ("cam_", "orbit_", "move_")                                     # the step's own records of the camera layer: the others are written by the take, Battle, Layout, the place
TAKE = ("take_",)                                                     # the take's records: their objects are built by Time, the camera build does not


def _spec(name, target, az, el, dist, lens, res, G=None):
    a, e = math.radians(az), math.radians(el)
    p = [target[0] + dist * math.cos(a) * math.cos(e), target[1] + dist * math.sin(a) * math.cos(e), target[2] + dist * math.sin(e)]
    if G is not None:
        try: p[2] = max(p[2], float(G.height(p[0], p[1])) + 2.0)   # not underground
        except Exception: pass
    z = [p[i] - target[i] for i in range(3)]; n = math.sqrt(sum(c * c for c in z)); z = [c / n for c in z]
    x = [-z[1], z[0], 0.0]; n = math.sqrt(x[0] ** 2 + x[1] ** 2) or 1.0; x = [x[0] / n, x[1] / n, 0.0]
    y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]]
    mw = [[x[0], y[0], z[0], p[0]], [x[1], y[1], z[1], p[1]], [x[2], y[2], z[2], p[2]], [0, 0, 0, 1]]
    W, H = (int(v) for v in str(res).lower().split("x"))
    return {"name": name, "lens_mm": float(lens), "sensor_width_mm": 36.0, "sensor_height_mm": 24.0, "sensor_fit": "HORIZONTAL",
            "shift_x": 0.0, "shift_y": 0.0, "resolution": [W, H], "matrix_world": mw}


def _target(keys, ctx, G):
    """Camera target (above — the height of the target point above the ground or the anchor, m): the knob target "x,y" (by default on the ground) →
    the anchor anchor (of the relief or of the buildings, the anchor layer core.layers) → a building centre
    (castle | town | temple | edifice | classic) → the "*.top" peak of the place → the middle of the place. Without relief (G is None) the ground height is 0."""
    from core import layers as LY
    L, A = (G.loc if G is not None else {}), LY.anchors(ctx["scene"])
    hz = (lambda x, y: float(G.height(x, y))) if G is not None else (lambda x, y: 0.0)
    if keys.get("target"):
        x, y = (float(v) for v in str(keys["target"]).split(",")[:2]); return [x, y, hz(x, y) + float(keys.get("above", 0.0))]
    if keys.get("anchor") in A:
        p = A[keys["anchor"]]; return [p[0], p[1], p[2] + float(keys.get("above", 4.0))]
    b = next((A[k] for k in ("castle.centre", "town.centre", "temple.centre", "edifice.centre", "classic.centre") if k in A), None)
    if b:
        return [b[0], b[1], hz(b[0], b[1]) + float(keys.get("above", 8.0))]
    a = next((k for k in A if k.endswith(".top")), None)
    if a:
        p = A[a]; return [p[0], p[1], p[2] + float(keys.get("above", 4.0))]
    c = L.get("center", [0.0, 0.0]); return [float(c[0]), float(c[1]), hz(c[0], c[1]) + 4.0]


def _drift(a, b):
    """A pinned record a against the step's decision b (like stage diff): (shift, m; gaze rotation, °; other lens, sensor, shift
    or frame); a match — within 5 mm and 0.05° — gives None."""
    pa, pb = ([c["matrix_world"][i][3] for i in range(3)] for c in (a, b))
    va, vb = ([c["matrix_world"][i][2] for i in range(3)] for c in (a, b))   # the camera's z axis: the view is −z
    cos = sum(x * y for x, y in zip(va, vb)) / ((math.hypot(*va) * math.hypot(*vb)) or 1.0)
    d = (math.dist(pa, pb), math.degrees(math.acos(max(-1.0, min(1.0, cos)))),
         any(a.get(k) != b.get(k) for k in ("lens_mm", "sensor_width_mm", "sensor_height_mm", "sensor_fit", "shift_x", "shift_y"))
         or list(a.get("resolution") or []) != list(b.get("resolution") or []))
    return d if d[0] > 0.005 or d[1] > 0.05 or d[2] else None


def _short(names):
    """Record names, briefly: "orbit_08…orbit_11 (4)"."""
    ns = sorted(names)
    return ", ".join(ns) if len(ns) <= 3 else f"{ns[0]}…{ns[-1]} ({len(ns)})"


def plan_camera(M, keys, ctx):
    SD = ctx["scene"]; G = None
    if os.path.exists(os.path.join(SD, "relief.npz")):                 # the place relief is the site section, optional (may_read): the camera
        from core.surface.ground import Ground                         # is placed even without a place (a scene without relief — the ground at 0)
        G = Ground(SD)
    t = _target(keys, ctx, G)
    az, el = float(keys.get("azimuth", 200)), float(keys.get("elevation", 12))
    dist, lens, res = float(keys.get("distance", 160)), float(keys.get("lens", 35)), keys.get("res", "960x540")
    view = str(keys.get("view") or "wide")                             # camera label: cam_<view>
    want = [_spec(f"cam_{view}", t, az, el, dist, lens, res, G)]
    n = int(keys.get("orbit", 0) or 0)                                  # orbit: N cameras; an arc of arc° (360 — a full circle), over the orbit — a rise
    arc, rise, close = float(keys.get("arc", 360.0)), float(keys.get("rise", 0.0)), float(keys.get("close", 0.0))   # of rise° and a push-in of close
    for i in range(n):                                                 # the old path: a circle by azimuth (the place-aware move is the move knob)
        f = i / n if arc >= 360.0 else i / max(n - 1, 1)                  # a circle — without repeating the first camera; an arc — from end to end
        want.append(_spec(f"orbit_{i:02d}", t, az + arc * f, el + rise * f, dist * (1.0 - close * f), lens, res, G))
    mvs, mres = keys.get("move"), None
    if mvs:                                                            # place-aware move (move/): recipe, frames A and B from the place field, the rig;
        from components.camera import move as MV                       # a list of moves — records move_<j>_NNN
        from components.camera.move.field import Field
        fld, many = Field(SD), isinstance(mvs, list); mres = {"plan": [], "checks": [], "bad": [], "cams": []}
        for j, mv in enumerate(mvs if many else [mvs]):
            mv = dict(mv); mv.setdefault("res", res); mv["name"] = f"move_{j}" if many else "move"
            if not mv.get("subject"):                                 # without a subject — the step's target: a point on the ground, the height is above
                g0 = float(G.height(t[0], t[1])) if G is not None else 0.0
                mv["subject"] = {"point": t[:2], "h": max(2.0, t[2] - g0)}
            r = MV.shot(mv, SD, field=fld)
            mres["plan"].append(r["plan"]); mres["checks"].append(r["checks"]); mres["cams"] += r["cams"]
            mres["bad"] += [(f"{mv['name']}: " if many else "") + b for b in r["bad"]]
        if not many: mres["plan"], mres["checks"] = mres["plan"][0], mres["checks"][0]
        want += [{k: v for k, v in c.items() if k != "_look"} for c in mres["cams"]]
    cp = os.path.join(SD, "cameras.json")
    have = json.load(open(cp)) if os.path.exists(cp) else []
    own = lambda c: str(c.get("name", "")).startswith(OWN)
    pins = {c["name"]: c for c in have if own(c) and c.get("pinned")}  # approved on sketch (stage pin): the decision is kept as data, not recomputed
    specs, drift = [], []
    for s in want:
        p = pins.pop(s["name"], None)
        if p is None: specs.append(s); continue
        miss = [k for k in ("matrix_world", "lens_mm", "resolution") if not p.get(k)]
        if miss: raise ValueError(f"pinned record {s['name']} lacks {', '.join(miss)}: pin it again (core.services.stage pin) or release it (unpin)")
        specs.append(p); d = _drift(p, s)                              # the record stays as it is, with pinned; the step's decision is only for comparison
        if d: drift.append(d)
    at = next((i for i, c in enumerate(have) if own(c)), len(have))   # the step's own records go back to their previous place: the frame numbers (cam<K>) stay the same across stages
    rest = [c for c in have if not own(c)]                             # the step's own unpinned records and the pinned ones outside the step are dropped from the run scene
    cams = rest[:at] + specs + rest[at:]
    json.dump(cams, open(cp, "w"), ensure_ascii=False, indent=1)
    npin = sum(1 for s in specs if s.get("pinned"))
    out = {"target": t, "cameras": len(specs), "total": len(cams), "names": [s["name"] for s in specs], "azimuth": az, "elevation": el,
           "distance": dist, "pinned": npin}
    skip = ["no place relief (relief.npz): ground height is 0, the camera was not checked for 'at least 2 m above the ground'"] if G is None else []
    if mres is not None:                                               # move: the plan and checks go into the step result, "what is wrong" into the line
        out["move"] = {"plan": mres["plan"], "checks": mres["checks"]}
        if mres["bad"]: skip.append("move: " + "; ".join(mres["bad"]))
    if pins:                                                           # not silently; the pin is intact in the source scene — restore it with a knob or unpin
        out["pinned_out"] = sorted(pins)
        o, m = sorted(k for k in pins if k.startswith("orbit_")), sorted(k for k in pins if k.startswith("move_"))
        v = sorted(k for k in pins if k.startswith("cam_"))
        skip.append("pinned " + "; ".join(([f"orbit_NN outside the orbit (orbit {n}): {_short(o)}"] if o else [])
                                                + ([f"move_NNN outside the move (frames {len(mres['cams']) if mres else 0}): {_short(m)}"] if m else [])
                                                + ([f"cam_ of another view (view {view}): {_short(v)}"] if v else []))
                    + " — removed from the run scene, pinned in the source")
    if drift:                                                          # the knobs, the scene or the code changed since the pin — the step's decision is different
        out["pinned_drift"] = len(drift)
        skip.append(f"pinned records not recomputed: the step's decision differs for {len(drift)} of {npin} (up to {max(d[0] for d in drift):.1f} m, "
                    f"{max(d[1] for d in drift):.1f}°{'; lens or framing' if any(d[2] for d in drift) else ''})")
    if skip: out["_skipped"] = "; ".join(skip)
    return out


def build_camera(ctx):
    import bpy
    from core.blender import bl_scene as B
    from core.contract import contract as C
    sc = ctx["sc"]; cams = json.load(open(os.path.join(ctx["scene"], "cameras.json")))
    made = []
    for spec in cams:                                                  # pinned ones also from the record: matrix_world, lens, sensor, shift, frame
        name = spec.get("name") or f"cam{len(made) + 1}"
        if name.startswith(TAKE): continue                             # take_NNN objects are built by the take (Time) from its own records
        if bpy.data.objects.get(name) is not None and bpy.data.objects[name].type == "CAMERA": continue   # the object already exists — leave it alone
        ob = B.make_camera(spec, name, sc); made.append(name)          # the step's own and other writers' plan cameras (Layout, Battle): detail and forest are driven by them
        ob[C.SK_ID], ob[C.SK_KIND] = name, "camera"                    # contract tags of the objects: id — the name of the cameras.json record
    plan = ctx.get("plan") or {}
    out = {"cameras": len(made), "names": made, "pinned": int(plan.get("pinned") or 0)}   # pinned goes into the log line's statistics
    if plan.get("_skipped"): out["_skipped"] = plan["_skipped"]       # "skipped" of the plan (no relief, pinned records outside the step) goes into the step line
    return out
