"""Check of film-camera paths of a large scene without rendering: on every frame of a shot the camera and the look-at point use the same keys
that tools/bl_crowd.shot_camera builds (shake included), and the rays go through one tree of static geometry (core/blender/bl_static_bvh.py:
terrain, crowds, heroes, vehicles, effects and the shield are skipped). The same checks as in bl_cam_path.py (the gap to buildings, the
target visible), but on every frame rather than on keys, plus "a wall filling the frame" — a small grid of rays across the frame (depth
without a render): a path can avoid hitting a wall and still run right alongside it.
blender -b --factory-startup <scene.blend> -P core/blender/run.py -- components/camera/bl_shot_check.py -- <shots.json> <out.json> [gap=2] [close=12] [frac=0.6]
    [grid=16x9] [frame=1280x536] [step=1]
shots.json — a list of shots ({name, frames, path, aim, lens | lenses, shake}, as battle.yaml.shots.json) or {"cases": [{case, shot, …}]}.
Shot verdict: collision (the camera is inside a body or closer than 0.5 m) · gap (closer than "gap" m, or the frame is blocked closer than "close" m
for more than "frac" of the frames, three in a row) · obstructed (more than 40 % of the frame is covered by geometry closer than a third of the distance
to the target, 8–80 m, three frames in a row: an old shot — a wall filling half the frame from 35 m; clean trailer paths — up to 6 %) ·
target blocked (> 20 % of the frames) · clean. The answer is __RESULT__: the verdicts and the worst frames.
For a case with an expectation (bench/battle/shots_known.json): a known error is caught in its own frame window, a clean path — clean.
A shot with a target box (components/camera/battle_cam.py: 'box' {f, c, h[, alt]}, 'fmt' 9:16 | 16:9) is also checked "in frame" by the format's rules
(as in the solver: 9:16 — the centre in the upper half, the box not in the lower third; 16:9 — zones of 80 and 90 %), "target visible" (rays to the
centre and the top of the box through the static tree) and screen jerk; the field of view is as in Blender (the sensor along the long side of the frame)."""
import json, os, sys, time
import bpy
from mathutils import Vector
from core.blender import bl_kit as C
from core.blender import bl_static_bvh as SB

t0 = time.time(); a = sys.argv[sys.argv.index("--") + 1:]; SRC, OUT = a[0], a[1]; opt = dict(x.split("=", 1) for x in a[2:])
GAP, NEAR, SHARE = float(opt.get("gap", 2.0)), float(opt.get("close", 12.0)), float(opt.get("frac", 0.6))
FILL_K, FILL_LO, FILL_HI, FILL_SHARE = 1 / 3, 8.0, 80.0, 0.4    # "obstructed": closer than a third of the distance to the target (8–80 m) > 40 % of the frame
GW, GH = map(int, opt.get("grid", "16x9").split("x")); FW, FH = map(int, opt.get("frame", "1280x536").split("x")); STEP = int(opt.get("step", 1))
sc = bpy.context.scene; src = json.load(open(SRC)); cases = src["cases"] if isinstance(src, dict) else [{"case": s["name"], "shot": s} for s in src]
ST = SB.build(sc); print(f"BVH {ST.info}", flush=True)
Z = Vector((0, 0, 1)); DIRS = [Vector(v).normalized() for v in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                                                                (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)]]


def frames_of(sh):
    """Shot camera per frame: the same keys and shake as in the scene (C.shot_camera in a temporary collection) — the curve values, without frame_set."""
    from bpy_extras import anim_utils
    tmp = bpy.data.collections.new("_shot_check"); sc.collection.children.link(tmp)
    cam = C.shot_camera(sc, tmp, sh, phase=sh.get("_phase", 0.0)); aim = next(o for o in tmp.objects if o.name.startswith(sh["name"] + "_aim"))
    def curves(idb):
        ad = idb.animation_data
        if not ad or not ad.action: return {}
        cb = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
        return {(fc.data_path, fc.array_index): fc for fc in cb.fcurves}
    cc, ac, lc = curves(cam), curves(aim), curves(cam.data)
    a_, b_ = sh["frames"]; out = []
    for f in range(a_, b_ + 1, STEP):
        P = Vector([cc[("location", i)].evaluate(f) for i in range(3)]); A = Vector([ac[("location", i)].evaluate(f) for i in range(3)])
        out.append((f, P, A, lc[("lens", 0)].evaluate(f) if ("lens", 0) in lc else cam.data.lens))
    sw = cam.data.sensor_width
    for m in [m for m in sc.timeline_markers if m.name == sh["name"]]: sc.timeline_markers.remove(m)
    for o in list(tmp.objects): bpy.data.objects.remove(o)
    bpy.data.collections.remove(tmp)
    return out, sw


FMTS = {"9:16": (1080, 1920), "16:9": (1920, 1080)}
SIGNS = [Vector((sx, sy, sz)) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]


def box_at(bx, f):
    """Target box at frame f: the nearest sample (the box step is 2 frames); the second one (at the turn) if enabled."""
    import bisect
    k = min(max(bisect.bisect_left(bx["f"], f), 0), len(bx["f"]) - 1)
    if k > 0 and abs(bx["f"][k - 1] - f) < abs(bx["f"][k] - f): k -= 1
    out = [(Vector(bx["c"][k]), Vector(bx["h"][k]))]
    if bx.get("alt") and bx["alt"]["on"][k]: out.append((Vector(bx["alt"]["c"][k]), Vector(bx["alt"]["h"][k])))
    return out


def in_frame(P, fwd, r_, u_, hw, hh, C, H, portrait):
    def pr(X):
        d = X - P; z = d.dot(fwd); zz = max(z, 1e-3); return d.dot(r_) / (zz * hw), d.dot(u_) / (zz * hh), z
    cx, cy, cz = pr(C); pts = [pr(C + Vector((H.x * s.x, H.y * s.y, H.z * s.z))) for s in SIGNS]
    okc = cz > 0 and abs(cx) <= 0.8 and ((-0.1 <= cy <= 0.8) if portrait else abs(cy) <= 0.8)
    ylo = -0.33 if portrait else -0.9; front = [q for q in pts if q[2] > 0]
    share = sum(1 for x, y, z in front if abs(x) <= 0.9 and ylo <= y <= 0.9) / 8
    big = len(front) < 8 or (front and ((max(q[0] for q in front) - min(q[0] for q in front)) / 2 >= 0.9
                                        or (max(q[1] for q in front) - min(q[1] for q in front)) / 2 >= (0.6 if portrait else 0.9)))
    return bool(okc and (share >= 0.75 or big)), (cx, cy)


def check(sh):
    fr, sw = frames_of(sh); rows = []; FWs, FHs = FMTS.get(sh.get("fmt"), (FW, FH)); bx = sh.get("box"); portrait = FHs > FWs
    for f, P, A, lens in fr:
        fwd = (A - P).normalized(); r_ = fwd.cross(Z).normalized(); u_ = r_.cross(fwd)
        hw = sw / lens / 2; hh = hw * FHs / FWs                             # the sensor along the long side (Blender, sensor_fit AUTO)
        if portrait: hh = sw / lens / 2; hw = hh * FWs / FHs
        L_ = (A - P).length
        DF = min(max(FILL_K * L_, FILL_LO), FILL_HI)
        near = fill = 0; dmin = 1e9; inside = False
        gw, gh = (GH, GW) if portrait else (GW, GH)           # the ray grid follows the frame sides
        for j in range(gh):
            for i in range(gw):
                d = (fwd + r_ * (hw * (2 * (i + 0.5) / gw - 1)) + u_ * (hh * (1 - 2 * (j + 0.5) / gh))).normalized()
                hit, dist, nrm, _ = ST.ray(P, d, max(NEAR, DF))
                if hit:
                    near += dist <= NEAR; fill += dist <= DF
                    if dist <= NEAR:
                        dmin = min(dmin, dist)
                        if nrm.dot(d) > 0.05: inside = True              # from inside a body: the first face points away from the camera
        for d in DIRS:
            hit, dist, nrm, _ = ST.ray(P, d, GAP)
            if hit:
                dmin = min(dmin, dist)
                if nrm.dot(d) > 0.05: inside = True
        hit, dist, _, idx = ST.ray(P, fwd, L_)
        occl = bool(hit and dist < 0.85 * L_)
        row = {"f": f, "near": round(near / (gw * gh), 3), "fill": round(fill / (gw * gh), 3), "df": round(DF, 1),
               "dmin": round(dmin, 2) if dmin < 1e8 else None, "inside": inside, "occl": occl, "by": ST.who(idx) if occl else None, "P": [round(x, 1) for x in P]}
        if bx:                                                         # target box: in the frame (either of the two at the turn) and visible
            bs = box_at(bx, f); res_ = [in_frame(P, fwd, r_, u_, hw, hh, C, H, portrait) for C, H in bs]; C, H = bs[0]
            seen = False
            for X in (C, C + Vector((0, 0, H.z))):
                d = X - P; Ld = d.length; rt = H.length + 1.0; h_, dd, _, _ = ST.ray(P, d.normalized(), max(Ld - rt, 0.1))
                seen = seen or not h_
            row.update(inf=any(o for o, _ in res_), cen=res_[0][1], seen=seen, C=[round(x, 1) for x in C])
        rows.append(row)
    coll = [r for r in rows if r["inside"] or (r["dmin"] is not None and r["dmin"] < 0.5)]
    def runs(key, share):                                              # frames in runs of ≥ 3 in a row above the fraction
        out, cur = [], []
        for r in rows + [{"f": None, key: 0}]:
            if r[key] > share: cur.append(r["f"])
            else:
                if len(cur) >= 3: out += cur
                cur = []
        return out
    near_f, fill_f = runs("near", SHARE), runs("fill", FILL_SHARE); worst = max(rows, key=lambda r: r["fill"])
    gap = [r for r in rows if r["dmin"] is not None and r["dmin"] < GAP]
    occ = sum(r["occl"] for r in rows) / max(len(rows), 1)
    if bx: occ = sum(not r["seen"] for r in rows) / max(len(rows), 1)  # with a box — the target itself is blocked (the solver's gaze misses the centre)
    verdict = "collision" if coll else "gap" if gap or near_f else "obstructed" if fill_f else "target blocked" if occ > 0.2 else "clean"
    bad = sorted({r["f"] for r in coll} | {r["f"] for r in gap} | set(near_f) | set(fill_f))
    tg = {}
    if bx:                                                             # screen jerk — ignoring the jump when the main target changes
        cen = [r["cen"] for r in rows]; Cs = [Vector(r["C"]) for r in rows]; j2 = 0.0
        for i in range(1, len(rows) - 1):
            if (Cs[i + 1] - Cs[i - 1]).length > 10: continue
            j2 = max(j2, max(abs(cen[i + 1][k] - 2 * cen[i][k] + cen[i - 1][k]) for k in (0, 1)) / STEP ** 2)
        tg = {"within_frame": round(sum(r["inf"] for r in rows) / len(rows), 3), "target_visible": round(sum(r["seen"] for r in rows) / len(rows), 3),
              "jerk_screen": round(j2, 4), "fmt": sh.get("fmt"), "out_of_frame": [r["f"] for r in rows if not r["inf"]][:12]}
    return {"verdict": verdict, "frames": len(rows), **tg, "collide_frames": [r["f"] for r in coll][:12], "gap_frames": [r["f"] for r in gap][:12],
            "fill_frames": fill_f[:12], "bad_frames": bad, "fill_max": worst["fill"], "fill_max_frame": worst["f"], "occluded": round(occ, 3),
            "dmin": min((r["dmin"] for r in rows if r["dmin"] is not None), default=None), "example": (coll or gap or [worst])[0], "rows": rows}


res = []
for c in cases:
    r = check(dict(c["shot"])); r["case"] = c["case"]
    if "expect" in c:                                                  # a known error — caught in its own frame window; a clean path — clean
        r["expect"] = c["expect"]; w = c.get("window")
        r["ok"] = r["verdict"] == "clean" if c["expect"] == "clean" else bool([f for f in r["bad_frames"] if w is None or w[0] <= f <= w[1]])
    res.append(r); print(f"SHOT {c['case']}: {r['verdict']}" + (f" (expected {c['expect']})" if "expect" in c else ""), flush=True)
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=0)
summary = [{k: r.get(k) for k in ("case", "verdict", "expect", "ok", "collide_frames", "gap_frames", "fill_frames", "fill_max", "fill_max_frame",
                                  "occluded", "dmin", "within_frame", "target_visible", "jerk_screen", "fmt") if k in r} for r in res]
print("__RESULT__" + json.dumps({"shots": summary, "bvh": ST.info, "s": round(time.time() - t0, 1)}, ensure_ascii=False))
