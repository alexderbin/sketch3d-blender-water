"""Test stand of the Camera (Python, no Blender, ~1 s): the public entry api.py — shot movement (move, look_at) and framing (frame_*,
follow_*) on several inputs; the plan of the "camera" step — on relief (a synthetic slope) and without a place (ground height 0); pinned
records (pinned): its own name — as it is, part of the orbit, a shorter orbit and another view — "skipped", a record without a matrix — an
error; a build with bpy replaced by a stub: objects — the step's own and other writers' plan cameras, not take_NNN (the take builds those),
an existing object is untouched, new ones get the contract tags sk_id, sk_kind. The plan scenes live in a temporary folder.
The frame — a 36 mm sensor by width, as in api: tan of the half-angle = 18 / lens.
.venv/bin/python -m components.camera.bench.check   → lines "ok / FAIL", __RESULT__ {checks, failures, failed}; exit code is the number of failures"""
import json, math, os, sys, tempfile

import numpy as np

from components.camera import api as A
from components.camera import steps as S

FAIL, DONE = [], []
SIZE = (1280, 720)


def ok(name, cond, info=""):
    DONE.append(name); print(("ok   " if cond else "FAIL ") + name + (f" — {info}" if info else ""), flush=True)
    if not cond: FAIL.append(name)


def axes(M):
    """Camera matrix 4×4 → position, right, up, view direction (camera −Z)."""
    M = np.asarray(M, float); return M[:3, 3], M[:3, 0], M[:3, 1], -M[:3, 2]


def screen(M, p, lens, size=SIZE):
    """World point in the frame: (x, y) — fractions of the half-frame (|·| ≤ 1 — inside the frame), z — depth (> 0 — in front of the camera)."""
    pos, r, u, f = axes(M); d = np.asarray(p, float) - pos; z = float(d @ f); tx = 18.0 / lens
    return float(d @ r) / z / tx, float(d @ u) / z / (tx * size[1] / size[0]), z


def angle(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))))))


def moves():
    c = np.array([0.0, 0.0, 2.0]); p0 = np.array([30.0, 0.0, 12.0]); M0 = A.look_at(p0, c)
    pos, r, u, f = axes(M0)
    R = np.asarray(M0, float)[:3, :3]
    ok("look_at: the basis is orthonormal, right is horizontal, up is up, the view is on the target",
       np.allclose(R.T @ R, np.eye(3), atol=1e-9) and abs(r[2]) < 1e-9 and u[2] > 0 and angle(f, c - p0) < 1e-6)
    M, _ = A.move(M0, c, "orbit", 90); q = axes(M)[0]
    ok("orbit 90°: the same radius and height, rotated by 90°, looking at the centre",
       abs(np.linalg.norm((q - c)[:2]) - 30) < 1e-6 and abs(q[2] - p0[2]) < 1e-9 and abs(angle((q - c)[:2], (p0 - c)[:2]) - 90) < 1e-6
       and angle(axes(M)[3], c - q) < 1e-6, f"{np.round(q, 3)}")
    M, _ = A.move(M0, c, "orbit", 90, e=0.5); q = axes(M)[0]
    ok("orbit 90° at half the way — 45°", abs(angle((q - c)[:2], (p0 - c)[:2]) - 45) < 1e-6)
    M, _ = A.move(M0, c, "dolly", 0.5); q = axes(M)[0]
    ok("dolly 0.5: midway to the centre, looking at the centre", np.allclose(q, (p0 + c) / 2) and angle(axes(M)[3], c - q) < 1e-6)
    M, _ = A.move(M0, c, "dolly", -0.5); q = axes(M)[0]
    ok("dolly −0.5: pull back — the distance ×1.5", abs(np.linalg.norm(q - c) / np.linalg.norm(p0 - c) - 1.5) < 1e-9)
    M, _ = A.move(M0, c, "crane", 3); q = axes(M)[0]
    ok("crane 3 m: up by 3 m, the same x and y", np.allclose(q, p0 + [0, 0, 3]))
    M, _ = A.move(M0, c, "truck", 2); q, _, _, f2 = axes(M)
    ok("truck 2 m: sideways along the right vector, the view stays parallel", np.allclose(q, p0 + r * 2) and angle(f2, f) < 1e-6)
    M, _ = A.move(M0, c, "pan", 30); q, _, _, f2 = axes(M)
    ok("pan 30°: the same position, the view turned by 30° in the horizontal plane", np.allclose(q, p0) and abs(angle(f2[:2], f[:2]) - 30) < 1e-6)
    M, lens2 = A.move(M0, c, "zoom", 2, e=0.5, lens=35)
    ok("zoom ×2 at half the way: 35 → 52.5 mm, the same position", abs(lens2 - 52.5) < 1e-9 and np.allclose(axes(M)[0], p0), f"{lens2}")
    M, _ = A.move(M0, c, "orbit", 60, aim=False); q, _, _, f2 = axes(M)
    ok("orbit without aiming: the whole camera rotates — the centre stays in the middle of the frame", angle(f2, c - q) < 1e-6)
    M, _ = A.move(M0, c, "dolly", 0.3, aim=False)
    ok("dolly without aiming: the camera rotation is unchanged", np.allclose(np.asarray(M)[:3, :3], R))
    try: A.move(M0, c, "spin", 1); bad = False
    except ValueError: bad = True
    ok("unknown movement — ValueError", bad)


def frames():
    pts = [(-6, 0), (-3, 1), (0, -1), (3, 0.5), (6, 0)]; hz = 1.0; n = (0.0, -1.0)
    for lens in (28.0, 50.0):
        d = A.frame_group(pts, hz, n, lens=lens, size=SIZE); M = d["matrix_world"]; pos = np.array(d["location"]); cm = np.mean(pts, 0)
        sc = [screen(M, (x, y, hz + 0.9), lens) for x, y in pts]
        ok(f"frame_group {lens:.0f} mm: camera on the axis side, the whole group in frame",
           float((pos[:2] - cm) @ n) > 0 and all(abs(x) <= 1 and z > 0 for x, _, z in sc), f"x: {[round(x, 2) for x, _, _ in sc]}")
    xy, h = (4.0, 2.0), 1.8
    for yaw in (math.radians(-90), math.radians(90)):                   # facing the cameras or with the back to them — the same side (the 180° rule)
        d = A.frame_subject(xy, yaw, hz, h, n, plan="medium", size=SIZE); M = d["matrix_world"]; pos = np.array(d["location"])
        lo, hi = screen(M, (*xy, hz), d["lens_mm"]), screen(M, (*xy, hz + h), d["lens_mm"])
        ok(f"frame_subject medium (heading {math.degrees(yaw):.0f}°): from the axis side, the whole figure in frame",
           float((pos[:2] - np.array(xy)) @ n) > 0 and abs(lo[1]) <= 1 and abs(hi[1]) <= 1, f"y of feet {lo[1]:.2f}, of head {hi[1]:.2f}")
    d = A.frame_subject(xy, math.radians(-90), hz, h, n, plan="closeup", size=SIZE); M = d["matrix_world"]
    lo, hi = screen(M, (*xy, hz), d["lens_mm"]), screen(M, (*xy, hz + h * 0.95), d["lens_mm"])
    ok("frame_subject close: 50 mm, head in frame, feet out of frame", d["lens_mm"] == 50.0 and abs(hi[1]) <= 1 and lo[1] < -1,
       f"y of feet {lo[1]:.2f}, of head {hi[1]:.2f}")
    P1, P2, Q1, Q2 = (0.0, 0.0), (8.0, 0.0), (0.0, 0.0), (8.0, 1.0); n0 = np.array([-(Q2[1] - Q1[1]), Q2[0] - Q1[0]])
    sides = []
    for a, b in ((P1, P2), (P2, P1)):                                    # the order of the heroes is swapped — the same side, the line is not crossed
        d = A.frame_two(a, b, Q1, Q2, a, b, hz, 1.8, 1.8, side=1, size=SIZE); M = d["matrix_world"]; pos = np.array(d["location"])
        sc = [screen(M, (*p, hz + 1.0), d["lens_mm"]) for p in (a, b)]
        sides.append(float((pos[:2] - (np.array(a) + b) / 2) @ n0) > 0)
        ok("frame_two: both in frame", all(abs(x) <= 1 and z > 0 for x, _, z in sc), f"x: {[round(x, 2) for x, _, _ in sc]}")
    ok("frame_two: the side is set by the start of the piece — the same when the heroes are swapped", all(sides))
    sh, tg = np.array([0.0, 0.0]), np.array([20.0, 5.0]); d = A.frame_over(sh, 1.7, tg, 1.2, size=SIZE); M = d["matrix_world"]
    pos = np.array(d["location"]); f2 = (tg - sh) / np.linalg.norm(tg - sh); r2 = np.array([f2[1], -f2[0]]); x, y, z = screen(M, (*tg, 1.2), d["lens_mm"])
    ok("frame_over: behind and to the right of the shooter, 0.4 m above the head, the target in the centre of the frame",
       np.linalg.norm(pos[:2] - tg) > np.linalg.norm(sh - tg) and float((pos[:2] - sh) @ r2) > 0 and abs(pos[2] - 2.1) < 1e-9
       and abs(x) < 1e-9 and abs(y) < 1e-9 and z > 0)
    d = A.frame_point((5, -20, 6), (0, 0, 1), lens=40, size=SIZE); x, y, z = screen(d["matrix_world"], (0, 0, 1), 40)
    ok("frame_point: camera at the point, the target in the centre of the frame", np.allclose(axes(d["matrix_world"])[0], (5, -20, 6)) and abs(x) < 1e-9 and abs(y) < 1e-9)
    ok("follow_pos: heading 0° — forward 5 m, left 2.5 m, up 1.8 m", np.allclose(A.follow_pos((1.0, 2.0), 3.0, 0.0), (6.0, 4.5, 4.8)))
    pos, hero = np.array([0.0, -10.0, 2.0]), np.array([0.0, 0.0, 1.0])
    for lead, far in (((0.3, 0.0, 1.0), False), ((12.0, 0.0, 1.0), True)):
        L = A.follow_look(pos, hero, lead, lens=35.0); th = math.degrees(math.atan(6.0 / 35.0))
        good = abs(np.linalg.norm(L - pos) - np.linalg.norm(hero - pos)) < 1e-9 and (
            abs(angle(L - pos, hero - pos) - th) < 1e-6 if far else angle(L - pos, np.array(lead) - pos) < 1e-6)
        ok(f"follow_look: lead {'far — the gaze stays within a third of the frame' if far else 'near — looking at it'}", good)
    d = A.frame_follow((0.0, 0.0), 0.0, 0.0, (0.0, 0.0, 1.0), (40.0, 0.0, 1.0), size=SIZE)
    x, _, z = screen(d["matrix_world"], (0.0, 0.0, 1.0), d["lens_mm"])
    ok("frame_follow: the hero on a third of the frame (no farther), in front of the camera", abs(x) <= 1 / 3 + 1e-6 and z > 0, f"hero x {x:.3f}")


def relief(folder):
    """Synthetic place: a slope z = 10 + 0.1·x (near grid 0.5 m over 200 m, belt 10 m over 1 km), no platform."""
    xs = np.arange(-100, 100.5, 0.5); Xn = np.meshgrid(xs, xs)[0]; xf = np.arange(-500, 510, 10.0); Xf = np.meshgrid(xf, xf)[0]
    np.savez(os.path.join(folder, "relief.npz"), near=(10 + 0.1 * Xn).astype(np.float32), near_o=np.array([-100.0, -100.0, 0.5]),
             far=(10 + 0.1 * Xf).astype(np.float32), far_o=np.array([-500.0, -500.0, 10.0]), level=np.full(Xn.shape, -1e3, np.float32),
             pad=np.zeros(Xn.shape, np.uint8))
    json.dump({"far_z": 0.0, "center": [0.0, 0.0], "anchors": {"hill.top": [20.0, 0.0, 12.0]}}, open(os.path.join(folder, "location.json"), "w"))


def plans():
    keys = {"view": "wide", "azimuth": 200, "elevation": 12, "distance": 100, "lens": 35, "orbit": 4, "res": "640x360"}
    with tempfile.TemporaryDirectory() as T:                            # scene without a place: only the buildings' anchors and other writers' cameras
        json.dump({"castle.centre": [10.0, 20.0, 5.0]}, open(os.path.join(T, "anchors.json"), "w"))
        json.dump([{"name": "take_001"}, {"name": "cam_old"}], open(os.path.join(T, "cameras.json"), "w"))
        r = S.plan_camera({}, keys, {"scene": T}); cams = json.load(open(os.path.join(T, "cameras.json")))
        z0 = 8.0 + 100 * math.sin(math.radians(12))
        ok("plan without a place: does not fail, the target is the building centre at height 0 + 8 m, 'skipped' in the step line",
           r["target"] == [10.0, 20.0, 8.0] and bool(r.get("_skipped")), str(r["target"]))
        ok("plan without a place: camera by azimuth and elevation, orbit 4, other writers' cameras intact, its own old ones replaced",
           abs(cams[1]["matrix_world"][2][3] - z0) < 1e-6 and [c["name"] for c in cams] == ["take_001", "cam_wide"] + [f"orbit_{i:02d}" for i in range(4)],
           str([c["name"] for c in cams]))
    with tempfile.TemporaryDirectory() as T:                            # classic (Buildings writes classic.centre) — also the building centre
        json.dump({"classic.centre": [-3.0, 4.0, 2.0]}, open(os.path.join(T, "anchors.json"), "w"))
        r = S.plan_camera({}, dict(keys, orbit=0), {"scene": T})
        ok("plan without a place: the classic centre (classic.centre) — the target at height 0 + 8 m", r["target"] == [-3.0, 4.0, 8.0], str(r["target"]))
    with tempfile.TemporaryDirectory() as T:                            # scene with relief: the target height follows the ground, the camera is not underground
        relief(T)
        r = S.plan_camera({}, dict(keys, target="30,0", orbit=0), {"scene": T})
        ok("plan on relief: target 'x,y' — on the ground (z = 13), no 'skipped'", np.allclose(r["target"], [30.0, 0.0, 13.0]) and not r.get("_skipped"),
           str(np.round(r["target"], 3)))
        r = S.plan_camera({}, dict(keys, target="30,0", above=1.5, orbit=0), {"scene": T})
        ok("plan on relief: target 'x,y' with above 1.5 — 1.5 m above the ground (z = 14.5)", np.allclose(r["target"], [30.0, 0.0, 14.5]),
           str(np.round(r["target"], 3)))
        r = S.plan_camera({}, dict(keys, anchor="hill.top", elevation=-20, orbit=0), {"scene": T})
        p = json.load(open(os.path.join(T, "cameras.json")))[0]["matrix_world"]; x, y, z = (p[i][3] for i in range(3)); g = 10 + 0.1 * x
        ok("plan on relief: camera from below (elevation −20°) — not lower than 2 m above the ground", abs(r["target"][2] - 16.0) < 1e-9 and z >= g + 2.0 - 1e-6,
           f"z {z:.2f}, ground {g:.2f}")


def pins():
    """Pinned records (pinned — core.services.stage pin): the step plan on a run scene that holds a pinned cameras.json."""
    keys = {"view": "wide", "azimuth": 200, "elevation": 12, "distance": 100, "lens": 35, "orbit": 4, "res": "640x360"}
    mark = "out/dev/runs/x/a, 2026-10-31 12:00"
    with tempfile.TemporaryDirectory() as T, tempfile.TemporaryDirectory() as R:
        relief(T); relief(R); cp = os.path.join(T, "cameras.json")
        S.plan_camera({}, dict(keys, azimuth=230), {"scene": R}); ref = {c["name"]: c for c in json.load(open(os.path.join(R, "cameras.json")))}
        S.plan_camera({}, keys, {"scene": T}); cams = json.load(open(cp))
        for c in cams[:3]: c["pinned"] = mark                         # cam_wide, orbit_00, orbit_01 are pinned — part of the orbit
        cams[0]["matrix_world"][0][3] += 5.0                           # the approved camera is not the one the step would choose
        take = {"name": "take_001", "frame": 1, "pinned": mark}; cams.append(take); json.dump(cams, open(cp, "w"))
        r = S.plan_camera({}, dict(keys, azimuth=230), {"scene": T}); got = json.load(open(cp)); by = {c["name"]: c for c in got}
        ok("pinned, azimuth +30°: its own as they are (with pinned), orbit_02 and 03 — by the step's decision, take_001 intact, order unchanged",
           all(by[c["name"]] == c for c in cams[:3]) and by["take_001"] == take and [c["name"] for c in got] == [c["name"] for c in cams]
           and all(by[k]["matrix_world"] == ref[k]["matrix_world"] and "pinned" not in by[k] for k in ("orbit_02", "orbit_03")),
           str([c["name"] for c in got]))
        ok("pinned: pinned 3, the step's decision differs for three — in 'skipped'",
           r["pinned"] == 3 and r.get("pinned_drift") == 3 and "not recomputed" in (r.get("_skipped") or ""), str(r.get("_skipped")))
    with tempfile.TemporaryDirectory() as T:
        relief(T); cp = os.path.join(T, "cameras.json")
        S.plan_camera({}, keys, {"scene": T}); cams = json.load(open(cp))
        for c in cams: c["pinned"] = mark
        json.dump(cams, open(cp, "w"))
        r = S.plan_camera({}, keys, {"scene": T})
        ok("all pinned, the same knobs: 5 from the records, the same file, no 'skipped'",
           r["pinned"] == 5 and not r.get("_skipped") and json.load(open(cp)) == cams, str(r.get("_skipped")))
        r = S.plan_camera({}, dict(keys, orbit=2), {"scene": T}); names = [c["name"] for c in json.load(open(cp))]
        ok("orbit shorter than the pinned one (4 → 2): orbit_02, orbit_03 removed from the run scene, the names are in 'skipped'",
           names == ["cam_wide", "orbit_00", "orbit_01"] and r.get("pinned_out") == ["orbit_02", "orbit_03"]
           and "outside the orbit (orbit 2): orbit_02, orbit_03" in (r.get("_skipped") or ""), f"{names}; {r.get('_skipped')}")
        r = S.plan_camera({}, dict(keys, orbit=2, view="gate"), {"scene": T}); names = [c["name"] for c in json.load(open(cp))]
        ok("another view: pinned cam_wide removed, cam_gate decided, the name is in 'skipped'",
           names == ["cam_gate", "orbit_00", "orbit_01"] and r["pinned"] == 2 and "(view gate): cam_wide" in (r.get("_skipped") or ""),
           f"{names}; {r.get('_skipped')}")
    with tempfile.TemporaryDirectory() as T:
        json.dump([{"name": "cam_wide", "pinned": mark}], open(os.path.join(T, "cameras.json"), "w"))
        try: S.plan_camera({}, dict(keys, orbit=0), {"scene": T}); bad = False
        except ValueError: bad = True
        ok("a pinned record without matrix_world is an error, not a silent recomputation", bad)


def build():
    """Build without Blender (bpy and bl_scene replaced by stubs): which records become objects and what goes into the log line."""
    import types
    made, new, objs = [], {}, {"plan_2": types.SimpleNamespace(type="CAMERA")}  # the plan's camera object is already in the .blend
    fake = {"bpy": types.SimpleNamespace(data=types.SimpleNamespace(objects=objs)),
            "core.blender.bl_scene": types.SimpleNamespace(make_camera=lambda spec, name, sc: (made.append(name), new.setdefault(name, {}))[1])}
    saved = {k: sys.modules.get(k) for k in fake}; sys.modules.update(fake)
    try:
        with tempfile.TemporaryDirectory() as T:
            json.dump([{"name": n} for n in ("plan_1", "plan_2", "cam_wide", "orbit_00", "take_001", "take_002")], open(os.path.join(T, "cameras.json"), "w"))
            r = S.build_camera({"sc": None, "scene": T, "plan": {"pinned": 2, "_skipped": "pinned …"}})
    finally:
        for k, v in saved.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v
    ok("build: objects — the step's own and other writers' plan cameras; take_NNN — none (the take builds them); an existing object is untouched",
       made == ["plan_1", "cam_wide", "orbit_00"], str(made))
    ok("build: the plan's pinned and 'skipped' go into the return value (the log line)", r["pinned"] == 2 and r.get("_skipped") == "pinned …", str(r))
    ok("build: new objects carry the contract tags — sk_id (the record name), sk_kind camera (no no_sk_* warning)",
       made and all(new[n] == {"sk_id": n, "sk_kind": "camera"} for n in made), str(new))


def solver():
    """Wrappers of the framing solver (framing_fast) against the unwrapped one: the frame is within 0.01 mm, the refusal and the cloud projection are the same, several times faster."""
    import random, time
    import core.paths
    core.paths.component_ext("camera", "armature")
    from armature_core import framing as F
    from components.camera import framing_fast
    rng = random.Random(3); C = [[rng.uniform(-0.4, 0.4), rng.uniform(-0.3, 0.3), rng.uniform(0, 1.9)] for _ in range(40)]
    ks = [dict(height_frac=hf, end_x_frac=0.62, target_y_frac=0.5, azimuth_deg=az, elevation_deg=el, lens_mm=ln, sensor_mm=36.0, width=W, height=H,
               radius_bounds=(0.5, 400.0), passes=3) for hf, az, el, ln, W, H in ((0.7, 50, -18, 24, 864, 1536), (0.3, 35, 3, 35, 1536, 864),
                                                                                 (0.88, -35, 0, 50, 1536, 1536))]

    def run():
        t = time.perf_counter(); out = [F.solve_camera(C, C, **k) for k in ks]; sec = time.perf_counter() - t
        try: F.solve_camera(C, C, **dict(ks[0], end_x_frac=40.0)); err = None   # a position 40 frame widths away — unreachable
        except F.FramingError as e: err = str(e)
        return out, sec, err, F._extent(C, [0.1, 0.2, 1.0], 6.0, 30, 10, 35, 36.0, 1536, 864)
    a = run(); framing_fast.install(F); b = run()
    dr = max(abs(x["radius"] - y["radius"]) for x, y in zip(a[0], b[0])); dt = max(math.dist(x["target"], y["target"]) for x, y in zip(a[0], b[0]))
    ok("solver with wrappers: the same frame within 0.01 mm (radius, target), several times faster", dr < 1e-5 and dt < 1e-5 and b[1] < a[1] / 5,
       f"radius {dr * 1000:.4f} mm, target {dt * 1000:.4f} mm; {a[1]:.2f} → {b[1]:.3f} s")
    ok("solver with wrappers: the refusal (an unreachable frame) is the same FramingError; the cloud projection gives the same numbers",
       a[2] is not None and a[2] == b[2] and max(abs(x - y) for x, y in zip(a[3], b[3])) < 1e-12)


def move_by_site():
    """Place-aware move (components/camera/move): its own test stand bench/move_check.py, the shared "ok / FAIL" line."""
    from components.camera.bench import move_check
    move_check.run(ok)


def main():
    for part in (moves, frames, plans, pins, build, solver, move_by_site):
        try: part()
        except Exception as e: ok(f"{part.__name__}: no exceptions", False, f"{type(e).__name__}: {e}")
    print("__RESULT__" + json.dumps({"checks": len(DONE), "failures": len(FAIL), "failed": FAIL}, ensure_ascii=False))
    return len(FAIL)


if __name__ == "__main__":
    sys.exit(main())
