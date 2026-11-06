"""Test stand of the place-aware move (components/camera/move; Python, no Blender, ~5 s): the "frame ↔ camera" conversion both ways, time
curves, the place field on a synthetic castle (a slope, a hill, a ring of 10 m walls, four 18 m towers, a 20 m keep, a camp anchor), an arc
orbit against the old circle at a wall, rigs (rail, crane, drone), following the hero along a wall, fitting under the speed ceiling, the
move knob of the camera step (move_NNN records, pinned ones kept as they are). Called from bench/check.py (the shared "ok / FAIL" line)."""
import json, math, os, tempfile

import numpy as np

from components.camera import move as MV
from components.camera import steps as S
from components.camera.move import frame as F
from components.camera.move import rig as G
from components.camera.move import timing as T
from components.camera.move.field import Field


def ground(x, y):
    return 10 + 0.08 * x + 6 * math.exp(-(x * x + y * y) / (2 * 35 ** 2))


def castle(T_):
    """Synthetic castle on a slope: layers relief.npz, location.json (camp anchor), surfaces.json (walls, towers), occupancy.json
    (keep), anchors.json (castle.keep, castle.gate)."""
    xs = np.arange(-150, 150.5, 0.5); X, Y = np.meshgrid(xs, xs)
    Z = 10 + 0.08 * X + 6 * np.exp(-(X ** 2 + Y ** 2) / (2 * 35 ** 2))
    xf = np.arange(-1000, 1010, 10.0); Xf, _ = np.meshgrid(xf, xf)
    np.savez(os.path.join(T_, "relief.npz"), near=Z.astype(np.float32), near_o=np.array([-150.0, -150.0, 0.5]),
             far=(10 + 0.08 * Xf).astype(np.float32), far_o=np.array([-1000.0, -1000.0, 10.0]), level=np.full(Z.shape, -1e3, np.float32),
             pad=np.zeros(Z.shape, np.uint8))
    json.dump({"far_z": 0.0, "center": [0, 0], "anchors": {"camp": [120.0, -60.0, 20.0]}}, open(os.path.join(T_, "location.json"), "w"))
    ring = [[25 * math.cos(a), 25 * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 40, endpoint=False)]
    items = [{"tag": "w0_out", "part": "outer", "strip": {"pts": ring, "top": [ground(*p) + 10 for p in ring], "base": [ground(*p) for p in ring],
                                                          "t": 2.6, "closed": True}}]
    for k, a in enumerate(np.linspace(0.25 * math.pi, 2.25 * math.pi, 4, endpoint=False)):
        c = (25 * math.cos(a), 25 * math.sin(a)); pts = [[c[0] + 4 * math.cos(b), c[1] + 4 * math.sin(b)] for b in np.linspace(0, 2 * math.pi, 12, endpoint=False)]
        items.append({"tag": f"t{k}", "part": "towers", "strip": {"pts": pts, "top": [ground(*c) + 18] * 12, "base": [ground(*c)] * 12, "t": 1.0, "closed": True}})
    json.dump({"items": items}, open(os.path.join(T_, "surfaces.json"), "w"))
    json.dump({"castle.keep": [0.0, 0.0, ground(0, 0) + 24], "castle.gate": [25.0, 0.0, ground(25, 0)]}, open(os.path.join(T_, "anchors.json"), "w"))
    json.dump({"items": [{"kind": "building", "poly": [[-6, -6], [6, -6], [6, 6], [-6, 6]], "base": ground(0, 0), "eave": ground(0, 0) + 20}]},
              open(os.path.join(T_, "occupancy.json"), "w"))


def run(ok):
    # ── frame ↔ camera ─────────────────────────────────────────────────────────────────────────
    worst = 0.0
    for subj in ({"pos": [3.0, -2.0, 1.0], "h": 1.8, "yaw": 30.0}, {"pos": [0.0, 0.0, 16.0], "h": 24.0, "yaw": -120.0}):
        for fr in ({"size": 0.6, "yaw": 40, "pitch": 8, "sx": 0.5, "sy": 0.55}, {"size": 1.4, "yaw": -150, "pitch": -12, "sx": 0.33, "sy": 0.4},
                   {"size": 0.25, "yaw": 95, "pitch": 55, "sx": 0.7, "sy": 0.62, "lens": 85}):
            c = F.place(subj, fr); f = F.full(fr, subj["h"]); m = F.measure(c["pos"], c["look"], c["lens"], subj, f["aim"])
            worst = max(worst, abs(m["size"] / f["size"] - 1), abs((m["yaw"] - f["yaw"] + 180) % 360 - 180) / 100,
                        abs(m["pitch"] - f["pitch"]) / 100, abs(m["sx"] - f["sx"]), abs(m["sy"] - f["sy"]))
    ok("frame ↔ camera: place → measure returns the size, yaw, pitch and screen position (to 1e-6)", worst < 1e-6, f"{worst:.2e}")
    # ── time ───────────────────────────────────────────────────────────────────────────────────
    good = True
    for mode in T.MODES:
        u = T.progress(97, mode, 0.15, 0.0)
        good &= bool(u[0] == 0 and abs(u[-1] - 1) < 1e-12 and np.all(np.diff(u) >= -1e-12))
    u = T.progress(97, "ease", 0.15); v = np.diff(u) * 96
    ok("time: the curves are monotonic, 0 → 1; ease — a plateau ×1/(1−ramp) (1.18), the edges start from zero",
       good and abs(v[45:50].mean() - 1 / 0.85) < 0.02 and v[0] < 0.05 and v[-1] < 0.05, f"plateau {v[45:50].mean():.3f}, edges {v[0]:.3f} {v[-1]:.3f}")
    u = T.progress(96, "ease", 0.12, hold_end=0.5, fps=24); us = T.progress(100, "settle", move_share=0.6)
    ok("time: a 0.5 s hold — the arrival frame and 12 frames stand still; settle — the path is covered by 60 % of the plan",
       bool(np.all(u[-13:] == 1.0) and u[-14] < 1.0 and us[60] == 1.0 and us[58] < 1.0))
    # ── cut ────────────────────────────────────────────────────────────────────────────────────
    from components.camera.move import checks as K
    subj = {"pos": [0.0, 0.0, 0.0], "h": 1.8, "yaw": 0.0}
    ca = F.place(subj, {"size": 0.8, "yaw": 40}); cb = F.place(subj, {"size": 0.85, "yaw": 55}); cc = F.place(subj, {"size": 1.6, "yaw": 80})
    cd = F.place(subj, {"size": 0.8, "yaw": -60})
    _, b1 = K.cut(ca, cb, subj); _, b2 = K.cut(ca, cc, subj); _, b3 = K.cut(ca, cd, subj, axis=0.0)
    ok("cut: 15° and ×1.06 — a jump; 40° and ×2 — acceptable; across the axis at 0° — 'the axis is crossed by the cut'",
       any("jump" in x for x in b1) and not any("jump" in x for x in b2) and any("scene axis" in x for x in b3), f"{b1[:1]}; {b2}; {b3[:1]}")
    o, b = K.sequence([{"sec": 3, "recipe": "arc"}, {"sec": 2.5, "recipe": "push"}, {"sec": 4, "recipe": "arc"}, {"sec": 30, "recipe": "pan"}],
                      MV.profile({"format": "fiction"})[1])
    ok("episode plan against the profile: no static shots where the format wants 40 % and a 30 s shot — both come out 'wrong'",
       len(b) == 2 and o["static_share"] == 0.0, f"{o}; {b}")
    with tempfile.TemporaryDirectory() as T_:
        castle(T_); fld = Field(T_)
        # ── place field ────────────────────────────────────────────────────────────────────────────
        wz = ground(25 * math.cos(0.1), 25 * math.sin(0.1)) + 10
        ok("field: wall — masonry top and mark 1, tower — 2, keep — 3, meadow — ground",
           abs(float(fld.top(25 * math.cos(0.1), 25 * math.sin(0.1))) - wz) < 0.6 and int(fld.kind_at(25 * math.cos(0.1), 25 * math.sin(0.1))) == 1
           and int(fld.kind_at(25 * math.cos(math.pi / 4), 25 * math.sin(math.pi / 4))) == 2 and int(fld.kind_at(0, 0)) == 3
           and int(fld.kind_at(60, -40)) == 0, f"wall top {float(fld.top(25 * math.cos(0.1), 25 * math.sin(0.1))):.2f} vs {wz:.2f}")
        a, b = [60.0, 3.0, ground(60, 3) + 1.7], [15.0, 3.0, ground(15, 3) + 1.0]
        hi = [60.0, 3.0, ground(60, 3) + 80.0]
        ok("field: the court behind the wall is not visible from the meadow at eye level, visible from above; behind the hero at a wall — a wall",
           not bool(fld.sees(a, b)[0]) and bool(fld.sees(hi, b)[0]) and fld.behind([60.0, 3.0, ground(60, 3) + 1.7], [32.0, 3.0, ground(32, 3) + 1.2])[0][0] == "wall")
        # ── orbit: an arc by the place versus a circle ─────────────────────────────────────────────
        r = MV.shot({"recipe": "arc", "subject": {"anchor": "castle.keep"}, "size": 0.45, "pitch": 12, "format": "documentary", "sec": 8, "rig": "drone"}, T_, field=fld)
        c, p = r["checks"], r["plan"]
        ok("keep orbit: an arc of 15–180° (not a circle), the camera is free, the keep is visible, there is a hold at the end",
           15.0 <= c["arc_deg"] <= 180.0 and c["clear_min"] > 0.25 and c["visible"] >= 0.95 and c["hold_end_s"] >= 0.4,
           f"arc {c['arc_deg']}°, clearance {c['clear_min']} m, visible {c['visible']}, edge p90 {c['shift_p90']} %/s, fit {p['fit']}")
        r = MV.shot({"recipe": "arc", "subject": {"anchor": "castle.keep"}, "size": 0.45, "pitch": 12, "format": "documentary", "sec": 12, "rig": "drone",
                     "bg_b": "camp"}, T_, field=fld)
        ok("orbit with background B 'camp': at the end the camp anchor is behind the keep, the arc is no longer than 180°",
           bool(r["plan"].get("bg")) and "camp" in (r["plan"]["bg"][1][2], *r["plan"]["bg"][1][3]) and abs(r["plan"]["sweep"]) <= 180,
           json.dumps(r["plan"].get("bg"), ensure_ascii=False) + f"; arc {r['plan'].get('sweep')}°")
        hero = [29.5, -6.0]; keys = {"view": "wide", "azimuth": 0, "elevation": 8, "distance": 4.0, "lens": 24, "orbit": 24, "res": "1280x720",
                                     "target": f"{hero[0]},{hero[1]}", "above": 1.1}
        S.plan_camera({}, keys, {"scene": T_}); old = [cc for cc in json.load(open(os.path.join(T_, "cameras.json"))) if cc["name"].startswith("orbit_")]
        Po = np.array([[cc["matrix_world"][i][3] for i in range(3)] for cc in old]); inside_old = int((fld.clear(Po, r=0.3) < 0).sum())
        r = MV.shot({"recipe": "arc", "subject": {"point": hero, "h": 1.8}, "size": 0.6, "pitch": 6, "format": "fiction", "sec": 5, "rig": "steadicam"}, T_, field=fld)
        Pn = np.array([cc["matrix_world"][i][3] for cc in r["cams"] for i in range(3)]).reshape(-1, 3)
        hn = Pn[:, 2] - fld.ground(Pn[:, 0], Pn[:, 1]); inside_new = int((fld.clear(Pn, r=0.3) < 0).sum())
        ok("hero by a wall: the old circle (4 m) goes into the wall, the place-aware arc does not; the operator stays 0.6–2.0 m above the ground",
           inside_old > 0 and inside_new == 0 and hn.min() >= 0.59 and hn.max() <= 2.01 and r["checks"]["visible"] >= 0.95,
           f"circle in the wall {inside_old} of {len(Po)}, arc {inside_new} of {len(Pn)}, height {hn.min():.2f}–{hn.max():.2f} m, arc {r['checks']['arc_deg']}°")
        # ── rigs ───────────────────────────────────────────────────────────────────────────────────
        line = np.stack([np.linspace(-40, 40, 30), np.full(30, -30.0), np.full(30, 20.0)], 1)
        _, notes = G.project(line, "dolly", fld)
        P, _ = G.project(np.array([[60.0, 0.0, ground(60, 0) + 6], [64.0, 0.0, ground(64, 0) + 4.5], [68.0, 0.0, ground(68, 0) + 2.0]]), "crane", fld)
        rad = np.linalg.norm(P - P.mean(0), axis=1)
        D, _ = G.project(np.stack([np.linspace(-60, 60, 40), np.zeros(40), np.full(40, 20.0)], 1), "drone", fld)
        env = fld.envelope(D, r=5.0, window=1)
        ok("rigs: a rail over the hill 'will not fit'; the crane — an arc without a jerk; a drone above the castle — at least 3 m above the tops",
           any("rail will not fit" in s for s in notes) and np.all(np.diff(P[:, 2]) < 0) and float((D[:, 2] - env).min()) >= 2.99,
           f"{notes[:1]}; drone above the top ≥ {float((D[:, 2] - env).min()):.2f} m")
        # ── following ──────────────────────────────────────────────────────────────────────────────
        n = 120; xs = np.linspace(-20, -12.8, n); ys = np.full(n, -27.5)                # walks along the wall on the outside, 1.5 m/s
        track = {"pos": np.stack([xs, ys, [ground(x, y) for x, y in zip(xs, ys)]], 1), "yaw": np.zeros(n), "h": 1.8}
        r = MV.shot({"recipe": "follow", "size": 0.7, "format": "fiction", "rig": "steadicam"}, T_, track=track, field=fld)
        L = [np.asarray(cc["_look"]) for cc in r["cams"]]; Pc = [np.asarray(cc["location"]) for cc in r["cams"]]
        sx = [float(F.project(p, l, r["cams"][0]["lens_mm"], 16 / 9, a)[0]) for p, l, a in zip(Pc, L, F.aim_point({"pos": track["pos"], "h": 1.8}, 0.6))]
        ok("following along a wall: the side is the free one (not the wall), the hero is visible and stays near the middle (± 0.2 of the frame), no jerks",
           r["plan"]["free_share"] >= 0.95 and r["checks"]["visible"] >= 0.95 and max(abs(x - 0.5) for x in sx) <= 0.2 and not any("jerk" in b for b in r["bad"]),
           f"side {r['plan']['side']}°, free {r['plan']['free_share']}, screen position {min(sx):.2f}–{max(sx):.2f}")
        r = MV.shot({"recipe": "push", "subject": {"point": [0.0, -9.0], "h": 1.8}, "size": 0.35, "amount": 0.6, "format": "trailer", "sec": 3}, T_, field=fld)
        ok("push-in at a building in a courtyard: a valid position is chosen along the whole move — not through walls, no jerk, the hero is visible",
           r["checks"]["clear_min"] >= 0.25 and r["checks"]["visible"] >= 0.95 and not any("jerk" in b or "through" in b for b in r["bad"]),
           f"clearance {r['checks']['clear_min']} m, step ≤ {r['checks']['step_max_m']} m, {r['bad'][:1]}")
        a = MV.shot({"recipe": "static", "subject": {"point": [60, 10], "h": 1.8}, "size": 0.85, "format": "fiction", "sec": 4, "rig": "handheld",
                     "yaw": 40}, T_, field=fld)
        b = MV.shot({"recipe": "static", "subject": {"point": [60, 10], "h": 1.8}, "size": 0.85, "format": "fiction", "sec": 4, "rig": "handheld",
                     "yaw": 40, "hand": "handycam_run"}, T_, field=fld)
        ta, tb = a["checks"]["turn_max_deg"], b["checks"]["turn_max_deg"]
        ok("hand from a recording of a live camera: the 'quiet hand' hold shakes slightly, 'run' — several times stronger, the hero is visible",
           0.005 < ta < 0.5 and tb > 3 * ta and a["checks"]["visible"] >= 0.95, f"turn per frame up to {ta}° versus {tb}°")
        n = 24 * 20; xs = np.linspace(-58, -30, n); ys = np.full(n, -27.5)                # 20 s, 1.4 m/s; houses at 1.7 m: on the right at the start, on the left at the end
        track = {"pos": np.stack([xs, ys, [ground(x, y) for x, y in zip(xs, ys)]], 1), "yaw": np.zeros(n), "h": 1.8}
        with tempfile.TemporaryDirectory() as T2:
            castle(T2); occ = json.load(open(os.path.join(T2, "occupancy.json")))
            occ["items"].append({"kind": "house", "poly": [[-62, -36], [-45, -36], [-45, -29.2], [-62, -29.2]], "base": ground(-55, -33), "h": 6.0})
            occ["items"].append({"kind": "house", "poly": [[-42, -25.8], [-26, -25.8], [-26, -19], [-42, -19]], "base": ground(-34, -22), "h": 6.0})
            json.dump(occ, open(os.path.join(T2, "occupancy.json"), "w")); fld2 = Field(T2)
            r0 = MV.shot({"recipe": "follow", "size": 0.7, "format": "fiction", "rig": "steadicam"}, T2, track=track, field=fld2)
            r = MV.shot({"recipe": "follow", "size": 0.7, "format": "fiction", "rig": "steadicam", "sides": [90, -90, 30, -30]}, T2, track=track, field=fld2)
        sd = r["plan"]["sides"]
        ok("long following (20 s) between houses: when free — from the front (a neutral angle); sideways only — the side changes smoothly, the hero is visible",
           r0["plan"]["side"] in (0.0, 180.0) and r0["plan"]["free_share"] >= 0.95 and len(sd) >= 2 and r["plan"]["free_share"] >= 0.9
           and r["checks"]["visible"] >= 0.95 and not any("jerk" in b for b in r["bad"]),
           f"all sides — {r0['plan']['sides']}; sideways only — {sd}, free {r['plan']['free_share']}, visible {r['checks']['visible']}")
        # ── fitting and the step knob ──────────────────────────────────────────────────────────────
        r = MV.shot({"recipe": "push", "subject": {"point": [60, 10], "h": 1.8}, "size": 0.6, "amount": 2.5, "format": "lore", "sec": 3}, T_, field=fld)
        ok("fit: too strong a push-in — the move is shortened to the format's ceiling, the p90 scale is not above the ceiling",
           r["plan"]["fit"] < 1.0 and r["checks"]["scale_p90"] <= 4.5 + 1e-6, f"fit {r['plan']['fit']}, p90 scale {r['checks']['scale_p90']} %/s")
        keys = {"view": "wide", "azimuth": 200, "elevation": 12, "distance": 100, "lens": 35, "orbit": 0, "res": "640x360",
                "move": {"recipe": "arc", "subject": {"anchor": "castle.keep"}, "size": 0.45, "pitch": 12, "format": "documentary", "sec": 2, "fps": 12, "rig": "drone"}}
        out = S.plan_camera({}, keys, {"scene": T_}); cams = json.load(open(os.path.join(T_, "cameras.json")))
        mv = [cc for cc in cams if cc["name"].startswith("move_")]
        ok("camera step, move knob: move_NNN records with frame and fps, the result — the plan and the move checks",
           len(mv) == 24 and mv[0]["frame"] == 1 and mv[-1]["frame"] == 24 and mv[0]["fps"] == 12 and "move" in out and "_look" not in mv[0],
           f"{len(mv)} records; {out.get('move', {}).get('plan', {}).get('sweep')}°")
        for cc in cams:
            if cc["name"] == "move_005": cc["pinned"] = "bench, 2026-11-02"; cc["matrix_world"][0][3] += 1.0; x5 = cc["matrix_world"][0][3]
        json.dump(cams, open(os.path.join(T_, "cameras.json"), "w"))
        S.plan_camera({}, keys, {"scene": T_}); m5 = next(cc for cc in json.load(open(os.path.join(T_, "cameras.json"))) if cc["name"] == "move_005")
        ok("camera step: pinned move_005 as it is (the 1 m shift is intact), the others — by the move's decision", m5.get("pinned") and abs(m5["matrix_world"][0][3] - x5) < 1e-9)
