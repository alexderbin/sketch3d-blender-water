"""Test bench of the forest stand-ins: the stand-in box (bl_stand_in.parts — the same vertices the draft builds;
Blender is not needed) against the box of a real tree of the same species, state and height (bl_plant, tier 0, no tilt) — a tolerance of
10 % in width (the mean of x and y; plus sampling noise for species with a large spread of trees) and in height against the mean box of the check
seeds; the check uses seeds and heights for which the silhouette (stand_in.json) was not measured.
.venv/bin/python -m components.plants.bench.stand_in_bench            → a table by species and __RESULT__ {n_failures, n_checks}; ~1 s
.venv/bin/python -m core.services.guard heavy --wait 1800 --what plants_stand_in -- .venv/bin/python -m components.plants.bench.stand_in_bench --measure
    → Blender (~4 min): the catalogue trees (cases) → stand_in.json (silhouettes — role cal) and stand_in_golden.json (check boxes — role val)."""
import json, os, subprocess, sys, tempfile, time, types

HERE = os.path.dirname(os.path.realpath(__file__)); ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "core")): ROOT = os.path.dirname(ROOT)
import core.paths
import numpy as np
from components.plants import forest as FO
from components.plants import plants as PL
from components.plants import stand_in as PS

GOLD = os.path.join(HERE, "stand_in_golden.json"); TOL = 0.10; MAX_FACES = 200
_R = np.random.default_rng(2026)                                      # seeds scattered: neighbouring ones (101, 102, …) give, in the m_tree engine,
CAL_SEEDS = tuple(int(x) for x in _R.choice(np.arange(1, 100000), 8, replace=False))   # similar trees (measured: the block 101–106 is
VAL_SEEDS = tuple(int(x) for x in _R.choice(np.arange(1, 100000), 4, replace=False))   # narrower than the block 201–204 by 15 % for the willow); the forest takes scattered ones


def cases():
    """Measurement trees (species, age, state, height, seed, role): for each tree species of the catalogue — the forest states (forest.plan,
    with_specs): mature, young, broken, winter (for deciduous), dead. Silhouette (cal) — three heights × two seeds: the heights are the range
    of the state in the forest (mature 0.5–1 of the species' mean height, young 0.12–0.42; a shrub has its own range); check (val) — other heights
    and seeds. Silhouette — 3 heights × 8 seeds, check — 2 × 4: for willow, shrubs and dead trees the width varies with the seed by 10–25 %."""
    out = []
    for sid in PL.knowledge()["species"]:
        S = PL.species(sid); hab = S["habit"]
        if hab not in ("tree", "shrub", "palm"): continue
        lo, hi = S["height"]; Hm = (lo + hi) / 2; tree = hab != "shrub"; fixed = bool(S.get("state")); ever = FO.evergreen(sid)
        if tree: A, Av, Y, Yv = [f * Hm for f in (0.5, 0.75, 1.0)], [f * Hm for f in (0.6, 0.9)], [f * Hm for f in (0.12, 0.25, 0.42)], [f * Hm for f in (0.18, 0.35)]
        else: A = [0.7 * lo, Hm, hi]; Av = [(A[0] + Hm) / 2, (Hm + hi) / 2]

        def add(age, state, Hc, Hv):
            out.extend([(sid, age, state, h, s_, "cal") for h in Hc for s_ in CAL_SEEDS] + [(sid, age, state, h, s_, "val") for h in Hv for s_ in VAL_SEEDS])
        add("adult", "healthy", A, Av)
        if tree: add("young", "healthy", Y, Yv)
        if fixed: continue                                            # dead standing wood is already bare
        if tree: add("old", "broken", A[1:], Av[1:]); add("adult", "dead", A, Av); add("young", "dead", Y, Yv[:1])
        if not ever: add("adult", "winter", A, Av)
        if tree and not ever: add("young", "winter", Y, Yv)
    return out


def spec_of(sid, age, state, H, seed):
    return PL.resolve(sid, age, state, seed=seed, height=round(H, 2), lean={"deg": 0.0, "az": 0.0, "recover": 0.6})


def measure():
    """Measurement in Blender (call it under guard heavy) → stand_in.json and stand_in_golden.json; → a summary."""
    PL.textures(); C = cases(); specs = [spec_of(*c[:5]) for c in C]; role = {(sp["id"], sp["height"]): c[5] for c, sp in zip(C, specs)}
    d = tempfile.mkdtemp(prefix="stand_in_"); sj, oj = os.path.join(d, "specs.json"), os.path.join(d, "trees.json")
    json.dump({"plants": specs}, open(sj, "w"), ensure_ascii=False)
    p = subprocess.run(["blender", "-b", "--factory-startup", "--python-exit-code", "1", "-P", core.paths.RUN, "--",
                        os.path.join(HERE, "bl_stand_in_measure.py"), "--", sj, oj], capture_output=True, text=True, cwd=ROOT, timeout=3000)
    line = next((l for l in p.stdout.splitlines() if l.startswith("__RESULT__")), None)
    if p.returncode or not line or not os.path.exists(oj): raise SystemExit("measurement failed: " + (p.stderr or p.stdout)[-800:])
    raw = json.load(open(oj)); T = raw["trees"]
    for r in T: r["role"] = role.get((r["id"], r["H"]), "val")
    cal = PS.calibrate([r for r in T if r["role"] == "cal"], raw["tex"])
    cal["measured"] = time.strftime("%Y-%m-%d"); cal["source"] = "bl_plant tier 0, no tilt; bench/stand_in_bench.py --measure"
    json.dump(cal, open(PS.CAL, "w"), ensure_ascii=False, indent=1)
    keep = ("species", "age", "state", "seed", "H", "foliage", "min", "max", "z0", "z1", "sec")
    rows = [json.dumps({k: r.get(k) for k in keep}, ensure_ascii=False) for r in T if r["role"] == "val"]   # one record per line: the diff stays readable
    open(GOLD, "w").write('{"measured": "%s", "trees": [\n%s\n]}\n' % (cal["measured"], ",\n".join(rows)))
    PS._CAL = None
    sec = [r["sec"] for r in T]
    return {"trees": len(T), "build_s": round(sum(sec), 1), "slowest": max(T, key=lambda r: r["sec"])["id"], "blender": json.loads(line[len("__RESULT__"):])}


def frame_of(r):
    """Stand-in box of a tree from measurement record r (on the ground 0): widths x, y and the top — from the vertices of bl_stand_in.parts; faces per tree."""
    sys.modules.setdefault("bpy", types.ModuleType("bpy"))            # the core primitives are numpy; bpy is needed only to build the object
    from components.plants import bl_stand_in as BSI
    sp = spec_of(r["species"], r["age"], r["state"], r["H"], r["seed"])
    t = {"species": r["species"], "age": r["age"], "state": r["state"], "h": r["H"], "x": 0.0, "y": 0.0, "seed": r["seed"], "spec": sp,
         "lean": {"deg": 0.0, "az": 0.0}, "push": [0.0, 0.0], "cr": 0.5 * float(PL.species(r["species"]).get("width", 0.5)) * r["H"]}
    P, ok = BSI.parts(t, 0.0); V = np.vstack([p["V"] for p in P])
    return float(np.ptp(V[:, 0])), float(np.ptp(V[:, 1])), float(V[:, 2].max()), sum(len(p["F"]) for p in P), ok


def check():
    """The stand-in box against the mean box of real trees (check seeds) of the same species, state and height. Width (x and y on
    average): within the tolerance TOL — "exact"; within TOL + sampling noise — "within spread" (noise — 2σ of the difference of means: the species' tree spread
    from stand_in.json × √(1/silhouette seeds + 1/check seeds); for willow, acacia, the "alien tree", trees of the same size differ
    by 10–35 %, and the mean over 8 and 4 seeds is itself imprecise); otherwise — failure. Height — TOL. Faces — up to MAX_FACES."""
    G = json.load(open(GOLD)); grp, rows, bad = {}, [], []
    for r in G["trees"]:
        if r.get("max"): grp.setdefault((r["species"], r["age"], r["state"], r["H"]), []).append(r)
    for (sid, age, st, H), L in grp.items():
        ex, ey, top, nf, ok = frame_of(L[0]); W = [(r["max"][0] - r["min"][0] + r["max"][1] - r["min"][1]) / 2 for r in L]
        w_ = sum(W) / len(W); t_ = sum(r["max"][2] for r in L) / len(L); ew = (ex + ey) / 2 / w_ - 1; eh = top / t_ - 1
        c = PS.pick(sid, PS.cond(age, st, L[0]["foliage"])) or {}
        noise = 2 * c.get("spread", 0.0) * (1 / max(c.get("seeds", 1), 1) + 1 / len(L)) ** 0.5
        kind = "exact" if abs(ew) <= TOL else "within spread" if abs(ew) <= TOL + noise else "failure"
        fail = kind == "failure" or abs(eh) > TOL or nf > MAX_FACES or not ok
        rows.append((sid, age, st, H, ew, eh, noise, nf, ok, fail, kind))
        if fail: bad.append(f"{sid} {age} {st} H={H:.1f}: width {ew:+.0%} (noise {noise:.0%}), height {eh:+.0%}")
    print(f"{'kind':<13}{'age':<8}{'state':<10}{'H, m':>6}{'width':>9}{'noise':>7}{'height':>9}{'n_faces':>8}")
    for sid, age, st, H, ew, eh, noise, nf, ok, fail, kind in rows:
        print(f"{sid:<13}{age:<8}{st:<10}{H:>6.1f}{ew:>+9.1%}{noise:>7.0%}{eh:>+9.1%}{nf:>8}  {kind}" + ("" if ok else ", not measured") + ("  OUT OF TOLERANCE" if fail else ""))
    w = np.abs([x[4] for x in rows]); h = np.abs([x[5] for x in rows]); un = [s for s in PL.knowledge()["species"]
        if PL.species(s)["habit"] in ("tree", "shrub", "palm") and s not in PS.calibration()["species"]]
    return {"failures": len(bad) + len(un), "checks": len(rows), "exact": sum(x[10] == "exact" for x in rows), "within_spread": sum(x[10] == "within spread" for x in rows),
            "width_err_mean": round(float(w.mean()), 3), "width_err_max": round(float(w.max()), 3), "height_err_max": round(float(h.max()), 3),
            "faces_max": max(x[7] for x in rows), "uncalibrated": un, "out_of_tolerance": bad}


if __name__ == "__main__":
    t0 = time.time(); res = {}
    if "--measure" in sys.argv: res["measure"] = measure()
    res.update(check()); res["sec"] = round(time.time() - t0, 1)
    print("__RESULT__", json.dumps(res, ensure_ascii=False))
    sys.exit(1 if res["failures"] else 0)
