"""Reference test of the "Site" component: the presets library/terrain/presets/*.yaml with seed 1 → terrain (terrain.build, the project's Python,
no Blender) → a fingerprint: the heights of the near and far grids and the water level (to a centimetre), the terrain summary (stats) and the result of the site
checks (checks). Compared with golden.json next to it. Without names — the reference sites (12 natural ones); presets without a reference (12 settlement sites)
are not compared, they are listed in a "no reference" line. The reference of a new site is recorded only after
the frame is checked: by name with --record. A change that alters the terrain is deliberate: --record in the same commit, with the reason in the message.
.venv/bin/python -m components.site.bench.bench [name …] [--record]   → __RESULT__ and exit code = the number of mismatches; ~2–3 min"""
import glob, hashlib, json, os, sys, time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from components.site import terrain as T  # noqa: E402

GOLD = os.path.join(os.path.dirname(os.path.realpath(__file__)), "golden.json")


def h(a):
    return hashlib.sha1(np.round(np.nan_to_num(np.asarray(a, np.float64), nan=-9999.0), 2).tobytes()).hexdigest()[:16]


def fingerprint(name):
    t = time.time(); o = T.build(T.preset(name, 1))
    ch = o["loc"]["checks"]
    return {"relief": h(o["near"].H), "far": h(o["far"].H), "water": h(o["level"]), "summary": o["loc"]["stats"],
            "checks": [len(ch), sum(1 for c in ch if c.get("ok", True))], "s": round(time.time() - t, 1)}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]; write = "--record" in sys.argv
    presets = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(ROOT, "library/terrain/presets/*.yaml")))
    gold = json.load(open(GOLD)) if os.path.exists(GOLD) else {}
    names = args or sorted(gold)                                       # without names — those in the reference (and --record: re-record the same ones)
    unchecked = [] if args else [n for n in presets if n not in gold]  # presets without a reference are not compared, but are listed
    fails, t0 = 0, time.time()
    for n in names:
        if n not in presets:
            fails += 1; print("FAIL", n, "— no preset library/terrain/presets/" + n + ".yaml"); continue
        fp = fingerprint(n); g = gold.get(n)
        if write: gold[n] = fp; print("WRITE", n, fp["relief"], fp["checks"], fp["s"]); continue
        if g is None:
            fails += 1; print("FAIL", n, fp["s"], "s — no reference (record it by name with --record after the frame is checked):",
                              {k: fp[k] for k in ("relief", "checks")}); continue
        diff = [k for k in ("relief", "far", "water", "summary", "checks") if g.get(k) != fp[k]]
        fails += bool(diff)
        print("OK  " if not diff else "FAIL", n, fp["s"], "s", "" if not diff else {k: g.get(k) for k in diff} | {"now": {k: fp[k] for k in diff}})
    if unchecked: print(f"no reference ({len(unchecked)}, not compared):", ", ".join(unchecked))
    if write: json.dump(gold, open(GOLD, "w"), ensure_ascii=False, indent=1)
    print("__RESULT__" + json.dumps({"places": len(names), "mismatches": fails, "no_golden": len(unchecked), "s": round(time.time() - t0, 1)},
                                    ensure_ascii=False))
    sys.exit(fails)


if __name__ == "__main__":
    main()
