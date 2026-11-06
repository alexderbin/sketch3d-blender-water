"""Reference test of the "Plants and details" component (pass / fail): on a copy of the urban_squad scene (as in the smoke check) — trees of two
species, boulders; the answer of each step (__RESULT__: species, vertices, heights) and the fingerprint of the appended scene.obj against golden.json.
The banner (cloth) moved to the Cloth component (components/cloth) — its test bench is there.
.venv/bin/python -m components.plants.bench.bench [--record]   → line __RESULT__ {n_failures, n_checks}; ~1 min."""
import hashlib, json, os, shutil, subprocess, sys, tempfile, time
import core.paths

HERE = os.path.dirname(os.path.realpath(__file__)); ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "core")): ROOT = os.path.dirname(ROOT)
GOLD = os.path.join(HERE, "golden.json"); t0 = time.time()
# the shrub (shrub) is not in the reference: the number of vertices differs between runs with the same seed (3304 / 3412) — nondeterminism of the shrub skeleton
CASES = [("spruce", "bl_tree.py", ["pine", "6,9", "1"]),
         ("tree", "bl_tree.py", ["oak", "-6,10", "1"]), ("boulders", "bl_boulder.py", ["rock", "-1,6", "n=2", "spread=1"])]
S = os.path.join(tempfile.mkdtemp(prefix="plants_"), "scene"); os.makedirs(S)
for f in ("scene.obj", "world.json", "figkit.json", "cameras.json"): shutil.copy(os.path.join(ROOT, "bench/scenes/urban_squad", f), S)
got = {}; BASE = len(open(os.path.join(S, "scene.obj")).read())
for name, tool, args in CASES:
    p = subprocess.run(["blender", "-b", "--factory-startup", "-P", core.paths.RUN, "--", os.path.join(ROOT, "components/plants", tool), "--", S, *args],
                       capture_output=True, text=True, cwd=ROOT, timeout=600)
    line = next((l for l in p.stdout.splitlines() if l.startswith("__RESULT__")), None)
    got[name] = json.loads(line[len("__RESULT__"):]) if line else {"error": (p.stderr or p.stdout)[-300:]}
def parts(path, skip):                                              # appended parts of scene.obj: vertices and bounding size (to 1 cm)
    P, cur = {}, None
    for l in open(path).read()[skip:].splitlines():
        if l[:2] in ("o ", "g "): cur = l[2:].strip(); P.setdefault(cur, [])
        elif l.startswith("v ") and cur: P[cur].append(tuple(float(x) for x in l.split()[1:4]))
    return {k: [len(v)] + [round(f(c[i] for c in v), 2) for f in (min, max) for i in range(3)] for k, v in P.items() if v}
got["scene.obj"] = parts(os.path.join(S, "scene.obj"), BASE)       # the file bytes differ between runs (write order)
got["world.json"] = hashlib.sha256(open(os.path.join(S, "world.json"), "rb").read()).hexdigest()[:16]
if "--record" in sys.argv:
    json.dump(got, open(GOLD, "w"), ensure_ascii=False, indent=1); print("written", GOLD)
gold = json.load(open(GOLD)); bad = [k for k in gold if gold[k] != got.get(k)]
for k in bad: print("MISMATCH", k, "\n  was:", json.dumps(gold[k], ensure_ascii=False)[:400], "\n  now:", json.dumps(got.get(k), ensure_ascii=False)[:400])
print("__RESULT__", json.dumps({"failures": len(bad), "checks": len(gold), "sec": round(time.time() - t0, 1)}, ensure_ascii=False))
sys.exit(len(bad))
