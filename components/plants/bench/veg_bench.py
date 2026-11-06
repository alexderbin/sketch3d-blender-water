"""Vegetation reference test (pass / fail): (1) knowledge → specifications (plants.resolve) — a JSON fingerprint; (2) the Blender build — vertices,
cards, bounding size, a vertex fingerprint by parts (determinism by seed); (3) the forest computation at a catalogue site — counts by species, classes,
layers, trunks per ha; (4) the biome's herb layer — species composition. Against veg_golden.json.
.venv/bin/python -m components.plants.bench.veg_bench [--record] → __RESULT__ {n_failures, n_checks}; ~30 s."""
import hashlib, json, os, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.realpath(__file__)); ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "core")): ROOT = os.path.dirname(ROOT)
import core.paths
from components.plants import plants as PL
from components.plants import forest as FO

GOLD = os.path.join(HERE, "veg_golden.json"); t0 = time.time(); got = {}
CASES = [("oak", "adult", "healthy", 3), ("spruce", "adult", "healthy", 4), ("birch", "young", "healthy", 5), ("palm", "adult", "healthy", 6),
         ("hazel", "adult", "healthy", 7), ("oak", "old", "dead", 8), ("kapok", "adult", "healthy", 9), ("meadow_grass", "adult", "healthy", 10),
         ("daisy", "adult", "healthy", 11), ("fern", "adult", "healthy", 12)]
PL.textures(); specs = [PL.resolve(*c[:3], seed=c[3]) for c in CASES]
got["specs"] = {s["id"]: hashlib.sha256(json.dumps(s, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12] for s in specs}
tmp = os.path.join(tempfile.mkdtemp(prefix="veg_"), "specs.json"); json.dump({"plants": specs}, open(tmp, "w"), ensure_ascii=False)
p = subprocess.run(["blender", "-b", "--factory-startup", "-P", core.paths.RUN, "--", os.path.join(HERE, "bl_veg_bench.py"), "--", tmp], capture_output=True, text=True, cwd=ROOT, timeout=900)
line = next((l for l in p.stdout.splitlines() if l.startswith("__RESULT__")), None)
got["build"] = json.loads(line[len("__RESULT__"):]) if line else {"error": (p.stderr or p.stdout)[-400:]}
R = FO.plan(os.path.join(ROOT, "out/relief/cat_real_fortress_hill_1"), 1.0, 1, box=(-60, -60, 60, 60))
got["forest"] = {k: R["stats"][k] for k in ("trees_n", "shrubs_n", "by_species", "by_class", "by_layer", "stems_per_ha", "closure")}
got["meadow"] = {b: sorted(PL.meadow_specs(b, K=1)) for b in ("temperate", "desert", "jungle", "alpine")}
from components.plants import leaf_optics as LO                                   # leaf optics from pigments (PROSPECT-D): reflectance and transmittance shares in the visible range
got["optics"] = {n: [LO.optics(n)["R_vis"], LO.optics(n)["T_vis"]] for n in ("broad", "broad_thin", "needle", "grass", "autumn_yellow", "dry_grass")}
if "--record" in sys.argv:
    json.dump(got, open(GOLD, "w"), ensure_ascii=False, indent=1); print("written", GOLD)
gold = json.load(open(GOLD)); bad = []
for k in gold:
    if isinstance(gold[k], dict) and isinstance(got.get(k), dict):
        for kk in gold[k]:
            if gold[k][kk] != got[k].get(kk): bad.append(f"{k}/{kk}")
    elif gold[k] != got.get(k): bad.append(k)
for k in bad:
    a, b = k.split("/") if "/" in k else (k, None); g0 = gold[a][b] if b else gold[a]; g1 = (got.get(a) or {}).get(b) if b else got.get(a)
    print("MISMATCH", k, "\n  was:", json.dumps(g0, ensure_ascii=False)[:300], "\n  now:", json.dumps(g1, ensure_ascii=False)[:300])
print("__RESULT__", json.dumps({"failures": len(bad), "checks": sum(len(v) if isinstance(v, dict) else 1 for v in gold.values()), "sec": round(time.time() - t0, 1)}, ensure_ascii=False))
sys.exit(1 if bad else 0)
