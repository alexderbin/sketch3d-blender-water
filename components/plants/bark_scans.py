"""Poly Haven bark scans (CC0) for the foreground: species → scan (BARK), downloading 1k jpg (colour, GL normals, roughness; ~2.5 MB per scan)
into library/plants/bark/<scan>/. Bark in Blender — bl_plant.bark_material: if there is a scan, it is applied over the m_tree engine's unwrap, otherwise a procedural pattern.
.venv/bin/python -m components.plants.bark_scans fetch — download the missing ones (birch and "alien" are procedural)"""
import core.paths
import json, os, sys, urllib.request

DIR = os.path.join(core.paths.ROOT, "library", "plants", "bark")
BARK = {"oak": "jolcham_oak_bark_01", "beech": "japanese_hackberry_bark", "poplar": "bark_willow", "willow": "bark_willow_02",
        "spruce": "bark_brown_02", "pine": "pine_bark", "fir": "chinese_hackberry_bark", "cypress": "japanese_cedar_bark",
        "palm": "palm_tree_bark", "acacia": "bark_brown_01", "kapok": "japanese_zelkova_bark", "snag": "tree_bark_03",
        "hazel": "trident_maple_bark", "juniper": "chinese_cedar_bark"}
MAPS = {"Diffuse": "diff", "nor_gl": "nor_gl", "Rough": "rough"}


def paths(scan):
    d = os.path.join(DIR, scan); return {v: os.path.join(d, f"{scan}_{v}_1k.jpg") for v in MAPS.values()}


def _get(url, path=None):
    """Poly Haven answers 403 to the default Python user agent — a custom header."""
    req = urllib.request.Request(url, headers={"User-Agent": "sketch3d-plants/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    if path: open(path, "wb").write(data); return len(data)
    return json.loads(data)


def fetch(scans=None):
    got = {}
    for scan in sorted(set(scans or BARK.values())):
        P = paths(scan)
        if all(os.path.exists(p) for p in P.values()): got[scan] = "present"; continue
        os.makedirs(os.path.dirname(P["diff"]), exist_ok=True)
        files = _get(f"https://api.polyhaven.com/files/{scan}"); n = 0
        for k, v in MAPS.items(): n += _get(files[k]["1k"]["jpg"]["url"], P[v])
        got[scan] = round(n / 1e6, 1)
    return got


if __name__ == "__main__":
    if sys.argv[1:2] == ["fetch"]: print("__RESULT__" + json.dumps(fetch(), ensure_ascii=False))
    else: print(__doc__)
