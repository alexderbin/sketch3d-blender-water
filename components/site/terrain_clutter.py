"""Small clutter near the camera (components/site/bl_terrain_clutter.py): downloading Poly Haven scans per library/terrain/clutter.yaml.
.venv/bin/python -m components.site.terrain_clutter fetch [--cfg library/terrain/rocks.yaml --out library/rocks/polyhaven]
→ library/clutter/polyhaven/<id>/<id>_<res>.blend + textures/, index.json
(the mean linear color of the color texture over the opaque pixels — to tint an instance by the ground). Poly Haven license — CC0."""
import json, os, sys, urllib.request
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "library/clutter/polyhaven"


def _get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "sketch3d/1.0"}), timeout=180).read()


def resolve(biome):
    """The biome's clutter config for Blender (which has no yaml) — in the site's surface.json: the biome's set per layer replaces the
    general one (in a desert — no grass)."""
    cfg = yaml.safe_load((ROOT / "library/terrain/clutter.yaml").read_text())
    dens = {lay: dict(v) for lay, v in cfg["density"]["default"].items()}
    for lay, v in (cfg["density"].get(biome) or {}).items(): dens[lay] = dict(v)
    return {k: cfg[k] for k in ("reach", "full", "lod", "assets")} | {"density": dens}


def fetch(cfg_path="library/terrain/clutter.yaml", out=None):
    """Poly Haven scans per a config (assets: role → id) → <out>/<id>/<id>_<res>.blend + textures/, index.json (mean color, object
    bounding box). One loader for clutter (clutter.yaml) and rocks (rocks.yaml, out=library/rocks/polyhaven)."""
    OUT_ = Path(out) if out else OUT
    cfg = yaml.safe_load((ROOT / cfg_path).read_text()); res = cfg.get("res", "1k")
    idx, mb = {}, 0.0
    for role, A in cfg["assets"].items():
        aid = A["id"]; d = OUT_ / aid
        F = json.loads(_get(f"https://api.polyhaven.com/files/{aid}"))["blend"][res]["blend"]
        for rel, url in [(os.path.basename(F["url"]), F["url"])] + [(k, v["url"]) for k, v in F.get("include", {}).items()]:
            p = d / rel; p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists(): data = _get(url); p.write_bytes(data); mb += len(data) / 1e6
        diff = next(iter(sorted(d.glob("textures/*_diff_*.jpg"))), None)
        alpha = next(iter(sorted(d.glob("textures/*_alpha_*.png"))), None)
        mean = None
        if diff:
            c = np.power(np.asarray(Image.open(diff).convert("RGB").resize((128, 128)), np.float32) / 255.0, 2.2).reshape(-1, 3)
            if alpha: c = c[np.asarray(Image.open(alpha).convert("L").resize((128, 128))).ravel() > 127]
            mean = [round(float(x), 4) for x in c.mean(0)]
        idx[aid] = {"blend": str((d / os.path.basename(F["url"])).relative_to(ROOT)), "mean": mean}
        print(f"{role}: {aid}", flush=True)
    (OUT_ / "index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=1))
    print(f"done → {(OUT_ / 'index.json').relative_to(ROOT)}; downloaded {mb:.0f} MB")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] != ["fetch"]: raise SystemExit(__doc__)
    fetch(*([a[a.index("--cfg") + 1]] if "--cfg" in a else []), **({"out": ROOT / a[a.index("--out") + 1]} if "--out" in a else {}))
