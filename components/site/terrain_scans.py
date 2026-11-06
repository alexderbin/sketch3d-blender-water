"""Ground photoscans for the site shader (core/surface/bl_terrain_surf.py): downloaded from ambientCG per library/terrain/scans.yaml.
.venv/bin/python -m components.site.terrain_scans fetch [--res 1K]  → library/textures/terrain/<id>/<id>_<res>-JPG_<map>.jpg, index.json
(mean color and roughness); backfill — add roughness to a finished index.json without downloading
ambientCG license — CC0. The shader removes tile repetition by itself (noise offsets, macro patches, larger in the distance)."""
import io, json, sys, urllib.request, zipfile
from pathlib import Path
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "library/textures/terrain"


def rough_of(p):
    """Mean roughness of a scan map (as the shader reads it: Non-Color, 0…1)."""
    return round(float((np.asarray(Image.open(p).convert("L").reduce(8), np.float32) / 255.0).mean()), 3)


def backfill():
    """Add to a finished index.json what fetch now computes itself (roughness), without downloading."""
    p = OUT / "index.json"; J = json.loads(p.read_text())
    for aid, S in J["scans"].items():
        if S.get("Roughness") and (ROOT / S["Roughness"]).exists(): S["rough"] = rough_of(ROOT / S["Roughness"])
    p.write_text(json.dumps(J, ensure_ascii=False, indent=1)); return {k: v.get("rough") for k, v in J["scans"].items()}


def fetch(res=None):
    cfg = yaml.safe_load((ROOT / "library/terrain/scans.yaml").read_text())
    res = res or cfg.get("res", "2K")
    idx, mb = {}, 0.0
    for aid, S in cfg["scans"].items():
        d = OUT / aid
        d.mkdir(parents=True, exist_ok=True)
        want = {m: d / f"{aid}_{res}-JPG_{m}.jpg" for m in cfg["maps"]}
        if not all(p.exists() for p in want.values()):
            url = f"https://ambientcg.com/get?file={aid}_{res}-JPG.zip"
            req = urllib.request.Request(url, headers={"User-Agent": "sketch3d/1.0"})
            data = urllib.request.urlopen(req, timeout=300).read(); mb += len(data) / 1e6
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for m, p in want.items():
                    name = next((n for n in z.namelist() if n.endswith(f"_{m}.jpg")), None)
                    if name is not None: p.write_bytes(z.read(name))
            print(f"{aid} {res} — {len(data) / 1e6:.1f} MB archive", flush=True)
        idx[aid] = {"size": S["size"], "kind": S.get("kind", ""), "veg": bool(S.get("veg")), **{m: str(p.relative_to(ROOT)) for m, p in want.items() if p.exists()}}
        if want["Color"].exists():                                  # mean color (linear) — for matching the tone to the site
            c = np.asarray(Image.open(want["Color"]).convert("RGB").resize((64, 64)), np.float32) / 255.0
            idx[aid]["mean"] = [round(float(x), 4) for x in np.power(c, 2.2).mean((0, 1))]
        if want.get("Roughness") and want["Roughness"].exists():    # mean roughness — handed to the shader (there is no pattern in the distance — the highlight follows the mean)
            idx[aid]["rough"] = rough_of(want["Roughness"])
    (OUT / "index.json").write_text(json.dumps({"res": res, "scans": idx}, ensure_ascii=False, indent=1))
    print(f"done → {(OUT / 'index.json').relative_to(ROOT)}; downloaded {mb:.0f} MB")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in ("fetch", "backfill"):
        raise SystemExit(__doc__)
    print(backfill()) if a[0] == "backfill" else fetch(a[a.index("--res") + 1] if "--res" in a else None)
