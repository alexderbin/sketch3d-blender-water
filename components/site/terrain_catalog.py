"""Catalog of ready-made sites (library/terrain/presets/*.yaml): each preset × seeds → a site, the tile texture, Blender views
(no bridge) → the preset sheet library/terrain/catalog/<name>.jpg (a row per seed: plan from above, overall, from below or
along) and the index library/terrain/catalog/INDEX.jpg (a preset is a row: the overall view of the first seed, title, stories)
+ library/terrain/CATALOG.md. The index tile and the run record — catalog/thumbs/<name>.jpg|json: the index is assembled from
all presets, a run of a subset does not cut it down.
.venv/bin/python -m components.site.terrain_catalog [name …] [--seeds 1,2,3] [--base real]   ~35 s per site; a heavy run — guard.py gate before the start.
--base real — sites on a real base (the library components/site/terrain_library.py, "find or make"): sheets — catalog/real/, scenes — cat_real_*."""
import core.paths         # the code is here, the core — core/<part> through kernel
import glob, json, os, sys, time
import yaml
from PIL import Image, ImageDraw, ImageFont
ROOT = core.paths.ROOT
from components.site import terrain as T

LIB = os.path.join(ROOT, "library", "terrain"); OUT = os.path.join(LIB, "catalog"); TH = os.path.join(OUT, "thumbs")
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def font(n):
    try: return ImageFont.truetype(FONT, n)
    except Exception: return ImageFont.load_default()


def tile(p, w, h):
    im = Image.open(p).convert("RGB"); k = max(w / im.width, h / im.height); im = im.resize((int(im.width * k) + 1, int(im.height * k) + 1))
    x, y = (im.width - w) // 2, (im.height - h) // 2; return im.crop((x, y, x + w, y + h))


def main():
    global OUT, TH
    a = sys.argv[1:]; seeds = [1, 2, 3]; base = None
    if "--base" in a: i = a.index("--base"); base = a[i + 1]; a = a[:i] + a[i + 2:]; OUT = os.path.join(OUT, base); TH = os.path.join(OUT, "thumbs")
    if "--seeds" in a: i = a.index("--seeds"); seeds = [int(x) for x in a[i + 1].split(",")]; a = a[:i] + a[i + 2:]
    names = a or [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(os.path.join(T.PRESETS, "*.yaml")))]
    os.makedirs(TH, exist_ok=True); ran, t_all = [], time.time()
    for nm in names:
        P = yaml.safe_load(open(os.path.join(T.PRESETS, nm + ".yaml"))); res_ = []
        for sd in seeds:
            t0 = time.time()
            try: r = T.scene(None, f"cat_{base + '_' if base else ''}{nm}_{sd}", T.preset(nm, sd, **({"base": base} if base else {})),
                             f"preset {nm} #{sd}" + (f" (base {base})" if base else ""), bridge=False)
            except SystemExit as e: print(f"{nm} #{sd}: error {e}", flush=True); continue
            res_.append((sd, r)); print(f"{nm} #{sd}: {time.time() - t0:.0f} s, knobs {r['preset']['knobs']}", flush=True)
        if not res_: continue
        W, H = 480, 270; S = H; f1, f2 = font(15), font(13)
        sheet = Image.new("RGB", (S + 2 * W + 10, 40 + len(res_) * (H + 26)), (22, 22, 22)); d = ImageDraw.Draw(sheet)
        d.text((8, 8), f"{P.get('title', nm)} ({nm}) — {', '.join(P.get('cases', []))}", fill=(255, 220, 90), font=f1)
        for k, (sd, r) in enumerate(res_):
            y = 40 + k * (H + 26); fold = os.path.join(ROOT, r["folder"]); sh = dict(zip(r["cams"], r["shots"] or []))
            d.text((8, y), f"№{sd}: " + ", ".join(f"{a}={b}" for a, b in r["preset"]["knobs"].items()), fill=(220, 220, 220), font=f2)
            sheet.paste(tile(os.path.join(fold, "sheet.png"), S, S) if os.path.exists(os.path.join(fold, "sheet.png")) else Image.new("RGB", (S, S)), (0, y + 18))
            second = next((c for c in r["cams"] if c != "wide"), None)   # the second view is the preset's first story camera
            for j, cam in enumerate(["wide", second]):
                if cam and cam in sh: sheet.paste(tile(os.path.join(ROOT, sh[cam]) if not os.path.isabs(sh[cam]) else sh[cam], W, H), (S + 5 + j * (W + 5), y + 18))
        sheet.save(os.path.join(OUT, nm + ".jpg"), quality=85)
        sh0 = dict(zip(res_[0][1]["cams"], res_[0][1]["shots"] or [])).get("wide")
        if sh0: tile(sh0 if os.path.isabs(sh0) else os.path.join(ROOT, sh0), 420, 236).save(os.path.join(TH, nm + ".jpg"), quality=88)
        json.dump({"seeds": [s for s, _ in res_], "sheet": f"{os.path.relpath(OUT, LIB)}/{nm}.jpg"}, open(os.path.join(TH, nm + ".json"), "w")); ran.append((nm, len(res_)))
    Y = sorted(glob.glob(os.path.join(T.PRESETS, "*.yaml"))); rows = []   # all presets that have a run record; the title and knobs come from the preset
    for p in Y:
        nm = os.path.splitext(os.path.basename(p))[0]; j = os.path.join(TH, nm + ".json")
        if os.path.exists(j): rows.append((nm, yaml.safe_load(open(p)), json.load(open(j))))
    if rows:                                                        # index: a row per preset
        W, H = 420, 236; C = 3; f1, f2 = font(15), font(12)
        idx = Image.new("RGB", (C * (W + 6), ((len(rows) + C - 1) // C) * (H + 54)), (22, 22, 22)); d = ImageDraw.Draw(idx)
        for k, (nm, P, _) in enumerate(rows):
            x, y = (k % C) * (W + 6), (k // C) * (H + 54); th = os.path.join(TH, nm + ".jpg")
            if os.path.exists(th): idx.paste(tile(th, W, H), (x, y + 50))
            d.text((x + 4, y + 4), f"{P.get('title', nm)}", fill=(255, 220, 90), font=f1)
            d.text((x + 4, y + 26), f"{nm} · {P.get('biome')} · " + ", ".join(P.get("cases", [])[:3]), fill=(210, 210, 210), font=f2)
        idx.save(os.path.join(OUT, "INDEX.jpg"), quality=85)
    md = ["# Ready-made sites — preset catalog", "", "A preset is a site concept (relief, biome, knobs with ranges, scatters of shapes); a seed gives its own site of the same",
          "concept. Scene: `.venv/bin/python -m components.site.terrain preset <name> [seed] [scene] [knob=value …]`; in your own location: `relief: {preset:",
          "<name>, seed: N}`. Sheets — `catalog/<name>.jpg`, index — `catalog/INDEX.jpg`. Rebuild the catalog — `components/site/terrain_catalog.py`.", ""]
    md += ["| Preset | Biome | For stories | Knobs | Sheet |", "|---|---|---|---|---|"]
    for nm, P, e in rows:
        md.append(f"| {P.get('title')} (`{nm}`) | {P.get('biome')} | {', '.join(P.get('cases') or [])} | "
                  + ", ".join(f"{k} {v}" for k, v in (P.get("knobs") or {}).items()) + f" | [{e['sheet']}]({e['sheet']}) |")
    if rows and len(rows) == len(Y): open(os.path.join(LIB, "CATALOG.md" if OUT.endswith("catalog") else f"CATALOG-{os.path.basename(OUT)}.md"), "w").write("\n".join(md) + "\n")
    print("__RESULT__" + json.dumps({"presets": len(ran), "variants": sum(n for _, n in ran), "in_index": len(rows), "index": os.path.relpath(os.path.join(OUT, "INDEX.jpg"), ROOT),
                                     "s": round(time.time() - t_all)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
