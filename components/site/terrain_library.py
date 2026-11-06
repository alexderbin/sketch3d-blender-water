"""Library of real bases: USGS 3DEP relief windows (lidar, 1 m step, public domain) by shape type — for "find or make". Regions and
type requirements — library/terrain/real/regions.yaml.
Collection: survey of a region (2 km, 4 m step) → 400 m windows on a 100 m grid → analysis (terrain_analysis) → those that satisfy
the requirements, best by the type's score, not closer than 500 m to each other → a 600 m window at a 1 m step → re-check of the
requirements → npz (uint16 heights at 1 cm) + summary → library/terrain/real/<type>/<name>.npz, index.json; sheet — real/<type>.jpg.
.venv/bin/python -m components.site.terrain_library build [type …] [--per 4]     (~1.4 MB download and ~0.5 MB on disk per window)
.venv/bin/python -m components.site.terrain_library sheet [type …]
.venv/bin/python -m components.site.terrain_library prune          — re-check the collected windows against the current requirements
In code: L.load(name) → (H, m; step, m; meta); L.find(type, seed, need) → the name of a suitable window."""
import core.paths as _core_paths         # the code is here, the core — core/<part> through kernel
import io, json, math, os, sys, time, urllib.request

import numpy as np

ROOT = _core_paths.ROOT
from core.surface import terrain_analysis as TA

LIB = os.path.join(ROOT, "library", "terrain", "real")
URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"


def fetch(lat, lon, half_m, n, tries=3):
    """A 2·half_m × 2·half_m m window around a point, n × n samples; rows run from south to north (the site's y grows to the north)."""
    from PIL import Image
    dl, dn = half_m / 111320, half_m / (111320 * math.cos(math.radians(lat)))
    q = (f"?bbox={lon - dn},{lat - dl},{lon + dn},{lat + dl}&bboxSR=4326&imageSR=4326&size={n},{n}&format=tiff&pixelType=F32"
         "&interpolation=RSP_BilinearInterpolation&f=image")
    for k in range(tries):
        try:
            Z = np.array(Image.open(io.BytesIO(urllib.request.urlopen(URL + q, timeout=120).read())), np.float64)[::-1].copy()
            bad = ~np.isfinite(Z) | (Z < -500)                          # no data (over the sea) — fill with the window's lowest height: at the coast it is sea
            if bad.any(): Z[bad] = float(np.min(Z[~bad])) if (~bad).any() else 0.0
            return Z
        except Exception as e:
            if k == tries - 1: raise
            time.sleep(3 * (k + 1))


def summary(H, res, need=None, zabs=None, area=None):
    """Window summary for the requirements: drop, cliffs, slope, flat share, the stream, saddle, summit and bowl nearest the center, sea
    share (by absolute height zabs), fine texture. Points are in meters from the window center. area — the catchment at the center,
    computed over the region (inside the window the stream is cut off: its catchment lies beyond the edge)."""
    ny, nx = H.shape; c = np.array([(ny - 1) / 2, (nx - 1) / 2]); S = TA.slope(H, res)
    out = {"relief": float(np.percentile(H, 98) - np.percentile(H, 2)), "steep": float((S > 40).mean()), "slope50": float(np.median(S)),
           "rough": float(np.std((H - TA.blur(H, 2.5 / res))[5:-5, 5:-5])), "dead_flat": float((S < 0.5).mean())}
    if zabs is not None: out["sea"] = float((zabs <= 0.5).mean())
    need = need or {}
    if "flat_max" in need:
        C = TA.geomorphons(H, res, 20.0); v = C[C >= 0]; out["flat"] = float((v == 0).mean()) if len(v) else 1.0
    if "stream" in need:
        if area is None:
            F = TA.fill(H); A = TA.accumulate(F, TA.receivers(F, res), res); r = int(need["stream"]["r"] / res)
            area = float(A[int(c[0]) - r:int(c[0]) + r + 1, int(c[1]) - r:int(c[1]) + r + 1].max())
        out["stream_area"] = float(area)
    if "saddle" in need:
        sd = [d for s_ in (20.0, 30.0, 50.0) for d in TA.saddles(H, res, s_, n=4)]
        near = [d for d in sd if math.dist(d["ij"], c) * res <= need["saddle"]["r"]]
        out["saddle_drop"] = max([d["drop"] for d in near], default=0.0)
        if near: b = max(near, key=lambda d: d["drop"]); out["saddle"] = {"at": [(b["ij"][1] - c[1]) * res, (b["ij"][0] - c[0]) * res], "road_deg": b["road_deg"]}
    for key, sign in (("summit", 1), ("bowl", -1)):
        if key in need:
            pk = TA.peaks(H, res, sign=sign, min_rel=2.0); near = [d for d in pk if math.dist(d["ij"], c) * res <= need[key]["r"]]
            if key == "summit": near = [d for d in near if d.get("flat_top_m2", 0) >= need[key].get("top", 0)]
            out[f"{key}_rel"] = max([d["drop"] for d in near], default=0.0)
            if near: b = max(near, key=lambda d: d["drop"]); out[key] = {"at": [(b["ij"][1] - c[1]) * res, (b["ij"][0] - c[0]) * res]}
    return out


def ok(s, need, coarse=False):
    rng = lambda k: k not in need or need[k][0] <= s[k] <= need[k][1]
    checks = [rng("relief"), rng("slope50"), rng("sea") if "sea" in s else True, s.get("dead_flat", 0) <= need.get("dead_flat_max", 0.6)]
    if not coarse: checks += [rng("steep"), s.get("rough", 1) >= need.get("rough_min", 0.05), s.get("flat", 0) <= need.get("flat_max", 1)]
    if "stream" in need: checks.append(s.get("stream_area", 0) >= need["stream"]["area"])
    if "saddle" in need: checks.append(s.get("saddle_drop", 0) >= need["saddle"]["drop"])
    if "summit" in need: checks.append(s.get("summit_rel", 0) >= need["summit"]["rel"])
    if "bowl" in need: checks.append(s.get("bowl_rel", 0) >= need["bowl"]["rel"])
    return all(checks)


def score(s, keys):
    return tuple(round(s.get(k, 0.0), 3) for k in keys)


def config():
    import yaml
    return yaml.safe_load(open(os.path.join(LIB, "regions.yaml")))["types"]


def build(types=None, per=4, log=print):
    T = config(); idx_p = os.path.join(LIB, "index.json"); idx = json.load(open(idx_p)) if os.path.exists(idx_p) else {}
    for tp, cf in T.items():
        if types and tp not in types: continue
        need = cf.get("need", {}); os.makedirs(os.path.join(LIB, tp), exist_ok=True)
        for reg, lat, lon in cf["regions"]:
            if any(v["type"] == tp and v["region"] == reg for v in idx.values()): log(f"{tp}/{reg}: already present"); continue
            t0 = time.time()
            try: Z = fetch(lat, lon, 1000.0, 500)                                    # survey: 2 km, 4 m step
            except Exception as e: log(f"{tp}/{reg}: survey failed to download ({e})"); continue
            res = 4.0; cand = []; AR = None
            if "stream" in need: FR = TA.fill(Z); AR = TA.accumulate(FR, TA.receivers(FR, res), res)   # catchment — over the whole region
            area_at = lambda i, j: None if AR is None else float(AR[i - int(need["stream"]["r"] / res):i + int(need["stream"]["r"] / res) + 1,
                                                                     j - int(need["stream"]["r"] / res):j + int(need["stream"]["r"] / res) + 1].max())
            for i in range(50, 450, 25):
                for j in range(50, 450, 25):
                    W = Z[i - 50:i + 50, j - 50:j + 50]
                    s = summary(W - W.min(), res, {k: v for k, v in need.items() if k != "flat_max"}, zabs=W, area=area_at(i, j))
                    if ok(s, need, coarse=True): cand.append((score(s, cf.get("score", ["relief"])), i, j))
            cand.sort(reverse=True); chosen = []
            for sc_, i, j in cand:
                if all(math.hypot(i - a, j - b) * res >= 500 for _, a, b in chosen): chosen.append((sc_, i, j))
                if len(chosen) >= per * 2: break
            kept = 0
            for sc_, i, j in chosen:
                if kept >= per: break
                la, lo = lat + (i - 249.5) * res / 111320, lon + (j - 249.5) * res / (111320 * math.cos(math.radians(lat)))
                try: F = fetch(la, lo, 300.0, 600)
                except Exception as e: log(f"  window failed to download ({e})"); continue
                core = F[100:500, 100:500]; s = summary(core - core.min(), 1.0, need, zabs=core, area=area_at(i, j))
                nf = {k: ({**v, **{q: v[q] * 0.6 for q in ("drop", "rel") if q in v}} if k in ("saddle", "bowl", "summit") else v) for k, v in need.items()}
                if not ok(s, nf): continue                         # the shape is confirmed by the survey (4 m); at 1 m its drop looks different — threshold ×0.6
                name = f"{tp}_{reg}_{kept + 1}"; zmin = float(F.min()); dz = max(0.01, float(F.max() - zmin) / 65000)   # 1 cm; steep ones — coarser
                q = np.round((F - zmin) / dz).clip(0, 65535).astype(np.uint16)
                meta = {"type": tp, "region": reg, "lat": la, "lon": lo, "zmin": zmin, "z_step": dz, "step_m": 1.0, "size_m": 600,
                        "summary": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}}
                np.savez_compressed(os.path.join(LIB, tp, name + ".npz"), z=q, meta=json.dumps(meta, ensure_ascii=False))
                idx[name] = {"type": tp, "region": reg, "file": f"{tp}/{name}.npz", **{k: meta[k] for k in ("lat", "lon")}, "summary": meta["summary"]}
                kept += 1; json.dump(idx, open(idx_p, "w"), ensure_ascii=False, indent=1)
            log(f"{tp}/{reg}: candidates {len(cand)}, taken {kept}, {time.time() - t0:.0f} s")
    return idx


def prune(log=print):
    """Re-check of the collected windows against the current requirements (and the general ones: no more than 60 % nearly flat) —
    unsuitable ones are deleted."""
    T = config(); p = os.path.join(LIB, "index.json"); idx = json.load(open(p)); gone = []
    for name, v in list(idx.items()):
        need = T.get(v["type"], {}).get("need", {}); H, res, meta = load(name); core = H[100:500, 100:500]
        s = summary(core - core.min(), res, need, zabs=core + meta["zmin"], area=v["summary"].get("stream_area"))
        if not ok(s, need):
            os.remove(os.path.join(LIB, v["file"])); idx.pop(name); gone.append(name); log(f"removed {name}")
        else: idx[name]["summary"] = {k: (round(x, 3) if isinstance(x, float) else x) for k, x in s.items()}
    json.dump(idx, open(p, "w"), ensure_ascii=False, indent=1); return gone


def load(name):
    p = os.path.join(LIB, json.load(open(os.path.join(LIB, "index.json")))[name]["file"]); z = np.load(p)
    meta = json.loads(str(z["meta"])); return z["z"].astype(np.float64) * float(meta.get("z_step", 0.01)), float(meta.get("step_m", 1.0)), meta


def find(tp, seed=1, need=None):
    """A suitable window of the type (the seed chooses); need — extra requirements on the window summary (the same fields)."""
    idx = json.load(open(os.path.join(LIB, "index.json"))); names = sorted(n for n, v in idx.items() if v["type"] == tp and (not need or ok(v["summary"], need)))
    if not names: return None
    import zlib                                                     # seeds walk through the type's windows without repeats (each type has its own permutation)
    perm = np.random.default_rng(zlib.crc32(tp.encode())).permutation(len(names))
    return names[int(perm[(int(seed) - 1) % len(names)])]


def sheet(types=None, px=200):
    from PIL import Image, ImageDraw, ImageFont
    try: fnt = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 12)
    except Exception: fnt = ImageFont.load_default()
    p = os.path.join(LIB, "index.json"); idx = json.load(open(p)) if os.path.exists(p) else {}; T = config()
    for tp in T:
        if types and tp not in types: continue
        names = sorted(n for n, v in idx.items() if v["type"] == tp)
        if not names: continue
        C = min(6, len(names)); S = Image.new("RGB", (C * (px + 4), ((len(names) + C - 1) // C) * (px + 18) + 22), (24, 24, 24)); d = ImageDraw.Draw(S)
        d.text((4, 4), f"{tp}: {T[tp].get('title', '')} — {len(names)} windows of 600 m", fill=(255, 220, 90), font=fnt)
        for k, n in enumerate(names):
            H, res, meta = load(n); gy, gx = np.gradient(H, res); nn = np.dstack([-gx, -gy, np.ones_like(gx)]); nn /= np.linalg.norm(nn, axis=2, keepdims=True)
            l = np.array([-0.5, 0.5, 0.7]); l /= np.linalg.norm(l); sh = np.clip(nn @ l, 0, 1)
            col = np.dstack([sh * 225 + 20, sh * 215 + 25, sh * 195 + 30]); im = Image.fromarray(col[::-1].astype(np.uint8)).resize((px, px))
            x, y = (k % C) * (px + 4), 22 + (k // C) * (px + 18); S.paste(im, (x, y)); d.text((x + 2, y + px + 2), n[len(tp) + 1:][:30], fill=(220, 220, 220), font=fnt)
        S.save(os.path.join(LIB, f"{tp}.jpg"), quality=86)


def main():
    a = sys.argv[1:]; per = 4
    if "--per" in a: i = a.index("--per"); per = int(a[i + 1]); a = a[:i] + a[i + 2:]
    if not a: print(__doc__); return
    if a[0] == "build":
        idx = build(a[1:] or None, per, log=lambda m: print(m, flush=True)); sheet(a[1:] or None)
        print("__RESULT__" + json.dumps({"windows": len(idx), "by_type": {t: sum(1 for v in idx.values() if v["type"] == t) for t in config()}}, ensure_ascii=False))
    elif a[0] == "sheet": sheet(a[1:] or None); print("__RESULT__", LIB)
    elif a[0] == "prune": g = prune(); sheet(); print("__RESULT__" + json.dumps({"removed": g}, ensure_ascii=False))


if __name__ == "__main__":
    main()
