"""Measurement of real man-made shapes from lidar: hillforts, town ramparts, terps, lynchets, sunken roads, quarries and spoil
heaps. The windows come from open relief maps without buildings and forest (sites.yaml: England EA 1 m, Netherlands AHN 0.5 m,
USA 3DEP 1 m); they are stored as the Site's real bases (uint16 heights at 1 cm + meta) — library/terrain/real/works/.
We look as archaeologists do: hillshade + a local relief model (LRM, Hesse 2010: height minus the smoothed one — ramparts and
moats are visible on any slope).
.venv/bin/python -m components.site.towns.measures.lidar fetch [name …]   → windows, index.json (≈ 1–6 MB download per window)
.venv/bin/python -m components.site.towns.measures.lidar sheet [name …]   → measures/works.jpg
In code: load(name) → (H, m; step, m; meta), rows run from south to north, x is east."""
import core.paths
import io, json, math, os, sys, time, urllib.request

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(core.paths.ROOT, "library", "terrain", "real", "works")
EA = ("https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m-2022/wcs?service=WCS&version=2.0.1"
      "&request=GetCoverage&CoverageId=13787b9a-26a4-4775-8523-806d13af58fc__Lidar_Composite_Elevation_DTM_1m&format=image/tiff")
AHN = "https://service.pdok.nl/rws/ahn/wcs/v1_0?service=WCS&version=2.0.1&request=GetCoverage&CoverageId=dtm_05m&format=image/tiff"
LIC = {"ea": "Environment Agency LIDAR Composite DTM 2022 1 m, Open Government Licence v3",
       "ahn": "AHN4 dtm_05m (PDOK / Rijkswaterstaat), CC0", "3dep": "USGS 3DEP, public domain"}
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def sites():
    return yaml.safe_load(open(os.path.join(HERE, "sites.yaml")))["sites"]


def _get(url, tries=3):
    for k in range(tries):
        try: return urllib.request.urlopen(url, timeout=180).read()
        except Exception:
            if k == tries - 1: raise
            time.sleep(4 * (k + 1))


def _wcs(base, crs, axes, lat, lon, half):
    from pyproj import Transformer
    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    q = f"&subset={axes[0]}({x - half:.1f},{x + half:.1f})&subset={axes[1]}({y - half:.1f},{y + half:.1f})"
    Z = np.array(Image.open(io.BytesIO(_get(base + q))), np.float64)[::-1].copy()   # TIFF rows run from the north: flip
    return Z


def fill(Z, bad=None):
    """No data (houses, water in AHN; the edge in EA) → holes: a mask and filling by normalized convolution from σ 1 m upward — the
    values stay within the range of their neighbors (Telea inpainting on a float grid gave −48…+60 m). Moat water uses the same mask."""
    bad = (~np.isfinite(Z) | (np.abs(Z) > 1e4)) if bad is None else bad
    if not bad.any(): return Z, bad
    M = (~bad).astype(np.float32); Z0 = np.where(bad, 0, Z).astype(np.float32); F = np.where(bad, np.nan, Z); todo = bad.copy(); s = 1.0
    while todo.any() and s < 1024:
        den = cv2.GaussianBlur(M, (0, 0), s); num = cv2.GaussianBlur(Z0, (0, 0), s)
        ok = todo & (den > 0.05); F[ok] = num[ok] / den[ok]; todo &= ~ok; s *= 2
    F[todo] = float(np.nanmedian(F)); return F.astype(np.float64), bad


def fetch(name, s):
    t0 = time.time(); half = float(s["half"])
    if s["src"] == "ea": Z = _wcs(EA, 27700, "EN", s["lat"], s["lon"], half); res = 1.0
    elif s["src"] == "ahn": Z = _wcs(AHN, 28992, "xy", s["lat"], s["lon"], half); res = 0.5
    else:
        from components.site import terrain_library as TL
        res = float(s.get("res", 1.0)); Z = TL.fetch(s["lat"], s["lon"], half, int(round(2 * half / res)))
    if s.get("res") and s["src"] != "3dep" and float(s["res"]) != res:
        k = float(s["res"]) / res; Z = cv2.resize(Z, (int(Z.shape[1] / k), int(Z.shape[0] / k)), interpolation=cv2.INTER_AREA); res = float(s["res"])
    Z, bad = fill(Z)
    zmin = float(Z.min()); q = np.round((Z - zmin) / 0.01).clip(0, 65535).astype(np.uint16)
    meta = {"name": name, "ru": s["ru"], "src": s["src"], "license": LIC[s["src"]], "lat": s["lat"], "lon": s["lon"], "step_m": res,
            "z_step": 0.01, "zmin": round(zmin, 3), "kind": s["kind"], "no_data": round(float(bad.mean()), 4)}
    os.makedirs(LIB, exist_ok=True)
    np.savez_compressed(os.path.join(LIB, name + ".npz"), z=q, nodata=np.packbits(bad), shape=np.array(bad.shape), meta=json.dumps(meta, ensure_ascii=False))
    ip = os.path.join(LIB, "index.json"); idx = json.load(open(ip)) if os.path.exists(ip) else {}
    idx[name] = {k: meta[k] for k in ("ru", "src", "kind", "lat", "lon", "step_m", "license")}; idx[name]["file"] = name + ".npz"
    json.dump(idx, open(ip, "w"), ensure_ascii=False, indent=1)
    return {"name": name, "shape": list(Z.shape), "step": res, "drop": round(float(np.ptp(Z)), 1), "holes": meta["no_data"], "s": round(time.time() - t0, 1)}


def load(name):
    z = np.load(os.path.join(LIB, name + ".npz")); meta = json.loads(str(z["meta"]))
    bad = np.unpackbits(z["nodata"])[:int(np.prod(z["shape"]))].reshape(z["shape"]).astype(bool)
    return z["z"].astype(np.float64) * meta["z_step"] + meta["zmin"], float(meta["step_m"]), {**meta, "bad": bad}


def shade(H, res, az=(315, 45, 135, 225), alt=40):
    """Hillshade from four directions (multidirectional — ramparts of any orientation are visible)."""
    gy, gx = np.gradient(H, res); sl = np.arctan(np.hypot(gx, gy)); asp = np.arctan2(-gx, gy); out = 0
    for a in az:
        A, h = math.radians(a), math.radians(alt)
        out = out + np.clip(math.sin(h) * np.cos(sl) + math.cos(h) * np.sin(sl) * np.cos(A - asp), 0, 1)
    return out / len(az)


def lrm(H, res, sigma_m=15.0):
    """Local relief model (Hesse 2010): height minus the height smoothed with σ — a rampart (+), a moat (−)."""
    return H - cv2.GaussianBlur(H, (0, 0), sigma_m / res)


def tile(name, px=520):
    H, res, meta = load(name)
    S = shade(H, res); L = np.clip(lrm(H, res) / 2.0, -1, 1)              # ±2 m
    rgb = np.stack([S, S, S], -1) * 0.85 + 0.1
    red = np.array([0.85, 0.25, 0.15]); blue = np.array([0.15, 0.35, 0.85])
    wpos, wneg = np.clip(L, 0, 1)[..., None], np.clip(-L, 0, 1)[..., None]
    rgb = rgb * (1 - 0.6 * (wpos + wneg)) + 0.6 * (wpos * red + wneg * blue) * (0.4 + 0.6 * rgb)
    rgb[meta["bad"]] = rgb[meta["bad"]] * 0.5 + np.array([0.2, 0.5, 0.7]) * 0.5   # holes (water, houses) — in light blue
    im = Image.fromarray((np.clip(rgb, 0, 1)[::-1] * 255).astype(np.uint8)).resize((px, px), Image.LANCZOS)
    d = ImageDraw.Draw(im); f = ImageFont.truetype(FONT, 15)
    m_px = px / (H.shape[1] * res); bar = 100 if H.shape[1] * res < 1500 else 500
    d.rectangle([10, px - 22, 10 + bar * m_px, px - 16], fill=(255, 255, 255)); d.text((14 + bar * m_px, px - 28), f"{bar} m", fill=(255, 255, 255), font=f)
    return im, meta, H


def sheet(names, cols=4, px=520):
    f = ImageFont.truetype(FONT, 16); cells = []
    for n in names:
        im, meta, H = tile(n, px)
        cap = Image.new("RGB", (px, 46), (24, 24, 24)); d = ImageDraw.Draw(cap)
        d.text((6, 3), meta["ru"][:62], fill=(235, 235, 235), font=f)
        d.text((6, 24), f"{n} · {meta['src']} {meta['step_m']:g} m · window {H.shape[1] * meta['step_m']:.0f} m · drop {np.ptp(H):.0f} m", fill=(170, 170, 170), font=f)
        c = Image.new("RGB", (px, px + 46)); c.paste(cap, (0, 0)); c.paste(im, (0, 46)); cells.append(c)
    rows = math.ceil(len(cells) / cols); W = Image.new("RGB", (cols * px + (cols - 1) * 6, rows * (px + 46) + (rows - 1) * 6), (10, 10, 10))
    for i, c in enumerate(cells): W.paste(c, ((i % cols) * (px + 6), (i // cols) * (px + 46 + 6)))
    p = os.path.join(HERE, "works.jpg"); W.save(p, quality=88); return p


if __name__ == "__main__":
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    S = sites(); names = args or list(S)
    if cmd == "fetch":
        for n in names:
            try: print(json.dumps(fetch(n, S[n]), ensure_ascii=False), flush=True)
            except Exception as e: print(json.dumps({"name": n, "error": str(e)[:200]}, ensure_ascii=False), flush=True)
    elif cmd == "refill":                                           # refill the holes of the saved windows (no download)
        for n in names:
            H, res, m = load(n)
            if not m["bad"].any(): continue
            Z, bad = fill(np.where(m["bad"], np.nan, H), m["bad"]); zmin = float(Z.min()); m.pop("bad"); m["zmin"] = round(zmin, 3)
            np.savez_compressed(os.path.join(LIB, n + ".npz"), z=np.round((Z - zmin) / 0.01).clip(0, 65535).astype(np.uint16), nodata=np.packbits(bad),
                                shape=np.array(bad.shape), meta=json.dumps(m, ensure_ascii=False))
            print(n, "drop", round(float(np.ptp(Z)), 1), flush=True)
    elif cmd == "sheet": print("__RESULT__" + json.dumps({"sheet": sheet([n for n in names if os.path.exists(os.path.join(LIB, n + ".npz"))])}, ensure_ascii=False))
    else: raise SystemExit(__doc__)
