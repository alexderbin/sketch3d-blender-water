"""Measurement of the profiles of man-made shapes in lidar windows (lidar.py) → measures/works.json, profiles — measures/profiles.jpg.
- Ramparts (kind fort, town): crests are found by the transverse curvature (the Hessian of the smoothed height) and the rise
  above the local relief; across the crest — a profile: the drop to the bottom on both sides (outward — the larger one: that is
  where the moat is), a moat if there is a rise beyond the bottom (counterscarp), the width at half height, the face steepness,
  the crest width; the number of lines — by rays from the middle (multi-rampart) and the spacing between them.
- Sunken roads (lane): the same for hollows — the depth to the brows, the width at the brows and at the floor, the side steepness.
- Mounds (mound): rays from the summit — the height above the surroundings, the radius of the top and of the foot, the slope
  steepness and its spread between sides.
- Steps (field, pit): a profile along the dip — the ledge height, the shelf width, the slope of the shelf and of the ledge.
- Ridge-and-furrow arable land (field): the local relief spectrum in 96 m squares — the spacing and amplitude of the ridges.
- Spoil heap (heap): the slope steepness (angle of repose) and the heights above the foot.
In AHN, water (moats) and houses are holes: water is set to the level of the lowest bank around it (the moat depth is measured to the water).
.venv/bin/python -m components.site.towns.measures.measure [name …]"""
import json, math, os, sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from components.site.towns.measures import lidar as L

HERE = os.path.dirname(os.path.abspath(__file__))


def water(H, bad):
    """Holes → bank level: for each connected hole, the 5th percentile of heights in a 3-cell ring around it."""
    if not bad.any(): return H
    H = H.copy(); n, lab = cv2.connectedComponents(bad.astype(np.uint8))
    for i in range(1, n):
        m = lab == i; ring = cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~bad
        if ring.any(): H[m] = np.percentile(H[ring], 5)
    return H


def bil(H, res, x, y):
    """Height at points (m from the window corner; rows run from the south), bilinear."""
    fx, fy = np.clip(x / res, 0, H.shape[1] - 1.001), np.clip(y / res, 0, H.shape[0] - 1.001)
    ix, iy = fx.astype(int), fy.astype(int); tx, ty = fx - ix, fy - iy
    return (H[iy, ix] * (1 - tx) * (1 - ty) + H[iy, ix + 1] * tx * (1 - ty) + H[iy + 1, ix] * (1 - tx) * ty + H[iy + 1, ix + 1] * tx * ty)


def hessian(H, res, s_m):
    S = cv2.GaussianBlur(H, (0, 0), max(s_m / res, 0.5)); gy, gx = np.gradient(S, res)
    hyy, hyx = np.gradient(gy, res); hxy, hxx = np.gradient(gx, res); hxy = 0.5 * (hxy + hyx)
    tr, dt = 0.5 * (hxx + hyy), np.sqrt((0.5 * (hxx - hyy)) ** 2 + hxy ** 2)
    l1, l2 = tr - dt, tr + dt                                         # l1 is the smallest (crest: ≪ 0), l2 is the largest (hollow: ≫ 0)
    a1 = 0.5 * np.arctan2(2 * hxy, hxx - hyy) + np.pi / 2              # direction of the eigenvector l1 (across the crest)
    return S, l1, l2, a1


def spaced(mask, res, gap_m, nmax, rng):
    iy, ix = np.nonzero(mask)
    if not len(iy): return np.zeros((0, 2), int)
    o = rng.permutation(len(iy)); cell = max(1, int(gap_m / res)); seen = set(); out = []
    for j in o:
        k = (iy[j] // cell, ix[j] // cell)
        if k in seen: continue
        seen.add(k); out.append((iy[j], ix[j]))
        if len(out) >= nmax: break
    return np.array(out)


def profile(H, res, p, a, L_m, step=None):
    step = step or res / 2; s = np.arange(-L_m, L_m + 1e-9, step)
    x = p[1] * res + s * math.cos(a); y = p[0] * res + s * math.sin(a)
    return s, bil(H, res, x, y)


def _slope_max(s, z, i0, i1):
    if abs(i1 - i0) < 2: return 0.0
    a, b = sorted((i0, i1)); g = np.abs(np.gradient(z[a:b + 1], s[a:b + 1])); k = max(1, int(round(2.0 / (s[1] - s[0])))) | 1
    if len(g) > k: g = np.convolve(g, np.ones(k) / k, mode="same")
    return float(np.degrees(np.arctan(g.max())))


def bank(s, z, near=3.0, far=30.0):
    """A profile across a crest → rampart measurements (or None)."""
    st = s[1] - s[0]; c = len(s) // 2; w = int(near / st)
    i0 = c - w + int(np.argmax(z[c - w:c + w + 1])); z0 = z[i0]
    res = {}
    for side, rng_ in (("a", range(i0 - 1, max(0, i0 - int(far / st)) - 1, -1)), ("b", range(i0 + 1, min(len(s), i0 + int(far / st) + 1)))):
        idx = list(rng_)
        if not idx: return None
        zz = z[idx]; j = int(np.argmin(zz)); it = idx[j]; drop = z0 - z[it]
        beyond = idx[j:j + int(15 / st)]; rise = float(z[beyond].max() - z[it]) if len(beyond) > 1 else 0.0
        half = z0 - 0.5 * drop; jh = next((k for k in idx if z[k] < half), it)
        res[side] = {"drop": float(drop), "trough": it, "rise": rise, "half": abs(s[jh] - s[i0]), "dist": abs(s[it] - s[i0]),
                     "slope": _slope_max(s, z, i0, it)}
    o, i = (res["a"], res["b"]) if res["a"]["drop"] >= res["b"]["drop"] else (res["b"], res["a"])
    if i["drop"] < 0.3: return None
    top = z0 - 0.1 * i["drop"]; k = i0; j = i0
    while k > 0 and z[k - 1] > top: k -= 1
    while j < len(z) - 1 and z[j + 1] > top: j += 1
    return {"h_out": round(o["drop"], 2), "h_in": round(i["drop"], 2), "ditch": round(o["rise"], 2) if o["rise"] >= 0.3 else 0.0,
            "w_half": round(o["half"] + i["half"], 1), "to_ditch": round(o["dist"], 1), "slope_out": round(o["slope"], 1),
            "slope_in": round(i["slope"], 1), "crest_w": round((j - k) * st, 1)}


def lane(s, z, far=25.0):
    st = s[1] - s[0]; c = len(s) // 2; w = int(3 / st); i0 = c - w + int(np.argmin(z[c - w:c + w + 1])); zb = z[i0]
    sh = []
    for idx in (range(i0 - 1, max(0, i0 - int(far / st)) - 1, -1), range(i0 + 1, min(len(s), i0 + int(far / st) + 1))):
        idx = list(idx)
        if not idx: return None
        j = int(np.argmax(z[idx])); sh.append((idx[j], z[idx[j]]))
    d = min(sh[0][1], sh[1][1]) - zb
    if d < 0.8: return None
    lv = zb + 0.9 * d; bt = zb + 0.15 * d; k, j = i0, i0
    while k > 0 and z[k - 1] < lv: k -= 1
    while j < len(z) - 1 and z[j + 1] < lv: j += 1
    kb, jb = i0, i0
    while kb > 0 and z[kb - 1] < bt: kb -= 1
    while jb < len(z) - 1 and z[jb + 1] < bt: jb += 1
    return {"depth": round(float(d), 2), "w_top": round((j - k) * st, 1), "w_bottom": round((jb - kb) * st, 1),
            "slope": round(max(_slope_max(s, z, i0, sh[0][0]), _slope_max(s, z, i0, sh[1][0])), 1)}


def q(v, ps=(25, 50, 75, 90)):
    v = np.asarray([x for x in v if x is not None], float)
    return {f"p{p}": round(float(np.percentile(v, p)), 2) for p in ps} if len(v) else {}


def linear(H, res, sign, n=500, seed=1):
    """Crests (sign +1) or hollows (−1) over the whole window → cross profiles."""
    S, l1, l2, a1 = hessian(H, res, 1.5); LR = L.lrm(H, res, 15.0)
    if sign > 0: m = (l1 < -0.04) & (LR > 0.6)
    else: m = (l2 > 0.04) & (LR < -0.6)
    m &= np.pad(np.ones((H.shape[0] - 2 * int(40 / res), H.shape[1] - 2 * int(40 / res)), bool), int(40 / res))   # not at the window edge
    pts = spaced(m, res, 8.0, n, np.random.default_rng(seed)); out = []; prof = []
    for p in pts:
        a = a1[p[0], p[1]] - (0 if sign > 0 else np.pi / 2); s, z = profile(H, res, p, a, 40.0)   # across: for a hollow — the vector l2
        r = bank(s, z) if sign > 0 else lane(s, z)
        if r: out.append(r); prof.append((s, z, r))
    return out, prof


def radial_lines(H, res, c, r0, r1, n=48):
    """The number of crests (prominent ≥ 1 m above the neighboring lows) along rays from c and the spacing between them."""
    cnt, gaps = [], []
    for a in np.linspace(0, 2 * math.pi, n, endpoint=False):
        r = np.arange(r0, r1, res / 2); x = c[0] + r * math.cos(a); y = c[1] + r * math.sin(a)
        ok = (x > 0) & (y > 0) & (x < H.shape[1] * res - res) & (y < H.shape[0] * res - res)
        if ok.sum() < 20: continue
        z = bil(H, res, x[ok], y[ok]); r = r[ok]; z = z - np.convolve(np.pad(z, 60, mode="edge"), np.ones(121) / 121, mode="valid")
        pk = [i for i in range(4, len(z) - 4) if z[i] == z[i - 4:i + 5].max()]
        good = []
        for i in pk:
            lo_a = z[max(0, i - 40):i].min() if i > 0 else z[i]; lo_b = z[i + 1:i + 41].min() if i < len(z) - 1 else z[i]
            if z[i] - max(lo_a, lo_b) >= 1.0: good.append(r[i])
        cnt.append(len(good)); gaps += list(np.diff(good))
    return {"lines_per_ray": q(cnt, (25, 50, 75)), "step_m": q(gaps, (25, 50, 75))}


def mound(H, res):
    Hs = cv2.GaussianBlur(H, (0, 0), 10 / res); ny, nx = H.shape; m = np.zeros_like(Hs, bool); b = int(0.25 * nx); m[b:-b, b:-b] = True
    iy, ix = np.unravel_index(np.argmax(np.where(m, Hs, -1e9)), Hs.shape); c = (ix * res, iy * res); rmax = 0.48 * nx * res
    r = np.arange(0, rmax, res); P = []; smax = []
    for a in np.linspace(0, 2 * math.pi, 36, endpoint=False):
        z = bil(H, res, c[0] + r * math.cos(a), c[1] + r * math.sin(a)); P.append(z)
        zs = np.convolve(np.pad(z, 3, mode="edge"), np.ones(7) / 7, mode="valid"); smax.append(float(np.degrees(np.arctan(np.abs(np.diff(zs)).max() / res))))
    P = np.array(P); med = np.median(P, 0); zt = float(np.median(P[:, r < 8])); zb = float(np.median(med[r > 0.8 * rmax])); h = zt - zb
    rt = float(r[np.argmax(med < zt - 0.08 * h)]); rb = float(r[np.argmax(med < zb + 0.08 * h)])
    g = np.degrees(np.arctan(np.abs(np.gradient(np.convolve(np.pad(med, 5, mode="edge"), np.ones(11) / 11, mode="valid"), res))))
    return {"h": round(h, 2), "r_top": round(rt, 1), "r_base": round(rb, 1), "slope_max_median": round(float(g.max()), 1),
            "slope_max_by_side": q(smax, (10, 50, 90)), "center_m": [round(c[0] - nx * res / 2), round(c[1] - ny * res / 2)]}, (r, med)


def steps(H, res, n=160, L_m=60.0, seed=2, min_slope=5.0):
    S = cv2.GaussianBlur(H, (0, 0), 10 / res); gy, gx = np.gradient(S, res); sl = np.degrees(np.arctan(np.hypot(gx, gy)))
    m = sl > min_slope; e = int(L_m / res) + 2; m[:e] = m[-e:] = False; m[:, :e] = m[:, -e:] = False
    pts = spaced(m, res, L_m / 3, n, np.random.default_rng(seed)); R = []; prof = []
    for p in pts:
        a = math.atan2(gy[p[0], p[1]], gx[p[0], p[1]]); s, z = profile(H, res, p, a, L_m, res)
        k = max(1, int(round(2.0 / res))) | 1; zs = np.convolve(np.pad(z, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
        g = np.degrees(np.arctan(np.gradient(zs, s))); mean = float(np.degrees(np.arctan((z[-1] - z[0]) / (s[-1] - s[0]))))
        steep = (g > max(mean * 1.6, mean + 6)) & (g > 12)
        e_ = np.diff(np.r_[0, steep.astype(int), 0]); a0, a1 = np.nonzero(e_ == 1)[0], np.nonzero(e_ == -1)[0] - 1
        ris = [(i, j) for i, j in zip(a0, a1) if (s[j] - s[i]) >= res]
        if len(ris) < 2: continue
        hs = [float(z[min(j + 1, len(z) - 1)] - z[max(i - 1, 0)]) for i, j in ris]; mids = [0.5 * (s[i] + s[j]) for i, j in ris]
        tr = []
        for (i, j), (i2, j2) in zip(ris[:-1], ris[1:]):
            if i2 - j > 2: tr.append(float(np.degrees(np.arctan((z[i2] - z[j]) / (s[i2] - s[j])))))
        R.append({"riser_m": round(float(np.median(hs)), 2), "tread_m": round(float(np.median(np.diff(mids))), 1),
                  "tread_slope": round(float(np.median(tr)), 1) if tr else None, "riser_deg": round(float(np.median([g[i:j + 1].max() for i, j in ris])), 1),
                  "slope_deg": round(mean, 1)})
        prof.append((s, z, R[-1]))
    return R, prof


def furrows(H, res, box=96.0):
    LR = L.lrm(H, res, 5.0); k = int(box / res); out = []
    win = np.outer(np.hanning(k), np.hanning(k))
    for y0 in range(0, H.shape[0] - k, k):
        for x0 in range(0, H.shape[1] - k, k):
            t = LR[y0:y0 + k, x0:x0 + k]; P = np.abs(np.fft.fftshift(np.fft.fft2((t - t.mean()) * win))) ** 2
            fy, fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(k, res)), np.fft.fftshift(np.fft.fftfreq(k, res)), indexing="ij")
            fr = np.hypot(fx, fy); band = (fr > 1 / 25.0) & (fr < 1 / 4.0)
            if not band.any(): continue
            j = np.argmax(np.where(band, P, 0)); pk = P.flat[j]; bg = np.median(P[band])
            if pk / max(bg, 1e-12) < 60: continue
            out.append({"step_m": round(1 / fr.flat[j], 1), "amp_m": round(float(2 * math.sqrt(2) * t.std()), 2)})
    return out


def heap(H, res):
    gy, gx = np.gradient(cv2.GaussianBlur(H, (0, 0), 2 / res), res); sl = np.degrees(np.arctan(np.hypot(gx, gy)))
    return {"slope": q(sl[sl > 10].ravel()[::7], (50, 75, 90, 95, 99)), "drop": round(float(np.percentile(H, 99) - np.percentile(H, 1)), 1)}


def run(name):
    H, res, meta = L.load(name); H = water(H, meta["bad"]); k = meta["kind"]; out = {"ru": meta["ru"], "kind": k, "step": res}; prof = []
    if k in ("fort", "town"):
        R, prof = linear(H, res, +1); out["bank_profiles"] = len(R)
        for f in ("h_out", "h_in", "ditch", "w_half", "to_ditch", "slope_out", "slope_in", "crest_w"): out[f] = q([r[f] for r in R])
        big = [r for r in R if r["h_out"] >= np.percentile([x["h_out"] for x in R], 75)] if R else []
        out["large"] = {f: q([r[f] for r in big], (50,)).get("p50") for f in ("h_out", "h_in", "ditch", "w_half", "slope_out", "slope_in", "crest_w")}
        S, l1, l2, a1 = hessian(H, res, 1.5); LR = L.lrm(H, res, 15.0); m = (l1 < -0.04) & (LR > 1.0); iy, ix = np.nonzero(m)
        if len(iy):
            c = (float(np.average(ix, weights=LR[iy, ix])) * res, float(np.average(iy, weights=LR[iy, ix])) * res)
            out.update(radial_lines(H, res, c, 20.0, 0.48 * H.shape[1] * res)); out["center_m"] = [round(c[0] - H.shape[1] * res / 2), round(c[1] - H.shape[0] * res / 2)]
    elif k == "lane":
        R, prof = linear(H, res, -1); out["profiles"] = len(R)
        for f in ("depth", "w_top", "w_bottom", "slope"): out[f] = q([r[f] for r in R])
    elif k == "mound":
        out.update(mound(H, res)[0])
    elif k in ("field", "pit"):
        R, prof = steps(H, res, L_m=60.0 if k == "field" else 400.0, min_slope=5.0 if k == "field" else 15.0); out["profiles"] = len(R)
        for f in ("riser_m", "tread_m", "tread_slope", "riser_deg", "slope_deg"): out[f] = q([r[f] for r in R])
        if k == "field":
            F = furrows(H, res); out["furrows"] = {"squares": len(F), "step_m": q([f["step_m"] for f in F], (25, 50, 75)), "amp_m": q([f["amp_m"] for f in F], (25, 50, 75))}
    elif k == "heap":
        out.update(heap(H, res))
    return out, prof, (H, res)


def draw(rows, path, W=1700, h=150):
    f = ImageFont.truetype(L.FONT, 14); im = Image.new("RGB", (W, h * len(rows)), (18, 18, 18)); d = ImageDraw.Draw(im)
    for i, (name, out, prof) in enumerate(rows):
        y0 = i * h; d.text((8, y0 + 4), f"{name} — {out['ru']}", fill=(230, 230, 230), font=f)
        if not prof: continue
        pick = sorted(prof, key=lambda p: -(p[2].get("h_out") or p[2].get("depth") or p[2].get("riser_m") or 0))
        pick = [pick[len(pick) // 10], pick[len(pick) // 3], pick[len(pick) // 2]] if len(pick) >= 3 else pick
        for j, (s, z, r) in enumerate(pick):
            x0 = 10 + j * (W // 3); ww = W // 3 - 30; zz = z - z.min(); span = max(zz.max(), 1.0); sc = min((h - 40) / span, ww / (s[-1] - s[0]))
            pts = [(x0 + (s[k] - s[0]) * sc, y0 + h - 10 - zz[k] * sc) for k in range(len(s))]
            d.line(pts, fill=(120, 200, 255), width=2)
            d.text((x0, y0 + 22), " ".join(f"{k_}={v}" for k_, v in r.items())[:110], fill=(170, 170, 170), font=f)
    im.save(path, quality=88)


if __name__ == "__main__":
    S = L.sites(); names = sys.argv[1:] or [n for n in S if os.path.exists(os.path.join(L.LIB, n + ".npz"))]
    jp = os.path.join(HERE, "works.json"); allr = json.load(open(jp)) if os.path.exists(jp) else {}; rows = []
    for n in names:
        out, prof, _ = run(n); allr[n] = out; rows.append((n, out, prof)); print(n, json.dumps(out, ensure_ascii=False)[:400], flush=True)
    json.dump(allr, open(jp, "w"), ensure_ascii=False, indent=1)
    draw([r for r in rows if r[2]], os.path.join(HERE, "profiles.jpg"))
