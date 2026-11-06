"""Foliage cards: an RGBA image of one organ by the species' leaf type (project Python: numpy + PIL; Blender has no PIL — the images
are written in advance and cached by key). The type comes from the species knowledge (species.yaml → leaf.type): entire leaf, serrate, lobed (oak),
palmate (maple), lanceolate (willow), needle bundle (pine), needle twig (spruce, fir), scale (cypress, juniper), grass, petal.
A leaf is drawn with the petiole down at the middle of the bottom edge: in Blender the card is placed with its bottom toward the branch, its long side along the leaf.
.venv/bin/python -m components.plants.leaves <folder> [type …] → <folder>/<type>.png + __RESULT__"""
import json, math, os, sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

N = 256                                           # side of the organ image; a leaf card is a square, the leaf is fitted with margins


def _noise(n, scale, seed):
    """Smooth n × n noise: a sum of sines with random phases (no external libraries), 0…1."""
    r = np.random.default_rng(seed); y, x = np.mgrid[0:n, 0:n] / n; v = np.zeros((n, n))
    for k in range(6):
        f = scale * (1.7 ** k); a, b, p = r.normal(size=3)
        v += math.pow(0.6, k) * np.sin(2 * math.pi * f * (a * x + b * y) + 6.28 * p)
    v -= v.min(); return v / max(v.max(), 1e-9)


def _outline(kind, t):
    """Half-width of the leaf (a fraction of its length) by t ∈ [0,1] from the petiole to the tip."""
    if kind in ("ovate", "serrate", "birch"):  return 0.36 * np.maximum(np.sin(np.pi * np.clip(t, 0, 1) ** 0.8), 0) ** 0.9
    if kind == "lanceolate":                   return 0.12 * np.sin(np.pi * np.clip(t, 0, 1) ** 0.7)
    if kind == "lobed":                        # oak: lobes along the edge, deep notches
        base = 0.33 * np.maximum(np.sin(np.pi * np.clip(t, 0, 1) ** 0.9), 0) ** 0.8
        return base * (0.62 + 0.38 * np.abs(np.cos(np.pi * 4.5 * t)) ** 0.6)
    return 0.3 * np.sin(np.pi * np.clip(t, 0, 1))


def broad(kind="ovate", seed=1, color=(0.22, 0.36, 0.12)):
    """A leaf: an outline by type, petiole, midrib and lateral veins, colour patches; alpha — the outline. The palette is in linear ≈ sRGB."""
    img = np.zeros((N, N, 4), np.float32); y, x = np.mgrid[0:N, 0:N].astype(np.float32)
    pet = 0.14; L0, L1 = N * 0.97, N * 0.04                      # the petiole at the bottom (y is large) — the tip at the top
    t = (L0 - y) / (L0 - L1); u = (x - N / 2) / (L0 - L1)        # t — along the leaf, u — across (in fractions of the length)
    tl = (t - pet) / (1 - pet)
    if kind == "palmate":                                        # maple: five lobes from one point
        cx, cy = N / 2, L0 - pet * (L0 - L1); ang = np.arctan2(-(y - cy), x - cx); rr = np.hypot(x - cx, y - cy) / (L0 - L1)
        lob = np.zeros_like(rr)
        for a, ln in ((90, 0.80), (40, 0.70), (140, 0.70), (-5, 0.42), (185, 0.42)):
            d = np.angle(np.exp(1j * (ang - math.radians(a))))
            lob = np.maximum(lob, ln * np.clip(1 - np.abs(d) / 0.62, 0, 1) ** 0.35)
        lob *= 1 - 0.07 * np.abs(np.sin(ang * 9))                 # teeth along the lobe edges
        inside = rr < lob * (1 + 0.04 * np.sin(rr * 90))
    else:
        w = _outline(kind, tl)
        if kind in ("serrate", "birch"):                         # small teeth along the edge
            w = w * (1 - 0.06 * (np.abs(np.sin(np.pi * 22 * tl)) ** 2))
        inside = (tl > 0) & (tl < 1) & (np.abs(u) < w)
    stem = (t >= -0.02) & (t <= pet + 0.02) & (np.abs(u) < 0.012)
    vein = (tl > 0) & (tl < 0.97) & (np.abs(u) < 0.008 + 0.006 * (1 - tl))
    side = np.zeros_like(t, bool)
    for k in range(1, 8):                                        # lateral veins in a herringbone pattern
        t0 = k / 8.5; side |= (np.abs(tl - t0 - np.abs(u) * 0.9) < 0.006) & (np.abs(u) > 0.01) & inside
    n1 = _noise(N, 3, seed); n2 = _noise(N, 11, seed + 5)
    base = np.array(color, np.float32)
    col = base[None, None, :] * (0.78 + 0.32 * n1[..., None]) * (0.92 + 0.12 * n2[..., None])
    col = col * (1 + 0.25 * (np.abs(u)[..., None] / 0.36))       # lighter toward the edge (thinner)
    col = np.where(vein[..., None] | side[..., None], col * 1.35 + 0.02, col)
    col = np.where(stem[..., None], np.array([0.20, 0.17, 0.08]), col)
    a = (inside | stem).astype(np.float32)
    img[..., :3] = np.clip(col, 0, 1); img[..., 3] = a
    return img


def needles(kind="pine", seed=1, color=(0.12, 0.22, 0.10)):
    """Needles on a twig: the shoot runs up the centre from bottom to top, needles on both sides (pine — long bundles, spruce/fir — short ones in a comb
    pattern). The image is placed as a card along the shoot."""
    r = np.random.default_rng(seed); im = Image.new("RGBA", (N, N), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    c = lambda k: tuple(int(255 * min(1, v * k)) for v in color) + (255,)
    x0 = N / 2; d.line([(x0, N - 4), (x0 + r.uniform(-6, 6), 8)], fill=(70, 55, 30, 255), width=4)
    n, ln, ang = (90, 0.30, 35) if kind == "pine" else (220, 0.16, 62) if kind == "spruce" else (200, 0.17, 80)
    for i in range(n):
        tt = r.uniform(0.02, 0.98); yy = N - 4 - tt * (N - 12); s = r.choice([-1, 1])
        a = math.radians(ang + r.uniform(-15, 15)); L = N * ln * r.uniform(0.75, 1.15) * (1 - 0.35 * tt if kind != "pine" else 1)
        x1 = x0 + s * L * math.sin(a); y1 = yy - L * math.cos(a)
        d.line([(x0, yy), (x1, y1)], fill=c(r.uniform(0.7, 1.4)), width=2 if kind == "pine" else 3)
    im = im.filter(ImageFilter.SMOOTH)
    a = np.asarray(im).astype(np.float32) / 255; return a


def scales(seed=1, color=(0.14, 0.24, 0.10)):
    """Scale leaves (cypress, thuja, juniper): a branching flat twig of small scales."""
    r = np.random.default_rng(seed); im = Image.new("RGBA", (N, N), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    def br(x, y, a, L, w, k):
        if L < 8 or k > 4: return
        x1, y1 = x + L * math.sin(a), y - L * math.cos(a)
        d.line([(x, y), (x1, y1)], fill=tuple(int(255 * v * r.uniform(0.8, 1.3)) for v in color) + (255,), width=max(2, int(w)))
        for j in range(3):
            tt = 0.35 + 0.25 * j; xs, ys = x + (x1 - x) * tt, y + (y1 - y) * tt
            br(xs, ys, a + (0.7 if j % 2 else -0.7) + r.uniform(-0.2, 0.2), L * 0.5, w * 0.75, k + 1)
        br(x1, y1, a + r.uniform(-0.2, 0.2), L * 0.55, w * 0.8, k + 1)
    br(N / 2, N - 4, 0, N * 0.45, 9, 0)
    return np.asarray(im.filter(ImageFilter.SMOOTH)).astype(np.float32) / 255


def cluster(kind, seed=1, n=7):
    """Cluster: a twig with n leaves scattered about (like scan clusters) — one card instead of n; the leaf is the image broad()."""
    leaf = Image.fromarray((np.clip(TYPES[kind](seed), 0, 1) * 255).astype(np.uint8), "RGBA"); r = np.random.default_rng(seed)
    M = N * 2; im = Image.new("RGBA", (M, M), (0, 0, 0, 0)); d = ImageDraw.Draw(im); x0, y0, x1, y1 = M / 2, M - 6, M / 2 + r.uniform(-20, 20), M * 0.12
    d.line([(x0, y0), (x1, y1)], fill=(72, 58, 34, 255), width=5)
    for i in range(n):
        t = 0.15 + 0.85 * i / max(n - 1, 1); x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t; s = 1 if i % 2 else -1
        a = s * r.uniform(35, 65) if i < n - 1 else r.uniform(-10, 10); sz = int(M * r.uniform(0.34, 0.44) * (0.75 + 0.25 * t))
        lf = leaf.resize((sz, sz), Image.LANCZOS).rotate(-a, expand=True, resample=Image.BICUBIC)
        k = math.radians(a); bx, by = x + math.sin(k) * 0, y                                            # the petiole is at a point on the twig
        im.alpha_composite(lf, (int(bx - lf.width / 2 + math.sin(k) * sz * 0.45), int(by - lf.height / 2 - math.cos(k) * sz * 0.45)))
    return np.asarray(im).astype(np.float32) / 255


def blade(seed=1):
    """A grass blade — a white ribbon with a vein and a light tip (the colour is per vertex in Blender); alpha — a pointed tip."""
    y, x = np.mgrid[0:N, 0:N].astype(np.float32); t = 1 - y / N; u = np.abs(x - N / 2) / (N / 2)
    w = np.clip(1 - t, 0, 1) ** 0.35; a = (u < w).astype(np.float32); n1 = _noise(N, 6, seed)
    v = 0.82 + 0.18 * n1 - 0.12 * (u < 0.06) + 0.1 * t
    img = np.zeros((N, N, 4), np.float32); img[..., 0] = v; img[..., 1] = v; img[..., 2] = v * 0.95; img[..., 3] = a; return img


def flower(kind, seed=1):
    """A flower from above (a card facing up or toward the camera): daisy, buttercup, poppy, cornflower, clover, umbel, lupine spike, plantain spike."""
    r = np.random.default_rng(seed); im = Image.new("RGBA", (N, N), (0, 0, 0, 0)); d = ImageDraw.Draw(im); c = N / 2
    P = {"daisy": ((245, 245, 238), (225, 180, 30), 18, 0.46, 0.12), "buttercup": ((240, 200, 20), (200, 150, 10), 5, 0.42, 0.10),
         "poppy": ((205, 30, 20), (30, 25, 20), 4, 0.46, 0.10), "cornflower": ((60, 90, 205), (40, 50, 120), 9, 0.44, 0.08),
         "campanula": ((120, 90, 200), (90, 70, 160), 5, 0.40, 0.08)}
    if kind in P:
        pc, cc, n, R, rc = P[kind]
        for i in range(n):
            a = 2 * math.pi * i / n + r.uniform(-0.1, 0.1); L = N * R * r.uniform(0.9, 1.05); w = N * (0.08 if n > 12 else 0.2 if n < 6 else 0.13)
            pts = [(c + math.cos(a + o) * w * (1 if j else 0.3) + math.cos(a) * L * j, c + math.sin(a + o) * w * (1 if j else 0.3) + math.sin(a) * L * j)
                   for j, o in ((0, -1.4), (0.55, -1.0), (1, 0), (0.55, 1.0), (0, 1.4))]
            kk = r.uniform(0.9, 1.05); col = tuple(min(255, int(v * kk)) for v in pc) + (255,); d.polygon(pts, fill=col)
        d.ellipse([c - N * rc, c - N * rc, c + N * rc, c + N * rc], fill=cc + (255,))
    elif kind == "clover":
        for i in range(60):
            a = r.uniform(0, 2 * math.pi); rr = N * 0.36 * math.sqrt(r.uniform(0, 1)); x, y = c + rr * math.cos(a), c + rr * math.sin(a)
            d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(int(220 * r.uniform(0.85, 1)), int(120 * r.uniform(0.8, 1.1)), 170, 255))
    elif kind == "umbel":                                     # umbel from above: umbellets all around the circle, denser toward the edge
        for i in range(26):
            a = r.uniform(0, 2 * math.pi); rr = N * 0.40 * math.sqrt(r.uniform(0.05, 1)); x, y = c + rr * math.cos(a), c + rr * math.sin(a)
            for j in range(10):
                b = r.uniform(0, 2 * math.pi); q = r.uniform(0, 16); d.ellipse([x + q * math.cos(b) - 4, y + q * math.sin(b) - 4, x + q * math.cos(b) + 4, y + q * math.sin(b) + 4], fill=(238, 238, 226, 255))
    elif kind == "plantain":                                   # plantain spike: a thin brown cylinder, stamens as light dots
        for i in range(70):
            t = i / 70; y = N - 8 - t * (N - 16); w = N * 0.05
            for s in (-1, 0, 1):
                if r.random() < 0.7: d.ellipse([c + s * w - 4, y - 4, c + s * w + 4, y + 4], fill=(int(96 * r.uniform(0.8, 1.15)), int(84 * r.uniform(0.8, 1.15)), int(52 * r.uniform(0.8, 1.1)), 255))
            if r.random() < 0.25: xx = c + r.uniform(-1.6, 1.6) * w; d.ellipse([xx - 2, y - 2, xx + 2, y + 2], fill=(214, 206, 170, 255))
    elif kind == "spike":                                      # lupine spike: a side view, the flower cluster narrows toward the top
        for i in range(40):
            t = i / 40; y = N - 10 - t * (N - 20); w = N * 0.16 * (1 - 0.7 * t)
            for s in (-1, 1):
                if r.random() < 0.8: d.ellipse([c + s * w * 0.5 - 7, y - 5, c + s * w * 0.5 + 7, y + 5], fill=(int(140 * r.uniform(0.8, 1.1)), int(100 * r.uniform(0.8, 1.1)), int(175 * r.uniform(0.85, 1.05)), 255))
    return np.asarray(im.filter(ImageFilter.SMOOTH)).astype(np.float32) / 255


TYPES = {"ovate": lambda s: broad("ovate", s), "serrate": lambda s: broad("serrate", s, (0.20, 0.34, 0.11)),
         "birch": lambda s: broad("birch", s, (0.25, 0.40, 0.12)), "lanceolate": lambda s: broad("lanceolate", s, (0.24, 0.36, 0.16)),
         "lobed": lambda s: broad("lobed", s, (0.19, 0.30, 0.10)), "palmate": lambda s: broad("palmate", s, (0.22, 0.34, 0.10)),
         "pine": lambda s: needles("pine", s), "spruce": lambda s: needles("spruce", s, (0.09, 0.18, 0.10)),
         "fir": lambda s: needles("fir", s, (0.10, 0.20, 0.11)), "scale": lambda s: scales(s), "blade": blade}
for _k in ("daisy", "buttercup", "poppy", "cornflower", "campanula", "clover", "umbel", "spike", "plantain"):
    TYPES["flower_" + _k] = (lambda k: lambda s: flower(k, s))(_k)
for _k in ("ovate", "serrate", "birch", "lanceolate", "lobed", "palmate"):
    TYPES["cluster_" + _k] = (lambda k: lambda s: cluster(k, s))(_k)


def cutout(rgba, max_pts=10):
    """Card cut-out: alpha → convex polygons of the organs in UV [0,1] (v up — from the petiole).
    Cluster: the thin twig is removed by morphological opening, each leaf gets its own polygon; needles and scales — one shoot outline."""
    import cv2
    A = (np.asarray(rgba)[..., 3] > 0.5).astype(np.uint8); H, W = A.shape; polys = []
    k = max(3, W // 64) | 1; B = cv2.morphologyEx(A, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(B if B.sum() > 0.25 * A.sum() else A)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < (W * H) * 0.002: continue
        pts = np.column_stack(np.nonzero(lab == i))[:, ::-1].astype(np.int32); hull = cv2.convexHull(pts)
        eps = 0.01 * cv2.arcLength(hull, True); ap = cv2.approxPolyDP(hull, eps, True)[:, 0, :]
        while len(ap) > max_pts: eps *= 1.4; ap = cv2.approxPolyDP(hull, eps, True)[:, 0, :]
        polys.append([[round(float(x) / W, 4), round(1 - float(y) / H, 4)] for x, y in ap])
    if not polys:                                                    # thin (needles): the outline of the whole shoot
        pts = np.column_stack(np.nonzero(A))[:, ::-1].astype(np.int32); hull = cv2.convexHull(pts)
        ap = cv2.approxPolyDP(hull, 0.01 * cv2.arcLength(hull, True), True)[:, 0, :][:max_pts]
        polys.append([[round(float(x) / W, 4), round(1 - float(y) / H, 4)] for x, y in ap])
    return polys


def write(folder, kinds=None, seed=1):
    """Organ images + cut-outs (<type>.cut.json) into a folder; drawn once."""
    os.makedirs(folder, exist_ok=True); out = {}
    for k in kinds or TYPES:
        p = os.path.join(folder, f"{k}.png"); c = os.path.join(folder, f"{k}.cut.json")
        if not os.path.exists(p):
            a = TYPES[k](seed); Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "RGBA").save(p)
        if not os.path.exists(c):
            json.dump(cutout(np.asarray(Image.open(p).convert("RGBA")).astype(np.float32) / 255), open(c, "w"))
        out[k] = p
    return out


if __name__ == "__main__":
    print("__RESULT__" + json.dumps(write(sys.argv[1], sys.argv[2:] or None), ensure_ascii=False))
