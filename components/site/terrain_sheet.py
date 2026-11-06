"""Top-down site sheet (components/site/terrain.py): on the left — the near plot (ground color × hillshade, contours, water, paths,
plan anchors, scale, north), on the right — the belt out to the horizon with the plot's frame and the checks result. The plan can
be checked from the sheet without a render; the same view works as a "map as frame" shot. .venv/bin/python -m components.site.terrain_sheet <site_folder> [out.png]"""
import json, math, os, re, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from core.surface.ground import Ground

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def font(n):
    try: return ImageFont.truetype(FONT, n)
    except Exception: return ImageFont.load_default()


def shade(G, X, Y, e):
    gx, gy = G.grad(X, Y, e); n = np.stack([-gx, -gy, np.ones_like(gx)], -1); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    L = np.array([-0.5, 0.5, 0.7]); L /= np.linalg.norm(L)
    return np.clip(n @ L, 0, 1)


def contours(Z, step):
    q = np.floor(Z / step); return (q != np.roll(q, 1, 0)) | (q != np.roll(q, 1, 1))


_TEX = {}


def tex_colors(G, X, Y):
    """Color from the baked tile texture (relief_tex.png, components/site/terrain_tex.py) at world points — the sheet as a 3D view; if absent — Ground.colors."""
    tp, tj = os.path.join(G.folder, "relief_tex.png"), os.path.join(G.folder, "relief_tex.json")
    if not (os.path.exists(tp) and os.path.exists(tj)): return G.colors(X, Y)
    if tp not in _TEX: _TEX[tp] = (np.asarray(Image.open(tp).convert("RGB"), np.float32)[::-1] / 255, json.load(open(tj)))
    T, m = _TEX[tp]; xs, ys, k = np.asarray(m["xs"]), np.asarray(m["ys"]), m["px"]
    u = (np.interp(X, xs, np.arange(len(xs))) + 0.5) * k - 0.5; v = (np.interp(Y, ys, np.arange(len(ys))) + 0.5) * k - 0.5
    i = np.clip(np.rint(u).astype(int), 0, T.shape[1] - 1); j = np.clip(np.rint(v).astype(int), 0, T.shape[0] - 1)
    return T[j, i]


def panel(G, b, px, step_c=None, far=False):
    """Image of the box b = (x0, y0, x1, y1) of size px: color × hillshade, contours; north is up."""
    xs = np.linspace(b[0], b[2], px); ys = np.linspace(b[3], b[1], px); X, Y = np.meshgrid(xs, ys); e = (b[2] - b[0]) / px
    Z = G.height(X, Y); C = tex_colors(G, X, Y)
    sh = shade(G, X, Y, max(e, G.nres)); img = C * (0.55 + 0.6 * sh)[..., None]
    rng = float(np.ptp(Z)); step = step_c or next((s for s in (1, 2, 5, 10, 20, 50, 100) if rng / s <= 28), 200)
    cm = contours(Z, step); img[cm] *= 0.72
    if not far:
        dep = G.water(X, Y); w = np.clip(dep / 1.5, 0.35, 0.85)[..., None] * (dep > 0.03)[..., None]
        img = img * (1 - w) + np.array([0.28, 0.45, 0.62]) * w
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)), step


def sheet(folder, out):
    G = Ground(folder); loc = G.loc; nb = G.near_bounds(); fb = G.far_bounds()
    W, P = 1240, 800; im = Image.new("RGB", (W, P + 40), (24, 24, 24)); d = ImageDraw.Draw(im)
    left, step = panel(G, nb, P); im.paste(left, (0, 40))
    to = lambda x, y: ((x - nb[0]) / (nb[2] - nb[0]) * P, 40 + (nb[3] - y) / (nb[3] - nb[1]) * P)
    col = {"road": (150, 105, 60), "river": (60, 110, 210), "ridge": (60, 50, 40), "valley": (90, 70, 50), "cliff": (200, 60, 40)}
    for nm, p in loc["paths"].items():
        Q = np.asarray(p["pts"]); pts = [to(x, y) for x, y, _ in Q]
        if p["kind"] in col and len(pts) > 1: d.line(pts, fill=col[p["kind"]], width=2 if p["kind"] != "road" else 3)
    pub = set(loc.get("public") or loc["anchors"])
    f13 = font(13)
    for nm, (x, y, z) in loc["anchors"].items():
        if nm not in pub or not (nb[0] <= x <= nb[2] and nb[1] <= y <= nb[3]): continue
        if re.search(r"\.(edge|foot|rim|ring)_[nesw]$|\.(start|end|mid|side_l|side_r)$", nm) and nm not in loc.get("named", []): continue
        u, v = to(x, y); d.ellipse([u - 4, v - 4, u + 4, v + 4], fill=(230, 40, 40), outline=(255, 255, 255))
        d.text((u + 6, v - 8), f"{nm} {z:.0f}", fill=(255, 255, 255), font=f13, stroke_width=2, stroke_fill=(0, 0, 0))
    span = nb[2] - nb[0]; bar = next(s for s in (10, 20, 50, 100, 200, 500) if s >= span / 6)
    bpx = bar / span * P; d.line([(14, 40 + P - 16), (14 + bpx, 40 + P - 16)], fill=(255, 255, 255), width=4)
    d.text((18 + bpx, 40 + P - 28), f"{bar} m", fill=(255, 255, 255), font=f13, stroke_width=2, stroke_fill=(0, 0, 0))
    d.polygon([(P - 24, 56), (P - 30, 74), (P - 18, 74)], fill=(255, 255, 255)); d.text((P - 28, 76), "S", fill=(255, 255, 255), font=f13)
    st = loc["stats"]
    d.text((8, 8), f"{os.path.basename(os.path.normpath(folder))}: plot {span:.0f} m, step {G.nres:g} m, heights {st['z_min']}…{st['z_max']} m, "
                   f"contours every {step} m, steeper than 35° — {st['steep_35'] * 100:.0f} %, water — {st['water_share'] * 100:.1f} %",
           fill=(240, 220, 120), font=font(15))
    R = W - P - 20; right, _ = panel(G, fb, R, far=True); im.paste(right, (P + 10, 40))
    tf = lambda x, y: (P + 10 + (x - fb[0]) / (fb[2] - fb[0]) * R, 40 + (fb[3] - y) / (fb[3] - fb[1]) * R)
    d.rectangle([tf(nb[0], nb[3]), tf(nb[2], nb[1])], outline=(255, 60, 60), width=2)
    d.text((P + 14, 44), f"belt {fb[2] - fb[0]:.0f} m, mountains up to {st['far_max']} m", fill=(255, 255, 255), font=f13, stroke_width=2, stroke_fill=(0, 0, 0))
    y = 40 + R + 10; f12 = font(12)
    for c in loc["checks"]:
        txt = ("✓ " if c["ok"] else "✗ ") + c["check"] + ": " + ", ".join(f"{k} {v}" for k, v in c.items() if k not in ("check", "ok"))
        d.text((P + 12, y), txt[:70], fill=(150, 220, 150) if c["ok"] else (255, 120, 120), font=f12); y += 15
        if y > P + 30: break
    im.save(out)
    return out


if __name__ == "__main__":
    f = os.path.abspath(sys.argv[1]); print(sheet(f, sys.argv[2] if len(sys.argv) > 2 else os.path.join(f, "sheet.png")))
