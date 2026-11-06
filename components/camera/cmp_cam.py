"""Comparison of Blender and sketch25 frames by object id and depth + a sheet for viewing.
Usage: .venv/bin/python -m components.camera.cmp_cam <field folder> <camera 1..>
Reads out/<name>/sk25/cam{k}-buf.* and out/<name>/blender/cam{k}-{id.png,idmap.json,passes.exr,workbench.png,eevee.png}."""
import json
import os
import sys

import cv2
import numpy as np
import OpenEXR
from PIL import Image, ImageDraw, ImageFont

root, k = sys.argv[1], int(sys.argv[2])
sk, bl = os.path.join(root, "sk25"), os.path.join(root, "blender")

# --- sketch25 labels
buf = np.load(f"{sk}/cam{k}-buf.npz")
meta = json.load(open(f"{sk}/cam{k}-buf.json"))
names = {int(g): v for g, v in meta["names"].items()}


def canon_sk(g):
    if g < 0:
        return "sky"
    i, kind = names[g]
    return None if kind in ("member", "crowd") else ("ground" if kind == "ground" else i)


# --- Blender labels
idmap = json.load(open(f"{bl}/cam{k}-idmap.json"))
inv = {v: n for n, v in idmap.items()}
im = np.array(Image.open(f"{bl}/cam{k}-id.png").convert("RGB")).astype(np.int32)
bid = im[..., 0] + 256 * im[..., 1]


def canon_bl(n):
    if n is None:
        return "sky"
    n = n.split(".")[0]
    if n.startswith("figure_"):
        n = n[len("figure_"):]
    return "ground" if n.startswith("ground") else n


H, W = bid.shape
lut_sk = {int(g): canon_sk(int(g)) for g in np.unique(buf["gid"])}
lut_bl = {int(v): (canon_bl(inv.get(int(v))) if v else "sky") for v in np.unique(bid)}
names_all = sorted({n for n in list(lut_sk.values()) + list(lut_bl.values()) if n})
code = {n: i + 1 for i, n in enumerate(names_all)}
A = np.zeros((H, W), np.int32)       # sketch25 (0 — a tree crown / a crowd member, excluded)
Bm = np.zeros((H, W), np.int32)
for g, n in lut_sk.items():
    A[buf["gid"] == g] = code[n] if n else 0
for v, n in lut_bl.items():
    Bm[bid == v] = code[n]
valid = A > 0
agree = (A == Bm) & valid

rep = {"frame": f"{W}x{H}", "pixels compared": int(valid.sum()), "matched, %": round(100 * agree.sum() / valid.sum(), 2),
       "objects": {}}
for n, c in code.items():
    a, b = (A == c), ((Bm == c) & valid)
    u = (a | b).sum()
    if u > 200:
        rep["objects"][n] = {"sk25_px": int(a.sum()), "blender_px": int(b.sum()), "IoU": round(float((a & b).sum() / u), 3)}

# --- depth (along the camera axis)
f = OpenEXR.File(f"{bl}/cam{k}-passes.exr")
zb = None
for p in f.parts:
    if p.name().endswith("Depth"):
        zb = np.array(list(p.channels.values())[0].pixels)
zs = buf["depth_z"]
m = agree & (zs > 0) & np.isfinite(zb) & (zb < 1e6) & (A != code["sky"])
rel = np.abs(zb[m] - zs[m]) / zs[m]
rep["depth"] = {"n_pixels": int(m.sum()), "median error, %": round(float(np.median(rel) * 100), 3),
                  "95th percentile, %": round(float(np.percentile(rel, 95) * 100), 2)}
for n in ("wall", "tower", "ground"):
    mm = m & (A == code.get(n, -1))
    if mm.sum() > 100:
        r = np.abs(zb[mm] - zs[mm]) / zs[mm]
        rep["depth"][n] = {"median, %": round(float(np.median(r) * 100), 3), "p95, %": round(float(np.percentile(r, 95) * 100), 2)}
json.dump(rep, open(f"{root}/cam{k}-compare.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(rep, ensure_ascii=False, indent=1))

# --- sheet
def load(p):
    return Image.open(p).convert("RGB")


ref = load(f"{sk}/cam{k}-toneline.png")
wb, ev = load(f"{bl}/cam{k}-workbench.png"), load(f"{bl}/cam{k}-eevee.png")
# Blender object edges over the sketch25 toneline
edge = np.zeros((H, W), bool)
edge[:, 1:] |= Bm[:, 1:] != Bm[:, :-1]
edge[1:, :] |= Bm[1:, :] != Bm[:-1, :]
ov = np.array(ref).copy()
ov[cv2.dilate(edge.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0] = (255, 0, 0)
dis = np.full((H, W, 3), 235, np.uint8)
dis[valid] = (205, 205, 205)
dis[valid & ~agree] = (220, 30, 30)
S = (768, 432)
tiles = [("sketch25 toneline", ref), ("Blender Workbench", wb), ("Blender Eevee (sun, shadows)", ev),
         ("Blender edges (red) on sketch25", Image.fromarray(ov)),
         (f"id mismatch: {100 - rep['matched, %']:.2f}% pix.", Image.fromarray(dis)),
         ("sketch25 ids", load(f"{sk}/cam{k}-ids.png"))]
FONT = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 15)
sheet = Image.new("RGB", (S[0] * 3, (S[1] + 20) * 2), "white")
for i, (t, img) in enumerate(tiles):
    x, y = (i % 3) * S[0], (i // 3) * (S[1] + 20)
    sheet.paste(img.resize(S, Image.LANCZOS), (x, y + 20))
    ImageDraw.Draw(sheet).text((x + 6, y + 1), t, fill=(0, 0, 0), font=FONT)
sheet.save(f"{root}/cam{k}-sheet.jpg", quality=88)
print("sheet:", f"{root}/cam{k}-sheet.jpg")
