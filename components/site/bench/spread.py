"""A measure of the "sameness" of the ground in a lit frame (bl_relief look=lit): the ground is the ground shader's AOV mask ground (lit.exr;
without it — all geometry by alpha), the frame is split into 12 × 7 cells, and a cell with more than 60 % ground gets its mean colour in Lab. Two layers:
"surface" — the surface colour without light (Diffuse Color), "frame" — lit.png (light and haze). Numbers: the spread of lightness and chroma
of the cells (std. dev. of L and of a, b), the share of "single-colour" cells — ΔE to the mean under 4 (the eye sees one blotch). Compare the
same frame before and after.
.venv/bin/python -m components.site.bench.spread <frame folder with lit.exr and lit.png> […] → a line per frame and __RESULT__ JSON"""
import json, os, sys

import cv2
import numpy as np


def channels(path):
    import OpenEXR
    ch = {}
    for p in OpenEXR.File(path).parts:
        for name, c in p.channels.items():
            px = np.asarray(c.pixels, np.float32); name = name.replace("ViewLayer.", "")
            if px.ndim == 3:
                for i, k in enumerate("RGBA"[:px.shape[2]]): ch[k if name in ("RGBA", "RGB", "Combined") else f"{name}.{k}"] = px[..., i]
            else: ch[name] = px
    return ch


def _lab(rgb_lin):
    srgb = np.where(rgb_lin <= 0.0031308, 12.92 * rgb_lin, 1.055 * np.power(np.clip(rgb_lin, 0, None), 1 / 2.4) - 0.055)
    return cv2.cvtColor(np.clip(srgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)          # L 0–100, a, b


def _cells(lab, m, nx, ny):
    H, W = m.shape; out = []
    for j in range(ny):
        for i in range(nx):
            sl = (slice(j * H // ny, (j + 1) * H // ny), slice(i * W // nx, (i + 1) * W // nx)); mm = m[sl]
            if mm.mean() > 0.6: out.append(lab[sl][mm].mean(0))
    if len(out) < 6: return None
    C = np.array(out); dE = np.linalg.norm(C - C.mean(0), axis=1)
    return {"cells": len(C), "sL": round(float(C[:, 0].std()), 2), "sab": round(float(np.hypot(C[:, 1].std(), C[:, 2].std())), 2),
            "uniform_share": round(float((dE < 4).mean()), 2)}


def spread(d, nx=12, ny=7):
    ch = channels(os.path.join(d, "lit.exr"))
    gr = next((v for k, v in ch.items() if k.endswith("ground") or k.endswith("ground.X") or k.endswith("ground.V")), None)
    m = (gr > 0.5) if gr is not None else (ch["A"] > 0.99)
    res = {}
    dc = [ch.get(f"Diffuse Color.{c}") for c in "RGB"]
    if all(x is not None for x in dc): res["surface"] = _cells(_lab(np.stack(dc, -1)), m, nx, ny)
    img = cv2.cvtColor(cv2.imread(os.path.join(d, "lit.png")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    res["frame"] = _cells(cv2.cvtColor(img, cv2.COLOR_RGB2LAB), m, nx, ny)
    return res


if __name__ == "__main__":
    res = {}
    for d in sys.argv[1:]:
        r = spread(d); res[d] = r; print(os.path.basename(d.rstrip("/")), r, flush=True)
    print("__RESULT__" + json.dumps(res, ensure_ascii=False))
