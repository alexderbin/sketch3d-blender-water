"""Leaf optics: pigments and structure → reflectance and transmittance spectra (PROSPECT-D, Féret et al. 2017; the prosail 2.0.5 code — ext/prosail
as is, with the adapter below) → colour in linear sRGB / Rec.709 (CIE 1931 2°, illuminant D65 — ext/cie): the leaf "albedo" for reflection and for transmission.
The sun comes from the node's lamp: warm evening light × yellow-green transmittance = gold when backlit with no tinting.
.venv/bin/python -m components.plants.leaf_optics [set …] — a table of the colours of the sets (__RESULT__)"""
import core.paths
import functools, json, os, sys, types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); EXT = os.path.join(HERE, "ext")
# Pigments by PROSPECT-D: N — structure (layers), cab — chlorophyll a+b (µg/cm²), car — carotenoids, ant — anthocyanins, brown — brown pigments,
# cw — water (cm), cm — dry matter (g/cm²). The sets are typical values from the PROSPECT literature (LOPEX93, ANGERS; [published]) — checked against
# colour and transmittance: a green leaf transmits 5–15 % of the light, yellow-green; needles — almost nothing.
SETS = {
    "broad":       {"N": 1.5, "cab": 40, "car": 10, "ant": 0.0, "brown": 0.0, "cw": 0.012, "cm": 0.0050},   # a deciduous leaf in summer
    "broad_thin":  {"N": 1.3, "cab": 28, "car": 7, "ant": 0.0, "brown": 0.0, "cw": 0.009, "cm": 0.0032},    # thin: beech, birch, linden
    "broad_thick": {"N": 1.8, "cab": 50, "car": 12, "ant": 0.0, "brown": 0.0, "cw": 0.016, "cm": 0.0080},   # thick, leathery: oak, tropics
    "young":       {"N": 1.2, "cab": 15, "car": 5, "ant": 0.3, "brown": 0.0, "cw": 0.007, "cm": 0.0025},    # spring, young
    "needle":      {"N": 2.6, "cab": 55, "car": 13, "ant": 0.0, "brown": 0.0, "cw": 0.030, "cm": 0.0200},   # needles: thick, dark
    "grass":       {"N": 1.5, "cab": 35, "car": 9, "ant": 0.0, "brown": 0.0, "cw": 0.010, "cm": 0.0040},    # grass blade
    "dry_grass":   {"N": 1.8, "cab": 2, "car": 5, "ant": 0.0, "brown": 0.9, "cw": 0.001, "cm": 0.0060},     # dry grass, straw
    "autumn_yellow": {"N": 1.5, "cab": 3, "car": 12, "ant": 0.0, "brown": 0.15, "cw": 0.010, "cm": 0.0050},
    "autumn_red":  {"N": 1.5, "cab": 5, "car": 8, "ant": 12.0, "brown": 0.2, "cw": 0.010, "cm": 0.0050},
    "dead":        {"N": 2.0, "cab": 0.5, "car": 3, "ant": 0.0, "brown": 2.0, "cw": 0.001, "cm": 0.0070},   # dry brown leaf
}


@functools.lru_cache(None)
def _prospect():
    sys.modules.setdefault("numba", types.SimpleNamespace(jit=lambda *a, **k: (lambda f: f)))   # the package's SAIL is not needed — a stub
    p = os.path.join(EXT, "prosail")
    if p not in sys.path: core.paths.component_ext("plants", "prosail")
    import prosail
    from prosail.prospect_d import calctav
    return prosail.run_prospect, (1.0 - calctav(40.0, prosail.spectral_lib.prospectd.nr))


@functools.lru_cache(None)
def _cie():
    X = np.loadtxt(os.path.join(EXT, "cie", "CIE_xyz_1931_2deg.csv"), delimiter=",")
    D = np.loadtxt(os.path.join(EXT, "cie", "CIE_std_illum_D65.csv"), delimiter=",")
    wl = np.arange(400, 701); cmf = np.stack([np.interp(wl, X[:, 0], X[:, k]) for k in (1, 2, 3)], 1); d65 = np.interp(wl, D[:, 0], D[:, 1])
    M = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]])
    return wl, cmf * d65[:, None] / float((cmf[:, 1] * d65).sum()), M


def rgb(spec_wl, spec):
    """A spectrum (a 0–1 share per wavelength) → linear sRGB under D65 light (a white surface → 1, 1, 1)."""
    wl, W, M = _cie(); s = np.interp(wl, spec_wl, spec); xyz = s @ W
    return np.clip(M @ xyz, 0.0, None)


@functools.lru_cache(None)
def optics(name="broad", **over):
    """A set of pigments (+ corrections) → {'R': total reflectance, 'Rd': the leaf body (without the cuticle reflection — the shader gloss provides that),
    'T': transmittance, 'F': surface reflectance (mean), 'R_vis', 'T_vis', 'A_vis': shares in the visible range} — rgb in linear."""
    p = dict(SETS[name]); p.update(over); run, rs = _prospect()
    wv, r, t = run(p["N"], p["cab"], p["car"], p["brown"], p["cw"], p["cm"], ant=p["ant"])
    vis = (wv >= 400) & (wv <= 700)
    return {"R": rgb(wv, r).round(4).tolist(), "Rd": np.maximum(rgb(wv, np.clip(r - rs, 0, None)), 0.003).round(4).tolist(), "T": rgb(wv, t).round(4).tolist(),
            "F": round(float(rs[vis].mean()), 4),
            "R_vis": round(float(r[vis].mean()), 4), "T_vis": round(float(t[vis].mean()), 4), "A_vis": round(float(1 - r[vis].mean() - t[vis].mean()), 4)}


if __name__ == "__main__":
    names = sys.argv[1:] or list(SETS)
    print("__RESULT__" + json.dumps({n: optics(n) for n in names}, ensure_ascii=False))
