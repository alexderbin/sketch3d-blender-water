"""Water-column color from the substance in it: absorption a(λ) and backscattering b_b(λ) of pure water, phytoplankton
(chlorophyll), dissolved organics (humus, tannins — CDOM) and suspended matter (silt, clay, rock flour) → the column
reflectance r_rs = 0.089·u + 0.125·u², u = b_b/(a + b_b) (Gordon 1988, Lee 1998), above the water Rrs = 0.52·r_rs/(1 − 1.7·r_rs)
→ the column color for the material (diffuse albedo π·Rrs in linear sRGB at D65) and the attenuation K(λ) ≈ a + b_b (a shoal,
the bed is visible); the check is the Forel–Ule scale index by the chromaticity of Rrs (hue angle, Novoa et al. 2013, table 6).
Pure water — Pope & Fry 1997 (from the table; 450, 550, 650 nm checked against NASA); water scattering — Morel 1974; the
absorption shapes of plankton, organics and suspended matter are Gaussians and exponentials after ocean-optics textbooks [estimate].
  .venv/bin/python -m components.water.optics — the presets table: color, K, FU index."""
import math

import numpy as np

LAM = np.arange(400.0, 701.0, 10.0)
A_W = np.array([0.00663, 0.00473, 0.00454, 0.00495, 0.00635, 0.00922, 0.00979, 0.0106, 0.0127, 0.0150, 0.0204, 0.0325, 0.0409,
                0.0434, 0.0474, 0.0565, 0.0619, 0.0695, 0.0896, 0.1351, 0.2224, 0.2644, 0.2755, 0.2916, 0.3108, 0.340, 0.410,
                0.439, 0.465, 0.516, 0.624])                    # m⁻¹, Pope & Fry 1997
D65 = np.array([82.75, 91.49, 93.43, 86.68, 104.86, 117.01, 117.81, 114.86, 115.92, 108.81, 109.35, 107.80, 104.79, 107.69,
                104.41, 104.05, 100.00, 96.33, 95.79, 88.69, 90.01, 89.60, 87.70, 83.29, 83.70, 80.03, 80.21, 82.28, 78.28,
                69.72, 71.61])
FU_ANGLE = [227.168, 220.977, 209.994, 190.779, 163.084, 132.999, 109.054, 94.037, 83.346, 74.572, 67.957, 62.186, 56.435,
            50.665, 45.129, 39.769, 34.906, 30.439, 26.337, 22.741]   # Novoa et al. 2013, table 6: FU i holds while the angle > the angle i


def _g(x, mu, s1, s2):
    return np.exp(-0.5 * ((x - mu) / np.where(x < mu, s1, s2)) ** 2)


def cmf(lam):
    """CIE 1931 color matching functions — the analytic approximation of Wyman, Sloan, Shirley 2013."""
    x = 1.056 * _g(lam, 599.8, 37.9, 31.0) + 0.362 * _g(lam, 442.0, 16.0, 26.7) - 0.065 * _g(lam, 501.1, 20.4, 26.2)
    y = 0.821 * _g(lam, 568.8, 46.9, 40.5) + 0.286 * _g(lam, 530.9, 16.3, 31.1)
    z = 1.217 * _g(lam, 437.0, 11.8, 36.0) + 0.681 * _g(lam, 459.0, 26.0, 13.8)
    return x, y, z


XYZ2RGB = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
_X, _Y, _Z = cmf(LAM)


def rgb_of(refl):
    """Spectral reflectance (0…1) → linear sRGB at D65 (white 1 → (1, 1, 1))."""
    XYZ = np.array([(refl * D65 * c).sum() for c in (_X, _Y, _Z)]) / (D65 * _Y).sum()
    white = np.array([(D65 * c).sum() for c in (_X, _Y, _Z)]) / (D65 * _Y).sum()
    rgb = XYZ2RGB @ XYZ; rgb_w = XYZ2RGB @ white
    return np.maximum(rgb / rgb_w, 0.0)


def iop(chl=0.05, cdom=0.01, sed=0.0, flour=0.0):
    """Absorption a(λ) and backscattering b_b(λ), m⁻¹. chl — chlorophyll, mg/m³; cdom — organics absorption at 440 nm, m⁻¹;
    sed — suspended silt and clay, g/m³; flour — glacial rock flour, g/m³ (fine and light — it strongly scatters blue)."""
    a_ph = 0.06 * chl ** 0.65 * (np.exp(-0.5 * ((LAM - 440) / 32) ** 2) + 0.45 * np.exp(-0.5 * ((LAM - 675) / 14) ** 2) + 0.12)
    a_g = cdom * np.exp(-0.015 * (LAM - 440))
    a_d = 0.04 * sed * np.exp(-0.011 * (LAM - 440)) + 0.004 * flour * np.exp(-0.011 * (LAM - 440))
    b_bw = 0.0019 * (400 / LAM) ** 4.32
    b_bp = 0.0006 * chl ** 0.62 * (550 / LAM) + 0.012 * sed * (550 / LAM) ** 0.5 + 0.02 * flour * (550 / LAM) ** 1.2
    return A_W + a_ph + a_g + a_d, b_bw + b_bp


def look(chl=0.05, cdom=0.01, sed=0.0, flour=0.0):
    """Column look: body — the diffuse albedo of deep water (linear sRGB), K — attenuation per R, G, B (m⁻¹, weighted by the CMF),
    rrs — Rrs(λ), fu — the Forel–Ule index, angle — the hue angle, °."""
    a, bb = iop(chl, cdom, sed, flour)
    u = bb / (a + bb); r = 0.089 * u + 0.125 * u * u
    Rrs = 0.52 * r / (1 - 1.7 * r)
    body = rgb_of(math.pi * Rrs / 0.94)
    K = a + bb
    w = [np.maximum(XYZ2RGB[i] @ np.stack([_X, _Y, _Z]), 0) for i in range(3)]   # the wavelength weight in the R, G, B channels
    Krgb = [float(-np.log((np.exp(-K * 1.0) * wi * D65).sum() / (wi * D65).sum())) for wi in w]   # per 1 m of path
    X, Y, Z = ((Rrs * c).sum() for c in (_X, _Y, _Z))
    x, y = X / (X + Y + Z), Y / (X + Y + Z)
    ang = math.degrees(math.atan2(y - 1 / 3, x - 1 / 3)) % 360
    fu = next((i + 1 for i, t in enumerate(FU_ANGLE) if ang > t), 21)
    return {"body": tuple(float(c) for c in body), "K": tuple(Krgb), "fu": fu, "angle": round(ang, 1), "rrs": Rrs}


PRESETS = {                                     # the substance in the water by type; fu — what is expected on the scale
    "ocean": dict(chl=0.05, cdom=0.005, fu=(1, 3)),
    "ocean_blue": dict(chl=0.2, cdom=0.01, fu=(2, 4)),
    "coastal": dict(chl=2.0, cdom=0.15, sed=2.0, fu=(6, 10)),
    "north_sea": dict(chl=1.5, cdom=0.2, sed=5.0, fu=(7, 11)),
    "lagoon": dict(chl=0.1, cdom=0.01, sed=0.5, fu=(2, 5)),
    "mountain_lake": dict(chl=0.5, cdom=0.05, fu=(3, 7)),
    "glacial_lake": dict(chl=0.2, cdom=0.02, flour=8.0, fu=(3, 6)),
    "pond_clear": dict(chl=2.0, cdom=0.3, sed=1.0, fu=(9, 14)),
    "pond_green": dict(chl=30.0, cdom=0.8, sed=3.0, fu=(10, 15)),
    "forest_pond": dict(chl=3.0, cdom=6.0, sed=2.0, fu=(17, 21)),
    "muddy_river": dict(chl=2.0, cdom=0.8, sed=80.0, fu=(16, 21)),
}


def preset(name):
    p = dict(PRESETS[name]); p.pop("fu", None)
    return look(**p)


if __name__ == "__main__":
    for nm, p in PRESETS.items():
        q = dict(p); want = q.pop("fu"); L = look(**q)
        ok = want[0] <= L["fu"] <= want[1]
        print(f"{nm:14s} FU {L['fu']:>2} (expected {want[0]}–{want[1]}) {'ok' if ok else '—'}  angle {L['angle']:6.1f}  "
              f"column {tuple(round(c, 4) for c in L['body'])}  K {tuple(round(k, 3) for k in L['K'])}")
