"""The wave engine: one model for all waters — a spectrum from wind, fetch and depth (JONSWAP capped by Pierson–Moskowitz, a TMA
depth correction), the swell as a separate spectrum, directionality (Hasselmann, Horvath's swell elongation, the "scattered"
fraction), dispersion with capillarity ω² = (g·k + σ/ρ·k³)·tanh(k·h). The bands are FFT tiles of different sizes, each with its
own range of wavelengths (no double counting); sharp crests — a displacement toward the crest (λ); the Jacobian → foam with
memory (the sum of past injections with decay and wind drift — a pure function of time, frames in any order); a time loop — the
frequencies are multiples of 2π/T. Slopes finer than the bands — the remainder up to Cox–Munk (the material roughness). Pure
numpy — in both the project's Python and Blender.
  .venv/bin/python -m components.water.waves [u10=8] [fetch=2000] [depth=50] [swell_hs=1 swell_tp=10] — a summary of the water state."""
import math
import sys

import numpy as np

G = 9.81
ST = 7.28e-5                                   # σ/ρ, m³/s²: the surface tension of water over the density
MONAHAN = (3.84e-6, 3.41)                      # whitecap fraction W = a·U10^b (Monahan, O'Muircheartaigh 1980)
COX_MUNK = 5.12e-3                             # mean square slope of clean water σ² = 5.12·10⁻³·U12.5 (Cox, Munk 1954, without the 0.003)


def omega(k, depth=None):
    """Frequency from the wavenumber: ω² = (g·k + σ/ρ·k³)·tanh(k·h); depth None — deep."""
    w2 = G * k + ST * k ** 3
    if not _deep(depth): w2 = w2 * np.tanh(np.minimum(k * depth, 20.0))
    return np.sqrt(w2)


def _deep(depth):
    """No depth given (None, 0) — deep water; an array of depths — per point."""
    return depth is None or (np.isscalar(depth) and not depth)


def domega_dk(k, depth=None):
    """The group velocity dω/dk of the same dispersion."""
    gk, dg = G * k + ST * k ** 3, G + 3 * ST * k * k
    if _deep(depth): return dg / (2 * np.sqrt(gk))
    T = np.tanh(np.minimum(k * depth, 20.0))
    return (dg * T + gk * depth * (1 - T * T)) / (2 * np.sqrt(np.maximum(gk * T, 1e-12)))


def wavenumber(w, depth=None):
    """k from ω (inverse dispersion, Newton): for a wavelength and a period."""
    k = np.maximum(np.asarray(w, float) ** 2 / G, 1e-6)
    for _ in range(30):
        k = np.maximum(k - (omega(k, depth) - w) / domega_dk(k, depth), 1e-6)
    return k


def jonswap_params(u10, fetch=None):
    """(α, ωp, γ) from the wind at 10 m and the fetch, m: JONSWAP (Hasselmann et al. 1973) — α = 0.076·χ^−0.22, fp·U/g = 3.5·χ^−0.33,
    χ = g·F/U²; the ceiling is the Pierson–Moskowitz fully developed sea (α 0.0081, fp·U/g 0.13, a peak without enhancement γ = 1);
    γ goes from 3.3 for a young sea to 1 for a developed one. fetch None — a fully developed sea."""
    u = max(float(u10), 0.05)
    chi = 1e12 if not fetch else G * float(fetch) / (u * u)
    nu = max(3.5 * chi ** -0.33, 0.13)
    return max(0.076 * chi ** -0.22, 0.0081), 2 * math.pi * nu * G / u, 1.0 + 2.3 * min(max((nu / 0.13 - 1) / 0.5, 0.0), 1.0)


def jonswap(w, alpha, wp, gamma=3.3):
    """S(ω), m²·s: α·g²·ω⁻⁵·exp(−5/4·(ωp/ω)⁴)·γ^r, r = exp(−(ω − ωp)²/(2σ²ωp²)), σ 0.07 / 0.09."""
    w = np.maximum(w, 1e-6)
    s = np.where(w <= wp, 0.07, 0.09)
    return alpha * G * G * w ** -5 * np.exp(-1.25 * (wp / w) ** 4) * gamma ** np.exp(-((w - wp) ** 2) / (2 * s * s * wp * wp))


def tma(w, depth=None):
    """Kitaigorodskii depth correction (TMA): Φ(ωh), ωh = ω·√(h/g)."""
    if not depth: return 1.0
    wh = w * math.sqrt(depth / G)
    return np.where(wh <= 1, 0.5 * wh * wh, np.where(wh < 2, 1 - 0.5 * (2 - wh) ** 2, 1.0))


def _lgamma(z):
    """ln Γ(z) by Stirling with a shift of 3 (z > 0, accuracy ~1e-7)."""
    z3 = z + 3.0
    st = (z3 - 0.5) * np.log(z3) - z3 + 0.5 * math.log(2 * math.pi) + 1 / (12 * z3) - 1 / (360 * z3 ** 3)
    return st - np.log(z * (z + 1) * (z + 2))


def spread(w, wp, u10, theta, swell=0.0, mix=1.0, smul=1.0):
    """D(θ), ∫ over [−π, π] = 1: Q(s)·|cos(θ/2)|^(2s) — Hasselmann (1980) by ω/ωp; swell — swell elongation s += 16·tanh(ωp/ω)·ξ²
    (Horvath 2015); mix — the directional fraction (the rest is scattered)."""
    x = np.maximum(w / wp, 1e-3)
    s = np.where(x <= 1, 6.97 * x ** 4.06, 9.77 * x ** (-2.33 - 1.45 * (u10 * wp / G - 1.17)))
    s = np.clip(s * smul + 16 * np.tanh(1 / x) * swell * swell, 1e-3, 2000.0)   # smul > 1 — narrower at the peak (Mitsuyasu s_max ~10–20)
    D = np.exp(_lgamma(s + 1) - _lgamma(s + 0.5) - math.log(2 * math.sqrt(math.pi))
               + 2 * s * np.log(np.maximum(np.abs(np.cos(theta / 2)), 1e-30)))
    return mix * D + (1 - mix) / (2 * math.pi)


BEAUFORT_HS = [0.0, 0.1, 0.2, 0.6, 1.0, 2.0, 3.0, 4.0, 5.5, 7.0, 9.0, 11.5, 14.0, 16.0]   # height by force (the lower bound of the likely range;
BEAUFORT_U = [0.0, 0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7, 40.0]   # open sea, the Beaufort scale); wind — the start of the force


def beaufort_hs(u10):
    """Storm wave height from the wind (the Beaufort table, linear between forces): the Pierson–Moskowitz fully developed sea
    overestimates at forces 10–12 (a storm is limited by time)."""
    return float(np.interp(u10, BEAUFORT_U, BEAUFORT_HS))


def spectrum_hs(u10, fetch=None):
    """Hs of the engine's JONSWAP spectrum (the integral of S(ω) without bands) — for choosing the fetch."""
    al, wp, ga = jonswap_params(u10, fetch)
    w = np.linspace(0.3 * wp, 12 * wp, 6000); S = jonswap(w, al, wp, ga)
    return 4 * math.sqrt(float(((S[1:] + S[:-1]) * np.diff(w)).sum() / 2))


def storm_fetch(u10, hs):
    """The fetch at which the engine's spectrum gives the height hs at wind u10 (bisection on the fetch): a "storm by height" — the same
    spectrum, younger than a fully developed sea; higher than fully developed — None (a fully developed sea)."""
    u = max(float(u10), 0.5)
    if spectrum_hs(u, None) <= hs: return None
    lo, hi = math.log(10.0), math.log(5e7)
    for _ in range(40):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if spectrum_hs(u, math.exp(m)) < hs else (lo, m)
    return math.exp(0.5 * (lo + hi))


def u_at(u10, z, z0=2e-4):
    """Wind speed at height z by the logarithmic profile above water (z0 ~ 0.2 mm)."""
    return u10 * math.log(z / z0) / math.log(10.0 / z0)


def plan_bands(lp, near=2.0, N=256, ratio=5.3, m=6.0, top=None):
    """Band tiles: the first is ~10 peak wavelengths (not less than 4·near), each next one is smaller by ratio (incommensurate, so the
    tiles do not coincide), until the finest becomes ~2·near (near — the nearest water from the camera, m). Wavelength ranges: for a
    band i>0 the longest wave is L_i/m (at least m waves on the tile), the shortest one in the last band is 4 points per wave."""
    L = [max(10.0 * lp, 4.0 * near, 0.5)] if top is None else [float(top)]
    kp = 2 * math.pi / lp                                           # keep making them finer until we reach the nearest water and until the peak
    while (L[-1] / ratio >= max(2.0 * near, 0.25) or math.pi * N / (2 * L[-1]) < 4 * kp) and len(L) < 6:   # gets a tail in the bands
        L.append(L[-1] / ratio)
    k_lo = [2 * math.pi / L[0]] + [2 * math.pi * m / l for l in L[1:]]
    k_hi = k_lo[1:] + [math.pi * N / L[-1] / 2.0]
    return [dict(L=l, N=N, klo=a, khi=b) for l, a, b in zip(L, k_lo, k_hi)]


class Band:
    """An FFT tile of L×L m, N×N points; modes with |k| in [klo, khi)."""

    def __init__(self, L, N, klo, khi):
        self.L, self.N, self.klo, self.khi = float(L), int(N), float(klo), float(khi)
        f = 2 * math.pi * np.fft.fftfreq(self.N, d=self.L / self.N)
        self.kx, self.ky = np.meshgrid(f, f)                           # [row = y, column = x]
        self.k = np.hypot(self.kx, self.ky)
        self.mask = (self.k >= self.klo) & (self.k < self.khi)
        self.dk = 2 * math.pi / self.L
        ki = np.where(self.k > 0, self.k, 1.0)
        self.ux, self.uy = self.kx / ki, self.ky / ki                   # k/|k|
        self.neg = (-np.arange(self.N)) % self.N                        # the index of −k


class Sea:
    """Water state. u10 — wind at 10 m, m/s; wind_to — where it blows toward (x, y); fetch — the fetch, m (None — a fully developed
    sea); depth — depth, m (None — deep); swell — {hs, tp, to: (x, y), xi: elongation 0…2} or None; chop — the crest sharpness λ;
    near — the nearest water from the camera, m (what is finer than the bands goes into roughness); loop — the loop period, s
    (None — no loop); seed — the seed.
    eval(t) → band fields; foam(t) — foam with memory; stats() — Hs, Tp, slopes, whitecaps."""

    def __init__(self, u10=5.0, wind_to=(1.0, 0.0), fetch=None, depth=None, swell=None, chop=1.0, near=2.0, N=256, seed=1,
                 loop=None, mix=0.85, gamma=None, bands=None, foam=None, hs=None, focus=None, smul=1.0):
        if hs is not None: fetch = storm_fetch(u10, hs)                 # the height is given (a storm by Beaufort) — the fetch is chosen to match it
        self.u10, self.fetch, self.depth, self.chop, self.loop = float(u10), fetch, depth, float(chop), loop
        a = math.atan2(wind_to[1], wind_to[0]); self.wind_to = (math.cos(a), math.sin(a)); self.wind_a = a
        self.alpha, self.wp, g_ = jonswap_params(self.u10, fetch); gamma = g_ if gamma is None else float(gamma)
        self.lp = float(2 * math.pi / wavenumber(self.wp, depth))       # peak wavelength, m
        self.parts = [dict(kind="wind", alpha=self.alpha, wp=self.wp, a=a, gamma=gamma, xi=0.0, mix=mix, u=self.u10, smul=float(smul))]
        if swell:
            sa = math.atan2(*reversed(swell.get("to", wind_to))); swp = 2 * math.pi / float(swell["tp"])
            self.parts.append(dict(kind="swell", alpha=1.0, wp=swp, a=sa, gamma=float(swell.get("gamma", 5.0)), xi=float(swell.get("xi", 1.2)),
                                   mix=1.0, u=G / swp * 0.82, hs=float(swell["hs"])))
            self.lp = max(self.lp, float(2 * math.pi / wavenumber(swp, depth)))
        self.bands = [Band(**b) for b in (bands or plan_bands(self.lp, near, N))]
        rng = np.random.default_rng(seed)
        for p in self.parts:                                            # swell — α from the given height (m0 = (Hs/4)²)
            if p["kind"] == "swell":
                w = np.linspace(p["wp"] * 0.3, p["wp"] * 6, 4000)
                S_ = jonswap(w, 1.0, p["wp"], p["gamma"]) * tma(w, depth); m0 = float(((S_[1:] + S_[:-1]) * np.diff(w)).sum() / 2)
                p["alpha"] = (p["hs"] / 4) ** 2 / m0
        for b in self.bands:
            w = omega(np.maximum(b.k, 1e-9), depth)
            psi = self.psi(b.kx, b.ky)
            lo = b.mask & (b.k < 16 * b.dk)                             # at the start of a band the grid is coarse — the average over a 4×4 cell
            if lo.any():
                o = (np.arange(4) - 1.5) / 4 * b.dk
                psi[lo] = np.mean([self.psi(b.kx[lo] + dx, b.ky[lo] + dy) for dx in o for dy in o], 0)
            b.psi = np.where(b.mask, psi, 0.0)                          # Ψ(kx, ky), m⁴: ∫∫Ψ dkx dky = m0
            amp = 0.5 * np.sqrt(b.psi) * b.dk
            b.h0 = amp * (rng.standard_normal(b.k.shape) + 1j * rng.standard_normal(b.k.shape))
            b.hm = np.conj(b.h0[np.ix_(b.neg, b.neg)])                  # h0*(−k)
            b.w = w if not loop else np.round(w * loop / (2 * math.pi)) * (2 * math.pi / loop)
            b.lam = 2 * math.pi / b.khi                                 # the shortest wave of the band, m
        self.focus = self._focus(focus) if focus else None              # rogue wave: the group converges at a point and a moment
        f = dict(thr=None, width=0.3, grow=4.0, tau=3.0, dt=0.1, drift=0.03, lmin=0.4, lfrac=60.0, N=128, cover=None)
        f.update(foam or {}); self.fp = f
        self._foam_setup()
        if f["thr"] is None: self.calibrate_foam(f["cover"])

    def _focus(self, fo):
        """A rogue wave by linear focusing (as in wave tanks): for a part of the modes of the first band the phases converge at the point
        fo["xy"] at the moment fo["t"] — the crest there is fo["crest"] m (default fo["k"]·Hs, k = 1.7: a wave of ~2.5 Hs from the
        trough, above the 2 Hs "rogue wave" threshold); the modes are picked at random across the whole band spectrum (a narrow group at
        the peak barely disperses and does not stand out); the modes lie near the wind direction (±12°): the focus is a line across the
        travel direction — a wall with a crest hundreds of meters long (from all directions — a pyramid); the other modes are random
        (the surrounding sea is the same). A mode h0(k) contributes 2·Re[h0·e^{i(k·x − ωt)}] to η — the phase is ωt − k·x. The group
        gathers and disperses by dispersion (~one peak period before and after). → {xy, t, crest, modes}."""
        b = self.bands[0]; x, y = (float(v) for v in fo["xy"]); tf = float(fo.get("t", 0.0))
        hs = 4 * math.sqrt(sum(float(bb.psi.sum()) * bb.dk ** 2 for bb in self.bands))
        A = float(fo.get("crest") or fo.get("k", 1.7) * hs)
        th = (np.arctan2(b.ky, b.kx) - self.wind_a + math.pi) % (2 * math.pi) - math.pi   # modes near the wind direction (±spread):
        amp = 2 * np.abs(b.h0) * b.mask * (np.abs(th) < math.radians(float(fo.get("spread", 12.0))))   # the focus is a line — a wall, not a pyramid
        r = np.random.default_rng(int(fo.get("seed", 99))).random(amp.shape)
        lo, hi = 0.0, 1.0                                                # modes across the whole spectrum width (a fraction p at random): the group
        for _ in range(30):                                              # a short one converges and disperses within ~one peak period
            p_ = 0.5 * (lo + hi); lo, hi = (p_, hi) if amp[r < p_].sum() < A else (lo, p_)
        sel = np.nonzero((r < hi) & (amp > 0)); n_ = len(sel[0])
        ph = b.w[sel] * tf - (b.kx[sel] * x + b.ky[sel] * y)
        b.h0[sel] = np.abs(b.h0[sel]) * np.exp(1j * ph)
        b.hm = np.conj(b.h0[np.ix_(b.neg, b.neg)])
        return {"xy": (x, y), "t": tf, "crest": round(A, 2), "modes": n_, "hs": round(hs, 2)}

    def psi(self, kx, ky):
        """Spectrum Ψ(kx, ky), m⁴ — the sum of the parts (wind, swell): S(ω)·Φ(ωh)·D(θ)·dω/dk / k."""
        k = np.maximum(np.hypot(kx, ky), 1e-9); w = omega(k, self.depth); out = np.zeros_like(k)
        for p in self.parts:
            th = (np.arctan2(ky, kx) - p["a"] + math.pi) % (2 * math.pi) - math.pi
            out += jonswap(w, p["alpha"], p["wp"], p["gamma"]) * tma(w, self.depth) * spread(w, p["wp"], p["u"], th, p["xi"], p["mix"], p.get("smul", 1.0))
        return out * domega_dk(k, self.depth) / k

    # --- fields ---
    def spectrum(self, b, t):
        """Complex amplitudes c(k, t) = h0·e^{iωt} + h0*(−k)·e^{−iωt}."""
        e = np.exp(-1j * b.w * t)                                       # e^{−iωt}: the mode h0(k) runs toward +k — downwind (Tessendorf's
        return b.h0 * e + b.hm * np.conj(e)                             # convention is e^{+iωt} — opposite: the waves would run against the wind and the foam)

    def eval(self, t, i=None):
        """Band fields at time t: {dx, dy, dz, hx, hy, xx, yy, xy} — displacement (m), height slopes, displacement derivatives
        (by the undisplaced tile coordinates, row — y). i — a single band."""
        out = []
        for j, b in enumerate(self.bands):
            if i is not None and j != i: continue
            c, lam, n2 = self.spectrum(b, t), self.chop, b.N * b.N
            X, Y = 1j * lam * b.ux * c, 1j * lam * b.uy * c
            Hx, Hy = 1j * b.kx * c, 1j * b.ky * c
            ki = np.where(b.k > 0, b.k, 1.0)
            Xx, Yy, Xy = -lam * b.kx * b.kx / ki * c, -lam * b.ky * b.ky / ki * c, -lam * b.kx * b.ky / ki * c
            f1, f2 = np.fft.ifft2(c + 1j * X) * n2, np.fft.ifft2(Y + 1j * Hx) * n2
            f3, f4 = np.fft.ifft2(Hy + 1j * Xx) * n2, np.fft.ifft2(Yy + 1j * Xy) * n2
            out.append(dict(dz=f1.real, dx=f1.imag, dy=f2.real, hx=f2.imag, hy=f3.real, xx=f3.imag, yy=f4.real, xy=f4.imag))
        return out if i is None else out[0]

    def _foam_setup(self):
        """Foam — only from waves longer than lmin and λp/lfrac (ripples break without entraining air — without foam), on an Nf² grid of
        the same tile: the band modes that fit on it; the step back in time and the wind drift are multipliers (no exp per step)."""
        f = self.fp; kf = 2 * math.pi / max(f["lmin"], self.lp / f["lfrac"])
        v = f["drift"] * self.u10; wx, wy = self.wind_to
        for b in self.bands:
            Nf = min(int(f["N"]), b.N); idx = np.r_[0:Nf // 2, b.N - Nf // 2:b.N]; ix = np.ix_(idx, idx)
            on = (b.k[ix] <= kf) & b.mask[ix]
            b.f_on = bool(on.any() and b.klo < kf)
            if not b.f_on: continue
            b.f_N, b.f_kx, b.f_ky = Nf, b.kx[ix], b.ky[ix]
            b.f_h0, b.f_hm, b.f_w = np.where(on, b.h0[ix], 0), np.where(on, b.hm[ix], 0), b.w[ix]
            s_ = np.exp(-1j * (b.f_kx * wx + b.f_ky * wy) * v * f["dt"])   # drift per step: the field at x − v·age
            b.f_ra, b.f_rb = np.exp(1j * b.f_w * f["dt"]) * s_, np.exp(-1j * b.f_w * f["dt"]) * s_   # a step back in time
            ki = np.where(b.k[ix] > 0, b.k[ix], 1.0)
            b.f_xx, b.f_yy, b.f_xy = -self.chop * b.f_kx ** 2 / ki, -self.chop * b.f_ky ** 2 / ki, -self.chop * b.f_kx * b.f_ky / ki

    def foam(self, t, i=None, thr=None):
        """Band foam 0…1 (an Nf² grid of the band tile; None — a band without foam) at time t: E = Σ injection(J(x − v·a, t − a))·e^{−a/τ},
        a = jΔ; foam = 1 − e^{−grow·Δ·E}; the injection is where the Jacobian is below the threshold thr (soft, width width)."""
        f, out = self.fp, []
        thr = f["thr"] if thr is None else thr
        M = int(math.ceil(2.5 * f["tau"] / f["dt"]))
        for j, b in enumerate(self.bands):
            if i is not None and j != i: continue
            if not b.f_on or thr is None or thr <= -9: out.append(None); continue
            e = np.exp(-1j * b.f_w * t); A, B = b.f_h0 * e, b.f_hm * np.conj(e)
            E, n2, dec = np.zeros((b.f_N, b.f_N)), b.f_N * b.f_N, math.exp(-f["dt"] / f["tau"])
            wgt = 1.0
            for _ in range(M + 1):
                c = A + B
                f1 = np.fft.ifft2(c * (b.f_xx + 1j * b.f_yy)) * n2; xy = (np.fft.ifft2(c * b.f_xy) * n2).real
                J = (1 + f1.real) * (1 + f1.imag) - xy * xy
                E += np.clip((thr - J) / f["width"], 0.0, 1.0) * wgt
                A *= b.f_ra; B *= b.f_rb; wgt *= dec
            out.append(1.0 - np.exp(-f["grow"] * f["dt"] * E))
        return out if i is None else out[0]

    def foam_old(self, t, tau=12.0, dt=1.0, grow=0.8):
        """Storm foam of two ages (an Nf² band grid): "now" — where a crest breaks at time t, and old — with a memory τ, s (where waves
        broke during the last ~2.5τ: foam streaks stretch from there). The step dt is larger than in foam: the crest trail in patches
        every c·dt is smoothed. → [(now, old) | None] per band."""
        from scipy import ndimage
        f, out = self.fp, []
        thr = f["thr"]; M = int(math.ceil(2.5 * tau / dt)); v = f["drift"] * self.u10; wx, wy = self.wind_to
        for b in self.bands:
            if not b.f_on or thr is None or thr <= -9: out.append(None); continue
            s_ = np.exp(-1j * (b.f_kx * wx + b.f_ky * wy) * v * dt)
            ra, rb = np.exp(1j * b.f_w * dt) * s_, np.exp(-1j * b.f_w * dt) * s_
            e = np.exp(-1j * b.f_w * t); A, B = b.f_h0 * e, b.f_hm * np.conj(e)
            E, n2, dec, wgt, now = np.zeros((b.f_N, b.f_N)), b.f_N * b.f_N, math.exp(-dt / tau), 1.0, None
            for j in range(M + 1):
                c = A + B
                f1 = np.fft.ifft2(c * (b.f_xx + 1j * b.f_yy)) * n2; xy = (np.fft.ifft2(c * b.f_xy) * n2).real
                inj = np.clip((thr - ((1 + f1.real) * (1 + f1.imag) - xy * xy)) / f["width"], 0.0, 1.0)
                if j == 0: now = inj
                E += inj * wgt; A *= ra; B *= rb; wgt *= dec
            cb = math.sqrt(G * b.L / 8.0 / (2 * math.pi))                 # stamps every c·dt (c for waves of ~L/4 of the band): smooth over them
            sg = max(2.0, 0.6 * cb * dt / (b.L / b.f_N))
            out.append((now, 1.0 - np.exp(-grow * dt * ndimage.gaussian_filter(E, sg, mode="wrap"))))
        return out

    def coverage(self, t, thr=None, n=12000, seed=0):
        """The fraction of the area where the foam of all bands together is > 0.5 (at n random points of the first band's tile)."""
        pts = np.random.default_rng(seed).uniform(0, self.bands[0].L, (n, 2)); keep = np.ones(n)
        for b, F in zip(self.bands, self.foam(t, thr=thr)):
            if F is not None: keep *= 1 - sample(F, b.L, pts)
        return float((1 - keep > 0.5).mean())

    def calibrate_foam(self, cover=None, times=(3.0, 11.0)):
        """The Jacobian threshold from the whitecap fraction: cover (default — Monahan by U10); if there is nothing to expect (< 0.01 %) or
        the foam does not build up even at a threshold of 0.9 — there is no foam (thr = −10)."""
        cover = min(MONAHAN[0] * self.u10 ** MONAHAN[1], 0.12) if cover is None else float(cover)   # in measurements it saturates at ~10–15 % toward 20+ m/s
        cov = lambda th: float(np.mean([self.coverage(t, th) for t in times]))
        if cover < 1e-4 or not any(b.f_on for b in self.bands) or cov(0.9) < cover * 0.5:
            self.fp["thr"] = -10.0; return self.fp["thr"]
        lo, hi = -1.0, 0.9
        for _ in range(10):
            th = (lo + hi) / 2
            lo, hi = (th, hi) if cov(th) < cover else (lo, th)
        self.fp["thr"] = (lo + hi) / 2
        return self.fp["thr"]

    # --- numbers ---
    def stats(self):
        """Hs (4√m0), peak period and length, the mean square slope of the bands (along and across the wind), the remainder up to
        Cox–Munk, the expected whitecap fraction after Monahan."""
        m0 = sum(float(b.psi.sum()) * b.dk ** 2 for b in self.bands)
        ca, sa = self.wind_to
        sl = []
        for b in self.bands:
            ka = b.kx * ca + b.ky * sa; kc = -b.kx * sa + b.ky * ca
            sl.append((float((b.psi * ka * ka).sum()) * b.dk ** 2, float((b.psi * kc * kc).sum()) * b.dk ** 2))
        res = sum(a + c for a, c in sl)
        cm = COX_MUNK * u_at(self.u10, 12.5)
        return {"hs": 4 * math.sqrt(m0), "tp": 2 * math.pi / self.wp, "lp": self.lp,
                "bands": [round(b.L, 3) for b in self.bands], "lam_min": [round(b.lam, 4) for b in self.bands],
                "slope2": [round(a + c, 6) for a, c in sl], "slope2_along": round(sum(a for a, _ in sl), 6),
                "slope2_cross": round(sum(c for _, c in sl), 6), "slope2_cox_munk": round(cm, 6),
                "slope2_rest": round(max(cm - res, 0.0), 6), "whitecaps": MONAHAN[0] * self.u10 ** MONAHAN[1]}


def sample(F, L, xy):
    """A tile field F (N×N, row — y) at points xy (M×2, m) — bilinear, wrapping around."""
    N = F.shape[0]; u = (np.asarray(xy, float) / L * N) % N
    i0, j0 = np.floor(u[:, 1]).astype(int), np.floor(u[:, 0]).astype(int); fy, fx = u[:, 1] - i0, u[:, 0] - j0
    i0, j0 = i0 % N, j0 % N                                             # (−1e-15) % N gives exactly N — an index past the edge
    i1, j1 = (i0 + 1) % N, (j0 + 1) % N
    return (F[i0, j0] * (1 - fx) * (1 - fy) + F[i0, j1] * fx * (1 - fy) + F[i1, j0] * (1 - fx) * fy + F[i1, j1] * fx * fy)


def surface_at(sea, fields, xy, iters=4):
    """Surface height and normal above the world points xy (M×2) from the eval(t) fields: the undisplaced point x0 is found by
    iterating x0 = p − D(x0) (Atlas); → (z, n)."""
    p = np.asarray(xy, float); x0 = p.copy()
    for _ in range(iters):
        D = np.zeros_like(p)
        for b, F in zip(sea.bands, fields): D += np.stack([sample(F["dx"], b.L, x0), sample(F["dy"], b.L, x0)], 1)
        x0 = p - D
    z, hx, hy, xx, yy = (sum(sample(F[k], b.L, x0) for b, F in zip(sea.bands, fields)) for k in ("dz", "hx", "hy", "xx", "yy"))
    n = np.stack([-hx / (1 + xx), -hy / (1 + yy), np.ones_like(z)], 1)
    return z, n / np.linalg.norm(n, axis=1, keepdims=True)


if __name__ == "__main__":
    kv = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    sw = dict(hs=float(kv["swell_hs"]), tp=float(kv.get("swell_tp", 10)), to=(1.0, 0.3)) if "swell_hs" in kv else None
    sea = Sea(u10=float(kv.get("u10", 8)), fetch=float(kv["fetch"]) if "fetch" in kv else None,
              depth=float(kv["depth"]) if "depth" in kv else None, swell=sw, near=float(kv.get("near", 2)))
    for k_, v_ in sea.stats().items(): print(f"{k_}: {v_}")


class Rain:
    """Rain rings: a drop is a Gaussian dimple (width s, depth amp), then linear waves with the same dispersion (capillary-gravity:
    the short rings run ahead), damping by viscosity and a surface film. Tile L×L, N×N; rate — drops per m² per second (light rain
    ~100, moderate ~300, downpour ~1000); the drops of a birth frame form one impulse map (one FFT per group) and live for life s.
    eval(t) → band fields (dz, hx, hy; no displacement)."""

    def __init__(self, rate=300.0, L=2.0, N=512, seed=1, life=1.2, amp=1.2e-3, s=0.004, fps=24.0, film=1.2):
        self.L, self.N, self.rate, self.life, self.fps, self.film = float(L), int(N), float(rate), float(life), float(fps), float(film)
        f = 2 * math.pi * np.fft.fftfreq(self.N, d=self.L / self.N)
        self.kx, self.ky = np.meshgrid(f, f); self.k = np.hypot(self.kx, self.ky)
        self.w = omega(np.maximum(self.k, 1e-6))
        kc = 0.45 * math.pi * self.N / self.L                           # cutoff near Nyquist: otherwise the grid shows a checkerboard
        self.A = (-amp * 2 * math.pi * s * s / self.L ** 2) * np.exp(-0.5 * (self.k * s) ** 2) * np.exp(-(self.k / kc) ** 4)   # Fourier series of the dimple
        self.seed = int(seed)
        self.lam, self.klo, self.khi = 4 * self.L / self.N, 2 * math.pi / self.L, math.pi * self.N / self.L / 2
        self._cache = {}

    def _group(self, j):
        """The spectrum of the drops of birth frame j (deterministic by the seed and the frame number)."""
        if j in self._cache: return self._cache[j]
        rng = np.random.default_rng((self.seed * 1000003 + j) % (2 ** 32))
        n = rng.poisson(self.rate * self.L * self.L / self.fps)
        I = np.zeros((self.N, self.N))
        if n:
            ij = rng.integers(0, self.N, (n, 2)); np.add.at(I, (ij[:, 0], ij[:, 1]), rng.uniform(0.5, 1.5, n))
        S = np.fft.fft2(I) * self.A
        if len(self._cache) > 400: self._cache.clear()
        self._cache[j] = S
        return S

    def eval(self, t):
        j1 = int(math.floor(t * self.fps)); j0 = j1 - int(math.ceil(self.life * self.fps))
        C = np.zeros((self.N, self.N), complex)
        for j in range(j0, j1 + 1):
            age = t - j / self.fps
            if age < 0: continue
            C += self._group(j) * (np.cos(self.w * age) * np.exp(-(2 * 1e-6 * self.k ** 2 + self.film) * age))
        n2 = self.N * self.N
        f1 = np.fft.ifft2(C + 1j * (1j * self.kx * C)) * n2; f2 = np.fft.ifft2(1j * self.ky * C) * n2
        z = np.zeros((self.N, self.N))
        return dict(dz=f1.real, hx=f1.imag, hy=f2.real, dx=z, dy=z, xx=z, yy=z, xy=z)


def caustics(F, L, sun=(0.0, 0.0, 1.0), depth=0.6, n_w=1.333, ss=2, blur=1.0):
    """Caustics on a flat bed of depth depth under a band tile (the fields F = eval of one band, a tile L×L): the sun (toward the
    sun — sun) is refracted by the surface slopes (Snell), the rays land on the bed; a histogram of hits, mean 1 (a calm surface —
    exactly 1). ss — rays per point along a side (the slopes are subsampled bilinearly); → an N×N map of the same tile."""
    N = F["hx"].shape[0]
    c = (np.arange(N * ss) + 0.5) / ss - 0.5
    X, Y = np.meshgrid(c, c)
    pts = np.stack([X.ravel(), Y.ravel()], 1) * (L / N)
    hx, hy = sample(F["hx"], L, pts), sample(F["hy"], L, pts)
    nrm = np.stack([-hx, -hy, np.ones_like(hx)], 1); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    I = -np.asarray(sun, float) / np.linalg.norm(sun)                   # incident ray (downward)
    eta = 1.0 / n_w; ci = -(nrm @ I)
    k = 1 - eta * eta * (1 - ci * ci)
    T = eta * I[None, :] + (eta * ci - np.sqrt(np.maximum(k, 0)))[:, None] * nrm   # refracted ray
    t = depth / np.maximum(-T[:, 2], 1e-3)
    xb, yb = pts[:, 0] + T[:, 0] * t, pts[:, 1] + T[:, 1] * t
    u, v = (xb / L * N) % N, (yb / L * N) % N
    i0, j0 = np.floor(v).astype(int), np.floor(u).astype(int); fy, fx = v - i0, u - j0
    H = np.zeros((N, N))
    for di, dj, w in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)), (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
        np.add.at(H, ((i0 + di) % N, (j0 + dj) % N), w)
    if blur > 0:                                                        # a light wrap-around blur (hit noise)
        kf = 2 * math.pi * np.fft.fftfreq(N); KX, KY = np.meshgrid(kf, kf)
        H = np.real(np.fft.ifft2(np.fft.fft2(H) * np.exp(-0.5 * blur * blur * (KX * KX + KY * KY))))
    return H / max(H.mean(), 1e-9)
