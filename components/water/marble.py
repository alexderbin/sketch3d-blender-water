"""Marble foam: the pattern of the old foam of a storm. Foam after breaking tears into lace (a network of threads around holes), the
surface current stretches the lace along the wind into veins and gathers it into streaks. The current: jets along the wind,
varying across (the shear is the main stretching), weak 2D eddies, changing over time; Langmuir rollers (convergence every λ, two
scales, the rollers meander). A backward characteristics map: a tile point → where its water came from (the paint is not
smeared), the thickening at convergence — by the divergence along the path. The tile is periodic; the x axis runs along the wind,
y across.
  marble(L, n, seed) → (n, n, 2): young (stretched T/3) and old (T) foam 0…1."""
import math

import numpy as np
from scipy import ndimage


def _noise(n, L, size, rng):
    """Periodic noise n² of a tile L: white noise through a Gaussian window ~size, m; mean 0, σ 1."""
    k = np.fft.fftfreq(n, L / n) * 2 * math.pi; K2 = k[None, :] ** 2 + k[:, None] ** 2
    f = np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * np.exp(-K2 * (size / 2.0) ** 2 / 2)).real
    return (f - f.mean()) / (f.std() + 1e-12)


class Flow:
    """Surface current of a tile L on an m² grid: stream-function modes with the tile's wavenumbers; aniso < 1 — the modes are longer
    along the wind (jets); turb — the velocity σ, m/s; rows — rollers per tile (large, small), conv — the convergence speed toward a
    roller, m/s; jet — a jet along the wind at the convergence of the large rollers, m/s; meander — the meandering of the rollers, m."""

    def __init__(self, L, rng, m=128, modes=40, lam=(1.5, 16.0), aniso=0.08, turb=0.045, rows=(5, 16), conv=(0.04, 0.015),
                 jet=0.07, meander=0.5):
        self.L, self.m = float(L), int(m)
        kmin, kmax = 2 * math.pi / lam[1], 2 * math.pi / lam[0]; ks = []; amax = int(L / lam[0] * aniso) + 1
        while len(ks) < modes:                                          # integer wavenumbers of the tile (periodicity)
            a, b = int(rng.integers(-amax, amax + 1)), int(rng.integers(-int(L / lam[0]), int(L / lam[0]) + 1))
            kx, ky = 2 * math.pi * a / L, 2 * math.pi * b / L
            if kmin <= math.hypot(kx / aniso, ky) <= kmax and b != 0: ks.append((kx, ky))
        ks = np.array(ks); self.kx, self.ky = ks[:, 0], ks[:, 1]; k = np.hypot(self.kx / aniso, self.ky)
        A = k ** -2.0; A *= turb / math.sqrt(0.5 * float((A ** 2 * (self.kx ** 2 + self.ky ** 2)).sum()))
        self.A, self.ph = A, rng.uniform(0, 2 * math.pi, len(A))
        self.om = rng.uniform(0.03, 0.15, len(A)) * rng.choice([-1, 1], len(A))   # evolution of the jets and eddies, rad/s
        g = np.arange(self.m) * self.L / self.m; self.gx, self.gy = np.meshgrid(g, g)
        self.rows = [2 * math.pi * r / L for r in rows]; self.conv, self.jet = [float(c) for c in conv], float(jet)
        self.mk = [2 * math.pi * j / L for j in (1, 2, 3)]              # meandering of the rollers along the wind
        self.ma = [rng.uniform(0.3, 1.0, 3) * meander / (1 + i) for i in range(len(rows))]
        self.mp = [rng.uniform(0, 2 * math.pi, 3) for _ in rows]; self.mw = [rng.uniform(0.01, 0.04, 3) for _ in rows]

    def grid(self, t):
        """(u, v, div) on an m² grid at time t, m/s and 1/s."""
        X, Y = self.gx, self.gy; u = np.zeros_like(X); v = np.zeros_like(X); dv = np.zeros_like(X)
        for A, kx, ky, ph, om in zip(self.A, self.kx, self.ky, self.ph, self.om):
            c = A * np.cos(kx * X + ky * Y + ph + om * t); u += ky * c; v -= kx * c
        for i, (kr, B) in enumerate(zip(self.rows, self.conv)):
            eta = sum(a * np.sin(k * X + p + w * t) for a, k, p, w in zip(self.ma[i], self.mk, self.mp[i], self.mw[i]))
            s = kr * (Y - eta)
            v += B * np.sin(s); dv += B * kr * np.cos(s)                # convergence toward a roller at s = π (+2πj)
            if i == 0: u -= self.jet * np.cos(s)                        # at the convergence the water is faster downwind
        return u, v, dv


def _at(F, X, Y, L):
    """Bilinear lookup in the periodic grid F at points (X, Y), m."""
    m = F.shape[0]
    return ndimage.map_coordinates(F, [np.mod(Y / L * m, m).ravel(), np.mod(X / L * m, m).ravel()], order=1,
                                   mode="grid-wrap").reshape(X.shape)


def trace(flow, X, Y, T, dt):
    """Backward along the current from time T to 0: where the water of points (X, Y) came from; the thickening exp(−∫div). → (X0, Y0, thickening)."""
    S = np.zeros_like(X); t = T; L = flow.L
    while t > 1e-9:
        h = min(dt, t)
        u, v, _ = flow.grid(t)
        Xm, Ym = X - 0.5 * h * _at(u, X, Y, L), Y - 0.5 * h * _at(v, X, Y, L)   # step midpoint (RK2)
        u, v, dv = flow.grid(t - 0.5 * h)
        X, Y = X - h * _at(u, Xm, Ym, L), Y - h * _at(v, Xm, Ym, L)
        S += _at(dv, Xm, Ym, L) * h; t -= h
    return X, Y, np.exp(-S)


def lace(n, L, rng, cell=2.4, soft=0.08, patch=16.0, cover=0.7):
    """Foam lace: the network threads are the zero lines of smooth noise (cells ~cell, m), sharp (e^{−d/(soft·cell)}); each piece of
    thread has its own strength 0…1 (uniform, noise ~cell) and clusters (large noise, the fraction cover at full strength). The
    strength threshold in the material is the density: the lower the threshold, the more threads, and the thickness grows only
    logarithmically."""
    a = _noise(n, L, cell, rng); gy, gx = np.gradient(a, L / n); gr = np.hypot(gx, gy) + 1e-6
    net = np.exp(-(np.abs(a) / gr) / (soft * cell))                    # |a|/|∇a| — the distance to the zero line, m
    st = _noise(n, L, cell * 1.5, rng); st = np.argsort(np.argsort(st.ravel())).reshape(st.shape) / st.size   # strength — uniform
    p = _noise(n, L, patch, rng); thr = float(np.quantile(p, 1 - cover))
    return net * st * np.clip(0.6 + 0.4 * (p - thr) / 0.8, 0.3, 1)


def marble(L=48.0, n=512, seed=1, T=40.0, dt=4.0, **flow_kw):
    """A marble foam tile of L×L m, n²: the lace stretched by the current; → (n, n, 2) float32: [..., 0] — young (T/3), [..., 1] — old
    (T); 0…1 is the rank of the foam strength (thread × strength × thickening at the roller convergence): a threshold τ leaves the
    fraction 1 − τ of the tile."""
    rng = np.random.default_rng(seed)
    flow = Flow(L, rng, **flow_kw)
    dye = lace(512, L, rng)
    g = (np.arange(n) + 0.5) * L / n; X, Y = np.meshgrid(g, g); m = dye.shape[0]
    out = []
    for Tk in (T / 3.0, T):
        X0, Y0, conc = trace(flow, X, Y, Tk, dt)
        c = ndimage.map_coordinates(dye, [np.mod(Y0 / L * m, m).ravel(), np.mod(X0 / L * m, m).ravel()], order=3,
                                    mode="grid-wrap").reshape(X.shape)
        q = ndimage.gaussian_filter(np.clip(c, 0, 1) * conc ** 0.7, 0.8, mode="wrap").ravel()   # without grains of thin threads
        out.append((np.argsort(np.argsort(q)) / (q.size - 1)).reshape(X.shape))   # rank: the fraction of the tile above the threshold τ is exactly 1 − τ
    return np.stack(out, -1).astype(np.float32)


_CACHE = {}


def tile(seed=1, L=48.0, n=512):
    """The marble tile with a process-level memory (the scene's cameras share one tile; ~2 s for the first)."""
    key = (int(seed), float(L), int(n))
    if key not in _CACHE: _CACHE[key] = marble(L=L, n=n, seed=seed)
    return _CACHE[key]


if __name__ == "__main__":                                               # test: tile → image, time
    import sys
    import time
    t0 = time.time(); M = marble(seed=int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    print(f"marble {M.shape}: {time.time() - t0:.2f} s; young — fraction > 0.5: {(M[..., 0] > 0.5).mean():.3f}, "
          f"old: {(M[..., 1] > 0.5).mean():.3f}")
    out = sys.argv[1] if len(sys.argv) > 1 else "marble.pgm"
    img = np.concatenate([M[..., 0], M[..., 1]], 1)
    with open(out, "wb") as f:
        f.write(f"P5 {img.shape[1]} {img.shape[0]} 255\n".encode()); f.write((np.clip(img, 0, 1) * 255).astype(np.uint8).tobytes())
