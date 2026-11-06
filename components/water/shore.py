"""Surf: waves at the shore by the bed depth, a field on the world grid.
The phase is the eikonal |∇ψ| = k(d) from the waterline (refraction: the crests turn along the isobaths; Dijkstra on the grid —
scipy.sparse.csgraph); the height is shoaling by the energy flux H = H0·√(cg0/cg(d)) until breaking at H = γ·d (γ ≈ 0.78,
McCowan); the breaker type is given by the Iribarren number ξ = tan β/√(H0/L0) (spilling < 0.5, plunging ("tube") 0.5–3.3,
surging > 3.3); after breaking — a bore (a steep front, a gentle back) with a foam trail; swash on the beach — the run-up height
after Stockdon (2006), the tongue of water runs up and drains back. Three components with close periods form wave groups (sets).
Pure numpy + scipy (the project's Python and Blender).
  Shore(ground, level, sea, bounds).eval(t) → {h, hx, hy, foam} on the grid; at(xy, t) — at vertices; swash(xy, z_ground, t) — the swash."""
import math

import numpy as np

from components.water import waves as W

G = 9.81


def _ss(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _bilinear(F, x0, y0, res, x, y):
    ny, nx = F.shape
    u = np.clip((np.asarray(x) - x0) / res, 0, nx - 1.001); v = np.clip((np.asarray(y) - y0) / res, 0, ny - 1.001)
    i, j = v.astype(int), u.astype(int); fy, fx = v - i, u - j
    return (F[i, j] * (1 - fx) * (1 - fy) + F[i, j + 1] * fx * (1 - fy) + F[i + 1, j] * (1 - fx) * fy + F[i + 1, j + 1] * fx * fy)


class Shore:
    """Surf of the plot bounds (x0, y0, x1, y1) with step res, m. ground — the bed and the shore f(x, y) → z; sea — waves.Sea (where the
    period and height come from: the swell if there is one, otherwise the wind wave); comps — (period fraction, height fraction) of
    the components; gamma — the breaking index."""

    def __init__(self, ground, level, sea, bounds, res=0.4, seed=1, comps=((1.0, 1.0), (0.88, 0.55), (1.14, 0.45)), gamma=0.78,
                 foam_tau=3.0):
        from scipy import ndimage
        self.level, self.res, self.gamma, self.foam_tau = float(level), float(res), float(gamma), float(foam_tau)
        x0, y0, x1, y1 = bounds
        self.nx, self.ny = int(round((x1 - x0) / res)) + 1, int(round((y1 - y0) / res)) + 1
        self.x0, self.y0 = float(x0), float(y0)
        X, Y = np.meshgrid(self.x0 + np.arange(self.nx) * res, self.y0 + np.arange(self.ny) * res)
        self.X, self.Y = X, Y
        Z = ground(X, Y); self.Z = Z; d = self.level - Z; self.d = d
        wet = d > 0.01; self.wet = wet
        sw = [p for p in sea.parts if p["kind"] == "swell"]
        if sw:
            p = max(sw, key=lambda q: q.get("hs", 0)); T0, H0 = 2 * math.pi / p["wp"], float(p["hs"])
        else:
            st = sea.stats(); T0, H0 = st["tp"], st["hs"]
        self.T0, self.H0 = float(T0), 0.75 * float(H0)                  # the height of the surf's "average noticeable" wave
        self.L0 = G * self.T0 ** 2 / (2 * math.pi)
        self.s_land = ndimage.distance_transform_edt(~wet) * res        # on land — the distance to the water, m
        gy, gx = np.gradient(Z, res); slope = np.hypot(gx, gy)
        band = np.abs(d) < max(0.6, 0.5 * self.H0)
        self.tanb = max(float(np.median(slope[band])) if band.any() else 0.05, 0.005)
        self.xi = self.tanb / math.sqrt(max(self.H0, 0.05) / self.L0)
        self.kind = "spilling" if self.xi < 0.5 else "pipe" if self.xi < 3.3 else "surging"
        b = self.tanb; HL = self.H0 * self.L0                           # run-up R2 (Stockdon et al. 2006), m vertically
        self.R = 1.1 * (0.35 * b * math.sqrt(HL) + 0.5 * math.sqrt(HL * (0.563 * b * b + 0.004)))
        self.s_run = self.R / b                                         # horizontally
        self.wz = np.where(wet, 1 - _ss(d, 0.22 * self.L0, 0.45 * self.L0), 0.0)   # where the surf is (shallower than ~L0/2); farther out — the open sea
        edge = np.minimum.reduce([X - self.x0, (self.x0 + (self.nx - 1) * res) - X, Y - self.y0, (self.y0 + (self.ny - 1) * res) - Y])
        self.wz = self.wz * _ss(edge, 0.0, min(40.0, 0.15 * (self.nx - 1) * res))   # toward the grid edge — smoothly into the open sea (no seam)
        rng = np.random.default_rng(seed)
        nzx, nzy = rng.uniform(0, 100, 2)                               # crest bending along the shore: phase noise (40 m)
        self.noise = 0.9 * (np.sin(X / 37.0 + nzx) * np.cos(Y / 41.0 + nzy) + 0.5 * np.sin((X + Y) / 23.0 + nzy))
        self.comps = []
        dd = np.maximum(d, 0.03)
        for fT, fA in comps:
            T = self.T0 * fT; w = 2 * math.pi / T
            k = np.where(wet, W.wavenumber(np.full(dd.shape, w), dd), np.nan)
            cg = np.where(wet, W.domega_dk(np.nan_to_num(k, nan=1.0), dd), 1.0); cg0 = G / (2 * w)
            Hs = fA * self.H0 * np.sqrt(cg0 / np.maximum(cg, 1e-3))     # shoaling
            Hb = self.gamma * dd
            self.comps.append(dict(w=w, psi=self._eikonal(k), H=np.where(wet, np.minimum(Hs, Hb), 0.0),
                                   r=np.where(wet, np.clip(Hs / Hb, 0, 2), 0.0), ph=rng.uniform(0, 2 * math.pi), fA=fA))

    def _eikonal(self, k):
        """Phase from the waterline: a path over the wet cells with cost k (rad/m) — Dijkstra from the cells at land; → ψ, rad (land — 0)."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        wet = self.wet; ny, nx = wet.shape; idx = -np.ones(wet.shape, np.int64); idx[wet] = np.arange(wet.sum())
        rows, cols, wts = [], [], []
        def pair(a, dy, dx):                                            # (a cell, a neighbor with offset dy ≥ 0, dx)
            xa, xb = (slice(0, nx - dx), slice(dx, nx)) if dx >= 0 else (slice(-dx, nx), slice(0, nx + dx))
            return a[0:ny - dy, xa], a[dy:ny, xb]
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            wa, wb = pair(wet, dy, dx); ia, ib = pair(idx, dy, dx); ka, kb = pair(k, dy, dx)
            m = wa & wb; L = self.res * math.hypot(dy, dx)
            rows.append(ia[m]); cols.append(ib[m]); wts.append(0.5 * (ka[m] + kb[m]) * L)
        r, c, w_ = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
        n = int(wet.sum()); Gm = coo_matrix((np.r_[w_, w_], (np.r_[r, c], np.r_[c, r])), shape=(n, n)).tocsr()
        from scipy import ndimage
        shore = wet & ndimage.binary_dilation(~wet)                     # wet cells at the land are sources
        src = idx[shore]
        psi = np.zeros(wet.shape)
        if len(src):
            dist = dijkstra(Gm, directed=False, indices=src, min_only=True)
            psi[wet] = np.where(np.isfinite(dist), dist, 0.0)
            wm = wet.astype(float); sg = 2.0                            # smoothing: a path over 8 neighbors leaves a "comb" across the crests
            psi = np.where(wet, ndimage.gaussian_filter(psi * wm, sg) / np.maximum(ndimage.gaussian_filter(wm, sg), 1e-6), 0.0)
        return psi

    def eval(self, t):
        """The surf field at time t: h — water rise, m; hx, hy — slopes; foam — foam 0…1 (bore and trail)."""
        h = np.zeros(self.d.shape); foam = np.zeros(self.d.shape)
        for c in self.comps:
            th = c["w"] * t + c["psi"] + c["ph"] + self.noise           # the crests run toward the shore (ψ decreases toward the waterline)
            u = np.mod(th, 2 * math.pi)                                 # 0 — a crest has passed; toward 2π — the next one approaches
            r = np.clip(c["r"], 0, 1.5); br = _ss(r, 0.85, 1.05)         # breaking fraction
            kap = 1.0 + 4.0 * np.clip(r, 0, 1)                          # the crest is sharper at breaking (cnoidal)
            smooth = (np.exp(kap * (np.cos(u) - 1)) - np.i0(kap) * np.exp(-kap)) / (1 - np.exp(-2 * kap))   # mean 0, range 1
            front = 0.18                                                 # bore: a jump at the front (width ~0.18 rad), the back slopes down
            bore = np.where(u < 2 * math.pi - front, 1 - u / (2 * math.pi - front), (u - (2 * math.pi - front)) / front) - 0.5
            prof = (1 - br) * smooth + br * bore
            h += c["H"] * prof * self.wz
            if c["fA"] >= 0.99:                                         # foam comes from the main component
                foam = np.maximum(foam, br * self.wz * np.exp(-(u / c["w"]) / self.foam_tau) * _ss(u, 0.0, 0.08))
        hy, hx = np.gradient(h, self.res)
        return {"h": h, "hx": hx, "hy": hy, "foam": foam}

    def at(self, F, xy):
        """The frame field (eval) at points xy (M×2)."""
        return {k: _bilinear(v, self.x0, self.y0, self.res, xy[:, 0], xy[:, 1]) for k, v in F.items()}

    def swash(self, xy, t):
        """Swash at land points xy: after the bore the tongue of water runs up by s_run and drains within ~0.75 of a period. → (film
        thickness, m; foam 0…1) at each point (on water — 0)."""
        s = _bilinear(self.s_land, self.x0, self.y0, self.res, xy[:, 0], xy[:, 1])
        c = self.comps[0]
        nz = _bilinear(self.noise, self.x0, self.y0, self.res, xy[:, 0], xy[:, 1])
        u = np.mod(c["w"] * t + c["ph"] + nz, 2 * math.pi); tw = u / c["w"]; Tr = 0.75 * self.T0
        xf = self.s_run * np.sin(math.pi * np.clip(tw / Tr, 0, 1))      # tongue front: up and back
        on = (s > 0) & (s < xf)
        q = np.where(on, 1 - s / np.maximum(xf, 1e-3), 0.0)
        hfilm = np.where(on, 0.012 + 0.05 * q * (1 - tw / Tr).clip(0, 1), 0.0)
        fm = np.where(on, np.clip(0.35 + 0.65 * _ss(1 - q, 0.75, 1.0), 0, 1) * (1 - 0.6 * (tw / Tr).clip(0, 1)), 0.0)
        return hfilm, fm

    def summary(self):
        return {"T0": round(self.T0, 2), "H0": round(self.H0, 2), "L0": round(self.L0, 1), "tanb": round(self.tanb, 4),
                "xi": round(self.xi, 3), "breaker": self.kind, "runup_m": round(self.R, 2), "runup_dist_m": round(self.s_run, 1),
                "grid": [self.nx, self.ny]}
