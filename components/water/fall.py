"""Waterfalls: all kinds use one model of a jet over a rock (the numbers of the kinds are in falls.yaml).
The lip: the flow is critical — depth h_c = (q²/g)^{1/3}, speed v_c = √(g·h_c) (q is the discharge per meter, m²/s); before the
lip there is an approach (the level drops, the water accelerates: a smooth "glassy" arc). Then the jet flies (air drag grows with
breakup: a solid jet has a speed limit of ~40 m/s, strands and clumps ~18 m/s) — or, hitting the rock, slides along it (along
the slope g·sin α minus friction, a limit of ~6 m/s at 70°) and detaches where the rock falls away from under the jet. Contact
comes from the shape of the rock, not from the kind: under a niche the jet is free, on a slope with a convex lip it slides.
Breakup: by the path L_b ≈ 1 + 4√q m the jet tears into strands and drops and turns white; on the rock — three times faster.
Below is the plunge pool: impact along the line impact (foam, ripple rings, water mist ~0.35 of the height — bl_fall).
  Fall(kind).sheet(t) → jets: vertices (across × along × 3) and attributes; face(x, z) — the rock (y outward); summary() — the numbers."""
import math
import os

import numpy as np
import yaml

G = 9.81
HERE = os.path.dirname(os.path.abspath(__file__))
FALLS = yaml.safe_load(open(os.path.join(HERE, "falls.yaml"), encoding="utf-8"))


def _ss(x, lo, hi):
    t = np.clip((np.asarray(x, float) - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _vn(x, y, seed=0):
    """Value noise on a lattice, 0…1, smooth."""
    xi, yi = np.floor(x), np.floor(y); fx, fy = x - xi, y - yi
    h = lambda i, j: (np.sin(i * 127.1 + j * 311.7 + seed * 74.7) * 43758.5453) % 1.0
    u, v = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    return (h(xi, yi) * (1 - u) + h(xi + 1, yi) * u) * (1 - v) + (h(xi, yi + 1) * (1 - u) + h(xi + 1, yi + 1) * u) * v


def fbm(x, y, seed=0, oct_=4):
    s, w, tot = 0.0, 1.0, 0.0
    for k in range(oct_):
        s = s + w * _vn(x * 2 ** k, y * 2 ** k, seed + 13 * k); tot += w; w *= 0.5
    return s / tot


class Fall:
    """A waterfall of the kind kind (falls.yaml) with the overrides over; the lip runs along the x axis (−width/2…width/2) at the height
    z_lip, the water flows toward +y, the rock is y = face(x, z) (zero at the lip). The jets follow the lip fractions segments."""

    def __init__(self, kind="plunge", z_lip=None, seed=1, nu=48, nt=120, x0=0.0, y0=0.0, approach=3.0, white_in=False, **over):
        p = dict(FALLS[kind]); p.update(over); self.kind, self.p = kind, p
        self.H, self.W, self.q = float(p["height"]), float(p["width"]), float(p["q"])
        self.z_lip = float(self.H if z_lip is None else z_lip); self.z_pool = self.z_lip - self.H
        self.x0, self.y0 = float(x0), float(y0)                         # the lip in the world: the middle at (x0, y0), the water flows toward +y
        self.hc = (self.q ** 2 / G) ** (1 / 3); self.vc = math.sqrt(G * self.hc)
        self.Lb = 1.0 + 4.0 * math.sqrt(self.q); self.seed = int(seed); self.rough = float(p.get("rough", 0.4))
        self.spread = float(p.get("spread", 0.1)); self.approach, self.white_in = float(approach), bool(white_in)
        self.segs = [(self.x0 - self.W / 2 + a * self.W, self.x0 - self.W / 2 + b * self.W) for a, b in (p.get("segments") or [[0.0, 1.0]])]
        self.nu, self.nt = int(nu), int(nt)
        self._trace()

    # --- rock ---
    def face(self, x, z):
        """The rock under the lip: the y of the surface (outward is +) at world points (x, z); the roughness rough — noise of ~2 m."""
        x, z = np.asarray(x, float), np.asarray(z, float); d = np.clip(self.z_lip - z, 0, None); p = self.p
        if p["face"] == "slope":                                       # convex lip: the rock follows the flight parabola, then a slope
            base = np.maximum(d / math.tan(math.radians(float(p["angle"]))), 0.98 * self.vc * np.sqrt(2 * d / G))
        else:                                                           # niche: the rock under the lip recedes inward
            base = -float(p.get("recess", 0.1)) * self.H * _ss(d, 0.3, 0.25 * self.H + 1.0)
        r_, sd = self.rough, self.seed                                  # rock relief: blocks of ~6 m, protrusions of ~2 m, columns along the plumb, strata
        big = (fbm(x / 6.0 + 3.1 * sd, z / 6.0, sd) - 0.5) * 2.0 * 1.4 * r_
        mid = (fbm(x / 2.0 + 7.3 * sd, z / 2.0, sd + 5) - 0.5) * 2.0 * 0.6 * r_
        col = 0.35 * r_ * np.abs(np.sin(x / 0.95 + 2.5 * fbm(x / 4.0, z / 9.0, sd + 9)))
        strata = 0.3 * r_ * _ss(np.mod(z / 1.7 + 0.6 * fbm(x / 5.0, z / 5.0, sd + 13), 1.0), 0.7, 0.95)
        return self.y0 + base + (big + mid - col + strata) * _ss(d, 0.2, 1.2)

    # --- jet path ---
    def _trace(self, dt=0.008):
        """The path of each lip point down to the pool: flight with drag, sliding along the rock (contact — where the jet would go into the stone or runs closer than cling to a convex rock)."""
        X = np.concatenate([np.linspace(a, b, self.nu) for a, b in self.segs]); n = len(X)
        y = np.full(n, self.y0, float); z = np.full(n, self.z_lip + 0.715 * self.hc)   # at the lip the depth is ~0.715·h_c
        vy, vz = np.full(n, self.vc), np.zeros(n); s = np.zeros(n); aer = np.zeros(n)
        hist = []; t = 0.0; tmax = 3.0 * math.sqrt(2 * self.H / G) + 2.0; cf = G * math.sin(math.radians(70)) / 36.0
        cling = 0.35 if self.p["face"] == "slope" else 0.0              # a thin jet clings to a convex rock (adhesion)
        while t < tmax:
            vt = 40.0 - 22.0 * np.clip(aer, 0, 1); k = G / vt ** 2         # limit: a solid jet ~40 m/s, strands and clumps ~18
            sp = np.hypot(vy, vz)
            vy = vy - k * sp * vy * dt; vz = vz - (G + k * sp * vz) * dt
            yn, zn = y + vy * dt, z + vz * dt
            th = np.clip(self.q / np.maximum(sp, 0.3), 0.02, 0.5)
            yf = self.face(X, zn) + 0.5 * th
            con = (yn < yf + cling) & (zn < self.z_lip - 0.05)
            if con.any():                                               # sliding: along the rock, g·sin α − friction
                dF = (self.face(X[con], zn[con] + 0.05) - self.face(X[con], zn[con] - 0.05)) / 0.1
                nn = np.sqrt(1 + dF * dF); ty, tz = -dF / nn, -1.0 / nn
                v_t = np.maximum(vy[con] * ty + vz[con] * tz, 0.5)
                v_t = v_t + (G / nn - cf * v_t * v_t) * dt
                vy[con], vz[con] = v_t * ty, v_t * tz; yn[con] = yf[con]
            ds = np.hypot(yn - y, zn - z); s += ds; aer += ds / self.Lb * np.where(con, 3.0, 1.0)
            y, z = yn, zn; t += dt
            hist.append((t, y.copy(), z.copy(), np.hypot(vy, vz), aer.copy(), con.copy()))
            if (z < self.z_pool - 0.4).all(): break
        T = np.array([h[0] for h in hist]); Y = np.array([h[1] for h in hist]); Z = np.array([h[2] for h in hist])
        V = np.array([h[3] for h in hist]); A = np.array([h[4] for h in hist]); C = np.array([h[5] for h in hist], float)
        # approach: ts — up to the lip (−T_a…0), the level drops to 0.715·h_c, the speed rises to v_c
        na = max(self.nt // 6, 4); Ta = self.approach / (0.75 * self.vc)
        out = {k: np.zeros((n, na + self.nt)) for k in ("t", "y", "z", "v", "b", "c")}
        for i in range(n):                                              # each little jet has its own path length to the pool
            zi = Z[:, i]; end = int(np.argmax(zi < self.z_pool - 0.35)) if (zi < self.z_pool - 0.35).any() else len(T) - 1
            tk = np.linspace(0.0, T[end], self.nt)
            ta = np.linspace(-Ta, 0.0, na, endpoint=False); fa = (ta + Ta) / Ta
            out["t"][i] = np.r_[ta, tk]
            out["y"][i] = np.r_[self.y0 - self.approach * (1 - fa) ** 1.3, np.interp(tk, T, Y[:, i])]
            out["z"][i] = np.r_[self.z_lip + self.hc * (1 - 0.285 * fa ** 2), np.interp(tk, T, zi)]
            out["v"][i] = np.r_[self.vc * (0.5 + 0.5 * fa), np.interp(tk, T, V[:, i])]
            out["b"][i] = np.r_[np.full(na, 1.0 if self.white_in else 0.0), np.clip(np.interp(tk, T, A[:, i]), 0, 1) ** 0.7]
            out["c"][i] = np.r_[np.zeros(na), np.interp(tk, T, C[:, i])]
        self.X, self.path = X, out
        zc = self.z_pool
        self.impact = np.stack([X, [np.interp(zc, out["z"][i][::-1], out["y"][i][::-1]) for i in range(n)], np.full(n, zc)], 1)

    def sheet(self, t):
        """Jets at time t: [{P (nu × nt′ × 3), tau, b, v, u (0…1 across), um (m along the lip), c — contact}] per jet. A free jet flutters
        (noise by detachment time: the drift outward and sideways grows toward the bottom) and spreads toward the bottom by spread of the width."""
        out = []; P = self.path
        for k, (a, b_) in enumerate(self.segs):
            sl = slice(k * self.nu, (k + 1) * self.nu); X = self.X[sl][:, None]; xc = 0.5 * (a + b_)
            y, z, tau = P["y"][sl], P["z"][sl], P["t"][sl]
            fall_ = np.clip((self.z_lip - z) / self.H, 0, 1)
            x = X + self.spread * (X - xc) * fall_ ** 0.8                 # spreading toward the bottom
            u = np.linspace(0, 1, self.nu)[:, None] * np.ones_like(z)
            w = (fbm(u * 3.0 + 7 * k, (t - tau) * 1.3 + 11 * k, self.seed + k) - 0.5) * 2.0
            wx = (fbm(u * 2.0 + 5 * k, (t - tau) * 0.9 + 3 * k, self.seed + 40 + k) - 0.5) * 2.0
            free = 1 - P["c"][sl]; amp = 0.025 * self.H * fall_ ** 1.5 * free
            P3 = np.stack([x + 0.5 * amp * wx, y + amp * w, z], -1)
            out.append({"P": P3, "tau": tau, "b": P["b"][sl], "v": P["v"][sl], "u": u, "um": x - (self.x0 - self.W / 2), "c": P["c"][sl]})
        return out

    def summary(self):
        P = self.path; i0 = self.path["t"].shape[1] - self.nt
        tf = float(np.mean(P["t"][:, -1])); vi = float(np.mean(P["v"][:, -1]))
        return {"kind": self.kind, "h_c": round(self.hc, 3), "v_c": round(self.vc, 2), "fall_s": round(tf, 2), "v_impact": round(vi, 1),
                "throw_m": round(float(np.mean(self.impact[:, 1] - self.y0)), 2), "contact": round(float(P["c"][:, i0:].mean()), 2),
                "breakup_m": round(self.Lb, 2), "free_fall_s": round(math.sqrt(2 * self.H / G), 2)}


if __name__ == "__main__":                                               # the numbers of the kinds
    for k in FALLS:
        if FALLS[k].get("face") == "steps": continue
        print(Fall(k).summary())
