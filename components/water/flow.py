"""River current: the velocity field along the channel axis — v0 at midstream, falling off toward the banks (profile √(1 − (n/b)²)),
the direction is tangent to the axis; the midstream speed comes from Manning's formula by slope and depth (bodies.py) or is given;
behind stones — a foam trail and ripples (a wake downstream), along the banks — a foam strip. The field on the world grid → an
image for the material: advecting the ripples by the current in two phases offset by half a cycle,
phase noise against "pulsing".
Torrent: the level along the axis — slope and steps (profile); after a step the jet accelerates to √(v² + 2gh) and is damped by a
hydraulic jump (Bélanger's conjugate depth d₂ = d₁/2·(√(1 + 8Fr₁²) − 1), the roller ~6·d₂ — white water and a hump), after the
roller — a tail of standing waves; in fast water (Froude > 0.85) — standing waves λ = 2πv²/g (the phase stands still: the wave
runs upstream at the flow speed), crests steeper than 0.07 are white ("haystacks"); at a stone — a pillow (stagnation head v²/2g),
a pit behind it and a V-wake at the Mach angle (sin θ = 1/Fr). The field breathes over time.
The waterfall pool (Pool): impact along a line — the water spreads from it and is carried to the outlet, foam at the impact and
trails, ripple rings run from the impact, at the impact the water boils in a hump — fields of the same kind as the river's.
  River(path, width, v0, bounds, …).field → {vx, vy, speed, foam}; at(xy) — at points; eval(t) → {h, hx, hy, white}; level_at(x, y)."""
import math

import numpy as np

G = 9.81


def _ss(x, lo, hi):
    t = np.clip((np.asarray(x, float) - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _dense(path, step):
    P = np.asarray(path, float)[:, :2]
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(int(np.linalg.norm(b - a) / step), 1); out += [a + (b - a) * (i / n) for i in range(1, n + 1)]
    return np.array(out)


def manning(depth, slope, n=0.035):
    """Speed by Manning (a wide channel, R ≈ depth), m/s."""
    return depth ** (2 / 3) * math.sqrt(max(slope, 1e-6)) / n


def throw(h, v):
    """The jet's reach from a step of height h at speed v: horizontal, m."""
    return v * math.sqrt(2 * max(h, 0.0) / G)


def profile(s, slope=0.0, drops=(), z0=0.0, v=2.0):
    """Water level along the axis, m: z0 − slope·s and steps (s_d, h): past the lip the level drops by h over the length of the jet's flight."""
    s = np.asarray(s, float); z = z0 - slope * s
    for sd, h in drops:
        ln = max(0.5, throw(h, v) * 1.3)
        z = z - h * _ss(s, sd - 0.15 * ln, sd + ln)
    return z


def jump(v0, depth, h):
    """A step h on a flow (v0, depth): the jet speed at the foot, the depths before and after the jump, Froude, the roller length."""
    q = v0 * depth; v1 = math.sqrt(v0 * v0 + 2 * G * h); d1 = q / v1; fr1 = v1 / math.sqrt(G * d1)
    d2 = d1 / 2 * (math.sqrt(1 + 8 * fr1 * fr1) - 1)
    return {"v1": v1, "d1": d1, "d2": d2, "fr1": fr1, "roll": 4.0 * d2}


class River:
    """Channel: path — the axis (x, y points), width — the width at the water, m; v0 — midstream speed, m/s (None — by Manning from
    slope, depth, n); rocks — stones in the flow [(x, y, r)]; bounds — the field extent; res — step, m. Torrent: slope — the level
    slope, m/m; drops — steps [(s, height)] along the axis length from its start; depth — depth at midstream; z0 — the level at the
    start of the axis."""

    def __init__(self, path, width, v0, bounds, res=0.25, rocks=(), bank_foam=0.35, wake=6.0, slope=0.0, drops=(), depth=1.0,
                 z0=0.0, n=0.035, seed=1):
        self.res = float(res); x0, y0, x1, y1 = bounds; self.x0, self.y0 = float(x0), float(y0)
        self.nx, self.ny = int(round((x1 - x0) / res)) + 1, int(round((y1 - y0) / res)) + 1
        X, Y = np.meshgrid(self.x0 + np.arange(self.nx) * res, self.y0 + np.arange(self.ny) * res)
        P = _dense(path, 0.5); T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
        sP = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        from scipy.spatial import cKDTree
        tree = cKDTree(P); dist, j = tree.query(np.stack([X.ravel(), Y.ravel()], 1))
        dist, j = dist.reshape(X.shape), j.reshape(X.shape)
        t = T[j]; nvec = np.stack([-t[..., 1], t[..., 0]], -1)
        side = np.sign(((np.stack([X, Y], -1) - P[j]) * nvec).sum(-1))
        b = width / 2.0; q = np.clip(dist / b, 0, 1)
        prof = np.sqrt(np.clip(1 - q * q, 0, 1)) * (dist < b)
        if v0 is None: v0 = manning(depth, slope, n)
        self.drops = [(float(sd), float(h)) for sd, h in drops]
        self.slope, self.depth, self.z0 = float(slope), float(depth), float(z0)
        vP = np.full(len(P), float(v0)); self.jumps = []
        for sd, h in self.drops:                                        # the jet from a step is faster — to the end of the roller
            J = jump(v0, depth, h); J.update(s=sd, h=h, toe=sd + throw(h, v0)); self.jumps.append(J)
            vP += (J["v1"] - v0) * _ss(sP, sd - 0.3, sd + 0.4) * (1 - _ss(sP, J["toe"], J["toe"] + J["roll"]))
        sp = vP[j] * prof
        self.s, self.offset = sP[j], side * dist                         # along the axis and across (the sign is the bank), m
        self.level = profile(self.s, slope, self.drops, z0, v0); self.wet = dist < b + 1.0
        foam = bank_foam * np.exp(-((b - dist) / 0.8) ** 2) * (dist < b)   # along the banks — a foam strip
        vx, vy = sp * t[..., 0], sp * t[..., 1]
        dl = np.maximum(depth * np.clip(1 - q * q, 0.08, 1), 0.05)       # depth across: a bowl
        fr = sp / np.sqrt(G * dl); self.froude = fr
        rng = np.random.default_rng(seed)
        H0 = np.zeros(X.shape); Hb = np.zeros(X.shape); white = np.zeros(X.shape)   # standing / breathing hump; white water
        self.rapid = bool(slope > 0.005 or self.drops or v0 / math.sqrt(G * depth) > 0.8)
        if self.rapid:
            kP = G / np.maximum(vP, 0.4) ** 2                            # standing wave: k = g/v²
            phiP = np.r_[0.0, np.cumsum(0.5 * (kP[1:] + kP[:-1]) * np.diff(sP))]
            ox, oy = rng.uniform(0, 100, 2)
            bend = 0.9 * np.sin(self.offset / (0.45 * b) + ox) + 0.5 * np.sin(self.s / 9.0 + oy)   # crests are curved, the "haystacks" scattered
            ph = phiP[j] + bend
            cross = 0.6 + 0.4 * np.cos(math.pi * self.offset / (0.55 * b) + 0.7 * np.sin(self.s / 13.0 + ox))
            A = 0.22 * dl * _ss(fr, 0.85, 1.4) * (0.55 + 0.45 * np.sin(self.s / 7.3 + oy) * np.cos(self.offset / 3.1 + ox)) * prof
            ak = A * (G / np.maximum(sp, 0.4) ** 2)                      # steepness of the standing wave
            Hb += A * np.cos(ph) * cross
            white = np.maximum(white, _ss(ak, 0.025, 0.065) * _ss(np.cos(ph), 0.35, 0.9) * cross)
            for J in self.jumps:                                        # jump: a roller at the foot — white water and a hump, then a tail of waves
                p1, p2 = rng.uniform(0, 2 * math.pi, 2)                 # the roller's foot is uneven across the width (±1 m)
                toe = J["toe"] + 0.7 * np.sin(self.offset / 1.7 + p1) + 0.4 * np.sin(self.offset / 0.8 + p2)
                u = self.s - toe; Lr = J["roll"]
                rise = _ss(u, -0.4, 0.8); roll = rise * (1 - _ss(u, 0.35 * Lr, Lr)) * (dist < b)
                H0 += 0.35 * (J["d2"] - J["d1"]) * roll * prof
                white = np.maximum(white, rise * (1 - _ss(u, 0.6 * Lr, 1.3 * Lr)) * (0.75 + 0.25 * np.sin(self.offset / 1.3 + p2)) * prof ** 0.3)
                lt = 2 * math.pi * v0 * v0 / G; tail = _ss(u, 0.8 * Lr, 1.2 * Lr) * np.exp(-(u - Lr).clip(0) / (3 * lt))
                Hb += 0.3 * (J["d2"] - J["d1"]) * tail * np.cos(2 * math.pi * (u - Lr) / lt + 0.6 * bend) * prof
                white = np.maximum(white, 0.6 * _ss(u, Lr, 1.3 * Lr) * np.exp(-(u - 1.3 * Lr).clip(0) / 3.0) * prof)   # the roller foam is carried away
        for (rx, ry, rr) in rocks:                                      # stone: flow around it (sideways) and a foam wake downstream
            dx, dy = X - rx, Y - ry; d = np.hypot(dx, dy)
            jr = tree.query([rx, ry])[1]; tr = T[jr]
            along = dx * tr[0] + dy * tr[1]; across = -dx * tr[1] + dy * tr[0]
            inside = d < rr
            vx = np.where(inside, 0.0, vx); vy = np.where(inside, 0.0, vy)
            near = np.exp(-((d - rr) / (0.6 * rr + 0.2)) ** 2) * (d >= rr)
            sg = np.sign(across); vx += near * 0.35 * v0 * sg * -tr[1]; vy += near * 0.35 * v0 * sg * tr[0]   # flow around sideways
            wk = np.exp(-(across / (rr * (1 + along.clip(0) / (2 * wake)))) ** 2) * np.exp(-along.clip(0) / wake) * (along > -rr * 0.5)
            foam = np.maximum(foam, np.clip(0.95 * wk + 0.9 * near, 0, 1) * (d >= rr * 0.9))
            if self.rapid:
                vr = float(vP[jr]); dh = vr * vr / (2 * G); frr = vr / math.sqrt(G * depth)
                up = 1 - _ss(along, -0.4 * rr, 0.5 * rr)                 # the pillow is on the upstream side of the stone, smoothly
                H0 += 0.8 * dh * np.exp(-((d - rr) / (0.6 * rr + 0.2)) ** 2) * up * prof
                H0 -= 0.6 * dh * np.exp(-((along - 1.3 * rr) / rr) ** 2 - (across / (0.8 * rr)) ** 2) * (along > 0) * prof   # pit
                if frr > 1.0:                                           # V-wake: the Mach angle
                    tn = math.tan(math.asin(1.0 / frr)); w_ = 0.25 + 0.08 * along.clip(0)
                    Hb += 0.35 * dh * np.exp(-((np.abs(across) - rr - along.clip(0) * tn) / w_) ** 2) * np.exp(-along.clip(0) / (8 * rr)) * (along > 0) * prof
                white = np.maximum(white, _ss(vr, 1.2, 2.5) * np.exp(-((d - rr) / (0.5 * rr + 0.15)) ** 2) * (0.5 + 0.5 * up))   # foaming breaker around the stone
                white = np.maximum(white, 0.85 * _ss(vr, 1.2, 2.5) * np.exp(-((along - 1.2 * rr) / (1.2 * rr)) ** 2 - (across / (0.9 * rr)) ** 2) * (along > 0))
        self.vx, self.vy = vx, vy; self.speed = np.hypot(vx, vy); self.foam = np.clip(foam, 0, 1)
        self.v0, self.width = float(v0), float(width)
        self.H0, self.Hb, self.white = H0, Hb, np.clip(white, 0, 1)
        self.breath = rng.uniform(0, 2 * math.pi) + 0.8 * np.sin(X / 5.3) * np.cos(Y / 4.7)   # phase of the wave breathing by place

    def at(self, xy):
        """Velocity and foam at points xy (M×2): (vx, vy, speed, foam)."""
        from components.water.shore import _bilinear
        return tuple(_bilinear(F, self.x0, self.y0, self.res, xy[:, 0], xy[:, 1]) for F in (self.vx, self.vy, self.speed, self.foam))

    def eval(self, t):
        """The torrent surface at time t: h — above the axis level, m; hx, hy — slopes; white — white water 0…1. Standing waves breathe
        (±15 %, ~7 s), the jump roller sways (±10 %)."""
        h = self.H0 * (1 + 0.1 * np.sin(1.7 * t + self.breath)) + self.Hb * (1 + 0.15 * np.sin(0.9 * t + self.breath))
        hy, hx = np.gradient(h, self.res)
        return {"h": h, "hx": hx, "hy": hy, "white": self.white}

    def level_at(self, x, y):
        """Water level at points (x, y): along the axis — slope and steps; farther than 1 m from the waterline — NaN (not water)."""
        from components.water.shore import _bilinear
        x, y = np.asarray(x, float), np.asarray(y, float)
        lv = _bilinear(self.level, self.x0, self.y0, self.res, x, y)
        wet = _bilinear(self.wet.astype(float), self.x0, self.y0, self.res, x, y) > 0.5
        inside = (x >= self.x0) & (x <= self.x0 + (self.nx - 1) * self.res) & (y >= self.y0) & (y <= self.y0 + (self.ny - 1) * self.res)
        return np.where(wet & inside, lv, np.nan)

    def summary(self):
        fr = self.froude[self.froude > 0]
        return {"v0": round(self.v0, 2), "froude_max": round(float(fr.max()) if fr.size else 0.0, 2),
                "jumps": [{k: round(v, 2) for k, v in J.items()} for J in self.jumps],
                "white_share": round(float((self.white > 0.5).sum() / max((self.speed > 0).sum(), 1)), 3)}


class Pool:
    """The pool under a waterfall: impact along the line impact [(x, y)] at the speed v_imp, m/s; on top the water spreads from the line
    (~0.25·v_imp at the impact, decaying over ~0.3·radius) and is carried to the outlet out (m/s); the foam is solid at the impact,
    farther out — trails; ripple rings (λ ~0.8 m) run from the impact by the dispersion ω² = gk; at the impact the water boils in a
    hump (white water ~1.8 m). The fields are as in River (the water material)."""

    def __init__(self, impact, v_imp, bounds, res=0.2, radius=10.0, out=(0.0, 0.3), seed=1):
        from scipy.spatial import cKDTree
        self.res = float(res); x0, y0, x1, y1 = bounds; self.x0, self.y0 = float(x0), float(y0)
        self.nx, self.ny = int(round((x1 - x0) / res)) + 1, int(round((y1 - y0) / res)) + 1
        X, Y = np.meshgrid(self.x0 + np.arange(self.nx) * res, self.y0 + np.arange(self.ny) * res)
        I = _dense(np.asarray(impact, float)[:, :2], 0.2); d, j = cKDTree(I).query(np.stack([X.ravel(), Y.ravel()], 1))
        d, j = d.reshape(X.shape), j.reshape(X.shape); dd = np.maximum(d, 1e-3)
        ux, uy = (X - I[j, 0]) / dd, (Y - I[j, 1]) / dd                 # from the impact line
        R = float(radius); sp = 0.25 * v_imp * np.exp(-d / (0.3 * R))
        self.vx, self.vy = ux * sp + out[0], uy * sp + out[1]; self.speed = np.hypot(self.vx, self.vy)
        rng = np.random.default_rng(seed); a, b_ = rng.uniform(0, 2 * math.pi, 2)
        streak = 0.5 + 0.5 * np.sin(np.arctan2(uy, ux) * 9.0 + a + 0.4 * np.sin(d / 1.7 + b_))   # trails — as rays from the impact
        self.foam = np.clip(np.exp(-d / (0.22 * R)) + 0.5 * np.exp(-d / (0.6 * R)) * streak, 0, 1)
        self.white = _ss(1.8 - d, 0.0, 1.8); self.H0 = 0.15 * np.exp(-(d / 1.2) ** 2); self.Hb = np.zeros(X.shape)
        self.dist, self.R, self.rapid = d, R, True
        self.breath = rng.uniform(0, 2 * math.pi) + 0.8 * np.sin(X / 2.3) * np.cos(Y / 2.9)
        self.v_imp, self.v0, self.width, self.drops, self.jumps = float(v_imp), 0.12 * v_imp + 0.3, 0.0, [], []   # v0 — the speed at the surface

    def at(self, xy):
        from components.water.shore import _bilinear
        return tuple(_bilinear(F, self.x0, self.y0, self.res, xy[:, 0], xy[:, 1]) for F in (self.vx, self.vy, self.speed, self.foam))

    def eval(self, t):
        k = 2 * math.pi / 0.8; w = math.sqrt(G * k)
        ring = 0.025 * np.exp(-self.dist / (0.45 * self.R)) * _ss(self.dist, 0.5, 2.0) * np.sin(k * self.dist - w * t + self.breath)
        h = self.H0 * (1 + 0.25 * np.sin(4.0 * t + self.breath)) + ring
        hy, hx = np.gradient(h, self.res)
        return {"h": h, "hx": hx, "hy": hy, "white": self.white}
