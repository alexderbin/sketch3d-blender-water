# source: technique — Hans Theobald Beyer, "Implementation of a method for hydraulic erosion" (2015); numbers
#   — Sebastian Lague, Hydraulic-Erosion (MIT), Erosion.cs: inertia, capacity, deposit, erode, evaporate,
#   gravity, brush radius. Differences: droplets as a numpy batch (not one by one), heights in meters of a
#   grid of any step, speed by physics (v² += Δh·g downhill), 3 × 3 brush.
"""Relief erosion for components/site/terrain.py: water (droplets roll down the slope, erode and deposit — gullies, ridges,
alluvial fans) and scree (material steeper than the angle of repose slides down — trains at the feet). numpy only: both .venv
and Blender.
erode(H, res, amount=1.0, droplets=0.7, seed=0, protect=None) — H (m) is changed in place; amount — strength (0 — no erosion),
droplets — droplets per grid node; protect — a 0..1 mask where the relief is not to be touched (platforms, roads, water).
thermal(H, res, talus=35, iters=40) — scree: when the difference to a neighbor exceeds tan(talus)·step, half of the excess goes downhill."""
import math
import numpy as np


def _bil(H, x, y):
    ix, iy = x.astype(np.int64), y.astype(np.int64); fx, fy = x - ix, y - iy
    h00, h10, h01, h11 = H[iy, ix], H[iy, ix + 1], H[iy + 1, ix], H[iy + 1, ix + 1]
    h = h00 * (1 - fx) * (1 - fy) + h10 * fx * (1 - fy) + h01 * (1 - fx) * fy + h11 * fx * fy
    gx = (h10 - h00) * (1 - fy) + (h11 - h01) * fy; gy = (h01 - h00) * (1 - fx) + (h11 - h10) * fx
    return h, gx, gy, ix, iy, fx, fy


BRUSH = [(dx, dy, w) for dy in (-1, 0, 1) for dx in (-1, 0, 1) for w in [max(0.0, 1.0 - math.hypot(dx, dy) / 1.9)]]
_BW = sum(w for _, _, w in BRUSH)


def erode(H, res, amount=1.0, droplets=0.7, seed=0, steps=40, inertia=0.08, capacity=3.0, min_slope=0.02, deposit=0.25, erode_=0.25,
          evaporate=0.03, gravity=9.0, protect=None, batch=None, lim=0.25):
    """Droplets: heights in grid steps (H / res) — slopes and capacity are the same at any step; amount scales the erosion and deposition."""
    if amount <= 0: return H
    ny, nx = H.shape; rng = np.random.default_rng(seed); Hc = H / res; keep = None
    if protect is not None: keep = np.clip(protect, 0, 1); H0 = Hc.copy()
    n = int(nx * ny * droplets); k = 0.6 * amount                    # k — the erosion fraction per step (strength)
    batch = batch or max(1500, int(nx * ny * 0.02))                 # at once — no more than 2 % of the nodes: the droplets of a batch do not converge on one node
    for b0 in range(0, n, batch):
        m = min(batch, n - b0)
        px = rng.uniform(1, nx - 3, m); py = rng.uniform(1, ny - 3, m)
        dx = np.zeros(m); dy = np.zeros(m); v = np.ones(m); w = np.ones(m); sed = np.zeros(m); alive = np.ones(m, bool)
        for _ in range(steps):
            if not alive.any(): break
            a = np.nonzero(alive)[0]
            h, gx, gy, ix, iy, fx, fy = _bil(Hc, px[a], py[a])
            dx[a] = dx[a] * inertia - gx * (1 - inertia); dy[a] = dy[a] * inertia - gy * (1 - inertia)
            L = np.hypot(dx[a], dy[a]); flat = L < 1e-9
            if flat.any():                                           # on flat ground — a random direction
                t = rng.uniform(0, 2 * math.pi, flat.sum()); dx[a[flat]], dy[a[flat]] = np.cos(t), np.sin(t); L[flat] = 1.0
            dx[a] /= L; dy[a] /= L
            nx_, ny_ = px[a] + dx[a], py[a] + dy[a]
            out = (nx_ < 1) | (nx_ >= nx - 2) | (ny_ < 1) | (ny_ >= ny - 2)
            nh = np.where(out, h, _bil(Hc, np.clip(nx_, 1, nx - 2.001), np.clip(ny_, 1, ny - 2.001))[0])
            dh = nh - h                                              # < 0 — downhill (in grid steps)
            cap = np.maximum(-dh, min_slope) * v[a] * w[a] * capacity
            depo = (sed[a] > cap) | (dh > 0)
            amt_d = np.clip(np.where(dh > 0, np.minimum(dh, sed[a]), (sed[a] - cap) * deposit), 0.0, sed[a]) * depo   # no more than the sediment
            amt_e = np.clip(np.where(~depo, np.minimum((cap - sed[a]) * erode_ * k, -0.3 * dh), 0.0), 0.0, None)   # strength — erosion only; per step
            # no more than 30 % of the drop: otherwise a pit deepens itself (feedback)
            # deposit — into the 4 nodes of the cell (bilinear), erode — with a 3 × 3 brush around the nearest node; a
            # node's change per step — at most lim
            cx, cy = np.clip(np.rint(px[a]).astype(np.int64), 1, nx - 2), np.clip(np.rint(py[a]).astype(np.int64), 1, ny - 2)
            I = [(iy + oy) * nx + ix + ox for ox, oy in ((0, 0), (1, 0), (0, 1), (1, 1))] + [(cy + oy) * nx + cx + ox for ox, oy, wt in BRUSH if wt > 0]
            Wt = [amt_d * (1 - fx) * (1 - fy), amt_d * fx * (1 - fy), amt_d * (1 - fx) * fy, amt_d * fx * fy] + [-amt_e * wt / _BW for _, _, wt in BRUSH if wt > 0]
            dH = np.bincount(np.concatenate(I), np.concatenate(Wt), minlength=nx * ny).reshape(ny, nx)   # one count instead of 13 np.add.at
            Hc += np.clip(dH, -lim, lim)
            sed[a] = np.maximum(sed[a] + amt_e - amt_d, 0.0)
            v[a] = np.minimum(np.sqrt(np.maximum(v[a] ** 2 - dh * gravity, 0.0)), 6.0)
            w[a] *= (1 - evaporate); px[a], py[a] = nx_, ny_
            alive[a[out | (w[a] < 0.05)]] = False
    if keep is not None: Hc = Hc * (1 - keep) + H0 * keep
    H[:] = Hc * res
    return H


def thermal(H, res, talus=35.0, iters=40, rate=0.5, protect=None):
    """Scree: where the difference to a neighbor (4 directions) exceeds tan(talus)·step, the fraction rate of the excess goes downhill; protect — do not touch."""
    t = math.tan(math.radians(talus)) * res; keep = None
    if protect is not None: keep = np.clip(protect, 0, 1); H0 = H.copy()
    for _ in range(iters):
        D = np.zeros_like(H)
        for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
            d = H - np.roll(H, sh, axis=ax); ex = np.maximum(d - t, 0) * rate / 4
            if ax == 0: ex[0 if sh == 1 else -1, :] = 0
            else: ex[:, 0 if sh == 1 else -1] = 0
            D -= ex; D += np.roll(ex, -sh, axis=ax)
        H += D
    if keep is not None: H[:] = H * (1 - keep) + H0 * keep
    return H


def resize(A, shape):
    """Bilinear grid resizing (numpy)."""
    ny, nx = A.shape; my, mx = shape
    y = np.linspace(0, ny - 1, my); x = np.linspace(0, nx - 1, mx); y0 = np.minimum(y.astype(int), ny - 2); x0 = np.minimum(x.astype(int), nx - 2)
    fy, fx = (y - y0)[:, None], (x - x0)[None, :]
    return (A[y0][:, x0] * (1 - fy) * (1 - fx) + A[y0][:, x0 + 1] * (1 - fy) * fx + A[y0 + 1][:, x0] * fy * (1 - fx) + A[y0 + 1][:, x0 + 1] * fy * fx)


def erode_ms(H, res, amount=1.0, droplets=0.5, seed=0, macro=4, talus=38.0, thermal_iters=30):
    """Two-scale erosion: large (a grid macro times coarser — valleys and ridges tens of meters across), then fine (gullies), then
    scree. Returns H (m, changed in place)."""
    if amount <= 0: return H
    if macro and macro > 1:
        f = int(macro); ny, nx = H.shape
        Hs = resize(H, (max(8, ny // f), max(8, nx // f))); H0 = Hs.copy()
        erode(Hs, res * (ny - 1) / (Hs.shape[0] - 1), amount, droplets * 1.6, seed)
        H += resize(Hs - H0, H.shape)
    erode(H, res, amount * 0.7, droplets, seed + 1)
    if thermal_iters: thermal(H, res, talus, thermal_iters)
    return H
