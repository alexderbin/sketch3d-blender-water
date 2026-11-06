"""2.5D place field for the camera: the top of obstacles and what they are, at any point, without Blender.

The ground is core.surface.ground.Ground (relief.npz); walls and towers are masonry strips (surfaces.json: stations, the top by stations;
a tower is closed and filled entirely, a wall is a strip of thickness t); buildings and houses are occupancy (occupancy.json: eave + 3 m under
the roof, for a house — base + h + 2); without masonry — the walls and towers of the occupancy get a nominal height (10 and 18 m above the ground);
trees are the canopy (canopy.json: crown cr, height h). The raster step is res over the place's near grid (or over the layers' box); outside it — only
the ground. Queries are vectorised: top, ground, kind_at, sees (the ray does not go under a top), cast (the first hit of a ray), behind (what is behind
a subject on the camera ray), clear (clearance above a top), envelope (the envelope of the tops under a path). Pure numpy (works in Blender too)."""
import math

import numpy as np

KINDS = ("ground", "wall", "tower", "building", "tree")
WALL, TOWER, BUILDING, TREE = 1, 2, 3, 4
FAR = 300.0                                                          # m: farther from the subject, the ground behind it is "far"


class Field:
    def __init__(self, scene=None, res=1.0, margin=60.0):
        from core import layers as LY
        self.scene, self.res, self.G, self.skipped = scene, float(res), None, []
        if scene:
            import os
            if os.path.exists(os.path.join(scene, "relief.npz")):
                from core.surface.ground import Ground
                self.G = Ground(scene)
        if self.G is None:
            self.skipped.append("no place relief (relief.npz): the ground is the plane 0")
        self.anchors = LY.anchors(scene) if scene else {}
        surf = (LY.surfaces(scene).get("items") or []) if scene else []
        occ = LY.occupancy(scene) if scene else []
        trees = LY.canopy(scene) if scene else []
        box = self._box(surf, occ, trees, margin)
        self.x0, self.y0 = box[0], box[1]
        self.W = int(math.ceil((box[2] - box[0]) / self.res)) + 1; self.H = int(math.ceil((box[3] - box[1]) / self.res)) + 1
        X, Y = np.meshgrid(self.x0 + self.res * np.arange(self.W), self.y0 + self.res * np.arange(self.H))
        self.O = np.full((self.H, self.W), -np.inf); self.K = np.zeros((self.H, self.W), np.uint8)
        gz = self.ground(X, Y)
        for it in surf:                                              # masonry: walls as a strip, towers whole
            s = it.get("strip") or {}; pts = s.get("pts") or []
            if len(pts) < 2: continue
            top = np.broadcast_to(np.asarray(s.get("top", 0.0), float), (len(pts),))
            t = float(np.max(np.asarray(s.get("t", 2.0), float)))
            if it.get("part") == "towers":
                self._fill(self._poly(pts), float(top.max()), TOWER)
            else:
                P = list(pts) + ([pts[0]] if s.get("closed") else []); T = list(top) + ([top[0]] if s.get("closed") else [])
                for a, b, ta, tb in zip(P[:-1], P[1:], T[:-1], T[1:]):
                    self._fill(self._line(a, b, t), max(ta, tb), WALL)
        for it in surf:                                              # wall openings (gates, gaps along the face length) — cut them out: one can see through a gate
            if it.get("part") == "towers" or not it.get("gaps"): continue
            from core.surface.strip import strip_from
            st = strip_from(it["strip"]); t = float(np.max(np.asarray(it["strip"].get("t", 2.0), float)))
            for u0, u1 in it["gaps"]:
                for u in np.arange(float(u0) + 0.25, float(u1) - 0.1, 0.5):
                    (sl, m) = self._disc(np.asarray(st.world(u, 0.0, 0.0), float).reshape(-1)[:2], t)
                    cut = m & (self.K[sl] == WALL); self.O[sl][cut] = -np.inf; self.K[sl][cut] = 0
        have_walls = bool(surf)
        for it in occ:                                               # occupancy: buildings and houses; walls and towers — if there is no masonry
            k = it.get("kind")
            if k in ("building", "house") and len(it.get("poly") or []) >= 3:
                if it.get("eave") is not None: z = float(it["eave"]) + 3.0
                elif it.get("h") is not None: z = float(it.get("base", 0.0)) + float(it["h"]) + 2.0
                else: z = None
                m = self._poly(it["poly"])
                self._fill(m, z if z is not None else gz + 8.0, BUILDING)
            elif k == "wall" and not have_walls and len(it.get("line") or []) >= 2:
                for a, b in zip(it["line"][:-1], it["line"][1:]):
                    self._fill(self._line(a, b, float(it.get("w", 3.0))), gz + 10.0, WALL)
            elif k == "tower" and it.get("xy"):                       # an occupancy tower that is absent from the masonry (a keep): the height — from
                x, y, r = float(it["xy"][0]), float(it["xy"][1]), float(it.get("r", 4.0))   # the anchor on it, otherwise 18 m above the ground
                if int(self.kind_at(x, y)) in (WALL, TOWER): continue
                on = [a for a in self.anchors.values() if len(a) > 2 and math.hypot(a[0] - x, a[1] - y) <= r]
                z = max(a[2] for a in on) + 2.0 if on else float(self.ground(x, y)) + 18.0
                self._fill(self._disc((x, y), r), z, TOWER)
        for tr in trees:                                             # crowns: a subgrid of the circle, the height above the ground
            self._fill(self._disc((float(tr["x"]), float(tr["y"])), float(tr.get("cr", 3.0))), gz + float(tr.get("h", 12.0)), TREE)
        if not (surf or occ):
            self.skipped.append("no building layers (surfaces.json, occupancy.json): obstacles are the ground only")

    # ── raster ──────────────────────────────────────────────────────────────────────────────
    def _box(self, surf, occ, trees, margin):
        if self.G is not None:
            return self.G.near_bounds()
        pts = [p[:2] for it in surf for p in ((it.get("strip") or {}).get("pts") or [])]
        pts += [p[:2] for it in occ for p in (it.get("poly") or it.get("line") or [])]
        pts += [(t["x"], t["y"]) for t in trees]
        if not pts:
            return (-margin, -margin, margin, margin)
        P = np.asarray(pts, float)
        return (P[:, 0].min() - margin, P[:, 1].min() - margin, P[:, 0].max() + margin, P[:, 1].max() + margin)

    def _sub(self, x0, y0, x1, y1):
        """Raster subgrid over a box: row and column slices, cell centres (X, Y)."""
        j0 = max(0, int(math.floor((x0 - self.x0) / self.res))); j1 = min(self.W, int(math.ceil((x1 - self.x0) / self.res)) + 1)
        i0 = max(0, int(math.floor((y0 - self.y0) / self.res))); i1 = min(self.H, int(math.ceil((y1 - self.y0) / self.res)) + 1)
        X, Y = np.meshgrid(self.x0 + self.res * np.arange(j0, j1), self.y0 + self.res * np.arange(i0, i1))
        return (slice(i0, i1), slice(j0, j1)), X, Y

    def _poly(self, pts):
        """Polygon (even-odd rule) — a mask of the subgrid."""
        P = np.asarray(pts, float)[:, :2]; sl, X, Y = self._sub(*P.min(0), *P.max(0)); m = np.zeros(X.shape, bool)
        for (xa, ya), (xb, yb) in zip(P, np.roll(P, -1, 0)):
            c = (ya > Y) != (yb > Y)
            with np.errstate(divide="ignore", invalid="ignore"):
                m ^= c & (X < (xb - xa) * (Y - ya) / (yb - ya) + xa)
        return sl, m

    def _line(self, a, b, w):
        """A strip of width w along segment ab with round ends — a mask of the subgrid."""
        a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]; hw = w / 2 + self.res / 2
        sl, X, Y = self._sub(min(a[0], b[0]) - hw, min(a[1], b[1]) - hw, max(a[0], b[0]) + hw, max(a[1], b[1]) + hw)
        d = b - a; t = np.clip(((X - a[0]) * d[0] + (Y - a[1]) * d[1]) / max(float(d @ d), 1e-12), 0, 1)
        return sl, np.hypot(X - a[0] - t * d[0], Y - a[1] - t * d[1]) <= hw

    def _disc(self, c, r):
        r = max(r, self.res / 2); sl, X, Y = self._sub(c[0] - r, c[1] - r, c[0] + r, c[1] + r)
        return sl, np.hypot(X - c[0], Y - c[1]) <= r

    def _fill(self, mask, z, kind):
        sl, m = mask; z = np.asarray(z, float); z = z[sl] if z.ndim == 2 else np.broadcast_to(z, m.shape)
        O, K = self.O[sl], self.K[sl]; up = m & (z > O); O[up] = z[up]; K[up] = kind

    def _ij(self, x, y):
        j = np.floor((np.asarray(x, float) - self.x0) / self.res + 0.5).astype(int)
        i = np.floor((np.asarray(y, float) - self.y0) / self.res + 0.5).astype(int)
        inside = (i >= 0) & (i < self.H) & (j >= 0) & (j < self.W)
        return np.clip(i, 0, self.H - 1), np.clip(j, 0, self.W - 1), inside

    # ── queries ─────────────────────────────────────────────────────────────────────────────
    def ground(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return self.G.height(x, y) if self.G is not None else np.zeros(np.broadcast(x, y).shape)

    def top(self, x, y):
        """Top of the obstacle (or the ground) at a point, m."""
        i, j, inside = self._ij(x, y); g = self.ground(x, y)
        return np.where(inside, np.maximum(g, self.O[i, j]), g)

    def kind_at(self, x, y):
        """Label at a point: 0 — ground, 1 — wall, 2 — tower, 3 — building, 4 — tree."""
        i, j, inside = self._ij(x, y)
        k = np.where(inside, self.K[i, j].astype(int), 0)
        return np.where(np.where(inside, self.O[i, j], -np.inf) > self.ground(x, y) + 0.3, k, 0)

    def clear(self, p, r=1.0):
        """Clearance of a point (a camera) above the tops of obstacles within radius r, m; < 0 — inside."""
        p = np.atleast_2d(np.asarray(p, float)); a = np.linspace(0, 2 * math.pi, 8, endpoint=False)
        xs = np.concatenate([p[:, :1], p[:, :1] + r * np.cos(a)[None]], 1); ys = np.concatenate([p[:, 1:2], p[:, 1:2] + r * np.sin(a)[None]], 1)
        return p[:, 2] - self.top(xs, ys).max(1)

    def sees(self, A, B, near=0.5, skip_end=1.5, step=0.75):
        """Whether point B is visible from A (vectorised over pairs): the ray does not go under the tops of obstacles; at the ends there is a margin
        (the camera, the body and the ground under the subject)."""
        A, B = np.atleast_2d(np.asarray(A, float)), np.atleast_2d(np.asarray(B, float))
        L = np.linalg.norm(B - A, axis=1); M = int(min(400, max(16, math.ceil(float(L.max()) / step))))
        s = np.linspace(0.0, 1.0, M)[None, :]
        s = np.clip(s, (near / np.maximum(L, 1e-9))[:, None], (1 - skip_end / np.maximum(L, 1e-9))[:, None])
        P = A[:, None, :] + (B - A)[:, None, :] * s[..., None]
        return ~(P[..., 2] < self.top(P[..., 0], P[..., 1]) - 0.05).any(1)

    def cast(self, C, D, far=4000.0, d0=0.5):
        """First hit of rays C + t·D (vectorised): (t, point, label); on a miss — t = inf, label −1."""
        C, D = np.atleast_2d(np.asarray(C, float)), np.atleast_2d(np.asarray(D, float))
        D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
        t = np.concatenate([np.arange(d0, 60.0, 0.5), 60.0 * 1.02 ** np.arange(int(math.log(far / 60.0) / math.log(1.02)) + 1)])
        P = C[:, None, :] + D[:, None, :] * t[None, :, None]
        hit = P[..., 2] < self.top(P[..., 0], P[..., 1])
        any_ = hit.any(1); k = np.argmax(hit, 1)
        tt = np.where(any_, t[k], np.inf); pt = C + D * np.where(any_, t[k], 0.0)[:, None]
        lab = np.where(any_, self.kind_at(pt[:, 0], pt[:, 1]), -1)
        return tt, pt, lab

    def behind(self, C, P, skip=2.0):
        """What stands behind subject P on the ray from camera C (vectorised over pairs): [(label, distance behind the subject, anchor), …];
        labels — sky, far (ground farther than FAR m), ground, wall, tower, building, tree; the anchor is the nearest one within 25 m of the hit."""
        C, P = np.atleast_2d(np.asarray(C, float)), np.atleast_2d(np.asarray(P, float))
        D = P - C; D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
        t, pt, lab = self.cast(P + D * skip, D)
        names = list(self.anchors); A = np.asarray([self.anchors[n][:2] for n in names], float) if names else None
        out = []
        for ti, p, k in zip(t, pt, lab):
            if not np.isfinite(ti): out.append(("sky", None, None)); continue
            name = KINDS[int(k)] if k > 0 else ("far" if ti + skip > FAR else "ground")
            near = None
            if A is not None:
                dd = np.hypot(A[:, 0] - p[0], A[:, 1] - p[1]); m = int(np.argmin(dd))
                if dd[m] < 25.0: near = names[m]
            out.append((name, float(ti + skip), near))
        return out

    def envelope(self, xy, r=5.0, window=7):
        """Envelope of the tops under a path: the maximum within radius r around each point, then a moving maximum and a window mean."""
        xy = np.atleast_2d(np.asarray(xy, float))[:, :2]; a = np.linspace(0, 2 * math.pi, 12, endpoint=False)
        rs = np.array([0.0] + [r / 2] * 12 + [r] * 12); aa = np.concatenate([[0.0], a, a])
        e = self.top(xy[:, :1] + rs * np.cos(aa), xy[:, 1:2] + rs * np.sin(aa)).max(1)
        if len(e) > 2 and window > 1:
            window = int(window) | 1; h = window // 2; pad = np.pad(e, h, mode="edge")
            e = np.array([pad[i:i + window].max() for i in range(len(e))])
            pad = np.pad(e, h, mode="edge"); e = np.convolve(pad, np.ones(window) / window, mode="valid")
        return e
