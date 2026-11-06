"""Water in Blender (components/water): a mesh from the camera — the screen projected onto the water level (dense near the camera,
sparser toward the horizon; Earth curvature), displacement by the engine bands (waves.Sea) per frame with attenuation by the
mesh step, band images for the material (slopes, displacement derivatives, height, foam) and a material: the normal per frame
point from the bands, bands finer than a point go into roughness (together with the remainder up to Cox–Munk), Fresnel and sky,
water-column color, light through the crests (Atlas), foam with lace.
Frame: Water.update(t) — engine fields → images and vertices (nothing goes to disk)."""
import math

import bpy
import numpy as np

from components.water import waves as W

R_EARTH = 6.371e6


class N:
    """Shader nodes in code: n(type, inputs…) — a node; links and numbers go to the inputs."""

    def __init__(self, nt):
        self.nt = nt

    def n(self, t, *ins, **kw):
        nd = self.nt.nodes.new(t)
        for k, v in kw.items():
            setattr(nd, k, v)
        for i, v in enumerate(ins):
            self.put(nd.inputs[i], v)
        return nd

    def put(self, sock, v):
        if v is None: return
        if hasattr(v, "is_linked") or hasattr(v, "links"): self.nt.links.new(v, sock)
        elif isinstance(v, (tuple, list)) and len(v) == 3 and len(sock.default_value) == 4: sock.default_value = (*v, 1.0)
        else: sock.default_value = v

    def m(self, op, a, b=None, c=None):
        nd = self.n("ShaderNodeMath", operation=op)
        for i, v in enumerate((a, b, c)): self.put(nd.inputs[i], v)
        return nd.outputs[0]

    def v(self, op, a, b=None, s=None):
        nd = self.n("ShaderNodeVectorMath", operation=op)
        self.put(nd.inputs[0], a)
        if b is not None: self.put(nd.inputs[1], b)
        if s is not None: self.put(nd.inputs["Scale"], s)
        return nd.outputs["Value" if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else "Vector"]

    def xyz(self, x, y, z):
        return self.n("ShaderNodeCombineXYZ", x, y, z).outputs[0]

    def sep(self, v):
        return self.n("ShaderNodeSeparateXYZ", v).outputs

    def rgb(self, c):
        nd = self.n("ShaderNodeRGB"); nd.outputs[0].default_value = (*c, 1.0); return nd.outputs[0]

    def mix(self, f, a, b):
        nd = self.n("ShaderNodeMix", data_type="RGBA"); self.put(nd.inputs["Factor"], f)
        self.put(nd.inputs[6], a); self.put(nd.inputs[7], b); return nd.outputs[2]

    def ss(self, x, lo, hi):
        """smoothstep(lo, hi, x) — Map Range with smoothing."""
        nd = self.n("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP")
        self.put(nd.inputs["Value"], x); nd.inputs["From Min"].default_value = lo; nd.inputs["From Max"].default_value = hi
        return nd.outputs[0]


def image(name, n, m=None):
    """A band's data image: float RGBA n×m (m — n), Non-Color, the channels are independent."""
    m = n if m is None else m
    im = bpy.data.images.get(name)
    if im is not None and tuple(im.size) != (n, m): bpy.data.images.remove(im); im = None
    if im is None:
        im = bpy.data.images.new(name, n, m, alpha=True, float_buffer=True)
    im.colorspace_settings.name = "Non-Color"; im.alpha_mode = "CHANNEL_PACKED"
    return im


def frame_tan(cam, sc):
    """Half-tangents of the camera's field of view (x, y) by the Blender frame (sensor, fit and shift are accounted for — by magnitude)."""
    fr = [v for v in cam.data.view_frame(scene=sc)]
    d = abs(fr[0].z)
    return max(abs(v.x) for v in fr) / d, max(abs(v.y) for v in fr) / d


def near_for(cam, sc, level=0.0, N=256, margin=0.12):
    """The engine's near knob (waves.plan_bands: the finest tile ≈ 2·near) from the camera: the footprint of a frame point on the
    nearest visible water (the bottom of the frame) × N/4 — the shortest band wave ≈ 2 frame points at the near edge."""
    tx, ty = frame_tan(cam, sc); pix = 2 * tx / sc.render.resolution_x
    R = np.array(cam.matrix_world.to_3x3()); o = np.array(cam.matrix_world.translation); h = max(o[2] - level, 0.05)
    d = R @ np.array([0.0, -ty * (1 + margin), -1.0]); d /= np.linalg.norm(d)
    if d[2] >= -1e-3: return 2.0
    dist = h / -d[2]; fp = dist * pix * 1.5 / math.sqrt(-d[2])
    return max(fp * N / 4.0, 0.05)


def vnoise(x, y, seed=0):
    """Value noise on a lattice (a sine hash), 0…1, smooth (smoothstep) — for gust patches."""
    xi, yi = np.floor(x), np.floor(y); fx, fy = x - xi, y - yi
    h = lambda i, j: (np.sin(i * 127.1 + j * 311.7 + seed * 74.7) * 43758.5453) % 1.0
    u, v = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    return (h(xi, yi) * (1 - u) + h(xi + 1, yi) * u) * (1 - v) + (h(xi, yi + 1) * (1 - u) + h(xi + 1, yi + 1) * u) * v


def fbm(x, y, seed=0, oct_=4):
    s, w, tot = 0.0, 1.0, 0.0
    for k in range(oct_):
        s = s + w * vnoise(x * 2 ** k, y * 2 ** k, seed + 13 * k); tot += w; w *= 0.5
    return s / tot


def ss_np(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class Water:
    """A camera's water: sea — waves.Sea; level — the level, m; body — the water-column color (linear RGB, as reflected by the column);
    scatter — the color of light through the crests; sun — (direction to the sun (x, y, z), color × strength) for lighting the
    crests; grid — vertices across the screen (x, y); margin — a reserve beyond the frame (the displacement drags the edge in);
    far — the mesh range, m (default — the horizon × 1.3). ground — the bed: f(x, y) → z (a shoal: the bed is visible, waves fade
    toward the shore); K — water-column attenuation per R, G, B, m⁻¹ (optics.look); gust — ripple patches from gusts ("cat's paws")
    and slicks: {scale: m, floor: the ripple fraction in a lull, slick: the strength of the smooth streaks, drift: patch speed as a
    fraction of the wind}; only for bands shorter than 3 m."""

    def __init__(self, sc, sea, cam=None, level=0.0, name="water", grid=(400, 300), margin=0.12, body=(0.004, 0.018, 0.03),
                 scatter=(0.02, 0.16, 0.13), sun=None, foam_rgb=(0.82, 0.85, 0.85), far=None, coll=None, debug=None,
                 ground=None, K=None, gust=None, bottom=(0.1, 0.09, 0.075), refr_far=22.0, extra=None, shore=None, flow=None,
                 level_fn=None):
        self.sc, self.sea, self.cam, self.level, self.name = sc, sea, cam or sc.camera, float(level), name
        self.gx, self.gy, self.margin, self.far = int(grid[0]), int(grid[1]), float(margin), far
        self.body, self.scatter, self.sun, self.foam_rgb, self.debug = body, scatter, sun, foam_rgb, debug
        self.ground, self.K, self.gust, self.bottom, self.refr_far = ground, K, gust, bottom, float(refr_far)
        self.level_fn = level_fn                                        # level by points (the Site's rivers and lakes): f(x, y) → z | NaN — not water
        self.shore = shore                                              # surf (shore.Shore): a world field → displacement, slopes, foam
        self.flow = flow                                                # current (flow.River): the flow carries the ripples and foam
        self.rimg = None
        if flow is not None:
            self.fimg = image(f"{name}_flow", flow.nx, flow.ny)
            self.fimg.pixels.foreach_set(np.stack([flow.vx, flow.vy, flow.foam, flow.speed], -1).astype(np.float32).ravel()); self.fimg.update()
            if getattr(flow, "rapid", False): self.rimg = image(f"{name}_rapid", flow.nx, flow.ny)   # torrent: the hump, slopes, white water
        if shore is not None:
            self.simg = (image(f"{name}_shore_d", shore.nx, shore.ny), image(f"{name}_shore_n", shore.nx, shore.ny))
        self.gusty = [b.L / 6.0 < 3.0 for b in sea.bands]               # ripple bands (gusts damp and wake them)
        self.storm = sea.u10 >= 10.8                                    # force 6+: foam of three ages, veins — marble (below that — lace)
        if self.storm:
            self.margin = max(self.margin, 0.3)                          # the crest drift (λ·A up to ~6 m) drags the mesh edge into the frame
            from components.water import marble
            Mt = marble.tile(); one = np.ones(Mt.shape[:2] + (1,), np.float32)
            self.mimg = image(f"{name}_marble", Mt.shape[0]); self.mimg.pixels.foreach_set(np.concatenate([Mt, 0 * one, one], -1).ravel())
            self.mimg.update(); self.mL = 48.0
        self.imgs = [(image(f"{name}_b{i}_d", b.N), image(f"{name}_b{i}_n", b.N)) for i, b in enumerate(sea.bands)]
        self.extra = list(extra or [])                                  # extra bands (rain rings: waves.Rain) — into the material only
        self.ximgs = [(image(f"{name}_x{i}_d", r.N), image(f"{name}_x{i}_n", r.N)) for i, r in enumerate(self.extra)]
        self.xvar = []
        for r in self.extra:
            F0 = r.eval(5.0); self.xvar.append(float((F0["hx"] ** 2 + F0["hy"] ** 2).mean()))
        self.obj = self._mesh(coll or sc.collection)
        self.obj.data.materials.append(self.material())
        self.base = None
        self.place()

    # --- mesh ---
    def place(self):
        """Camera mesh: screen rays with a margin → intersection with the water level (above the horizon — the range ring far), Earth
        curvature; the vertex base (undisplaced x, y — the band coordinate), the mesh step at each vertex (band attenuation), UV wxy."""
        cam, sc = self.cam, self.sc
        tx, ty = frame_tan(cam, sc); m = 1 + self.margin
        u, v = np.meshgrid(np.linspace(-1, 1, self.gx) * tx * m, np.linspace(-1, 1, self.gy) * ty * m)
        Rm = np.array(cam.matrix_world.to_3x3()); o = np.array(cam.matrix_world.translation)
        d = np.stack([u, v, -np.ones_like(u)], -1) @ Rm.T
        h = max(o[2] - self.level, 0.05)
        far = self.far or min(1.3 * math.sqrt(2 * R_EARTH * h) + 200.0, cam.data.clip_end * 0.95)
        dh = np.hypot(d[..., 0], d[..., 1]) + 1e-12
        t = np.where(d[..., 2] < -1e-6, h / np.maximum(-d[..., 2], 1e-9), np.inf)
        dist = np.minimum(t * dh, far)                                  # distance along the ground
        x, y = o[0] + d[..., 0] / dh * dist, o[1] + d[..., 1] / dh * dist
        z = self.level - dist * dist / (2 * R_EARTH)
        if self.level_fn is not None:                                   # level by points: where there is no water — deep underground
            lv = self.level_fn(x, y); z = np.where(np.isfinite(lv), lv - dist * dist / (2 * R_EARTH), self.level - 1e3)
        P = np.stack([x, y, z], -1)
        sx = np.linalg.norm(np.diff(P, axis=1), axis=-1); sy = np.linalg.norm(np.diff(P, axis=0), axis=-1)
        s = np.maximum(np.pad(sx, ((0, 0), (0, 1)), mode="edge"), np.pad(sy, ((0, 1), (0, 0)), mode="edge"))
        self.base, self.step, self.dist = P.reshape(-1, 3), s.ravel(), dist.ravel()
        me = self.obj.data
        me.vertices.foreach_set("co", self.base.astype(np.float32).ravel())
        li = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", li)
        me.uv_layers["wxy"].data.foreach_set("uv", self.base[li, :2].astype(np.float32).ravel())
        self.depth = np.full(len(self.base), 1e3)
        if self.ground is not None:                                     # depth at the vertex; the waves fade toward the waterline
            self.depth = np.clip(self.base[:, 2] - self.ground(self.base[:, 0], self.base[:, 1]), 0.0, 1e3)
        h_att = max(0.6 * self.sea.stats()["hs"], 0.03)
        self.att = ss_np(self.depth, 0.0, h_att)
        if self.shore is not None:                                      # in the surf the sea's random waves give way to the crests running toward the shore
            from components.water.shore import _bilinear
            self.wz = _bilinear(self.shore.wz, self.shore.x0, self.shore.y0, self.shore.res, self.base[:, 0], self.base[:, 1])
            H0 = self.shore.H0                                          # the random waves fade only in the breaking zone — by depth,
            self.att = self.att * (1 - 0.85 * (1 - ss_np(self.depth, 2.0 * H0, 8.0 * H0)))   # not by the edge of the surf mesh (otherwise a seam)
            self.zg = self.ground(self.base[:, 0], self.base[:, 1]) if self.ground is not None else np.full(len(self.base), -1e3)
        for nm, val in (("depth", self.depth), ("att", self.att)):
            a = me.attributes.get(nm) or me.attributes.new(nm, "FLOAT", "POINT"); a.data.foreach_set("value", val.astype(np.float32))
        me.update()
        # attenuation of the band displacement by the mesh step: the band stays intact while the step < λ_min/3 and
        # fades out by the step λ_max/3
        self.fade = []
        for b in self.sea.bands:
            lo, hi = math.log(b.lam / 3), math.log(max(b.L / 6.0, b.lam * 2) / 3)
            self.fade.append(1 - ss_np(np.log(np.maximum(self.step, 1e-6)), lo, hi))
        return P

    def _mesh(self, coll):
        nx, ny = self.gx, self.gy
        ii = np.arange(nx * ny).reshape(ny, nx)
        q = np.stack([ii[:-1, :-1], ii[:-1, 1:], ii[1:, 1:], ii[1:, :-1]], -1).reshape(-1, 4)
        me = bpy.data.meshes.new(self.name)
        me.vertices.add(nx * ny); me.loops.add(q.size); me.polygons.add(len(q))
        me.loops.foreach_set("vertex_index", q.ravel().astype(np.int32))
        me.polygons.foreach_set("loop_start", np.arange(0, q.size, 4, dtype=np.int32))
        me.polygons.foreach_set("loop_total", np.full(len(q), 4, np.int32))
        me.uv_layers.new(name="wxy"); me.update(); me.shade_smooth()
        ob = bpy.data.objects.new(self.name, me); coll.objects.link(ob)
        ob["kind"] = "water"; ob["sk_layer"] = "water"
        ob.visible_shadow = False                                       # water casts no shadow: the light reaches the bed (the caustics are our own)
        return ob

    # --- frame ---
    def update(self, t):
        """Frame at time t, s: band fields and foam → material images; vertex displacement by the bands with attenuation by the mesh step."""
        F, foam = self.sea.eval(t), self.sea.foam(t)
        self.last = F                                                   # the frame's fields go to the floating object (waves.surface_at)
        D = np.zeros_like(self.base)
        xy = self.base[:, :2]
        gm = self.gust_at(t)
        old = self.sea.foam_old(t) if self.storm else [None] * len(F)  # storm: "current" foam and old foam (the source of the streaks)
        nold = max(sum(o is not None for o in old), 1)
        zl = np.zeros(len(xy))                                          # long waves (bands 0–1) — for the Stokes correction
        XX, YY, XY = np.zeros(len(xy)), np.zeros(len(xy)), np.zeros(len(xy))   # Jacobian of the shift at the vertices — against crest folding
        for bi, (b, f, fo, ol, (imd, imn), fd, gy) in enumerate(zip(self.sea.bands, F, foam, old, self.imgs, self.fade, self.gusty)):
            fz = np.zeros((b.N, b.N)) if fo is None else _upsample(fo, b.N)
            r0 = r1 = np.zeros((b.N, b.N))                              # height image: R — foam "now", G — old foam (toward the mean 1 / bands)
            if ol is not None: r0, r1 = _upsample(ol[0], b.N), _upsample(ol[1] / max(float(ol[1].mean()), 0.02) / nold, b.N)
            imd.pixels.foreach_set(np.stack([r0, r1, f["dz"], fz], -1).astype(np.float32).ravel()); imd.update()
            imn.pixels.foreach_set(np.stack([f["hx"], f["hy"], f["xx"], f["yy"]], -1).astype(np.float32).ravel()); imn.update()
            k_ = fd * self.att * (gm if gy else 1.0)
            on = k_ > 1e-3
            if on.any():
                p = xy[on]
                dzb = W.sample(f["dz"], b.L, p)
                D[on] += k_[on, None] * np.stack([W.sample(f["dx"], b.L, p), W.sample(f["dy"], b.L, p), dzb], 1)
                if bi < 2: zl[on] += k_[on] * dzb
                if self.storm:
                    XX[on] += k_[on] * W.sample(f["xx"], b.L, p); YY[on] += k_[on] * W.sample(f["yy"], b.L, p); XY[on] += k_[on] * W.sample(f["xy"], b.L, p)
        if self.storm:                                                  # no folding: where the shift folds the surface (J < 0.25) —
            J = (1 + XX) * (1 + YY) - XY * XY; jmin = 0.25                  # an overhang that is "about to collapse" but does not — the shift is smaller:
            unf = np.where(J < jmin, (1 - jmin) / np.maximum(1 - J, 1e-3), 1.0); D[:, :2] *= unf[:, None]   # a sharp crest, without an overhang
            m_ = self.obj.data; a_ = m_.attributes.get("unfold") or m_.attributes.new("unfold", "FLOAT", "POINT")
            a_.data.foreach_set("value", unf.astype(np.float32))
        if self.storm:                                                  # second-order Stokes (narrow-band): η + k_p/2·(η² − σ²) —
            kp = 2 * math.pi / self.sea.lp; v_ = float(np.mean(zl * zl)) if len(zl) else 0.0   # crests higher and sharper, troughs wider
            D[:, 2] += 0.5 * kp * (zl * zl - v_) * self.att
        if self.sun is not None and self.ground is not None:            # caustics on the bed: sun rays through a fine band
            b_ = self.sea.bands[-1]; C = W.caustics(F[-1], b_.L, sun=self.sun[0], depth=0.6)
            im = image(f"{self.name}_caustic", b_.N); im.pixels.foreach_set(np.stack([C, C, C, np.ones_like(C)], -1).astype(np.float32).ravel())
            im.update(); self.caustic = (im, b_.L)
        for r, (imd, imn) in zip(self.extra, self.ximgs):
            f = r.eval(t)
            imd.pixels.foreach_set(np.stack([f["dx"], f["dy"], f["dz"], np.zeros_like(f["dz"])], -1).astype(np.float32).ravel()); imd.update()
            imn.pixels.foreach_set(np.stack([f["hx"], f["hy"], f["xx"], f["yy"]], -1).astype(np.float32).ravel()); imn.update()
        me = self.obj.data
        a = me.attributes.get("gust") or me.attributes.new("gust", "FLOAT", "POINT"); a.data.foreach_set("value", np.broadcast_to(gm, self.att.shape).astype(np.float32))
        fv = np.zeros(len(self.base)); dep = self.depth
        if self.shore is not None:                                      # surf: the water rise and the foam of the bore; on the sand — the swash tongue
            Fs = self.shore.eval(t); S = self.shore.at(Fs, xy); sh = self.shore
            imd, imn = self.simg; z0 = np.zeros_like(Fs["h"])
            imd.pixels.foreach_set(np.stack([z0, z0, Fs["h"], Fs["foam"]], -1).astype(np.float32).ravel()); imd.update()
            imn.pixels.foreach_set(np.stack([Fs["hx"], Fs["hy"], z0, z0], -1).astype(np.float32).ravel()); imn.update()
            wet = self.depth > 0
            D[:, 2] += np.where(wet, S["h"], 0.0)
            film, ffoam = sh.swash(xy, t)
            land = ~wet; on = land & (film > 0)
            D[:, 2] = np.where(on, self.zg + film - self.level, np.where(land, -0.05, D[:, 2]))   # land: a film on top of the sand, otherwise beneath it
            fv = np.where(on, ffoam, 0.0)
            dep = np.where(wet, np.maximum(self.depth + S["h"], 0.0), film)
        if self.rimg is not None:                                       # torrent: standing waves, rollers, pillows — the displacement and the image
            from components.water.shore import _bilinear
            Fr = self.flow.eval(t); fw = self.flow
            D[:, 2] += _bilinear(Fr["h"], fw.x0, fw.y0, fw.res, xy[:, 0], xy[:, 1])
            self.rimg.pixels.foreach_set(np.stack([Fr["h"], Fr["hx"], Fr["hy"], Fr["white"]], -1).astype(np.float32).ravel()); self.rimg.update()
        fc = np.zeros(len(self.base))                                    # band foam at the vertices (storm spray — bl_spray)
        for bb, fo in zip(self.sea.bands, foam):
            if fo is not None: fc = 1 - (1 - fc) * (1 - W.sample(fo, bb.L, xy))
        if self.storm and self.sea.u10 >= 17.2:                         # a breaking crest is not smooth: bumps of ±0.08·Hs where there is foam on the crest
            hs_ = self.sea.stats()["hs"]; crest = ss_np(D[:, 2], 0.2 * hs_, 0.55 * hs_)
            lump = fbm(xy[:, 0] / 4.0 + 0.6 * t, xy[:, 1] / 4.0 - 0.4 * t, 21, 3) - 0.5
            D[:, 2] += 2 * 0.08 * hs_ * lump * crest * ss_np(fc, 0.2, 0.7)
        for nm, val in (("foam_v", fv), ("depth", dep), ("foam_c", fc)):
            a = me.attributes.get(nm) or me.attributes.new(nm, "FLOAT", "POINT"); a.data.foreach_set("value", np.asarray(val, np.float32))
        me.vertices.foreach_set("co", (self.base + D).astype(np.float32).ravel()); me.update()
        self.obj["water_t"] = float(t)
        for mt in me.materials:
            nd = mt.node_tree.nodes.get("water_time") if mt and mt.node_tree else None
            if nd: nd.outputs[0].default_value = float(t)
        return D

    def gust_at(self, t):
        """Ripple patches at the vertices at time t: gusts — large noise moving downwind (drift·U10), slicks — smooth streaks stretched along
        the wind; → a ripple multiplier floor…1 (without gust — 1)."""
        if not self.gust: return np.ones(len(self.base))
        g_ = self.gust; ca, sa = self.sea.wind_to; xy = self.base[:, :2]
        al = xy[:, 0] * ca + xy[:, 1] * sa - g_.get("drift", 0.8) * self.sea.u10 * t; cr = -xy[:, 0] * sa + xy[:, 1] * ca
        L = float(g_.get("scale", 40.0))
        e0, e1 = g_.get("edge", (0.42, 0.62))                           # patch threshold: higher — more lulls (mirror-smooth areas)
        pat = ss_np(fbm(al / L, cr / L, 7, 4), e0, e1)
        sl = ss_np(fbm(al / (L * 3.0), cr / (L * 0.25), 11, 3), 0.55, 0.75) * float(g_.get("slick", 0.5))
        fl = float(g_.get("floor", 0.08))
        return fl + (1 - fl) * pat * (1 - sl)

    # --- material ---
    def material(self):
        """Material: the normal from the bands (attenuation by the frame point), roughness from the unresolved slopes, Fresnel IOR 1.333,
        the water column — diffuse with the body color, light through the crests — emission after Atlas, foam — lace by the band foam fraction."""
        sea, cam, sc = self.sea, self.cam, self.sc
        mat = bpy.data.materials.new(self.name)
        nt = mat.node_tree; nt.nodes.clear(); g = N(nt)
        st = sea.stats()
        uv = g.n("ShaderNodeUVMap", uv_map="wxy").outputs[0]
        tm = g.n("ShaderNodeValue", label="water_time"); tm.name = "water_time"; tm.outputs[0].default_value = 0.0
        tx, _ = frame_tan(cam, sc); pix = 2 * tx / sc.render.resolution_x            # angle of a frame point, rad
        fl = None
        if self.flow is not None:                                       # current: velocity (x, y), foam, magnitude — a world image
            fw = self.flow; Wf, Hf = (fw.nx - 1) * fw.res, (fw.ny - 1) * fw.res
            uf = g.v("ADD", g.v("MULTIPLY", g.v("SUBTRACT", uv, (fw.x0, fw.y0, 0.0)), (1.0 / Wf, 1.0 / Hf, 1.0)), (0.5 / fw.nx, 0.5 / fw.ny, 0.0))
            ft = g.n("ShaderNodeTexImage", image=self.fimg, interpolation="Linear", extension="EXTEND"); g.put(ft.inputs[0], uf)
            fc = g.sep(ft.outputs["Color"]); fvec = g.xyz(fc[0], fc[1], 0.0)
            Tc = min(1.6, 2.0 / max(getattr(fw, "v0", 1.0), 0.5))       # advection cycle, s: two phases offset by half a cycle; on fast
            # water it is shorter — the shift per cycle ≤ ~2 m, otherwise the current's shear stretches the ripples into
            # "curls"
            nph = g.n("ShaderNodeTexNoise", noise_dimensions="2D"); g.put(nph.inputs["Vector"], uv); nph.inputs["Scale"].default_value = 0.15
            p1 = g.m("FRACT", g.m("ADD", g.m("DIVIDE", tm.outputs[0], Tc), nph.outputs["Fac"]))
            p2 = g.m("FRACT", g.m("ADD", p1, 0.5))
            fl = dict(vec=fvec, foam=fc[2], speed=ft.outputs["Alpha"], u1=g.v("SUBTRACT", uv, g.v("SCALE", fvec, s=g.m("MULTIPLY", p1, Tc))),
                      u2=g.v("SUBTRACT", uv, g.v("SCALE", fvec, s=g.m("MULTIPLY", p2, Tc))),
                      w2=g.m("ABSOLUTE", g.m("SUBTRACT", g.m("MULTIPLY", p1, 2.0), 1.0)))
        cd = g.n("ShaderNodeCameraData")
        geo = g.n("ShaderNodeNewGeometry")
        cosv = g.m("MAXIMUM", g.m("ABSOLUTE", g.v("DOT_PRODUCT", geo.outputs["Normal"], geo.outputs["Incoming"])), 0.02)
        fp = g.m("DIVIDE", g.m("MULTIPLY", cd.outputs["View Distance"], pix * 1.5), g.m("SQRT", cosv))   # footprint of a frame point, m
        # (stretched along the view direction by 1/cos — we take the geometric mean of the axes, otherwise the
        # distance is blurred across)
        lfp = g.m("LOGARITHM", g.m("MAXIMUM", g.m("MULTIPLY", fp, 2.0), 1e-5), math.e)
        att = g.n("ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name="att").outputs["Fac"]     # the waves fade toward the waterline
        gst = g.n("ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name="gust").outputs["Fac"]    # gust patches
        ag = g.m("MULTIPLY", att, gst)
        hx = hy = xx = yy = hz = var = None; keep = None; now_keep = old_sum = None
        layers = [(b.L, b.N, b.lam, max(b.L / 6.0, b.lam * 2), st["slope2"][i], im_, self.gusty[i]) for i, (b, im_) in enumerate(zip(sea.bands, self.imgs))]
        layers += [(r.L, r.N, r.lam, 0.12, v_, im_, False) for r, im_, v_ in zip(self.extra, self.ximgs, self.xvar)]
        for i, (bL, bN, blam, blong, bvar, (imd, imn), bgust) in enumerate(layers):
            u_ = g.v("ADD", g.v("SCALE", uv, s=1.0 / bL), (0.5 / bN, 0.5 / bN, 0.0))
            tn = g.n("ShaderNodeTexImage", image=imn, interpolation="Linear", extension="REPEAT"); g.put(tn.inputs[0], u_)
            adv = fl is not None and len(sea.bands) - 2 <= i < len(sea.bands)   # the river's fine bands — advected by the current (two phases)
            if adv:
                tA = g.n("ShaderNodeTexImage", image=imn, interpolation="Linear", extension="REPEAT")
                g.put(tA.inputs[0], g.v("ADD", g.v("SCALE", fl["u1"], s=1.0 / bL), (0.5 / bN, 0.5 / bN, 0.0)))
                tB = g.n("ShaderNodeTexImage", image=imn, interpolation="Linear", extension="REPEAT")
                g.put(tB.inputs[0], g.v("ADD", g.v("SCALE", fl["u2"], s=1.0 / bL), (0.5 / bN, 0.5 / bN, 0.0)))
                class _T: pass
                tn = _T(); tn.outputs = {"Color": g.mix(fl["w2"], tA.outputs["Color"], tB.outputs["Color"]),
                                         "Alpha": g.m("ADD", g.m("MULTIPLY", tA.outputs["Alpha"], g.m("SUBTRACT", 1.0, fl["w2"])), g.m("MULTIPLY", tB.outputs["Alpha"], fl["w2"]))}
            use_d = i < 3 and i < len(sea.bands)                         # height and foam — for the three large bands (Metal allows ≤ 16 textures)
            if use_d: td = g.n("ShaderNodeTexImage", image=imd, interpolation="Linear", extension="REPEAT"); g.put(td.inputs[0], u_)
            lo, hi = math.log(blam), math.log(blong)
            f = g.m("SUBTRACT", 1.0, g.ss(lfp, lo, hi))                  # 1 — the band is resolved by the frame points
            am = ag if bgust else att
            if fl is not None and i < len(sea.bands): am = g.m("MULTIPLY", am, g.m("ADD", 1.0, g.m("MULTIPLY", g.ss(fl["speed"], 0.4, 2.0), 1.2)))   # fast water — stronger ripples
            fa = g.m("MULTIPLY", f, am)
            c = g.sep(tn.outputs["Color"])
            terms = (g.m("MULTIPLY", c[0], fa), g.m("MULTIPLY", c[1], fa), g.m("MULTIPLY", c[2], fa), g.m("MULTIPLY", tn.outputs["Alpha"], fa))
            hx, hy, xx, yy = [t_ if a_ is None else g.m("ADD", a_, t_) for a_, t_ in zip((hx, hy, xx, yy), terms)]
            if use_d: z_ = g.m("MULTIPLY", g.sep(td.outputs["Color"])[2], fa); hz = z_ if hz is None else g.m("ADD", hz, z_)
            uv_ = g.m("MULTIPLY", g.m("MULTIPLY", g.m("SUBTRACT", 1.0, f), bvar), g.m("MULTIPLY", am, am))
            var = uv_ if var is None else g.m("ADD", var, uv_)
            if use_d: k_ = g.m("SUBTRACT", 1.0, td.outputs["Alpha"]); keep = k_ if keep is None else g.m("MULTIPLY", keep, k_)
            if use_d and self.storm:                                    # foam "now" — together across the bands, old foam — as a sum (mean 1)
                cc = g.sep(td.outputs["Color"]); nk = g.m("SUBTRACT", 1.0, cc[0])
                now_keep = nk if now_keep is None else g.m("MULTIPLY", now_keep, nk)
                old_sum = cc[1] if old_sum is None else g.m("ADD", old_sum, cc[1])
        if self.shore is not None:                                      # surf: slopes and foam of the world field (no attenuation — the waves are long)
            sh = self.shore; Wd, Hd = (sh.nx - 1) * sh.res, (sh.ny - 1) * sh.res
            us = g.v("ADD", g.v("MULTIPLY", g.v("SUBTRACT", uv, (sh.x0, sh.y0, 0.0)), (1.0 / Wd, 1.0 / Hd, 1.0)), (0.5 / sh.nx, 0.5 / sh.ny, 0.0))
            sd = g.n("ShaderNodeTexImage", image=self.simg[0], interpolation="Linear", extension="EXTEND"); g.put(sd.inputs[0], us)
            sn = g.n("ShaderNodeTexImage", image=self.simg[1], interpolation="Linear", extension="EXTEND"); g.put(sn.inputs[0], us)
            c_ = g.sep(sn.outputs["Color"]); hx = g.m("ADD", hx, c_[0]); hy = g.m("ADD", hy, c_[1])
            hz = g.m("ADD", hz, g.sep(sd.outputs["Color"])[2])
            keep = g.m("MULTIPLY", keep, g.m("SUBTRACT", 1.0, sd.outputs["Alpha"]))
        if fl is not None:                                              # river foam: along the banks and behind stones, the current carries the pattern
            def pat(u__):                                               # lace: a network of noise ridges (foam threads), finer — jitter
                nn = g.n("ShaderNodeTexNoise", noise_dimensions="2D", noise_type="RIDGED_MULTIFRACTAL"); g.put(nn.inputs["Vector"], u__)
                nn.inputs["Scale"].default_value = 2.6; nn.inputs["Detail"].default_value = 3.0; nn.inputs["Distortion"].default_value = 0.8
                nn.inputs["Offset"].default_value = 1.0; nn.inputs["Gain"].default_value = 2.0; return nn.outputs["Fac"]
            pn = g.m("ADD", g.m("MULTIPLY", pat(fl["u1"]), g.m("SUBTRACT", 1.0, fl["w2"])), g.m("MULTIPLY", pat(fl["u2"]), fl["w2"]))
            ff = g.m("MULTIPLY", g.m("MULTIPLY", fl["foam"], g.ss(pn, 0.85, 1.45)), 0.85)
            keep = g.m("MULTIPLY", keep, g.m("SUBTRACT", 1.0, ff))
            if self.rimg is not None:                                   # white water: the flow field × boiling (4D noise) and the advection lace
                rt = g.n("ShaderNodeTexImage", image=self.rimg, interpolation="Linear", extension="EXTEND"); g.put(rt.inputs[0], uf)
                rc = g.sep(rt.outputs["Color"]); hx = g.m("ADD", hx, rc[1]); hy = g.m("ADD", hy, rc[2])
                bo = g.n("ShaderNodeTexNoise", noise_dimensions="4D"); g.put(bo.inputs["Vector"], fl["u1"]); g.put(bo.inputs["W"], g.m("MULTIPLY", tm.outputs[0], 0.9))
                bo.inputs["Scale"].default_value = 0.6; bo.inputs["Detail"].default_value = 3.0   # boiling: bumps of ~1–2 m, changing in ~1 s
                bc = g.sep(bo.outputs["Color"]); kb = g.m("MULTIPLY", g.m("ADD", 0.12, g.m("MULTIPLY", rt.outputs["Alpha"], 0.35)), g.ss(fl["speed"], 0.8, 3.0))
                hx = g.m("ADD", hx, g.m("MULTIPLY", g.m("SUBTRACT", bc[0], 0.5), kb)); hy = g.m("ADD", hy, g.m("MULTIPLY", g.m("SUBTRACT", bc[1], 0.5), kb))
                ch = g.n("ShaderNodeTexNoise", noise_dimensions="4D"); g.put(ch.inputs["Vector"], fl["u1"]); g.put(ch.inputs["W"], g.m("MULTIPLY", tm.outputs[0], 1.4))
                ch.inputs["Scale"].default_value = 1.1; ch.inputs["Detail"].default_value = 4.0; ch.inputs["Roughness"].default_value = 0.6
                mix_ = g.m("ADD", g.m("MULTIPLY", g.m("SUBTRACT", pn, 1.0), 0.5), g.m("SUBTRACT", ch.outputs["Fac"], 0.5))
                wv = g.ss(g.m("ADD", g.m("MULTIPLY", rt.outputs["Alpha"], 1.3), g.m("MULTIPLY", mix_, 0.9)), 0.4, 0.7)
                keep = g.m("MULTIPLY", keep, g.m("SUBTRACT", 1.0, wv)); self._dbg_white = wv
                self._aer = g.m("MINIMUM", g.m("ADD", g.m("MULTIPLY", rt.outputs["Alpha"], 0.6), g.m("MULTIPLY", g.ss(fl["speed"], 1.5, 4.0), 0.25)), 1.0)
        fva = g.n("ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name="foam_v").outputs["Fac"]   # foam at the vertices (swash)
        keep = g.m("MULTIPLY", keep, g.m("SUBTRACT", 1.0, fva))
        if self.storm:                                                   # the normal follows the same damped shift as the vertices
            unf_ = g.n("ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name="unfold").outputs["Fac"]
            xx, yy = g.m("MULTIPLY", xx, unf_), g.m("MULTIPLY", yy, unf_)
        nx_ = g.m("DIVIDE", g.m("MULTIPLY", hx, -1.0), g.m("MAXIMUM", g.m("ADD", xx, 1.0), 0.15))
        ny_ = g.m("DIVIDE", g.m("MULTIPLY", hy, -1.0), g.m("MAXIMUM", g.m("ADD", yy, 1.0), 0.15))
        # facets (stochastic microfacets): the unresolved slopes (bands finer than a point and the remainder up to
        # Cox–Munk) — 60 % by cells in water coordinates with a random tilt, the cell ~ half the footprint of a frame
        # point (two octaves with blending): the sky inside a point is averaged by samples, the sun reflects in
        # individual facets — the sparkles of the glitter path; the facets shimmer over time
        unres = g.m("ADD", g.m("ADD", var, g.m("MULTIPLY", max(st["slope2_rest"], 2e-5), g.m("MULTIPLY", ag, ag))), 2e-6)
        amp = g.m("MULTIPLY", g.m("SQRT", g.m("MULTIPLY", unres, 0.3)), math.sqrt(3.0))      # uniform ±a: a² / 3 = 0.6·σ²/2
        lvl = g.m("LOGARITHM", g.m("MAXIMUM", g.m("MULTIPLY", fp, 0.5), 1e-4), 2.0)
        l0 = g.m("FLOOR", lvl); fr = g.m("SUBTRACT", lvl, l0)
        gx = gy = None
        for j, wgt in ((0, g.m("SUBTRACT", 1.0, fr)), (1, fr)):
            sc_ = g.m("POWER", 2.0, g.m("MULTIPLY", g.m("ADD", l0, float(j)), -1.0))          # 1 / cell
            vo = g.n("ShaderNodeTexVoronoi", feature="F1", voronoi_dimensions="4D"); g.put(vo.inputs["Vector"], uv)
            g.put(vo.inputs["Scale"], sc_); g.put(vo.inputs["W"], g.m("DIVIDE", g.m("MULTIPLY", tm.outputs[0], 3.0), sc_))
            rc = g.sep(vo.outputs["Color"])
            sx_ = g.m("MULTIPLY", g.m("SUBTRACT", rc[0], 0.5), wgt); sy_ = g.m("MULTIPLY", g.m("SUBTRACT", rc[1], 0.5), wgt)
            gx = sx_ if gx is None else g.m("ADD", gx, sx_); gy = sy_ if gy is None else g.m("ADD", gy, sy_)
        norm = g.m("DIVIDE", g.m("MULTIPLY", amp, 2.0), g.m("SQRT", g.m("ADD", g.m("POWER", fr, 2.0), g.m("POWER", g.m("SUBTRACT", 1.0, fr), 2.0))))
        gx, gy = g.m("MULTIPLY", gx, norm), g.m("MULTIPLY", gy, norm)
        nrm = g.v("NORMALIZE", g.xyz(g.m("ADD", nx_, gx), g.m("ADD", ny_, gy), 1.0))
        foam = g.m("SUBTRACT", 1.0, keep)                                # the foam fraction of the bands combined
        # roughness: σ² of the unresolved slopes → α ≈ σ (Beckmann), Blender roughness = √α
        s2 = g.m("MULTIPLY", unres, 0.4)
        rough = g.m("MINIMUM", g.m("MAXIMUM", g.m("POWER", s2, 0.25), 0.03), 0.65)
        # foam: patches (large noise) with ragged holes (fine noise) and threads along the cell edges; the more foam,
        # the more solid; the coordinates are stretched along the wind (streaks along the wind); in the distance the
        # pattern is replaced by its mean (a frame point is larger than the pattern)
        ca, sa = sea.wind_to
        along = g.m("ADD", g.m("MULTIPLY", g.sep(uv)[0], ca), g.m("MULTIPLY", g.sep(uv)[1], sa))
        cross = g.m("ADD", g.m("MULTIPLY", g.sep(uv)[0], -sa), g.m("MULTIPLY", g.sep(uv)[1], ca))
        fuv = g.xyz(g.m("MULTIPLY", along, 0.6), cross, 0.0)
        n1 = g.n("ShaderNodeTexNoise", noise_dimensions="2D"); g.put(n1.inputs["Vector"], fuv); n1.inputs["Scale"].default_value = 0.35
        n1.inputs["Detail"].default_value = 5.0; n1.inputs["Roughness"].default_value = 0.6
        n2 = g.n("ShaderNodeTexNoise", noise_dimensions="2D"); g.put(n2.inputs["Vector"], fuv); n2.inputs["Scale"].default_value = 4.0
        n2.inputs["Detail"].default_value = 3.0
        rd = g.n("ShaderNodeTexNoise", noise_dimensions="2D", noise_type="RIDGED_MULTIFRACTAL"); g.put(rd.inputs["Vector"], fuv)
        rd.inputs["Scale"].default_value = 1.8; rd.inputs["Detail"].default_value = 3.0; rd.inputs["Roughness"].default_value = 0.55
        rd.inputs["Distortion"].default_value = 0.6; rd.inputs["Offset"].default_value = 1.0; rd.inputs["Gain"].default_value = 2.0
        wf = g.ss(fp, 0.04, 0.5)                                         # 1 — the pattern is finer than a frame point
        jit = g.m("ADD", g.m("MULTIPLY", g.m("SUBTRACT", n1.outputs["Fac"], 0.5), 0.9), g.m("MULTIPLY", g.m("SUBTRACT", n2.outputs["Fac"], 0.5), 0.5))
        core = g.ss(g.m("ADD", foam, g.m("MULTIPLY", jit, g.m("SUBTRACT", 1.0, wf))), 0.45, 0.7)
        fil = g.m("MULTIPLY", g.ss(rd.outputs["Fac"], 0.75, 1.35), g.ss(foam, 0.03, 0.4))        # lace: a network of noise ridges
        fil = g.m("MULTIPLY", fil, g.m("SUBTRACT", 1.0, wf))
        fvis = g.m("MINIMUM", g.m("ADD", core, g.m("MULTIPLY", fil, 0.75)), 1.0)
        self.dbg = {"foam": foam, "fvis": fvis, "rough": rough, "lace": rd.outputs["Fac"], "hz": hz, "gust": gst, "att": att}
        if getattr(self, "_dbg_white", None) is not None: self.dbg["white"] = self._dbg_white
        fvis = g.m("ADD", g.m("MULTIPLY", fvis, g.m("SUBTRACT", 1.0, wf)), g.m("MULTIPLY", g.ss(foam, 0.1, 0.8), wf))
        fvis = g.m("MULTIPLY", fvis, 0.95)
        if self.storm:                                                  # storm (force 7+): foam of three ages — breaking now
            # (solid, the edge is torn into lace), recent (3 s memory: torn into lace by the marble tile), old (marble
            # veins along the wind; density — by force, by place — a ~30 s memory of breakings); the tile drifts with the
            # surface (3 % of the wind)
            A_old = float(np.interp(sea.u10, [10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7], [0.0, 0.02, 0.06, 0.12, 0.18, 0.26, 0.34]))
            al_d = g.m("SUBTRACT", along, g.m("MULTIPLY", tm.outputs[0], 0.03 * sea.u10))
            def mtex(s_, o_):
                t_ = g.n("ShaderNodeTexImage", image=self.mimg, interpolation="Linear", extension="REPEAT")
                g.put(t_.inputs[0], g.xyz(g.m("ADD", g.m("MULTIPLY", al_d, 1.0 / (self.mL * s_ * 4.0)), o_[0]),   # along the wind ×4
                                          g.m("ADD", g.m("MULTIPLY", cross, 2.8 / (self.mL * s_)), o_[1]), 0.0))   # across — finer
                return g.sep(t_.outputs["Color"])
            m1, m2 = mtex(1.0, (0.0, 0.0)), mtex(1.37, (0.31, 0.57))       # two tiles — no repetition; the max of two ranks: P(> τ) = 1 − τ²
            young, oldp = g.m("MAXIMUM", m1[0], m2[0]), g.m("MAXIMUM", m1[1], m2[1])
            now = g.m("SUBTRACT", 1.0, now_keep) if now_keep is not None else 0.0
            gate = old_sum if old_sum is not None else 0.0
            A_loc = g.m("MINIMUM", g.m("MULTIPLY", g.m("ADD", 0.55, g.m("MULTIPLY", gate, 0.45)), A_old), 0.5)
            def ramp(r_, A_):                                           # rank above the threshold τ = √(1 − A): the thread edge is pale, the core of
                tau = g.m("SQRT", g.m("SUBTRACT", 1.0, A_))             # a strong thread is white (the tile mean ≈ 0.5·A)
                return g.m("POWER", g.m("MAXIMUM", g.m("DIVIDE", g.m("SUBTRACT", r_, tau), g.m("MAXIMUM", g.m("SUBTRACT", 1.0, tau), 1e-3)), 0.0), 0.8)
            bub = g.n("ShaderNodeTexVoronoi", feature="DISTANCE_TO_EDGE"); g.put(bub.inputs["Vector"], uv); bub.inputs["Scale"].default_value = 5.0
            bub = g.m("ADD", g.m("MULTIPLY", g.ss(bub.outputs["Distance"], 0.0, 0.12), g.m("SUBTRACT", 1.0, g.ss(fp, 0.02, 0.1))),
                      g.ss(fp, 0.02, 0.1))                              # bubbles (~20 cm): up close the foam is a network of films, farther out — its mean
            m3 = mtex(0.3, (0.13, 0.71))                                   # threads inside a vein: the same tile finer (~6 cm)
            thr_ = g.m("ADD", g.m("MULTIPLY", ramp(m3[1], 0.5), g.m("SUBTRACT", 1.0, g.ss(fp, 0.03, 0.15))), g.m("MULTIPLY", g.ss(fp, 0.03, 0.15), 0.5))
            veins = g.m("MULTIPLY", g.m("MULTIPLY", g.m("POWER", ramp(oldp, A_loc), 1.3), 0.42),   # vein: a bright middle, a pale edge,
                        g.m("ADD", 0.3, g.m("MULTIPLY", g.m("MULTIPLY", bub, thr_), 0.7 / 0.5)))            # inside — threads and bubbles
            mid = g.n("ShaderNodeTexNoise", noise_dimensions="4D"); g.put(mid.inputs["Vector"], g.v("SCALE", geo.outputs["Position"], s=1 / 8.0))
            g.put(mid.inputs["W"], g.m("MULTIPLY", tm.outputs[0], 0.25)); mid.inputs["Detail"].default_value = 3.0; mid.inputs["Roughness"].default_value = 0.6
            holes = g.ss(mid.outputs["Fac"], 0.38, 0.6)                  # patches of ~8 m in the wave axes: the foam tears, dark water shows in the gaps
            A_f = g.m("MINIMUM", g.m("MULTIPLY", g.m("MULTIPLY", foam, 0.5), g.m("ADD", 0.25, g.m("MULTIPLY", holes, 1.1))), 1.0)
            lace_ = g.m("MULTIPLY", g.m("MINIMUM", g.m("MULTIPLY", ramp(young, A_f), 1.1), 0.9), g.m("ADD", 0.35, g.m("MULTIPLY", bub, 0.65)))
            crest_ = g.ss(hz, 0.12 * st["hs"], 0.4 * st["hs"])           # fresh foam — on the crests (breaking occurs at the top of the wave)
            boil = g.n("ShaderNodeTexNoise", noise_dimensions="4D"); g.put(boil.inputs["Vector"], g.v("SCALE", geo.outputs["Position"], s=0.45))
            g.put(boil.inputs["W"], g.m("MULTIPLY", tm.outputs[0], 1.2)); boil.inputs["Detail"].default_value = 4.0   # the breaker is in the wave axes:
            core_ = g.m("MULTIPLY", g.ss(g.m("ADD", g.m("ADD", g.m("MULTIPLY", now, 1.6), g.m("MULTIPLY", g.m("SUBTRACT", boil.outputs["Fac"], 0.5), 0.8)),
                                             g.m("MULTIPLY", g.m("SUBTRACT", holes, 0.5), 0.9)), 0.35, 0.6), crest_)   # moves with the crest, boils, tears
            pat = g.m("MAXIMUM", core_, g.m("MAXIMUM", lace_, veins))
            mean = g.m("MAXIMUM", g.m("MULTIPLY", g.ss(g.m("MULTIPLY", now, crest_), 0.0, 0.9), 0.75),   # far away — softly: a ~10 m field with a sharp
                       g.m("MAXIMUM", g.m("MULTIPLY", A_f, 0.6), g.m("MULTIPLY", A_loc, 0.35)))               # threshold gives "bricks" on the crests
            fvis = g.m("ADD", pat, g.m("MULTIPLY", g.m("SUBTRACT", mean, pat), g.ss(fp, 0.06, 0.3)))   # far away: veins finer than a point — by their mean
            fbump = g.m("ADD", g.m("MULTIPLY", young, 0.6), g.m("MULTIPLY", bub, 0.4)); self._fbump = fbump   # foam is a mass: relief
            self.dbg.update({"veins": veins, "now": now, "gate": g.m("MULTIPLY", gate, 0.5), "fvis": fvis})
        # water: a dielectric, IOR 1.333, the water column — diffuse (subsurface reflection of the column)
        bsdf = g.n("ShaderNodeBsdfPrincipled")
        g.put(bsdf.inputs["Base Color"], self.body); bsdf.inputs["IOR"].default_value = 1.333
        g.put(bsdf.inputs["Roughness"], rough); g.put(bsdf.inputs["Normal"], nrm)
        fb = g.n("ShaderNodeBsdfPrincipled"); g.put(fb.inputs["Base Color"], self.foam_rgb); fb.inputs["Roughness"].default_value = 0.7
        if getattr(self, "_fbump", None) is not None:                   # storm foam — uneven brightness (thickness, bubbles)
            g.put(fb.inputs["Base Color"], g.v("SCALE", g.xyz(*self.foam_rgb), s=g.m("ADD", 0.62, g.m("MULTIPLY", self._fbump, 0.45))))
        if getattr(self, "_fbump", None) is not None:                   # storm foam is a lumpy mass (it catches light), not a film
            bp = g.n("ShaderNodeBump"); bp.inputs["Strength"].default_value = 0.6; bp.inputs["Distance"].default_value = 0.08
            g.put(bp.inputs["Height"], self._fbump); g.put(bp.inputs["Normal"], nrm); g.put(fb.inputs["Normal"], bp.outputs[0])
        else: g.put(fb.inputs["Normal"], nrm)
        wsh = bsdf.outputs[0]
        if self.ground is not None and self.K is not None:              # shoal: the bed is visible through the water, the column attenuates it along the ray path
            depth = g.n("ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name="depth").outputs["Fac"]
            cn = g.m("ABSOLUTE", g.v("DOT_PRODUCT", nrm, geo.outputs["Incoming"]))
            mu = g.m("SQRT", g.m("MAXIMUM", g.m("SUBTRACT", 1.0, g.m("DIVIDE", g.m("SUBTRACT", 1.0, g.m("MULTIPLY", cn, cn)), 1.777)), 0.04))
            path = g.m("MULTIPLY", depth, g.m("ADD", g.m("DIVIDE", 1.0, mu), 1.25))        # down along the view and from the sun to the bed
            Tc = [g.m("EXPONENT", g.m("MULTIPLY", path, -float(k))) for k in self.K]
            Tg = g.m("ADD", g.m("ADD", g.m("MULTIPLY", Tc[0], 0.2126), g.m("MULTIPLY", Tc[1], 0.7152)), g.m("MULTIPLY", Tc[2], 0.0722))
            # far away: Eevee's screen-space refraction loses the bed (~20 m) — there the column and the bed are combined
            # as two streams: column·(1 − T) + bed·T
            g.put(bsdf.inputs["Base Color"], g.xyz(*[g.m("ADD", g.m("MULTIPLY", b_, g.m("SUBTRACT", 1.0, t_)), g.m("MULTIPLY", d_, t_))
                                                    for b_, d_, t_ in zip(self.body, self.bottom, Tc)]))
            wr = g.m("MULTIPLY", g.m("SUBTRACT", 1.0, g.ss(cd.outputs["View Distance"], self.refr_far * 0.6, self.refr_far)),
                     g.ss(cosv, 0.17, 0.24))                             # and by angle: shallower than ~10° to the water, the screen ray finds no bed
            win = g.sep(g.n("ShaderNodeTexCoord").outputs["Window"])       # and at the frame edges: the ray leaves the frame — and takes the sky
            zone = g.m("ADD", 0.14, g.m("MULTIPLY", g.m("MULTIPLY", g.m("MINIMUM", depth, 1.0), cosv), 0.35))   # the steeper the view and the deeper the bed —
            mrz = g.n("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP"); g.put(mrz.inputs["Value"], win[1])  # the farther the ray goes down out of the frame
            g.put(mrz.inputs["From Min"], g.m("SUBTRACT", zone, 0.12)); g.put(mrz.inputs["From Max"], zone)
            wr = g.m("MULTIPLY", wr, g.m("MULTIPLY", mrz.outputs[0], g.m("MULTIPLY", g.ss(win[0], 0.0, 0.06), g.ss(g.m("SUBTRACT", 1.0, win[0]), 0.0, 0.06))))
            tint = g.xyz(*[g.m("MINIMUM", g.m("DIVIDE", t_, g.m("MAXIMUM", Tg, 1e-4)), 2.0) for t_ in Tc])
            bt = g.n("ShaderNodeBsdfPrincipled"); g.put(bt.inputs["Base Color"], tint); bt.inputs["IOR"].default_value = 1.333
            bt.inputs["Transmission Weight"].default_value = 1.0; g.put(bt.inputs["Roughness"], rough); g.put(bt.inputs["Normal"], nrm)
            mx = g.n("ShaderNodeMixShader"); g.put(mx.inputs[0], g.m("MULTIPLY", Tg, wr)); g.put(mx.inputs[1], wsh); g.put(mx.inputs[2], bt.outputs[0])
            wsh = mx.outputs[0]; mat.use_raytrace_refraction = True; mat.thickness_mode = "SLAB"
            self.thin = True                                            # thickness 0: the ray to the bed without a second boundary (a slab)
            self.dbg["T"] = Tg; self.dbg["depth"] = g.m("MULTIPLY", depth, 0.2); self.dbg_refr = bt.outputs[0]
        if getattr(self, "_aer", None) is not None:                     # torrent: bubbles in the column — lighter and more turquoise
            bc = bsdf.inputs["Base Color"]; src = bc.links[0].from_socket if bc.is_linked else tuple(bc.default_value)[:3]
            g.put(bc, g.mix(self._aer, src, (0.09, 0.17, 0.16)))
        sh = g.n("ShaderNodeMixShader"); g.put(sh.inputs[0], fvis); g.put(sh.inputs[1], wsh); g.put(sh.inputs[2], fb.outputs[0])
        out_sh = sh.outputs[0]
        if self.sun:                                                     # light through the crests (Atlas, GDC 2019)
            L, Ls = self.sun
            V = geo.outputs["Incoming"]
            ldv = g.m("MAXIMUM", g.v("DOT_PRODUCT", L, g.v("SCALE", V, s=-1.0)), 0.0)
            ldn = g.v("DOT_PRODUCT", L, nrm)
            k1 = g.m("MULTIPLY", g.m("MULTIPLY", g.m("MAXIMUM", hz, 0.0), g.m("POWER", ldv, 4.0)),
                     g.m("POWER", g.m("MAXIMUM", g.m("SUBTRACT", 0.5, g.m("MULTIPLY", ldn, 0.5)), 0.0), 3.0))
            k1 = g.m("MULTIPLY", k1, 6.0 / max(st["hs"], 0.05))           # normalized by the wave height: the crest itself glows, not just any tall wave
            k2 = g.m("MULTIPLY", g.m("POWER", g.m("MAXIMUM", g.v("DOT_PRODUCT", V, nrm), 0.0), 2.0), 0.15)
            fr = g.n("ShaderNodeFresnel"); fr.inputs["IOR"].default_value = 1.333; g.put(fr.inputs["Normal"], nrm)
            amt = g.m("MULTIPLY", g.m("MULTIPLY", g.m("ADD", k1, k2), g.m("SUBTRACT", 1.0, fr.outputs[0])), g.m("SUBTRACT", 1.0, fvis))
            em = g.n("ShaderNodeEmission"); g.put(em.inputs["Color"], tuple(c * Ls / math.pi for c in self.scatter)); g.put(em.inputs["Strength"], amt)
            ad = g.n("ShaderNodeAddShader"); g.put(ad.inputs[0], out_sh); g.put(ad.inputs[1], em.outputs[0]); out_sh = ad.outputs[0]
        if self.sun: self.dbg["scatter"] = amt
        if self.debug == "refr" and getattr(self, "dbg_refr", None) is not None: out_sh = self.dbg_refr   # debug: refraction only
        if self.debug in self.dbg:                                       # debug: the value as gray emission
            em_ = g.n("ShaderNodeEmission"); g.put(em_.inputs["Color"], self.dbg[self.debug]); out_sh = em_.outputs[0]
        o = g.n("ShaderNodeOutputMaterial"); g.put(o.inputs["Surface"], out_sh)
        if getattr(self, "thin", False):                                 # thickness — by a node only: Eevee 5.2 does not accept a plain number in the socket
            th = g.n("ShaderNodeValue"); th.outputs[0].default_value = 0.0; g.put(o.inputs["Thickness"], th.outputs[0])   # (a plane has extent 0, the bed is not visible)
        return mat


def _upsample(F, n):
    """Field m×m → n×n (n is a multiple of m), bilinear and wrapping around."""
    m = F.shape[0]
    if m == n: return F
    c = np.arange(n) * m / n                                            # band point j is at x = j·L/N (the FFT grid)
    X, Y = np.meshgrid(c, c)
    return W.sample(F, float(m), np.stack([X.ravel(), Y.ravel()], 1)).reshape(n, n)
