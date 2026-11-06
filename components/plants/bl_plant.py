"""A plant as one model in Blender (tree, shrub, small plant, grass, flower — one logic: axes with branching orders,
organs on the axes). Input — a plant specification from plants.py (JSON: species, age, state, seed → parameters).
Woody axes (trunk, branches, roots) — the m_tree engine as is (its mesh transfer mesh_utils);
our own — roots as a second "tree" growing down with seating on the ground, trunk tilt, foliage as cards by mesh attributes (radius, direction),
herbaceous axes (stems, blades — ribbons), organs (leaf, branch cluster, needles, frond, flower), materials by species.
The functions are for the builders (forest, meadow, component step); the command — a sheet of species:
blender -b -P core/blender/run.py -- components/plants/bl_plant.py -- <plants.json> <out.blend> — all the file's plants in rows (a check)."""
import core.paths
import json, math, os, sys, time, types
import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); EXT = os.path.join(HERE, "ext")
core.paths.component_ext("plants", "mtree", "site"); core.paths.component_ext("plants")
if "mtree" not in sys.modules:                                       # their add-on package — as modules only (not registered in Blender)
    _pk = types.ModuleType("mtree"); _pk.__path__ = [os.path.join(EXT, "mtree")]; sys.modules["mtree"] = _pk
    _pc = types.ModuleType("mtree.python_classes"); _pc.__path__ = [os.path.join(EXT, "mtree", "python_classes")]; sys.modules["mtree.python_classes"] = _pc
from m_tree import m_tree as M                                        # noqa: E402
from mtree.python_classes.presets import tree_presets as TP           # noqa: E402  (their setup of branch parameters: nested fields, wrappers)
from mtree.python_classes.mesh_utils import create_mesh_from_cpp      # noqa: E402

TEX = os.path.join(core.paths.ROOT, "library", "plants", "tex")           # organ images (leaves.py), cached by type
_MATS = {}


# ---------------------------------------------------------------- woody axes: the m_tree engine
def _trunk(d, seed):
    t = M.TrunkFunction(); t.seed = int(seed)
    for k, v in d.items(): setattr(t, k, v)
    return t


def _branch(d, seed):
    b = M.BranchFunction(); b.seed = int(seed)
    for k, v in d.items(): TP._set_branch_param(b, k, v)
    return b


def woody_mesh(trunk, levels, seed, radial=12, smooth=2, growth=None):
    """Trunk + branch levels (each next one is the children of the previous) → a C++ mesh (one, seamless)."""
    tree = M.Tree(); tr = _trunk(trunk, seed); parent = tr
    for i, lv in enumerate(levels):
        b = _branch(lv, seed + 1 + i); parent.add_child(b); parent = b
    if growth:
        g = M.GrowthFunction(); g.seed = int(seed) + 9
        for k, v in growth.items(): setattr(g, k, v)
        parent.add_child(g)
    tree.set_trunk_function(tr); tree.execute_functions()
    mh = M.ManifoldMesher(); mh.radial_n_points = int(radial); mh.smooth_iterations = int(smooth)
    return mh.mesh_tree(tree)


def mesh_arrays(me):
    """A Blender mesh → (V, N, radius, direction) numpy."""
    n = len(me.vertices); V = np.empty(n * 3); me.vertices.foreach_get("co", V); Nn = np.empty(n * 3); me.vertices.foreach_get("normal", Nn)
    rad = np.zeros(n); D = np.zeros(n * 3)
    if "radius" in me.attributes: me.attributes["radius"].data.foreach_get("value", rad)
    if "direction" in me.attributes: me.attributes["direction"].data.foreach_get("vector", D)
    return V.reshape(-1, 3), Nn.reshape(-1, 3), rad, D.reshape(-1, 3)


def lean(P, H, deg, az, recover):
    """Trunk tilt (slope, light, wind): the axis leans from the vertical by deg at the ground and straightens toward the top by the fraction recover
    (phototropism, a "sabre-shaped" trunk) — the offset by height x(z) = tg·(z − rec·z²/2H)."""
    if not deg: return P
    t = math.tan(math.radians(deg)); z = np.clip(P[:, 2], 0, None); s = t * (z - recover * z * z / (2 * max(H, 1e-3)))
    Q = P.copy(); Q[:, 0] += s * math.cos(math.radians(az)); Q[:, 1] += s * math.sin(math.radians(az)); return Q


def crown_push(P, z0, H, push):
    """Crown toward the light (a forest edge, a gap between neighbours): the top of the crown shifts by the vector push (m) starting from the crown base z0, fully at the
    apex; the trunk below z0 is not touched."""
    if push is None or not np.any(push): return P
    k = np.clip((P[:, 2] - z0) / max(H - z0, 1e-3), 0, 1) ** 1.5; Q = P.copy(); Q[:, 0] += k * push[0]; Q[:, 1] += k * push[1]; return Q


def root_mesh(rs, r0, seed):
    """The root system is a second tree of the m_tree engine: a short "trunk" inside the root flare, the main roots — branches almost horizontal,
    sub-roots — their children. Type (species knowledge): flat (plate) — many roots at the surface; heart — slanting down; taproot — little
    visible; buttress — tall plates (compressed across)."""
    typ = rs.get("type", "heart"); n = rs.get("n", 8); L = rs.get("spread", 2.5)
    ang = {"plate": 92, "heart": 104, "tap": 120, "buttress": 95, "stilt": 140}.get(typ, 100)
    tr = {"length": 0.25, "start_radius": r0 * rs.get("flare", 1.35), "end_radius": r0, "shape": 0.6, "up_attraction": 1.0, "resolution": 6, "randomness": 0.0}
    lv = [{"start": 0.0, "end": 0.45, "length": L, "branches_density": n / (0.45 * 0.25), "start_angle": ang, "randomness": 0.45, "flatness": 1.0,
           "up_attraction": -0.05, "gravity_strength": 1.0, "stiffness": 0.3, "split_proba": 0.3, "split_radius": 0.6, "split_angle": 30,
           "phillotaxis": 137.5, "break_chance": 0.0, "resolution": 4, "start_radius": 0.92}]   # roots at the base are thick — a transition into the flare
    if L > 1.2: lv.append({"start": 0.2, "end": 0.9, "length": L * 0.3, "branches_density": 1.5, "start_angle": 35, "randomness": 0.4, "flatness": 1.0,
                           "up_attraction": -0.6, "gravity_strength": 3.0, "stiffness": 0.2, "split_proba": 0.05, "break_chance": 0.0, "resolution": 3,
                           "start_radius": 0.45})                    # sub-roots run shallow and down (otherwise they stick up like spikes)
    return woody_mesh(tr, lv, seed + 101, radial=10, smooth=2)


def seat_roots(P, Nn, rad, rs, ground, x0, y0, z_base, depth=None):
    """Seating the roots on the ground by a smooth transformation (without restoring the axes — that breaks on the smoothed mesh joints):
    the root follows the terrain; at the trunk it is raised by the exposure fraction exposure × the root radius, toward the edge the lift fades over the length
    spread × exposed_len and the root dives into the soil along its own slope; downhill it is more exposed (erosion); the planks (buttress) are
    stretched upward at the trunk. The roots' "trunk" (the flare, order 0) is in the soil, only its edge is above ground."""
    ex = float(rs.get("exposure", 0.35)); Lx = max(float(rs.get("spread", 2.5)) * float(rs.get("exposed_len", 0.45)), 0.3)
    d = np.hypot(P[:, 0], P[:, 1]); gz = (ground(x0 + P[:, 0], y0 + P[:, 1]) - z_base) if ground else np.zeros(len(P))
    side = np.zeros(len(P))
    if ground:                                                        # toward the downhill side — more is exposed
        e = 0.5; gx = float(ground(np.array([x0 + e]), np.array([y0]))[0] - ground(np.array([x0 - e]), np.array([y0]))[0]) / (2 * e)
        gy = float(ground(np.array([x0]), np.array([y0 + e]))[0] - ground(np.array([x0]), np.array([y0 - e]))[0]) / (2 * e); gn = math.hypot(gx, gy)
        if gn > 1e-4: side = -(P[:, 0] * gx + P[:, 1] * gy) / (np.maximum(d, 1e-3) * gn) * min(gn * 3, 1.0)
    trunk = (depth < 0.5) if depth is not None else (d < 0.05)
    R = float(np.quantile(rad[~trunk], 0.9)) if (~trunk).any() else float(rad.max())
    fall = np.exp(-d / Lx); lift = ex * R * 0.9 * (1 + 0.6 * side) * fall
    dive = R * 2.5 * np.clip(d / Lx, 0, None) ** 1.5                  # toward the edge the root dives into the soil — the ends and broken stubs of the sub-roots do not stick out
    z = P[:, 2].copy()
    if rs.get("type") == "buttress": z = np.where(z > 0, z * (1 + 0.5 * float(rs.get("height", 2.0)) * np.exp(-d / (0.4 * Lx))), z)
    Q = P.copy(); Q[:, 2] = z + gz + lift - dive
    Q[trunk, 2] = P[trunk, 2] + gz[trunk] - 0.18
    return Q


# ---------------------------------------------------------------- organs: cards and ribbons
def _tex_mean(im):
    """Mean colour of the opaque part of an organ image, linear (an 8-bit PNG in Blender keeps sRGB values as in the file)."""
    key = ("mean", im.name)
    if key not in _MATS:
        px = np.empty(len(im.pixels), np.float32); im.pixels.foreach_get(px); px = px.reshape(-1, 4); m = px[:, 3] > 0.5
        c = px[m, :3].mean(0) if m.any() else np.full(3, 0.5, np.float32)
        if not im.is_float: c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        _MATS[key] = np.maximum(c, 1e-3)
    return _MATS[key]


def sky_visibility(C, area, ndir=12, G=0.4, seed=0):
    """The share of sky above each organ of the crown (Reeves & Blau 1985 — light by depth in a crown; Beer's law in a canopy, Monsi & Saeki 1953):
    the foliage — an area cloud in a grid (leaf area density, m²/m³), from the card centre — ndir rays over the upper hemisphere
    (cosine weight), τ = G · Σ density · step (G — the leaf's shadow on the ray: projection ½ × clumping ~0.8); V = mean of e^(−τ). It does not
    depend on the light — computed at build time; Eevee does not occlude the sky with the crown (world light comes from probes), Cycles occludes it itself."""
    C = np.asarray(C, np.float64); lo = C.min(0) - 0.5; ext = C.max(0) + 0.5 - lo; h = max(0.4, float(ext.max()) / 32.0)
    dims = np.ceil(ext / h).astype(int) + 1; idx = np.clip(((C - lo) / h).astype(int), 0, dims - 1)
    lad = np.zeros(dims, np.float32); np.add.at(lad, tuple(idx.T), np.asarray(area, np.float32)); lad /= h ** 3
    r = np.random.default_rng(seed); u = (np.arange(ndir) + r.random(ndir)) / ndir; ph = r.random(ndir) * 2 * np.pi
    ct = np.sqrt(1.0 - u); st = np.sqrt(np.maximum(0.0, 1.0 - ct * ct))    # cosine weight: cos θ = √(1 − u)
    D = np.column_stack([st * np.cos(ph), st * np.sin(ph), ct]); K = int(np.linalg.norm(ext) / h) + 2; s_ = h * (np.arange(K) + 1.0)
    cell, inv = np.unique(idx, axis=0, return_inverse=True); P0 = lo + (cell + 0.5) * h   # we compute over the occupied cells — smoothly and several times faster
    V = np.zeros(len(cell))
    for d in D:
        Q = ((P0[:, None, :] + d[None, None, :] * s_[None, :, None] - lo) / h).astype(int)
        ok = np.all((Q >= 0) & (Q < dims), axis=2); Q = np.clip(Q, 0, dims - 1)
        tau = G * h * np.where(ok, lad[Q[..., 0], Q[..., 1], Q[..., 2]], 0.0).sum(1); V += np.exp(-tau)
    V = V[inv.ravel()]
    return (V / ndir).astype(np.float32)


def _tex_cov(mat):
    """The opaque share of an organ image (leaf area per card)."""
    im = next((n.image for n in mat.node_tree.nodes if n.type == "TEX_IMAGE" and n.image), None)
    if im is None: return 0.3
    key = ("cov", im.name)
    if key not in _MATS:
        px = np.empty(len(im.pixels), np.float32); im.pixels.foreach_get(px); _MATS[key] = float(px[3::4].mean())
    return _MATS[key]


def _organ(png, kind):
    b = os.path.basename(png)[:-4]
    if kind == "flower" or b.startswith("flower"): return "flower"
    if kind == "blade" or b == "blade": return "blade"
    return "shoot" if b in ("pine", "spruce", "fir", "scale") else "leaf"


def card_material(png, name=None, kind="leaf", cheap=False, optics=None):
    """An organ card: an RGBA image × a tint by corners (attribute tint) × the species' optics (optics from plants.organ_optics):
    leaf — the image's mean colour is fitted to the leaf-body reflectance Rd and to the transmittance T (PROSPECT-D: yellow-green when backlit, gold under a
    warm sun), needles — the image as is (the shadow inside the shoot is painted in), translucency — T/R; a grass blade — the species colour (tint),
    translucency — T/Rd of the grass; a flower — translucency 0.6. Cuticle gloss — the species' roughness, IOR 1.45; the underside is lighter (under) and more matte.
    Thickness 0 — a single boundary: by default Eevee takes the thickness from the object (for a tree — metres) and the translucency is 5 times weaker (measured;
    equals Cycles at 0). Alpha — a cut-out (Dithered). cheap — the far tier: the shadow has no transparency."""
    O = optics or {}; mode = _organ(png, kind)
    key = (png, kind, cheap, json.dumps(O, sort_keys=True) if O else None)
    if key in _MATS: return _MATS[key]
    nm = name or ("card_" + os.path.basename(png)[:-4] + ("_far" if cheap else "") + (("_" + O["set"]) if O.get("set") else ""))
    m = bpy.data.materials.new(nm); m.surface_render_method = "DITHERED"; m.use_backface_culling = False
    m.use_transparent_shadow = not cheap; m.thickness_mode = "SLAB"
    nt = m.node_tree; N_, L_ = nt.nodes, nt.links; out = N_.get("Material Output"); bs = N_.get("Principled BSDF")
    sid = lambda ss, i: next(x for x in ss if x.identifier == i)
    im = N_.new("ShaderNodeTexImage"); im.image = bpy.data.images.load(png, check_existing=True)
    att = N_.new("ShaderNodeAttribute"); att.attribute_name = "tint"; att.attribute_type = "GEOMETRY"
    mix = N_.new("ShaderNodeMix"); mix.data_type = "RGBA"; mix.blend_type = "MULTIPLY"; mix.inputs["Factor"].default_value = 1.0
    L_.new(im.outputs["Color"], sid(mix.inputs, "A_Color")); L_.new(att.outputs["Color"], sid(mix.inputs, "B_Color"))
    kR, kT, rough, under = np.ones(3), np.ones(3), (0.5 if mode != "blade" else 0.65), 1.0
    if O:
        mean = _tex_mean(im.image); rough = float(O.get("rough", rough))
        if mode == "leaf": kR, kT, under = np.asarray(O["Rd"]) / mean, np.asarray(O["T"]) / mean, float(O.get("under", 1.3))
        elif mode == "shoot": kT = np.clip(np.asarray(O["T"]) / np.maximum(np.asarray(O["R"]), 1e-3), 0, 1)
        elif mode == "blade":
            g = O.get("grass") or O; kT = np.clip(np.asarray(g["T"]) / np.maximum(np.asarray(g["Rd"]), 1e-3), 0, 3); under = 1.1
        else: kT = np.full(3, 0.6)
    def scaled(k):
        s_ = N_.new("ShaderNodeMix"); s_.data_type = "RGBA"; s_.blend_type = "MULTIPLY"; s_.inputs["Factor"].default_value = 1.0
        L_.new(sid(mix.outputs, "Result_Color"), sid(s_.inputs, "A_Color")); sid(s_.inputs, "B_Color").default_value = (*[float(x) for x in k], 1.0)
        return sid(s_.outputs, "Result_Color")
    base, trans = scaled(kR), scaled(kT)
    sky_k = float(O.get("sky", 0.0)) if O else 0.0                   # disabled: it also darkens the sun (measured: the foliage is 2–3 times darker than Cycles)
    if sky_k > 0:                                                     # the sky share above a leaf (sk_sky, at build time): Eevee does not occlude with the crown by itself;
        av = N_.new("ShaderNodeAttribute"); av.attribute_name = "sk_sky"; av.attribute_type = "GEOMETRY"   # Cycles does occlude: strength sk_sky_k = 0
        kv = N_.new("ShaderNodeValue"); kv.name = "sk_sky_k"; kv.outputs[0].default_value = sky_k; m["sk_sky_k"] = sky_k
        k1 = N_.new("ShaderNodeMath"); k1.operation = "SUBTRACT"; k1.inputs[0].default_value = 1.0; L_.new(kv.outputs[0], k1.inputs[1])
        fk = N_.new("ShaderNodeMath"); fk.operation = "MULTIPLY_ADD"; L_.new(av.outputs["Fac"], fk.inputs[0]); L_.new(kv.outputs[0], fk.inputs[1])
        L_.new(k1.outputs[0], fk.inputs[2])
        def occl(c):
            v_ = N_.new("ShaderNodeVectorMath"); v_.operation = "SCALE"; L_.new(c, v_.inputs[0]); L_.new(fk.outputs[0], v_.inputs["Scale"]); return v_.outputs[0]
        if O.get("sky_mode") == "trans": trans = occl(trans)            # translucency only: the sky from above through a leaf deep in the crown (evening under the canopy)
        else:
            base, trans = occl(base), occl(trans)
            sp_ = N_.new("ShaderNodeMath"); sp_.operation = "MULTIPLY"; L_.new(fk.outputs[0], sp_.inputs[0]); sp_.inputs[1].default_value = 0.5
            L_.new(sp_.outputs[0], bs.inputs["Specular IOR Level"])
    if under != 1.0:                                                  # the underside (the camera sees the back of the card): lighter, more matte
        geo = N_.new("ShaderNodeNewGeometry"); uf = N_.new("ShaderNodeMath"); uf.operation = "MULTIPLY_ADD"
        L_.new(geo.outputs["Backfacing"], uf.inputs[0]); uf.inputs[1].default_value = under - 1.0; uf.inputs[2].default_value = 1.0
        sc_ = N_.new("ShaderNodeVectorMath"); sc_.operation = "SCALE"; L_.new(base, sc_.inputs[0]); L_.new(uf.outputs[0], sc_.inputs["Scale"]); base = sc_.outputs[0]
        ru = N_.new("ShaderNodeMath"); ru.operation = "MULTIPLY_ADD"; L_.new(geo.outputs["Backfacing"], ru.inputs[0]); ru.inputs[1].default_value = 0.15
        ru.inputs[2].default_value = rough; L_.new(ru.outputs[0], bs.inputs["Roughness"])
    else: bs.inputs["Roughness"].default_value = rough
    tr = N_.new("ShaderNodeBsdfTranslucent"); ad = N_.new("ShaderNodeAddShader"); ms = N_.new("ShaderNodeMixShader"); tp = N_.new("ShaderNodeBsdfTransparent")
    L_.new(base, bs.inputs["Base Color"]); L_.new(trans, tr.inputs["Color"])
    tr_out = tr.outputs[0]
    bs.inputs["IOR"].default_value = 1.45; bs.inputs["Specular IOR Level"].default_value = 0.5 if O else 0.35
    L_.new(bs.outputs[0], ad.inputs[0]); L_.new(tr_out, ad.inputs[1])
    L_.new(im.outputs["Alpha"], ms.inputs["Fac"]); L_.new(tp.outputs[0], ms.inputs[1]); L_.new(ad.outputs[0], ms.inputs[2])
    th = N_.new("ShaderNodeValue"); th.name = "sk_thick"; th.outputs[0].default_value = 0.0; L_.new(th.outputs[0], out.inputs["Thickness"])
    m["sk_leak"] = float(np.dot([0.2126, 0.7152, 0.0722], np.asarray(O["T"]))) if O.get("T") else 0.0   # a leaf's shadow lets light through, like the leaf itself (bl_veg_light)
    px_ = np.empty(len(im.image.pixels), np.float32); im.image.pixels.foreach_get(px_); m["sk_cov"] = float(px_[3::4].mean())   # leaf share in the card
    L_.new(ms.outputs[0], out.inputs["Surface"]); _MATS[key] = m; return m


def bark_material(b):
    """Bark by species in the m_tree engine's unwrap (u — around, v — along the axis: the pattern runs along each branch): the pattern pattern —
    furrow (furrows: oak, poplar, willow), plates (plates: pine, larch), smooth (beech, fir), lenticel (birch: dark lenticels
    across), rings (palm: leaf-scar rings); tones c0 (furrow) / c1 (bark), noise — blotches; the relief — from the same pattern."""
    key = ("bark", json.dumps(b, sort_keys=True))
    if key in _MATS: return _MATS[key]
    sc_ = None
    if b.get("scan"):
        from components.plants import bark_scans; sc_ = bark_scans.paths(b["scan"])
    if sc_ and all(os.path.exists(f) for f in sc_.values()):          # a Poly Haven scan over the unwrap (bark_scans.py)
        m = bpy.data.materials.new("bark_" + b.get("name", "x")); nt = m.node_tree; N_, L_ = nt.nodes, nt.links; bs = N_.get("Principled BSDF")
        uv = N_.new("ShaderNodeUVMap"); uv.uv_map = "UVMap"; mp = N_.new("ShaderNodeMapping"); mp.inputs["Scale"].default_value = b.get("scan_scale", (2.0, 0.7, 1.0))
        L_.new(uv.outputs["UV"], mp.inputs["Vector"])
        def tex(path, data):
            t = N_.new("ShaderNodeTexImage"); t.image = bpy.data.images.load(path, check_existing=True)
            if data: t.image.colorspace_settings.name = "Non-Color"
            L_.new(mp.outputs["Vector"], t.inputs["Vector"]); return t
        dif, nor, rgh = tex(sc_["diff"], False), tex(sc_["nor_gl"], True), tex(sc_["rough"], True)
        hue = N_.new("ShaderNodeMix"); hue.data_type = "RGBA"; hue.blend_type = "MULTIPLY"; hue.inputs["Factor"].default_value = b.get("scan_tint_k", 0.35)
        sid = lambda ss, i: next(x for x in ss if x.identifier == i)
        L_.new(dif.outputs["Color"], sid(hue.inputs, "A_Color")); c1 = b.get("c1", [0.3, 0.27, 0.22]); k = 0.3 / max(np.mean(c1), 1e-3)
        sid(hue.inputs, "B_Color").default_value = (*[min(1.0, c * k) for c in c1], 1)   # toward the species tone — slightly
        L_.new(sid(hue.outputs, "Result_Color"), bs.inputs["Base Color"]); L_.new(rgh.outputs["Color"], bs.inputs["Roughness"])
        nm = N_.new("ShaderNodeNormalMap"); nm.inputs["Strength"].default_value = 1.0; L_.new(nor.outputs["Color"], nm.inputs["Color"]); L_.new(nm.outputs["Normal"], bs.inputs["Normal"])
        _MATS[key] = m; return m
    pat = b.get("pattern", "furrow"); m = bpy.data.materials.new("bark_" + b.get("name", "x")); nt = m.node_tree; N_, L_ = nt.nodes, nt.links
    bs = N_.get("Principled BSDF"); sid = lambda ss, i: next(x for x in ss if x.identifier == i)
    uv = N_.new("ShaderNodeUVMap"); uv.uv_map = "UVMap"; mp = N_.new("ShaderNodeMapping")
    sc = {"furrow": (7, 1.6, 1), "plates": (5, 1.2, 1), "smooth": (3, 1.0, 1), "lenticel": (2, 9.0, 1), "rings": (1.5, 12.0, 1)}[pat]
    mp.inputs["Scale"].default_value = b.get("uvscale", sc); L_.new(uv.outputs["UV"], mp.inputs["Vector"])
    vo = N_.new("ShaderNodeTexVoronoi"); vo.feature = "DISTANCE_TO_EDGE"; vo.inputs["Scale"].default_value = 1.0
    nz = N_.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 4.0; nz.inputs["Detail"].default_value = 6
    L_.new(mp.outputs["Vector"], vo.inputs["Vector"]); L_.new(mp.outputs["Vector"], nz.inputs["Vector"])
    edge = N_.new("ShaderNodeMapRange"); edge.inputs["From Min"].default_value = 0.0; edge.inputs["From Max"].default_value = b.get("furrow_w", 0.12 if pat != "smooth" else 0.03)
    L_.new(vo.outputs["Distance"], edge.inputs["Value"])                # 0 — in a furrow, 1 — on the bark
    if pat in ("lenticel", "rings"):                                    # stripes across the axis: a sine along v with noise
        sep = N_.new("ShaderNodeSeparateXYZ"); L_.new(mp.outputs["Vector"], sep.inputs[0]); wv = N_.new("ShaderNodeMath"); wv.operation = "SINE"
        ad = N_.new("ShaderNodeMath"); ad.operation = "ADD"; L_.new(sep.outputs["Y"], ad.inputs[0]); L_.new(nz.outputs["Fac"], ad.inputs[1])
        L_.new(ad.outputs[0], wv.inputs[0]); band = N_.new("ShaderNodeMapRange"); band.inputs["From Min"].default_value = 0.75 if pat == "lenticel" else 0.2
        band.inputs["From Max"].default_value = 0.95 if pat == "lenticel" else 0.9; L_.new(wv.outputs[0], band.inputs["Value"])
        inv = N_.new("ShaderNodeMath"); inv.operation = "SUBTRACT"; inv.inputs[0].default_value = 1.0; L_.new(band.outputs["Result"], inv.inputs[1])
        mask = inv.outputs[0] if pat == "lenticel" else band.outputs["Result"]
        if pat == "lenticel":                                           # lenticels are short: × noise patches
            ms = N_.new("ShaderNodeMath"); ms.operation = "MAXIMUM"; L_.new(inv.outputs[0], ms.inputs[0])
            gt = N_.new("ShaderNodeMath"); gt.operation = "GREATER_THAN"; gt.inputs[1].default_value = 0.55; L_.new(nz.outputs["Fac"], gt.inputs[0]); L_.new(gt.outputs[0], ms.inputs[1]); mask = ms.outputs[0]
    else: mask = edge.outputs["Result"]
    mix = N_.new("ShaderNodeMix"); mix.data_type = "RGBA"; c0, c1 = b.get("c0", [0.10, 0.085, 0.07]), b.get("c1", [0.24, 0.21, 0.17])
    sid(mix.inputs, "A_Color").default_value = (*c0, 1); sid(mix.inputs, "B_Color").default_value = (*c1, 1); L_.new(mask, mix.inputs["Factor"])
    var = N_.new("ShaderNodeMix"); var.data_type = "RGBA"; var.blend_type = "MULTIPLY"; var.inputs["Factor"].default_value = 1.0
    L_.new(sid(mix.outputs, "Result_Color"), sid(var.inputs, "A_Color"))
    vr = N_.new("ShaderNodeMapRange"); vr.inputs["To Min"].default_value = 0.75; vr.inputs["To Max"].default_value = 1.2; L_.new(nz.outputs["Fac"], vr.inputs["Value"])
    L_.new(vr.outputs["Result"], sid(var.inputs, "B_Color")); L_.new(sid(var.outputs, "Result_Color"), bs.inputs["Base Color"])
    bp = N_.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = b.get("bump", 0.6) * 0.6; bp.inputs["Distance"].default_value = 0.02
    L_.new(mask, bp.inputs["Height"]); L_.new(bp.outputs["Normal"], bs.inputs["Normal"])
    bs.inputs["Roughness"].default_value = b.get("rough", 0.85); _MATS[key] = m; return m


def stem_material(c):
    key = ("stem", tuple(c))
    if key in _MATS: return _MATS[key]
    m = bpy.data.materials.new("stem"); bs = m.node_tree.nodes.get("Principled BSDF"); bs.inputs["Base Color"].default_value = (*c, 1)
    bs.inputs["Roughness"].default_value = 0.6; _MATS[key] = m; return m


def make_mesh(name, V, F, uv=None, tint=None, mat=None, smooth=True):
    """A mesh from arrays (without from_pydata): V (n,3), F (m,k) — faces of one length k; uv/tint — per corner."""
    me = bpy.data.meshes.new(name); F = np.asarray(F, np.int32); k = F.shape[1]
    me.vertices.add(len(V)); me.vertices.foreach_set("co", np.asarray(V, np.float32).ravel())
    me.loops.add(F.size); me.loops.foreach_set("vertex_index", F.ravel())
    me.polygons.add(len(F)); me.polygons.foreach_set("loop_start", np.arange(0, F.size, k, dtype=np.int32))
    me.update(calc_edges=True)
    if uv is not None: me.uv_layers.new(name="UVMap").data.foreach_set("uv", np.asarray(uv, np.float32).ravel())
    if tint is not None:
        me.attributes.new("tint", "FLOAT_COLOR", "CORNER").data.foreach_set("color", np.asarray(tint, np.float32).ravel())
        me.attributes.new("sk_sky", "FLOAT", "POINT").data.foreach_set("value", np.ones(len(V), np.float32))   # sky share (the crown foliage has its own)
    if smooth: me.polygons.foreach_set("use_smooth", np.ones(len(F), bool))
    if mat: me.materials.append(mat)
    return me


def cards(P, L, Nl, size, aspect=1.0, fold=0.15, uvq=None):
    """Organ cards: base at P, length along L (unit), a plane with normal Nl, width = size·aspect; a fold along the midrib.
    Returns (V, F, uv) — 6 vertices and 2 quads per card."""
    n = len(P); s = np.asarray(size, float).reshape(-1, 1) * np.ones((n, 1)); w = s * aspect * 0.5; S = np.cross(Nl, L)
    S /= np.linalg.norm(S, axis=1, keepdims=True) + 1e-9; fo = Nl * (w * fold)
    q = np.stack([P - S * w, P + fo, P + S * w, P + S * w + L * s, P + L * s + fo, P - S * w + L * s], 1)
    F = (np.array([[0, 1, 4, 5], [1, 2, 3, 4]])[None] + 6 * np.arange(n)[:, None, None]).reshape(-1, 4)
    uv = np.array([[0, 0], [0.5, 0], [1, 0], [1, 1], [0.5, 1], [0, 1]], float)[[0, 1, 4, 5, 1, 2, 3, 4]]
    return q.reshape(-1, 3), F, np.tile(uv, (n, 1))


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def foliage(V, Nn, rad, D, lf, seed, H):
    """Foliage on the thin branches of a tree: the points are bark vertices with a radius below the quantile q (branch tips); the bias is toward the outer
    crown shell and the top (that is where the light is: the inner crown is bare — as in light-driven growth models, Pałubicki 2009); a card sits not on
    the bark itself but in a cloud around the branch of radius reach (invisible small twigs carry the clusters); the count is density × the length of thin branches,
    at most max. A leaf/cluster is at an angle to the branch (the petiole angle), its plane facing the light (up); needles run along the shoot ("paws")."""
    r = np.random.default_rng(seed); typ = lf["type"]; conif = typ in ("pine", "spruce", "fir", "scale")
    br = rad > 0
    if not br.any(): return None
    thr = np.quantile(rad[br], lf.get("q", 0.35)); cand = np.nonzero(br & (rad <= thr) & (V[:, 2] > lf.get("zmin", 0.0) * H))[0]
    if not len(cand): return None
    radial = lf.get("_radial", 12); seglen = lf.get("_seg", 0.33)
    twig_m = len(cand) / radial * seglen                              # length of the thin branches, m (ring vertices × step)
    n = int(min(lf.get("max", 45000), max(80, lf.get("per_m", 6.0) * twig_m)))
    C = V[cand]; cxy = np.median(C[:, :2], 0); rr = np.hypot(C[:, 0] - cxy[0], C[:, 1] - cxy[1]); rn = rr / max(np.quantile(rr, 0.98), 1e-3)
    hn = (C[:, 2] - C[:, 2].min()) / max(np.ptp(C[:, 2]), 1e-3)
    w = (0.2 + np.clip(rn, 0, 1.2)) ** lf.get("shell", 1.6) * (0.55 + hn) ** lf.get("top", 0.8); w /= w.sum()
    i = cand[r.choice(len(cand), n, p=w)]; P, n_, d = V[i], Nn[i], _unit(D[i] + 1e-6)
    reach = lf.get("reach", lf["size"] * 1.3)                         # a cloud of clusters around the branch
    perp = _unit(np.cross(d, r.normal(0, 1, (n, 3)))); off = (n_ * 0.6 + perp) * (reach * np.sqrt(r.random((n, 1))))
    out_ = _unit(np.column_stack([P[:, 0] - cxy[0], P[:, 1] - cxy[1], np.zeros(n)]) + 1e-6)
    P = P + off + out_ * reach * 0.35 * r.random((n, 1))              # slightly toward the outside of the crown
    a = r.uniform(*lf.get("angle", (0.15, 0.6) if conif else (0.6, 1.2)), (n, 1))
    L = d * np.cos(a) + n_ * np.sin(a); L = _unit(L + r.normal(0, lf.get("jitter", 0.3), (n, 3)) + out_ * 0.3 + np.array([0, 0, -lf.get("droop", 0.15)]))
    up = np.array([0, 0, 1.0]) * lf.get("up", 0.55) + out_ * 0.35 + r.normal(0, lf.get("spread", 0.45), (n, 3)); Nl = _unit(up - (up * L).sum(1, keepdims=True) * L)
    size = lf["size"] * r.uniform(0.75, 1.25, n); Vq, Fq, uv = cards(P, L, Nl, size, lf.get("aspect", 1.0), lf.get("fold", 0.15))
    tv = r.uniform(*lf.get("tint", (0.8, 1.15)), n); hue = np.asarray(lf.get("hue", [1, 1, 1]), float)
    tint = np.repeat(np.column_stack([tv * hue[0], tv * hue[1] * r.uniform(0.95, 1.05, n), tv * hue[2], np.ones(n)]), 8, 0)
    return Vq, Fq, uv, tint, n


def _bend_axis(base, dirv, L, k, bend, seed_rng, droop=1.0):
    """A stem/blade axis from k+1 points: initial direction dirv, a bend under its own weight (bend: 0 — straight, 1 — an arc down to the horizontal)."""
    t = np.linspace(0, 1, k + 1)[:, None]; h = _unit(dirv * np.array([1, 1, 0]) + 1e-6)
    ang0 = np.arccos(np.clip(dirv[2], -1, 1)); ang = ang0 + bend * droop * (math.pi / 2 - ang0 * 0.5) * t[:, 0] ** 1.6
    seg = L / k; P = [base]
    for j in range(1, k + 1):
        a = ang[j]; P.append(P[-1] + seg * (h * math.sin(a) + np.array([0, 0, math.cos(a)])))
    return np.array(P)


def ribbons(axes, widths, normals):
    """Ribbons along axes: axes (n, k+1, 3), width per point (n, k+1), the normal of the ribbon plane (n, 3) → (V, F, uv)."""
    n, kp, _ = axes.shape; T = np.gradient(axes, axis=1); T = _unit(T); S = _unit(np.cross(T, normals[:, None, :]))
    Lft = axes - S * widths[..., None] * 0.5; Rgt = axes + S * widths[..., None] * 0.5
    V = np.stack([Lft, Rgt], 2).reshape(n, kp * 2, 3); idx = np.arange(kp - 1)
    f = np.stack([2 * idx, 2 * idx + 1, 2 * idx + 3, 2 * idx + 2], 1); F = (f[None] + (kp * 2) * np.arange(n)[:, None, None]).reshape(-1, 4)
    t = np.linspace(0, 1, kp); uvp = np.stack([np.stack([np.zeros(kp), t], 1), np.stack([np.ones(kp), t], 1)], 1).reshape(-1, 2)
    uv_v = np.tile(uvp, (n, 1)); uv = uv_v[F.ravel()]
    return V.reshape(-1, 3), F, uv


def fronds(fr, top, r, stem_color):
    """Fronds (fern, palm — one organ): an arched rachis from the top + leaflet cards on both sides."""
    parts = []; nf = int(r.integers(*fr.get("n", [6, 10]))); P_, L_, N_, S_ = [], [], [], []; stems = []
    for j in range(nf):
        az = 2 * math.pi * j / nf + r.uniform(-0.2, 0.2); tilt = math.radians(r.uniform(*fr.get("angle", [30, 60])))
        dv = np.array([math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az), math.cos(tilt)])
        Lr = fr["len"] * r.uniform(0.8, 1.1); a = _bend_axis(top, dv, Lr, 8, fr.get("bend", 0.7), r); stems.append(a)
        npn = fr.get("pinnae", 14)
        for q in range(npn):
            t = 0.12 + 0.85 * q / npn; idx = t * 8; i0 = min(int(idx), 7); p = a[i0] + (a[i0 + 1] - a[i0]) * (idx - i0)
            tg = _unit(a[i0 + 1] - a[i0]); sd = _unit(np.cross(tg, [0, 0, 1.0]) + 1e-6)
            for s_ in (-1, 1):
                Ld = _unit(sd * s_ + tg * 0.45 + np.array([0, 0, -fr.get("droop", 0.15)])); P_.append(p); L_.append(Ld)
                N_.append(_unit(np.cross(Ld, tg) * s_ + np.array([0, 0, 0.6]))); S_.append(fr["leaflet"] * (1 - 0.6 * abs(t - 0.45)))
    P_, L_, N_ = np.array(P_), np.array(L_), np.array(N_); n = len(P_)
    Vq, Fq, uv = cards(P_, L_, N_, np.array(S_), fr.get("aspect", 0.35), 0.2); tv = r.uniform(0.85, 1.15, n)
    hue = np.asarray(fr.get("hue", [1, 1, 1]), float); tint = np.repeat(np.column_stack([tv * hue[0], tv * hue[1], tv * hue[2], np.ones(n)]), 8, 0)
    if fr.get("type") == "blade":                                     # a ribbon leaflet (palm): lighter toward the tip, darker at the rachis
        tt = np.tile(np.array([0, 0, 0, 1, 1, 1, 0, 0], float), n)[:, None]; tint[:, :3] *= (0.75 + 0.45 * tt)
    parts.append(("fronds", Vq, Fq, uv, tint, fr.get("type", "lanceolate")))
    wdr = np.array([np.linspace(fr.get("rachis_w", 0.01), fr.get("rachis_w", 0.01) * 0.3, 9) for _ in stems])
    V, F, uv = ribbons(np.array(stems), wdr, np.tile([0, 0, 1.0], (len(stems), 1)))
    parts.append(("rachis", V, F, uv, np.tile(np.append(np.asarray(stem_color, float), 1), (len(F) * 4, 1)), "blade")); return parts


def herb(sp, seed):
    """A herbaceous plant with the same axis logic: stems (count, angle from the vertical, length, bend under weight), organs on the axes —
    leaves by phyllotaxis, a basal rosette, fronds (fern, palm), a flower or inflorescence at the top; a grass — a tuft of
    ribbon blades (a clump: a shared tilt, height spread, dry blades, spikelets)."""
    r = np.random.default_rng(seed); H = sp["height"]; parts = []; hv = sp.get("habit")
    if hv == "grass":
        nb = int(r.integers(*sp.get("blades", [20, 50]))); k = 5; ax, wd, nm, tn = [], [], [], []
        lean_az = r.uniform(0, 2 * math.pi); lean_k = sp.get("clump_lean", 0.25)
        for j in range(nb):
            az = r.uniform(0, 2 * math.pi); tilt = math.radians(r.uniform(0, sp.get("spread", 35))) * (0.4 + 0.6 * r.random())
            dv = np.array([math.sin(tilt) * math.cos(az) + lean_k * math.cos(lean_az), math.sin(tilt) * math.sin(az) + lean_k * math.sin(lean_az), math.cos(tilt)])
            base = np.array([r.normal(0, sp.get("clump_r", 0.04)), r.normal(0, sp.get("clump_r", 0.04)), 0.0])
            L = H * r.uniform(0.45, 1.0); a = _bend_axis(base, _unit(dv), L, k, sp.get("bend", 0.5) * r.uniform(0.5, 1.3), r)
            w = sp.get("width", 0.006) * r.uniform(0.7, 1.3) * np.clip(1.0 - np.linspace(0, 1, k + 1) ** 2.2, 0.05, 1)
            ax.append(a); wd.append(w); nm.append(_unit(np.cross(dv, [math.cos(az + 1.5), math.sin(az + 1.5), 0]) + r.normal(0, 0.3, 3)))
            dry = r.random() < sp.get("dry", 0.1); c = np.array(sp.get("dry_color", [0.30, 0.23, 0.10]) if dry else sp.get("color", [0.10, 0.19, 0.04]))
            tn.append(c * r.uniform(0.8, 1.2))
        V, F, uv = ribbons(np.array(ax), np.array(wd), np.array(nm)); kp = k + 1
        tip = np.tile(np.repeat(np.linspace(0, 1, kp), 2), nb)       # lighter and yellower toward the tip
        base_c = np.repeat(np.array(tn), kp * 2, 0); col = base_c * (0.75 + 0.45 * tip[:, None]) + np.array([0.06, 0.04, -0.02]) * tip[:, None]
        tint = np.column_stack([col, np.ones(len(col))])[F.ravel()]
        parts.append(("blades", V, F, uv, tint, sp.get("tex", "blade")))
        if sp.get("heads", 0) > 0:                                     # spikelets on some of the blades
            nh = int(nb * sp["heads"] * 0.3); hp = np.array([a[-1] for a in ax[:nh]]) if nh else np.zeros((0, 3))
            if nh:
                Lh = np.array([_unit(a[-1] - a[-2]) for a in ax[:nh]]); up = _unit(np.cross(Lh, r.normal(0, 1, (nh, 3))))
                Vq, Fq, uvq = cards(hp - Lh * 0.02, Lh, up, H * 0.18, 0.35, 0.0)
                tq = np.repeat(np.column_stack([np.tile(np.array(sp.get("head_color", [0.32, 0.27, 0.13])), (nh, 1)) * r.uniform(0.8, 1.1, (nh, 1)), np.ones(nh)]), 8, 0)
                parts.append(("heads", Vq, Fq, uvq, tq, "cluster_lanceolate"))
        return parts
    # stems
    ns = int(r.integers(*sp.get("stems", [1, 3]))); k = 6; ax, wd, nm = [], [], []
    for j in range(ns):
        az = r.uniform(0, 2 * math.pi); tilt = math.radians(r.uniform(*sp.get("stem_angle", [0, 20])))
        dv = np.array([math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az), math.cos(tilt)])
        L = H * r.uniform(0.75, 1.0); a = _bend_axis(np.array([r.normal(0, 0.01), r.normal(0, 0.01), 0]), dv, L, k, sp.get("bend", 0.15), r)
        ax.append(a); wd.append(np.full(k + 1, sp.get("stem_w", 0.004)) * np.linspace(1, 0.6, k + 1)); nm.append(_unit(np.cross(dv, [math.cos(az), math.sin(az), 0])))
    if sp.get("stem_w", 0.004) > 0:
        for rot in (0, 1):                                            # a stem is two crossed ribbons (visible from any side)
            nm2 = np.array(nm) if rot == 0 else _unit(np.cross(np.array([_unit(a[-1] - a[0]) for a in ax]), np.array(nm)))
            V, F, uv = ribbons(np.array(ax), np.array(wd), nm2)
            col = np.array(sp.get("stem_color", [0.08, 0.14, 0.035])); tint = np.tile(np.append(col, 1), (len(F) * 4, 1))
            parts.append((f"stem{rot}", V, F, uv, tint, "blade"))
    lf = sp.get("leaf")
    if lf:                                                            # leaves along the stem (phyllotaxis) and a rosette
        P_, L_, N_ = [], [], []
        for a in ax:
            nl = lf.get("n", 6)
            for q in range(nl):
                t = 0.12 + 0.75 * q / max(nl, 1); idx = t * (len(a) - 1); i0 = int(idx); p = a[i0] + (a[min(i0 + 1, len(a) - 1)] - a[i0]) * (idx - i0)
                tg = _unit(a[min(i0 + 1, len(a) - 1)] - a[i0]); az = q * math.radians(lf.get("phyllotaxis", 137.5)) + r.uniform(-0.3, 0.3)
                out = _unit(np.cross(tg, [math.cos(az), math.sin(az), 0.0]) + 1e-6); out = _unit(np.cross(out, tg))
                ang = math.radians(lf.get("angle", 50)); Ld = _unit(tg * math.cos(ang) + out * math.sin(ang) + np.array([0, 0, -lf.get("droop", 0.2)]))
                P_.append(p); L_.append(Ld); N_.append(_unit(np.cross(Ld, np.cross([0, 0, 1.0], Ld)) + r.normal(0, 0.3, 3)))
        for q in range(int(lf.get("basal", 0))):                      # basal rosette
            az = r.uniform(0, 2 * math.pi); Ld = _unit(np.array([math.cos(az), math.sin(az), lf.get("basal_up", 0.6)]))
            P_.append(np.array([0, 0, 0.005])); L_.append(Ld); N_.append(_unit(np.cross(Ld, np.cross([0, 0, 1.0], Ld))))
        if P_:
            P_, L_, N_ = np.array(P_), np.array(L_), np.array(N_); n = len(P_); size = lf["size"] * r.uniform(0.75, 1.2, n)
            Vq, Fq, uv = cards(P_, L_, N_, size, lf.get("aspect", 1.0), 0.18); tv = r.uniform(0.85, 1.15, n)
            hue = np.asarray(lf.get("hue", [1, 1, 1]), float); tint = np.repeat(np.column_stack([tv * hue[0], tv * hue[1], tv * hue[2], np.ones(n)]), 8, 0)
            parts.append(("leaves", Vq, Fq, uv, tint, lf["type"]))
    fr = sp.get("frond")
    if fr: parts += fronds(fr, np.array([0, 0, fr.get("z", 0.0)]), r, sp.get("stem_color", [0.09, 0.13, 0.045]))
    fl = sp.get("flower")
    if fl:                                                            # flowers at the tops of the stems
        tips = np.array([a[-1] for a in ax]); n = len(tips)
        if fl.get("side"):                                            # an ear — as a card along the top of the stem
            Ld = np.array([_unit(a[-1] - a[-3]) for a in ax]); Nl = _unit(np.cross(Ld, r.normal(0, 1, (n, 3))))
            Vq, Fq, uv = cards(tips - Ld * fl["size"] * 0.8, Ld, Nl, np.full(n, fl["size"]), 0.4, 0.0)
        else:                                                         # a cup — facing up with a tilt + a second one at 60° (from the side it is not an "edge")
            up = _unit(np.array([0, 0, 1.0]) + r.normal(0, 0.35, (n, 3))); Ld = _unit(np.cross(up, r.normal(0, 1, (n, 3))))
            sz = np.full(n, fl["size"]) * r.uniform(0.85, 1.15, n); up2 = _unit(0.5 * up + 0.866 * np.cross(Ld, up))
            V1, F1, uv1 = cards(tips - Ld * sz[:, None] * 0.5, Ld, up, sz, 1.0, 0.0); V2, F2, uv2 = cards(tips - Ld * sz[:, None] * 0.5, Ld, up2, sz * 0.9, 1.0, 0.0)
            Vq, Fq, uv = np.vstack([V1, V2]), np.vstack([F1, F2 + len(V1)]), np.vstack([uv1, uv2])
        tint = np.ones((len(Fq) * 4, 4)); parts.append(("flowers", Vq, Fq, uv, tint, "flower_" + fl["type"]))
    return parts


# ---------------------------------------------------------------- the whole plant
def build(spec, where=(0.0, 0.0, None), ground=None, sc=None, name=None, coll=None):
    """A plant from a plants.resolve specification → Blender objects (bark, foliage/organs, roots) at the point where on the ground ground(x, y).
    Returns: (objects, cost)."""
    sc = sc or bpy.context.scene; coll = coll or sc.collection; t0 = time.time(); seed = int(spec.get("seed", 1))
    x0, y0, z0 = where; z0 = float(ground(np.array([x0]), np.array([y0]))[0]) if (z0 is None and ground) else (z0 or 0.0)
    gfun = (lambda X, Y: ground(np.asarray(X, float), np.asarray(Y, float))) if ground else None
    nm = name or spec.get("id", "plant"); objs = []; cost = {"verts": 0}
    H = float(spec["height"]); lod = int(spec.get("lod", 0))
    if spec.get("habit") in ("tree", "shrub", "palm") and spec.get("trunk"):
        W = dict(spec["woody"]); levels = spec.get("levels", [])
        if lod >= 1: W["radial"] = min(W.get("radial", 12), 6 if lod == 1 else 4); W["smooth"] = 1
        if lod >= 2 and len(levels) > 1: levels = levels[:1]           # far: only the first order of branches
        cm = woody_mesh(spec["trunk"], levels, seed, W.get("radial", 12), W.get("smooth", 2), spec.get("growth"))
        me = bpy.data.meshes.new(nm + "_bark"); create_mesh_from_cpp(me, cm)
        V, Nn, rad, D = mesh_arrays(me); h0 = max(float(V[:, 2].max()), 1e-3); k = H / h0
        V = V * k; rad = rad * k
        if spec.get("bark_radius"):                                    # trunk thickness by species (diameter at the ground), independent of the scale
            r0 = float(rad[V[:, 2] < 0.3 * H].max()) if (V[:, 2] < 0.3 * H).any() else rad.max()
            kr = spec["bark_radius"] / max(r0, 1e-4); kr = float(np.clip(kr, 0.4, 2.5))
            A = V - Nn * rad[:, None]; V = A + Nn * (rad * kr)[:, None]; rad = rad * kr
        Lv = spec.get("lean") or {}; push = np.asarray(spec.get("push") or [0, 0], float)
        V = crown_push(lean(V, H, Lv.get("deg", 0), Lv.get("az", 0), Lv.get("recover", 0.6)), H * spec.get("crown_base", 0.35), H, push)
        me.vertices.foreach_set("co", V.astype(np.float32).ravel()); me.attributes["radius"].data.foreach_set("value", rad.astype(np.float32)); me.update()
        me.materials.append(bark_material(spec.get("bark", {}))); ob = bpy.data.objects.new(nm + "_bark", me); coll.objects.link(ob); objs.append(ob)
        cost["verts"] += len(V)
        lf = spec.get("leaves")
        if lf and spec.get("state") not in ("dead", "winter"):
            lf = dict(lf, _radial=W.get("radial", 12), _seg=1.0 / max(levels[-1].get("resolution", 3) if levels else 3, 1) * k)
            if lod >= 1:                                                # mid and far shots: fewer, larger cards — the same crown mass
                f = (0.3, 1.8) if lod == 1 else (0.1, 3.2); lf["per_m"] = lf.get("per_m", 6) * f[0] * (len(spec.get("levels", [])) / max(len(levels), 1))
                lf["size"] = lf["size"] * f[1]; lf["reach"] = lf.get("reach", lf["size"] / f[1] * 1.3) * 1.15
            V0, N0, r0_, D0 = mesh_arrays(me); out = foliage(V0, N0, r0_, D0, lf, seed + 5, H)
            if out:
                Vq, Fq, uv, tint, n = out; lm = make_mesh(nm + "_leaves", Vq, Fq, uv, tint, card_material(os.path.join(TEX, lf["type"] + ".png"), cheap=lod >= int(os.environ.get("SK_VEG_CHEAP_LOD", 1)), optics=spec.get("optics")), smooth=False)
                if float((spec.get("optics") or {}).get("sky", 0.0)) > 0:  # the sky share above a leaf — only if enabled in the optics (off by default)
                    q = Vq.reshape(n, 6, 3); ar = np.linalg.norm(q[:, 2] - q[:, 0], axis=1) * np.linalg.norm(q[:, 3] - q[:, 2], axis=1)
                    cov = _tex_cov(lm.materials[0]); t_ = time.time(); vs = sky_visibility(q.mean(1), ar * cov, seed=seed)
                    lm.attributes["sk_sky"].data.foreach_set("value", np.repeat(vs, 6)); cost["sky_s"] = round(time.time() - t_, 2); cost["sky_mean"] = round(float(vs.mean()), 3)
                lo = bpy.data.objects.new(nm + "_leaves", lm); coll.objects.link(lo); objs.append(lo); cost["leaves"] = n; cost["verts"] += len(Vq)
        if spec.get("frond") and spec.get("state") not in ("dead",):  # palm: fronds from the top of the trunk
            top = V[np.argmax(V[:, 2])]; fr = dict(spec["frond"])
            for pn, Vf, Ff, uvf, tf, tex in fronds(fr, top, np.random.default_rng(seed + 7), [0.14, 0.13, 0.06]):
                mat = card_material(os.path.join(TEX, tex + ".png"), kind="blade" if tex == "blade" else "leaf", optics=spec.get("optics"))
                mf = make_mesh(f"{nm}_{pn}", Vf, Ff, uvf, tf, mat, smooth=False); of = bpy.data.objects.new(f"{nm}_{pn}", mf)
                coll.objects.link(of); objs.append(of); cost["verts"] += len(Vf)
        rs = spec.get("roots") if lod < 2 else None
        if rs and rs.get("type") != "none":
            r_base = float(rad[V[:, 2] < 0.5].max()) if (V[:, 2] < 0.5).any() else float(rad.max())
            cr = root_mesh(rs, r_base, seed); rm = bpy.data.meshes.new(nm + "_roots"); create_mesh_from_cpp(rm, cr)
            Rv, Rn, Rr, _ = mesh_arrays(rm); dep = np.zeros(len(Rv))
            if "hierarchy_depth" in rm.attributes: rm.attributes["hierarchy_depth"].data.foreach_get("value", dep)
            Rv = seat_roots(Rv, Rn, Rr, rs, gfun, x0, y0, z0, dep); rm.vertices.foreach_set("co", Rv.astype(np.float32).ravel()); rm.update()
            rm.materials.append(bark_material(spec.get("bark", {}))); ro = bpy.data.objects.new(nm + "_roots", rm); coll.objects.link(ro); objs.append(ro)
            cost["verts"] += len(Rv)
    else:
        for pn, V, F, uv, tint, tex in herb(spec, seed):
            mat = card_material(os.path.join(TEX, tex + ".png"), kind="flower" if tex.startswith("flower") else "blade" if tex == "blade" else "leaf", optics=spec.get("optics"))
            me = make_mesh(f"{nm}_{pn}", V, F, uv, tint, mat, smooth=pn in ("blades",)); ob = bpy.data.objects.new(f"{nm}_{pn}", me)
            coll.objects.link(ob); objs.append(ob); cost["verts"] += len(V)
    for o in objs:
        o.location = (x0, y0, z0 - spec.get("sink", 0.05)); o["plant"] = spec.get("id", nm); o["kind"] = "plant"
    cost["s"] = round(time.time() - t0, 3); return objs, cost


def main():
    """Sheet of species (a check): all the file's plants side by side along the X axis, without ground; save a .blend."""
    a = sys.argv[sys.argv.index("--") + 1:]; P = json.load(open(a[0])); out = a[1]
    for o in list(bpy.data.objects): bpy.data.objects.remove(o)
    res = []; x = 0.0
    for sp in P["plants"]:
        w = max(sp.get("width", sp["height"] * 0.6), 0.3); x += w * 0.6
        objs, cost = build(sp, (x, sp.get("y", 0.0), 0.0), name=sp["id"]); x += w * 0.6 + 1.0
        res.append({"id": sp["id"], **cost})
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out), compress=True)
    print("__RESULT__" + json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
