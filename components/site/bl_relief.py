"""A site from a plan in Blender (components/site/terrain.py → relief.npz + location.json): one seamless ground mesh — step res in the core, sparser toward the edge of the
plot and toward the horizon (the grid lines are clustered, but the mesh is one — no cracks at the joins), the ground-type texture (Ground.colors), water —
its own mesh by the level of rivers and lakes. The object "terrain" (taken by the bridge bl_export_scene.py and the frame colour bl_color.py; the property
far_z — the far ground level), water — "water" (the property kind=water). The functions also serve the builder of a custom location.
blender -b --factory-startup -P core/blender/run.py -- components/site/bl_relief.py -- <site folder> <out.blend> [core=120] [px=4] [cams=<cameras.json>] [shots=<folder>]
  [veg=1 [veg_density=0.45]] — the vegetation of the Plants component (a computed forest, a meadow near the cameras) instead of the dressing stand-in trees and the small-items grass
cams — cameras in the cameras.json format: the active one is the first; shots — a view through each (Workbench with texture, ~0.3 s per frame)."""
import core.paths as _core_paths         # the code is here, the core — core/<part> through kernel
import json, math, os, sys, time
import bpy
import numpy as np
from mathutils import Matrix
from core.surface.ground import Ground, axes


def grid_mesh(name, X, Y, Z, uv=True, skirt=40.0):
    """An nx × ny grid of nodes (X, Y, Z) → a mesh of quads with UVs in index space. skirt — a skirt downward along the edge and a bottom:
    the mesh is closed — Workbench shadows (shadow volumes) have no false stripes from an open edge."""
    ny, nx = X.shape; co = np.stack([X, Y, Z], -1).reshape(-1, 3)
    ii = np.arange(nx * ny).reshape(ny, nx); q = np.stack([ii[:-1, :-1], ii[:-1, 1:], ii[1:, 1:], ii[1:, :-1]], -1).reshape(-1, 4).tolist()
    faces = q; n0 = len(co); ring = []
    if skirt:
        ring = np.concatenate([ii[0, :], ii[1:, -1], ii[-1, -2::-1], ii[-2:0:-1, 0]])   # the edge around the perimeter
        low = co[ring].copy(); low[:, 2] = Z.min() - skirt; co = np.vstack([co, low]); m = len(ring)
        faces = q + [[int(ring[k]), int(n0 + k), int(n0 + (k + 1) % m), int(ring[(k + 1) % m])] for k in range(m)] + [list(range(n0 + m - 1, n0 - 1, -1))]
    me = bpy.data.meshes.new(name); me.from_pydata(co.tolist(), [], faces); me.update()
    if uv:
        src = np.concatenate([np.arange(n0), ring]) if skirt else np.arange(n0)
        u = (src % nx) / (nx - 1); v = (src // nx) / (ny - 1)
        li = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", li)
        me.uv_layers.new(name="ground").data.foreach_set("uv", np.stack([u[li], v[li]], 1).ravel().astype(np.float32))
    me.shade_smooth()
    return me


def image(name, rgb):
    """A Blender image from an sRGB array (h, w, 3), rows bottom to top; packed into the .blend."""
    h, w, _ = rgb.shape; im = bpy.data.images.new(name, w, h, alpha=False)
    im.pixels.foreach_set(np.concatenate([rgb, np.ones((h, w, 1))], -1).astype(np.float32).ravel()); im.pack()
    return im


def material(name, rgb=None, img=None, rough=0.9, mean=None):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree
    bs = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"); bs.inputs["Roughness"].default_value = rough
    if img is not None:
        tx = nt.nodes.new("ShaderNodeTexImage"); tx.image = img; tx.interpolation = "Linear"; tx.extension = "EXTEND"
        nt.links.new(tx.outputs["Color"], bs.inputs["Base Color"]); nt.nodes.active = tx    # Workbench "texture" takes the active image
        if mean is not None: m.diffuse_color = (*(float(c) ** 2.2 for c in mean), 1.0)   # the mean colour set explicitly (4K pixels as a list take long)
    if rgb is not None:
        lin = tuple(float(c) ** 2.2 for c in rgb); bs.inputs["Base Color"].default_value = (*lin, 1.0); m.diffuse_color = (*lin, 1.0)
    return m


def apply_surface(ter, G, set_=None, far=True):
    """The ground shader assembled from ready parts by the site recipe (surface.json): scans up close (core/surface/bl_terrain_surf), the belt — the far shader
    (core/surface/bl_terrain_vista); far="baked" — the belt as a baked image. No recipe or masks — the ground stays an image
    (the one Workbench shows). → a summary."""
    sj = os.path.join(G.folder, "surface.json")
    if not (os.path.exists(sj) and os.path.exists(os.path.join(G.folder, "relief_w.npz"))): return {"surf": "no surface.json or relief_w.npz"}
    from core.surface import bl_terrain_surf as TS; from components.site import terrain_tex as TT
    sj_ = json.load(open(sj)); rec = (sj_.get("sets") or {}).get(set_, sj_["recipe"])
    bio = G.loc.get("biome") or TT.GROUND_BIOME.get(G.loc.get("ground", "grass"), "temperate")
    return TS.apply(ter, G, rec, sj_.get("seed", 1), TT.snowline_of(G.loc, bio), sj_.get("masks"), far)


def build_terrain(G, name="terrain", core=None, px=4, sc=None, surface=True):
    """The site ground: a tensor-product grid (dense in the core), heights — Ground.height, colour — a texture in the grid index space;
    surface — the ground shader by the site recipe (apply_surface): True — scans up close and the far shader on the belt, "vista" — the far shader on the whole
    ground (cheap), "baked" — the belt as an image; False — an image only."""
    sc = sc or bpy.context.scene; t0 = time.time()
    xs, ys = axes(G, core); X, Y = np.meshgrid(xs, ys); Z = G.height(X, Y)
    me = grid_mesh(name, X, Y, Z); me.set_sharp_from_angle(angle=math.radians(40))   # cliff edges are sharp, slopes smooth (no stripes on the walls)
    ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob)
    t1 = time.time(); tp, tj = os.path.join(G.folder, "relief_tex.png"), os.path.join(G.folder, "relief_tex.json")
    ok = os.path.exists(tp) and os.path.exists(tj) and (lambda m: len(m["xs"]) == len(xs) and len(m["ys"]) == len(ys))(json.load(open(tj)))
    if ok:                                                          # the baked texture of the game's tiles (components/site/terrain_tex.py) — the same grid lines
        img = bpy.data.images.load(tp); img.pack(); mj = json.load(open(tj)); src = "tiles " + mj["biome"]; mean = mj.get("mean")
    else:                                                           # fallback colour — Ground.colors
        tu = (np.arange(len(xs) * px) + 0.5) / px - 0.5; tv = (np.arange(len(ys) * px) + 0.5) / px - 0.5     # texel centres in line indices
        TX, TY = np.meshgrid(np.interp(tu, np.arange(len(xs)), xs), np.interp(tv, np.arange(len(ys)), ys))
        rgb = G.colors(TX, TY); img = image("relief_ground", rgb); src = "color"; mean = rgb.reshape(-1, 3).mean(0).tolist()
    me.materials.append(material("relief_ground", img=img, mean=mean))
    ob["far_z"] = G.far_z; ob["relief"] = G.folder; ob["kind"] = "ground"
    res = {"verts": len(me.vertices), "lines": [len(xs), len(ys)], "tex": [img.size[0], img.size[1]], "tex_src": src, "s_mesh": round(t1 - t0, 2),
           "s_tex": round(time.time() - t1, 2)}
    if core is None: res.update(build_belt(G, sc=sc)[1])            # the mountain belt — its own mesh (silhouettes)
    if surface and core is None: res.update(apply_surface(ob, G, far={"vista": "all", "baked": "baked"}.get(surface, True)))
    return ob, res


BELT_SINK = 0.5                                                     # belt nodes under the near mesh are lowered by this much, m (so they do not z-fight with it)


def build_belt(G, name="terrain_belt", sc=None):
    """The mountain belt as a separate mesh — the nodes of the site's belt grid (step G.fres, 4.7 m), heights as is: in the shared mesh the belt cells reach 25 m —
    faceted mountain silhouettes (like in games of 2004). The near plot is a hole with an overlap of one cell (the nodes inside are lower by BELT_SINK,
    under the near mesh); UVs are in the axis indices of the shared mesh: the same images (the baked one for Workbench, the belt data for the far shader).
    The belt faces are removed from "terrain" (the vertices, UVs and attributes stay): made transparent they cost ~4.7 s of a frame. → (object | None, info)."""
    sc = sc or bpy.context.scene; t0 = time.time()
    old = bpy.data.objects.get(name); home = None                  # the belt's collection is the same as the previous one's (when Buildings rebuild the ground),
    if old is not None:                                             # otherwise the "site" section's: the frame cost and the step's contribution are reported per component
        home = old.users_collection[0] if old.users_collection else None
        me0 = old.data; bpy.data.objects.remove(old)
        if me0 is not None and me0.users == 0: bpy.data.meshes.remove(me0)
    H = G.Hf; ny, nx = H.shape; r = float(G.fres); x0, y0, x1, y1 = G.near_bounds()
    if G.far_bounds()[2] - G.far_bounds()[0] <= (x1 - x0) + 4 * r: return None, {"belt": "no belt"}
    X, Y = np.meshgrid(G.fx0 + np.arange(nx) * r, G.fy0 + np.arange(ny) * r)
    cx, cy = (X[:-1, :-1] + X[1:, 1:]) / 2, (Y[:-1, :-1] + Y[1:, 1:]) / 2
    keep = ~((cx > x0 + r) & (cx < x1 - r) & (cy > y0 + r) & (cy < y1 - r))   # belt cells and the overlap ring
    ii = np.arange(nx * ny).reshape(ny, nx)
    q = np.stack([ii[:-1, :-1], ii[:-1, 1:], ii[1:, 1:], ii[1:, :-1]], -1)[keep]
    used = np.unique(q); rm = -np.ones(nx * ny, np.int64); rm[used] = np.arange(len(used))
    din = np.minimum.reduce([X - x0, x1 - X, Y - y0, y1 - Y]).ravel()[used]   # depth into the near plot, m
    Z = H.ravel()[used] - BELT_SINK * np.clip(din / r, 0, 1)
    co = np.column_stack([X.ravel()[used], Y.ravel()[used], Z]).astype(np.float32)
    me = bpy.data.meshes.new(name); me.vertices.add(len(co)); me.vertices.foreach_set("co", co.ravel())
    f = rm[q].astype(np.int32); me.loops.add(f.size); me.loops.foreach_set("vertex_index", f.ravel())
    me.polygons.add(len(f)); me.polygons.foreach_set("loop_start", np.arange(0, f.size, 4, dtype=np.int32)); me.polygons.foreach_set("loop_total", np.full(len(f), 4, np.int32))
    me.update(); me.shade_smooth()
    xs, ys = axes(G); u = np.interp(co[:, 0], xs, np.arange(len(xs))) / (len(xs) - 1); v = np.interp(co[:, 1], ys, np.arange(len(ys))) / (len(ys) - 1)
    li = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", li)
    me.uv_layers.new(name="ground").data.foreach_set("uv", np.stack([u[li], v[li]], 1).ravel().astype(np.float32))
    ter = bpy.data.objects.get("terrain"); img = None
    if ter is not None and ter.data.materials:                      # Workbench — the same baked image as the ground's
        img = next((n.image for n in ter.data.materials[0].node_tree.nodes if n.type == "TEX_IMAGE" and n.image), None)
    mb = material("relief_belt", img=img); me.materials.append(mb)
    if ter is not None and ter.data.materials: mb.diffuse_color = ter.data.materials[0].diffuse_color
    home = home or bpy.data.collections.get("site") or (ter.users_collection[0] if ter is not None and ter.users_collection else sc.collection)
    ob = bpy.data.objects.new(name, me); home.objects.link(ob)
    ob["kind"] = "ground"; ob["far_z"] = G.far_z; ob["relief"] = G.folder; ob["sk_layer"] = "site"
    cut = 0
    if ter is not None:                                             # the shared mesh covers only the near plot (the belt is this mesh)
        import bmesh
        bm = bmesh.new(); bm.from_mesh(ter.data)
        out_ = [f_ for f_ in bm.faces if (lambda c: c.x < x0 or c.x > x1 or c.y < y0 or c.y > y1)(f_.calc_center_median())]
        cut = len(out_); bmesh.ops.delete(bm, geom=out_, context="FACES_ONLY"); bm.to_mesh(ter.data); bm.free(); ter.data.update()
    return ob, {"belt_verts": len(co), "belt_faces": len(f), "terrain_faces_cut": cut, "s_belt": round(time.time() - t0, 2)}


def build_water(G, name="water", stride=2, sc=None):
    """Water: nodes of the near grid where a river and lake level is set (with a one-node margin — the edge goes under the bank), z — the level."""
    sc = sc or bpy.context.scene; L = G.level[::stride, ::stride]; ok = ~np.isnan(L)
    if not ok.any(): return None, {}
    pad = np.pad(ok, 1); grow = pad[1:-1, 1:-1] | pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]
    Lf = np.where(ok, L, np.nan)
    for _ in range(2):                                              # the level in the margin nodes comes from the neighbours
        P = np.pad(Lf, 1, constant_values=np.nan); nb_ = np.stack([P[:-2, 1:-1], P[2:, 1:-1], P[1:-1, :-2], P[1:-1, 2:]])
        Lf = np.where(np.isnan(Lf) & grow, np.nanmax(np.where(np.isnan(nb_), -1e9, nb_), 0), Lf)
    Lf = np.where(Lf < -1e8, np.nan, Lf); r = G.nres * stride
    ys_, xs_ = np.nonzero(grow); j0, j1, i0, i1 = ys_.min(), ys_.max() + 1, xs_.min(), xs_.max() + 1
    sub = Lf[j0:j1, i0:i1]; X, Y = np.meshgrid(G.nx0 + np.arange(i0, i1) * r, G.ny0 + np.arange(j0, j1) * r)
    v = ~np.isnan(sub); idx = -np.ones(sub.shape, int); idx[v] = np.arange(v.sum())
    q = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4); q = q[(q >= 0).all(1)]
    co = np.stack([X[v], Y[v], sub[v]], -1)
    me = bpy.data.meshes.new(name); me.from_pydata(co.tolist(), [], q.tolist()); me.update(); me.shade_smooth()
    lava = G.loc.get("fluid") == "lava" or G.loc.get("biome") == "volcanic"   # at a volcano, rivers and lakes are lava
    me.materials.append(material("lava" if lava else "water", rgb=(0.95, 0.38, 0.08) if lava else (0.24, 0.36, 0.45), rough=0.6 if lava else 0.12))
    ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob); ob["kind"] = "water"
    res = {"water_verts": len(co)}
    if any(f.get("id") == "sea" or str(f.get("id", "")).endswith("_sea") for f in G.loc.get("features") or []) and not lava:   # the sea (sea, <archetype>_sea) extends to the horizon: the level exists only in the near
        lv = float(np.nanmedian(Lf[~np.isnan(Lf)])); x0, y0, x1, y1 = G.near_bounds(); R_ = 8000.0   # grid, beyond it — a flat surface as a frame
        X4, Y4 = [x0 - R_, x0, x1, x1 + R_], [y0 - R_, y0, y1, y1 + R_]                         # (land above the level covers it by itself)
        V_ = [(X4[i], Y4[j], lv) for j in range(4) for i in range(4)]
        F_ = [(j * 4 + i, j * 4 + i + 1, (j + 1) * 4 + i + 1, (j + 1) * 4 + i) for j in range(3) for i in range(3) if (i, j) != (1, 1)]
        sm = bpy.data.meshes.new(name + "_sea"); sm.from_pydata(V_, [], F_); sm.update(); sm.materials.append(me.materials[0])
        so = bpy.data.objects.new(name + "_sea", sm); sc.collection.objects.link(so); so["kind"] = "water"; res["sea_level"] = round(lv, 2)
    return ob, res


def rock_shapes(n=10, seed=3, sub=3, crag=False):
    """n rock shapes: an icosphere distorted by noise (ridges + fine detail), a flat bottom; unit size. crag — a faceted boulder: the first subdivision,
    a strong fracture, elongated (an outcrop in a wall)."""
    import bmesh
    from mathutils import Vector, noise
    out = []
    for k in range(n):
        bm = bmesh.new(); bmesh.ops.create_icosphere(bm, subdivisions=sub, radius=1.0)
        o = Vector((k * 13.7 + seed, k * 7.1, k * 3.3))
        for v in bm.verts:
            p = v.co * 1.3 + o
            r = 1 + 0.32 * noise.ridged_multi_fractal(p, 1.0, 2.0, 3, 1.0, 2.0) - 0.25 + 0.12 * noise.fractal(p * 2.3, 1.0, 2.0, 3)
            if crag: r = 1 + 0.55 * noise.ridged_multi_fractal(p * 0.8, 1.0, 2.0, 2, 1.0, 2.0) - 0.35
            v.co = v.co * r
            if crag: v.co.z *= 1.6                                  # taller than wide: an outcrop along the dip of the wall
            v.co.z = max(v.co.z, -0.35)                             # flat bottom: the rock sits on the slope, not a ball
        V = np.array([v.co[:] for v in bm.verts]); F = [[v.index for v in f.verts] for f in bm.faces]; bm.free()
        V -= (V.max(0) + V.min(0)) / 2; V /= np.maximum(np.ptp(V, 0) / 2, 1e-6)
        out.append((V, F))
    return out


def _thin(P, spacing, rng):
    """Thinning points on a grid of spacing: one random point per cell (a fast substitute for Poisson disk sampling)."""
    if not len(P): return P
    P = P[rng.permutation(len(P))]; key = np.floor(P[:, :2] / spacing).astype(np.int64)
    _, first = np.unique(key[:, 0] * 1000003 + key[:, 1], return_index=True)
    return P[np.sort(first)]


def dress_rocks(G, sc=None, seed=5, name="rocks", talus=1.0, faces=1.0, limit=700):
    """Dress the cliffs (as the game does — rocks at steep ground): scree at the foot (a strip up to 4 m by the steep ground, where it is gentler than 32°: many small,
    few large — a power law, 0.5–3.5 m) and sparse boulders in the wall (step 14 m, 2.5–6 m, sunk by 40 %). One part, "rocks"."""
    sc = sc or bpy.context.scene; rng = np.random.default_rng(seed); H = G.Hn; r = G.nres
    gy, gx = np.gradient(H, r); S = np.degrees(np.arctan(np.hypot(gx, gy)))
    steep = (S > 42) | (G.masks["cliff"] > 0.5); k = max(1, int(round(4.0 / r)))
    grow = steep.copy()
    for _ in range(k):                                              # a strip by the steep ground
        p_ = np.pad(grow, 1); grow = grow | p_[:-2, 1:-1] | p_[2:, 1:-1] | p_[1:-1, :-2] | p_[1:-1, 2:]
    dry = (np.nan_to_num(G.level, nan=-1e9) < H - 0.2) & (G.masks["road"] < 0.1) & (G.masks["pad"] < 0.1)   # dry, not on a roadbed or pad
    nb = G.near_bounds(); m = 6.0                                   # (scree at an embankment slope was creeping onto the road)
    def pts(mask):
        j, i = np.nonzero(mask); P = np.column_stack([G.nx0 + i * r, G.ny0 + j * r])
        return P[(P[:, 0] > nb[0] + m) & (P[:, 0] < nb[2] - m) & (P[:, 1] > nb[1] + m) & (P[:, 1] < nb[3] - m)]
    T = _thin(pts(grow & ~steep & (S < 32) & dry), 2.2 / max(talus, 0.05), rng)
    T = T[rng.random(len(T)) < 0.55 * talus]                        # the scree is not continuous
    Fc = _thin(pts(steep & dry), 14.0 / max(faces, 0.05), rng)
    items = [(x, y, float(np.clip(0.5 * rng.random() ** -0.6, 0.5, 3.5)), 0.25) for x, y in T]
    items += [(x, y, rng.uniform(2.5, 6.0), 0.4) for x, y in Fc]
    if STANDS:                                                      # not into a figure: 2.5 m from the figures' spots
        A_ = np.array(STANDS); items = [it for it in items if np.min(np.hypot(A_[:, 0] - it[0], A_[:, 1] - it[1])) > 2.5 + it[2] / 2]
    if AVOID:                                                       # not into a building: outside the footprints (+1 m and half the rock size)
        from components.site.terrain import sdf_poly
        items = [it for it in items if all(sdf_poly(np.array([it[0]]), np.array([it[1]]), P_)[0] > 1.0 + it[2] / 2 for P_ in AVOID)]
    items = items[:limit]
    if not items: return None, {"rocks": 0}
    shapes = rock_shapes(seed=seed); V_, F_, off = [], [], 0
    for x, y, size, sink in items:
        V, F = shapes[rng.integers(len(shapes))]
        sx, sy, sz = size * rng.uniform(0.8, 1.2), size * rng.uniform(0.6, 0.9), size * rng.uniform(0.45, 0.75)
        a = rng.uniform(0, 2 * math.pi); R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        P = (V * [sx / 2, sy / 2, sz / 2]) @ R.T + [x, y, float(G.height(x, y)) - sz * sink]
        V_.append(P); F_ += [[off + q for q in f] for f in F]; off += len(P)
    me = bpy.data.meshes.new(name); me.from_pydata(np.vstack(V_).tolist(), [], F_); me.update(); me.shade_smooth(); box_uv(me, 4.0)
    me.materials.append(tile_material("rock", "rock", G, (0.47, 0.45, 0.42), 4.0))
    ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob); ob["kind"] = "rock"
    return ob, {"rocks": len(items), "talus": int(min(len(T), limit)), "face_rocks": int(len(items) - min(len(T), limit)), "rock_verts": off}


def _lathe(rings, seg, noise_amp, seed, closed_top=True):
    """A mesh "by rings": rings — [(centre xyz, normal axis u, normal axis v, radius u, radius v)]; noise — by angle and ring number."""
    from mathutils import Vector, noise
    o = Vector((seed * 3.1, seed * 1.7, seed * 0.9)); V = []
    for k, (c, u, v, ru, rv) in enumerate(rings):
        for j in range(seg):
            a = 2 * math.pi * j / seg; d = math.cos(a) * np.asarray(u) * ru + math.sin(a) * np.asarray(v) * rv
            p = np.asarray(c, float) + d
            n_ = 1 + noise_amp * noise.noise(Vector(p) * 0.28 + o) + noise_amp * 0.6 * noise.noise(Vector(p) * 0.9 + o) + noise_amp * 0.3 * noise.noise(Vector(p) * 2.4 + o)
            V.append(np.asarray(c, float) + d * n_)
    R = len(rings); F = [[k * seg + j, k * seg + (j + 1) % seg, (k + 1) * seg + (j + 1) % seg, (k + 1) * seg + j] for k in range(R - 1) for j in range(seg)]
    if closed_top: F.append(list(range((R - 1) * seg, R * seg))); F.append(list(range(seg))[::-1])
    return np.array(V), F


def landmark_mesh(L_, G, seed):
    """A landmark (not a height map): arch — an arch (a cross-section along a semicircle, abutments into the ground), spire — a rock pillar (cap — the top cap),
    overhang — a stone overhang at a cliff (outward — down the slope), megaliths — a ring of standing stones. Vertices are in world metres."""
    k = L_["kind"]; at = np.asarray(L_["at"], float); a = math.radians(L_.get("rot", 0)); R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    z0 = float(G.height(at[0], at[1]))
    if k == "arch":
        span, h, th = L_.get("span", 24.0), L_.get("h", 16.0), L_.get("thick", 4.0); rings = []
        for i in range(-3, 31):
            t = math.pi * min(max(i, 0), 28) / 28; ext = (-i if i < 0 else i - 28 if i > 28 else 0) * 1.0   # the abutments go into the ground
            p = np.array([-span / 2 * math.cos(t), 0, h * math.sin(t) - ext]); tg = np.array([span / 2 * math.sin(t), 0, h * math.cos(t)]); tg /= np.linalg.norm(tg) + 1e-9
            nrm = np.cross(tg, [0, 1, 0]); w = th * (1 + 0.7 * (1 - math.sin(t)))
            rings.append((p, nrm, [0, 1, 0], w * 0.55, w * 0.75))
        V, F = _lathe(rings, 12, 0.32, seed, closed_top=True)
        legs = np.array([[-span / 2, 0], [span / 2, 0]]) @ R[:2, :2].T + at; zb = float(G.height(legs[:, 0], legs[:, 1]).min())
        return V @ R.T + [at[0], at[1], zb - 0.5], F
    if k == "spire":
        h, r = L_.get("h", 18.0), L_.get("r", 3.5); rings = []
        for i in range(13):
            z = -2 + (h + 2) * i / 12; rr = r * (1 - 0.45 * i / 12) * (1 + 0.15 * math.sin(i * 1.7 + seed))
            rings.append(([0, 0, z], [1, 0, 0], [0, 1, 0], rr, rr * 0.85))
        if L_.get("cap", True): rings += [([0, 0, h + 0.2], [1, 0, 0], [0, 1, 0], r * 1.25, r * 1.1), ([0, 0, h + 1.4], [1, 0, 0], [0, 1, 0], r * 1.15, r)]
        V, F = _lathe(rings, 12, 0.24, seed); return V @ R.T + [at[0], at[1], z0], F
    if k == "overhang":
        gx, gy = G.grad(at[0], at[1]); d = -np.array([float(gx), float(gy)]); d /= np.linalg.norm(d) + 1e-9   # outward — down the slope
        L2, w, th = L_.get("length", 10.0), L_.get("w", 7.0), L_.get("thick", 2.2); rings = []
        for i in range(10):
            s_ = -3 + (L2 + 3) * i / 9; taper = 1 - 0.55 * max(0, s_) / L2
            rings.append(([s_, 0, -0.15 * max(0, s_)], [0, 0, 1], [0, 1, 0], th / 2 * taper, w / 2 * (0.6 + 0.4 * taper)))
        V, F = _lathe(rings, 10, 0.3, seed); a2 = math.atan2(d[1], d[0]); R2 = np.array([[math.cos(a2), -math.sin(a2), 0], [math.sin(a2), math.cos(a2), 0], [0, 0, 1]])
        return V @ R2.T + [at[0], at[1], z0 - th * 0.3], F
    if k == "megaliths":
        n, rad, h = int(L_.get("n", 9)), L_.get("r", 9.0), L_.get("h", 4.0); VV, FF = [], []
        for j in range(n):
            t = 2 * math.pi * j / n; c = at + rad * np.array([math.cos(t), math.sin(t)]); hh = h * (0.8 + 0.4 * ((j * 7919) % 5) / 4)
            lean = math.radians(((j * 37) % 17 - 8)); rings = [([math.sin(lean) * z, 0, z], [1, 0, 0], [0, 1, 0], 0.95 * (1 - 0.18 * z / hh), 0.42)
                                                                  for z in np.linspace(-1, hh, 6)]   # a tilted slab, not a cylinder
            V, F = _lathe(rings, 5, 0.22, seed + j); Rt = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
            V = V @ Rt.T + [c[0], c[1], float(G.height(c[0], c[1]))]; FF += [[len(VV) + q for q in f] for f in F]; VV += list(V)
        return np.array(VV), FF
    raise ValueError(f"landmark {k}?")


def box_uv(me, period=6.0):
    """UV by cubic projection per face: a face takes the axis of the nearest normal — walls do not stretch into stripes (Workbench has no triplanar projection)."""
    nP = len(me.polygons); N = np.empty(nP * 3); me.polygons.foreach_get("normal", N); N = np.abs(N.reshape(-1, 3)); ax = N.argmax(1)
    ls = np.empty(nP, np.int32); me.polygons.foreach_get("loop_start", ls); lt = np.empty(nP, np.int32); me.polygons.foreach_get("loop_total", lt)
    pid = np.repeat(np.arange(nP), lt); li = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", li)
    co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)[li]; a = ax[pid]
    u = np.where(a == 0, co[:, 1], co[:, 0]); v = np.where(a == 2, co[:, 1], co[:, 2])
    uv = me.uv_layers.new(name="box"); uv.data.foreach_set("uv", (np.stack([u, v], 1) / period).ravel().astype(np.float32))


def tile_material(name, layer, G, rgb, period=6.0):
    """The game's tile material of a biome layer (terrain_tex.BIOMES), repeated; without tiles — a flat colour rgb."""
    try:
        import importlib; TT = importlib.import_module("components.site.terrain_tex")
        biome = G.loc.get("biome") or TT.GROUND_BIOME.get(G.loc.get("ground", "grass"), "temperate"); nm = TT.BIOMES[biome][layer][0][0]
        src = bpy.data.images.load(os.path.join(TT.TILES, nm + "_dif.jpg"), check_existing=True); w, h = src.size
        px = np.empty(w * h * 4, np.float32); src.pixels.foreach_get(px); px = px.reshape(-1, 4)   # layer brightness (as in the ground texture)
        lum = {**TT.LUM, **TT.LUM_BIOME.get(biome, {})}.get(layer, 0.45); px[:, :3] = np.clip(px[:, :3] * lum / max(float(px[:, :3].mean()), 1e-3), 0, 1)
        img = bpy.data.images.new(f"{name}_{nm.split('/')[-1]}", w, h); img.pixels.foreach_set(px.ravel()); img.pack(); bpy.data.images.remove(src)
        m = material(name, img=img, rough=0.85, mean=rgb); tx = next(n for n in m.node_tree.nodes if n.type == "TEX_IMAGE"); tx.extension = "REPEAT"
        return m
    except Exception as e:
        print("tile", layer, e, flush=True); return material(name, rgb=rgb, rough=0.85)


def build_landmarks(G, sc=None, name="landmarks"):
    """Site landmarks (location.json → landmarks) as one part "landmarks" (kind rock): a unique silhouette that a height map cannot give."""
    sc = sc or bpy.context.scene; Ls = G.loc.get("landmarks") or []
    if not Ls: return None, {"landmarks": 0}
    VV, FF = [], []
    for i, L_ in enumerate(Ls):
        V, F = landmark_mesh(L_, G, 11 + i); FF += [[len(VV) + q for q in f] for f in F]; VV += list(V)
    me = bpy.data.meshes.new(name); me.from_pydata(np.array(VV).tolist(), [], FF); me.update(); me.shade_smooth(); box_uv(me, 8.0)
    me.materials.append(tile_material("landmark_rock", "cliff", G, {"desert": (0.62, 0.42, 0.30), "alpine": (0.52, 0.53, 0.56),
                                                                     "volcanic": (0.24, 0.23, 0.23)}.get(G.loc.get("biome"), (0.50, 0.48, 0.44)), 8.0))
    ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob); ob["kind"] = "rock"
    return ob, {"landmarks": len(Ls), "landmark_verts": len(VV)}


def _box(c, t, n, hx, hy, z0, z1):
    """A box: centre c (xy), axis t along and n across, half-sizes hx, hy, heights z0…z1 → (V, F)."""
    V = [[c[0] + t[0] * sx * hx + n[0] * sy * hy, c[1] + t[1] * sx * hx + n[1] * sy * hy, z] for z in (z0, z1) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return V, [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]


def bridge_mesh(B, G):
    """A bridge (location.json → bridges, a deck path from dry land to dry land): a deck with parapets — a "trough" cross-section along the path, piers in the riverbed
    every ~8 m, abutments in the banks. Vertices are in world metres."""
    P = np.asarray(B["path"], float); w = float(B.get("w", 5.2)) / 2; lv = float(B.get("level", P[:, 2].min() - 2))
    T_ = np.gradient(P[:, :2], axis=0); T_ /= np.linalg.norm(T_, axis=1, keepdims=True) + 1e-9; N_ = np.column_stack([-T_[:, 1], T_[:, 0]])
    sec = [(-w, -0.7), (w, -0.7), (w, 0.75), (w - 0.35, 0.75), (w - 0.35, 0.05), (-w + 0.35, 0.05), (-w + 0.35, 0.75), (-w, 0.75)]   # (across, height)
    m = len(sec); V = [[P[k, 0] + N_[k, 0] * u, P[k, 1] + N_[k, 1] * u, P[k, 2] + z] for k in range(len(P)) for u, z in sec]
    F = [[k * m + q, k * m + (q + 1) % m, (k + 1) * m + (q + 1) % m, (k + 1) * m + q] for k in range(len(P) - 1) for q in range(m)]
    F += [list(range(m))[::-1], [(len(P) - 1) * m + q for q in range(m)]]
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))]; n_p = max(0, int(L[-1] / 8))
    parts = [(np.interp(s_, L, P[:, 0]), np.interp(s_, L, P[:, 1]), np.interp(s_, L, P[:, 2]), 0.6, lv - 2.5) for s_ in L[-1] * np.arange(1, n_p + 1) / (n_p + 1)]
    for k, out in ((0, -1), (len(P) - 1, 1)):                       # abutments: from the deck into the bank, one metre outward
        parts.append((P[k, 0] + T_[k, 0] * out, P[k, 1] + T_[k, 1] * out, P[k, 2], 1.6, min(float(G.height(P[k, 0], P[k, 1])), P[k, 2]) - 2.0))
    for x, y, z, hx, zb in parts:
        k_ = int(np.argmin(np.hypot(P[:, 0] - x, P[:, 1] - y))); Vb, Fb = _box((x, y), T_[k_], N_[k_], hx, w + 0.25, zb, z - 0.05)
        F += [[len(V) + q for q in f] for f in Fb]; V += Vb
    return np.array(V), F


def build_bridges(G, sc=None, name="bridges"):
    """The site's bridges as one part "bridges" (kind wall, like buildings): the stone of the biome's pad by cubic projection."""
    import bmesh
    sc = sc or bpy.context.scene; Bs = G.loc.get("bridges") or []
    if not Bs: return None, {"bridges": 0}
    VV, FF = [], []
    for B in Bs:
        V, F = bridge_mesh(B, G); FF += [[len(VV) + q for q in f] for f in F]; VV += list(V)
    me = bpy.data.meshes.new(name); me.from_pydata(np.array(VV).tolist(), [], FF); bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces); bm.to_mesh(me); bm.free(); me.update(); box_uv(me, 3.0)
    me.materials.append(tile_material("bridge_stone", "pad", G, (0.52, 0.49, 0.45), 3.0))
    ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob); ob["kind"] = "wall"
    return ob, {"bridges": len(Bs)}


STANDS, AVOID = [], []                                              # fighters' spots and building footprints (m): the rock dressing avoids them

# forests by biome: patch density, species (share), slope limit; "below snow" — for mountains
FOREST = {"temperate": {"density": 0.38, "kinds": {"broad": 0.65, "conifer": 0.25, "shrub": 0.10}},
          "jungle": {"density": 0.7, "kinds": {"broad": 0.45, "palm": 0.35, "shrub": 0.20}},
          "alpine": {"density": 0.22, "kinds": {"conifer": 0.9, "shrub": 0.1}, "below_snow": True},
          "desert": {"density": 0.03, "kinds": {"dead": 0.6, "shrub": 0.4}}, "volcanic": {"density": 0.04, "kinds": {"dead": 1.0}},
          "wasteland": {"density": 0.06, "kinds": {"dead": 0.8, "shrub": 0.2}}, "alien": {"density": 0.0, "kinds": {}}}
CROWN = {"temperate": (0.26, 0.37, 0.15), "jungle": (0.17, 0.34, 0.12), "alpine": (0.14, 0.25, 0.17), "desert": (0.35, 0.33, 0.22),
         "volcanic": (0.18, 0.16, 0.14), "wasteland": (0.28, 0.25, 0.18), "alien": (0.30, 0.22, 0.30)}


def blob_tree(kind, rng):
    """A light tree (200–400 vertices): the trunk is a hexagonal prism, the crown — lumps (deciduous), tiers of cones (conifer), an umbrella (palm),
    bare branches (dead), a low lump (shrub). Height 1 = the tree's height; returns (V, F, the share of trunk vertices)."""
    import bmesh
    from mathutils import Matrix, Vector, noise
    bm = bmesh.new(); h = 1.0

    def cyl(r0, r1, z0, z1, x=0.0, y=0.0, seg=6):
        g = bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=r0, radius2=r1, depth=z1 - z0)
        bmesh.ops.translate(bm, verts=g["verts"], vec=(x, y, (z0 + z1) / 2))

    def blob(c, r, sub=1, squash=1.0):
        g = bmesh.ops.create_icosphere(bm, subdivisions=sub, radius=r); o = Vector((rng.uniform(0, 50), rng.uniform(0, 50), rng.uniform(0, 50)))
        for v in g["verts"]:
            v.co = v.co * (1 + 0.28 * noise.noise(v.co * 3.0 / r + o)); v.co.z *= squash; v.co += Vector(c)
    if kind == "broad":
        cyl(0.045, 0.03, 0, 0.55)
        for _ in range(int(rng.integers(3, 6))):
            a = rng.uniform(0, 2 * math.pi); d = rng.uniform(0.05, 0.18); blob((d * math.cos(a), d * math.sin(a), rng.uniform(0.55, 0.8)), rng.uniform(0.16, 0.26), 1, 0.85)
    elif kind == "conifer":
        cyl(0.03, 0.02, 0, 0.25)
        n_ = int(rng.integers(3, 5))
        for k in range(n_):
            z = 0.18 + k * 0.75 / n_; r = 0.24 * (1 - k / n_) + 0.05
            g = bmesh.ops.create_cone(bm, cap_ends=True, segments=8, radius1=r, radius2=0.0, depth=0.32); bmesh.ops.translate(bm, verts=g["verts"], vec=(0, 0, z + 0.16))
    elif kind == "palm":
        lean = rng.uniform(-0.12, 0.12)
        for k in range(5): cyl(0.03, 0.026, k * 0.17, k * 0.17 + 0.18, lean * k / 5, 0)
        for k in range(6):
            a = k * math.pi / 3 + rng.uniform(-0.2, 0.2); blob((lean + 0.18 * math.cos(a), 0.18 * math.sin(a), 0.86), 0.12, 1, 0.35)
    elif kind == "dead":
        cyl(0.04, 0.02, 0, 0.75)
        for k in range(int(rng.integers(3, 6))):
            a = rng.uniform(0, 2 * math.pi); z = rng.uniform(0.35, 0.7); L = rng.uniform(0.15, 0.3)
            g = bmesh.ops.create_cone(bm, cap_ends=True, segments=4, radius1=0.012, radius2=0.004, depth=L)
            R = Matrix.Rotation(rng.uniform(0.6, 1.1), 4, Vector((-math.sin(a), math.cos(a), 0)))
            bmesh.ops.rotate(bm, verts=g["verts"], cent=(0, 0, 0), matrix=R); bmesh.ops.translate(bm, verts=g["verts"], vec=(math.cos(a) * L * 0.4, math.sin(a) * L * 0.4, z))
    else:                                                           # shrub
        for _ in range(int(rng.integers(2, 4))):
            a = rng.uniform(0, 2 * math.pi); d = rng.uniform(0.05, 0.15); blob((d * math.cos(a), d * math.sin(a), 0.25), rng.uniform(0.22, 0.32), 1, 0.7)
    V = np.array([v.co[:] for v in bm.verts]); F = [[v.index for v in f.verts] for f in bm.faces]; bm.free()
    return V, F


def dress_trees(G, sc=None, seed=11, name="trees", limit=1600):
    """Groves by biome: noise patches (the biome density or relief.forest), not steeper than 28°, dry, not on a road or pad, not near fighters and
    buildings, in mountains — below the snow line; a 5–9 m step on a grid, size 0.8–1.25 of the species norm. One part, "trees" (kind tree)."""
    sc = sc or bpy.context.scene; L = G.loc; biome = L.get("biome") or {"snow": "alpine", "sand": "desert", "ash": "volcanic", "mud": "wasteland"}.get(L.get("ground"), "temperate")
    cfg = dict(FOREST.get(biome, FOREST["temperate"])); cfg.update(L.get("forest") or {})
    if L.get("forest") is False or not cfg.get("density") or not cfg.get("kinds"): return None, {"trees": 0}
    from components.site.terrain import fbm
    rng = np.random.default_rng(seed); nb = G.near_bounds(); step = 2.0
    xs = np.arange(nb[0] + 6, nb[2] - 6, step); ys = np.arange(nb[1] + 6, nb[3] - 6, step); X, Y = np.meshgrid(xs, ys)
    patch = fbm(X / 55.0, Y / 55.0, 3, seed) * 0.5 + 0.5; thr = 1.0 - float(cfg["density"])
    ok = (patch > thr) & (G.slope(X, Y) < 28) & (G.water(X, Y) < 0.01) & (G.mask("road", X, Y) < 0.2) & (G.mask("pad", X, Y) < 0.2)
    if cfg.get("below_snow"):
        sl = L.get("snowline"); ok &= (G.height(X, Y) < float(sl) - 4) if sl not in (None, "none") else ok
    P = np.column_stack([X[ok], Y[ok]]) + rng.uniform(-1, 1, (int(ok.sum()), 2))
    dens = (patch[ok] - thr) / max(1 - thr, 1e-3)                    # denser in the middle of a patch, sparser toward the edge
    P = P[rng.random(len(P)) < 0.15 + 0.6 * dens]
    if len(P):
        P = P[rng.permutation(len(P))]; key = np.floor(P / 6.5).astype(np.int64); _, first = np.unique(key[:, 0] * 1000003 + key[:, 1], return_index=True); P = P[first]
    if STANDS:
        A_ = np.array(STANDS); P = np.array([p for p in P if np.min(np.hypot(A_[:, 0] - p[0], A_[:, 1] - p[1])) > 3.0]).reshape(-1, 2)
    if AVOID:
        from components.site.terrain import sdf_poly
        P = np.array([p for p in P if all(sdf_poly(np.array([p[0]]), np.array([p[1]]), Q)[0] > 3.0 for Q in AVOID)]).reshape(-1, 2)
    if len(P):                                                      # story locations are open: clearings at the plan's anchors, cleared strips along roads and rivers
        from components.site.terrain import line_info
        keep = np.ones(len(P), bool)
        for a_ in L.get("public", []):
            if a_ in L["anchors"]: x_, y_, _ = L["anchors"][a_]; keep &= np.hypot(P[:, 0] - x_, P[:, 1] - y_) > 12.0
        for pth in L.get("paths", {}).values():
            if pth["kind"] in ("road", "river"):
                Q_ = np.asarray(pth["pts"])[:, :2]; d_ = line_info(P[:, 0], P[:, 1], Q_)[0]; keep &= d_ > (pth.get("w") or 4) / 2 + 6.0
        P = P[keep]
    P = P[:limit]
    if not len(P): return None, {"trees": 0}
    kinds = list(cfg["kinds"]); wk = np.array([cfg["kinds"][k] for k in kinds], float); wk /= wk.sum()
    H0 = {"broad": 11.0, "conifer": 14.0, "palm": 12.0, "dead": 8.0, "shrub": 2.2}
    proto = {k: [blob_tree(k, np.random.default_rng(seed * 31 + j)) for j in range(4)] for k in kinds}
    V_, F_, C_, off = [], [], [], 0
    for x, y in P:
        k = kinds[int(rng.choice(len(kinds), p=wk))]; V, F = proto[k][int(rng.integers(4))]; s = H0[k] * rng.uniform(0.8, 1.25)
        a = rng.uniform(0, 2 * math.pi); R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        Q = (V * s) @ R.T + [x, y, float(G.height(x, y)) - 0.2]; V_.append(Q); F_ += [[off + q for q in f] for f in F]; off += len(Q)
    me = bpy.data.meshes.new(name); me.from_pydata(np.vstack(V_).tolist(), [], F_); me.update(); me.shade_smooth()
    me.materials.append(material("crown", rgb=CROWN.get(biome, CROWN["temperate"]), rough=0.9))
    ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob); ob["kind"] = "tree"
    return ob, {"trees": int(len(P)), "tree_verts": off}


def dress_from(G, D, sc=None, seed=5, trees=True):
    """Dressing by the terrain_dress plan (dress.json): rocks (diameter s, sunk by sink, tilted toward the normal by tilt) — the part "rocks"
    (kind rock), trees and shrubs (species sub, height — the species' H0 × s) — the part "trees" (kind tree)."""
    sc = sc or bpy.context.scene; rng = np.random.default_rng(seed); L = G.loc
    biome = L.get("biome") or "temperate"; res = {}
    R = [p for p in D["places"] if p["kind"] == "rock"]; T = [p for p in D["places"] if p["kind"] == "tree"]
    if R:
        shapes = rock_shapes(seed=seed); crags = rock_shapes(seed=seed + 7, sub=1, crag=True); V_, F_, off = [], [], 0; flat = []
        for p in R:
            sh_ = crags if p.get("facet") else shapes; V, F = sh_[rng.integers(len(sh_))]; s = p["s"]; flat += [bool(p.get("facet"))] * len(F)
            sx, sy, sz = s * rng.uniform(0.8, 1.2), s * rng.uniform(0.6, 0.9), s * rng.uniform(0.45, 0.75)
            a = math.radians(p["rot"]); Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
            n = np.asarray(G.normal(p["x"], p["y"]), float).reshape(3); up = np.array([0, 0, 1.0]); t = float(p.get("tilt", 0))
            z_ = (1 - t) * up + t * n; z_ /= np.linalg.norm(z_); x_ = np.cross([0, 1.0, 0], z_); x_ /= np.linalg.norm(x_) + 1e-9; y_ = np.cross(z_, x_)
            B = np.column_stack([x_, y_, z_])                         # tilt along the ground normal
            P = ((V * [sx / 2, sy / 2, sz / 2]) @ Rz.T) @ B.T + [p["x"], p["y"], p["z"] - sz * float(p.get("sink", 0))]
            V_.append(P); F_ += [[off + q for q in f] for f in F]; off += len(P)
        me = bpy.data.meshes.new("rocks"); me.from_pydata(np.vstack(V_).tolist(), [], F_); me.update(); me.shade_smooth(); box_uv(me, 4.0)
        if any(flat): me.polygons.foreach_set("use_smooth", [not f for f in flat]); me.update()   # faceted ones — flat faces, edges visible
        me.materials.append(tile_material("rock", "rock", G, (0.47, 0.45, 0.42), 4.0))
        ob = bpy.data.objects.new("rocks", me); sc.collection.objects.link(ob); ob["kind"] = "rock"; res.update(rocks=len(R), rock_verts=off)
    if T and trees:                                                 # trees=False — the trees are placed by the Plants component (veg=1)
        H0 = {"broad": 11.0, "conifer": 14.0, "palm": 12.0, "dead": 8.0, "shrub": 2.2}
        kinds = sorted({p["sub"] or "broad" for p in T}); proto = {k: [blob_tree(k, np.random.default_rng(seed * 31 + j)) for j in range(4)] for k in kinds}
        V_, F_, off = [], [], 0
        for p in T:
            k = p["sub"] or "broad"; V, F = proto[k][int(rng.integers(4))]; s = H0.get(k, 10.0) * p["s"]
            a = math.radians(p["rot"]); Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
            Q = (V * s) @ Rz.T + [p["x"], p["y"], p["z"] - 0.2]; V_.append(Q); F_ += [[off + q for q in f] for f in F]; off += len(Q)
        me = bpy.data.meshes.new("trees"); me.from_pydata(np.vstack(V_).tolist(), [], F_); me.update(); me.shade_smooth()
        me.materials.append(material("crown", rgb=CROWN.get(biome, CROWN["temperate"]), rough=0.9))
        ob = bpy.data.objects.new("trees", me); sc.collection.objects.link(ob); ob["kind"] = "tree"; res.update(trees=len(T), tree_verts=off)
    res["layers"] = D.get("by_layer"); return res


def camera_from(spec, sc, name="cam"):
    cd = bpy.data.cameras.new(name); cd.lens = spec["lens_mm"]; cd.sensor_width = spec.get("sensor_width_mm", 36.0)
    cd.sensor_fit = spec.get("sensor_fit", "AUTO"); cd.clip_start, cd.clip_end = 0.1, 40000.0
    ob = bpy.data.objects.new(name, cd); sc.collection.objects.link(ob); ob.matrix_world = Matrix(spec["matrix_world"])
    return ob


def look(sc, mode="workbench"):
    """A view for the eye and the sheet: Workbench, colour — texture, studio light, cavities; the sky is light. mode=eevee — the ground shader
    (bl_terrain_surf) under sun and sky: a check of the surfaces."""
    if mode == "eevee":
        sc.render.engine = "BLENDER_EEVEE"; sc.eevee.taa_render_samples = 16; sc.view_settings.view_transform = "AgX"
        w = bpy.data.worlds.new("sky"); w.use_nodes = True; bg = w.node_tree.nodes["Background"]
        bg.inputs[0].default_value = (0.55, 0.64, 0.78, 1); bg.inputs[1].default_value = 0.9; sc.world = w
        for o in sc.objects:
            if o.type == "LIGHT" and o.data.type == "SUN": o.data.energy = 3.5
        return
    sc.render.engine = "BLENDER_WORKBENCH"; d = sc.display.shading
    d.light, d.color_type, d.show_cavity = "STUDIO", "TEXTURE", True
    d.background_type, d.background_color = "VIEWPORT", (0.78, 0.83, 0.9)
    sc.display.shading.show_specular_highlight = False


def lit_shots(sc, folder, cams, objs, out):
    """Frames with the site node's light and haze — like bl_color light=far (core/light/far.py, bl_far.py): the plan's sky and lamps by biome,
    the far terrain, lit.exr with depth + sky.exr → far.py post (haze by depth, AgX) → <out>/<camera>.png. A check of surfaces
    in real light: under bare sun without haze the panorama is not readable."""
    import shutil, subprocess
    from core.light import bl_far
    py, fp = os.path.join(_core_paths.ROOT, ".venv", "bin", "python"), os.path.join(out, "far_plan.json")
    r = subprocess.run([py, "-m", "core.light.far", "resolve", folder, f"out={fp}"], capture_output=True, text=True)
    if r.returncode: raise SystemExit("far.py resolve: " + r.stderr[-600:])
    FAR = json.load(open(fp)); FAR["path"] = fp
    for o in [o for o in sc.objects if o.type == "LIGHT"]: bpy.data.objects.remove(o)
    sc.world = bl_far.build_world(FAR); bl_far.render_horizon(sc, FAR, os.path.splitext(fp)[0] + "_horizon.json")
    bl_far.attach_terrain(sc, FAR, None)
    sc.render.engine = "BLENDER_EEVEE"; sc.eevee.taa_render_samples = 16; bl_far.attach(sc, FAR)
    dirs, shots = [], []
    for c, ob in zip(cams, objs):
        d = os.path.join(out, "lit", c.get("name", "cam")); os.makedirs(d, exist_ok=True)
        sc.camera = ob; sc.render.resolution_x, sc.render.resolution_y = c["resolution"]; bl_far.fit_clip(ob, FAR); bl_far.per_camera(sc, c)
        sc.render.film_transparent = False; bl_far.render_sky(sc, os.path.join(d, "sky.exr"))
        json.dump({"plan": fp, "hash": FAR["hash"], "ev": FAR["light"]["ev"], "mode": "lit", "camera": c, "fx": False, "frame": None},
                  open(os.path.join(d, "far.json"), "w"))
        sc.render.film_transparent = True; im = sc.render.image_settings; vl = sc.view_layers[0]; vl.use_pass_z = True
        vl.use_pass_diffuse_color = True                            # the surface colour without light and the ground mask (the AOV ground of the ground shader) —
        if "ground" not in vl.aovs: a_ = vl.aovs.add(); a_.name, a_.type = "ground", "VALUE"   # for the measure components/site/bench/spread.py
        im.media_type, im.file_format, im.color_depth, im.color_mode = "MULTI_LAYER_IMAGE", "OPEN_EXR_MULTILAYER", "32", "RGBA"
        sc.render.filepath = os.path.join(d, "lit.exr"); bpy.ops.render.render(write_still=True); dirs.append(d)
        im.media_type, im.file_format, im.color_depth = "IMAGE", "PNG", "8"; vl.use_pass_z = vl.use_pass_diffuse_color = False
    r = subprocess.run([py, "-m", "core.light.far", "post", *dirs], capture_output=True, text=True)
    if r.returncode: raise SystemExit("far.py post: " + r.stderr[-600:])
    for c, d in zip(cams, dirs):
        p = os.path.join(out, f"{c.get('name', 'cam')}.png"); shutil.copy(os.path.join(d, "lit.png"), p); shots.append(p)
    return shots


def main():
    a = sys.argv[sys.argv.index("--") + 1:]; folder, out = os.path.abspath(a[0]), os.path.abspath(a[1])
    opt = dict(kv.split("=", 1) for kv in a[2:])
    for ob in list(bpy.data.objects): bpy.data.objects.remove(ob)
    sc = bpy.context.scene; G = Ground(folder); res = {}
    ter, r1 = build_terrain(G, core=float(opt["core"]) if "core" in opt else None, px=int(opt.get("px", 4)), sc=sc, surface=False); res.update(r1)
    if opt.get("surf", "1") != "0":                                 # the ground shader from ready parts — for Eevee; Workbench — an image; surf_far=baked|0
        sf = opt.get("surf_far", "1"); res.update(apply_surface(ter, G, opt.get("surf_set"), {"0": False, "baked": "baked"}.get(sf, True)))
    wat, r2 = build_water(G, sc=sc); res.update(r2); lm, r5 = build_landmarks(G, sc=sc); res.update(r5)
    res.update(build_bridges(G, sc=sc)[1])
    if opt.get("rocks", "0") == "1":                                # rocks — as scans by the rocks.json plan (the Site step rocks; no plan — computed now)
        from components.site import bl_rocks as BRK
        rj = os.path.join(folder, "rocks.json")
        if not os.path.exists(rj):
            import subprocess
            subprocess.run([os.path.join(_core_paths.ROOT, ".venv", "bin", "python"), "-m", "components.site.terrain_rocks", folder], capture_output=True, cwd=_core_paths.ROOT)
        if os.path.exists(rj): res.update(BRK.build(G, json.load(open(rj)), sc=sc))
    if opt.get("dress", "1") != "0":                                # dressing the cliffs with rocks and groves by biome (dress=0 — none)
        t_ = time.time(); dj = os.path.join(folder, "dress.json")
        if os.path.exists(dj): res.update(dress_from(G, json.load(open(dj)), sc=sc, trees=opt.get("veg", "0") != "1"))   # one mechanism: the terrain_dress plan
        else: rk, r3 = dress_rocks(G, sc=sc); res.update(r3); tr, r4 = dress_trees(G, sc=sc); res.update(r4)
        res["s_dress"] = round(time.time() - t_, 2)
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun); sun.rotation_euler = (math.radians(50), 0, math.radians(40))
    cams = json.load(open(opt["cams"])) if "cams" in opt else []
    objs = [camera_from(c, sc, c.get("name", f"cam{i + 1}")) for i, c in enumerate(cams)]
    if objs: sc.camera = objs[0]; sc.render.resolution_x, sc.render.resolution_y = cams[0]["resolution"]
    if objs and opt.get("near", "0") == "1" and any(o.name.startswith("terrain") for o in sc.objects):   # ground near the camera — a close-up
        from core.surface import bl_terrain_near as TN; t_ = time.time(); res.update(TN.build(ter, G, objs, int(opt.get("near_levels", 3)), sc=sc)); res["s_near"] = round(time.time() - t_, 1)
    if opt.get("veg", "0") == "1":                                  # vegetation of the Plants component instead of stand-in trees and scan grass:
        from components.plants import api as PL                  # a computed forest (forest.py), a meadow near the cameras (bl_meadow) — a debt: this is another component's code — Plants
        t_ = time.time(); eyes = []
        for c, ob in zip(cams, objs):
            d_ = ob.matrix_world.to_3x3() @ __import__("mathutils").Vector((0, 0, -1)); p_ = ob.matrix_world.translation; eyes.append((p_.x, p_.y, p_.z, math.atan2(d_.y, d_.x)))
        res["plants"] = PL.dress_place(G, eyes, sc=sc, density=float(opt.get("veg_density", 0.45))); res["s_veg"] = round(time.time() - t_, 1)
    if objs and opt.get("clutter", "0") == "1":                     # small items near the camera: grass, pebbles, twigs (Poly Haven scans, instances)
        from components.site import bl_terrain_clutter as TC
        from components.site import terrain_tex as TT_
        sj_ = json.load(open(os.path.join(folder, "surface.json"))) if os.path.exists(os.path.join(folder, "surface.json")) else {}
        bio_ = G.loc.get("biome") or TT_.GROUND_BIOME.get(G.loc.get("ground", "grass"), "temperate")
        cfg_ = sj_.get("clutter")
        if cfg_ and opt.get("veg", "0") == "1":                      # the grass is placed by the Plants meadow — the small items keep pebbles, twigs, rocks
            cfg_ = dict(cfg_, density={k: {a: v for a, v in d.items() if not a.startswith(("grass", "weeds"))} for k, d in cfg_["density"].items()})
        res.update(TC.build(G, objs, cfg_, masks=sj_.get("masks"), snowline=TT_.snowline_of(G.loc, bio_), sc=sc))
    shots = []
    if "shots" in opt and objs:
        os.makedirs(opt["shots"], exist_ok=True)
        if opt.get("look") == "lit": shots = lit_shots(sc, folder, cams, objs, opt["shots"])
        else: look(sc, opt.get("look", "workbench"))
        for c, ob in zip(cams, objs) if opt.get("look") != "lit" else ():
            sc.camera = ob; sc.render.resolution_x, sc.render.resolution_y = c["resolution"]
            p = os.path.join(opt["shots"], f"{c.get('name', 'cam')}.png"); sc.render.filepath = p; bpy.ops.render.render(write_still=True); shots.append(p)
        sc.camera = objs[0]; sc.render.resolution_x, sc.render.resolution_y = cams[0]["resolution"]
    os.makedirs(os.path.dirname(out), exist_ok=True); bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
    print("__RESULT__" + json.dumps({"blend": out, **res, "cams": len(objs), "shots": shots}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
