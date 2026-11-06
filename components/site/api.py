"""Public entry of Site (ARCHITECTURE.md: the mechanism stays with its owner, neighbours call api): ground from the terrain and the site from a plan. Buildings
edit the terrain (terraces, moat, courtyard) and call from here to rebake the ground and rebuild its object; Layout builds
its plan's site from here — the terrain from a description, the site folder, texture, sheet, scene and cameras by anchors, in Blender — the site parts and
dressing (_BL, on first access). The terrain and the surface recipe are the Site layer."""


def refresh(folder):
    """The ground again from the edited terrain of the scene folder: the texture (relief_tex), moisture and masks (relief_w.npz), the surface recipe
    and small items (surface.json) — the project's Python. → the result of refresh.run."""
    from components.site import refresh as R
    return R.run(folder)


def build_terrain(G, sc=None):
    """The ground object "terrain" from the terrain G (core.surface.ground.Ground) in Blender — rebuilt from scratch. → (object, info)."""
    import bpy
    from components.site import bl_relief
    from components.site.steps import SURFACE
    sc = sc or bpy.context.scene                                    # the ground is as set by the "Site" step (the ground knob)
    return bl_relief.build_terrain(G, sc=sc, surface=SURFACE.get(sc.get("sk_ground", "scans"), True))


# ── site from a plan (Layout) ─────────────────────────────────────────────────────────────────────────────────────────────
def place_build(spec, fast=False):
    """A site from a description {"relief": {…}} (landforms, archetypes, anchors, pads and supports) — terrain.build. fast — anchors only."""
    from components.site import terrain as T
    return T.build(spec, fast=fast)


def place_save(o, folder):
    """A site (the result of place_build) into a folder: relief.npz, location.json and companion files (terrain.save)."""
    from components.site import terrain as T
    return T.save(o, folder)


def place_texture(folder):
    """The site's ground texture — the game's tiles by biome (terrain_tex.bake)."""
    from components.site import terrain_tex
    return terrain_tex.bake(folder)


def place_sheet(folder, out):
    """A site sheet in texture colour (terrain_sheet.sheet; the old path)."""
    from components.site import terrain_sheet
    return terrain_sheet.sheet(folder, out)


def place_finish(scene_dir, folder):
    """A scene from the site folder: water and rocks have their own kind, far_z, relief.npz alongside (terrain.finish_scene)."""
    from components.site import terrain as T
    return T.finish_scene(scene_dir, folder)


def place_views(G, spec):
    """Site cameras by anchors (terrain.views) — as a list of camera descriptions."""
    from components.site import terrain as T
    return T.views(G, spec)


_BL = ("build_water", "build_landmarks", "build_bridges", "dress_rocks", "dress_trees", "AVOID", "STANDS")   # Blender: bl_relief


def __getattr__(name):
    """Site parts and dressing in Blender (bl_relief) — on first access; AVOID, STANDS — lists of the build module: building footprints and
    fighters' spots that the rock and tree dressing avoids (Layout fills them in before dress_*)."""
    if name in _BL:
        from components.site import bl_relief
        return getattr(bl_relief, name)
    raise AttributeError(f"components.site.api: no '{name}'")
