"""Public entry point of Water (ARCHITECTURE.md: the mechanism lives with its owner, neighbors and the core call the entry point):
the core calls the light hook (core/hooks.py) — when a scene is opened by the frames output, the water is rebuilt from the
object numbers (bl_live.attach), the frame is computed on the fly. The wave engine and the water-column color work without
Blender: sea_state (the state of a water body), look (the substance in the water)."""


def light_changed(sc, P):
    """The light hook: the scene's live water (frame and render handlers); no water — nothing happens. → the number of water bodies."""
    if not any(o.get("water_params") for o in sc.objects): return 0
    from components.water import bl_live
    return bl_live.attach(sc, P)


def sea_state(u10, fetch=None, depth=None, **kw):
    """Water state without Blender: waves.Sea(...).stats() — Hs, Tp, λp, slopes, whitecap fraction."""
    from components.water import waves
    return waves.Sea(u10=u10, fetch=fetch, depth=depth, foam={"thr": -10.0}, **kw).stats()


def look(preset):
    """A substance in the water (optics.PRESETS) → water-column color, attenuation, the Forel–Ule index."""
    from components.water import optics
    return {k: v for k, v in optics.preset(preset).items() if k != "rrs"}
