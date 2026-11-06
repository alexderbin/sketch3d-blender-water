"""Public entry of Plants (ARCHITECTURE.md: the mechanism stays with its owner, neighbours call api): light through foliage and plants in the maps.
The core calls them through the passport's extension points (hooks, core/hooks.py): light — after the world and lamps of the light plan (core/light/bl_far.attach),
maps_before / maps_after — around the split of the scene into map parts (core/outputs/bl_parts.collect); the core does not import Plants."""


def light_changed(sc, P):
    """The scene's light is set: the foliage shadow via doubles offset from the plan P's sun/moon (bl_veg_light). → the number of doubles."""
    if not any(o.get("kind") == "plant" for o in sc.objects): return 0
    from components.plants import bl_veg_light
    return bl_veg_light.aim(sc, P)


def maps_before(sc):
    """The scene's plants → proxies for the maps (bl_maps.for_maps: crowns as hulls, a trunk); no plants — nothing. → a summary."""
    if not any(o.get("kind") == "plant" for o in sc.objects): return None
    from components.plants import bl_maps
    return bl_maps.for_maps(sc)


def maps_after(sc):
    """After the maps: remove the proxies, the original plants as they were (bl_maps.undo_maps)."""
    from components.plants import bl_maps
    bl_maps.undo_maps(sc)


def dress_place(*a, **kw):
    """Computed forest and meadow near the cameras in place of the old site-scene path (bl_forest.dress_place; called by Site — components/site/bl_relief.py)."""
    from components.plants import bl_forest
    return bl_forest.dress_place(*a, **kw)
