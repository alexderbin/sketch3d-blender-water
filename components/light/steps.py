"""Entry of the "Light and sky" component for the step-by-step build: the light step.
plan (project Python): the scene's light description (far.yaml | battle.yaml | site biome | world.json) + the knobs time (far.TIME), weather
(far.WEATHER), style (far.STYLE), sky_image → the node plan far_plan.json (far.plan_of); shadows: off — sun.shadow = false in the plan
(the sun/moon lamp without shadow: bl_far.add_lamps). build (Blender): world, far terrain, city, clouds, lights, lamps and the AgX view — into the scene;
old lamps are removed. gives: light_plan (path of the node plan), exposure — for outputs and effects.
Sky weather: the weather knob, otherwise sky_weather from the gives of the Materials weather step (it comes before light in the step pipeline: the weather
layer of materials works over all the assembled geometry, the node's sky works from its preset)."""
import json, os


def shadows_on(v):
    """Knob shadows: on | off → does the sun/moon cast a shadow in the frame? Empty — on. Unquoted YAML reads on/off as true/false — both forms are accepted."""
    if v is None or isinstance(v, bool): return v is not False
    s = str(v).strip().lower()
    if s in ("on", "true", "yes", "1"): return True
    if s in ("off", "false", "no", "0"): return False
    raise ValueError(f"shadows: '{v}' — allowed values are on | off")


def plan_light(M, keys, ctx):
    from core.light import far
    t = keys.get("time")                                                 # node time far.TIME (dawn … night)
    over = {"time": t} if t else {}
    for k in ("weather", "style"):
        if keys.get(k): over[k] = keys[k]
    if keys.get("sky_image"): over["sky_image"] = keys["sky_image"]             # sky as a picture (path[:rotation] or {path, yaw, sun…})
    if not over.get("weather") and ctx["gives"].get("sky_weather"):      # the "weather" step (Materials) comes before light: the sky follows its preset, as with frames.py --weather
        over["weather"] = ctx["gives"]["sky_weather"]
    out = os.path.join(ctx["out"], "far_plan.json")
    P = far.plan_of(ctx["scene"], over, out=out)
    shadow = shadows_on(keys.get("shadows"))
    if not shadow:                                                       # shadows: off — the lamp casts no shadow (a forest-scene draft:
        P["sun"]["shadow"] = False                                       # the sun's shadow takes 24–30 % of the take frame); the plan is changed in place
        tmp = f"{out}.{os.getpid()}"
        with open(tmp, "w") as f: json.dump({k: v for k, v in P.items() if k != "path"}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, out)
    return {"far_plan": out, "time": over.get("time"), "weather": over.get("weather"), "exposure": P["light"]["ev"],
            "far_relief": bool(P.get("terrain")), "shadows": "on" if shadow else "off"}


def build_light(ctx):
    import bpy
    from core.light import bl_far
    sc = ctx["sc"]; fp = ctx["plan"]["far_plan"]
    P = json.load(open(fp)); P["path"] = fp
    for o in [o for o in sc.objects if o.type == "LIGHT" and not o.get("sk_keep")]: bpy.data.objects.remove(o)   # sk_keep — neighbours' lamps (hearths, battle flashes)
    sc.world = bl_far.build_world(P)                                 # the plan's sky comes first: the haze needs the horizon table
    bl_far.render_horizon(sc, P, os.path.splitext(fp)[0] + "_horizon.json")
    ground_far = bpy.data.objects.get("ground_far")
    n_ter = len(bl_far.attach_terrain(sc, P, ground_far))            # the plan's far terrain (mountains to the horizon)
    bl_far.attach_city(sc, P)
    fx = bl_far.attach_clouds(sc, P) + bl_far.attach_fires(sc, P)
    lamps = bl_far.attach(sc, P)                                     # the plan's world, lamps, AgX and exposure (the sun/moon shadow follows sun.shadow)
    return {"light_plan": fp, "exposure": P["light"]["ev"], "lamps": len(lamps), "shadow_lamps": sum(o.data.use_shadow for o in lamps),
            "far_relief": n_ter, "effects": len(fx)}
