# Light — time-of-day sky, sun, moon, haze, distant mountains and night lighting for Blender with AgX tone mapping

Light is a procedural sky and time-of-day lighting system for Blender, written in Python: a few words such as
`time: sunset` and `weather: haze` set the sun, moon, sky, haze, distant mountains and night look of a scene. It gives
every scene one "light node": sun and moon for a time of day, a physically based sky, haze layers, cloud decks and
cumulus, distant terrain and a city beyond the site, fires and flashes, night rendering and Blender's AgX view.
Everything the node knows is computed once in Python (`core/light/far.py`: description → plan) and Blender only takes
the numbers (`core/light/bl_far.py`), so frames, sketches, previews and map packs share one light. The `light` step goes
last before the outputs — changing the time of day rebuilds only the light and the frames — and gives the light plan
and the exposure to the outputs and to effects.

**Use it to:**

- Set the time of day — dawn, sunrise, noon, evening, sunset, afterglow, dusk, night — with the matching sun and sky of
  a Blender scene: `light: {time: sunset}`
- Change the sky weather — clear, cumulus, haze, overcast, storm, rain, war_smoke, mist — for an overcast day, a misty
  dawn or a night fire: `light: {time: day, weather: overcast}`
- Light a night scene with a moon (its phase follows the sun), stars and a blue night shift: `light: {time: night}`
- Use an equirectangular HDRI or photo panorama as the camera's sky while the node keeps lighting the scene:
  `light: {sky_image: {path: <panorama.hdr>, yaw: 90, sun: image}}`
- Put distant mountain ridges, mesas or hills under haze at the horizon, with Earth curvature (by the site's biome
  unless told otherwise): in the scene's `far.yaml`, `far: {terrain: {kind: ridged, height: 2600}}`
- Render rain or snow as a white-on-black mask per frame: `components/light/bl_weather.py`

## Features

- **Presets**: 11 times of day — dawn, sunrise, morning, noon, day, afternoon, evening, sunset, afterglow, dusk, night;
  8 sky weathers — clear, cumulus, haze, overcast, storm, rain, war_smoke, mist; styles `real` and `anime`
  (saturation, exposure, night tint, stars, bloom, light shafts).
- **Sky**: Blender's Multiple Scattering sky from the sun, from the moon at night with a night tint; sun disc, moon with
  its phase from the sun direction, stars sized to ~1.3 px of each camera — seen by camera rays only, while lamps light
  the scene (no double sun). Lamp colour and strength come from a calibration table by sun elevation
  (`core/light/far_sky_table.json`).
- **Haze by depth**: transmittance and in-scattering per layer; horizon colour by azimuth from a table of 16 sky renders
  (cached by plan hash); the sky under the geometry is composited in post.
- **Distant terrain**: a ray mesh from the edge of the site's mountain belt out to tens of kilometres with Earth
  curvature and horizon dip — real heights from the site (`far_heights.npz`) or ridges, mesas and hills by biome.
- **Clouds of the node**: decks as discs up to 120 km on their altitude, cumulus as clusters of sprite puffs (Wang 2004)
  with single scattering through a sphere and two-stream diffusion; cloud and smoke shadows only from a sun above the
  horizon; clouds drift with the wind frame by frame.
- **City, fires, flashes**: a distant city with windows lit by the time of day; smoke columns, flame cards, fire glow in
  haze and from below onto clouds; explosions and lightning with per-frame envelopes.
- **Night**: scotopic blue shift (Jensen et al. 2000), moon fill and fire glow fill for scenes with their own lamps,
  exposure by style; scene lamps and emission are rescaled to the node's exposure.
- **Post** (`far.py post`): lit geometry + camera sky → haze by depth → AgX as in Blender (its matrices and
  `AgX_Base_sRGB.cube`; looks such as punchy or medium_high_contrast) → bloom and light shafts through gaps (Mitchell
  2007, GPU Gems 3).
- **Sky image** (`sky_image`): an equirectangular 2:1 panorama for the camera while the node lights the scene; the sun
  is taken from the image, from the node or given as azimuth and elevation; yaw, exposure, drift.
- **Wind for the whole scene** (`core/light/wind.py`): Beaufort force or m/s at 10 m, gusts, turbulence, log profile by
  roughness, deterministic by seed — read by water, cloth and effects.
- **Rain and snow layer** (`bl_weather.py`): particles from an emitter that follows the camera (rain 0.05–1 drops/s/m²,
  9–20 m/s; snow 2–26 per m²), white on a black scene — a mask per frame.

## How it works

The description is the scene's `far.yaml` (key `far:`: time, weather, style, sun, moon, terrain, clouds, city, fires,
flashes, wind, sky_image, ev); without it the node defaults to the sun of `world.json`, and on a site to the weather and
distant terrain of its biome (temperate — cumulus and 1.6 km ridges, alpine — clear and 2.6 km ridges, desert — haze
and mesas…). Plan (Python, `steps.py:plan_light`): description + step keys → `far.plan_of` → `far_plan.json`; with
`shadows: off` the sun lamp casts no shadow. Build (Blender, `steps.py:build_light`): old lamps are removed; world,
lamps, distant terrain, city, clouds, fires and the AgX view come from `core/light/bl_far.py`; hook `light` lets others
react (light through foliage, live water). If the sky weather is not given, it comes from the `weather` step.

## Steps and keys

| Step | Reads → writes | Keys (defaults) |
|---|---|---|
| `light` | — (+ site biome, sky weather of `weather`) → `light`; gives `light_plan`, `exposure` | `time` (scene light or site biome), `weather` (sky weather of the `weather` step), `style`, `sky_image`, `shadows` (on) |

All keys with descriptions: [component.yaml](/components/light/component.yaml)

## Usage

```yaml
steps:
  - relief: {preset: fortress_hill, seed: 7}
  - camera: {anchor: fort.top, azimuth: 200, elevation: 12, distance: 160}
  - light: {time: night}                       # last before the outputs
  # - light: {time: dusk, weather: haze, style: anime}
  # - light: {time: day, sky_image: {path: <panorama.hdr>, yaw: 90, sun: image}}
  # - light: {time: noon, shadows: off}        # cheaper sketch of a forest scene
```

```sh
.venv/bin/python -m core.light.far resolve <scene folder> [description.yaml] [time= weather= style= sun_az= moon_az= moon_el=] [out=<plan.json>]
blender -b --factory-startup -P core/blender/run.py -- core/light/bl_far.py -- <plan.json> <folder> [size=64]   # sky cube, "one sun" check
.venv/bin/python -m core.light.far post <frame folder> …                       # lit.exr + sky.exr → lit.png
.venv/bin/python -m core.outputs.frames <scene folder> <out> all --light dusk [--weather rain] [--sky <panorama>[:yaw°]]
blender -b --factory-startup -P core/blender/run.py -- components/light/bl_weather.py -- <scene folder> <camera K> <rain|snow> <out folder>
blender -b --factory-startup -P core/blender/run.py -- components/light/bench/sky_aim.py -- <plan.json> <folder>
```

## Driving it from an AI agent

One recipe line, last before the outputs (a new light then rebuilds only the light and the frames):

```yaml
  - light: {time: dusk, weather: haze, style: anime, shadows: off}   # shadows: off — a cheaper draft
```

- Check without Blender: `.venv/bin/python -m core.manager.pipeline <recipe.yaml> --dry`. The five keys and their
  values are in `components/light/component.yaml`; sun azimuth, moon, terrain, city, fires, wind and the rest are the
  `far:` description in the scene's `far.yaml`.
- A run prints `__RESULT__{"pipeline", "out", "failures", …}` as its last line; the run folder holds `journal.md` and,
  with the `frames` output, a contact sheet `contact.jpg`. The numbers of a light alone, with no Blender:
  `.venv/bin/python -m core.light.far resolve <scene folder> time=dusk weather=haze` prints
  `__RESULT__{"plan", "sun", "ev", …}`.
- Request to keys: "a foggy dawn" → `time: dawn, weather: mist`; "an overcast afternoon" →
  `time: afternoon, weather: overcast`; "a burning town at night" → `time: night, weather: war_smoke`;
  "this photo as the sky" → `sky_image`.

## Files

| File | Role |
|---|---|
| `steps.py` | step entry: plan of the node, build into the `.blend`, `shadows` |
| `bl_weather.py` | rain or snow layer as a mask |
| `bench/sky_aim.py` | sky image probe: offset of the brightest pixel from the node's sun |
| `core/light/far.py` | description → plan; haze, terrain, clouds, city, fires, flashes, night; post and AgX |
| `core/light/bl_far.py` | Blender: world, lamps, distant terrain, clouds, city, fires, horizon table, frames |
| `core/light/wind.py`, `core/light/far_sky_table.json` | scene wind model; sky light by sun elevation |

## Assets

None — fully procedural. AgX tables are read from Blender's own colour management: `SK_OCIO_DIR`, otherwise the folder
next to `blender` on `PATH`. A sky image is any 2:1 equirectangular panorama you provide.

## Limitations

- The node's clouds are sprite puffs and decks; volumetric clouds are the separate `clouds` component and do not
  replace them yet.
- Cloud and smoke shadows come only from the sun; fires of the node cast no shadows.
- A new distant terrain costs ~15 s and ~1.2 GB once per plan hash; the first plan of a scene adds ~16 s of horizon renders.
- With a forest, the shadow of the sun lamp is the most expensive part of a frame (~25–30 %); use `shadows: off` for sketches.

## FAQ

**How do I set the time of day and a physically based sky in a Blender scene from Python?** Put `light: {time: sunset}`
in a recipe (11 times from `dawn` to `night`, 8 sky weathers). Without a recipe,
`.venv/bin/python -m core.light.far resolve <scene folder> time=sunset` computes the plan — sun, moon, sky, haze,
exposure — in Python and numpy, and Blender takes the numbers (`core/light/bl_far.py`).

**Can I use an HDRI or a panorama photo as the sky?** Yes: `sky_image: {path: <panorama>, yaw: 90, sun: image}` takes
any 2:1 equirectangular image (`.hdr`, `.exr`, `.jpg`, `.png`). The camera sees the picture, the light comes from the
node, and the sun is taken from the image (`sun: image`), from the node (`sun: node`) or set as `[azimuth, elevation]`.

**How do I make a night scene with moonlight in Blender?** Use `light: {time: night}`: the moon with its phase, stars, a
blue night shift and a moon fill for scenes with their own lamps. `style` sets the night exposure and tint: `anime` is
brighter, `real` is darker.

**How do I render rain or snow as a separate layer?** Run `bl_weather.py` in Blender: it writes a white-on-black mask
per frame, with a particle emitter that follows the camera.

```sh
blender -b --factory-startup -P core/blender/run.py -- components/light/bl_weather.py -- <scene folder> <camera K> <rain|snow> <out folder>
```

**Can I use it without the rest of sketch3d?** Yes, it also ships as its own repository, `sketch3d-blender-sky-light`, with the
core it needs (`core/light/`). The plan is computed in Python; frames are rendered in Blender 5.2 with the AgX view.
