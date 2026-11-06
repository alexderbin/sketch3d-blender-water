# Water — procedural ocean, lake, river and waterfall generator for Blender: FFT waves, foam, water colour

> Part of [sketch3d](https://github.com/alexderbin/sketch3d-blender) — AI-ready 3D automation for Blender: procedural scenes from YAML recipes. This repository is a standalone slice: the `water` component with the sketch3d core it runs on and the components it builds on (`camera`, `light`, `plants`, `site`). An AI agent can drive it by [AGENTS.md](AGENTS.md).

Water is a procedural ocean, sea, lake, pond, river and waterfall generator for Blender and Python: waves, foam, surf
and water colour come from the wind, the fetch, the depth and what is dissolved or suspended in the water. One water
model for every body of water: the differences are numbers of the place — wind, fetch, depth, matter in the water,
current — not separate scenes. A numpy wave engine (JONSWAP/TMA spectra, FFT bands, foam with memory) drives a
camera-projected mesh in Blender and is evaluated on the fly for each frame, so nothing is written to disk. The `water`
step refines the site layer: it reads the site's water level and the cameras, finds the water bodies, plans their sea
state and builds one water surface per camera.

**Use it to:**

- Make a realistic ocean or sea from a wind speed, from a light breeze to a gale, with waves, whitecaps and foam, built
  for each camera and each frame: `water: {wind: 12, optics: ocean_blue}`
- Colour a lake, pond or river by what is in the water — clear mountain lake, green pond, brown forest pond,
  glacier-fed lake, muddy river (11 presets): `water: {optics: glacial_lake}`
- Animate water for a video or a camera fly-around, with frames rendered in any order and nothing baked to disk
  (`components/water/bench/water_bay_orbit.yaml`): `camera: {orbit: 4, arc: 40}`, `water: {}`, `frames: {video: true}`
- Find the sea, lakes, ponds and rivers of a terrain and get their area, depth, fetch, wave height and period, river
  speed and Froude number (`water.json`): `.venv/bin/python -m components.water.bench.plan harbour_bay wind=18`
- Calculate sea-state numbers without Blender — wave height, peak period and length, whitecap share, slopes:
  `.venv/bin/python -m components.water.waves u10=12 fetch=2000 depth=50`
- Render surf on a beach, a mountain river, rapids, a waterfall (8 kinds), a storm by Beaufort force or a view from a
  ship's deck: `scene=beach` in `components/water/bench/stand.py` (also `river`, `rapids`, `fall`, `storm`, `deck`;
  command under Usage)

## Features

- **Spectrum from wind, fetch and depth**: JONSWAP with the Pierson–Moskowitz cap, TMA depth correction, a separate swell
  spectrum, directional spreading (Hasselmann, Horvath for swell), dispersion with capillarity
  ω² = (gk + σ/ρ·k³)·tanh(kh). FFT bands of different tile sizes without double counting, choppy crests, seamless loops.
- **Foam with memory** from the Jacobian: past injections decay and drift with the wind — a pure function of time, so
  frames render in any order; whitecap share checked against Monahan. Storm foam in three ages and marbled streaks of old
  foam stretched by surface currents and Langmuir rows (`marble.py`).
- **Storm by Beaufort force**: spindrift torn from crests from force 8, spray haze from force 9 (`bl_spray.py`); a rogue
  wave by phase focusing at a chosen point and moment; rain rings, caustics; `surface_at` for floating objects.
- **Water colour from optics** (`optics.py`): absorption and backscatter of pure water (Pope & Fry 1997), phytoplankton,
  dissolved organics and suspended sediment → remote-sensing reflectance (Gordon 1988, Lee 1998) → body colour and
  attenuation for shallows; checked against the Forel–Ule scale (Novoa et al. 2013). 11 presets: ocean, ocean_blue,
  coastal, north_sea, lagoon, mountain_lake, glacial_lake, pond_clear, pond_green, forest_pond, muddy_river.
- **Water bodies from the site** (`bodies.py`): connected bodies, area, depth, kind (sea, lake, pond, river), fetch
  upwind; rivers get slope, Manning velocity and Froude number (regime: glide, current, riffle, threshold).
- **Surf** (`shore.py`): refraction by an eikonal from the waterline, shoaling by energy flux, breaking at H = 0.78·d
  (McCowan), breaker type by the Iribarren number, bores with foam, swash run-up (Stockdon 2006), wave groups.
- **Rivers** (`flow.py`): current across the channel, flow-map advection in two phases offset by half a cycle, foam wakes
  behind rocks; white water — steps, hydraulic jump (Bélanger), standing waves λ = 2πv²/g; plunge pools.
- **Waterfalls** (`fall.py`, `falls.yaml`): one jet model — critical depth at the lip, free fall with air drag or sliding
  along the rock, break-up after L ≈ 1 + 4√q m, mist at the foot; 8 kinds: plunge, horsetail, fan, curtain, ribbon,
  segmented, punchbowl, tiered.
- **In the frame** (`bl_water.py`): a screen-projected grid to the horizon with Earth curvature, displacement per frame,
  per-pixel normals from the band images, sub-pixel slopes as roughness (Cox–Munk), Fresnel sky reflection, body colour,
  light through crests, lace foam. A water frame costs 0.2–0.3 s (~200 k vertices).

## How it works

Plan (Python): `bodies.bodies` finds the bodies on the site grid (`core.surface.ground.Ground`), the wind comes from the
scene (`core.light.wind`, Beaufort scale, gusts), `kinds.yaml` picks the optics preset by body kind and biome; the
sea state (Hs, peak period and length, whitecaps) and the Forel–Ule index of the water go to `water.json`. Build (Blender,
`bl_live.py`): one `water_<camera>` surface per scene camera at the site's water level (rivers downhill, lakes, sea to
the horizon); the site's own water mesh is hidden from the render. The engine numbers are stored on the objects
(`water_params`); the `light` hook (`api.light_changed`) rebuilds the engine when the frames output opens the scene,
advances time per frame and updates only the current camera's water before rendering.

## Steps and keys

| Step | Reads → writes | Keys (defaults) |
|---|---|---|
| `water` | `site`, cameras (+ scene wind from `light`) → `site`; gives `water_plan` | `wind` (scene wind; m/s at 10 m), `optics` (by body kind and biome, `kinds.yaml`) |

Plan only in the `sketch` profile (the site's water stays as a stand-in). All keys with descriptions:
[component.yaml](/components/water/component.yaml)

## Usage

```yaml
steps:
  - relief: {preset: harbour_bay, seed: 2}
  - camera: {view: bay, anchor: bay.mouth, azimuth: 200, elevation: 6, distance: 260, lens: 30, res: 1280x720}
  - water: {}                          # or {wind: 12, optics: north_sea}
  - light: {}
```

```sh
.venv/bin/python -m core.manager.pipeline components/water/bench/water_bay.yaml [--dry] [--until water]
.venv/bin/python -m components.water.bench.bench        # wave engine vs JONSWAP, Pierson–Moskowitz, Cox–Munk, Monahan
.venv/bin/python -m components.water.bench.plan [preset,…]  # the water step on site presets, no Blender
.venv/bin/python -m components.water.waves [u10=8] [fetch=2000] [depth=50] [swell_hs=1 swell_tp=10]
.venv/bin/python -m components.water.optics             # presets: colour, attenuation, Forel–Ule index
blender -b --factory-startup --python-exit-code 1 -P core/blender/run.py -- components/water/bench/stand.py -- out/dev/water/sea
```

`bench/water_bay_orbit.yaml` is the same bay with a four-camera orbit and video. `bench/stand.py` renders showcase
scenes — sea (default), lake, pond, beach, river, storm, rapids, fall, deck — chosen by a key=value argument
`scene=<name>`; the scene names are in the `SCENES` table at the end of the script and the options (wind, swell, t,
frames, fps, samples, width) in its docstring. Some scenes take their own: `force=` (Beaufort force) for storm, `type=`
(one of the 8 kinds of `falls.yaml`) for fall, `waves=` for deck.

## Driving it from an AI agent

An agent writes a recipe: a YAML file with the header of `components/water/bench/water_bay.yaml` (`pipeline`, `scene`,
`profile`, `build: stepwise`) and these lines in `steps:`; `water` goes after `relief` and `camera` (it reads the site
and the cameras):

```yaml
steps:
  - relief: {preset: harbour_bay, seed: 2}
  - camera: {view: bay, anchor: bay.mouth, azimuth: 200, elevation: 6, distance: 260, lens: 30, res: 1280x720}
  - water: {wind: 18, optics: north_sea}
  - light: {time: dusk, weather: storm}
outputs:
  - frames: {samples: 16, maps: false}
```

- Check without Blender: `.venv/bin/python -m core.manager.pipeline <recipe.yaml> --dry` rejects an unknown step or key,
  a value outside the passport (such as `optics: swamp`) or a wrong step order, and prints a line per step.
- All keys, defaults and descriptions: `components/water/component.yaml`; the optics presets: `PRESETS` in
  `components/water/optics.py`; the preset per body kind and biome: `components/water/kinds.yaml`.
- A run prints one `__RESULT__{…}` JSON line and writes the run folder `out/dev/runs/<pipeline>/<scene>/<profile>/`:
  `journal.md`, `plan.json` and `build.json` per step, `water.json` in the `water` folder (per body: kind, area, depth,
  fetch, wave height and period, water preset; for a river speed and Froude number), `contact.jpg` of the frames.
- Request to keys: "a stormy sea at dusk" → `water: {wind: 18, optics: north_sea}` (m/s at 10 m; force 8 is 17.2–20.7)
  and `light: {time: dusk, weather: storm}`; "a clear tropical lagoon" → `water: {wind: 4, optics: lagoon}`.

## Files

| File | Role |
|---|---|
| `waves.py`, `marble.py` | wave spectrum, FFT bands, foam with memory, rain rings, caustics; marbled storm foam |
| `optics.py`, `bodies.py`, `kinds.yaml` | water colour from matter; bodies, fetch, Manning; optics and gusts per body kind |
| `shore.py`, `flow.py` | surf; river current and white water |
| `fall.py`, `falls.yaml` | waterfall jet model and its 8 kinds |
| `bl_water.py`, `bl_live.py`, `bl_spray.py`, `bl_fall.py` | Blender: water surface, live per-camera water, spray, falls and mist |
| `steps.py`, `api.py`, `bench/` | step entry; `light` hook; engine bench, plan check, showcase, bay recipes |

## Assets

The step is fully procedural. The showcase scenes of `bench/stand.py` load 4K Poly Haven HDRIs (CC0, such as
`belfast_sunset_puresky_4k.hdr`; the list is in the script) from `library/sky/` — download them from polyhaven.com —
and scans fetched by the `site` component (rocks into `library/rocks/polyhaven/`, ground scans).

## Limitations

- In the pipeline step, surf and river currents are not yet driven from the site layer; rapids, waterfalls, the deck
  view, swell, the rogue wave, spindrift and spray haze are mechanisms with showcase scenes, not step keys.
- Matter in the water is one preset per body kind and biome; ray tracing is off in pipeline frames (shallows and
  reflections come from probes); water is not part of the control maps.

## FAQ

### How do I make a realistic ocean in Blender with Python?

Put `water: {wind: 12, optics: ocean_blue}` after `relief` and `camera` in a recipe and run
`.venv/bin/python -m core.manager.pipeline <recipe.yaml>`. A numpy wave engine (a JONSWAP spectrum from wind, fetch and
depth, FFT bands, foam with memory) drives a mesh projected from the camera to the horizon, and the colour comes from
what is in the water.

### Do I have to bake or cache the water simulation?

No. The water is evaluated on the fly for each frame (0.2–0.3 s, ~200 k vertices) and nothing is written to disk; foam
is a pure function of time, so frames can be rendered in any order.

### How do I make a river, rapids, a waterfall or surf on a beach?

With the showcase scenes `river`, `rapids`, `fall` and `beach` of `bench/stand.py` (they need the sky images and scans
listed under Assets); the `water` step does not drive them from the site layer yet. A waterfall:

```sh
blender -b --factory-startup --python-exit-code 1 -P core/blender/run.py -- components/water/bench/stand.py -- \
  out/dev/water/fall scene=fall type=plunge
```

### Can I use it without the rest of sketch3d?

Yes. It also ships as its own repository, `sketch3d-blender-water`, with the core it needs. The wave engine and the water colour
need no Blender: `components.water.api.sea_state(u10, fetch, depth)` returns wave height, period, length and whitecaps,
and `components.water.api.look(preset)` the colour, attenuation and Forel–Ule index; the surface and its material
(`bl_water.py`) need Blender.

### Does it need a GPU?

No. The wave engine is numpy on the CPU, and a water frame costs 0.2–0.3 s.

## Install

Blender 5.2 as `blender` on `PATH`, Python 3.13, `ffmpeg`.

```sh
git clone https://github.com/alexderbin/sketch3d-blender-water.git && cd sketch3d-blender-water
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m core.services.venv_root
.venv/bin/python -m core.light.far agx      # bake the AgX look tables once (runs Blender for ~10 s)
.venv/bin/python -m pytest tests
```

Commands above run from the repository root. How recipes, steps and layers work — [ARCHITECTURE.md](ARCHITECTURE.md);
how an AI agent runs, checks and reads the pipeline — [AGENTS.md](AGENTS.md).

## License

MIT — see [LICENSE](LICENSE).
