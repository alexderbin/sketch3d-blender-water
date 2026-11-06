# Site — procedural terrain generator for Blender: heightmaps, erosion, rivers, roads, mountain ranges, USGS lidar

Site is a procedural terrain generator for Blender and Python. It builds the ground every other component stands on: a
near plot on a 0.5 m heightmap with rivers, roads, flat pads, named anchors and self-checks, plus a mountain belt out to
the horizon. A plot is a tree of shape primitives (Génevaux et al. 2015) expanded from a catalog preset or your own plan
and detailed by erosion and rock strata; the belt is noise, a real USGS 3DEP lidar window, or a mosaic of such windows.
`relief` is the first step of a recipe: it reads nothing and writes the relief layer (`relief.npz`, `location.json`)
that neighbours read through `core.surface.ground.Ground`.

**Use it to:**

- Generate a terrain for a scene from a ready landform — fortress hill, alpine pass, river crossing, harbour bay, desert
  canyon, glacier — and get another place of the same idea from another seed (24 presets):
  `relief: {preset: fortress_hill, seed: 7}`
- Put a mountain range around a playable area that runs to the horizon, from noise (`ridged`, `fbm`, `mesa`) or from
  real mountain shapes, with the `far` key of `relief`: `far: {kind: ridged, h: 240}` or
  `far: {kind: mosaic, types: {granite: 1, strata: 0.5}}`
- Make a terrain from real elevation data (USGS 3DEP lidar, where it has data; download the windows first, see Assets):
  `far: {kind: real, window: teton, k: 0.7}` for the mountains, `base: real` for the plot
- Shape man-made earthworks on the heightmap — hillfort ramparts and ditches, a tell, terps, lynchets, hollow ways,
  quarries — aged by slope diffusion: `relief: {preset: hillfort_contour, seed: 3, works: multivallate}`
- Scatter boulders, cliff faces, outcrops and scree over the terrain, placed from its own data (photo-scans as
  instances): `rocks: {density: 1, crags: 1}`
- Export a heightmap and masks for a game engine or another 3D program, as 16-bit height PNGs, 8-bit masks and
  `export.json`: `.venv/bin/python -m components.site.terrain export <site folder> [out]`

## Features

- **24 presets** (`library/terrain/presets/`, table in `library/terrain/CATALOG.md`): fortress hill, alpine pass, river
  crossing, desert canyon, coastal cliffs, harbour bay, glacier, volcanic plateau, crater field, hillforts, tell, terps.
  Each has a biome, knob ranges and camera views; the seed gives another place of the same idea.
- **Archetypes with anchors**: `fort_hill`, `gorge`, `pass`, `crossing`, `cliff_edge`, `arena`, `crater_field`, `valley`
  expand into features (hill, ridge, cliff, river, lake, crater, bowl, terraces, rampart, moat, pad, road…) and publish
  anchors such as `fort.top`, `x.ford` or path points `road@0.3`.
- **Geomorphology rules**: droplet and thermal erosion only where relief in a ~100 m window exceeds 20–60 m (low hills
  stay smooth, Perron et al. 2009); free hills under 150 m get round tops and slopes ≤ 28°; strata on steep ground
  (cornices, buttresses, gullies); shores with a beach, a steep drop and a bed; roads routed by A* under a grade limit.
- **Man-made relief** (`terrain_works.py`): ramparts and ditches, tells, terps, lynchets, hollow ways, quarries, dikes —
  aged by linear slope diffusion (Culling 1960, Hanks 1984), calibrated on lidar profiles of real sites (`towns/measures/`).
- **Mountain belt** (~3 km, 4.7 m grid): noise (`ridged`, `fbm`, `mesa`); a real 6 × 6 km window with the plot set in
  its valley (`real`); or a `mosaic` of massifs from 14 windows by type (granite, strata, mesa, volcanic, hills). A 90 km
  window of the same place continues to ~30 km (`far_heights.npz`) for the distant terrain of `light`.
- **Find or make** (`terrain_fit.py`): on a real lidar base the archetype takes the thalweg, saddle, summit or bowl it
  finds and builds only what is missing.
- **Self-checks** in `location.json`: flat pads, road grade, rivers flowing downhill, seam with the belt, anchors present.
- **Rocks**: boulders, cliff faces, outcrops and scree placed from terrain data (one point per footprint cell, big before
  small), instanced from photo-scans; belt cliffs near the camera become rock walls.
- **Export**: 16-bit height PNGs, 8-bit masks (water depth, road, pad, cliff, slope, walkable) and `export.json`.

## How it works

Plan (Python, numpy only): `terrain.preset` resolves knobs → `terrain.build` evaluates the primitive tree (profile by
distance to a point, line or contour + add, smooth max, carve or flatten) and the belt → `terrain.save` writes
`relief.npz` (grids, water level, masks road/pad/cliff/pave) and `location.json` (anchors, paths, checks, biome);
`terrain_tex.bake` adds the ground texture, layer weights and moisture (`relief_w.npz`); `surface.json` lists the
biome's scans. Build (Blender, `bl_relief.py`): seamless ground mesh `terrain`, belt mesh `terrain_belt`, water,
landmarks, bridges; the ground shader comes from `core/surface/` (scans near the camera, a vista shader on the belt).
Buildings edit the relief and call `api.refresh`, `api.build_terrain`. Knowledge — YAML in `library/terrain/`.

## Steps and keys

| Step | Reads → writes | Keys (defaults) |
|---|---|---|
| `relief` | — → `site`; gives `biome` | `preset` (alpine_pass), `seed` (1), `ground` (scans; vista, baked), `far` (preset's belt), `size`, `res` (preset's grid); preset knobs within catalog ranges |
| `rocks` | `site` (+ building occupancy) → `site` | `density` (1), `belt` (0), `crags` (1), `seed` (11) |

Hook `frame_near`: grass tufts, pebbles, twigs and moss near the cameras in frames of a `.blend` scene.
All keys with descriptions: [component.yaml](/components/site/component.yaml)

## Usage

```yaml
steps:
  - relief: {preset: fortress_hill, seed: 7, h: 28}       # h is a knob of this preset
  # - relief: {preset: alpine_pass, seed: 3, far: {kind: real, window: teton, k: 0.7}}
  # - relief: {preset: alpine_pass, seed: 3, far: {kind: mosaic, types: {granite: 1, strata: 0.5}}}
  - rocks: {density: 1}
```

```sh
.venv/bin/python -m core.manager.pipeline core/manager/pipelines/castle_rock.yaml --until relief
.venv/bin/python -m components.site.terrain build <plan.yaml> [folder]     # no Blender: relief.npz, location.json, sheet.png
.venv/bin/python -m components.site.terrain preset <name> [seed] [scene] [knob=value …]
.venv/bin/python -m components.site.terrain export <site folder> [out]
.venv/bin/python -m components.site.bench.bench [name …] [--record]       # 12 reference sites vs bench/golden.json
```

## Driving it from an AI agent

An agent writes a recipe: a YAML file with the header of `core/manager/pipelines/castle_rock.yaml` (`pipeline`, `scene`,
`profile`, `build: stepwise`) and these lines in `steps:` (`camera`, `light` and `frames` only add a picture):

```yaml
steps:
  - relief: {preset: alpine_pass, seed: 3, snowline: 28}      # snowline is a knob of this preset
  - rocks: {density: 1.2}
  - camera: {anchor: pass.saddle, azimuth: 200, elevation: 15, distance: 220, res: 960x540}
  - light: {time: sunset}
outputs:
  - frames: {samples: 8}
```

- Check without Blender: `.venv/bin/python -m core.manager.pipeline <recipe.yaml> --dry` rejects an unknown step or key,
  a value outside the passport (such as a preset that is not in the catalog) or a wrong step order, and prints a line
  per step.
- All keys, defaults and descriptions: `components/site/component.yaml`; the knobs of each preset and their ranges:
  `library/terrain/CATALOG.md`.
- A run prints one `__RESULT__{…}` JSON line and writes the run folder `out/dev/runs/<pipeline>/<scene>/<profile>/`:
  `journal.md`, `plan.json` and `build.json` per step, and `contact.jpg`, a contact sheet of the frames.
- Request to keys: "a snowy mountain pass with a rocky skyline" → `relief: {preset: alpine_pass, h: 85, snowline: 22}`,
  `rocks: {density: 1.5}`.

## Files

| File | Role |
|---|---|
| `steps.py`, `api.py`, `refresh.py` | step entry; entry for neighbours; ground refresh after relief edits |
| `terrain.py`, `terrain_erosion.py`, `terrain_works.py` | primitives, archetypes, anchors, checks, export; erosion; earthworks |
| `terrain_ranges.py`, `terrain_library.py`, `terrain_fit.py` | real belt windows and mosaic; real bases; find or make |
| `terrain_tex.py`, `terrain_sheet.py` | ground texture, layer weights, belt maps; top-view sheet |
| `terrain_dress.py`, `terrain_rocks.py` | dressing by ecotope rules; rock placement |
| `bl_relief.py`, `bl_rocks.py`, `bl_crags.py`, `bl_terrain_clutter.py` | Blender: ground, belt, water; rocks; belt walls; clutter |

## Assets

Presets and noise belts are fully procedural. Downloads go to `library/` (not in git):

- Ground scans, ambientCG, CC0: `.venv/bin/python -m components.site.terrain_scans fetch`
- Clutter scans, Poly Haven, CC0: `.venv/bin/python -m components.site.terrain_clutter fetch`
- Rock scans, Poly Haven, CC0 (~150 MB):
  `.venv/bin/python -m components.site.terrain_clutter fetch --cfg library/terrain/rocks.yaml --out library/rocks/polyhaven`
- Belt windows, USGS 3DEP, public domain (14 × ~25 MB): `.venv/bin/python -m components.site.terrain_ranges catalog`
- Real bases, USGS 3DEP: `.venv/bin/python -m components.site.terrain_library build [type …] [--per 4]`

## Limitations

- The baked ground texture (`terrain_tex.py`) samples a ground-tile image set (the `ground_tiles` folder in `library/textures/`, files
  `<tile>_dif.jpg`, names in `BIOMES` of `terrain_tex.py`) that is not part of this repository. Without the tiles the
  bake paints flat tones with soft blotches by tile name: `relief_tex.png`, `relief_w.npz`, `relief_nrm.png` and
  `relief_far.png` (the scan shader, `rocks`, clutter and `grass` read them) are still written; the ground is plainer.
- Real windows exist only where USGS 3DEP has data; a real belt is not eroded.
- Plan time ~25–40 s per preset at 0.5 m (half of it the texture), growing with (size / res)².

## FAQ

### How do I generate a procedural terrain in Blender with Python?

Put `relief: {preset: fortress_hill, seed: 7}` in a recipe and run
`.venv/bin/python -m core.manager.pipeline <recipe.yaml>`: the heightmap is computed in numpy and the meshes are built
in Blender by `bl_relief.py`. Without Blender,
`.venv/bin/python -m components.site.terrain build <plan.yaml> [folder]` writes `relief.npz`, `location.json` and a
top-view `sheet.png`; the plan file can be as short as `relief: {preset: fortress_hill, seed: 7}`.

### Can I get the terrain as a heightmap PNG for a game engine or another program?

Yes. `.venv/bin/python -m components.site.terrain export <site folder> [out]` writes 16-bit height PNGs of the plot and
of the belt, 8-bit masks (water depth, road, pad, cliff, slope, walkable) and `export.json` with the origin, step,
height range, anchors and paths.

### Can I build a terrain from real elevation data?

Yes, where USGS 3DEP has data. Download belt windows with `.venv/bin/python -m components.site.terrain_ranges catalog`
and use `far: {kind: real, window: <name>}` or `kind: mosaic` in `relief`; real bases for the plot come from
`.venv/bin/python -m components.site.terrain_library build` and are switched on with `base: real`.

### Can I use it without the rest of sketch3d?

Yes. It also ships as its own repository, `sketch3d-blender-terrain`, with the core it needs. The plan side is plain Python
(numpy, SciPy, OpenCV, Pillow, PyYAML); Blender is needed only to build the meshes.

### Does it need a GPU?

No. The terrain plan runs on the CPU, about 25–40 s per preset at 0.5 m, half of it the ground texture; the meshes are
built by Blender in background mode (`blender -b`).
