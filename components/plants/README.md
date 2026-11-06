# Plants — procedural tree, forest and meadow generator for Blender: 33 species, forests simulated on terrain

Plants is a procedural vegetation generator for Blender and Python: it grows trees, shrubs, palms, grasses, flowers and
ferns, and simulates whole forests on a terrain. Everything that grows is one model: axes with branching orders and
organs on the axes, set per species in `species.yaml` (33 species). Woody axes are built by the `m_tree` engine; a
forest is simulated from site data (suitability, growth, shading, self-thinning) instead of being scattered, and the
herb layer is instanced only where the cameras see it. The steps read the site layer (and building occupancy, cameras,
ground cover when present) and write objects into the layout section plus the canopy layer `canopy.json`, which `grass`
reads for shade and litter.

**Use it to:**

- Generate a forest on a terrain, with species chosen by moisture, slope and biome, then grown, shaded and self-thinned
  in steps instead of scattered: `forest: {density: 1.0, seed: 1}`
- Dial a forest from open woodland to a closed canopy, limit it to chosen species and set it to autumn or winter:
  `forest: {density: 0.3, kinds: "birch,hazel", season: autumn}`
- Clear the trees out of the camera's way once the shot is set: `forest_clear: {margin: 2, front: 12}`
- Scatter grass, flowers and ferns only where the cameras see them, thinning with distance and capped in count:
  `meadow: {density: 40, to: 70, budget: 40000}`
- Make a single tree, shrub, palm, flower or fern of a species at an age and state, such as an old broken oak:
  `.venv/bin/python -m components.plants.plants sheet <out.json> oak:old:broken`, then
  `blender -b -P core/blender/run.py -- components/plants/bl_plant.py -- <out.json> <out.blend>`
- Preview a forest quickly with crown stand-ins instead of full trees: `forest: {look: stand_in}`

## Features

- **One model for every habit**: 13 trees, a palm, 4 shrubs, 5 grasses, 8 flowers, a fern, a herb. Per branching order:
  density, phyllotaxis, splits, angles, length, breaks, up-attraction, gravity × stiffness; crown envelopes (cone,
  sphere, flame…), weeping and umbrella forms.
- **Age and state**: seedling, young, adult, old; winter, dead, broken, autumn. **Roots** as a second tree growing down,
  seated on the terrain (flat, heart, tap, buttress, stilt); **lean** on slopes and edges, crowns shifted towards light.
- **Foliage**: cards on the outer and upper crown, leaf images drawn procedurally (`leaves.py`: toothed, oak, maple,
  willow, needles, scales, fronds, blades, petals); leaf colour from pigments (PROSPECT-D, Féret et al. 2017; CIE 1931,
  D65) for reflection and transmission; light through foliage in Eevee with offset shadow twins (`bl_veg_light.py`).
- **Forest as a simulation** (`forest.py`): suitability per species (moisture, slope, landforms, water, roads, snowline,
  biome) → glades → sowing → 10 × 10-year steps of growth on Chapman–Richards curves, shade from taller neighbours by
  shade tolerance, self-thinning, seeding from adults → Kraft classes and layers → shrubs at the forest edge.
  Density 0.3 ≈ 50 stems/ha (open woodland), 1.0 ≈ 230/ha, 1.6 — closed canopy; 0.3–1.1 s per plot.
- **Detail by pixels**: tiers by tree height in the camera frames — unique trees with roots near the camera (≤ 22 m, up
  to 14), instanced variants beyond, hulls of a few hundred triangles far away; sub-pixel twigs removed, cards trimmed
  to the leaf outline. `forest_clear` re-tiers once the cameras exist and clears crowns at the lens.
- **Meadow by the cameras** (`meadow.py`): tufts only inside camera frustums, density ρ₀·min(1, (near/d)^1.6) by 3D
  distance to the eye, cameras merged by the most demanding one, a budget cap; with ground cover the group and species
  come from `cover.npz` (flowers in clumps, grass variants by dryness).
- **Sketch stand-ins** (`look: stand_in`): crown solids measured from real trees of each species (`stand_in.json`).
- **Hooks**: `maps_before` / `maps_after` swap plants for leaf cut-out proxies in control maps (`bl_maps.py`); `light`
  adds light through foliage after the scene light.

## How it works

Plan (Python): `plants.resolve` turns species + age + state + seed into a JSON spec; `forest.plan` writes the stand
(`forest.json`: species, age, Kraft class, lean, crown shift) and the canopy layer; `meadow.py` picks tuft points and
tiers. Build (Blender): `bl_plant.py` builds one plant (woody axes by `m_tree`, roots, herb ribbons, organ cards);
`bl_forest.py` and `bl_meadow.py` instance variants. Knowledge — `species.yaml`: sizes, biomes, moisture, shade
tolerance, branching orders, foliage, roots, bark, pigments, ages and states.

## Steps and keys

| Step | Reads → writes | Keys (defaults) |
|---|---|---|
| `forest` | `site` (+ occupancy, cameras) → `layout`, `canopy.json` | `density` (1.0), `seed` (1), `area` (near grid), `kinds` (by biome), `unique` (14), `unique_to` (22 m), `season` (summer; winter, autumn), `look` (full; stand_in on sketch) |
| `forest_clear` | cameras (+ layout) → `layout`, `canopy.json` | `margin` (2 m), `front` (12 m), `look` (full) |
| `meadow` | `site` (+ canopy, cameras) → `layout` | `density` (40 tufts/m²), `to` (70 m), `variants` (3), `seed` (1), `season`, `budget` (40000), `cards` (a card set), `card_px` (60), `herb_px` (20), `card_overlap` (6); off on sketch |
| `plants` (legacy) | `site` (+ occupancy, cameras) → `layout` | `trees` (8), `boulders` (10), `seed` (1), `species`, `boulder_kind`, `size`, `from`, `to`, `slope`, `gap`, `boulder_gap` |
| `trees`, `boulders` (legacy commands) | `site` → `layout` | `species` (oak), `seed` (1); `kind` (rock) |

All keys with descriptions: [component.yaml](/components/plants/component.yaml)

## Usage

```yaml
steps:
  - relief: {preset: river_crossing, seed: 1}
  - forest: {density: 1.0, seed: 1}
  - cover: {seed: 1}
  - camera: {anchor: x.ford, azimuth: 205, elevation: 25, distance: 120, res: 960x540}
  - forest_clear
  - meadow: {density: 40, to: 70}
  - light: {time: day}
```

```sh
.venv/bin/python -m core.manager.pipeline core/manager/pipelines/forest_meadow.yaml [--until forest] [--profile sketch]
.venv/bin/python -m components.plants.plants list                   # species by biome
.venv/bin/python -m components.plants.plants sheet <out.json> [species[:age[:state]] …]
.venv/bin/python -m components.plants.bench.veg_bench [--record]    # specs, build, forest, meadow, optics vs veg_golden.json
.venv/bin/python -m components.plants.bench.stand_in_bench [--measure]
```

The forest part of `veg_bench` reads a catalog site on a real base (fortress hill, seed 1) prepared by
`components.site.terrain_catalog`.

## Driving it from an AI agent

An agent writes a recipe: a YAML file with the header of `core/manager/pipelines/forest_meadow.yaml` (`pipeline`,
`scene`, `profile`, `build: stepwise`) and these lines in `steps:`; this component's are `forest`, `forest_clear` and
`meadow` (`plants`, `trees`, `boulders` are legacy):

```yaml
steps:
  - relief: {preset: river_crossing, seed: 1}
  - forest: {density: 0.4, kinds: "birch,hazel", season: autumn, seed: 4}
  - cover: {seed: 1}
  - camera: {anchor: x.ford, azimuth: 205, elevation: 25, distance: 120, res: 960x540}
  - forest_clear
  - meadow: {density: 40, to: 70}
  - light: {time: afternoon}
outputs:
  - frames: {samples: 8}
```

- Check without Blender: `.venv/bin/python -m core.manager.pipeline <recipe.yaml> --dry` rejects an unknown step or key,
  a value outside the passport (such as `season: spring`) or a wrong step order, and prints a line per step.
- All keys, defaults and descriptions: `components/plants/component.yaml`; species, sizes, biomes, moisture windows and
  shade tolerance: `components/plants/species.yaml`.
- A run prints one `__RESULT__{…}` JSON line and writes the run folder `out/dev/runs/<pipeline>/<scene>/<profile>/`:
  `journal.md`, `plan.json` and `build.json` per step (`plan.json` of `forest` has `trees_n`, `stems_per_ha`, `closure`,
  `by_species`), `forest.json` and `canopy.json` in `scene/`, and `contact.jpg`, a contact sheet of the frames.
- Request to keys: "a sparse birch wood in late autumn" →
  `forest: {density: 0.4, kinds: "birch,hazel", season: autumn}`.

## Files

| File | Role |
|---|---|
| `plants.py`, `leaves.py`, `leaf_optics.py` | species → spec; leaf images; leaf colours |
| `forest.py`, `meadow.py`, `stand_in.py` | forest simulation; meadow points; stand-in silhouettes |
| `bl_plant.py`, `bl_forest.py`, `bl_meadow.py`, `bl_cards.py` | Blender: plant, forest, meadow, grass cards |
| `bl_veg_light.py`, `bl_maps.py`, `bl_stand_in.py` | light through foliage; map proxies; stand-in forest |
| `bl_tree.py`, `bl_boulder.py`, `steps.py`, `api.py` | legacy trees and boulders; step entry; hooks |

## Assets

- Bark scans, Poly Haven, CC0 (14 scans, ~40 MB): `.venv/bin/python -m components.plants.bark_scans fetch`; without them
  bark is procedural.
- Not included, expected in `components/plants/ext/`: `mtree/` — the tree engine package with its compiled `m_tree`
  core for Python 3.13 under `mtree/site/`; `cie/` — CIE 1931 2° colour-matching functions and illuminant D65
  (`CIE_xyz_1931_2deg.csv`, `CIE_std_illum_D65.csv`); `tree_skeletons/` — the tree skeleton package of the legacy steps.
  Leaf optics need `pip install prosail` (PROSPECT-D).
- Legacy boulders use Blender's A.N.T. Landscape extension.

## Limitations

- No installer for `ext/` yet. Full forests and the meadow need `m_tree` (stand-ins do not); the plans of `forest`
  and `meadow` compute leaf optics and need `prosail` and the CIE tables.
- The cheap far tier of the meadow uses a card set in `library/plants/cards/<set>/` (`.glb` + `cards.yaml`); none is
  included, so the meadow draws tufts up to `to` and the ground shader takes over beyond.
- The tree engine gives similar trees for consecutive seeds; take spread-out seeds for variants.

## FAQ

### How do I generate a procedural forest in Blender with Python?

Add `forest: {density: 1.0, seed: 1}` after `relief` in a recipe and run it with
`.venv/bin/python -m core.manager.pipeline <recipe.yaml>`. The forest is simulated first (species suitability,
10 × 10-year growth steps, shade, self-thinning, forest edge) and then built in Blender. Without Blender,
`.venv/bin/python -m components.plants.forest <site folder> [density] [seed]` writes `forest.json` and a top-view
`forest_map.png` and prints the stand statistics as a `__RESULT__` line; it needs `prosail` and the CIE tables (see
Limitations).

### How many trees can it handle, and how does it keep the scene light?

Detail follows the pixels: up to `unique` (14) trees within `unique_to` (22 m) of a camera get their own geometry with
roots, farther trees are instanced variants, and far trees are hulls of a few hundred triangles. The meadow is capped by
`budget` (40000 tufts), and `look: stand_in` replaces trees with crown solids for drafts.

### Which tree species are included?

33 species: 13 trees (oak, beech, birch, poplar, willow, spruce, pine, fir, cypress, acacia, kapok, alien_tree, snag),
a palm, 4 shrubs, 5 grasses, 8 flowers, a fern and a herb. `.venv/bin/python -m components.plants.plants list` prints
them by biome; the data is in `species.yaml`.

### Can I use it without the rest of sketch3d?

Yes. It also ships as its own repository, `sketch3d-blender-plants`, with the core it needs. It reads a site folder
(`relief.npz`, `location.json`), such as `sketch3d-blender-terrain` writes. The tree engine `m_tree` and the CIE tables (see
Assets) are not included, and leaf optics need `pip install prosail`.

### Does it need a GPU?

No. The forest simulation is numpy and SciPy on the CPU, 0.3–1.1 s per plot; the geometry is built by Blender in
background mode.
