# Camera — camera placement and camera moves for Blender in Python: orbit, dolly, crane, arc, hero framing, coverage

Camera is a Python component for Blender that places and animates scene cameras automatically: orbit and fly-around
cameras, planned dolly, crane, arc, pan, truck and zoom moves that avoid walls, buildings, trees and terrain, and hero
framing by shot type. It places the scene's cameras and writes the `cameras` layer (`cameras.json`). The
`camera` step aims at a target — a point, an anchor of the site or the buildings, a building's centre — adds orbits and
plans camera moves *by the place*: a recipe finds frames A and B where the site allows them, lays the path on a rig,
times it by a format profile and checks it with numbers — NumPy on a 2.5D field built from the scene layers, no Blender.
Command tools add moves from a base camera, fly-throughs, coverage analysis, hero framing by intent and camera matching
from a picture. Cameras go before light: re-lighting never moves them.

**Use it to:**

- Fly a camera around a castle, a building or any point of the terrain, with a rise and a push-in —
  `camera: {orbit: 12, arc: 90, rise: 8, close: 0.3}`.
- Plan a dolly-in, pull-back, crane, truck, pan, arc or zoom that the terrain and buildings allow, timed like a trailer,
  a fiction scene or a documentary, or frame one comic panel as a move of zero length —
  `camera: {move: {recipe: arc, subject: {anchor: castle.gate}, format: trailer}}`, `api.panel(spec, scene)`.
- Frame a character as a wide, medium, close-up, heroic, overhead, two-shot, over-the-shoulder, from-cover or
  point-of-view shot, rejecting views blocked by walls (needs the hero's joints in `figkit.json`; the Blender command
  is in Usage) — `cam_intent.py <scene> <hero> over_shoulder pair=<hero2>`.
- Add a push-in, orbit, crane, truck, pan or zoom to a camera you already have, with the path repaired around
  buildings — `python -m components.camera.cam_move <scene> <K> dolly=0.25 12` — or fly it along waypoints on a smooth
  spline — `python -m components.camera.cam_path <scene> path.json 48`.
- Measure how much of a site or an object the cameras show and add the missing views, as a Blender command like
  `cam_intent.py` — `components/camera/bl_cover.py <scene> add=3 write=1`.
- Fit a Blender camera to a reference picture by its horizon and a person of known height —
  `components.camera.cam_from_picture <scene> <hero> <picture> --horizon <y> --feet <u,v> --head <u,v>`.

## Features

- **Targets**: `target` x, y → `anchor` → building centre (`castle`, `town`, `temple`, `edifice`, `classic` `.centre`) →
  a site `*.top` anchor → site centre; the camera stays at least 2 m above the terrain. **Orbits**: N cameras
  `orbit_NN` along an `arc` with a rise and a push-in. **Pinned cameras** approved on `sketch`
  (`core.services.stage pin`) are rebuilt exactly on `preview` and `final`.
- **Frame space** (toric, after Lino & Christie 2015: subject size, yaw, pitch, screen position ↔ camera) over a **2.5D
  site field** from terrain, masonry, occupancy and tree crowns that also tells *what is behind the subject* (sky, far,
  ground, wall, tower, building, tree) — a move can be asked for by its backgrounds A and B.
- **Recipes**: static, pan, push, pull, truck, arc, crane, follow, zoom. An orbit is an arc, not a circle — its span
  comes from the format (e.g. 20–60° for fiction); 360° only with `sweep`.
- **Rigs with limits**: tripod, dolly on a level rail (refuses a ground drop over 0.6 m), crane (boom 9 m, telescope
  15 m), steadicam and handheld (0.6–2.0 m above the smoothed ground), drone (3 m over the obstacle envelope), virtual;
  never closer than 0.3 m to an obstacle; handheld shake replayed from recordings of a real camera.
- **Timing**: curves `linear`, `ease`, `smooth`, `settle`, `accel`, a hold before the cut; screen-edge speed (% of the
  frame width per second) is measured and capped at the format's p90 — a too fast move is shortened.
- **Format profiles** (`data/profiles.yaml`): `trailer`, `fiction`, `lore`, `history_short`, `documentary`, `comic`;
  camera roles `companion`, `witness`, `spy`, `god`, `self`, `threat`.
- **Checks**: aim visible in ≥ 95 % of frames, clearance ≥ 0.25 m, jerk, "the move changes nothing"; cuts (30° and
  20 % rules, axis of action) via `api.cut_check`.
- **Command tools**: moves from a base camera with collision-free path repair (a clearance to buildings and the hero
  kept in view); a Catmull–Rom fly-through; coverage — the share of surfaces seen by the cameras with occlusion rays and
  greedy extra views; a camera from a reference picture by its horizon and a hero of known height; hero framing by intent
  (`wide`, `medium`, `closeup`, `heroic`, `overhead`, `two_shot`, `over_shoulder`, `from_cover`, `pov`): candidates
  around the hero, occlusion rays, the framing solver (~50× faster wrappers), the 180° line for pairs.

## How it works

Plan (`steps.py:plan_camera`, Python): target, `cam_<view>`, orbit cameras and moves (`move.shot` → `move_NNN` records
with `frame` and `fps`) → the run scene's `cameras.json`. Other writers' records stay, and the step's own records keep
their position, so frame folders `cam<K>` keep their numbers between profiles. Build (Blender): camera objects in the
step's collection, tagged `sk_id` and `sk_kind: camera`; Constructions and Plants read them for camera-driven detail.

## Steps and keys

| Step | Reads → writes | Keys (defaults) |
|---|---|---|
| `camera` | — (optional: site, buildings) → cameras | `view` (wide), `azimuth` (200°), `elevation` (12°), `distance` (160 m), `lens` (35 mm), `res` (960x540), `orbit` (0), `arc` (360°), `rise` (0°), `close` (0), `target`, `anchor`, `above` (4 m at an anchor, 8 m at a building), `move` |
| `camera` command entry (`cam_intent.py`) | figures → cameras | `intent` (medium), `hero`, `fmt`, `pair`, `az`, `el`, `lens`, `x`, `hf`, `name` |

`move` (a dict or a list): `recipe`, `subject` (`{anchor | point, h}`; default — the target), frame values (`size`,
`yaw`, `pitch`, …), `rig`, `format` (fiction), `role`, `sec`, `fps` (24), recipe knobs (`bg_a`, `bg_b`, `sweep`, …).
All keys with descriptions: [component.yaml](/components/camera/component.yaml)

## Usage

```yaml
steps:
  - relief: {preset: fortress_hill, seed: 7}
  - building: {type: castle, culture: german, scale: medium, seed: 7}
  - camera: {view: gate, azimuth: 200, elevation: 10, distance: 170, orbit: 8, res: 640x360}
  # an arc around the gate, trailer timing, 3 s at 12 fps → move_000 … move_035
  # - camera: {view: gate, distance: 170, move: {recipe: arc, subject: {anchor: castle.gate}, format: trailer, sec: 3, fps: 12}}
  - light: {time: day}
```

```sh
.venv/bin/python -m core.services.stage pin <run> --names cam_,orbit_       # keep approved cameras on later profiles
.venv/bin/python -m components.camera.bench.check                          # api, plans, pins, tags, moves; no Blender
# tools over a scene folder with cameras.json (e.g. <run>/scene)
.venv/bin/python -m components.camera.cam_move <scene folder> <K> orbit=10 12     # or dolly=0.25 | crane=2 | truck= | pan= | zoom=
.venv/bin/python -m components.camera.cam_path <scene folder> path.json 48 --name fly   # [{"pos": […], "look": […], "lens": 28}, …]
blender -b --factory-startup -P core/blender/run.py -- components/camera/cam_intent.py -- <scene folder> <hero> medium fmt=9:16
```

## Driving it from an AI agent

An agent writes one `camera` line in a recipe and lets the step choose the positions from the place:

```yaml
steps:
  - relief: {preset: fortress_hill, seed: 7}
  - building: {type: castle, culture: german, scale: medium, seed: 7}
  - camera: {view: gate, distance: 170, move: {recipe: arc, subject: {anchor: castle.gate}, bg_b: sky, format: trailer, sec: 3, fps: 12}}
  - light: {time: day}
outputs:
  - frames: {cameras: all}
```

- Check without Blender: `.venv/bin/python -m core.manager.pipeline <recipe.yaml> --dry` (step order, layers, step
  keys); try a move on `--profile sketch` first. All keys with defaults and descriptions:
  `components/camera/component.yaml`.
- A run prints one `__RESULT__{…}` JSON line (`out` is the run folder) and leaves `journal.md` there, the frames with
  `contact.jpg` in `NN_frames/`, and the move's `plan` and `checks` in `NN_camera/plan.json`; a bad move (blocked aim,
  clearance, speed over the format ceiling) is printed on the step line as `skipped: move: …`.
- Request → keys: "a slow orbit of the gate, ending on the sky" → inside `move`: `recipe: arc`,
  `subject: {anchor: castle.gate}`, `bg_b: sky`, `format: documentary`, `sec: 8`; "orbit 90°, rising" → `orbit: 12`,
  `arc: 90`, `rise: 8`; "descend from a wide view to the gate" → `recipe: crane`, `dir: down`.
- Freeze approved cameras: `.venv/bin/python -m core.services.stage pin <run> --names cam_,orbit_,move_`.

## Files

| File | Role |
|---|---|
| `steps.py`, `api.py` | the step; public entry (`shot`, `panel` — a comic panel, `move`, `look_at`, `frame_*`, `cut_check`, …) |
| `move/` | moves by the place: `frame.py`, `field.py`, `recipes.py`, `rig.py`, `timing.py`, `checks.py` |
| `data/` | format profiles and roles; handheld recordings |
| `cam_intent.py`, `framing_fast.py` | hero framing by intent; faster solver wrappers |
| `cam_move.py`, `bl_cam_path.py`, `cam_path.py` | moves from a camera with path repair; spline fly-through |
| `bl_cover.py`, `cam_from_picture.py`, `bench/` | coverage; camera from a picture; checks (`check.py`) |

## Assets

The step and the moves need nothing. `data/shake.json` holds handheld camera recordings (CC0). Hero framing imports the
`armature_core` module, not included: place it so that `components/camera/ext/armature/armature_core/` exists.

## Limitations

- Hero framing, the camera from a picture and hero-centred moves need figures (`figkit.json`) from an actors step that
  is not part of this release; without `armature_core` the solver check of `bench/check.py` fails.
- Command tools work on a scene folder; collision checks, coverage and intent framing also need its geometry
  (`scene.obj`, `world.json`), not the `.blend` of a stepwise run.
- The field is 2.5D (gate openings cut in): no overhangs. `follow` needs a hero track (`api.shot` only).
- Several profile numbers are estimates still to be checked against measured footage.

## FAQ

**How do I make a camera orbit around a building or a point in Blender with Python?** Add `orbit: N` to the `camera`
step: it writes cameras `orbit_00…` around the target (a building's centre, `anchor: <name>` or `target: "x,y"`); `arc`
sets the angle (360 is a full circle), `rise` tilts up, `close` pushes in, and `frames: {video: true}` renders the
fly-around. For a camera that already exists, this adds 12 cameras along a 10° orbit:

```sh
.venv/bin/python -m components.camera.cam_move <scene folder> <K> orbit=10 12
```

**How do I keep a Blender camera from clipping through walls or the ground during a move?** Plan the move with `move:`
in the `camera` step: it searches positions on a 2.5D field of terrain, masonry, buildings and tree crowns, keeps rigs
at least 0.3 m from obstacles and prints a blocked aim (visible in under 95 % of frames) or a clearance under 0.25 m as
`skipped: move: …`. For an existing camera, `cam_move` repairs the path (a 0.3 m gap to buildings, the hero kept in
view) when the folder has `scene.obj`; `detour=0` turns the repair off.

**Can it check a cut in code — the 30° and 20 % rules or the axis of action?** Yes: `components.camera.api.cut_check`
takes the last frame of shot A and the first of shot B (`{pos, look, lens}`) with the subject and returns numbers with
a list of what is wrong — a jump, a camera-height change, a crossed axis. `api.plan_check(shots, fmt)` checks a whole
episode plan against a format profile.

**Can I use it without the rest of sketch3d?** Yes. Camera also ships as its own repository, `sketch3d-blender-camera`, with the
core it needs. `components.camera.api` (`look_at`, `move`, `shot`, `panel`, `frame_*`, `cut_check`) is NumPy and runs in
plain Python or in Blender; `cam_move` and `cam_path` need only a scene folder with `cameras.json`.

**Does it need Blender or a GPU?** Planning — targets, orbits, moves and their checks — is Python and NumPy: no
Blender, no GPU. Headless Blender is needed to create the camera objects (`build_camera`), for hero framing, path repair
and coverage.
