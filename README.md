# Foxhole Map Exporter

Exporting [Foxhole](https://store.steampowered.com/app/505460/Foxhole/) maps and full pipeline for map mod creation. Updated for U67.

Every stage leaves something usable behind, so you can stop wherever your
project does:

| You want | Stop at | You get |
|---|---|---|
| Raw game data | [Step 1](#step-1---export-game-files) | Per-region JSON of every placed object and spline, meshes as `.pskx`/`.psk`, 16-bit heightmaps, terrain weightmaps |
| 3D scenes | [Steps 2-3](#step-2---generate-blender-scenes) | Each region (or the whole map) assembled in Blender: terrain, water, structures, roads, ready for renders, flythroughs or other 3D work |
| Map imagery | [Steps 4-5](#step-4---render-region-bakes) | World-sized PNG layers on one shared canvas: terrain, heightmaps, contours, roads, beaches, structures, ranges, alerts and more |
| A map mod | [Step 6](#step-6---toolset) | A `.pak` that replaces the in-game map with an image you composed from those layers, or anything else drawn on the same canvas |

The finished layers are published with every
[release](https://github.com/Tsekho/fh_map_exporter/releases/latest).

### Credits

The amazing idea of thresholded water depth coloring was adopted from
[Knight of Science's fork](https://github.com/Knight-of-Science/fh_map_exporter).

## Making a Map Mod

1. Download the layer zips from the
   [latest release](https://github.com/Tsekho/fh_map_exporter/releases/latest)
   and stack the layers you want in any image editor. They all share one
   20528x12704 canvas, so they line up as-is.
2. Export the result as a PNG.
3. Build the mod:

   ```bash
   pip install numpy opencv-python etcpak texture2ddecoder
   python 6_toolset.py mod MyMap.png -n MyMap
   ```

4. Copy `War-WindowsNoEditor_MyMap.pak` into `Foxhole/War/Content/Paks/`,
   next to the game's own `War-WindowsNoEditor.pak`.

Or skip the image editor and write the stack down as a
[recipe](#recipes): `recipe.json` rebuilds the released mod, so start by
editing that.

```bash
python 6_toolset.py recipe recipe.json      # -> War-WindowsNoEditor_CustomMapMod.pak
```

See [Step 6](#step-6---toolset) for the other things the toolset does.

## Pipeline

```text
0_make_release.py      ->  builds Exporter.exe from C# source
1_export.py            ->  Exporter.exe reads .pak → export/_json/, _meshes/, _heightmap/, _layers/
                           and records per-region JSON hashes → export/hashes.txt
2_blend_all.py         ->  full-map Blender scenes → export/blend/<MapName>.blend
3_blend_spills.py      ->  per-region .blend with neighbor spill → export/blend_spill/<Region>.blend
4_render_spills.py     ->  top-down per-region bakes → export/{ao,heightmap_landscape,heightmap_water,
                           roads,beaches,bridges_aim,id/<cat>,split_layers/<layer>,svg_layers/<layer>}/<Region>.png
5_finalize_exports.py  ->  stitches bakes into world PNGs and assembles final composites
                           → export/_final/{technical,assembly,id,split_layers,svg_layers}/
6_toolset.py           ->  world PNG <-> region PNGs <-> map mod .pak; layer recipe -> map mod .pak
```

## Requirements

Toolset only (step 6):

- Python 3.10 or newer (3.14 included: the toolset doesn't need `bpy`)
- **numpy**, **opencv-python**, **etcpak** (BC7 encoding), **texture2ddecoder**
  (BC7 decoding, for `unpack`). All four ship prebuilt wheels, so there is
  nothing to compile.

Full pipeline, in addition:

- Foxhole
- Python 3.10-3.13 (no `bpy` on 3.14)
- **bpy** (steps 2-4), **cairosvg** (step 4, `-svg`)
- Blender 5
- [.NET 10 SDK](https://dotnet.microsoft.com/download) (step 0 only)
- Cairo (the easiest way to install it - [GTK for Windows Runtime Environment](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer))

```bash
pip install -r requirements.txt
```

Clone with submodules (CUE4Parse is required for step 0):

```bash
git clone --recurse-submodules https://github.com/Tsekho/fh_map_exporter.git
```

## Usage

### Interactive Prompts

Steps 2-5 without a target argument open an arrow-key picker with
everything ticked: up/down to move, space to tick one on or off, `a`/`n`
for all/none, enter to run, esc to cancel. Enter straight away does what
`-a` does. Selected items get a progress bar each (one per worker in a
parallel run); `-v` shows the full log instead of just warnings. Step 6
without arguments opens a menu of its actions instead.

Piped and non-TTY runs fall back to a numbered prompt and plain output.
`FH_NO_TUI=1` (or `NO_COLOR`) forces that fallback in a terminal.

### Step 0 - Build the Exporter

Compiles `Exporter/` and outputs `Exporter.exe` at the repo root. A pre-built binary is included.

```bash
python 0_make_release.py
```

### Exporter.exe - Standalone Usage

`Exporter.exe` is a self-contained win-x64 binary.

```text
Exporter.exe -i <pak_path> -o <export_path> [-t] [-a <asset_path>]
```

| Argument | Description |
|---|---|
| `-i <pak_path>` | Path to the `.pak` file **or** its containing directory |
| `-o <export_path>` | Output folder (`_json/`, `_meshes/`, ... are created inside) |
| `-t`, `--texture` | Terrain layers and heightmaps |
| `-a <asset_path>` | Single asset, e.g. `War/Content/Maps/HomeRegionC.umap` (`.umap` optional). Omit to export all maps under `War/Content/Maps`. |

Proxy maps (`Proxy_*`, the engine's landscape blending regions) are
skipped; nothing downstream reads them.

```bash
Exporter.exe -i "C:\...\War-WindowsNoEditor.pak" -o export -a War/Content/Maps/HomeRegionC
Exporter.exe -i "C:\...\Paks" -o export -t
```

### Step 1 - Export Game Files

Runs `Exporter.exe` over every map. Writes:

| Path | Contents |
|---|---|
| `export/_json/` | Per-map JSON (symbols, groups, blueprints, splines + transforms) |
| `export/_meshes/` | Static/skeletal meshes as `.pskx` / `.psk` |
| `export/_heightmap/` | 16-bit grayscale heightmaps (2200×2200 px, 1 m/px) |
| `export/_layers/` | Per-region terrain weightmap layers (8-bit grayscale) |
| `export/hashes.txt` | Short SHA-256 of every region JSON (tracked in git) |

```bash
python 1_export.py
```

After the export it prints a summary, a diff of exported meshes against
`utils/catalogue.json`, and, when a previous `hashes.txt` exists, which
region JSONs changed, appeared or disappeared since then. That tells you
which regions a game update touched and need steps 2-5 again.

`War-WindowsNoEditor.pak` is located automatically by searching every Steam
library folder (any drive) for a Foxhole install. If that fails, or Foxhole
is installed somewhere unusual, set the `FOXHOLE_PAK_PATH` environment
variable to the full path of the `.pak` file.

### Step 2 - Generate Blender Scenes

```bash
python 2_blend_all.py                    # interactive
python 2_blend_all.py OarbreakerHex
python 2_blend_all.py -a
python 2_blend_all.py -nt OarbreakerHex  # exclude terrain
```

Output: `export/blend/<MapName>.blend`. JSONs whose stem isn't listed
in `utils/region_centers.json` (e.g. `MainMenu.json`) are skipped so
only real regions produce a `.blend`.

### Step 3 - Build Region Spill Scenes

Per-region `.blend` with a 200 m spill from hexagonal neighbors. Spill
categories are defined in `utils/catalogue.json` (each key is a
category name with a list of mesh names). Only the focus region's
terrain and water are fully included; neighbor spill contributes only
non-water categories. Every category must have a color entry in
`CATEGORY_COLORS` (`utils/config.py`); the reserved names `terrain`
(built from the heightmap), `water` (cloned as `deep_water` occluders
at `DEEP_WATER_DEPTH` m), and `deep_water` are handled specially.

```bash
python 3_blend_spills.py                # interactive
python 3_blend_spills.py OarbreakerHex
python 3_blend_spills.py -a
```

Output: `export/blend_spill/<Region>.blend`.

**Purge list.** Some regions hold paired scenario objects of which only one
spawns in game, one far more often than the other. Keeping both clutters
the map, so `PURGE` in `utils/config.py` lists instances to drop, per
region and per JSON section: `symbols`, `groups` and `splines` by mesh name
and the instance's own transform, `blueprints` by class and `_self`
transform. This step leaves them out of every build (as the focus region
and as neighbor spill). The JSONs on disk are never modified. A run ends
with a list of the purged instances it touched, and an entry that no longer
matches anything (the game moved or removed it) is reported as a `[WARN]`
so the list can be updated.

### Step 4 - Render Region Bakes

Opens each spill `.blend` and renders top-down `TILE_SIZE`×`TILE_SIZE` bakes. Without
any of `-svg`/`-ao`/`-hm`/`-id`/`-r`/`-b`/`-sl`, all bakes are produced.
The `-svg` pass reads `export/_json/<region>.json` directly and runs
before the `.blend` is opened; `-b` needs no Blender either (it reads the
weightmaps and the region's ID bakes from disk); all others need Blender.

```bash
python 4_render_spills.py               # interactive, all bakes
python 4_render_spills.py OarbreakerHex
python 4_render_spills.py -a            # all regions
python 4_render_spills.py -a -ao -id    # subset
```

Outputs (per-region PNGs; the flag producing each is in parentheses):

- `ao/` (`-ao`) - grayscale AO bake with slope shading.
- `heightmap_landscape/`, `heightmap_water/` (`-hm`) - grayscale bakes.
  `heightmap_landscape` passes through water; `heightmap_water` stops
  on the water surface.
- `roads/` (`-r`) - RGBA spline coverage (SSAA, colored via `SPLINE_COLORS`).
- `beaches/` (`-b`) - shore sand, `BEACH_COLOR`. Barges deploy their ramp only
  onto ground whose physical material is Sand or WetSand, and `_layers/` is
  named by physical material, so a pixel is sand when one of `BEACH_LAYERS`
  is its dominant weightmap layer. Only sand reachable from water through
  sand within `BEACH_FADE_PX` counts; alpha is full up to `BEACH_FULL_PX`
  from the water, fades to 0 at `BEACH_FADE_PX`, and is masked to
  `terrain × (not water)`. Reads `id/water` and `id/terrain`: like step 5's
  stage prerequisites, a `-b` run without `-id` pulls `-id` in for every
  region whose ID bake is missing or older than its spill `.blend`.
  Foxhole's beach landscape splines are editor-only sculpt guides with no
  game mesh, so the exporter drops them.
- `id/<category>/` (`-id`) - 8-bit coverage per category (`ID_SSAA`),
  including the water coverage in `id/water/`.
- `split_layers/<layer>/` (`-sl`) - RGBA Cycles AO per split layer; each
  category is tinted via its color from `SPLIT_LAYERS`.
- `svg_layers/<layer>/` (`-svg`) - cairosvg raster of `utils/svg/<cat>/<name>.svg`
  instanced via `<use>` at every JSON transform. UE cm map to SVG px as
  `x = x_cm * 1776 / 189000 + 1024`.
- `bridges_aim/` (`-svg`) - procedural bridge aligning lines (own folder, not
  under `svg_layers/`); stitched into `_final/assembly/bridges_aim.png` in step 5.
  Routed through `id/water`, which an `-svg` run without `-id` pulls in the
  same way `-b` does.

Tunables in `utils/config.py`: `TERRAIN_WHITELIST` (categories that
participate in ao/hm/id), `SPLIT_LAYERS`, `SVG_LAYERS`, `SPLINE_CATEGORIES`,
`SPLINE_COLORS`, `SPLINE_LAYER_SSAA`, `ID_SSAA`, `BEACH_*`.

Performance knobs for this step: see [Parallel Execution](#parallel-execution).

### Step 5 - Finalize Exports

Stitches every step-4 bake into world-sized PNGs, derives heightmap
products, and assembles the composites a map is built from. All assembly
work is in-memory; only the listed files are written.

Each output below is a named stage. With no arguments the arrow-key picker
opens with every stage ticked; names can also be given on the command line.
A stage whose outputs are newer than its inputs (the hex mask and
`region_centers.json` count as inputs of every stage) and whose config
settings are unchanged since it was last built is reported as up to date
and skipped, so re-running after changing one bake or one setting only
redoes what depends on it. Each stage lists the settings it reads (e.g.
`base_layer`: `LAYER_COLORS`, `ID_RECOLOR`, `SHADES_BLUR_*`, `HM_SPLIT_M`);
their fingerprints are kept in `_final/.stage_settings.json`. Code changes
are not tracked: `-f` rebuilds regardless.

```bash
python 5_finalize_exports.py                 # pick outputs interactively
python 5_finalize_exports.py base_layer rdz  # named stages
python 5_finalize_exports.py -a              # every stage
python 5_finalize_exports.py -a -f           # ... and ignore the mtime check
```

Stage names: `ao`, `roads`, `beaches`, `bridges_aim`, `split_layers`,
`svg_layers`, `id`, `fly_alert`, `contour`, `heightmap_simple`,
`dive_alert`, `base_layer`, `contours`, `rdz`, `grid_pattern`,
`no_intel_pattern`, `ranges`.

Shared intermediates (world alpha, ID coverage, heightmaps, shades) are
stitched on first use and freed once no remaining stage needs them, so
asking for one stage only pays for that stage's inputs. Masks always come
from the step-4 tiles; the only cross-stage file dependency is `rdz` and
`ranges` reading the stitched `svg_layers`, which is pulled into the run
when stale.

Output layout (under `export/_final/`):

- `technical/ao.png` - stitched AO (slope shading baked in via step 4).
- `technical/heightmap_simple.png` - 8-bit from `heightmap_water`;
  shade 60 = z=0 m, 1 shade = 0.5 m.
- `technical/contour.png` - black RGBA lines where `hm // 250`
  increments across a 4-neighbor boundary, masked to terrain.
- `assembly/roads.png`, `assembly/beaches.png` - stitched from step 4.
- `assembly/fly_alert.png` - `utils/img/fly_alert_pattern.png` tiled, alpha
  ramped between `FLY_ALERT_MIN_M` and `FLY_ALERT_MAX_M`, gated by
  `rocks_cov`.
- `assembly/dive_alert.png` - depth-graded overlay: each submerged pixel is
  coloured by its depth below the water surface via `DIVE_ALERT_GRADIENT`
  (a list of depth ranges, each with a start/end `#RRGGBB[AA]` colour pair
  interpolated linearly across the range; depths past the last range are
  transparent). Band transitions are smoothed by a water-masked normalized
  blur (`DIVE_ALERT_BLUR_*`) that never bleeds onto land or weakens the
  shoreline edge. Alpha is gated by `water_cov`.
- `assembly/base_layer.png` - single terrain composite:
  `terrain_recolor` (weighted blend of `ID_RECOLOR` per non-water
  category, nearest-filled) + `shades` (terrain weightmaps with
  `LAYER_COLORS`, `SHADES_BLUR_*`) + `highs × ground` (add) +
  `lows × ground` (difference) + `water_recolor` (multiply) + `ao`
  (multiply); alpha = world hex mask.
- `assembly/contours.png` - contour blurred 3×3, alpha × `(0.5*water + ground)`.
- `assembly/rdz.png` - the one-hex `utils/img/rdz_pattern.png` cloned into
  every region like the patterns below, alpha punched by
  `svg_layers/rdz_grace`.
- `assembly/grid_pattern.png`, `assembly/no_intel_pattern.png` - the
  one-hex patterns `utils/img/grid_pattern.png` and
  `utils/img/no_intel_pattern.png` cloned into every region and clipped
  to its hex. To restyle the grid, redraw that single 2048×2048 tile.
- `assembly/ranges.png` - alpha-over of range svg_layers:
  `ranges_tap × ground`, `ranges_intel`, `ranges_ai × ground`,
  `ranges_mh`, `ranges_cg × water`, `ranges_aag`.
- `assembly/bridges_aim.png` - bridge aligning lines, built procedurally in `4_render_spills.py` (per-region tiles in their own `bridges_aim/` folder).
- `id/<cat>.png`, `split_layers/<layer>.png`, `svg_layers/<layer>.png`
  - verbatim stitches of the per-region bakes.

- `assembly/roads_fix.png` - `utils/img/roads_fix.png` copied as is, a
  hand-made world layer meant to sit directly under `roads.png`.

The release zips are these folders.

### Step 6 - Toolset

Everything between a world image and the game. A **world PNG** is any
image on the 20528×12704 canvas the `_final` layers use. **Region PNGs**
are one image per region, named after it (`AcrithiaHex.png`, ...), in the
game's own texture size of 2048×1776.

```bash
python 6_toolset.py                                  # menu
python 6_toolset.py recipe recipe.json [-n NAME] [-o DIR] [--store]
python 6_toolset.py mod    World.png [-n NAME] [-o DIR] [--store]
python 6_toolset.py mod    RegionsDir/ [-n NAME] [-o DIR] [--store]
python 6_toolset.py unpack War-WindowsNoEditor_X.pak [-o DIR]
python 6_toolset.py break  World.png [--1k] [-o DIR]
python 6_toolset.py stitch RegionsDir/ [-o World.png]
```

| Action | Does | Output (default) |
|---|---|---|
| `recipe` | composites the layers a [recipe](#recipes) lists, then builds a map mod from them | `./War-WindowsNoEditor_<NAME>.pak` (NAME from the recipe) |
| `mod` | builds a map mod from a world PNG or a folder of region PNGs | `./War-WindowsNoEditor_<NAME>.pak` |
| `unpack` | decodes a map mod back into region PNGs | `./<pak name>/<Region>.png` |
| `break` | cuts a world PNG into region PNGs, each clipped to its hex; `--1k` halves them to 1024×888 | `./<World>/<Region>.png` |
| `stitch` | pastes a folder of region PNGs back onto a world PNG | `./<folder>.png` |

Outputs land in the current directory, so the toolset can be run from a
work folder: `python path/to/6_toolset.py`. In the menu, inputs are picked
from the current directory or pasted as a path.

A region folder may hold only some regions: a mod built from it replaces
just those and leaves the rest of the map as the game draws it. Folder
images may be 2048×1776 or a full 2048×2048 hex (trimmed automatically).
`unpack` reads mods built here. Paks made by other tools may use a newer format, which it refuses with an
error rather than misreading.

Textures are compressed to BC7 with [etcpak](https://github.com/K0lb3/etcpak)'s
build of bc7enc, spread over every core (a full map takes a few seconds).

#### Recipes

A recipe is a JSON file listing world layers bottom to top. `recipe`
composites them on the world canvas and packs the result, without an image
editor. The [recipe.json](recipe.json) at the repo root stacks the `_final`
layers the same way the released `TseMap_Red` mod does, and builds
`War-WindowsNoEditor_CustomMapMod.pak`. Copy and edit it to make your own.

```json
{
  "name": "CustomMapMod",
  "root": "export/_final",
  "layers": [
    "assembly/base_layer.png",
    {"file": "assembly/contours.png", "opacity": 64},
    {"file": "svg_layers/foliage.png", "mask": "id/terrain.png"},
    {"shadow": ["assembly/roads.png", "svg_layers/bridges.png"]},
    "assembly/roads.png",
    {"file": "assembly/rdz.png", "opacity": 100, "color": "#000000"}
  ]
}
```

- `name` - mod name, overridden by `-n`. Defaults to the recipe file's name.
- `root` - folder the layer paths are relative to, itself relative to the
  recipe file. Point it at wherever the release zips were extracted.
- `layers` - each entry is a path, or an object with one source and any
  number of adjustments:

| Key | Meaning |
|---|---|
| `file` | a world PNG |
| `shadow` | a list of world PNGs: a soft drop shadow of their combined shape (black unless `color` is set); `blur` (default 1, Gaussian sigma in px) and `strength` (default 1.5, alpha multiplier) tune it |
| `opacity` | 0-255, scales the layer's alpha like paint.net's layer opacity |
| `mask` | a greyscale world PNG multiplied into the alpha (black hides) |
| `color` | `"#rrggbb"`: recolors the whole layer and keeps its alpha (`"#000000"` on `rdz` gives the Dark variant) |

A missing file, or an image that isn't 20528×12704, stops
the build before anything is written. A full-map composite holds two
world-sized images in memory at once (about 2 GB).

#### As a Library

As a library (from a script at the repo root):

```python
from toolset import mapmod

world = mapmod.load_image("MyMap.png")          # BGRA, as cv2 reads it
tiles = mapmod.break_world(world)               # {"MapAcrithiaHex": 1776x2048x4, ...}
textures = mapmod.encode_all(tiles)             # {name: BC7 bytes}
mapmod.write_mod(mapmod.mod_path(".", "MyMap"), textures)

textures = mapmod.read_mod("War-WindowsNoEditor_MyMap.pak")
tile = mapmod.decode(textures["MapAcrithiaHex"])
```

`toolset/pak.py` is the bare `.pak` reader and writer, if you want to ship
other files the same way.

## Parallel Execution

Steps 2, 3, and 4 fan out to subprocesses when more than one item is
queued; `utils/parallel.py` holds the fan-out. Worker count comes from
`NUM_WORKERS` and `NUM_WORKERS_SPILLS` in `utils/config.py` - set to `1`
for serial execution.

Step 4 tuning:

- `NUM_WORKERS_SPILLS` is bounded by **memory, not cores** - each worker
  holds a region scene plus its BVH, and overshooting RAM pages.
- `BAKE_ROW_THREADS` splits cores across workers (`0` = automatic). The
  raycast bakes hold the GIL, so throughput comes from processes, not
  threads.
- `NUM_WORKERS_SVG` replaces the above for an SVG-only run, which never
  opens a `.blend`.
- `GPU_SERIALIZE_RENDERS` makes Cycles passes queue for the GPU via
  `utils/gpu_lock.py` instead of thrashing it; `CYCLES_USE_CPU_WITH_GPU`
  adds CPU devices, which only pays off for a lone render.
- `BVH_CACHE_REUSE` shares one BVH between consecutive bakes over the same
  objects. Disable it if `id/` output must be bit-identical - a reused
  tree can tie-break coincident faces differently.

## Project Structure

```text
.
├── 0_make_release.py           # Builds Exporter.exe from C# source
├── 1_export.py                 # Runs Exporter.exe against game .pak, tracks JSON hashes
├── 2_blend_all.py              # Full-map Blender scenes
├── 3_blend_spills.py           # Per-region spill .blend (applies the purge list)
├── 4_render_spills.py          # Top-down bakes per region
├── 5_finalize_exports.py       # Stitches bakes + layers into world PNGs
├── 6_toolset.py                # World PNG ⇄ region PNGs ⇄ map mod, recipe → map mod
├── recipe.json                 # Template recipe: rebuilds the TseMap_Red release mod
├── Exporter/                   # C# exporter source (.NET 10, win-x64)
│   ├── Program.cs
│   ├── MapExporter.cs
│   ├── LandscapeStitcher.cs
│   ├── TransformMath.cs
│   ├── JsonOutput.cs
│   └── Constants.cs
├── toolset/
│   ├── mapmod.py               # Regions, break/stitch, BC7, map-mod read/write
│   ├── recipe.py               # Layer recipes: parse + composite onto the world canvas
│   ├── pak.py                  # Unreal .pak (v3) container reader/writer
│   └── assets/                 # Per-region uasset headers + upscaled WorldMapBG
├── utils/
│   ├── config.py               # Shared constants and tunables (incl. PURGE)
│   ├── png.py                  # 16/8-bit PNG read/write
│   ├── psk.py                  # PSK/PSKX parser + mesh cache
│   ├── blender.py              # Materials, transforms, terrain, splines, GN instancing
│   ├── map.py                  # Map class (scene construction)
│   ├── bake.py                 # Top-down bakers (AO, heightmap, ID, coverage)
│   ├── svg_render.py           # SVG layer builder + cairosvg rasterizer
│   ├── beaches.py              # Beaches layer from Sand/WetSand weightmaps
│   ├── regions.py              # Region geometry, deep water, spill builder
│   ├── parallel.py             # Subprocess fan-out helper
│   ├── tui.py                  # Arrow-key pickers, prompts, progress bars
│   ├── progress.py             # Trackers feeding the bars; worker log capture
│   ├── gpu_lock.py             # Machine-wide lock serializing Cycles renders
│   ├── foxhole_locator.py      # Finds War-WindowsNoEditor.pak across Steam libraries
│   ├── region_centers.json     # World-space pixel coords of all regions
│   ├── catalogue.json          # Per-category mesh lists for step 3 spill
│   ├── svg/<category>/<name>.svg  # Source icons instanced into svg_layers
│   └── img/                    # Hand-made image inputs
│       ├── mask.png            # One-hex mask (2048×2048)
│       ├── grid_pattern.png    # One-hex grid, cloned by step 5
│       ├── no_intel_pattern.png # One-hex no-intel hatching, cloned by step 5
│       ├── fly_alert_pattern.png # World texture sampled by fly_alert
│       ├── rdz_pattern.png     # One-hex RDZ hatching, cloned by step 5
│       └── roads_fix.png       # World layer patching road gaps (ships in assembly.zip)
└── CUE4Parse/                  # Git submodule - Unreal Engine asset reader
```

## Transform Format

### Regular Objects

Every placed object in the exported JSON is a 9-element array:

```text
[x, y, z, sx, sy, sz, pitch, yaw, roll]
```

Coordinates are UE world-space centimetres; rotations are degrees.
`utils/blender.py` converts these to Blender metres/radians.

### Splines

Spline mesh components are exported as 23-element arrays containing the cubic hermite spline parameters needed to reconstruct the deformed mesh:

```text
[0..2]   world start position (x, y, z in cm)
[3..5]   world start tangent (x, y, z in cm)
[6..8]   world end position (x, y, z in cm)
[9..11]  world end tangent (x, y, z in cm)
[12]     start roll (radians)
[13..14] start offset (x, y, component-space 2D)
[15..16] start scale (x, y)
[17]     end roll (radians)
[18..19] end offset (x, y, component-space 2D)
[20..21] end scale (x, y)
[22]     forward axis (0=X, 1=Y, 2=Z)
```
