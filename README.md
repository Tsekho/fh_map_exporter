# Foxhole Map Exporter

Python pipeline for exporting and processing maps from [Foxhole](https://store.steampowered.com/app/505460/Foxhole/).
Updated for U66 Devbranch Phase 2.

### Credits

The amazing idea of thresholded water depth coloring was adopted from
[Knight of Science's fork](https://github.com/Knight-of-Science/fh_map_exporter).

## Pipeline

```text
0_make_release.py      ->  builds Exporter.exe from C# source
1_export.py            ->  Exporter.exe reads .pak → export/_json/, _meshes/, _heightmap/, _layers/
2_blend_all.py         ->  full-map Blender scenes → export/blend/<MapName>.blend
3_blend_spills.py      ->  per-region .blend with neighbor spill → export/blend_spill/<Region>.blend
4_render_spills.py     ->  top-down per-region bakes → export/{ao,heightmap_landscape,heightmap_water,
                           roads,beaches,bridges_aim,id/<cat>,split_layers/<layer>,svg_layers/<layer>}/<Region>.png
5_finalize_exports.py  ->  stitches bakes into world PNGs and assembles final composites
                           → export/_final/{technical,assembly,id,split_layers,svg_layers}/
6_breaker.py           ->  interactive: break any world PNG back into per-region tiles
                           → <input_stem>/<Region>.png
```

## Requirements

- Foxhole
- Python 3.10–3.13 (no `bpy` on 3.14)
- **numpy**, **opencv-python** (steps 4 & 5), **bpy** (steps 2–4), **cairosvg** (step 4, `-svg`)
- Blender 5
- [.NET 10 SDK](https://dotnet.microsoft.com/download) (step 0 only)
- Cairo (the easiest way to install it - [GTK for Windows Runtime Environment](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer))
```bash
pip install numpy opencv-python bpy cairosvg
```

Clone with submodules (CUE4Parse is required):

```bash
git clone --recurse-submodules https://github.com/Tsekho/fh_map_exporter.git
```

## Usage

### Step 0 - Build the exporter

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
| `-o <export_path>` | Output folder (`meshes/` subdirectory is created) |
| `-t`, `--texture` | Terrain layers and heightmaps |
| `-a <asset_path>` | Single asset, e.g. `War/Content/Maps/HomeRegionC.umap` (`.umap` optional). Omit to export all maps under `War/Content/Maps`. |

```bash
Exporter.exe -i "C:\...\War-WindowsNoEditor.pak" -o export -a War/Content/Maps/HomeRegionC
Exporter.exe -i "C:\...\Paks" -o export -t
```

### Step 1 - Export Game Files

Clears `export/` and runs `Exporter.exe`. Writes:

| Directory | Contents |
|---|---|
| `export/_json/` | Per-map JSON (symbols, groups, blueprints + transforms) |
| `export/_meshes/` | Static/skeletal meshes as `.pskx` / `.psk` |
| `export/_heightmap/` | 16-bit grayscale heightmaps (2200×2200 px, 1 m/px) |
| `export/_layers/` | Per-region terrain weightmap layers (8-bit grayscale) |

```bash
python 1_export.py
```

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

### Step 4 - Render Region Bakes

Opens each spill `.blend` and renders top-down `TILE_SIZE`×`TILE_SIZE` bakes. Without
any of `-svg`/`-ao`/`-hm`/`-id`/`-r`/`-b`/`-sl`, all bakes are produced.
The `-svg` pass reads `export/_json/<region>.json` directly and runs
before the `.blend` is opened; all others need Blender.

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
- `roads/` (`-r`), `beaches/` (`-b`) - RGBA spline coverage (SSAA, colored via
  `SPLINE_COLORS`). Beaches alpha is masked to `terrain × (not water)`.
- `id/<category>/` (`-id`) - 8-bit coverage per category (`ID_SSAA`),
  including the water coverage in `id/water/`.
- `split_layers/<layer>/` (`-sl`) - RGBA Cycles AO per split layer; each
  category is tinted via its color from `SPLIT_LAYERS`.
- `svg_layers/<layer>/` (`-svg`) - cairosvg raster of `utils/svg/<cat>/<name>.svg`
  instanced via `<use>` at every JSON transform. UE cm map to SVG px as
  `x = x_cm * 1776 / 189000 + 1024`.
- `bridges_aim/` (`-svg`) - procedural bridge aligning lines (own folder, not
  under `svg_layers/`); stitched into `_final/assembly/bridges_aim.png` in step 5.

Tunables in `utils/config.py`: `TERRAIN_WHITELIST` (categories that
participate in ao/hm/id), `SPLIT_LAYERS`, `SVG_LAYERS`, `SPLINE_CATEGORIES`,
`SPLINE_COLORS`, `SPLINE_LAYER_SSAA`, `ID_SSAA`.

### Step 5 - Finalize Exports

Stitches every step-4 bake into world-sized PNGs, derives heightmap
products, and assembles composites consumed by the map mod generator.
All assembly work is in-memory; only the listed files are written.

```bash
python 5_finalize_exports.py
```

Output layout (under `export/_final/`):

- `technical/ao.png` - stitched AO (slope shading baked in via step 4).
- `technical/heightmap_simple.png` - 8-bit from `heightmap_water`;
  shade 60 = z=0 m, 1 shade = 0.5 m.
- `technical/contour.png` - black RGBA lines where `hm // 250`
  increments across a 4-neighbor boundary, masked to terrain.
- `assembly/roads.png`, `assembly/beaches.png` - stitched from step 4.
- `assembly/fly_alert.png` - `utils/fly_alert_pattern.png` tiled, alpha
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
- `assembly/rdz.png` - `utils/rdz_pattern.png` tiled, alpha punched by
  `svg_layers/rdz_grace`.
- `assembly/ranges.png` - alpha-over of range svg_layers:
  `ranges_tap × ground`, `ranges_intel`, `ranges_ai × ground`,
  `ranges_mh`, `ranges_cg × water`, `ranges_aag`.
- `assembly/bridges_aim.png` - bridge aligning lines, built procedurally in `4_render_spills.py` (per-region tiles in their own `bridges_aim/` folder).
- `id/<cat>.png`, `split_layers/<layer>.png`, `svg_layers/<layer>.png`
  - verbatim stitches of the per-region bakes.

**Important**: This exporter's output is intended for use with the
[map mod generator](https://github.com/Tsekho/fh_map_mod_generator) that allows to convert composed layers into a standalone map mod via a single function call. For any other use you'll most likely want to break it apart into separate regions - see step 6.

### Step 6 - Break World PNG Into Regions

Interactive helper that slices any world-sized PNG back into per-region
tiles using `utils/region_centers.json` and `utils/mask.png`. Lists
`*.png` files in the current directory (or accepts a pasted path), then
asks whether to downscale to 1k.

```bash
python 6_breaker.py
```

For each region, extracts a `TILE_SIZE`×`TILE_SIZE` crop centred on the region's
world-pixel coordinates, multiplies `mask.png` into the alpha channel,
then saves either:

- default: **2048x1776**
- 1k mode: **1024x888** (default map textures size)

Output: `<input_stem>/<Region>.png` (created in the current directory).

## Parallel Execution

Steps 2, 3, and 4 fan out to multiple subprocesses when more than one
item is queued. Worker count is controlled by `NUM_WORKERS` and `NUM_WORKERS_SPILLS` in
`utils/config.py`. Set it to `1` for serial execution.
The fan-out logic lives in `utils/parallel.py`.

## Project Structure

```text
.
├── 0_make_release.py           # Builds Exporter.exe from C# source
├── 1_export.py                 # Runs Exporter.exe against game .pak
├── 2_blend_all.py              # Full-map Blender scenes
├── 3_blend_spills.py           # Per-region spill .blend
├── 4_render_spills.py          # Top-down bakes per region
├── 5_finalize_exports.py       # Stitches bakes + layers into world PNGs
├── 6_breaker.py                # Breaks a world PNG back into per-region tiles
├── Exporter/                   # C# exporter source (.NET 10, win-x64)
│   ├── Program.cs
│   ├── MapExporter.cs
│   ├── LandscapeStitcher.cs
│   ├── TransformMath.cs
│   ├── JsonOutput.cs
│   └── Constants.cs
├── utils/
│   ├── config.py               # Shared constants and tunables
│   ├── png.py                  # 16/8-bit PNG read/write
│   ├── psk.py                  # PSK/PSKX parser + mesh cache
│   ├── blender.py              # Materials, transforms, terrain, splines, GN instancing
│   ├── map.py                  # Map class (scene construction)
│   ├── bake.py                 # Top-down bakers (AO, heightmap, ID, coverage)
│   ├── svg_render.py           # SVG layer builder + cairosvg rasterizer
│   ├── regions.py              # Region geometry, deep water, spill builder
│   ├── parallel.py             # Subprocess fan-out helper
│   ├── region_centers.json     # World-space pixel coords of all regions
│   ├── mask.png                # Per-region hex mask
│   ├── fly_alert_pattern.png   # BGRA texture sampled by the fly_alert overlay
│   ├── rdz_pattern.png         # BGRA texture punched by svg_layers/rdz_grace
│   ├── svg/<category>/<name>.svg  # Source icons instanced into svg_layers
│   └── catalogue.json          # Per-category mesh lists for step 3 spill
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
