# Foxhole Map Exporter

Python pipeline for exporting and processing maps from [Foxhole](https://store.steampowered.com/app/fox).

## Pipeline

```text
0_make_release.py      ->  builds Exporter.exe from C# source
1_export.py            ->  Exporter.exe reads .pak → export/_json/, _meshes/, _heightmap/, _layers/
2_blend_all.py         ->  builds full-map Blender scenes → export/blend/<MapName>.blend
3_blend_spills.py      ->  builds per-region .blend with 200 m neighbor spill (rocks/glaciers/
                           landscape_meshes only; neighbor terrain excluded)
                           → export/blend_spill/<Region>.blend
4_render_spills.py     ->  top-down bakes per region
                           → export/{ao,heightmap_landscape,id,water}/<Region>.png
5_finalize_exports.py  ->  derives heightmap products, stitches every bake + terrain
                           layer into world-sized PNGs
                           → export/_final/
```

## Requirements

### Game

- Foxhole

### Python Dependencies

- Python 3.10–3.13 (no `bpy` on 3.14)
- **numpy**
- **opencv-python** (steps 4 & 5)
- **bpy** / Blender Python environment (steps 2–4)

```bash
pip install numpy opencv-python bpy
```

### Blender

- Blender 5

### Build Only (Step 0)

- [.NET 10 SDK](https://dotnet.microsoft.com/download)

Clone the repository with submodules (CUE4Parse is a required submodule):

```bash
git clone --recurse-submodules https://github.com/Tsekho/fh_map_exporter.git
```

## Usage

### Step 0 - Build the exporter

Compiles `Exporter/` and outputs `Exporter.exe` at the repo root. Only needed when you want to rebuild from source; a pre-built `Exporter.exe` is included.

```bash
python 0_make_release.py
```

### Exporter.exe - Standalone Usage

`Exporter.exe` is a self-contained win-x64 binary and can be run directly without the Python wrapper.

```text
Exporter.exe -i <pak_path> -o <export_path> [-t] [-a <asset_path>]
```

| Argument | Description |
|---|---|
| `-i <pak_path>` | Path to the `.pak` file **or** its containing directory |
| `-o <export_path>` | Output folder (`meshes/` subdirectory will be created) |
| `-t`, `--texture` | Terrain layers and heightmaps |
| `-a <asset_path>` | Single asset to export, e.g. `War/Content/Maps/HomeRegionC.umap` (`.umap` extension is optional). If omitted, all umaps under `War/Content/Maps` are exported in alphabetical order. |

```bash
# Export a single map
Exporter.exe -i "C:\...\War-WindowsNoEditor.pak" -o export -a War/Content/Maps/HomeRegionC

# Export a specific umap with explicit extension
Exporter.exe -i "C:\...\War-WindowsNoEditor.pak" -o export -a War/Content/Maps/Master/AcrithiaHex.umap

# Export all maps + textures from a Paks directory
Exporter.exe -i "C:\...\Paks" -o export -t
```

### Step 1 - Export Game Files

Clears `export/` and runs `Exporter.exe` against the game `.pak`. Writes:

| Directory | Contents |
|---|---|
| `export/_json/` | Per-map JSON (symbols, groups, blueprints + transforms) |
| `export/_meshes/` | Static/skeletal meshes as `.pskx` / `.psk` |
| `export/_heightmap/` | 16-bit grayscale heightmaps (2200×2200 px, 1 m/px) |
| `export/_layers/` | Per-region terrain weightmap layers (8-bit grayscale PNG) |

```bash
python 1_export.py
```

### Step 2 - Generate Blender Scenes

Reads JSON and meshes from `export/` and builds `.blend` files.

```bash
python 2_blend_all.py                    # interactive selection
python 2_blend_all.py OarbreakerHex      # specific map
python 2_blend_all.py -a                 # every map
python 2_blend_all.py -nt OarbreakerHex  # exclude heightmap terrain
```

Output goes to `export/blend/<MapName>.blend`.

### Step 3 - Build Region Spill Scenes

Generates a `.blend` file per region that includes a 200 m spill of rocks,
glaciers and landscape meshes from each hexagonal neighbor. Only the focus
region's terrain is included (neighbor terrain is excluded). This is the
scene consumed by step 4.

```bash
python 3_blend_spills.py                # interactive selection
python 3_blend_spills.py OarbreakerHex
python 3_blend_spills.py -a
```

Output goes to `export/blend_spill/<Region>.blend`.

### Step 4 - Render Region Bakes

Opens each spill `.blend` and renders top-down 2048×2048 bakes per region.
If none of `-ao` / `-hm` / `-id` is passed, all bakes are produced.

```bash
python 4_render_spills.py               # interactive selection, all bakes
python 4_render_spills.py OarbreakerHex # one region, all bakes
python 4_render_spills.py -a            # every region
python 4_render_spills.py -a -hm        # only heightmap
python 4_render_spills.py -a -ao -id    # AO + id + water, skip heightmap
```

Output goes to `export/ao/`, `export/heightmap_landscape/`, `export/id/`,
`export/water/`.

### Step 5 - Finalize Exports

This step only writes stitched world-sized PNGs; it never produces
per-region intermediates. Heightmap-derived products are computed
per-tile in memory and pasted directly onto world canvases.

World-sized products derived from `heightmap_landscape`:

- `heightmap_highs` - 1 shade = +0.5 m above the 10 m split (0 below / void)
- `heightmap_lows` - 1 shade = -0.5 m below the 10 m split (0 above / void)
- `curvature_peaks` - positive half of the 2D Laplacian of elevation
- `curvature_dips` - negative half of the 2D Laplacian of elevation
- `slopes` - slope magnitude as a black RGBA overlay (alpha 0 = flat,
  alpha saturates at 64 for any slope ≥ ~14°, capping opacity at ~25%)
- `fly_alert` - textured RGBA overlay built from
  `utils/fly_alert_pattern.png`: pattern RGB is passed through and an
  elevation coefficient (0 at 90 m, 1 at 100 m+) multiplies the
  pattern's alpha, so the pattern only shows through where terrain is
  high enough. The pattern is tiled to the world canvas size if needed.
- `contour` - stepwise black RGBA overlay built from the 16-bit
  heightmap (`step = hm // 250`), drawn where a pixel's step is
  exactly one greater than a 4-neighbor, masked to terrain id pixels

Stitched step-4 bakes (`ao`, `id`, `water`) are emitted as BGRA with
alpha = 0 outside the stitched hex mask. `slopes`, `fly_alert` and
`contour` are emitted as RGBA overlays with transparent backgrounds.

Two recolored products are also written from the stitched id/water
canvases using `ID_RECOLOR` from `utils/config.py`:

- `terrain_recolor.png` - id pixels remapped via `ID_RECOLOR` for every
  category except water (water becomes transparent; out-of-bounds is
  transparent).
- `water_recolor.png` - solid `ID_RECOLOR["water"]` wherever the water
  mask is > 0, transparent elsewhere.

Every terrain layer folder in `export/_layers/` is composited into a
single `shades.png`. Each layer gets a color from `LAYER_COLORS` in
`utils/config.py` (or a deterministic random bright color if not
listed), and layers are combined via "alpha betting": the layer with
the highest intensity at a pixel claims it and paints that pixel with
its color. Layers are masked to land pixels only (terrain id with water
excluded); pixels not claimed by any layer stay fully transparent. The
final image is saved as BGRA, and the resolved layer -> color mapping
is written to `shades_palette.json`.

```bash
python 5_finalize_exports.py
```

Output goes to `export/_final/`:

```text
export/_final/
    ao.png
    heightmap_highs.png
    heightmap_lows.png
    curvature_peaks.png
    curvature_dips.png
    slopes.png
    fly_alert.png
    id.png
    water.png
    terrain_recolor.png
    water_recolor.png
    contour.png
    shades.png
    shades_palette.json
```

The stitched PNGs under `export/_final/` are the only generated files
committed to the repo (see `.gitignore`). They are compatible with the
[map mod generator](https://github.com/Tsekho/fh_map_mod_generator).

## Parallel Execution

Steps 2, 3, and 4 fan out to multiple subprocesses whenever more than one
item is queued (either via `-a` or interactive "0 = all" selection). Each
work item runs in its own child process so every worker gets a fresh `bpy`
state; child stdout is streamed back with a `[label]` prefix so interleaved
output stays readable.

Worker count is controlled by `NUM_WORKERS` in `utils/config.py` (default
`4`). Set it to `1` to force serial execution in the parent process. The
fan-out logic lives in `utils/parallel.py`.

## Project Structure

```text
.
├── 0_make_release.py           # Builds Exporter.exe from C# source
├── 1_export.py                 # Runs Exporter.exe against game .pak
├── 2_blend_all.py              # Generates full-map Blender scenes
├── 3_blend_spills.py           # Generates per-region spill .blend
├── 4_render_spills.py          # Top-down bakes per region
├── 5_finalize_exports.py       # Stitches bakes + layers into world PNGs
├── Exporter/                   # C# exporter source (.NET 10, win-x64 self-contained)
│   ├── Program.cs              # CLI entry point; sets up CUE4Parse provider
│   ├── MapExporter.cs          # Reads .umap, resolves transforms, writes JSON + meshes
│   ├── LandscapeStitcher.cs    # Exports heightmaps and weightmap layers as PNGs
│   ├── TransformMath.cs        # UE4 Euler/quaternion ↔ rotation matrix conversions
│   ├── JsonOutput.cs           # Custom compact JSON serializer
│   └── Constants.cs            # Mesh filters, blueprint patches, name normalizations
├── utils/
│   ├── config.py               # Shared constants: paths, tile geometry, categories, NUM_WORKERS
│   ├── helpers.py              # PSK/PSKX parser, Blender helpers, bakers, Map class
│   ├── parallel.py             # Subprocess fan-out helper used by the "all"/multi-item modes
│   ├── region_centers.json     # World-space pixel coordinates of all ~55 map regions
│   ├── mask.png                # Hex-shaped mask applied per region tile during stitching
│   ├── fly_alert_pattern.png   # BGRA texture sampled by step 5's fly_alert overlay
│   └── catalogue.json          # Partially categorized asset names
└── CUE4Parse/                  # Git submodule - Unreal Engine asset reader
```

## Transform Format

Every placed object in the exported JSON is stored as a 9-element array:

```text
[x, y, z, sx, sy, sz, pitch, yaw, roll]
```

Coordinates are in Unreal Engine world-space centimetres. `utils/helpers.py` converts these to Blender metres/radians using the standard UE → Blender coordinate mapping. Rotations are in degrees.
