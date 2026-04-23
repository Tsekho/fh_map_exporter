"""
config.py
========
Shared constants for the Foxhole map exporter pipeline.

All paths are resolved relative to the repository root (the directory that
contains this ``utils`` package), so every ``N_*.py`` script works no matter
what the current working directory is when it runs.
"""

from pathlib import Path
from typing import Dict


# ------------------------------------------------------------------------------
#  Directories and files
# ------------------------------------------------------------------------------

REPO_ROOT           = Path(__file__).resolve().parent.parent


def short_path(p) -> str:
    """Return ``p`` relative to REPO_ROOT when possible, else just its name.

    Used for log output so messages don't dump 100+ char absolute paths.
    """
    try:
        return Path(p).resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        try:
            return Path(p).name
        except Exception:
            return str(p)

UTILS_DIR           = REPO_ROOT / "utils"
CENTRES_FILE        = UTILS_DIR / "region_centers.json"
CATALOGUE_FILE = UTILS_DIR / "catalogue.json"
MASK_FILE           = UTILS_DIR / "mask.png"
FLY_ALERT_PATTERN_FILE = UTILS_DIR / "fly_alert_pattern.png"
RDZ_PATTERN_FILE    = UTILS_DIR / "rdz_pattern.png"

EXPORT_DIR          = REPO_ROOT / "export"
JSON_DIR            = EXPORT_DIR / "_json"
MESHES_DIR          = EXPORT_DIR / "_meshes"
HEIGHTMAP_DIR       = EXPORT_DIR / "_heightmap"
LAYERS_DIR          = EXPORT_DIR / "_layers"
BLEND_DIR           = EXPORT_DIR / "blend"
SPILL_DIR           = EXPORT_DIR / "blend_spill"

AO_DIR              = EXPORT_DIR / "ao"
HM_LANDSCAPE_DIR    = EXPORT_DIR / "heightmap_landscape"
HM_WATER_DIR        = EXPORT_DIR / "heightmap_water"
ID_DIR              = EXPORT_DIR / "id"
ROADS_DIR           = EXPORT_DIR / "roads"
BEACHES_DIR         = EXPORT_DIR / "beaches"
SPLIT_LAYERS_DIR    = EXPORT_DIR / "split_layers"
SVG_DIR             = UTILS_DIR / "svg"
SVG_LAYERS_DIR      = EXPORT_DIR / "svg_layers"

FINAL_DIR           = EXPORT_DIR / "_final"


# ------------------------------------------------------------------------------
#  Exporter.exe build + invocation
# ------------------------------------------------------------------------------

EXPORTER_EXE        = REPO_ROOT / "Exporter.exe"

EXPORTER_PROJECT_DIR = REPO_ROOT / "Exporter"
EXPORTER_PROJECT     = EXPORTER_PROJECT_DIR / "Exporter.csproj"
EXPORTER_TFM         = "net10.0"
EXPORTER_RID         = "win-x64"
EXPORTER_PUBLISH_DIR = (
    EXPORTER_PROJECT_DIR / "bin" / "Release" / EXPORTER_TFM / EXPORTER_RID / "publish"
)

# ------------------------------------------------------------------------------
#  Parallelism
# ------------------------------------------------------------------------------

# Number of worker subprocesses used by
# 2_blend_all.py, 3_blend_spills.py, and 4_render_spills.py.
# Set to 1 to force serial execution in the parent process (no subprocesses).
NUM_WORKERS = 6
NUM_WORKERS_SPILLS = 3


FOXHOLE_PAK = Path(
    r"C:\Program Files (x86)\Steam\steamapps\common"
    r"\Foxhole\War\Content\Paks\War-WindowsNoEditor.pak"
)


# ------------------------------------------------------------------------------
#  Tile geometry
# ------------------------------------------------------------------------------

TILE_SIZE = 2048               # per-region bake resolution (px)
TILE_HALF = TILE_SIZE // 2     # half-extent used when stitching

PIXEL_SIZE_M = 1890.0 / 1776.0 # Blender metres per pixel
DEEP_WATER_DEPTH = 25.0

HM_SPLIT_M = 20.0
FLY_ALERT_MIN_M = 95.0
FLY_ALERT_MAX_M = 100.0

# Gaussian blur applied to shades.png in 5_finalize_exports.py as the last
# step before masking with the terrain coverage alpha. Kernel must be odd.
# Set SHADES_BLUR_KSIZE = 0 to disable the blur.
SHADES_BLUR_KSIZE = 3
SHADES_BLUR_SIGMA = 0.0  # 0 => cv2 derives sigma from kernel size

# AO bake slope shading (see utils/bake.py:render_ao).
# sh = 1 + C * (max(0, N.z)^P - 1); weak applies to most meshes,
# strong applies to objects flagged via pass_index == 1.
AO_SLOPE_POWER_WEAK   = 5.0
AO_SLOPE_C_WEAK       = 0.6
AO_SLOPE_POWER_STRONG = 5.0
AO_SLOPE_C_STRONG     = 0.75

# After the AO bake, pixels with shade >= AO_NEAR_WHITE_CUTOFF get
# clipped to 255 so near-white noise doesn't read as faint shading.
AO_NEAR_WHITE_CUTOFF = 252

# Step-4 (4_render_spills.py) tunables.
#
# Reserved category collection names that the renderer treats specially;
# anything else under <region>/<cat>/ in a spill .blend is a regular
# spill category discovered dynamically at render time.
RESERVED_CATS = ["terrain", "water", "deep_water", "splines"]

# Spill categories that participate in the normal ao/hm/id bakes.
# Any spill category NOT listed here is excluded from those bakes (its
# meshes still live in the .blend and may be used by other passes such
# as split layers). "water", "terrain" and "splines" are reserved
# categories with special handling; they are listed for clarity but
# always flow through the pipeline regardless.
TERRAIN_WHITELIST = [
    "water",
    "terrain",
    "rocks",
    "glaciers",
    "landscape_meshes_brown",
    "landscape_meshes_gray",
    "splines",
]

# Spline categories whose meshes act as terrain for heightmap/AO/id bakes.
TERRAIN_SPLINE_CATS = ["t1_road", "t2_road", "t3_road"]
# Spline categories rendered into the 'roads' and 'beaches' layers.
ROADS_CATS   = ["t1_road", "t2_road", "t3_road"]
BEACHES_CATS = ["beach", ]

# Regions whose terrain mesh sinks below spill meshes in places; verts
# below TERRAIN_CULL_MIN_Z (metres) are deleted before baking so rays
# hit the intended surface.
Z_FIX_REGIONS = ["HomeRegionC", "HomeRegionW"]
TERRAIN_CULL_MIN_Z = -0.5

# Supersampling rate per pixel side for the per-category ID coverage bake
# (4_render_spills.py). Each pixel fires ID_SSAA^2 rays; per-category
# outputs store the fraction that hit that category. Using SSAA replaces
# the old morphological close that plugged 1-px gaps between e.g. dock
# planks; with ID_SSAA >= 2 those gaps resolve smoothly instead.
ID_SSAA = 4

# Roads/beaches coverage bake: supersampling rate per pixel side and
# terrain drop (m). Terrain is temporarily lowered by this amount and
# added as an occluder so underground artefact splines buried deeper
# than the drop get culled while surface roads still read through.
SPLINE_LAYER_SSAA = 4
SPLINE_LAYER_TERRAIN_DROP = 4

# Split layers: each key is a layer name; its value maps spill
# categories -> per-category tint hex. Each layer is rendered as a
# standalone RGBA PNG (one per layer per region) under
# SPLIT_LAYERS_DIR/<layer>/<region>.png via a Cycles AO pass with a
# transparent film: only the layer's target meshes are visible (they
# occlude each other, so buildings of different categories still
# shadow correctly), producing smooth AA'd alpha edges and real AO
# that defines wall geometry. RGB is tinted per-object via the
# category color.
#
# Split-layer membership no longer affects ao/hm/id inclusion; that is
# controlled independently by TERRAIN_WHITELIST above. A category can
# therefore appear in both the whitelist and a split layer if desired.
SPLIT_LAYERS: Dict[str, Dict[str, str]] = {
    "houses":    {"houses":          "#2F2F2F"},
    "ghouses":   {"ghouses":         "#949494"},
    "industry":  {"industry":        "#404F3D"},
    "obstacles": {"obstacles_large": "#2F2F2F",
                  "obstacles_small": "#2F2F2F",
                  "walls_large":     "#2F2F2F",
                  "walls_small":     "#2F2F2F",
                  "sidewalks":       "#78787F",
                  "vehicles":        "#2F2F2F"}
}
# ignored foliage_collision, foliage_invis, foliage_no_collision
# roofs_ghouses, roofs_misc, ignore

# Shader-based silhouette edge darkening for split-layer renders. The
# map is rendered very zoomed out (each building is only 5-10 px
# across), so the last covered pixel of every footprint must read as a
# crisp dark rim. We do this inside the Cycles shader via a Bevel node,
# whose returned normal bends away from vertical near silhouette edges
# (roof/wall transitions). edge = (1 - max(0, bevel_N.z))^POWER *
# STRENGTH is then subtracted from the shade, so the darkening is
# baked into the render and picks up Cycles' native anti-aliasing on
# the boundary pixel.
#   _RADIUS_PX: width of the dark ring in pixels
#   _STRENGTH:  darkening at the very edge (0 = none, 1 = black)
#   _POWER:     falloff shape; >1 concentrates darkening near the edge
# Set SPLIT_LAYER_EDGE_SHADER_STRENGTH to 0.0 to disable.
# Note: the bevel-normal approach was tried first; it failed on
# silhouettes that have no 3D cliff (e.g. pyramid/sloped roofs), so we
# switched to a 2D-footprint distance map sampled per shading sample
# via a TexImage+Window node pair. Cycles' pixel AA then stamps the
# rim onto the genuine mesh edge regardless of 3D topology.
SPLIT_LAYER_EDGE_SHADER_RADIUS_PX: float = 3.0
SPLIT_LAYER_EDGE_SHADER_STRENGTH: float = 0.9
SPLIT_LAYER_EDGE_SHADER_POWER: float = 1.3

# ------------------------------------------------------------------------------
#  Renders
# ------------------------------------------------------------------------------

# Category names used for ID bakes and palette assignment come from
# utils/catalogue.json. CATEGORY_COLORS below supplies the paint
# color for each category (plus the reserved "terrain" category, which is
# built from the heightmap rather than a mesh list).
#
# Reserved categories with special behavior:
#   - "terrain":   built from the 16-bit heightmap (not a whitelist entry)
#   - "water":     whitelist entry; clones are spawned as "deep_water" at
#                  DEEP_WATER_DEPTH metres below
#   - "deep_water": auto-generated occluder clones of water
#
# Any other whitelist key is treated as a regular spill category.

CATEGORY_COLORS: Dict[str, str] = {
    "water":                   "#0000FF",
    "terrain":                 "#00FF00",
    "rocks":                   "#FF0000",
    "glaciers":                "#FFFFFF",
    "landscape_meshes_brown":  "#FF00FF",
    "landscape_meshes_gray":   "#00FFFF",
    "splines":                 "#A2DD43",

    "ghouses":                 "#FFAC46",
    "roofs_ghouses":           "#855B28",
    "houses":                  "#8D46FF",
    "industry":                "#46FF99",
    "roofs_misc":              "#4C268B",
    "obstacles_large":         "#FF1111",
    "obstacles_small":         "#FFFF11",
    "walls_large":             "#A40C0C",
    "walls_small":             "#939310",
    "sidewalks":               "#0EB3B3",
    "vehicles":                "#1111FF",
    "ignore":                  "#111111"
}

# Spline placement: which mesh names belong to which spline category.
# 3_blend_spills.py places these into the focus region's .blend under
# collection 'splines/<category>/'. 4_render_spills.py uses the
# t1_road / t2_road / t3_road / beach categories specially (see below).
SPLINE_CATEGORIES: Dict[str, list] = {
    "t1_road": [
        "Meshes__Environment__Roads__RoadT1Dirt01",
        "Meshes__Environment__Roads__RoadT1Dirt01Snow",
    ],
    "t2_road": [
        "Meshes__Environment__Roads__RoadT2PackedDirt01",
        "Meshes__Environment__Roads__RoadT2PackedDirt01Snow",
    ],
    "t3_road": [
        "Meshes__Environment__Roads__RoadT3Gravel01",
        "Meshes__Environment__Roads__RoadT3Gravel01Snow",
        "Meshes__Environment__Roads__RoadGreatMarch01"
    ],
    "beach": [
        "Engine__Content__EditorLandscapeResources__SplineEditorMesh"
    ],
}

# Per-spline-category colors used by the 'roads' / 'beaches' renders in
# 4_render_spills.py. Entries without a color here fall back to the
# generic 'splines' color.
SPLINE_COLORS: Dict[str, str] = {
    "t1_road":                 "#9DB8B1",
    "t2_road":                 "#AF8D6F",
    "t3_road":                 "#AA706D",
    "beach":                   "#B6A177",
}

# RGB color (hex) used by the dive_alert overlay in 5_finalize_exports.py;
# alpha fades from full at the water surface to 0 at DEEP_WATER_DEPTH.
DIVE_ALERT_COLOR = "#BA759C"

# Per-layer colors used by 5_finalize_exports.py when compositing the
# terrain weightmap layers under export/_layers/<layer>/ into a single
# materials.png. Keys match the layer folder names (case-insensitive
# lookup at consume time). Layers not listed here get a deterministic
# random bright color assigned at runtime.
#
# "_default" is a special entry: the fallback color used for terrain
# pixels that aren't claimed by any layer. Black is reserved for
# non-terrain pixels and must not be used here.
LAYER_COLORS: Dict[str, str] = {
    "K":                       "#CCCBC9",
    "Grass":                   "#CCB5A5",
    "a":                       "#CCB5A5",
    "Snow":                    "#DDDDDD",
    "SnowRough":               "#DDDDDD",
    "WetSand":                 "#C6B19B",
    "b":                       "#C6B19B",
    "Dirt":                    "#C6B19B",
    "Sand":                    "#C6B19B",
    "Extra02":                 "#C6B19B",
    "Rock":                    "#A3A3A3",
    "Stone":                   "#A3A3A3",
    "Cobble2":                 "#A3A3A3",
    "D":                       "#A3A3A3",
    "Ice":                     "#BCBEE2",
    "Road":                    "#B7A491",
    "TownStone":               "#B7A491",
    "Highway":                 "#B7A491",
    "DataLayer__":             "#B7A491",
    "E":                       "#B7A491",
    "G":                       "#B7A491",
    "MuddyGround":             "#B7A491",
    "TrenchDirt":              "#B7A491",
}

ID_RECOLOR: Dict[str, str] = {
    "water":                   "#DFE8ED",
    "terrain":                 "#9495A1",
    "rocks":                   "#727480",
    "glaciers":                "#FFFFFF",
    "deep_water":              "#9B9B9B",
    "landscape_meshes_brown":  "#90746B",
    "landscape_meshes_gray":   "#5A5A5A",
}

# ------------------------------------------------------------------------------
#  SVG layers (4_render_spills.py)
# ------------------------------------------------------------------------------

# Each key defines an output layer rendered via cairosvg into
# SVG_LAYERS_DIR/<layer>/<region>.png, later stitched by
# 5_finalize_exports.py into FINAL_DIR/svg_layers/<layer>.png.
#
# The value is an ORDERED list of SVG categories (subdirectories of
# utils/svg/); placements from earlier categories are drawn first, so
# later categories appear on top. Within a category, <use> elements are
# emitted in the order returned by sorted(glob("*.svg")).
#
# Each utils/svg/<category>/<name>.svg becomes a reusable <symbol
# id="<category>_<name>" overflow="visible"> wrapping the original
# svg's inner content. Placements are pulled from the region's
# export/_json/<region>.json: if <name> matches a blueprint key, the
# blueprint's per-instance "_self" transform is used; otherwise <name>
# is matched as a mesh across "symbols", "groups", and any nested
# blueprint mesh entries.
#
# UE world-space (cm) -> SVG pixel conversion is
# x_px = x_cm * 1776 / 189000 + 1024 (same for y); scale_x/scale_y/yaw
# are applied as-is to the <use> transform.
SVG_LAYERS: Dict[str, list] = {
    "bridges":      ["bridges"],
    "ranges_ai":    ["ranges_ai"],
    "ranges_cg":    ["ranges_cg"],
    "ranges_intel": ["ranges_intel"],
    "ranges_mh":    ["ranges_mh"],
    "ranges_tap":   ["ranges_tap"],
    "wells":        ["wells"],
    "foliage":      ["foliage_low", "foliage_medium", "foliage_tall"],
    "rdz_grace":    ["rdz_grace"],
    # "highlights":   ["stairs", "interiors"],
    # "bridges_aim":  ["bridges_aim"],
    # "drop_pads":    ["drop_pads"],
    # "urban":        ["tiers", "safehouses"],
    # "runways":      ["runways"],
    # "runways_aim":  ["runways_aim"],
    # "garrisons":    ["garrisons"],
}
