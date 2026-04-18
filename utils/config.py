"""
config.py
========
Shared constants for the Foxhole map exporter pipeline.

All paths are resolved relative to the repository root (the directory that
contains this ``utils`` package), so every ``N_*.py`` script works no matter
what the current working directory is when it runs.
"""

from pathlib import Path
from typing import Dict, Tuple


# ------------------------------------------------------------------------------
#  Directories and files
# ------------------------------------------------------------------------------

REPO_ROOT           = Path(__file__).resolve().parent.parent

UTILS_DIR           = REPO_ROOT / "utils"
CENTRES_FILE        = UTILS_DIR / "region_centers.json"
CATALOGUE_FILE      = UTILS_DIR / "catalogue.json"
MASK_FILE           = UTILS_DIR / "mask.png"

EXPORT_DIR          = REPO_ROOT / "export"
JSON_DIR            = EXPORT_DIR / "_json"
MESHES_DIR          = EXPORT_DIR / "_meshes"
HEIGHTMAP_DIR       = EXPORT_DIR / "_heightmap"
LAYERS_DIR          = EXPORT_DIR / "_layers"
BLEND_DIR           = EXPORT_DIR / "blend"
SPILL_DIR           = EXPORT_DIR / "blend_spill"

AO_DIR              = EXPORT_DIR / "ao"
HM_LANDSCAPE_DIR    = EXPORT_DIR / "heightmap_landscape"
ID_DIR              = EXPORT_DIR / "id"
WATER_DIR           = EXPORT_DIR / "water"

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

# Number of worker subprocesses used by the "all" modes of
# 2_blend_all.py, 3_blend_spills.py, and 4_render_spills.py.
# Set to 1 to force serial execution in the parent process (no subprocesses).
NUM_WORKERS = 4


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


# ------------------------------------------------------------------------------
#  Category / color constants
# ------------------------------------------------------------------------------

CATEGORY_NAMES: Tuple[str, ...] = (
    "terrain", "water", "rocks", "glaciers", "landscape_meshes", "deep_water",
)

CATEGORY_COLORS: Dict[str, str] = {
    "water":            "#0000FF",
    "terrain":          "#00FF00",
    "rocks":            "#FF0000",
    "glaciers":         "#FFFFFF",
    "landscape_meshes": "#FF00FF",
}

# Terrain category in BGR (OpenCV) order; equals CATEGORY_COLORS["terrain"].
TERRAIN_BGR: Tuple[int, int, int] = (0, 255, 0)

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
    "K":            "#E2DAC7",
    "Grass":        "#FFF9E5",
    "a":            "#FFF9E5",
    "Snow":         "#EFEFEF",
    "SnowRough":    "#EFEFEF",
    "WetSand":      "#FFEBD6",
    "b":            "#FFEBD6",
    "Dirt":         "#FFEBD6",
    "Sand":         "#FFEBD6",
    "Extra02":      "#FFEBD6",
    "Rock":         "#A3A3A3",
    "Stone":        "#A3A3A3",
    "Cobble2":      "#A3A3A3",
    "D":            "#A3A3A3",
    "Ice":          "#F9F2FF",
    "Road":         "#E8CFB4",
    "TownStone":    "#E8CFB4",
    "Highway":      "#E8CFB4",
    "DataLayer__":  "#E8CFB4",
    "E":            "#E8CFB4",
    "G":            "#E8CFB4",
    "MuddyGround":  "#E8CFB4",
    "TrenchDirt":   "#E8CFB4",
}

ID_RECOLOR: Dict[str, str] = {
    "water":            "#A8D5FF",
    "terrain":          "#E2DAC7",
    "rocks":            "#797B89",
    "glaciers":         "#FFFFFF",
    "landscape_meshes": "#D5C7D8",
    "deep_water":       "#C6C0B1"
}
