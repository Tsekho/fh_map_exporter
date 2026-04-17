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
HM_SIMPLE_DIR       = EXPORT_DIR / "heightmap_simple"
ID_DIR              = EXPORT_DIR / "id"
WATER_DIR           = EXPORT_DIR / "water"
CONTOUR_DIR         = EXPORT_DIR / "contour"

FINAL_DIR           = EXPORT_DIR / "_final"
LAYERS_MASKED_DIR   = EXPORT_DIR / "layers_masked"


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
