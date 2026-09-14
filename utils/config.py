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
CATALOGUE_FILE      = UTILS_DIR / "catalogue.json"
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
BRIDGES_AIM_DIR     = EXPORT_DIR / "bridges_aim"

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

# 4_render_spills.py is capped by RAM, not cores: ~4.9 GB peak per worker.
# Overshooting pages and is slower than running fewer workers.
NUM_WORKERS_SPILLS = 5

# The -svg pass never opens a .blend, so it isn't bound by that ceiling.
NUM_WORKERS_SVG = 12

# Row-pool threads inside each render worker. 0 = cores // NUM_WORKERS_SPILLS.
BAKE_ROW_THREADS = 0

# Reuse one BVH across the bakes that share an object set (heightmap+ID,
# heightmap_water+water coverage). A reused tree keeps the first caller's
# triangle order, so rays landing on coincident faces can tie-break
# differently; set False if ID output must be bit-identical.
BVH_CACHE_REUSE = True

# Queue Cycles renders through a machine-wide lock so workers take the GPU
# in turn instead of thrashing it (utils/gpu_lock.py).
GPU_SERIALIZE_RENDERS = True

# Let Cycles use the CPU alongside the GPU. A net loss with several workers
# in flight: those CPU devices compete with every other worker's raycast bake.
CYCLES_USE_CPU_WITH_GPU = False


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


# ------------------------------------------------------------------------------
#  Per-region Z normalisation (mirror of Exporter/Constants.cs)
# ------------------------------------------------------------------------------

# Z offset (cm) that Exporter.exe ADDS to every exported height and every JSON Z
# when it writes a region, so neighbouring regions meet seamlessly at the hex
# borders. Everything downstream of step 1 therefore carries it.
#
# DUPLICATED from Exporter/Constants.cs :: HeightOffsets. The C# side owns these
# values; this copy exists only so the Python stages can undo the shift. Edit
# both or neither - assert_height_offsets_match() in this module checks them.
#
# Why anything would undo it: the in-game altimeter reads the aircraft's raw
# world Z (the shipping exe feeds RootComponent's world-location Z straight into
# its Altitude* curves - no datum actor, no ground trace), and the game knows
# nothing about this normalisation. So an exported height is
# `raw world Z + offset`, and a stage comparing against an in-game altitude has
# to subtract the region's offset first. Only absolute-altitude consumers care
# (fly_alert); anything measuring a difference of two heights cancels it out.
#
# Keys are lower-cased region names as they appear in the asset path.
HEIGHT_OFFSETS_CM: Dict[str, float] = {
    "shackledchasmhex":    50,
    "clahstrahex":         42,
    "drownedvalehex":      40,
    "endlessshorehex":     50,
    "reaverspasshex":      42,
    "sableporthex":        50,
    "mooringcountyhex":    30,
    "stonecradlehex":      50,
    "weatheredexpansehex": 50,
    "stlicanshelfhex":     42,
    "tempestislandhex":    50,
    "wrestahex":           50,
    "farranaccoasthex":    50,
    "gutterhex":           50,
    "kingscagehex":        50,
    "westgatehex":         50,
    "fishermansrowhex":    50,
    "palantinebermhex":    50,
    "stemalandinghex":     42,
    "oarbreakerhex":       70,
    "lykosislehex":        50,
    "kuurastrandhex":      50,
    "paripeakhex":         50,
    "thefingershex":       50,
    "olaviswakehex":       50,
    "onyxhex":             50,
    "tyrantfoothillshex":  50,
    "pipersenclavehex":    50,
    "homeregionc":         450,
    "homeregionw":         450,
}


def height_offset_cm(region: str) -> float:
    """Z offset (cm) Exporter.exe added to ``region``, or 0.0 if it added none."""
    return HEIGHT_OFFSETS_CM.get(region.lower(), 0.0)


def assert_height_offsets_match() -> None:
    """Raise if HEIGHT_OFFSETS_CM has drifted from Exporter/Constants.cs.

    Cheap guard against the two copies diverging; the C# file is the source of
    truth. Silently returns when the C# source isn't present (release checkouts
    ship Exporter.exe without it).
    """
    import re

    cs = EXPORTER_PROJECT_DIR / "Constants.cs"
    if not cs.is_file():
        return
    text = cs.read_text(encoding="utf-8")
    try:
        block = text.split("HeightOffsets =")[1].split("};")[0]
    except IndexError:
        return
    theirs = {m.group(1).lower(): float(m.group(2))
              for m in re.finditer(r'\["([^"]+)"\]\s*=\s*([-\d.]+)', block)}
    ours = {k.lower(): float(v) for k, v in HEIGHT_OFFSETS_CM.items()}
    if theirs != ours:
        only_cs = sorted(set(theirs) - set(ours))
        only_py = sorted(set(ours) - set(theirs))
        differ = sorted(k for k in set(theirs) & set(ours)
                        if theirs[k] != ours[k])
        raise ValueError(
            "HEIGHT_OFFSETS_CM is out of sync with Exporter/Constants.cs "
            f"(only in Constants.cs: {only_cs}; only in config.py: {only_py}; "
            f"different values: {[(k, theirs[k], ours[k]) for k in differ]})"
        )

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
AO_SLOPE_POWER_SL     = 5.0
AO_SLOPE_C_SL         = 0.6

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
    "houses":    {"houses":          "#232323"},
    "ghouses":   {"ghouses":         "#A6AEBE"},
    "industry":  {"industry":        "#30533A"},
    "obstacles": {"obstacles_large": "#232323",
                  "obstacles_small": "#232323",
                  "walls_large":     "#232323",
                  "walls_small":     "#232323",
                  "sidewalks":       "#78787F",
                  "vehicles":        "#7E78A7"}
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

# Minimum HSV value (brightness) for split-layer renders. Any visible pixel
# (alpha > 0) whose V channel is below this threshold is lifted to this value,
# preventing over-dark outputs while leaving transparent pixels untouched.
MIN_SPLIT_LAYER_VALUE: int = 100

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
        "Meshes__Environment__Roads__RoadGreatMarch01",
        "Meshes__Environment__Roads__RoadGreatMarch01Snow",
    ],
    "beach": [
        "Engine__Content__EditorLandscapeResources__SplineEditorMesh",
    ],
}

# Per-spline-category colors used by the 'roads' / 'beaches' renders in
# 4_render_spills.py. Entries without a color here fall back to the
# generic 'splines' color.
SPLINE_COLORS: Dict[str, str] = {
    "t1_road":                 "#A2B4C1",
    "t2_road":                 "#C1AD83",
    "t3_road":                 "#B48780",
    "beach":                   "#B6A177",
}

# Depth-graded dive_alert overlay in 5_finalize_exports.py. Each entry maps
# a submersion-depth range [start_m, end_m) to a colour gradient: the overlay
# colour fades linearly from the first hex at start_m to the second hex at
# end_m. Hex accepts #RRGGBB (alpha 255) or #RRGGBBAA. Ranges must be listed
# ascending and contiguous; depths past the last range render transparent.
# The overlay is gated by water coverage, replacing the old
# obstacles/landscape 2-split.
DIVE_ALERT_GRADIENT = [
    (0.0,  1.4, "#86B29D88", "#86B29D88"), # Walkable
    (1.4,  1.7, "#62827288", "#62827288"), # Truck-drivable
    (1.7,  7.0, "#BA759C88", "#BA759C44"), # Surface ships might beach 
    (7.0, 14.2, "#6980EF88", "#6980EF44"), # Submarines might show on intel
]

# Gaussian blur smoothing dive_alert band transitions. Applied as a
# water-masked normalized blur: only pixels inside the water mask
# contribute and the result is renormalized by the blurred mask, so the
# overlay never bleeds onto land and its alpha doesn't weaken at the
# shoreline (the land->water edge stays sharp via the water_cov gate).
# Kernel must be odd; 0 disables.
DIVE_ALERT_BLUR_KSIZE = 3
DIVE_ALERT_BLUR_SIGMA = 0.0  # 0 => cv2 derives sigma from kernel size

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
    "glaciers":                "#E0E0E0",
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
    "bridges":       ["bridges"],
    "ranges_ai":     ["ranges_ai"],
    "ranges_cg":     ["ranges_cg"],
    "ranges_aag":    ["ranges_aag"],
    "ranges_intel":  ["ranges_intel"],
    "ranges_mh":     ["ranges_mh"],
    "ranges_tap":    ["ranges_tap"],
    "wells":         ["wells"],
    "foliage":       ["foliage_low", "foliage_medium", "foliage_tall"],
    "foliage_invis": ["foliage_invis"],
    "rdz_grace":     ["rdz_grace"],
    "highlights":    ["stairs", "interiors"],
    "drop_pads":     ["drop_pads"],
    "urban":         ["tiers", "safehouses"],
    "runways":       ["runways"],
    "runways_aim":   ["runways_aim"],
    "garrisons":     ["garrisons"],
#    "ladders":       ["ladders"],
}

# ------------------------------------------------------------------------------
#  Bridge aim lines (procedural; utils/svg_render.render_bridges_aim_layer)
# ------------------------------------------------------------------------------
#
# The "bridges_aim" layer is rendered procedurally into its own
# BRIDGES_AIM_DIR/<region>.png (not part of SVG_LAYERS) and is stitched by
# 5_finalize_exports.py into FINAL_DIR/assembly/bridges_aim.png. Instead of
# stamping the static utils/svg/bridges_aim/*.svg symbols, 4_render_spills.py
# computes the aim lines per region. Every bridge gets two sockets emanating from its centre
# along the passage axis (the bridge's local +x / -x). Each socket either:
#   * snaps to a facing socket on a nearby bridge, forming a smooth curve
#     that links the two crossings (so the line never crosses another
#     bridge), or
#   * extends straight outward and is truncated where it meets land, which
#     replaces the old water-erosion gate in 5_finalize_exports.py.
#
# All values are in tile pixels (1 px == PIXEL_SIZE_M metres).

# Blueprint class names (as they appear in export/_json/<region>.json) that
# are treated as bridges. Each placement's "_self" transform spawns one aim
# line pair. This replaces the old utils/svg/bridges_aim/*.svg registry.
BRIDGES_AIM_BLUEPRINTS = [
    "BPDrawbridgeA_C",
    "BPDrawbridgeB_C",
    "BPDrawbridgeC_C",
    "BPTrainBridgeA_C",
    "BPTrainBridgeC_C",
]

# Half-length of each aim line, measured from the bridge centre (matches the
# 100-unit reach of the legacy utils/svg/bridges_aim/*.svg lines).
BRIDGES_AIM_LENGTH_PX = 90.0
# Gap around the bridge centre where no line is drawn (legacy 5-unit gap).
BRIDGES_AIM_GAP_PX = 10.0
# Stroke width / colour of the rasterised aim lines.
BRIDGES_AIM_STROKE_PX = 1.2
BRIDGES_AIM_COLOR = "#FFFFFF"

# Snapping: two sockets on *different* bridges snap together when the two
# bridge centres are within this distance AND the sockets face one another
# (each points roughly toward the other bridge). The closest eligible pair
# is matched first (greedy).
BRIDGES_AIM_SNAP_DIST_PX = 200.0
# Minimum dot(socket_dir, unit_vector_to_other_bridge) for a pair to count
# as "facing" (1.0 = perfectly head-on, 0.5 ~= within 60 degrees).
BRIDGES_AIM_FACE_MIN_DOT = 0.7

# Both unpaired (outward) sockets and snapped pairs are routed through the
# NAVIGABLE water mask: the water coverage eroded by MIN_CLEARANCE_PX, so
# every cell of the route sits at least that far from any shore (the ship is
# modelled as a ball of that radius rolling down the channel). Navigability
# is read from the export/id/water coverage field (water when coverage >=
# BRIDGES_AIM_WATER_THRESH); the bridge deck reads as non-water and so acts
# as a wall -- routes never cross a bridge, and the two sockets of one bridge
# stay on opposite banks. The shortest grid route is then string-pulled into a
# taut polyline (it goes straight as far as the channel allows and only turns
# where an obstacle forces it) and rounded with a centripetal spline.
#
# Clearance guarantee: every control point keeps MIN_CLEARANCE_PX from shore,
# and the spline between them is sampled and repaired (by subdividing) until
# it, too, stays clear -- EXCEPT across spans shorter than MIN_CTRL_SPACING_PX,
# where smoothness wins over clearance (the curve is left alone rather than
# studded with control points every few pixels).
#
#   WATER_THRESH         coverage [0..255] at/above which a pixel is water
#   MIN_DEPTH_M          minimum submersion depth (m; water heightmap minus
#                        landscape heightmap) for a water pixel to count as
#                        navigable -- freighters run aground in shallower
#                        water. 0 disables the depth gate (coverage only).
#   MIN_CLEARANCE_PX     erosion radius / min distance-to-shore (px) kept by
#                        every control point and the curve spanning them
#   SNAP_TO_WATER_PX     max radius used to pull a bridge gap onto navigable
#                        water before routing (the gap sits on the deck)
#   CENTER_BIAS          OFF (0) by default. A small cost penalty for routing
#                        close to shore, nudging the line toward deeper water.
#                        Keep it low: large values make a long detour through
#                        open water cheaper than the direct channel, which
#                        produces wild U-shapes, and chase the river's medial
#                        axis, which makes the line squiggle. Taut routing
#                        (string-pulling) already keeps the line clear, so
#                        this is only a gentle optional nudge.
#   CHANNEL_PREF_PX      distance-to-shore (px) at/above which water is "deep
#                        enough"; no centring penalty is applied past it.
#   MIN_CTRL_SPACING_PX  control points closer than this are never split
#                        further to chase clearance (see exception above)
#   CURVE_CHECK_STEP_PX  spacing of samples used when testing curve clearance
#   REFINE_PASSES        max subdivision passes enforcing curve clearance
#   LATERAL_BIAS         penalty per pixel of sideways offset when picking an
#                        unpaired socket's endpoint (score = along-axis reach
#                        minus LATERAL_BIAS * |perpendicular offset|). The
#                        Dijkstra reach budget is measured in octile grid
#                        distance, so in open water the farthest-projection
#                        cell snaps to the nearest of the 8 compass directions
#                        instead of the socket axis; this penalty makes the
#                        on-axis cell win unless the channel genuinely bends
#                        the route. 0 restores raw projection.
#   END_DEPTH_BIAS       penalty per pixel that an endpoint's distance-to-
#                        shore falls below CHANNEL_PREF_PX. Without it the
#                        line ends wherever a few extra pixels of along-axis
#                        reach exist -- often a pocket right at the clearance
#                        boundary, which puts a tiny end-hook aiming the line
#                        into the shore. In a uniformly narrow river every
#                        cell is penalised alike, so reach is unaffected.
#   END_TRIM_PX          after routing, the tail of an unpaired line is cut
#                        back to the deepest cell within this many path-px of
#                        the end. Even with END_DEPTH_BIAS the along-axis
#                        reach can pay for a tail that descends toward shore;
#                        the spline's end tangent then points the tip at the
#                        bank. Trimming guarantees the line never *ends* on a
#                        shoreward approach. 0 disables.
#   PAIR_MAX_DETOUR      a snapped pair's water route is accepted only when its
#                        length is at most this multiple of the straight gap-
#                        to-gap distance. When two close, slightly misaligned
#                        bridges have their direct channel pinched shut by the
#                        erosion, A* would otherwise loop the long way around
#                        the far end of the other bridge's deck -- the U-shape
#                        that crosses back over it. Past this ratio the route
#                        is rejected in favour of a short, direct connector.
#   RELAX_CLEARANCE_PX   fallback clearances (px, descending) retried when a
#                        route fails at MIN_CLEARANCE_PX. Real channels pinch
#                        below the nominal ship radius -- under a bridge, in a
#                        narrow cut -- and a route hugging a slightly tighter
#                        band is still on water, whereas the unrouted direct
#                        connector is a straight line drawn over whatever lies
#                        between the two gaps. Relaxing beats not routing.
#   RELAXED_MAX_BOW_PX / RELAXED_MAX_BOW_FRAC
#                        a route that exists only at a relaxed clearance is
#                        rejected when it bows off the straight gap-to-gap
#                        chord by more than both of these (absolute px and
#                        fraction of the span) AND a direct connector would
#                        lie on water. Scraping through the last marginal
#                        pockets of an almost-empty navigable mask can find a
#                        path that detours wildly around nothing -- an L or a
#                        hook between two bridges that plainly face each
#                        other. Where the straight line is on water it is the
#                        better drawing. Both thresholds must trip: the
#                        fraction alone fires on near-zero spans, where a
#                        sub-pixel bow is a large ratio and means nothing.
#                        A full-clearance route is always trusted, however it
#                        bends -- that bend is a real channel.
#   DIRECT_MAX_SPAN_PX   last resort, when no clearance yields a route: a
#                        straight/bezier connector is drawn between two gaps
#                        only if they are at most this far apart (near-
#                        coincident placements of one crossing, where the
#                        connector hides under the decks) or the connector is
#                        verified to stay on water. Longer unverifiable spans
#                        draw nothing rather than a line across terrain.
#   ENTRY_MAX_ANGLE_DEG  hard cap on the angle between a socket's axis and the
#                        direction to its water entry cell. A ship leaves a
#                        bridge along the passage, so a line that exits the
#                        deck sideways is wrong however close the water is;
#                        past this angle the socket is left undrawn.
#   LEAD_IN_PX           straight run (px) along the socket axis before the
#                        route is allowed to bend. Control points inside this
#                        radius are dropped and an on-axis point is inserted,
#                        so the curve leaves the bridge on the passage axis
#                        instead of turning immediately off the deck.
#   LEAD_IN_MIN_DEV_DEG  the lead-in engages only when the route's natural
#                        departure is at least this far off the passage axis.
#                        Two bridges facing each other are rarely exactly
#                        collinear, and forcing a straight run out of each
#                        slightly-off axis bends the line one way and then
#                        the other -- an S across what should be a smooth
#                        link. Below this tolerance the departure is already
#                        plausible (ENTRY_MAX_ANGLE_DEG bounds it), so the
#                        curve is left smooth and nothing is inserted.
#   LEAD_IN_MAX_SPAN_FRAC
#                        cap on the lead-in as a fraction of the route it is
#                        attached to. A fixed LEAD_IN_PX is a small nudge on
#                        a long channel but most of a short bridge-to-bridge
#                        connector, where it leaves no room to blend back and
#                        turns a correction into a kink.
BRIDGES_AIM_WATER_THRESH = 200
BRIDGES_AIM_MIN_DEPTH_M = 4.0
BRIDGES_AIM_MIN_CLEARANCE_PX = 10.0
BRIDGES_AIM_SNAP_TO_WATER_PX = 30.0
BRIDGES_AIM_CENTER_BIAS = 3.0
BRIDGES_AIM_CHANNEL_PREF_PX = 30.0
BRIDGES_AIM_MIN_CTRL_SPACING_PX = 9.0
BRIDGES_AIM_CURVE_CHECK_STEP_PX = 1.5
BRIDGES_AIM_REFINE_PASSES = 24
BRIDGES_AIM_LATERAL_BIAS = 1.5
BRIDGES_AIM_END_DEPTH_BIAS = 1.0
BRIDGES_AIM_END_TRIM_PX = 25.0
BRIDGES_AIM_PAIR_MAX_DETOUR = 1.6
BRIDGES_AIM_RELAX_CLEARANCE_PX = (8.0, 6.0, 4.0, 2.0)
BRIDGES_AIM_DIRECT_MAX_SPAN_PX = 15.0
BRIDGES_AIM_RELAXED_MAX_BOW_PX = 4.0
BRIDGES_AIM_RELAXED_MAX_BOW_FRAC = 0.25
BRIDGES_AIM_ENTRY_MAX_ANGLE_DEG = 35.0
BRIDGES_AIM_LEAD_IN_PX = 12.0
BRIDGES_AIM_LEAD_IN_MIN_DEV_DEG = 25.0
BRIDGES_AIM_LEAD_IN_MAX_SPAN_FRAC = 0.25
