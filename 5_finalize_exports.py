"""
5_finalize_exports.py
========
Consolidate every per-region bake into a single world-sized export directory.

Only stitched world-sized PNGs are written; no per-region intermediates
are produced by this script.

Steps:
    1. Stitch the top-level bakes (ao / id / water) emitted by step 4.
       ao / id / water are written as BGRA where alpha = 0 outside the
       stitched hex mask (pixels not covered by any region tile are
       fully transparent).
    2. Derive world-sized products from heightmap_landscape (int16,
       0 m == 32768 raw, 1 raw == 0.01 m, raw 0 == void). Each region's
       tile is processed in memory and pasted onto the world canvas via
       np.maximum:
         - heightmap_highs: 1 shade == +0.5 m above the 10 m split.
                value = clip(round((meters - 10) * 2), 0, 255) for meters >= 10
                value = 0 for meters < 10 or void
         - heightmap_lows: 1 shade == -0.5 m below the 10 m split.
                value = clip(round((10 - meters) * 2), 0, 255) for meters <= 10
                value = 0 for meters > 10 or void
         - curvature_peaks: positive half of the 2D Laplacian of elevation.
                value = clip(round(laplacian * CURV_SCALE), 0, 255) > 0
         - curvature_dips: negative half of the 2D Laplacian of elevation.
                value = clip(round(-laplacian * CURV_SCALE), 0, 255) > 0
         - slopes: gradient magnitude (0 = flat, 255 = 45 deg+ slope).
                alpha = clip(round(|grad| * SLOPE_SCALE), 0, 64)
                emitted as RGBA solid black overlay, masked to terrain.
         - fly_alert: alert overlay at elevation. Textured via
                utils/fly_alert_pattern.png: pattern RGB passes through
                directly, and the elevation ramp (0 at 90 m, 255 at
                100 m+) multiplies the pattern's alpha as a coefficient.
         - contour: stepwise black lines where (hm // 250) step increments
                across a 4-neighbor boundary. Emitted as RGBA (solid
                black, full alpha where contour is drawn), masked to
                terrain pixels only.
       Curvature is m/m^2; slope magnitude is dimensionless. Pixels
       adjacent to void are masked out in curvature/slope outputs.
    3. Two recolored BGRA products are emitted from the stitched
       id / water canvases:
         - terrain_recolor: id pixels remapped via ID_RECOLOR for every
                category except water (water becomes transparent).
         - water_recolor: solid ID_RECOLOR["water"] wherever water.png > 0,
                transparent elsewhere.
    4. Stitch every folder under export/_layers/<layer>/ and composite
       them into a single export/_final/shades.png. Each layer is
       assigned a color (looked up in LAYER_COLORS by name, otherwise
       a deterministic random bright color). Layers are composited by
       "alpha betting": at every pixel, the layer with the highest
       intensity wins and paints that pixel with its color. Layers are
       masked to land pixels only (terrain id with water excluded);
       pixels not claimed by any layer stay fully transparent. The
       final image is saved as BGRA. The layer -> color mapping is
       written to shades_palette.json.

Output tree:
    export/
        _final/
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

Usage:
    python 5_finalize_exports.py
"""

import colorsys
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from utils.config import (
    AO_DIR,
    CATEGORY_COLORS,
    CENTRES_FILE,
    FINAL_DIR,
    HM_LANDSCAPE_DIR,
    ID_DIR,
    ID_RECOLOR,
    LAYER_COLORS,
    TERRAIN_BGR,
    LAYERS_DIR,
    MASK_FILE,
    PIXEL_SIZE_M,
    TILE_HALF,
    UTILS_DIR,
    WATER_DIR,
)

FLY_ALERT_PATTERN_FILE = UTILS_DIR / "fly_alert_pattern.png"


# Scale applied to the 2D Laplacian of elevation (m/m^2) before clipping
# to 8-bit. With CURV_SCALE = 50, one shade == 0.02 m/m^2, saturating at
# ~5.1 m/m^2.
CURV_SCALE = 50.0

# Scale applied to the gradient magnitude (rise/run) for slope alpha.
# With SLOPE_SCALE = 255, a slope of 1.0 (45 deg) saturates to 255
# (later clipped to 64 for the final overlay).
SLOPE_SCALE = 255.0

# Elevation (meters) at which the highs/lows split happens. Terrain
# below this elevation goes into heightmap_lows, above into highs.
HM_SPLIT_M = 10.0

# Fly-alert elevation ramp (meters). Alpha = 0 at/below FLY_ALERT_MIN_M
# and 255 at/above FLY_ALERT_MAX_M.
FLY_ALERT_MIN_M = 90.0
FLY_ALERT_MAX_M = 100.0


# ------------------------------------------------------------------------------
#  Shared helpers
# ------------------------------------------------------------------------------


def load_centres() -> Dict[str, Tuple[int, int]]:
    with open(CENTRES_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: (int(v[0]), int(v[1])) for k, v in raw.items()}


def load_mask() -> np.ndarray:
    img = cv2.imread(str(MASK_FILE), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"mask not found: {MASK_FILE}")
    return (img > 0).astype(np.uint8)


def canvas_size(centres: Dict[str, Tuple[int, int]]) -> Tuple[int, int]:
    max_y = max_x = 0
    for cx, cy in centres.values():
        max_y = max(max_y, cy + TILE_HALF)
        max_x = max(max_x, cx + TILE_HALF)
    return max_y, max_x


def _build_tile_map(src_dir: Path) -> Dict[str, Path]:
    if not src_dir.is_dir():
        return {}
    return {p.stem.lower(): p for p in src_dir.glob("*.png")}


def _apply_hex_mask(tile: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero everything outside the hex mask, preserving dtype + channels."""
    if tile.ndim == 2:
        return tile * mask.astype(tile.dtype)
    return tile * mask.astype(tile.dtype)[:, :, None]


def _hex_to_bgr(hex_str: str) -> Tuple[int, int, int]:
    s = hex_str.lstrip("#")
    r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    return (b, g, r)


def _random_bright_bgr(used: set, rng: random.Random) -> Tuple[int, int, int]:
    for _ in range(256):
        h = rng.random()
        s = rng.uniform(0.75, 1.0)
        v = rng.uniform(0.85, 1.0)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        bgr = (int(round(b * 255)), int(round(g * 255)), int(round(r * 255)))
        if bgr not in used and bgr != (0, 0, 0):
            return bgr
    return (255, 255, 255)


def _assign_layer_color(
    name: str,
    palette: Dict[str, str],
    used: set,
    rng: random.Random,
) -> Tuple[int, int, int]:
    hex_str = palette.get(name) or palette.get(name.lower())
    if hex_str is not None:
        bgr = _hex_to_bgr(hex_str)
    else:
        bgr = _random_bright_bgr(used, rng)
    used.add(bgr)
    return bgr


def _compute_world_alpha(
    centres: Dict[str, Tuple[int, int]],
    mask: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """World-sized 8-bit alpha: 255 inside any hex mask, 0 elsewhere."""
    alpha = np.zeros((height, width), dtype=np.uint8)
    mask_u8 = (mask.astype(np.uint8) * 255)
    for cx, cy in centres.values():
        y1, y2 = cy - TILE_HALF, cy + TILE_HALF
        x1, x2 = cx - TILE_HALF, cx + TILE_HALF
        dst = alpha[y1:y2, x1:x2]
        np.maximum(dst, mask_u8, out=dst)
    return alpha


def _write_with_alpha(
    canvas: np.ndarray,
    alpha: np.ndarray,
    out_path: Path,
) -> None:
    if canvas.ndim == 2:
        bgra = np.zeros((*canvas.shape, 4), dtype=np.uint8)
        bgra[..., 0] = canvas
        bgra[..., 1] = canvas
        bgra[..., 2] = canvas
        bgra[..., 3] = alpha
    elif canvas.shape[2] == 3:
        bgra = np.dstack([canvas, alpha])
    else:
        bgra = canvas.copy()
        bgra[..., 3] = np.minimum(bgra[..., 3], alpha)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgra)


def _build_recolor(
    id_canvas: np.ndarray,
    recolor: Dict[str, str],
    category_colors: Dict[str, str],
    exclude: Tuple[str, ...] = (),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recolor an id map canvas using *recolor*. Returns
    ``(bgr_canvas, opaque_mask)`` where ``opaque_mask`` is True wherever
    some category painted a pixel. ``deep_water`` is handled specially:
    step 4 paints it as pure black so pure-black id pixels map to its
    recolor entry.
    """
    h, w = id_canvas.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    opaque = np.zeros((h, w), dtype=bool)

    pairs: list = []
    for cat, src_hex in category_colors.items():
        if cat in exclude:
            continue
        dst_hex = recolor.get(cat)
        if dst_hex is None:
            continue
        pairs.append((_hex_to_bgr(src_hex), _hex_to_bgr(dst_hex)))

    if "deep_water" not in exclude:
        dw_hex = recolor.get("deep_water")
        if dw_hex is not None:
            pairs.append(((0, 0, 0), _hex_to_bgr(dw_hex)))

    for src_bgr, dst_bgr in pairs:
        m = (
            (id_canvas[..., 0] == src_bgr[0])
            & (id_canvas[..., 1] == src_bgr[1])
            & (id_canvas[..., 2] == src_bgr[2])
        )
        if m.any():
            out[m] = dst_bgr
            opaque |= m
    return out, opaque


# ------------------------------------------------------------------------------
#  Generic stitcher
# ------------------------------------------------------------------------------


def stitch(
    tile_map: Dict[str, Path],
    centres: Dict[str, Tuple[int, int]],
    mask: np.ndarray,
    height: int,
    width: int,
    *,
    channels: int,
    dtype: np.dtype,
    read_flag: int,
) -> np.ndarray:
    """
    Paste every tile onto a world canvas of (height, width) and return it.
    *channels* = 1 (grayscale) / 3 (BGR) / 4 (BGRA).
    Overlap resolved with np.maximum.
    """
    shape = (height, width) if channels == 1 else (height, width, channels)
    canvas = np.zeros(shape, dtype=dtype)
    total = len(centres)
    placed = 0

    for i, (name, (cx, cy)) in enumerate(centres.items(), 1):
        print(f"  {i}/{total}", end="\r")
        tile_path = tile_map.get(name.lower())
        if tile_path is None:
            continue

        tile = cv2.imread(str(tile_path), read_flag)
        if tile is None:
            print(f"\n  [WARN] unreadable tile: {tile_path}")
            continue

        if channels == 1 and tile.ndim == 3:
            tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        elif channels == 3 and tile.ndim == 2:
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        elif channels == 4 and tile.ndim == 2:
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGRA)
        elif channels == 4 and tile.ndim == 3 and tile.shape[2] == 3:
            tile = cv2.cvtColor(tile, cv2.COLOR_BGR2BGRA)

        if tile.dtype != dtype:
            tile = tile.astype(dtype)

        tile = _apply_hex_mask(tile, mask)

        y1, y2 = cy - TILE_HALF, cy + TILE_HALF
        x1, x2 = cx - TILE_HALF, cx + TILE_HALF
        dst = canvas[y1:y2, x1:x2]
        np.maximum(dst, tile, out=dst)
        placed += 1

    print(f"  {total}/{total}  ({placed} tiles placed)")
    return canvas


# ------------------------------------------------------------------------------
#  Heightmap-derived world products (in-memory, no per-region writes)
# ------------------------------------------------------------------------------


def build_heightmap_products(
    centres: Dict[str, Tuple[int, int]],
    mask: np.ndarray,
    height: int,
    width: int,
) -> Dict[str, np.ndarray]:
    """
    Iterate every heightmap_landscape tile, derive highs/lows/curvature/
    slopes/fly_alert/contour in memory, and paste onto world canvases.

    Contour uses each region's id.png to restrict to terrain pixels.

    Returns a dict of world-sized canvases:
        highs, lows, peaks, dips, slopes, fly_alert, contour_alpha
    All are 8-bit grayscale; contour_alpha is the alpha channel for
    the contour overlay (0 or 255).
    """
    highs = np.zeros((height, width), dtype=np.uint8)
    lows = np.zeros((height, width), dtype=np.uint8)
    peaks = np.zeros((height, width), dtype=np.uint8)
    dips = np.zeros((height, width), dtype=np.uint8)
    slopes = np.zeros((height, width), dtype=np.uint8)
    fly_alert = np.zeros((height, width), dtype=np.uint8)
    contour_alpha = np.zeros((height, width), dtype=np.uint8)

    if not HM_LANDSCAPE_DIR.is_dir():
        print(f"  [WARN] {HM_LANDSCAPE_DIR} not found; "
              f"skipping heightmap derivatives")
        return {
            "highs": highs, "lows": lows, "peaks": peaks, "dips": dips,
            "slopes": slopes, "fly_alert": fly_alert,
            "contour_alpha": contour_alpha,
        }

    lap_kernel = np.array(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    px2 = float(PIXEL_SIZE_M) ** 2

    split_raw = int(round(HM_SPLIT_M * 100.0)) + 32768  # raw threshold value
    mask_bool = mask.astype(bool)

    hm_map = _build_tile_map(HM_LANDSCAPE_DIR)
    id_map = _build_tile_map(ID_DIR)
    total = len(centres)
    placed = 0
    print(f"\n=== heightmap derivatives ({total} regions) ===")

    for i, (name, (cx, cy)) in enumerate(centres.items(), 1):
        print(f"  {i}/{total}", end="\r")
        p = hm_map.get(name.lower())
        if p is None:
            continue

        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if raw is None:
            print(f"\n  [WARN] unreadable: {p}")
            continue
        if raw.dtype != np.uint16:
            raw = raw.astype(np.uint16)
        placed += 1

        void = raw == 0
        meters = (raw.astype(np.float32) - 32768.0) / 100.0

        # highs / lows split at HM_SPLIT_M -----------------------------
        delta = meters - HM_SPLIT_M
        tile_highs = np.clip(np.round(delta * 2.0), 0, 255).astype(np.uint8)
        tile_lows = np.clip(np.round(-delta * 2.0), 0, 255).astype(np.uint8)
        tile_highs[void] = 0
        tile_lows[void] = 0

        # curvature ----------------------------------------------------
        elev = np.where(void, 0.0, meters).astype(np.float32)
        lap = cv2.filter2D(elev, cv2.CV_32F, lap_kernel,
                           borderType=cv2.BORDER_REPLICATE) / px2
        void_u8 = void.astype(np.uint8)
        void_neighborhood = cv2.dilate(
            void_u8, np.ones((3, 3), dtype=np.uint8), iterations=1,
        ).astype(bool)
        lap[void_neighborhood] = 0.0
        scaled_lap = lap * CURV_SCALE
        tile_peaks = np.clip(np.round(scaled_lap), 0, 255).astype(np.uint8)
        tile_dips = np.clip(np.round(-scaled_lap), 0, 255).astype(np.uint8)

        # slopes -------------------------------------------------------
        gx = cv2.Sobel(elev, cv2.CV_32F, 1, 0, ksize=3,
                       borderType=cv2.BORDER_REPLICATE) / (8.0 * PIXEL_SIZE_M)
        gy = cv2.Sobel(elev, cv2.CV_32F, 0, 1, ksize=3,
                       borderType=cv2.BORDER_REPLICATE) / (8.0 * PIXEL_SIZE_M)
        slope_mag = np.hypot(gx, gy)
        tile_slopes = np.clip(np.round(slope_mag * SLOPE_SCALE),
                              0, 255).astype(np.uint8)
        tile_slopes[void_neighborhood] = 0

        # fly_alert ----------------------------------------------------
        denom = max(FLY_ALERT_MAX_M - FLY_ALERT_MIN_M, 1e-6)
        fly_ratio = (meters - FLY_ALERT_MIN_M) / denom
        tile_fly = np.clip(np.round(fly_ratio * 255.0), 0, 255).astype(np.uint8)
        tile_fly[void] = 0

        # contour: step(px) exactly one greater than step of a 4-neighbor
        step = (raw // 250).astype(np.int32)
        contour = np.zeros(step.shape, dtype=bool)
        contour[1:, :]  |= (step[1:, :]  - step[:-1, :]) == 1
        contour[:-1, :] |= (step[:-1, :] - step[1:,  :]) == 1
        contour[:, 1:]  |= (step[:, 1:]  - step[:, :-1]) == 1
        contour[:, :-1] |= (step[:, :-1] - step[:, 1:])  == 1
        contour &= ~void

        # Restrict contour to terrain pixels using the id tile
        id_path = id_map.get(name.lower())
        if id_path is not None:
            id_tile = cv2.imread(str(id_path), cv2.IMREAD_COLOR)
            if id_tile is not None:
                terrain = (
                    (id_tile[..., 0] == TERRAIN_BGR[0])
                    & (id_tile[..., 1] == TERRAIN_BGR[1])
                    & (id_tile[..., 2] == TERRAIN_BGR[2])
                )
                contour &= terrain
        tile_contour = np.where(contour, 255, 0).astype(np.uint8)

        # Apply hex mask + paste to world canvases ---------------------
        for tile_arr, world in (
            (tile_highs, highs),
            (tile_lows, lows),
            (tile_peaks, peaks),
            (tile_dips, dips),
            (tile_slopes, slopes),
            (tile_fly, fly_alert),
            (tile_contour, contour_alpha),
        ):
            tile_arr[~mask_bool] = 0
            y1, y2 = cy - TILE_HALF, cy + TILE_HALF
            x1, x2 = cx - TILE_HALF, cx + TILE_HALF
            dst = world[y1:y2, x1:x2]
            np.maximum(dst, tile_arr, out=dst)

    print(f"  {total}/{total}  ({placed} tiles placed)")
    return {
        "highs": highs, "lows": lows, "peaks": peaks, "dips": dips,
        "slopes": slopes, "fly_alert": fly_alert,
        "contour_alpha": contour_alpha,
    }


# ------------------------------------------------------------------------------
#  Entry point
# ------------------------------------------------------------------------------


def main() -> int:
    t0 = time.time()

    try:
        centres = load_centres()
        mask = load_mask()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    height, width = canvas_size(centres)
    print(f"=== Finalizing exports ({width}x{height} px, {len(centres)} regions) ===")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    # World-sized alpha: 255 inside any region tile's hex mask, 0 outside.
    world_alpha = _compute_world_alpha(centres, mask, height, width)

    # 1) Stitch the step-4 bakes (ao / id / water) -------------------------
    def _stitch_then_alpha(
        label: str,
        src_dir: Path,
        *,
        channels: int,
        read_flag: int,
        out_name: str,
    ) -> np.ndarray | None:
        if not src_dir.is_dir():
            print(f"  [WARN] {src_dir} not found; skipping {label}")
            return None
        print(f"\n=== stitching {label} ===")
        tile_map = _build_tile_map(src_dir)
        canvas = stitch(tile_map, centres, mask, height, width,
                        channels=channels, dtype=np.uint8,
                        read_flag=read_flag)
        out_path = FINAL_DIR / out_name
        _write_with_alpha(canvas, world_alpha, out_path)
        print(f"  written: {out_path}")
        return canvas

    id_canvas = _stitch_then_alpha(
        "id", ID_DIR, channels=3, read_flag=cv2.IMREAD_COLOR,
        out_name="id.png",
    )
    _stitch_then_alpha(
        "ao", AO_DIR, channels=1, read_flag=cv2.IMREAD_GRAYSCALE,
        out_name="ao.png",
    )
    water_canvas = _stitch_then_alpha(
        "water", WATER_DIR,
        channels=1, read_flag=cv2.IMREAD_GRAYSCALE,
        out_name="water.png",
    )

    # terrain mask (world-sized) used for contour/slopes/fly_alert masking
    if id_canvas is None:
        print("[WARN] id canvas unavailable; terrain masking skipped")
        world_terrain = None
    else:
        world_terrain = (
            (id_canvas[..., 0] == TERRAIN_BGR[0])
            & (id_canvas[..., 1] == TERRAIN_BGR[1])
            & (id_canvas[..., 2] == TERRAIN_BGR[2])
        )

    # 2) Heightmap-derived world products (in-memory, stitched) -----------
    hm = build_heightmap_products(centres, mask, height, width)

    # highs / lows / peaks / dips -- BGRA with world_alpha
    for key, fname in (
        ("highs", "heightmap_highs.png"),
        ("lows", "heightmap_lows.png"),
        ("peaks", "curvature_peaks.png"),
        ("dips", "curvature_dips.png"),
    ):
        out_path = FINAL_DIR / fname
        _write_with_alpha(hm[key], world_alpha, out_path)
        print(f"  written: {out_path}")

    # slopes overlay: solid black, alpha = slope mag clipped to [0, 64],
    # masked to terrain pixels so flat/water is fully transparent.
    slopes_canvas = hm["slopes"]
    if world_terrain is not None:
        slopes_canvas = slopes_canvas * world_terrain.astype(np.uint8)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 3] = np.minimum(slopes_canvas, 64)
    slopes_out = FINAL_DIR / "slopes.png"
    cv2.imwrite(str(slopes_out), rgba)
    print(f"  written: {slopes_out}")

    # fly_alert overlay: textured via utils/fly_alert_pattern.png.
    # The elevation-derived alpha (0 at FLY_ALERT_MIN_M, 255 at
    # FLY_ALERT_MAX_M) multiplies the pattern's alpha, so the pattern's
    # RGB shows through only where the elevation ramp is non-zero. If the
    # pattern's size doesn't match the world canvas it is tiled to cover.
    fly_out = FINAL_DIR / "fly_alert.png"
    pattern = cv2.imread(str(FLY_ALERT_PATTERN_FILE), cv2.IMREAD_UNCHANGED)
    if pattern is None:
        print(f"  [WARN] {FLY_ALERT_PATTERN_FILE} not found; "
              f"falling back to solid white fly_alert")
        fly_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        fly_rgba[..., 0:3] = 255
        fly_rgba[..., 3] = hm["fly_alert"]
    else:
        if pattern.ndim == 2:
            pattern = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGRA)
        elif pattern.shape[2] == 3:
            pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2BGRA)
        ph, pw = pattern.shape[:2]
        if (ph, pw) != (height, width):
            reps_y = (height + ph - 1) // ph
            reps_x = (width + pw - 1) // pw
            pattern = np.tile(pattern, (reps_y, reps_x, 1))[:height, :width]
        fly_rgba = pattern.copy()
        coef = hm["fly_alert"].astype(np.uint16)
        fly_rgba[..., 3] = (
            (fly_rgba[..., 3].astype(np.uint16) * coef + 127) // 255
        ).astype(np.uint8)
    cv2.imwrite(str(fly_out), fly_rgba)
    print(f"  written: {fly_out}")

    # contour overlay: solid black, full alpha where step-contour is drawn.
    contour_alpha = hm["contour_alpha"]
    contour_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    contour_rgba[..., 3] = contour_alpha
    contour_out = FINAL_DIR / "contour.png"
    cv2.imwrite(str(contour_out), contour_rgba)
    print(f"  written: {contour_out}")

    # Recolor products -----------------------------------------------------
    if id_canvas is not None:
        terrain_bgr, terrain_opaque = _build_recolor(
            id_canvas, ID_RECOLOR, CATEGORY_COLORS, exclude=("water",),
        )
        terrain_alpha = np.where(
            terrain_opaque, world_alpha, 0
        ).astype(np.uint8)
        terrain_out = FINAL_DIR / "terrain_recolor.png"
        cv2.imwrite(str(terrain_out), np.dstack([terrain_bgr, terrain_alpha]))
        print(f"  written: {terrain_out}")
    else:
        print("  [WARN] id canvas unavailable; skipping terrain_recolor")

    water_hex = ID_RECOLOR.get("water")
    if water_canvas is not None and water_hex is not None:
        water_bgr_val = _hex_to_bgr(water_hex)
        water_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        water_hit = water_canvas > 0
        water_rgba[water_hit] = (
            water_bgr_val[0], water_bgr_val[1], water_bgr_val[2], 255,
        )
        water_rgba[..., 3] = np.minimum(water_rgba[..., 3], world_alpha)
        water_out = FINAL_DIR / "water_recolor.png"
        cv2.imwrite(str(water_out), water_rgba)
        print(f"  written: {water_out}")
    elif water_hex is None:
        print("  [WARN] ID_RECOLOR['water'] not set; skipping water_recolor")
    else:
        print("  [WARN] water canvas unavailable; skipping water_recolor")

    # Land mask = terrain id pixels with water excluded
    if world_terrain is None:
        world_land = None
    elif water_canvas is None:
        world_land = world_terrain
    else:
        world_land = world_terrain & (water_canvas == 0)

    # 3) per-layer compositing -> shades.png -----------------------------
    if not LAYERS_DIR.is_dir():
        print(f"\n[WARN] {LAYERS_DIR} not found; no per-layer stitching done")
    else:
        layer_dirs = sorted(d for d in LAYERS_DIR.iterdir() if d.is_dir())

        shades = np.zeros((height, width, 3), dtype=np.uint8)
        winner_alpha = np.zeros((height, width), dtype=np.uint8)

        used_colors: set = {_hex_to_bgr(c) for c in LAYER_COLORS.values()}
        rng = random.Random(0xF0)
        layer_palette: Dict[str, Tuple[int, int, int]] = {}

        for layer_dir in layer_dirs:
            layer = layer_dir.name
            print(f"\n=== stitching layer: {layer} ===")
            tile_map = _build_tile_map(layer_dir)
            canvas = stitch(
                tile_map, centres, mask, height, width,
                channels=1, dtype=np.uint8, read_flag=cv2.IMREAD_GRAYSCALE,
            )
            if world_land is not None:
                canvas = canvas * world_land.astype(np.uint8)

            color = _assign_layer_color(layer, LAYER_COLORS,
                                        used_colors, rng)
            layer_palette[layer] = color
            print(f"  color: BGR{color}")

            win = canvas > winner_alpha
            if win.any():
                shades[win] = color
                winner_alpha[win] = canvas[win]

        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[..., 0:3] = shades
        rgba[..., 3] = np.where(winner_alpha > 0, 255, 0).astype(np.uint8)

        shades_out = FINAL_DIR / "shades.png"
        cv2.imwrite(str(shades_out), rgba)
        print(f"\n  written: {shades_out}")

        palette_out = FINAL_DIR / "shades_palette.json"
        palette_json = {
            layer: "#{:02X}{:02X}{:02X}".format(bgr[2], bgr[1], bgr[0])
            for layer, bgr in layer_palette.items()
        }
        palette_out.write_text(
            json.dumps(palette_json, indent=2), encoding="utf-8"
        )
        print(f"  written: {palette_out}")

    print(f"\n=== SUCCESS (in {time.time() - t0:.2f}s) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
