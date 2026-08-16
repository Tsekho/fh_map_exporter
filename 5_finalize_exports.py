"""Stitch step-4 bakes into world PNGs and assemble final composites.

Writes to ``export/_final/``: ``technical/`` (ao, heightmap_simple, contour),
``assembly/`` (base_layer, beaches, roads, fly_alert, dive_alert, contours,
rdz, ranges, bridges_aim), and verbatim ``id/``, ``split_layers/``,
``svg_layers/``.

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
    BEACHES_DIR,
    BRIDGES_AIM_DIR,
    CENTRES_FILE,
    DIVE_ALERT_BLUR_KSIZE,
    DIVE_ALERT_BLUR_SIGMA,
    DIVE_ALERT_GRADIENT,
    FINAL_DIR,
    FLY_ALERT_PATTERN_FILE,
    HM_LANDSCAPE_DIR,
    HM_WATER_DIR,
    ID_DIR,
    ID_RECOLOR,
    LAYER_COLORS,
    LAYERS_DIR,
    MASK_FILE,
    RDZ_PATTERN_FILE,
    ROADS_DIR,
    SHADES_BLUR_KSIZE,
    SHADES_BLUR_SIGMA,
    SPLIT_LAYERS,
    SPLIT_LAYERS_DIR,
    SVG_LAYERS,
    SVG_LAYERS_DIR,
    TILE_HALF,
    HM_SPLIT_M,
    FLY_ALERT_MIN_M,
    FLY_ALERT_MAX_M,
)
from utils.tui import ScriptTUI
from utils.tui import log as tui_log
from utils.tui import step as tui_step
from utils.tui import warn as tui_warn


# Output subdirectories inside FINAL_DIR.
TECHNICAL_DIR = "technical"
ASSEMBLY_DIR = "assembly"

# Gaussian blur applied to the contour overlay before it lands in assembly/.
CONTOURS_BLUR_KSIZE = 3


def _saved(path: Path) -> None:
    """Report a save as a progress step. Path is shown relative to FINAL_DIR."""
    try:
        short = path.relative_to(FINAL_DIR).as_posix()
    except ValueError:
        short = path.name
    tui_step(f"saved  {short}")


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


def _write_rgba(rgba: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), rgba)


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
    """Paste every tile onto a world canvas; overlap resolved with np.maximum."""
    shape = (height, width) if channels == 1 else (height, width, channels)
    canvas = np.zeros(shape, dtype=dtype)
    total = len(centres)
    placed = 0
    last_logged = time.monotonic()

    for i, (name, (cx, cy)) in enumerate(centres.items(), 1):
        now = time.monotonic()
        if now - last_logged >= 0.5 or i == total:
            tui_log(f"  stitching {i}/{total}")
            last_logged = now
        tile_path = tile_map.get(name.lower())
        if tile_path is None:
            continue

        tile = cv2.imread(str(tile_path), read_flag)
        if tile is None:
            tui_warn(f"  [WARN] unreadable tile: {tile_path}")
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

    tui_log(f"  stitched {total}/{total}  ({placed} tiles placed)")
    return canvas


# ------------------------------------------------------------------------------
#  Heightmap-derived products (landscape + water)
# ------------------------------------------------------------------------------

def stitch_heightmap_landscape(
    centres: Dict[str, Tuple[int, int]],
    mask: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray | None:
    if not HM_LANDSCAPE_DIR.is_dir():
        tui_warn(f"  [WARN] {HM_LANDSCAPE_DIR} not found; "
                 f"skipping landscape heightmap products")
        return None
    tui_log("\n=== stitching heightmap_landscape ===")
    hm_map = _build_tile_map(HM_LANDSCAPE_DIR)
    return stitch(
        hm_map, centres, mask, height, width,
        channels=1, dtype=np.uint16, read_flag=cv2.IMREAD_UNCHANGED,
    )


def compute_highs_lows(
    raw_landscape: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (highs, lows) as uint8 grayscale arrays (no disk writes)."""
    void = raw_landscape == 0
    meters = (raw_landscape.astype(np.float32) - 32768.0) / 100.0
    delta = meters - HM_SPLIT_M
    highs = np.clip(np.round(delta * 2.0), 0, 255).astype(np.uint8)
    lows = np.clip(np.round(-delta * 2.0), 0, 255).astype(np.uint8)
    highs[void] = 0
    lows[void] = 0
    return highs, lows


def build_fly_alert(
    raw_landscape: np.ndarray,
    height: int,
    width: int,
    rocks_cov: np.ndarray | None,
    out_path: Path,
) -> None:
    tui_log("  building fly_alert...")
    void = raw_landscape == 0
    meters = (raw_landscape.astype(np.float32) - 32768.0) / 100.0
    denom = max(FLY_ALERT_MAX_M - FLY_ALERT_MIN_M, 1e-6)
    fly_ratio = (meters - FLY_ALERT_MIN_M) / denom
    fly_alert = np.clip(np.round(fly_ratio * 255.0), 0, 255).astype(np.uint8)
    fly_alert[void] = 0
    if rocks_cov is not None:
        fly_alert = (
            (fly_alert.astype(np.uint16) * rocks_cov.astype(np.uint16) + 127)
            // 255
        ).astype(np.uint8)
    pattern = cv2.imread(str(FLY_ALERT_PATTERN_FILE), cv2.IMREAD_UNCHANGED)
    if pattern is None:
        tui_warn(f"  [WARN] {FLY_ALERT_PATTERN_FILE} not found; "
                 f"falling back to solid white fly_alert")
        fly_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        fly_rgba[..., 0:3] = 255
        fly_rgba[..., 3] = fly_alert
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
        coef = fly_alert.astype(np.uint16)
        fly_rgba[..., 3] = (
            (fly_rgba[..., 3].astype(np.uint16) * coef + 127) // 255
        ).astype(np.uint8)
    _write_rgba(fly_rgba, out_path)
    _saved(out_path)


def build_contour(
    raw_landscape: np.ndarray,
    world_terrain: np.ndarray | None,
    height: int,
    width: int,
) -> np.ndarray:
    """Return the contour RGBA array (black lines with alpha)."""
    tui_log("  building contour...")
    void = raw_landscape == 0
    step = (raw_landscape // 250).astype(np.int32)
    contour = np.zeros(step.shape, dtype=bool)
    contour[1:, :]  |= (step[1:, :]  - step[:-1, :]) == 1
    contour[:-1, :] |= (step[:-1, :] - step[1:,  :]) == 1
    contour[:, 1:]  |= (step[:, 1:]  - step[:, :-1]) == 1
    contour[:, :-1] |= (step[:, :-1] - step[:, 1:])  == 1
    contour &= ~void
    if world_terrain is not None:
        contour &= world_terrain
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 3] = np.where(contour, 255, 0).astype(np.uint8)
    return rgba


def stitch_heightmap_water(
    centres: Dict[str, Tuple[int, int]],
    mask: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray | None:
    if not HM_WATER_DIR.is_dir():
        tui_warn(f"  [WARN] {HM_WATER_DIR} not found; skipping heightmap_simple")
        return None
    tui_log("\n=== stitching heightmap_water ===")
    hm_map = _build_tile_map(HM_WATER_DIR)
    return stitch(
        hm_map, centres, mask, height, width,
        channels=1, dtype=np.uint16, read_flag=cv2.IMREAD_UNCHANGED,
    )


def build_heightmap_simple(
    raw_water: np.ndarray,
    world_alpha: np.ndarray,
    out_path: Path,
) -> None:
    tui_log("  building heightmap_simple...")
    void = raw_water == 0
    meters = (raw_water.astype(np.float32) - 32768.0) / 100.0
    simple = np.clip(np.round(60.0 + meters * 2.0), 0, 255).astype(np.uint8)
    simple[void] = 0
    _write_with_alpha(simple, world_alpha, out_path)
    _saved(out_path)


def _hex_to_bgra(hex_str: str) -> Tuple[int, int, int, int]:
    """#RRGGBB or #RRGGBBAA -> (b, g, r, a); alpha defaults to 255."""
    s = hex_str.lstrip("#")
    r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    a = int(s[6:8], 16) if len(s) >= 8 else 255
    return (b, g, r, a)


def build_dive_alert(
    raw_landscape: np.ndarray,
    raw_water: np.ndarray,
    water_cov: np.ndarray,
    world_alpha: np.ndarray,
    out_path: Path,
) -> None:
    """RGBA overlay colouring each submerged pixel by its depth below the
    water surface via DIVE_ALERT_GRADIENT: within each [start_m, end_m)
    range the BGRA colour is interpolated linearly between the range's two
    stops; depths past the last range stay transparent. Alpha is gated by
    water_cov and the world hex mask."""
    tui_log("  building dive_alert...")
    valid = (raw_landscape != 0) & (raw_water != 0)
    land_m = (raw_landscape.astype(np.float32) - 32768.0) / 100.0
    water_m = (raw_water.astype(np.float32) - 32768.0) / 100.0
    depth = water_m - land_m
    valid &= depth > 0
    if not valid.any():
        tui_log("  [info] no submerged pixels; dive_alert skipped")
        return

    height, width = depth.shape
    bgra_f = np.zeros((height, width, 4), dtype=np.float32)
    for start_m, end_m, hex_a, hex_b in DIVE_ALERT_GRADIENT:
        seg = valid & (depth >= start_m) & (depth < end_m)
        if not seg.any():
            continue
        t = (depth[seg] - start_m) / max(end_m - start_m, 1e-6)
        ca = np.array(_hex_to_bgra(hex_a), dtype=np.float32)
        cb = np.array(_hex_to_bgra(hex_b), dtype=np.float32)
        bgra_f[seg] = ca[None, :] + (cb - ca)[None, :] * t[:, None]

    if DIVE_ALERT_BLUR_KSIZE and DIVE_ALERT_BLUR_KSIZE > 1:
        # Normalized masked blur: only submerged pixels contribute, and the
        # result is renormalized by the blurred mask so alpha keeps full
        # strength up to the water edge instead of fading against land.
        k = int(DIVE_ALERT_BLUR_KSIZE)
        if k % 2 == 0:
            k += 1
        sig = float(DIVE_ALERT_BLUR_SIGMA)
        eps = 1e-6
        m = valid.astype(np.float32)
        a = bgra_f[..., 3] * m
        m_b = cv2.GaussianBlur(m, (k, k), sig)
        a_b = cv2.GaussianBlur(a, (k, k), sig)
        a_b_safe = np.maximum(a_b, eps)
        for ch in range(3):
            pc_b = cv2.GaussianBlur(bgra_f[..., ch] * a, (k, k), sig)
            bgra_f[..., ch] = np.where(valid, pc_b / a_b_safe, 0.0)
        bgra_f[..., 3] = np.where(valid, a_b / np.maximum(m_b, eps), 0.0)

    alpha = bgra_f[..., 3] * (water_cov.astype(np.float32) / 255.0)
    alpha = np.minimum(alpha, world_alpha.astype(np.float32))

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0:3] = np.clip(np.round(bgra_f[..., 0:3]), 0, 255).astype(np.uint8)
    rgba[..., 3] = np.clip(np.round(alpha), 0, 255).astype(np.uint8)
    _write_rgba(rgba, out_path)
    _saved(out_path)


# ------------------------------------------------------------------------------
#  Terrain/water recolor + shades (in-memory; feed base_layer)
# ------------------------------------------------------------------------------

def build_terrain_recolor(
    id_coverage: Dict[str, np.ndarray],
    world_alpha: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray | None:
    """Weighted-blend BGR image (no alpha); uncovered in-bounds pixels
    are filled via nearest-claimed propagation. Returns None when no
    ID category (other than water) has coverage."""
    colored = np.zeros((height, width, 3), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)
    for cat, hex_color in ID_RECOLOR.items():
        if cat == "water":
            continue
        cov = id_coverage.get(cat)
        if cov is None:
            continue
        color = np.array(_hex_to_bgr(hex_color), dtype=np.float32)
        cov_f = cov.astype(np.float32)
        colored += cov_f[..., None] * color
        weight += cov_f

    out_bgr = np.zeros((height, width, 3), dtype=np.uint8)
    hit = weight > 0
    if not hit.any():
        return None
    out_bgr[hit] = np.clip(
        colored[hit] / weight[hit, None], 0, 255
    ).astype(np.uint8)
    in_bounds = world_alpha > 0
    need_fill = in_bounds & ~hit
    if need_fill.any():
        src_zero = (~hit).astype(np.uint8)
        _, labels = cv2.distanceTransformWithLabels(
            src_zero, cv2.DIST_L2, 3,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        ys, xs = np.where(hit)
        src_y = np.empty(ys.size + 1, dtype=np.int32)
        src_x = np.empty(xs.size + 1, dtype=np.int32)
        src_y[0] = 0; src_x[0] = 0
        src_y[1:] = ys; src_x[1:] = xs
        lab = np.clip(labels[need_fill].astype(np.int64), 1, ys.size)
        out_bgr[need_fill] = out_bgr[src_y[lab], src_x[lab]]
    return out_bgr


def build_water_recolor(
    water_cov: np.ndarray | None,
    world_alpha: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray | None:
    """RGBA: solid ID_RECOLOR['water'] with alpha = water_cov ∧ world_alpha."""
    water_hex = ID_RECOLOR.get("water")
    if water_cov is None or water_hex is None:
        return None
    bgr = _hex_to_bgr(water_hex)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    hit = water_cov > 0
    rgba[hit, 0] = bgr[0]
    rgba[hit, 1] = bgr[1]
    rgba[hit, 2] = bgr[2]
    rgba[..., 3] = np.minimum(water_cov, world_alpha)
    return rgba


def build_shades(
    centres: Dict[str, Tuple[int, int]],
    mask: np.ndarray,
    height: int,
    width: int,
    shade_alpha: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray] | None:
    """Stitch every LAYERS_DIR/<layer>/ folder and composite via
    alpha-betting into an RGB canvas. Returns (shades_bgr, shades_alpha)
    or None when LAYERS_DIR is missing / empty."""
    if not LAYERS_DIR.is_dir():
        tui_warn(f"[WARN] {LAYERS_DIR} not found; no per-layer stitching done")
        return None
    layer_dirs = sorted(d for d in LAYERS_DIR.iterdir() if d.is_dir())
    if not layer_dirs:
        return None

    claim_mask = (shade_alpha > 0) if shade_alpha is not None else None
    shades = np.zeros((height, width, 3), dtype=np.uint8)
    winner_alpha = np.zeros((height, width), dtype=np.uint8)

    used_colors: set = {_hex_to_bgr(c) for c in LAYER_COLORS.values()}
    rng = random.Random(0xF0)

    for layer_dir in layer_dirs:
        layer = layer_dir.name
        tui_log(f"\n=== stitching layer: {layer} ===")
        tile_map = _build_tile_map(layer_dir)
        canvas = stitch(
            tile_map, centres, mask, height, width,
            channels=1, dtype=np.uint8, read_flag=cv2.IMREAD_GRAYSCALE,
        )
        if claim_mask is not None:
            canvas = canvas * claim_mask.astype(np.uint8)
        color = _assign_layer_color(layer, LAYER_COLORS, used_colors, rng)
        tui_log(f"  color: BGR{color}")
        win = canvas > winner_alpha
        if win.any():
            shades[win] = color
            winner_alpha[win] = canvas[win]

    claimed = winner_alpha > 0
    if claimed.any():
        need_fill = ~claimed
        n_need = int(need_fill.sum())
        if n_need > 0:
            src_zero = (~claimed).astype(np.uint8)
            _, labels = cv2.distanceTransformWithLabels(
                src_zero, cv2.DIST_L2, 3,
                labelType=cv2.DIST_LABEL_PIXEL,
            )
            ys, xs = np.where(claimed)
            src_y = np.empty(ys.size + 1, dtype=np.int32)
            src_x = np.empty(xs.size + 1, dtype=np.int32)
            src_y[0] = 0; src_x[0] = 0
            src_y[1:] = ys; src_x[1:] = xs
            lab = np.clip(labels[need_fill].astype(np.int64), 1, ys.size)
            shades[need_fill] = shades[src_y[lab], src_x[lab]]
            tui_log(f"  filled {n_need} unassigned pixel(s) "
                    f"with nearest shade colour (blur bleed guard)")

    if SHADES_BLUR_KSIZE and SHADES_BLUR_KSIZE > 1:
        k = int(SHADES_BLUR_KSIZE)
        if k % 2 == 0:
            k += 1
        shades = cv2.GaussianBlur(shades, (k, k), float(SHADES_BLUR_SIGMA))

    if shade_alpha is not None:
        out_alpha = shade_alpha
    else:
        out_alpha = np.where(winner_alpha > 0, 255, 0).astype(np.uint8)
    return shades, out_alpha


# ------------------------------------------------------------------------------
#  Assembly composites (the new outputs)
# ------------------------------------------------------------------------------

def _alpha_over(
    base_rgb_f: np.ndarray,
    top_rgb_u8: np.ndarray,
    top_alpha_u8: np.ndarray,
) -> np.ndarray:
    """Return base_rgb_f with top composited over it via "normal" blending.
    base_rgb_f is float32 (0..255); top inputs are uint8. Works in place
    would be nice but we return a new array."""
    a = (top_alpha_u8.astype(np.float32) / 255.0)[..., None]
    return base_rgb_f * (1.0 - a) + top_rgb_u8.astype(np.float32) * a


def build_base_layer(
    terrain_recolor: np.ndarray | None,
    shades: Tuple[np.ndarray, np.ndarray] | None,
    highs: np.ndarray | None,
    lows: np.ndarray | None,
    water_recolor: np.ndarray | None,
    ao: np.ndarray | None,
    ground: np.ndarray,
    world_alpha: np.ndarray,
    out_path: Path,
) -> None:
    """Compose base_layer.png:

        terrain_recolor (base)
      + shades                  (normal alpha over)
      + highs  * ground         (add)
      + lows   * ground         (difference)
      + water_recolor           (multiply with alpha)
      + ao                      (multiply)
    """
    if terrain_recolor is None:
        tui_warn("  [WARN] terrain_recolor unavailable; skipping base_layer")
        return

    height, width = world_alpha.shape
    base = terrain_recolor.astype(np.float32)

    if shades is not None:
        shades_rgb, shades_alpha = shades
        base = _alpha_over(base, shades_rgb, shades_alpha)

    ground_f = ground[..., None]  # (H, W, 1), 0..1

    if highs is not None:
        add = highs.astype(np.float32)[..., None] * ground_f
        base = np.clip(base + add, 0, 255)

    if lows is not None:
        diff_layer = lows.astype(np.float32)[..., None] * ground_f
        base = np.abs(base - diff_layer)

    if water_recolor is not None:
        wa = (water_recolor[..., 3].astype(np.float32) / 255.0)[..., None]
        wrgb = water_recolor[..., 0:3].astype(np.float32) / 255.0
        # multiply-with-alpha: out = out * (wrgb * wa + (1 - wa))
        base = base * (wrgb * wa + (1.0 - wa))

    if ao is not None:
        ao_f = (ao.astype(np.float32) / 255.0)[..., None]
        base = base * ao_f

    base = np.clip(base, 0, 255).astype(np.uint8)
    rgba = np.dstack([base, world_alpha])
    _write_rgba(rgba, out_path)
    _saved(out_path)


def _load_svg_layer(name: str) -> np.ndarray | None:
    """Load export/_final/svg_layers/<name>.png (BGRA). Returns None when
    the stitched file hasn't been produced yet."""
    path = FINAL_DIR / "svg_layers" / f"{name}.png"
    if not path.is_file():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def _composite_over(dst_rgba: np.ndarray, top_rgba: np.ndarray) -> np.ndarray:
    """Standard Porter-Duff source-over in float32; returns uint8 RGBA."""
    da = dst_rgba[..., 3].astype(np.float32) / 255.0
    ta = top_rgba[..., 3].astype(np.float32) / 255.0
    drgb = dst_rgba[..., 0:3].astype(np.float32)
    trgb = top_rgba[..., 0:3].astype(np.float32)
    out_a = ta + da * (1.0 - ta)
    safe = np.maximum(out_a, 1e-6)[..., None]
    out_rgb = (trgb * ta[..., None] + drgb * da[..., None] * (1.0 - ta[..., None])) / safe
    out = np.zeros_like(dst_rgba)
    out[..., 0:3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(np.round(out_a * 255.0), 0, 255).astype(np.uint8)
    return out


def _mul_alpha(rgba: np.ndarray, coef01: np.ndarray) -> np.ndarray:
    """Return a copy of ``rgba`` with its alpha multiplied by ``coef01``
    (float32 in 0..1)."""
    out = rgba.copy()
    out[..., 3] = np.clip(
        np.round(out[..., 3].astype(np.float32) * coef01), 0, 255,
    ).astype(np.uint8)
    return out


def build_rdz(height: int, width: int, out_path: Path) -> None:
    """rdz_pattern with svg_layers/rdz_grace punching holes in its alpha."""
    pattern = cv2.imread(str(RDZ_PATTERN_FILE), cv2.IMREAD_UNCHANGED)
    if pattern is None:
        tui_warn(f"  [WARN] {RDZ_PATTERN_FILE} not found; skipping rdz")
        return
    if pattern.ndim == 2:
        pattern = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGRA)
    elif pattern.shape[2] == 3:
        pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2BGRA)
    ph, pw = pattern.shape[:2]
    if (ph, pw) != (height, width):
        reps_y = (height + ph - 1) // ph
        reps_x = (width + pw - 1) // pw
        pattern = np.tile(pattern, (reps_y, reps_x, 1))[:height, :width]

    grace = _load_svg_layer("rdz_grace")
    if grace is not None:
        keep = 1.0 - (grace[..., 3].astype(np.float32) / 255.0)
        pattern = _mul_alpha(pattern, keep)

    _write_rgba(pattern, out_path)
    _saved(out_path)


def build_ranges(
    height: int,
    width: int,
    ground01: np.ndarray,
    water01: np.ndarray,
    out_path: Path,
) -> None:
    """svg_layers: tap*ground + intel + ai*ground + mh + cg*water + aag, alpha-over."""
    layers_gates = [
        ("ranges_tap",   ground01),
        ("ranges_intel", None),
        ("ranges_ai",    ground01),
        ("ranges_mh",    None),
        ("ranges_cg",    water01),
        ("ranges_aag",   None),
    ]
    result = np.zeros((height, width, 4), dtype=np.uint8)
    any_hit = False
    for name, gate in layers_gates:
        img = _load_svg_layer(name)
        if img is None:
            tui_log(f"[skip] ranges: svg_layers/{name}.png missing")
            continue
        if gate is not None:
            img = _mul_alpha(img, gate)
        result = _composite_over(result, img)
        any_hit = True
    if not any_hit:
        tui_warn("  [WARN] no range svg_layers available; skipping ranges")
        return
    _write_rgba(result, out_path)
    _saved(out_path)


def build_contours_assembly(
    contour_rgba: np.ndarray,
    ground01: np.ndarray,
    water01: np.ndarray,
    out_path: Path,
) -> None:
    """3x3 gaussian of the contour overlay with alpha multiplied by
    (0.5 * water + ground). Produces the blurred copy consumed by the
    map compositor."""
    k = CONTOURS_BLUR_KSIZE
    if k % 2 == 0:
        k += 1
    blurred_alpha = cv2.GaussianBlur(contour_rgba[..., 3], (k, k), 0)
    coef = np.clip(0.5 * water01 + ground01, 0.0, 1.0)
    out_alpha = np.clip(
        np.round(blurred_alpha.astype(np.float32) * coef), 0, 255,
    ).astype(np.uint8)
    rgba = np.zeros_like(contour_rgba)
    rgba[..., 3] = out_alpha
    _write_rgba(rgba, out_path)
    _saved(out_path)


# ------------------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------------------

def _stitch_then_alpha(
    centres: Dict[str, Tuple[int, int]],
    mask: np.ndarray,
    height: int,
    width: int,
    world_alpha: np.ndarray,
    label: str,
    src_dir: Path,
    *,
    channels: int,
    read_flag: int,
    out_rel: str,
) -> np.ndarray | None:
    if not src_dir.is_dir():
        tui_log(f"[skip] {label}: source dir not found ({src_dir.name})")
        return None
    tui_log(f"\n=== stitching {label} ===")
    tile_map = _build_tile_map(src_dir)
    canvas = stitch(tile_map, centres, mask, height, width,
                    channels=channels, dtype=np.uint8,
                    read_flag=read_flag)
    out_path = FINAL_DIR / out_rel
    _write_with_alpha(canvas, world_alpha, out_path)
    _saved(out_path)
    return canvas


def _estimate_step_count() -> int:
    """Best-effort pre-count of the TUI progress bar's total steps."""
    est = 0
    if AO_DIR.is_dir():            est += 1  # technical/ao
    if ROADS_DIR.is_dir():         est += 1  # assembly/roads
    if BEACHES_DIR.is_dir():       est += 1  # assembly/beaches
    if BRIDGES_AIM_DIR.is_dir():   est += 1  # assembly/bridges_aim
    if SPLIT_LAYERS_DIR.is_dir():
        for _layer in SPLIT_LAYERS:
            if (SPLIT_LAYERS_DIR / _layer).is_dir(): est += 1
    if SVG_LAYERS_DIR.is_dir():
        for _layer in SVG_LAYERS:
            if (SVG_LAYERS_DIR / _layer).is_dir(): est += 1
    if ID_DIR.is_dir():
        est += sum(1 for d in ID_DIR.iterdir() if d.is_dir())
    if HM_LANDSCAPE_DIR.is_dir():  est += 3  # fly_alert, contour, contours
    if HM_WATER_DIR.is_dir():      est += 1  # heightmap_simple
    if HM_LANDSCAPE_DIR.is_dir() and HM_WATER_DIR.is_dir():
        est += 1  # dive_alert
    est += 3  # base_layer, rdz, ranges (may skip)
    return est


def _stitch_id_coverage(
    centres: Dict[str, Tuple[int, int]],
    mask: np.ndarray,
    height: int,
    width: int,
    world_alpha: np.ndarray,
) -> Dict[str, np.ndarray]:
    id_coverage: Dict[str, np.ndarray] = {}
    if not ID_DIR.is_dir():
        tui_warn(f"\n[WARN] {ID_DIR} not found; per-category ID coverage skipped")
        return id_coverage

    cat_dirs = sorted(d for d in ID_DIR.iterdir() if d.is_dir())
    for cat_dir in cat_dirs:
        cat = cat_dir.name
        tui_log(f"\n=== stitching id/{cat} ===")
        tile_map = _build_tile_map(cat_dir)
        if not tile_map:
            tui_log(f"  [skip] no tiles in {cat_dir}")
            continue
        canvas = stitch(
            tile_map, centres, mask, height, width,
            channels=1, dtype=np.uint8,
            read_flag=cv2.IMREAD_GRAYSCALE,
        )
        id_coverage[cat] = canvas
        out_path = FINAL_DIR / "id" / f"{cat}.png"
        _write_with_alpha(canvas, world_alpha, out_path)
        _saved(out_path)
    return id_coverage


def _run(
    centres: Dict[str, Tuple[int, int]],
    mask: np.ndarray,
    height: int,
    width: int,
) -> None:
    """Stitch every step-4 bake and assemble the final composites. Assumes
    a ScriptTUI is already active (see main())."""
    world_alpha = _compute_world_alpha(centres, mask, height, width)

    # -- technical/ao (also reused for base_layer multiply) --
    ao_canvas = _stitch_then_alpha(
        centres, mask, height, width, world_alpha,
        "ao", AO_DIR, channels=1, read_flag=cv2.IMREAD_GRAYSCALE,
        out_rel=f"{TECHNICAL_DIR}/ao.png",
    )

    # -- assembly/roads & assembly/beaches --
    _stitch_then_alpha(
        centres, mask, height, width, world_alpha,
        "roads", ROADS_DIR,
        channels=4, read_flag=cv2.IMREAD_UNCHANGED,
        out_rel=f"{ASSEMBLY_DIR}/roads.png",
    )
    _stitch_then_alpha(
        centres, mask, height, width, world_alpha,
        "beaches", BEACHES_DIR,
        channels=4, read_flag=cv2.IMREAD_UNCHANGED,
        out_rel=f"{ASSEMBLY_DIR}/beaches.png",
    )

    # -- assembly/bridges_aim (procedural per-region tiles, own folder) --
    if BRIDGES_AIM_DIR.is_dir():
        _stitch_then_alpha(
            centres, mask, height, width, world_alpha,
            "bridges_aim", BRIDGES_AIM_DIR,
            channels=4, read_flag=cv2.IMREAD_UNCHANGED,
            out_rel=f"{ASSEMBLY_DIR}/bridges_aim.png",
        )
    else:
        tui_warn(f"\n[WARN] {BRIDGES_AIM_DIR} not found; "
                 f"skipping bridges_aim stitching")

    # -- split_layers/<layer>.png (unchanged location) --
    if SPLIT_LAYERS_DIR.is_dir():
        for layer in SPLIT_LAYERS:
            src = SPLIT_LAYERS_DIR / layer
            _stitch_then_alpha(
                centres, mask, height, width, world_alpha,
                f"split_layers/{layer}", src,
                channels=4, read_flag=cv2.IMREAD_UNCHANGED,
                out_rel=f"split_layers/{layer}.png",
            )
    else:
        tui_warn(f"\n[WARN] {SPLIT_LAYERS_DIR} not found; "
                 f"skipping split_layer stitching")

    # -- svg_layers/<layer>.png (unchanged location; consumed below) --
    if SVG_LAYERS_DIR.is_dir():
        for layer in SVG_LAYERS:
            src = SVG_LAYERS_DIR / layer
            _stitch_then_alpha(
                centres, mask, height, width, world_alpha,
                f"svg_layers/{layer}", src,
                channels=4, read_flag=cv2.IMREAD_UNCHANGED,
                out_rel=f"svg_layers/{layer}.png",
            )
    else:
        tui_warn(f"\n[WARN] {SVG_LAYERS_DIR} not found; "
                 f"skipping svg_layer stitching")

    id_coverage = _stitch_id_coverage(centres, mask, height, width, world_alpha)
    terrain_cov = id_coverage.get("terrain")
    water_cov = id_coverage.get("water")
    rocks_cov = id_coverage.get("rocks")

    world_terrain = (terrain_cov > 0) if terrain_cov is not None else None

    # -- ground mask = terrain * (not water), 0..1 float --
    if terrain_cov is not None:
        if water_cov is not None:
            non_water = (255 - water_cov).astype(np.uint16)
            ground_u8 = (
                (terrain_cov.astype(np.uint16) * non_water + 127) // 255
            ).astype(np.uint8)
        else:
            ground_u8 = terrain_cov.copy()
        ground_u8 = np.minimum(ground_u8, world_alpha)
    else:
        ground_u8 = np.zeros((height, width), dtype=np.uint8)
    ground01 = ground_u8.astype(np.float32) / 255.0
    water01 = (water_cov.astype(np.float32) / 255.0
               if water_cov is not None
               else np.zeros((height, width), dtype=np.float32))

    # -- heightmap products --
    raw_landscape = stitch_heightmap_landscape(centres, mask, height, width)
    highs = lows = None
    contour_rgba = None
    if raw_landscape is not None:
        tui_log("\n=== deriving heightmap_landscape products ===")
        highs, lows = compute_highs_lows(raw_landscape)
        build_fly_alert(
            raw_landscape, height, width, rocks_cov,
            FINAL_DIR / ASSEMBLY_DIR / "fly_alert.png",
        )
        contour_rgba = build_contour(
            raw_landscape, world_terrain, height, width,
        )
        contour_out = FINAL_DIR / TECHNICAL_DIR / "contour.png"
        _write_rgba(contour_rgba, contour_out)
        _saved(contour_out)

    raw_water = stitch_heightmap_water(centres, mask, height, width)
    if raw_water is not None:
        tui_log("\n=== deriving heightmap_water products ===")
        build_heightmap_simple(
            raw_water, world_alpha,
            FINAL_DIR / TECHNICAL_DIR / "heightmap_simple.png",
        )

    if (
        raw_landscape is not None
        and raw_water is not None
        and water_cov is not None
    ):
        build_dive_alert(
            raw_landscape, raw_water, water_cov, world_alpha,
            FINAL_DIR / ASSEMBLY_DIR / "dive_alert.png",
        )
    else:
        tui_warn("  [WARN] missing heightmap/water coverage; skipping dive_alert")

    del raw_landscape, raw_water

    # -- in-memory intermediates for base_layer --
    tui_log("\n=== assembling base_layer inputs ===")
    terrain_recolor = (
        build_terrain_recolor(id_coverage, world_alpha, height, width)
        if id_coverage else None
    )
    water_recolor = build_water_recolor(water_cov, world_alpha, height, width)

    # shade_alpha = terrain * (not water) ∧ world_alpha (same as ground_u8)
    shade_alpha = ground_u8 if terrain_cov is not None else None
    shades_pair = build_shades(centres, mask, height, width, shade_alpha)

    # -- assembly/base_layer.png --
    build_base_layer(
        terrain_recolor, shades_pair, highs, lows, water_recolor, ao_canvas,
        ground01, world_alpha,
        FINAL_DIR / ASSEMBLY_DIR / "base_layer.png",
    )

    # -- assembly/contours.png (blurred contour with alpha gating) --
    if contour_rgba is not None:
        build_contours_assembly(
            contour_rgba, ground01, water01,
            FINAL_DIR / ASSEMBLY_DIR / "contours.png",
        )

    # -- assembly/rdz.png --
    build_rdz(height, width, FINAL_DIR / ASSEMBLY_DIR / "rdz.png")

    # -- assembly/ranges.png --
    build_ranges(
        height, width, ground01, water01,
        FINAL_DIR / ASSEMBLY_DIR / "ranges.png",
    )


def main() -> int:
    t0 = time.time()

    try:
        centres = load_centres()
        mask = load_mask()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    height, width = canvas_size(centres)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    title = f"Finalizing exports ({width}x{height} px, {len(centres)} regions)"
    with ScriptTUI(title=title, total=_estimate_step_count()):
        _run(centres, mask, height, width)
        tui_log(f"\n=== SUCCESS (in {time.time() - t0:.2f}s) ===")

    return 0


if __name__ == "__main__":
    sys.exit(main())
