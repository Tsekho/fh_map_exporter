"""Stitch step-4 bakes into world PNGs and assemble final composites.

Writes to ``export/_final/``: ``technical/`` (ao, heightmap_simple, contour),
``assembly/`` (base_layer, beaches, roads, fly_alert, dive_alert, contours,
rdz, ranges, bridges_aim), and verbatim ``id/``, ``split_layers/``,
``svg_layers/``.

Each output is a named stage, built only when asked for and only when its
inputs are newer than what is already on disk. Shared intermediates (the
stitched ID coverage, heightmaps, world alpha) are built on first use and
freed once no remaining stage needs them. Where one stage reads another's
written output (rdz and ranges read the stitched svg_layers), the producer
is pulled into the run when its files are missing or stale.

Usage:
    python 5_finalize_exports.py                 # pick outputs interactively
    python 5_finalize_exports.py base_layer rdz  # named stages
    python 5_finalize_exports.py -a              # every stage
    python 5_finalize_exports.py -a -f           # ... and rebuild regardless
"""

import argparse
import colorsys
import json
import os
import random
import sys
import time
import traceback
from functools import cached_property
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils import progress, tui

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
    TILE_SIZE,
    HM_SPLIT_M,
    FLY_ALERT_MIN_M,
    FLY_ALERT_MAX_M,
    assert_height_offsets_match,
    height_offset_cm,
)


# Output subdirectories inside FINAL_DIR.
TECHNICAL_DIR = "technical"
ASSEMBLY_DIR = "assembly"

# Gaussian blur applied to the contour overlay before it lands in assembly/.
CONTOURS_BLUR_KSIZE = 3


class _StepLogger:
    """Per-file reporting; only surfaces in -v runs and piped output."""

    def saved(self, path: Path) -> None:
        """Report a save. Path is shown relative to FINAL_DIR."""
        try:
            short = path.relative_to(FINAL_DIR).as_posix()
        except ValueError:
            short = path.name
        print(f"  saved  {short}")

    def info(self, msg: str) -> None:
        print(f"  {msg}")


LOG = _StepLogger()


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


def _imwrite(img: np.ndarray, out_path: Path) -> None:
    """Encode to a sibling temp file, then rename over the target.

    A whole-world PNG takes seconds to write; interrupted in place it would
    leave a truncated file with a fresh mtime, which up_to_date() would
    trust. The rename is atomic, so a kill leaves either the previous
    output or none -- both of which read as stale.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".partial.png")
    try:
        if not cv2.imwrite(str(tmp), img):
            raise OSError(f"cv2 failed to write {tmp}")
        os.replace(tmp, out_path)
    finally:
        if tmp.exists():
            tmp.unlink()


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
    _imwrite(bgra, out_path)


def _write_rgba(rgba: np.ndarray, out_path: Path) -> None:
    _imwrite(rgba, out_path)


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


def build_height_offset_m(
    centres: Dict[str, Tuple[int, int]],
    mask: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Per-pixel copy of the Z offset (m) Exporter.exe baked into each region.

    Laid out with the same tiling and hex mask as stitch(), so subtracting it
    from a stitched heightmap recovers the raw in-game world Z that the
    altimeter reads. See config.HEIGHT_OFFSETS_CM for why that matters.

    Overlaps resolve with np.maximum, matching stitch(). The hex mask makes
    tiles all but disjoint, and every offset is well under a metre apart
    anyway, so the choice only ever moves a border pixel by centimetres.
    """
    assert_height_offsets_match()
    canvas = np.zeros((height, width), dtype=np.float32)
    applied = 0

    for name, (cx, cy) in centres.items():
        offset_m = height_offset_cm(name) / 100.0
        if offset_m == 0.0:
            continue
        tile = _apply_hex_mask(
            np.full((TILE_SIZE, TILE_SIZE), offset_m, dtype=np.float32), mask)
        dst = canvas[cy - TILE_HALF: cy + TILE_HALF,
                     cx - TILE_HALF: cx + TILE_HALF]
        np.maximum(dst, tile, out=dst)
        applied += 1

    print(f"  height offsets applied to {applied}/{len(centres)} regions "
          f"(max {canvas.max():.2f} m)")
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
        print(f"  [WARN] {HM_LANDSCAPE_DIR} not found; "
              f"skipping landscape heightmap products")
        return None
    print(f"\n=== stitching heightmap_landscape ===")
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
    offset_m: np.ndarray | None = None,
) -> None:
    print("  building fly_alert...")
    void = raw_landscape == 0
    meters = (raw_landscape.astype(np.float32) - 32768.0) / 100.0
    # FLY_ALERT_MIN_M/MAX_M are in-game altimeter metres, which the game reads
    # as raw world Z. Undo the per-region seam normalisation to get there.
    if offset_m is not None:
        meters -= offset_m
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
        print(f"  [WARN] {FLY_ALERT_PATTERN_FILE} not found; "
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
    LOG.saved(out_path)


def build_contour(
    raw_landscape: np.ndarray,
    world_terrain: np.ndarray | None,
    height: int,
    width: int,
) -> np.ndarray:
    """Return the contour RGBA array (black lines with alpha)."""
    print("  building contour...")
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
        print(f"  [WARN] {HM_WATER_DIR} not found; skipping heightmap_simple")
        return None
    print(f"\n=== stitching heightmap_water ===")
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
    print("  building heightmap_simple...")
    void = raw_water == 0
    meters = (raw_water.astype(np.float32) - 32768.0) / 100.0
    simple = np.clip(np.round(60.0 + meters * 2.0), 0, 255).astype(np.uint8)
    simple[void] = 0
    _write_with_alpha(simple, world_alpha, out_path)
    LOG.saved(out_path)


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
    print("  building dive_alert...")
    valid = (raw_landscape != 0) & (raw_water != 0)
    land_m = (raw_landscape.astype(np.float32) - 32768.0) / 100.0
    water_m = (raw_water.astype(np.float32) - 32768.0) / 100.0
    depth = water_m - land_m
    valid &= depth > 0
    if not valid.any():
        print("  [info] no submerged pixels; dive_alert skipped")
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
    LOG.saved(out_path)


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
        print(f"\n[WARN] {LAYERS_DIR} not found; no per-layer stitching done")
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
        print(f"\n=== stitching layer: {layer} ===")
        tile_map = _build_tile_map(layer_dir)
        canvas = stitch(
            tile_map, centres, mask, height, width,
            channels=1, dtype=np.uint8, read_flag=cv2.IMREAD_GRAYSCALE,
        )
        if claim_mask is not None:
            canvas = canvas * claim_mask.astype(np.uint8)
        color = _assign_layer_color(layer, LAYER_COLORS, used_colors, rng)
        print(f"  color: BGR{color}")
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
            print(f"  filled {n_need} unassigned pixel(s) "
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
        print("  [WARN] terrain_recolor unavailable; skipping base_layer")
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
    LOG.saved(out_path)


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
        print(f"  [WARN] {RDZ_PATTERN_FILE} not found; skipping rdz")
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
    LOG.saved(out_path)


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
            LOG.info(f"[skip] ranges: svg_layers/{name}.png missing")
            continue
        if gate is not None:
            img = _mul_alpha(img, gate)
        result = _composite_over(result, img)
        any_hit = True
    if not any_hit:
        print("  [WARN] no range svg_layers available; skipping ranges")
        return
    _write_rgba(result, out_path)
    LOG.saved(out_path)


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
    LOG.saved(out_path)


# ------------------------------------------------------------------------------
#  Shared intermediates
# ------------------------------------------------------------------------------

class Ctx:
    """Everything a stage might need, stitched on first use and cached, so
    running one stage on its own only pays for that stage's inputs."""

    def __init__(self, centres: Dict[str, Tuple[int, int]],
                 mask: np.ndarray, height: int, width: int) -> None:
        self.centres = centres
        self.mask = mask
        self.height = height
        self.width = width

    def _stitch_dir(self, label: str, src_dir: Path, *, channels: int,
                    read_flag: int, dtype=np.uint8):
        if not src_dir.is_dir():
            LOG.info(f"[skip] {label}: source dir not found ({src_dir.name})")
            return None
        print(f"=== stitching {label} ===")
        return stitch(_build_tile_map(src_dir), self.centres, self.mask,
                      self.height, self.width, channels=channels,
                      dtype=dtype, read_flag=read_flag)

    @cached_property
    def world_alpha(self) -> np.ndarray:
        return _compute_world_alpha(self.centres, self.mask,
                                    self.height, self.width)

    @cached_property
    def ao(self):
        return self._stitch_dir("ao", AO_DIR, channels=1,
                                read_flag=cv2.IMREAD_GRAYSCALE)

    @cached_property
    def id_coverage(self) -> Dict[str, np.ndarray]:
        """{category: coverage} stitched from the per-region ID bakes."""
        out: Dict[str, np.ndarray] = {}
        if not ID_DIR.is_dir():
            print(f"  [WARN] {ID_DIR} not found; ID coverage unavailable")
            return out
        for cat_dir in sorted(d for d in ID_DIR.iterdir() if d.is_dir()):
            tile_map = _build_tile_map(cat_dir)
            if not tile_map:
                continue
            print(f"=== stitching id/{cat_dir.name} ===")
            out[cat_dir.name] = stitch(
                tile_map, self.centres, self.mask, self.height, self.width,
                channels=1, dtype=np.uint8, read_flag=cv2.IMREAD_GRAYSCALE,
            )
        return out

    @cached_property
    def raw_landscape(self):
        return stitch_heightmap_landscape(self.centres, self.mask,
                                          self.height, self.width)

    @cached_property
    def raw_water(self):
        return stitch_heightmap_water(self.centres, self.mask,
                                      self.height, self.width)

    @cached_property
    def height_offset_m(self) -> np.ndarray:
        return build_height_offset_m(self.centres, self.mask,
                                     self.height, self.width)

    @cached_property
    def highs_lows(self):
        if self.raw_landscape is None:
            return None, None
        return compute_highs_lows(self.raw_landscape)

    @cached_property
    def contour_rgba(self):
        if self.raw_landscape is None:
            return None
        terrain_cov = self.id_coverage.get("terrain")
        world_terrain = (terrain_cov > 0) if terrain_cov is not None else None
        return build_contour(self.raw_landscape, world_terrain,
                             self.height, self.width)

    @cached_property
    def ground_u8(self) -> np.ndarray:
        """Terrain coverage minus water, clipped to the world hexes."""
        terrain_cov = self.id_coverage.get("terrain")
        if terrain_cov is None:
            return np.zeros((self.height, self.width), dtype=np.uint8)
        water_cov = self.id_coverage.get("water")
        if water_cov is not None:
            non_water = (255 - water_cov).astype(np.uint16)
            ground = ((terrain_cov.astype(np.uint16) * non_water + 127)
                      // 255).astype(np.uint8)
        else:
            ground = terrain_cov.copy()
        return np.minimum(ground, self.world_alpha)

    @cached_property
    def ground01(self) -> np.ndarray:
        return self.ground_u8.astype(np.float32) / 255.0

    @cached_property
    def water01(self) -> np.ndarray:
        water_cov = self.id_coverage.get("water")
        if water_cov is None:
            return np.zeros((self.height, self.width), dtype=np.float32)
        return water_cov.astype(np.float32) / 255.0

    def release_except(self, keep: Sequence[str]) -> None:
        """Drop every cached canvas not in ``keep``. These are whole-world
        images -- a stitched heightmap is half a gigabyte."""
        cached = {name
                  for klass in type(self).__mro__
                  for name, value in vars(klass).items()
                  if isinstance(value, cached_property)}
        for name in list(self.__dict__):
            if name in cached and name not in keep:
                del self.__dict__[name]

    @cached_property
    def shades(self):
        shade_alpha = (self.ground_u8
                       if self.id_coverage.get("terrain") is not None
                       else None)
        return build_shades(self.centres, self.mask, self.height, self.width,
                            shade_alpha)


# ------------------------------------------------------------------------------
#  Stages
# ------------------------------------------------------------------------------

def _final(rel: str) -> Path:
    return FINAL_DIR / rel


class Stage:
    """One named output: what it writes, what it reads, how to build it."""

    def __init__(self, name: str, describe: str,
                 run: Callable[[Ctx], None],
                 outputs: Callable[[], List[Path]],
                 inputs: Callable[[], List[Path]],
                 needs: Sequence[str] = (),
                 requires: Sequence[str] = ()) -> None:
        self.name = name
        self.describe = describe
        self.run = run
        self.outputs = outputs
        self.inputs = inputs
        # Ctx attributes this stage may read; the rest are freed after it.
        self.needs = tuple(needs)
        # Stages whose written output this one reads, pulled into the run
        # when their files are missing or stale.
        self.requires = tuple(requires)

    def up_to_date(self) -> bool:
        """True when every output exists and no input has changed since. A
        missing input dir counts as unchanged: there is nothing to redo."""
        outs = self.outputs()
        if not outs or not all(p.is_file() and p.stat().st_size
                               for p in outs):
            return False
        oldest_out = min(p.stat().st_mtime for p in outs)
        return _newest_mtime(self.inputs()) <= oldest_out


def _newest_mtime(paths: Sequence[Path]) -> float:
    """Newest mtime among these files and the PNGs under these directories,
    or 0.0 when none exist."""
    newest = 0.0
    for path in paths:
        if path.is_file():
            newest = max(newest, path.stat().st_mtime)
        elif path.is_dir():
            for child in path.rglob("*.png"):
                try:
                    newest = max(newest, child.stat().st_mtime)
                except OSError:
                    continue
    return newest


def _stitch_stage(label: str, src_dir: Path, out_rel: str, *,
                  channels: int, read_flag: int) -> Callable[[Ctx], None]:
    """Stage body for a plain "stitch a tile folder, write it" output."""

    def _run(ctx: Ctx) -> None:
        canvas = ctx._stitch_dir(label, src_dir, channels=channels,
                                 read_flag=read_flag)
        if canvas is None:
            return
        out_path = _final(out_rel)
        _write_with_alpha(canvas, ctx.world_alpha, out_path)
        LOG.saved(out_path)

    return _run


def _run_ao(ctx: Ctx) -> None:
    if ctx.ao is None:
        return
    out_path = _final(f"{TECHNICAL_DIR}/ao.png")
    _write_with_alpha(ctx.ao, ctx.world_alpha, out_path)
    LOG.saved(out_path)


def _run_group(src_root: Path, layers: Sequence[str],
               out_dir: str) -> Callable[[Ctx], None]:
    """Stage body for a folder of sibling layers."""

    def _run(ctx: Ctx) -> None:
        if not src_root.is_dir():
            print(f"[WARN] {src_root} not found; skipping {out_dir}")
            return
        for layer in layers:
            canvas = ctx._stitch_dir(f"{out_dir}/{layer}", src_root / layer,
                                     channels=4,
                                     read_flag=cv2.IMREAD_UNCHANGED)
            if canvas is None:
                continue
            out_path = _final(f"{out_dir}/{layer}.png")
            _write_with_alpha(canvas, ctx.world_alpha, out_path)
            LOG.saved(out_path)

    return _run


def _run_id(ctx: Ctx) -> None:
    if not ctx.id_coverage:
        print(f"[WARN] no ID coverage to write")
        return
    for cat, canvas in ctx.id_coverage.items():
        out_path = _final(f"id/{cat}.png")
        _write_with_alpha(canvas, ctx.world_alpha, out_path)
        LOG.saved(out_path)


def _run_fly_alert(ctx: Ctx) -> None:
    if ctx.raw_landscape is None:
        print("  [WARN] no landscape heightmap; skipping fly_alert")
        return
    build_fly_alert(ctx.raw_landscape, ctx.height, ctx.width,
                    ctx.id_coverage.get("rocks"),
                    _final(f"{ASSEMBLY_DIR}/fly_alert.png"),
                    ctx.height_offset_m)


def _run_contour(ctx: Ctx) -> None:
    if ctx.contour_rgba is None:
        print("  [WARN] no landscape heightmap; skipping contour")
        return
    out_path = _final(f"{TECHNICAL_DIR}/contour.png")
    _write_rgba(ctx.contour_rgba, out_path)
    LOG.saved(out_path)


def _run_contours(ctx: Ctx) -> None:
    if ctx.contour_rgba is None:
        print("  [WARN] no landscape heightmap; skipping contours")
        return
    build_contours_assembly(ctx.contour_rgba, ctx.ground01, ctx.water01,
                            _final(f"{ASSEMBLY_DIR}/contours.png"))


def _run_heightmap_simple(ctx: Ctx) -> None:
    if ctx.raw_water is None:
        return
    build_heightmap_simple(ctx.raw_water, ctx.world_alpha,
                           _final(f"{TECHNICAL_DIR}/heightmap_simple.png"))


def _run_dive_alert(ctx: Ctx) -> None:
    water_cov = ctx.id_coverage.get("water")
    if ctx.raw_landscape is None or ctx.raw_water is None or water_cov is None:
        print("  [WARN] missing heightmap/water coverage; skipping dive_alert")
        return
    build_dive_alert(ctx.raw_landscape, ctx.raw_water, water_cov,
                     ctx.world_alpha,
                     _final(f"{ASSEMBLY_DIR}/dive_alert.png"))


def _run_base_layer(ctx: Ctx) -> None:
    print("=== assembling base_layer inputs ===")
    terrain_recolor = (
        build_terrain_recolor(ctx.id_coverage, ctx.world_alpha,
                              ctx.height, ctx.width)
        if ctx.id_coverage else None
    )
    water_recolor = build_water_recolor(ctx.id_coverage.get("water"),
                                        ctx.world_alpha,
                                        ctx.height, ctx.width)
    highs, lows = ctx.highs_lows
    build_base_layer(terrain_recolor, ctx.shades, highs, lows, water_recolor,
                     ctx.ao, ctx.ground01, ctx.world_alpha,
                     _final(f"{ASSEMBLY_DIR}/base_layer.png"))


def _run_rdz(ctx: Ctx) -> None:
    build_rdz(ctx.height, ctx.width, _final(f"{ASSEMBLY_DIR}/rdz.png"))


def _run_ranges(ctx: Ctx) -> None:
    build_ranges(ctx.height, ctx.width, ctx.ground01, ctx.water01,
                 _final(f"{ASSEMBLY_DIR}/ranges.png"))


def _id_outputs() -> List[Path]:
    if not ID_DIR.is_dir():
        return []
    return [_final(f"id/{d.name}.png")
            for d in sorted(ID_DIR.iterdir()) if d.is_dir()]


def _group_outputs(src_root: Path, layers: Sequence[str],
                   out_dir: str) -> List[Path]:
    return [_final(f"{out_dir}/{layer}.png") for layer in layers
            if (src_root / layer).is_dir()]


# Declaration order is run order: rdz and ranges read the stitched
# svg_layers world PNGs, so those must come first.
STAGES: List[Stage] = [
    Stage("ao", "technical/ao.png", _run_ao,
          lambda: [_final(f"{TECHNICAL_DIR}/ao.png")], lambda: [AO_DIR],
          needs=("world_alpha", "ao")),
    Stage("roads", "assembly/roads.png",
          _stitch_stage("roads", ROADS_DIR, f"{ASSEMBLY_DIR}/roads.png",
                        channels=4, read_flag=cv2.IMREAD_UNCHANGED),
          lambda: [_final(f"{ASSEMBLY_DIR}/roads.png")], lambda: [ROADS_DIR],
          needs=("world_alpha",)),
    Stage("beaches", "assembly/beaches.png",
          _stitch_stage("beaches", BEACHES_DIR, f"{ASSEMBLY_DIR}/beaches.png",
                        channels=4, read_flag=cv2.IMREAD_UNCHANGED),
          lambda: [_final(f"{ASSEMBLY_DIR}/beaches.png")],
          lambda: [BEACHES_DIR],
          needs=("world_alpha",)),
    Stage("bridges_aim", "assembly/bridges_aim.png",
          _stitch_stage("bridges_aim", BRIDGES_AIM_DIR,
                        f"{ASSEMBLY_DIR}/bridges_aim.png",
                        channels=4, read_flag=cv2.IMREAD_UNCHANGED),
          lambda: [_final(f"{ASSEMBLY_DIR}/bridges_aim.png")],
          lambda: [BRIDGES_AIM_DIR],
          needs=("world_alpha",)),
    Stage("split_layers", f"split_layers/<layer>.png ({len(SPLIT_LAYERS)})",
          _run_group(SPLIT_LAYERS_DIR, list(SPLIT_LAYERS), "split_layers"),
          lambda: _group_outputs(SPLIT_LAYERS_DIR, list(SPLIT_LAYERS),
                                 "split_layers"),
          lambda: [SPLIT_LAYERS_DIR],
          needs=("world_alpha",)),
    Stage("svg_layers", f"svg_layers/<layer>.png ({len(SVG_LAYERS)})",
          _run_group(SVG_LAYERS_DIR, list(SVG_LAYERS), "svg_layers"),
          lambda: _group_outputs(SVG_LAYERS_DIR, list(SVG_LAYERS),
                                 "svg_layers"),
          lambda: [SVG_LAYERS_DIR],
          needs=("world_alpha",)),
    Stage("id", "id/<category>.png", _run_id,
          _id_outputs, lambda: [ID_DIR],
          needs=("world_alpha", "id_coverage")),
    Stage("fly_alert", "assembly/fly_alert.png", _run_fly_alert,
          lambda: [_final(f"{ASSEMBLY_DIR}/fly_alert.png")],
          lambda: [HM_LANDSCAPE_DIR, ID_DIR / "rocks",
                   FLY_ALERT_PATTERN_FILE],
          needs=("raw_landscape", "id_coverage", "height_offset_m")),
    Stage("contour", "technical/contour.png", _run_contour,
          lambda: [_final(f"{TECHNICAL_DIR}/contour.png")],
          lambda: [HM_LANDSCAPE_DIR, ID_DIR / "terrain"],
          needs=("raw_landscape", "id_coverage", "contour_rgba")),
    Stage("heightmap_simple", "technical/heightmap_simple.png",
          _run_heightmap_simple,
          lambda: [_final(f"{TECHNICAL_DIR}/heightmap_simple.png")],
          lambda: [HM_WATER_DIR],
          needs=("raw_water", "world_alpha")),
    Stage("dive_alert", "assembly/dive_alert.png", _run_dive_alert,
          lambda: [_final(f"{ASSEMBLY_DIR}/dive_alert.png")],
          lambda: [HM_LANDSCAPE_DIR, HM_WATER_DIR, ID_DIR / "water"],
          needs=("raw_landscape", "raw_water", "id_coverage",
                 "world_alpha")),
    Stage("base_layer", "assembly/base_layer.png", _run_base_layer,
          lambda: [_final(f"{ASSEMBLY_DIR}/base_layer.png")],
          lambda: [AO_DIR, ID_DIR, LAYERS_DIR, HM_LANDSCAPE_DIR],
          needs=("world_alpha", "ao", "id_coverage", "shades",
                 "ground_u8", "ground01", "raw_landscape", "highs_lows")),
    Stage("contours", "assembly/contours.png", _run_contours,
          lambda: [_final(f"{ASSEMBLY_DIR}/contours.png")],
          lambda: [HM_LANDSCAPE_DIR, ID_DIR],
          needs=("raw_landscape", "contour_rgba", "id_coverage",
                 "ground_u8", "ground01", "water01", "world_alpha")),
    Stage("rdz", "assembly/rdz.png", _run_rdz,
          lambda: [_final(f"{ASSEMBLY_DIR}/rdz.png")],
          lambda: [RDZ_PATTERN_FILE, _final("svg_layers/rdz_grace.png")],
          needs=(), requires=("svg_layers",)),
    Stage("ranges", "assembly/ranges.png", _run_ranges,
          lambda: [_final(f"{ASSEMBLY_DIR}/ranges.png")],
          lambda: [ID_DIR] + [_final(f"svg_layers/{n}.png") for n in
                              ("ranges_tap", "ranges_intel", "ranges_ai",
                               "ranges_mh", "ranges_cg", "ranges_aag")],
          needs=("id_coverage", "ground_u8", "ground01", "water01",
                 "world_alpha"),
          requires=("svg_layers",)),
]


def with_prerequisites(
    selected: Sequence[Stage],
) -> Tuple[List[Stage], List[Tuple[str, str]]]:
    """Add any stage whose output a selected stage reads off disk, unless
    it is already up to date. Returns the expanded list in declaration
    order plus (added, because-of) pairs, so the run can say why it grew."""
    by_name = {s.name: s for s in STAGES}
    chosen = {s.name for s in selected}
    pulled: List[Tuple[str, str]] = []
    queue = list(selected)
    while queue:
        stage = queue.pop()
        for req_name in stage.requires:
            req = by_name.get(req_name)
            if req is None or req.name in chosen or req.up_to_date():
                continue
            chosen.add(req.name)
            pulled.append((req.name, stage.name))
            queue.append(req)
    return [s for s in STAGES if s.name in chosen], pulled


def pick_stages_interactive() -> Optional[List[Stage]]:
    return tui.select_many(
        STAGES, "Outputs to build",
        label_fn=lambda s: f"{s.name:<17}->  {s.describe}",
        short_fn=lambda s: s.name,
        noun="output",
    )


# ------------------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stitch step-4 bakes into world PNGs and assemble "
                    "final composites.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="stages: " + ", ".join(s.name for s in STAGES),
    )
    parser.add_argument("stages", nargs="*",
                        help="Stage names to build; omit for interactive")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Build every stage")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Rebuild even when outputs are newer than "
                             "their inputs")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print every log line instead of just progress "
                             "and warnings")
    args = parser.parse_args()

    by_name = {s.name: s for s in STAGES}
    if args.all:
        selected = list(STAGES)
    elif args.stages:
        unknown = [n for n in args.stages if n not in by_name]
        if unknown:
            print(f"ERROR: unknown stage(s): {', '.join(unknown)}")
            print(f"       known: {', '.join(by_name)}")
            return 1
        chosen = {n for n in args.stages}
        selected = [s for s in STAGES if s.name in chosen]
    else:
        picked = pick_stages_interactive()
        if picked is None:
            return 1
        # Declaration order, whatever order they were ticked in.
        chosen = {s.name for s in picked}
        selected = [s for s in STAGES if s.name in chosen]

    selected, pulled = with_prerequisites(selected)
    for added, because in pulled:
        print(f"    + {added} (its output is what {because} reads, "
              f"and it is missing or stale)")

    try:
        centres = load_centres()
        mask = load_mask()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    height, width = canvas_size(centres)
    print(f"=== Finalizing {len(selected)} output(s) "
          f"({width}x{height} px, {len(centres)} regions) ===")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(centres, mask, height, width)
    # Stages are run under capture_output(), which aims fd 1 at a pipe, so
    # the bars need their own handle on the terminal.
    console = progress.console_stream()

    t0 = time.time()
    failed: List[str] = []
    skipped = 0
    tracker = progress.Tracker()

    def _free_after(pos: int) -> None:
        """Drop cached canvases no later stage needs."""
        ctx.release_except({n for later in selected[pos + 1:]
                            for n in later.needs})

    with tui.Progress("Finalizing", unit="stage", step_unit="stage",
                      stream=console) as disp:
        disp.start("stages", "outputs", len(selected))
        for pos, stage in enumerate(selected):
            if not args.force and stage.up_to_date():
                disp.log(f"{tui.dim(tui.glyph(chr(0xB7), '-'))} "
                         f"{stage.name}  {tui.dim('up to date')}")
                disp.update("stages", advance=1, status=stage.name)
                skipped += 1
                _free_after(pos)
                continue
            disp.update("stages", status=stage.name)
            started = time.time()
            reason = trace = ""
            with progress.capture_output(disp, "stages", tracker,
                                         stage.name, args.verbose):
                try:
                    stage.run(ctx)
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    trace = traceback.format_exc().rstrip()
                    failed.append(stage.name)
            if not reason:
                disp.log(f"{tui.green(tui.glyph(chr(0x2713), '+'))} "
                         f"{stage.name}  "
                         f"{tui.dim(f'{time.time() - started:.1f}s')}")
            else:
                disp.log(tui.red(f"{tui.glyph(chr(0x2717), 'x')} "
                                 f"{stage.name}: {reason}"))
                disp.log(tui.dim(trace))
            disp.update("stages", advance=1)
            _free_after(pos)
        disp.finish("stages", not failed,
                    note=f"{len(failed)} failed")

    if console is not sys.stdout:
        console.close()

    took = time.time() - t0
    if skipped:
        print(f"    {skipped} stage(s) already up to date "
              f"(use -f to rebuild)")
    if failed:
        print(f"\n{len(failed)} stage(s) failed: {', '.join(failed)}")
        return 1
    print(f"\n=== SUCCESS (in {took:.2f}s) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
