"""
5_finalize_exports.py
========
Consolidate every per-region bake into a single world-sized export directory.

Steps:
    1. Derive heightmap_simple per region from heightmap_landscape:
           int16 -> int8 where 0 m == 60 and 1 shade == 0.5 m.
           Raw value 0 (void) and any height <= -30 m both map to 0.
    2. Stitch the five top-level bakes
           ao / heightmap_simple / id / water / contour
       into world-sized PNGs at export/_final/<bake>.png using tile centers
       from utils/region_centers.json, the hex mask from utils/mask.png, and
       np.maximum to resolve overlap.
    3. Stitch every folder under export/_layers/<layer>/ into
           export/_final/_<layer>.png
       Those stitched layers are then masked by the stitched id map so that
       only pixels whose id equals the terrain color (#00FF00) survive.
    4. Write the per-region, id-masked layer tiles to
           export/_final/layers_masked/<layer>/<region>.png

Output tree:
    export/_final/
        ao.png
        heightmap_simple.png
        id.png
        water.png
        contour.png
        _<layer>.png              (one per layer folder in export/_layers)
        layers_masked/
            <layer>/
                <region>.png

Usage:
    python 5_finalize_exports.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from utils.config import (
    AO_DIR,
    CENTRES_FILE,
    CONTOUR_DIR,
    FINAL_DIR,
    HM_LANDSCAPE_DIR,
    HM_SIMPLE_DIR,
    ID_DIR,
    TERRAIN_BGR,
    LAYERS_DIR,
    LAYERS_MASKED_DIR,
    MASK_FILE,
    TILE_HALF,
    WATER_DIR,
)


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
    """Zero out everything outside the hex mask, preserving dtype + channels."""
    if tile.ndim == 2:
        return tile * mask.astype(tile.dtype)
    return tile * mask.astype(tile.dtype)[:, :, None]


# ------------------------------------------------------------------------------
#  Heightmap simple conversion
# ------------------------------------------------------------------------------


def build_heightmap_simple() -> None:
    """
    Read every 16-bit heightmap in export/heightmap_landscape and write
    an 8-bit heightmap_simple variant to export/heightmap_simple.

    Mapping (per pixel):
        raw == 0                -> 0   (void)
        meters <= -30           -> 0
        otherwise               -> round(meters * 2) + 60, clipped to [0, 255]
    where meters = (raw - 32768) / 100.
    """
    if not HM_LANDSCAPE_DIR.is_dir():
        print(f"  [WARN] {HM_LANDSCAPE_DIR} not found; skipping heightmap_simple")
        return

    HM_SIMPLE_DIR.mkdir(parents=True, exist_ok=True)
    src_paths = sorted(HM_LANDSCAPE_DIR.glob("*.png"))
    print(f"\n=== heightmap_simple ({len(src_paths)} tiles) ===")

    for p in src_paths:
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if raw is None:
            print(f"  [WARN] unreadable: {p}")
            continue
        if raw.dtype != np.uint16:
            raw = raw.astype(np.uint16)

        meters = (raw.astype(np.int32) - 32768) / 100.0
        simple = np.round(meters * 2.0).astype(np.int32) + 60
        void = (raw == 0) | (meters <= -30.0)
        simple = np.clip(simple, 0, 255).astype(np.uint8)
        simple[void] = 0

        out = HM_SIMPLE_DIR / p.name
        cv2.imwrite(str(out), simple)


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

    *channels* = 1 (grayscale) / 3 (RGB) / 4 (RGBA).
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

        # Normalize channel count
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


def _stitch_dir(
    label: str,
    src_dir: Path,
    centres: Dict[str, Tuple[int, int]],
    mask: np.ndarray,
    height: int,
    width: int,
    *,
    channels: int,
    dtype: np.dtype,
    read_flag: int,
    out_path: Path,
) -> np.ndarray | None:
    if not src_dir.is_dir():
        print(f"  [WARN] {src_dir} not found; skipping {label}")
        return None
    print(f"\n=== stitching {label} ===")
    tile_map = _build_tile_map(src_dir)
    canvas = stitch(tile_map, centres, mask, height, width,
                    channels=channels, dtype=dtype, read_flag=read_flag)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    print(f"  written: {out_path}")
    return canvas


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

    # 1) derive heightmap_simple from heightmap_landscape -------------------
    build_heightmap_simple()

    # 2) stitch the five top-level bakes ------------------------------------
    id_canvas = _stitch_dir(
        "id", ID_DIR, centres, mask, height, width,
        channels=3, dtype=np.uint8, read_flag=cv2.IMREAD_COLOR,
        out_path=FINAL_DIR / "id.png",
    )
    _stitch_dir(
        "ao", AO_DIR, centres, mask, height, width,
        channels=1, dtype=np.uint8, read_flag=cv2.IMREAD_GRAYSCALE,
        out_path=FINAL_DIR / "ao.png",
    )
    _stitch_dir(
        "heightmap_simple", HM_SIMPLE_DIR, centres, mask, height, width,
        channels=1, dtype=np.uint8, read_flag=cv2.IMREAD_GRAYSCALE,
        out_path=FINAL_DIR / "heightmap_simple.png",
    )
    water_canvas = _stitch_dir(
        "water", WATER_DIR, centres, mask, height, width,
        channels=1, dtype=np.uint8, read_flag=cv2.IMREAD_GRAYSCALE,
        out_path=FINAL_DIR / "water.png",
    )
    _stitch_dir(
        "contour", CONTOUR_DIR, centres, mask, height, width,
        channels=4, dtype=np.uint8, read_flag=cv2.IMREAD_UNCHANGED,
        out_path=FINAL_DIR / "contour.png",
    )

    # 3/4) stitch per-layer tiles, masked by id == landscape ---------------
    if not LAYERS_DIR.is_dir():
        print(f"\n[WARN] {LAYERS_DIR} not found; no per-layer stitching done")
    else:
        # terrain mask (world-sized) derived from stitched id map, with
        # water pixels excluded so layers never bleed onto lakes/rivers.
        if id_canvas is None:
            print("[WARN] id canvas unavailable; per-layer masking is skipped")
            world_terrain = None
        else:
            world_terrain = (
                (id_canvas[..., 0] == TERRAIN_BGR[0])
                & (id_canvas[..., 1] == TERRAIN_BGR[1])
                & (id_canvas[..., 2] == TERRAIN_BGR[2])
            )
            if water_canvas is not None:
                world_terrain = world_terrain & (water_canvas == 0)

        layer_dirs = sorted(d for d in LAYERS_DIR.iterdir() if d.is_dir())
        for layer_dir in layer_dirs:
            layer = layer_dir.name
            print(f"\n=== stitching layer: {layer} ===")
            tile_map = _build_tile_map(layer_dir)
            canvas = stitch(
                tile_map, centres, mask, height, width,
                channels=1, dtype=np.uint8, read_flag=cv2.IMREAD_GRAYSCALE,
            )
            if world_terrain is not None:
                canvas = canvas * world_terrain.astype(np.uint8)

            # Emit the stitched layer as an all-white RGBA image whose alpha
            # channel carries the layer intensity: white stays opaque white,
            # black becomes fully transparent.
            rgba = np.empty((height, width, 4), dtype=np.uint8)
            rgba[..., 0:3] = 255
            rgba[..., 3] = canvas

            out_path = FINAL_DIR / f"_{layer}.png"
            cv2.imwrite(str(out_path), rgba)
            print(f"  written: {out_path}")

            # per-region masked tiles (grayscale, with water excluded)
            for name, (cx, cy) in centres.items():
                tile_path = tile_map.get(name.lower())
                if tile_path is None:
                    continue
                tile = cv2.imread(str(tile_path), cv2.IMREAD_GRAYSCALE)
                if tile is None:
                    continue
                tile = tile * mask.astype(tile.dtype)
                if world_terrain is not None:
                    y1, y2 = cy - TILE_HALF, cy + TILE_HALF
                    x1, x2 = cx - TILE_HALF, cx + TILE_HALF
                    local_terrain = world_terrain[y1:y2, x1:x2]
                    tile = tile * local_terrain.astype(tile.dtype)
                layer_out_dir = LAYERS_MASKED_DIR / layer
                layer_out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(layer_out_dir / f"{name}.png"), tile)

    print(f"\n=== SUCCESS (in {time.time() - t0:.2f}s) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
