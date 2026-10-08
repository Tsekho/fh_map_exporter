"""Foxhole world-map textures: regions, BC7 and the map-mod .pak.

The in-game map draws one texture per region, Map<Region>.uasset under
War/Content/Textures/UI/HexMaps/Processed: a 2048x1776 BC7 image, which is
a 2048x2048 hex tile (utils/img/mask.png) with 136 px trimmed off the top
and bottom. A map mod is a .pak replacing some or all of them, plus an
upscaled WorldMapBG so the ocean around the hexes keeps up.

Images are BGRA uint8 arrays throughout, as cv2 reads them.
"""

import json
import os
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import cv2
import etcpak
import numpy as np
import texture2ddecoder

from toolset.pak import read_pak, write_pak
from utils.config import CENTRES_FILE, MASK_FILE, TILE_HALF, TILE_SIZE

TEX_W, TEX_H = TILE_SIZE, 1776
TRIM = (TILE_SIZE - TEX_H) // 2     # 136 rows off the top and bottom
BC7_SIZE = TEX_W * TEX_H            # BC7 spends one byte per pixel

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
TEXTURE_DIR = "War/Content/Textures/UI/HexMaps/Processed"
BG_PATH = "War/Content/Textures/UI/WorldMap/WorldMapBG.uasset"
BG_FILE = ASSETS_DIR / "WorldMapBG.uasset"

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")

# Closes every texture uasset, after the BC7 payload.
_UASSET_FOOTER = (b"\x00\x08\x00\x00\xf0\x06\x00\x00\x01\x00\x00\x00\x00\x00"
                  b"\x00\x00\x0f\x00\x00\x00\x00\x00\x00\x00\xc1\x83\x2a\x9e")

# Canonical texture name (MapAcrithiaHex) -> the uasset bytes before the BC7.
HEADERS: Dict[str, bytes] = {
    p.name: p.read_bytes() for p in sorted((ASSETS_DIR / "headers").iterdir())
}
_BY_LOWER = {name.lower(): name for name in HEADERS}

# On Windows a process pool can't wait on more than 61 workers.
_MAX_WORKERS = 61


# ------------------------------------------------------------------------------
#  Regions
# ------------------------------------------------------------------------------

def canonical(name: str) -> str:
    """Texture name for a region however it is spelled: AcrithiaHex,
    acrithia, MapAcrithia and MapAcrithiaHex all resolve to
    MapAcrithiaHex. Raises ValueError for anything else."""
    stem = name.lower()
    for cand in (stem, "map" + stem, stem + "hex", "map" + stem + "hex"):
        if cand in _BY_LOWER:
            return _BY_LOWER[cand]
    raise ValueError(f"unknown region: {name}")


def short_name(tex: str) -> str:
    """File stem used for a texture's PNG: MapAcrithiaHex -> AcrithiaHex."""
    return tex[3:]


def centres() -> Dict[str, Tuple[int, int]]:
    """{texture name: world-pixel centre} for every region."""
    with open(CENTRES_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {canonical(k): (int(v[0]), int(v[1])) for k, v in raw.items()}


def world_size(cs: Mapping[str, Tuple[int, int]]) -> Tuple[int, int]:
    """(height, width) of the stitched world image."""
    return (max(cy for _, cy in cs.values()) + TILE_HALF,
            max(cx for cx, _ in cs.values()) + TILE_HALF)


def load_mask() -> np.ndarray:
    mask = cv2.imread(str(MASK_FILE), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"mask not found: {MASK_FILE}")
    if mask.shape != (TILE_SIZE, TILE_SIZE):
        raise ValueError(f"{MASK_FILE} is {mask.shape}, expected "
                         f"{(TILE_SIZE, TILE_SIZE)}")
    return mask


# ------------------------------------------------------------------------------
#  Images
# ------------------------------------------------------------------------------

def load_image(path) -> np.ndarray:
    """Any 8/16-bit gray, BGR or BGRA image as BGRA uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"could not read: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    if img.dtype == np.uint16:
        img = (img // 257).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _crop_hex(world: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """TILE_SIZE square centred at (cx, cy); outside the world is clear."""
    h, w = world.shape[:2]
    y1, x1 = cy - TILE_HALF, cx - TILE_HALF
    out = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)
    sy1, sy2 = max(y1, 0), min(y1 + TILE_SIZE, h)
    sx1, sx2 = max(x1, 0), min(x1 + TILE_SIZE, w)
    if sy2 > sy1 and sx2 > sx1:
        out[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = world[sy1:sy2, sx1:sx2]
    return out


def break_world(world: np.ndarray, one_k: bool = False,
                ) -> Dict[str, np.ndarray]:
    """{texture name: tile} cut from a stitched world image. Tiles are
    TEX_W x TEX_H with the hex mask in their alpha, or half that with
    ``one_k`` (no use for a mod, which wants full size)."""
    mask = load_mask()
    tiles: Dict[str, np.ndarray] = {}
    for tex, (cx, cy) in centres().items():
        tile = _crop_hex(world, cx, cy)
        np.minimum(tile[..., 3], mask, out=tile[..., 3])
        trim = TRIM
        if one_k:
            tile = cv2.resize(tile, (TILE_HALF, TILE_HALF),
                              interpolation=cv2.INTER_AREA)
            trim //= 2
        tiles[tex] = tile[trim:tile.shape[0] - trim]
    return tiles


def fit_tile(tile: np.ndarray, label: str) -> np.ndarray:
    """A region image as TEX_W x TEX_H: full hexes (TILE_SIZE square) are
    trimmed, anything else not already that size is refused."""
    if tile.shape[:2] == (TILE_SIZE, TILE_SIZE):
        return tile[TRIM:TILE_SIZE - TRIM]
    if tile.shape[:2] != (TEX_H, TEX_W):
        raise ValueError(f"{label} is {tile.shape[1]}x{tile.shape[0]}, "
                         f"expected {TEX_W}x{TEX_H} "
                         f"(or an untrimmed {TILE_SIZE}x{TILE_SIZE} hex)")
    return tile


def stitch_world(tiles: Mapping[str, np.ndarray]) -> np.ndarray:
    """Inverse of break_world: region tiles pasted into a world image,
    each clipped to its hex."""
    cs = centres()
    world = np.zeros((*world_size(cs), 4), dtype=np.uint8)
    hex_px = load_mask()[TRIM:TILE_SIZE - TRIM] > 0
    for tex, tile in tiles.items():
        cx, cy = cs[tex]
        y1, x1 = cy - TILE_HALF + TRIM, cx - TILE_HALF
        dst = world[y1:y1 + TEX_H, x1:x1 + TEX_W]
        dst[hex_px] = tile[hex_px]
    return world


def load_folder(folder) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """({texture name: tile}, skipped file names) for the region images in
    a folder, named after their region (AcrithiaHex.png, ...)."""
    tiles: Dict[str, np.ndarray] = {}
    skipped: List[str] = []
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            tex = canonical(path.stem)
        except ValueError:
            skipped.append(path.name)
            continue
        tiles[tex] = fit_tile(load_image(path), path.name)
    return tiles, skipped


# ------------------------------------------------------------------------------
#  BC7
# ------------------------------------------------------------------------------

def encode(tile: np.ndarray) -> bytes:
    """BC7 for one TEX_W x TEX_H BGRA tile."""
    if tile.shape != (TEX_H, TEX_W, 4):
        raise ValueError(f"tile is {tile.shape}, expected {(TEX_H, TEX_W, 4)}")
    rgba = np.ascontiguousarray(tile[..., [2, 1, 0, 3]])
    return etcpak.compress_bc7(rgba.tobytes(), TEX_W, TEX_H,
                               etcpak.BC7CompressBlockParams())


def decode(bc7: bytes) -> np.ndarray:
    """BGRA tile from BC7 data."""
    raw = texture2ddecoder.decode_bc7(bc7, TEX_W, TEX_H)
    return np.frombuffer(raw, dtype=np.uint8).reshape(TEX_H, TEX_W, 4).copy()


def _ignore_sigint() -> None:
    """Pool worker initializer: Ctrl+C is the parent's to handle; a worker
    finishes its tile (under a second) and is shut down."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def encode_all(tiles: Mapping[str, np.ndarray],
               on_done: Optional[Callable[[str], None]] = None,
               ) -> Dict[str, bytes]:
    """encode() every tile across a process pool (the encoder holds the
    GIL, so threads don't help). ``on_done(texture name)`` runs in this
    process as each one lands."""
    workers = max(1, min(os.cpu_count() or 1, len(tiles), _MAX_WORKERS))
    out: Dict[str, bytes] = {}
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_ignore_sigint) as pool:
        futures = {pool.submit(encode, tile): tex
                   for tex, tile in tiles.items()}
        try:
            for fut in as_completed(futures):
                tex = futures[fut]
                out[tex] = fut.result()
                if on_done is not None:
                    on_done(tex)
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return out


# ------------------------------------------------------------------------------
#  Mod .pak
# ------------------------------------------------------------------------------

def mod_path(directory, name: str) -> Path:
    """Where a mod called ``name`` goes: the game only mounts paks named
    after its own, so the War-WindowsNoEditor_ prefix is required."""
    return Path(directory) / f"War-WindowsNoEditor_{name}.pak"


def write_mod(path, textures: Mapping[str, bytes],
              compress: bool = True) -> None:
    """Pack {texture name: BC7} into a map mod at ``path``."""
    files: Dict[str, bytes] = {}
    for tex, bc7 in textures.items():
        if len(bc7) != BC7_SIZE:
            raise ValueError(f"{tex}: BC7 data is {len(bc7)} bytes, "
                             f"expected {BC7_SIZE}")
        files[f"{TEXTURE_DIR}/{tex}.uasset"] = (HEADERS[tex] + bc7
                                                + _UASSET_FOOTER)
    files[BG_PATH] = BG_FILE.read_bytes()
    write_pak(path, files, compress)


def read_mod(path) -> Dict[str, bytes]:
    """{texture name: BC7} for every region texture in a map mod."""
    out: Dict[str, bytes] = {}
    for name, data in read_pak(path).items():
        if not name.startswith(TEXTURE_DIR + "/"):
            continue
        tex = canonical(Path(name).stem)
        start = len(HEADERS[tex])
        if len(data) != start + BC7_SIZE + len(_UASSET_FOOTER):
            raise ValueError(f"{path}: {name} is {len(data)} bytes, not a "
                             f"{TEX_W}x{TEX_H} BC7 map texture")
        out[tex] = data[start:start + BC7_SIZE]
    return out
