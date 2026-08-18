"""Break a world PNG into per-region crops using mask.png.

For each region listed in ``utils/region_centers.json``, takes a
TILE_SIZE x TILE_SIZE crop centred on the region's world-pixel
coordinates, applies ``utils/mask.png`` as the alpha channel, optionally
downscales to 1k, and trims 136 px (or 68 px @ 1k) from the top and
bottom to produce the final per-region tile.

Interactive only:

    python 6_breaker.py

Output: ``<input_stem>/<region>.png``.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.config import CENTRES_FILE, JSON_DIR, MASK_FILE, TILE_SIZE
from utils.prompt import confirm, is_interactive, select, text
from utils.tui import ScriptTUI


# Trim amount (per side, in pixels) applied after optional downscale.
# 2048 - 2*136 = 1776; 1024 - 2*68 = 888.
TRIM_2K = 136
TRIM_1K = 68


def _load_centres() -> Dict[str, Tuple[int, int]]:
    """Region centres keyed by their original-cased name when possible.

    ``region_centers.json`` stores lowercase keys. If ``_json/`` exists
    we remap each key to the matching JSON stem (preserving its
    original case); otherwise we fall back to the lowercase key."""
    with open(CENTRES_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    case_map: Dict[str, str] = {}
    if JSON_DIR.is_dir():
        case_map = {p.stem.lower(): p.stem for p in JSON_DIR.glob("*.json")}

    return {
        case_map.get(k, k): (int(v[0]), int(v[1]))
        for k, v in raw.items()
    }


def _load_mask() -> np.ndarray:
    img = cv2.imread(str(MASK_FILE), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"mask not found: {MASK_FILE}")
    if img.shape != (TILE_SIZE, TILE_SIZE):
        raise ValueError(
            f"mask {MASK_FILE} is {img.shape}, expected "
            f"{(TILE_SIZE, TILE_SIZE)}"
        )
    return img


def _list_pngs(cwd: Path) -> List[Path]:
    return sorted(p for p in cwd.glob("*.png") if p.is_file())


def _prompt_manual_path() -> Path:
    while True:
        path_raw = text("Path to PNG: ").strip().strip('"').strip("'")
        if not path_raw:
            print("  empty path; try again")
            continue
        p = Path(path_raw).expanduser()
        if not p.is_file():
            print(f"  not a file: {p}")
            continue
        return p


def _pick_source_fallback(pngs: List[Path]) -> Optional[Path]:
    print("Available PNG files in current directory:")
    if pngs:
        for i, p in enumerate(pngs, 1):
            print(f"  {i:3}. {p.name}")
    else:
        print("  (none)")
    print("    0. Paste a path manually")

    while True:
        raw = input("\nSelect PNG (number or 0 to paste): ").strip()
        if raw == "0" or (not raw and not pngs):
            return _prompt_manual_path()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(pngs):
                return pngs[idx]
        print("  invalid selection; try again")


def _pick_source_interactive() -> Optional[Path]:
    cwd = Path.cwd()
    pngs = _list_pngs(cwd)

    if not is_interactive():
        return _pick_source_fallback(pngs)

    labels = [p.name for p in pngs] + ["Paste a path manually"]
    idx = select("Select PNG", labels)
    if idx is None:
        return None
    return _prompt_manual_path() if idx == len(pngs) else pngs[idx]


def _ask_1k() -> bool:
    if not is_interactive():
        while True:
            raw = input(
                "\nDownscale output to 1k (1024x888)? [y/N]: "
            ).strip().lower()
            if raw in ("", "n", "no"):
                return False
            if raw in ("y", "yes"):
                return True
            print("  please answer y or n")

    return confirm("Downscale output to 1k? (1024x888)", default=False)


def _load_source(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"could not read: {path}")
    # Normalise to BGRA uint8 so every downstream op is uniform.
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    if img.dtype != np.uint8:
        # Scale 16-bit or float inputs to 8-bit.
        if img.dtype == np.uint16:
            img = (img // 257).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _extract_tile(
    src: np.ndarray,
    cx: int,
    cy: int,
) -> np.ndarray:
    """Return a TILE_SIZE x TILE_SIZE BGRA tile centred at (cx, cy).

    Pixels outside the source image bounds are zero-filled (fully
    transparent)."""
    half = TILE_SIZE // 2
    h, w = src.shape[:2]

    y1, y2 = cy - half, cy + half
    x1, x2 = cx - half, cx + half

    out = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.uint8)

    src_y1 = max(y1, 0)
    src_y2 = min(y2, h)
    src_x1 = max(x1, 0)
    src_x2 = min(x2, w)
    if src_y2 <= src_y1 or src_x2 <= src_x1:
        return out

    dst_y1 = src_y1 - y1
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    dst_x1 = src_x1 - x1
    dst_x2 = dst_x1 + (src_x2 - src_x1)

    out[dst_y1:dst_y2, dst_x1:dst_x2] = src[src_y1:src_y2, src_x1:src_x2]
    return out


def _apply_mask(tile: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Multiply tile's alpha by mask (both uint8)."""
    m = mask.astype(np.uint16)
    a = tile[..., 3].astype(np.uint16)
    tile[..., 3] = ((a * m + 127) // 255).astype(np.uint8)
    return tile


def _finalize(tile: np.ndarray, one_k: bool) -> np.ndarray:
    if one_k:
        tile = cv2.resize(
            tile, (TILE_SIZE // 2, TILE_SIZE // 2),
            interpolation=cv2.INTER_AREA,
        )
        trim = TRIM_1K
    else:
        trim = TRIM_2K
    return tile[trim:tile.shape[0] - trim, :]


def main() -> int:
    try:
        centres = _load_centres()
        mask = _load_mask()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    src_path = _pick_source_interactive()
    if src_path is None:
        return 1
    one_k = _ask_1k()

    try:
        src = _load_source(src_path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1

    out_dir = Path.cwd() / src_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(centres)
    size_label = "1024x888" if one_k else "2048x1776"
    title = (
        f"breaking {src_path.name} ({src.shape[1]}x{src.shape[0]}) "
        f"into {total} regions @ {size_label}"
    )

    with ScriptTUI(title=title, total=total) as tui:
        tui.log(f"output: {out_dir}")
        for name, (cx, cy) in centres.items():
            tile = _extract_tile(src, cx, cy)
            tile = _apply_mask(tile, mask)
            tile = _finalize(tile, one_k)
            out_path = out_dir / f"{name}.png"
            cv2.imwrite(str(out_path), tile)
            tui.step(f"  {out_path.name}")

        tui.log("=== SUCCESS ===")

    return 0


if __name__ == "__main__":
    sys.exit(main())
