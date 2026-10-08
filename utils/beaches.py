"""Beaches layer from the terrain weightmaps (no Blender needed).

Barges deploy their ramp only where the ground's physical material is Sand
or WetSand, and export/_layers/ is named by physical material, so a beach
is shore-connected ground dominated by one of BEACH_LAYERS. See the
BEACH_* tunables in utils/config.py.
"""

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from utils.config import (
    BEACH_COLOR,
    BEACH_EDGE_BLUR_PX,
    BEACH_FADE_PX,
    BEACH_FULL_PX,
    BEACH_LAYERS,
    BEACHES_DIR,
    ID_DIR,
    LAYERS_DIR,
    short_path,
)
from utils.png import imwrite_atomic

# Landscape visibility (holes), not a material.
_VISIBILITY_LAYER = "datalayer__"


def _region_layers(region: str) -> Dict[str, np.ndarray]:
    """{layer name: uint8 weightmap} for every material layer of a region."""
    out: Dict[str, np.ndarray] = {}
    if not LAYERS_DIR.is_dir():
        return out
    for d in LAYERS_DIR.iterdir():
        if not d.is_dir() or d.name.lower() == _VISIBILITY_LAYER:
            continue
        p = d / f"{region}.png"
        if p.is_file():
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                out[d.name] = img
    return out


def beach_alpha(layers: Dict[str, np.ndarray], water_cov: np.ndarray,
                terrain_cov: np.ndarray) -> Optional[np.ndarray]:
    """Float alpha in [0, 1], or None when the region has no beach layer."""
    want = {n.lower() for n in BEACH_LAYERS}
    names = list(layers)
    beach_idx = [i for i, n in enumerate(names) if n.lower() in want]
    if not beach_idx:
        return None
    stack = np.stack([layers[n] for n in names])
    dom = stack.argmax(axis=0)
    sand = np.isin(dom, beach_idx) & (stack.max(axis=0) > 0)

    water = water_cov >= 128
    # Sand reachable from the water by stepping through sand, at most
    # BEACH_FADE_PX steps away: an inland sand patch never counts, and
    # neither does sand that only touches the shore through other ground.
    kernel = np.ones((3, 3), np.uint8)
    reach = water.copy()
    passable = sand | water
    for _ in range(int(BEACH_FADE_PX)):
        reach = cv2.dilate(reach.astype(np.uint8), kernel).astype(bool) & passable
    shore_sand = (reach & sand).astype(np.float32)
    if BEACH_EDGE_BLUR_PX > 0:
        shore_sand = cv2.GaussianBlur(shore_sand, (0, 0), BEACH_EDGE_BLUR_PX)

    dist = cv2.distanceTransform((~water).astype(np.uint8), cv2.DIST_L2, 5)
    span = max(float(BEACH_FADE_PX - BEACH_FULL_PX), 1e-6)
    fade = np.clip((BEACH_FADE_PX - dist) / span, 0.0, 1.0)

    land = (terrain_cov.astype(np.float32) / 255.0) * (1.0 - water_cov / 255.0)
    return fade * shore_sand * land


def render_beaches(region: str, mask: np.ndarray) -> bool:
    """Write BEACHES_DIR/<region>.png (RGBA). Needs the region's id/water
    and id/terrain bakes and its _layers weightmaps."""
    water_path = ID_DIR / "water" / f"{region}.png"
    terrain_path = ID_DIR / "terrain" / f"{region}.png"
    water_cov = cv2.imread(str(water_path), cv2.IMREAD_GRAYSCALE)
    terrain_cov = cv2.imread(str(terrain_path), cv2.IMREAD_GRAYSCALE)
    if water_cov is None or terrain_cov is None:
        print("  [WARN] beaches skipped (missing id/water or id/terrain; "
              "run -id first)")
        return False

    layers = _region_layers(region)
    shape = water_cov.shape
    bad = [n for n, a in layers.items() if a.shape != shape]
    if bad:
        print(f"  [WARN] beaches skipped (weightmap size mismatch: "
              f"{', '.join(bad)})")
        return False

    alpha = beach_alpha(layers, water_cov, terrain_cov)
    out = np.zeros(shape + (4,), np.uint8)
    if alpha is None:
        print("  [beaches] no Sand/WetSand layer; writing empty image")
    else:
        alpha[~mask] = 0.0
        h = BEACH_COLOR.lstrip("#")
        out[..., :3] = (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))
        out[..., 3] = np.round(alpha * 255.0).astype(np.uint8)
        out[out[..., 3] == 0, :3] = 0
    path = Path(BEACHES_DIR) / f"{region}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    imwrite_atomic(str(path), out)
    print(f"  [beaches] saved -> {short_path(str(path))}")
    return True
