"""Map-mod recipes: a JSON list of world layers composited into one image.

    {
      "name": "CustomMapMod",
      "root": "export/_final",
      "layers": [
        "assembly/base_layer.png",
        {"file": "assembly/contours.png", "opacity": 64},
        {"file": "svg_layers/foliage.png", "mask": "id/terrain.png"},
        {"shadow": ["assembly/roads.png", "svg_layers/bridges.png"]},
        ...
      ]
    }

Layers stack bottom to top. Each is a file path, or an object holding one
source and any number of adjustments:

  file      a world PNG
  shadow    a list of world PNGs: a soft drop shadow of their union,
            black unless "color" says otherwise; "blur" (default 1, the
            Gaussian sigma in px) and "strength" (default 1.5, alpha gain)
  opacity   0-255, scales the layer's alpha (paint.net's layer opacity)
  mask      a greyscale world PNG multiplied into the alpha (black hides)
  color     "#rrggbb": recolor every pixel, keeping the alpha

Paths are relative to "root", itself relative to the recipe file (default:
the recipe's own folder). Every image must be on the world canvas.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from toolset import mapmod

_KEYS = {"file", "shadow", "opacity", "mask", "color", "blur", "strength"}
_BAND = 512     # rows composited at once, to bound float32 temporaries


class RecipeError(ValueError):
    pass


@dataclass
class Layer:
    label: str
    file: Optional[Path] = None
    shadow: List[Path] = field(default_factory=list)
    opacity: int = 255
    mask: Optional[Path] = None
    color: Optional[Tuple[int, int, int]] = None    # BGR
    blur: float = 1.0
    strength: float = 1.5

    def files(self) -> List[Path]:
        return ([self.file] if self.file else self.shadow) + (
            [self.mask] if self.mask else [])


@dataclass
class Recipe:
    name: str
    layers: List[Layer]


# ------------------------------------------------------------------------------
#  Parsing
# ------------------------------------------------------------------------------

def _color(raw, where: str) -> Tuple[int, int, int]:
    s = str(raw).lstrip("#")
    try:
        if len(s) != 6:
            raise ValueError
        r, g, b = (int(s[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        raise RecipeError(f"{where}: color must be \"#rrggbb\", "
                          f"got {raw!r}") from None
    return b, g, r


def _layer(raw, root: Path, where: str) -> Layer:
    if isinstance(raw, str):
        raw = {"file": raw}
    if not isinstance(raw, dict):
        raise RecipeError(f"{where}: expected a path or an object")
    unknown = set(raw) - _KEYS
    if unknown:
        raise RecipeError(f"{where}: unknown key(s) {', '.join(sorted(unknown))}")
    if ("file" in raw) == ("shadow" in raw):
        raise RecipeError(f"{where}: needs exactly one of \"file\" or \"shadow\"")

    if "file" in raw:
        layer = Layer(label=Path(raw["file"]).stem, file=root / raw["file"])
    else:
        srcs = raw["shadow"]
        if isinstance(srcs, str):
            srcs = [srcs]
        if not srcs:
            raise RecipeError(f"{where}: empty shadow list")
        layer = Layer(label="shadow of " + ", ".join(Path(s).stem for s in srcs),
                      shadow=[root / s for s in srcs])
        layer.blur = float(raw.get("blur", layer.blur))
        layer.strength = float(raw.get("strength", layer.strength))
    if "file" in raw and ("blur" in raw or "strength" in raw):
        raise RecipeError(f"{where}: blur/strength only apply to a shadow")

    opacity = raw.get("opacity", 255)
    if not isinstance(opacity, int) or not 0 <= opacity <= 255:
        raise RecipeError(f"{where}: opacity must be an integer 0-255")
    layer.opacity = opacity
    if "mask" in raw:
        layer.mask = root / raw["mask"]
    if "color" in raw:
        layer.color = _color(raw["color"], where)
    return layer


def load(path) -> Recipe:
    """Parse and check a recipe file (the images aren't opened yet)."""
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        raise RecipeError(f"{path.name}: {exc}") from None
    if not isinstance(raw, dict) or not isinstance(raw.get("layers"), list):
        raise RecipeError(f"{path.name}: expected an object with a "
                          f"\"layers\" list")
    if not raw["layers"]:
        raise RecipeError(f"{path.name}: no layers")
    root = path.resolve().parent / raw.get("root", ".")
    layers = [_layer(lyr, root, f"{path.name}: layer {i + 1}")
              for i, lyr in enumerate(raw["layers"])]
    return Recipe(name=str(raw.get("name") or path.stem), layers=layers)


# ------------------------------------------------------------------------------
#  Compositing
# ------------------------------------------------------------------------------

def _load(path: Path, size: Tuple[int, int], gray: bool = False) -> np.ndarray:
    if gray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"could not read: {path}")
    else:
        img = mapmod.load_image(path)
    if img.shape[:2] != size:
        raise RecipeError(f"{path.name} is {img.shape[1]}x{img.shape[0]}, "
                          f"expected the {size[1]}x{size[0]} world canvas")
    return img


def _shadow(layer: Layer, size: Tuple[int, int]) -> np.ndarray:
    alpha = np.zeros(size, dtype=np.uint8)
    for src in layer.shadow:
        a = np.ascontiguousarray(_load(src, size)[..., 3])
        # union: alpha over alpha
        cv2.add(alpha, cv2.multiply(a, 255 - alpha, scale=1 / 255), dst=alpha)
        del a
    if layer.blur > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), layer.blur)
    alpha = cv2.convertScaleAbs(alpha, alpha=layer.strength)
    out = np.zeros((*size, 4), dtype=np.uint8)
    out[..., 3] = alpha
    return out


def _build(layer: Layer, size: Tuple[int, int]) -> np.ndarray:
    img = _shadow(layer, size) if layer.shadow else _load(layer.file, size)
    if layer.color is not None:
        img[..., :3] = layer.color
    if layer.mask is not None or layer.opacity != 255:
        alpha = np.ascontiguousarray(img[..., 3])
        if layer.mask is not None:
            cv2.multiply(alpha, _load(layer.mask, size, gray=True),
                         dst=alpha, scale=1 / 255)
        if layer.opacity != 255:
            lut = (np.arange(256) * layer.opacity // 255).astype(np.uint8)
            alpha = lut[alpha]
        img[..., 3] = alpha
    return img


def _over(dst: np.ndarray, src: np.ndarray) -> None:
    """Straight-alpha src over dst, in place, a band of rows at a time.
    Opaque src pixels are copied; only translucent ones (mostly edges)
    are blended."""
    for y0 in range(0, dst.shape[0], _BAND):
        y1 = y0 + _BAND
        sa = src[y0:y1, :, 3]
        part = sa < 255
        np.copyto(dst[y0:y1], src[y0:y1], where=~part[..., None])
        part &= sa > 0
        if not part.any():
            continue
        s = src[y0:y1][part].astype(np.float32) / 255
        d = dst[y0:y1][part].astype(np.float32) / 255
        a = s[:, 3:]
        da = d[:, 3:] * (1 - a)
        oa = a + da
        rgb = (s[:, :3] * a + d[:, :3] * da) / np.maximum(oa, 1e-6)
        out = np.concatenate([rgb, oa], axis=-1)
        dst[y0:y1][part] = np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


def missing(recipe: Recipe) -> List[Path]:
    """Every file the recipe names that isn't there."""
    return [f for lyr in recipe.layers for f in lyr.files()
            if not f.is_file()]


def compose(recipe: Recipe,
            on_layer: Optional[Callable[[Layer], None]] = None,
            ) -> np.ndarray:
    """The recipe's world image (BGRA); ``on_layer`` runs after each
    layer is composited."""
    size = mapmod.world_size(mapmod.centres())
    world = np.zeros((*size, 4), dtype=np.uint8)
    for layer in recipe.layers:
        img = _build(layer, size)
        _over(world, img)
        del img
        if on_layer is not None:
            on_layer(layer)
    return world
