"""Render per-region SVG layers into PNG.

For every layer in SVG_LAYERS (see utils/config.py), this module builds
a 2048x2048 SVG populated with <symbol>/<use> pairs from the region's
export/_json/<region>.json, then rasterizes it via cairosvg into
SVG_LAYERS_DIR/<layer>/<region>.png.

Each utils/svg/<category>/<name>.svg is wrapped once per region into a
<symbol id="<category>_<name>" overflow="visible"> whose children are
the original svg's inner content (no viewBox, so the symbol behaves as
a group and the svg's native coordinates are preserved). Every matching
placement in the region JSON emits one <use> with a transform of the
form `translate(tx ty) rotate(yaw) scale(sx sy)`. Categories are drawn
in the order listed in SVG_LAYERS[layer]; later categories paint on top
of earlier ones.
"""

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from utils.config import (
    JSON_DIR,
    SVG_DIR,
    SVG_LAYERS,
    SVG_LAYERS_DIR,
    TILE_SIZE,
)


# UE world-space is centimetres; 1 pixel = 1890/1776 m.
# cm -> px: x / 100 / (1890/1776) = x * 1776 / 189000. Origin offset by
# TILE_SIZE/2 so UE (0,0) lands at the region tile centre.
_SCALE = 1776.0 / 189000.0
_CENTER = TILE_SIZE / 2.0

_SVG_INNER_RE = re.compile(
    r"<svg\b[^>]*>(.*)</svg\s*>", re.DOTALL | re.IGNORECASE
)


def _xml_escape_attr(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
    )


def _load_svg_inner(path: Path) -> str:
    """Extract the inner content between the outermost <svg> tags."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    m = _SVG_INNER_RE.search(text)
    return m.group(1).strip() if m else ""


def _iter_mesh_placements(data: dict, name: str) -> Iterable[list]:
    """Yield every 9-tuple transform where `name` appears as a mesh
    (in "symbols", "groups", or nested inside a blueprint instance)."""
    for src_key in ("symbols", "groups"):
        src = data.get(src_key, {})
        for xf in src.get(name, []):
            yield xf
    for inst_list in data.get("blueprints", {}).values():
        for inst in inst_list:
            for xf in inst.get(name, []):
                yield xf


def _blueprint_self_placements(data: dict, name: str) -> List[list]:
    """Per-instance `_self` transforms for a blueprint class."""
    out: List[list] = []
    for inst in data.get("blueprints", {}).get(name, []):
        xf = inst.get("_self")
        if xf is not None:
            out.append(xf)
    return out


def _build_layer_svg(
    data: dict,
    categories: List[str],
) -> Tuple[str, int]:
    """Compose one layer's full SVG text. Returns (svg, n_placements).

    Symbols are emitted in first-use order inside <defs>; <use>
    elements are emitted in (category, svg-filename, placement-order)
    order so later categories paint on top of earlier ones.
    """
    bp_names = set(data.get("blueprints", {}).keys())

    symbol_defs: List[str] = []
    uses: List[str] = []
    seen: set = set()
    n_placed = 0

    for cat in categories:
        cat_dir = SVG_DIR / cat
        if not cat_dir.is_dir():
            continue
        for svg_path in sorted(cat_dir.glob("*.svg")):
            name = svg_path.stem
            if name in bp_names:
                xforms = _blueprint_self_placements(data, name)
            else:
                xforms = list(_iter_mesh_placements(data, name))
            if not xforms:
                continue

            sym_id = f"{cat}_{name}"
            if sym_id not in seen:
                inner = _load_svg_inner(svg_path)
                if not inner:
                    continue
                sid = _xml_escape_attr(sym_id)
                symbol_defs.append(
                    f'<symbol id="{sid}" overflow="visible">'
                    f"{inner}</symbol>"
                )
                seen.add(sym_id)

            sid = _xml_escape_attr(sym_id)
            for xf in xforms:
                # [x, y, z, sx, sy, sz, pitch, yaw, roll]
                if len(xf) < 9:
                    continue
                x, y = float(xf[0]), float(xf[1])
                sx, sy = float(xf[3]), float(xf[4])
                yaw = float(xf[7])
                tx = x * _SCALE + _CENTER
                ty = y * _SCALE + _CENTER
                uses.append(
                    f'<use href="#{sid}" transform="'
                    f'translate({tx:.3f} {ty:.3f}) '
                    f'rotate({yaw:.4f}) '
                    f'scale({sx:.5f} {sy:.5f})"/>'
                )
                n_placed += 1

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{TILE_SIZE}" height="{TILE_SIZE}" '
        f'viewBox="0 0 {TILE_SIZE} {TILE_SIZE}">'
        f'<defs>{"".join(symbol_defs)}</defs>'
        f'{"".join(uses)}'
        f'</svg>'
    )
    return svg, n_placed


def render_svg_layers(region_name: str) -> bool:
    """Rasterize every SVG_LAYERS entry for `region_name` into
    SVG_LAYERS_DIR/<layer>/<region_name>.png. Returns True unless an
    unrecoverable error (e.g. missing JSON) was hit."""
    import cairosvg  # local so the import is optional for non-svg runs

    json_path = JSON_DIR / f"{region_name}.json"
    if not json_path.is_file():
        print(f"  [WARN] SVG layers: no JSON at {json_path}; skipped")
        return False

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total_layers = len(SVG_LAYERS)
    w = len(str(max(total_layers, 1)))
    for i, (layer, cats) in enumerate(SVG_LAYERS.items(), 1):
        svg_text, n = _build_layer_svg(data, cats)
        out_dir = SVG_LAYERS_DIR / layer
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{region_name}.png"
        try:
            cairosvg.svg2png(
                bytestring=svg_text.encode("utf-8"),
                write_to=str(out_path),
                output_width=TILE_SIZE,
                output_height=TILE_SIZE,
            )
        except Exception as exc:
            print(f"  [WARN] svg layer '{layer}' rasterize failed: {exc}")
            continue
        print(f"    [{i:>{w}}/{total_layers}] {layer}: "
              f"{n} placement(s) -> {out_path.name}")
    return True
