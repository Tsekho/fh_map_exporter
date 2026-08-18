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

import heapq
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from utils.config import (
    BRIDGES_AIM_BLUEPRINTS,
    BRIDGES_AIM_CENTER_BIAS,
    BRIDGES_AIM_CHANNEL_PREF_PX,
    BRIDGES_AIM_COLOR,
    BRIDGES_AIM_CURVE_CHECK_STEP_PX,
    BRIDGES_AIM_FACE_MIN_DOT,
    BRIDGES_AIM_END_DEPTH_BIAS,
    BRIDGES_AIM_END_TRIM_PX,
    BRIDGES_AIM_GAP_PX,
    BRIDGES_AIM_LATERAL_BIAS,
    BRIDGES_AIM_LENGTH_PX,
    BRIDGES_AIM_MIN_CLEARANCE_PX,
    BRIDGES_AIM_MIN_CTRL_SPACING_PX,
    BRIDGES_AIM_PAIR_MAX_DETOUR,
    BRIDGES_AIM_REFINE_PASSES,
    BRIDGES_AIM_SNAP_DIST_PX,
    BRIDGES_AIM_SNAP_TO_WATER_PX,
    BRIDGES_AIM_STROKE_PX,
    BRIDGES_AIM_DIR,
    JSON_DIR,
    SVG_DIR,
    SVG_LAYERS,
    SVG_LAYERS_DIR,
    TILE_SIZE,
)
from utils.tui import log, warn

# The "bridges_aim" layer is rendered procedurally (see
# render_bridges_aim_layer) into its own BRIDGES_AIM_DIR rather than by
# stamping static symbols, so it is not part of SVG_LAYERS. This set is kept
# as a defensive guard for the generic _build_layer_svg loop.
_PROCEDURAL_LAYERS = {"bridges_aim"}


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
        warn(f"  SVG layers: no JSON at {json_path}; skipped")
        return False

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total_layers = len(SVG_LAYERS)
    w = len(str(max(total_layers, 1)))
    for i, (layer, cats) in enumerate(SVG_LAYERS.items(), 1):
        if layer in _PROCEDURAL_LAYERS:
            # Rendered separately (and after the ID bake) via
            # render_bridges_aim_layer; skip the generic symbol stamp.
            continue
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
            warn(f"  svg layer '{layer}' rasterize failed: {exc}")
            continue
        log(f"    [{i:>{w}}/{total_layers}] {layer}: "
              f"{n} placement(s) -> {out_path.name}")
    return True


# ------------------------------------------------------------------------------
#  Procedural bridge aim lines
# ------------------------------------------------------------------------------


class _Socket:
    """One aim-line emanating from a bridge centre along +/- the passage axis.

    cx, cy   bridge centre in tile pixels
    ux, uy   unit outward direction of this socket
    gap      pixels around the centre left blank before the line starts
    length   reach of the line from the centre (pre ground-cut)
    """

    __slots__ = ("bridge", "cx", "cy", "ux", "uy", "gap", "length", "matched")

    def __init__(self, bridge, cx, cy, ux, uy, gap, length):
        self.bridge = bridge
        self.cx = cx
        self.cy = cy
        self.ux = ux
        self.uy = uy
        self.gap = gap
        self.length = length
        self.matched = False


def _collect_bridge_sockets(data: dict) -> List[_Socket]:
    """Build two sockets per bridge placement (one per BRIDGES_AIM_BLUEPRINTS
    name found in the region JSON)."""
    bp_names = set(data.get("blueprints", {}).keys())
    sockets: List[_Socket] = []
    bridge_id = 0

    for name in BRIDGES_AIM_BLUEPRINTS:
        if name in bp_names:
            xforms = _blueprint_self_placements(data, name)
        else:
            xforms = list(_iter_mesh_placements(data, name))
        for xf in xforms:
            # [x, y, z, sx, sy, sz, pitch, yaw, roll]
            if len(xf) < 9:
                continue
            x, y = float(xf[0]), float(xf[1])
            sx = float(xf[3])
            yaw = float(xf[7])
            cx = x * _SCALE + _CENTER
            cy = y * _SCALE + _CENTER
            s = abs(sx) if sx else 1.0
            sign = -1.0 if sx < 0 else 1.0
            ang = math.radians(yaw)
            # Local +x axis after rotate(yaw) (SVG y points down).
            ux = sign * math.cos(ang)
            uy = sign * math.sin(ang)
            gap = BRIDGES_AIM_GAP_PX * s
            length = BRIDGES_AIM_LENGTH_PX * s
            sockets.append(_Socket(bridge_id, cx, cy, ux, uy, gap, length))
            sockets.append(_Socket(bridge_id, cx, cy, -ux, -uy, gap, length))
            bridge_id += 1
    return sockets


def _match_sockets(sockets: List[_Socket]) -> List[Tuple[_Socket, _Socket]]:
    """Greedily pair facing sockets on different bridges within the snap
    distance, closest pair first. Returns the matched pairs and marks each
    participating socket as ``matched``."""
    snap = BRIDGES_AIM_SNAP_DIST_PX
    face = BRIDGES_AIM_FACE_MIN_DOT

    candidates: List[Tuple[float, int, int]] = []
    n = len(sockets)
    for i in range(n):
        si = sockets[i]
        for j in range(i + 1, n):
            sj = sockets[j]
            if si.bridge == sj.bridge:
                continue
            wx = sj.cx - si.cx
            wy = sj.cy - si.cy
            dist = math.hypot(wx, wy)
            if dist <= 1e-6 or dist > snap:
                continue
            inv = 1.0 / dist
            wxh, wyh = wx * inv, wy * inv
            # Each socket must point roughly toward the other bridge.
            if (si.ux * wxh + si.uy * wyh) < face:
                continue
            if (sj.ux * -wxh + sj.uy * -wyh) < face:
                continue
            candidates.append((dist, i, j))

    candidates.sort(key=lambda c: c[0])
    pairs: List[Tuple[_Socket, _Socket]] = []
    for _dist, i, j in candidates:
        si, sj = sockets[i], sockets[j]
        if si.matched or sj.matched:
            continue
        si.matched = True
        sj.matched = True
        pairs.append((si, sj))
    return pairs


class _Nav:
    """Navigable-water context shared by every route in a region.

    ``mask[y, x]`` is True where the water coverage is eroded enough to keep
    a ship of radius ``clearance`` clear of any shore, i.e. distance-to-shore
    >= clearance. The bridge deck reads as non-water in ``water_dist`` and so
    is automatically excluded, acting as a wall that routes cannot cross."""

    __slots__ = ("water_dist", "clearance", "mask", "H", "W",
                 "bias", "pref")

    def __init__(self, water_dist: np.ndarray, clearance: float):
        self.water_dist = water_dist
        self.clearance = clearance
        self.mask = water_dist >= clearance
        self.H, self.W = self.mask.shape
        self.bias = float(BRIDGES_AIM_CENTER_BIAS)
        self.pref = max(float(BRIDGES_AIM_CHANNEL_PREF_PX), 1.0)

    def clear_at(self, x: float, y: float) -> bool:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= iy < self.H and 0 <= ix < self.W:
            return bool(self.mask[iy, ix])
        return False

    def shore_penalty(self, d: float) -> float:
        """Multiplicative step-cost surcharge (>= 0) that grows as the
        distance-to-shore ``d`` drops below the preferred channel depth, so
        routing favours the deep centre of the channel over the bank."""
        if d >= self.pref:
            return 0.0
        return self.bias * (self.pref - d) / self.pref


_SQ2 = math.sqrt(2.0)
_GRID_NBRS = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, _SQ2), (-1, 1, _SQ2), (1, -1, _SQ2), (1, 1, _SQ2),
)


def _dijkstra_reach(
    nav: _Nav, start: Tuple[int, int], budget: float
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], Tuple[int, int]],
           Dict[Tuple[int, int], float]]:
    """8-connected Dijkstra over the navigable mask from ``start``.

    Edge cost is the geometric step length surcharged by ``shore_penalty`` so
    the route prefers the channel centre. Expansion stops once the *geometric*
    geodesic distance exceeds ``budget`` (kept separately from the penalised
    cost so the reach stays a real pixel length). Returns (cost, came, geo)."""
    mask = nav.mask
    H, W = nav.H, nav.W
    wd = nav.water_dist
    sr, sc = start
    cost_g: Dict[Tuple[int, int], float] = {start: 0.0}
    geo: Dict[Tuple[int, int], float] = {start: 0.0}
    came: Dict[Tuple[int, int], Tuple[int, int]] = {}
    pq: List[Tuple[float, int, int]] = [(0.0, sr, sc)]
    while pq:
        cg, r, c = heapq.heappop(pq)
        if cg > cost_g.get((r, c), 1e18):
            continue
        if geo[(r, c)] >= budget:
            continue  # settled, but don't expand past the reach budget
        for dr, dc, step in _GRID_NBRS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W or not mask[nr, nc]:
                continue
            if dr != 0 and dc != 0 and (not mask[r + dr, c]
                                        or not mask[r, c + dc]):
                continue
            ncg = cg + step * (1.0 + nav.shore_penalty(float(wd[nr, nc])))
            if ncg < cost_g.get((nr, nc), 1e18):
                cost_g[(nr, nc)] = ncg
                geo[(nr, nc)] = geo[(r, c)] + step
                came[(nr, nc)] = (r, c)
                heapq.heappush(pq, (ncg, nr, nc))
    return cost_g, came, geo


def _trace_back(
    came: Dict[Tuple[int, int], Tuple[int, int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> List[Tuple[int, int]]:
    path = [goal]
    cur = goal
    while cur != start and cur in came:
        cur = came[cur]
        path.append(cur)
    path.reverse()
    return path


# ---- curve control points: simplify, then repair for clearance -------------

_CR_ALPHA = 0.5  # centripetal parameterisation (no overshoot, no cusps)


def _seg_bezier(C: List[Tuple[float, float]], i: int):
    """Cubic bezier (p0, c1, c2, p1) for the centripetal Catmull-Rom span
    C[i]->C[i+1].

    Centripetal (alpha=0.5) parameterisation is used instead of the uniform
    variant because uniform Catmull-Rom overshoots and forms little loops/
    cusps when control points are unevenly spaced -- the source of the stray
    S-shapes. Missing end neighbours are reflected (p0 = 2*p1 - p2) so the
    spline leaves its endpoints tangent to the first/last chord, i.e. it
    starts straight out of the bridge with no hook."""
    n = len(C)
    p1 = C[i]
    p2 = C[i + 1]
    p0 = C[i - 1] if i > 0 else (2.0 * p1[0] - p2[0], 2.0 * p1[1] - p2[1])
    p3 = C[i + 2] if i + 2 < n else (2.0 * p2[0] - p1[0], 2.0 * p2[1] - p1[1])

    def _tnext(t: float, a, b) -> float:
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        return t + (d ** _CR_ALPHA if d > 1e-9 else 1e-4)

    t0 = 0.0
    t1 = _tnext(t0, p0, p1)
    t2 = _tnext(t1, p1, p2)
    t3 = _tnext(t2, p2, p3)

    # Tangents at p1 and p2 (Barry-Goldman), then Hermite -> bezier.
    def _tan(a, b, e, ta, tb, te):
        # d/dt at b for the three-point non-uniform form.
        return (
            (b[0] - a[0]) / (tb - ta) - (e[0] - a[0]) / (te - ta)
            + (e[0] - b[0]) / (te - tb),
            (b[1] - a[1]) / (tb - ta) - (e[1] - a[1]) / (te - ta)
            + (e[1] - b[1]) / (te - tb),
        )

    m1 = _tan(p0, p1, p2, t0, t1, t2)
    m2 = _tan(p1, p2, p3, t1, t2, t3)
    f = (t2 - t1) / 3.0
    c1 = (p1[0] + m1[0] * f, p1[1] + m1[1] * f)
    c2 = (p2[0] - m2[0] * f, p2[1] - m2[1] * f)
    return p1, c1, c2, p2


def _bezier_at(p0, c1, c2, p1, u: float) -> Tuple[float, float]:
    mt = 1.0 - u
    a, b = mt * mt * mt, 3.0 * mt * mt * u
    c, d = 3.0 * mt * u * u, u * u * u
    return (a * p0[0] + b * c1[0] + c * c2[0] + d * p1[0],
            a * p0[1] + b * c1[1] + c * c2[1] + d * p1[1])


def _smooth_path_d(pts: List[Tuple[float, float]]) -> str:
    """Smooth SVG path string through ``pts`` (Catmull-Rom -> cubic beziers).
    Falls back to straight segments for fewer than three points."""
    n = len(pts)
    if n == 0:
        return ""
    if n < 3:
        return "M " + " L ".join(f"{x:.3f} {y:.3f}" for x, y in pts)
    d = [f"M {pts[0][0]:.3f} {pts[0][1]:.3f}"]
    for i in range(n - 1):
        _p0, c1, c2, p1 = _seg_bezier(pts, i)
        d.append(
            f"C {c1[0]:.3f} {c1[1]:.3f} {c2[0]:.3f} {c2[1]:.3f} "
            f"{p1[0]:.3f} {p1[1]:.3f}"
        )
    return " ".join(d)


def _los_clear(a: Tuple[float, float], b: Tuple[float, float],
               nav: _Nav, step: float = 1.0) -> bool:
    """True when the straight segment a->b is fully navigable (line of
    sight). Sampled at ~1 px so a thin bridge deck between the endpoints is
    never skipped."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy)
    n = max(1, int(L / max(step, 0.25)))
    for k in range(n + 1):
        t = k / n
        if not nav.clear_at(a[0] + dx * t, a[1] + dy * t):
            return False
    return True


def _string_pull(
    P: List[Tuple[float, float]], nav: _Nav
) -> List[Tuple[float, float]]:
    """Taut-string simplification of a dense navigable route.

    From each anchor, keep the *farthest* later vertex still in line of sight
    and make it the next anchor. The result goes as straight as the channel
    allows and only turns where an obstacle forces it -- so it neither
    squiggles down the medial axis nor detours, and every segment's straight
    chord is navigable (it can't cut across a bridge deck)."""
    n = len(P)
    if n <= 2:
        return list(P)
    out = [P[0]]
    i = 0
    while i < n - 1:
        nxt = i + 1
        for j in range(n - 1, i, -1):  # prefer the farthest visible vertex
            if _los_clear(P[i], P[j], nav):
                nxt = j
                break
        out.append(P[nxt])
        i = nxt
    return out


def _span_clear(C: List[Tuple[float, float]], k: int, nav: _Nav,
                step: float) -> bool:
    """True when the spline span C[k]->C[k+1] stays navigable (the endpoints
    are control points, assumed clear, so only the interior is sampled)."""
    p0, c1, c2, p1 = _seg_bezier(C, k)
    seglen = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    nsamp = max(2, int(seglen / max(step, 0.5)))
    for s in range(1, nsamp):
        x, y = _bezier_at(p0, c1, c2, p1, s / nsamp)
        if not nav.clear_at(x, y):
            return False
    return True


def _curve_control_points(
    P: List[Tuple[float, float]], nav: _Nav
) -> List[Tuple[float, float]]:
    """Turn a dense navigable route ``P`` into sparse curve control points.

    String-pulls ``P`` into a taut polyline (few, deliberate turns), then
    rounds it with the centripetal spline. Where that rounded span bulges out
    of the navigable band it is subdivided at the chord midpoint -- which is
    itself navigable, because every taut chord is line-of-sight clear -- until
    the curve is clear or the span drops below MIN_CTRL_SPACING_PX, the
    close-control-point exception where smoothness wins over clearance."""
    pts = _string_pull(P, nav)
    if len(pts) <= 2:
        return pts

    min_spacing = BRIDGES_AIM_MIN_CTRL_SPACING_PX
    step = BRIDGES_AIM_CURVE_CHECK_STEP_PX

    for _ in range(max(int(BRIDGES_AIM_REFINE_PASSES), 0)):
        inserts: Dict[int, Tuple[float, float]] = {}
        for k in range(len(pts) - 1):
            a, b = pts[k], pts[k + 1]
            if math.hypot(b[0] - a[0], b[1] - a[1]) <= min_spacing:
                continue  # exception: too close to bother enforcing
            if not _span_clear(pts, k, nav, step):
                inserts[k] = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
        if not inserts:
            break
        rebuilt: List[Tuple[float, float]] = []
        for k, p in enumerate(pts):
            rebuilt.append(p)
            if k in inserts:
                rebuilt.append(inserts[k])
        pts = rebuilt

    return pts


def _axis_entry(
    nav: _Nav, sx: float, sy: float, ux: float, uy: float, max_march: float
) -> Optional[Tuple[int, int]]:
    """Navigable entry cell in front of a bridge gap.

    First marches straight along the socket axis (sx, sy)+(ux, uy), so the
    aim line emanates straight out of the bridge instead of hooking sideways.
    If the axis ray misses the water (e.g. the bridge is slightly misaligned
    with the channel) it falls back to the nearest navigable cell that still
    lies *forward* of the gap (projection on the socket axis >= 0).

    The forward restriction is essential: snapping to the nearest cell in any
    direction can land the anchor behind the *other* bridge's deck, which
    forces the route to loop the long way around it -- the U-shape that
    crosses back over the bridge. Returns None when no water lies ahead."""
    n = int(max_march)
    for k in range(n + 1):
        ix = int(round(sx + ux * k))
        iy = int(round(sy + uy * k))
        if 0 <= iy < nav.H and 0 <= ix < nav.W and nav.mask[iy, ix]:
            return (iy, ix)

    ci, ri = int(round(sx)), int(round(sy))
    best: Optional[Tuple[int, int]] = None
    best_d = 1e18
    for dr in range(-n, n + 1):
        for dc in range(-n, n + 1):
            if dc * ux + dr * uy < 0.0:
                continue  # never anchor behind the bridge
            r, c = ri + dr, ci + dc
            if 0 <= r < nav.H and 0 <= c < nav.W and nav.mask[r, c]:
                d = dr * dr + dc * dc
                if d < best_d:
                    best_d, best = d, (r, c)
    return best


def _trim_shoreward_tail(
    pts: List[Tuple[float, float]], nav: _Nav, max_trim: float
) -> List[Tuple[float, float]]:
    """Cut back a route tail that descends toward shore.

    Within the last ``max_trim`` path-pixels, the route is truncated at its
    deepest cell (largest distance-to-shore). A tail whose depth only
    decreases -- an approach into the bank -- is removed entirely from the
    deepest point on; a tail through open water (depth flat or rising toward
    the end) is left untouched."""
    if max_trim <= 0.0 or len(pts) < 3:
        return pts
    wd = nav.water_dist
    best_i = len(pts) - 1
    best_d = float(wd[int(pts[-1][1]), int(pts[-1][0])])
    acc = 0.0
    i = len(pts) - 1
    while i > 1 and acc < max_trim:
        x1, y1 = pts[i]
        x0, y0 = pts[i - 1]
        acc += math.hypot(x1 - x0, y1 - y0)
        i -= 1
        d = float(wd[int(pts[i][1]), int(pts[i][0])])
        if d > best_d:
            best_d, best_i = d, i
    return pts[:best_i + 1]


def _trace_outward(sock: _Socket, nav: _Nav) -> Optional[List[Tuple[float, float]]]:
    """Route an unpaired socket outward along the channel.

    Enters the water on the socket axis, runs a budgeted centre-biased
    Dijkstra, and picks the endpoint that reaches furthest *along* the
    socket's outward direction (so the line follows the river -- or heads
    into open water for a large body -- and stops where the water ends).
    Returns a dense navigable polyline (anchor..goal) or None when no water
    is reachable near the bridge."""
    gx = sock.cx + sock.gap * sock.ux
    gy = sock.cy + sock.gap * sock.uy

    start = _axis_entry(nav, gx, gy, sock.ux, sock.uy,
                        BRIDGES_AIM_SNAP_TO_WATER_PX)
    if start is None:
        return None

    cost_g, came, geo = _dijkstra_reach(nav, start, sock.length)
    if len(geo) < 2:
        return None

    # Endpoint score: along-axis reach minus penalties for sideways offset
    # and for ending in shallow water.
    #
    # Sideways: the reach budget is octile grid distance, so in open water
    # the raw farthest-projection cell snaps to the nearest compass direction
    # (the reachable set is an octagon with vertices on the 8 grid
    # directions); penalising the perpendicular offset makes the on-axis cell
    # win unless the channel actually bends the route.
    #
    # Shallow: without a depth term the endpoint drifts into whichever
    # near-shore pocket offers a few extra pixels of reach, which is rarely
    # line-of-sight from the main chord and so leaves a tiny final vertex --
    # a hook aiming the line into the shore.
    sr, sc = start
    wd = nav.water_dist
    best = start
    best_score = -1e18
    for (r, c) in geo:
        dx, dy = c - sc, r - sr
        proj = dx * sock.ux + dy * sock.uy
        if proj <= 0.0:
            continue
        perp = abs(dx * -sock.uy + dy * sock.ux)
        shallow = max(0.0, nav.pref - float(wd[r, c]))
        score = (proj - BRIDGES_AIM_LATERAL_BIAS * perp
                 - BRIDGES_AIM_END_DEPTH_BIAS * shallow)
        if score > best_score:
            best_score, best = score, (r, c)

    if best == start:
        # No water ahead along the axis; follow the channel to its far end.
        best = max(geo, key=lambda rc: geo[rc])
        if best == start:
            return None

    cells = _trace_back(came, start, best)
    pts = [(float(c), float(r)) for r, c in cells]
    pts = _trim_shoreward_tail(pts, nav, BRIDGES_AIM_END_TRIM_PX)
    return pts if len(pts) >= 2 else None


def _astar_grid(
    nav: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
    depth: Optional[np.ndarray] = None,
    bias: float = 0.0, pref: float = 1.0,
) -> Optional[List[Tuple[int, int]]]:
    """8-connected A* over the boolean ``nav`` grid (True == navigable).

    Returns the cell path [start..goal] or None when no navigable route
    exists. Diagonal moves can't cut a land corner. When ``depth`` (a
    distance-to-shore array matching ``nav``) is given, the step cost is
    surcharged for shallow cells (``bias`` * how far below ``pref``), so the
    route follows the channel centre instead of hugging the clearance
    boundary -- this is what removes the staircase and the bank-to-bank
    wiggle. The octile heuristic stays admissible because the minimum step
    cost is still the geometric length (penalty >= 0)."""
    H, W = nav.shape
    sr, sc = start
    gr, gc = goal
    if not (nav[sr, sc] and nav[gr, gc]):
        return None

    SQ = math.sqrt(2.0)
    nbrs = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, SQ), (-1, 1, SQ), (1, -1, SQ), (1, 1, SQ),
    ]

    def hcost(r: int, c: int) -> float:
        dr, dc = abs(r - gr), abs(c - gc)
        lo, hi = (dr, dc) if dr < dc else (dc, dr)
        return (hi - lo) + SQ * lo

    def penalty(r: int, c: int) -> float:
        if depth is None:
            return 0.0
        d = float(depth[r, c])
        return bias * (pref - d) / pref if d < pref else 0.0

    g: Dict[Tuple[int, int], float] = {(sr, sc): 0.0}
    came: Dict[Tuple[int, int], Tuple[int, int]] = {}
    openh: List[Tuple[float, float, int, int]] = [(hcost(sr, sc), 0.0, sr, sc)]
    closed: set = set()

    while openh:
        _f, gcur, r, c = heapq.heappop(openh)
        if (r, c) in closed:
            continue
        closed.add((r, c))
        if (r, c) == (gr, gc):
            path = [(r, c)]
            while (r, c) in came:
                r, c = came[(r, c)]
                path.append((r, c))
            path.reverse()
            return path
        for dr, dc, cost in nbrs:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                continue
            if not nav[nr, nc]:
                continue
            if dr != 0 and dc != 0 and (not nav[r + dr, c]
                                        or not nav[r, c + dc]):
                continue  # don't squeeze diagonally past a land corner
            ng = gcur + cost * (1.0 + penalty(nr, nc))
            if ng < g.get((nr, nc), 1e18):
                g[(nr, nc)] = ng
                came[(nr, nc)] = (r, c)
                heapq.heappush(openh, (ng + hcost(nr, nc), ng, nr, nc))
    return None


def _trace_pair_path(
    si: "_Socket", sj: "_Socket", nav: _Nav
) -> Optional[List[Tuple[float, float]]]:
    """Route the navigable channel linking two snapped bridge gaps.

    Runs A* over the navigable mask inside a padded bounding box around the
    two gaps, so the route is guaranteed to keep MIN_CLEARANCE_PX from land
    while connecting both crossings -- bending around headlands/islands and
    never crossing another bridge deck (which reads as non-navigable).

    Returns the dense route between the two snapped anchors (no gap points),
    or None when no navigable route exists so the caller can fall back to a
    plain bezier."""
    ax, ay = si.cx + si.gap * si.ux, si.cy + si.gap * si.uy
    bx, by = sj.cx + sj.gap * sj.ux, sj.cy + sj.gap * sj.uy

    span = math.hypot(bx - ax, by - ay)
    # Pad generously so a detour around an obstacle still fits the window.
    pad = int(max(96.0, span))
    x0 = max(0, int(min(ax, bx)) - pad)
    x1 = min(nav.W, int(max(ax, bx)) + pad + 1)
    y0 = max(0, int(min(ay, by)) - pad)
    y1 = min(nav.H, int(max(ay, by)) + pad + 1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None

    sub = nav.mask[y0:y1, x0:x1]
    wsub = nav.water_dist[y0:y1, x0:x1]

    # Enter the water on each socket's axis, forward of the bridge only. If an
    # axis entry can't be found ahead (or lands outside this window) we give
    # up and let the caller draw the direct connector -- we deliberately do
    # NOT snap to the nearest water in any direction, since that can anchor
    # behind the other bridge and force a loop around it.
    ea = _axis_entry(nav, ax, ay, si.ux, si.uy, BRIDGES_AIM_SNAP_TO_WATER_PX)
    eb = _axis_entry(nav, bx, by, sj.ux, sj.uy, BRIDGES_AIM_SNAP_TO_WATER_PX)
    if ea is None or eb is None:
        return None

    sar, sac = ea[0] - y0, ea[1] - x0
    sbr, sbc = eb[0] - y0, eb[1] - x0
    hh, ww = y1 - y0, x1 - x0
    if not (0 <= sar < hh and 0 <= sac < ww and sub[sar, sac]
            and 0 <= sbr < hh and 0 <= sbc < ww and sub[sbr, sbc]):
        return None
    sa, sb = (sar, sac), (sbr, sbc)

    cells = _astar_grid(sub, sa, sb, depth=wsub,
                        bias=nav.bias, pref=nav.pref)
    if not cells:
        return None

    pts = [(float(x0 + cc), float(y0 + rr)) for rr, cc in cells]

    # Reject gross detours. When the direct channel between two close,
    # slightly misaligned bridges is pinched shut by the erosion, A* loops the
    # long way around the far end of the other deck and crosses back over it.
    # Such a route is far longer than the straight gap span; drop it so the
    # caller falls back to a short, direct connector instead.
    plen = sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
               for i in range(len(pts) - 1))
    if span > 1.0 and plen > BRIDGES_AIM_PAIR_MAX_DETOUR * span:
        return None
    return pts


def _path_elem(d: str) -> str:
    return (
        f'<path d="{d}" fill="none" stroke="{BRIDGES_AIM_COLOR}" '
        f'stroke-width="{BRIDGES_AIM_STROKE_PX}" stroke-linecap="round" '
        f'stroke-linejoin="round"/>'
    )


def _bridges_aim_svg(
    sockets: List[_Socket],
    pairs: List[Tuple[_Socket, _Socket]],
    nav: Optional[_Nav],
) -> Tuple[str, int]:
    """Compose the bridges_aim SVG. Returns (svg, n_lines).

    With a navigable field every line is routed through eroded water and
    reduced to a smooth, clearance-respecting spline. Without one (nav None)
    the lines degrade to the legacy straight stubs."""
    elems: List[str] = []

    # --- snapped pairs: connect the two gaps through the channel -----------
    for si, sj in pairs:
        ax, ay = si.cx + si.gap * si.ux, si.cy + si.gap * si.uy
        bx, by = sj.cx + sj.gap * sj.ux, sj.cy + sj.gap * sj.uy
        core = _trace_pair_path(si, sj, nav) if nav is not None else None
        if core is not None and len(core) >= 2:
            ctrl = _curve_control_points(core, nav)
            full = [(ax, ay)] + ctrl + [(bx, by)]
            d = _smooth_path_d(full)
        else:
            # Fallback: a plain facing-tangent bezier between the two gaps.
            k = math.hypot(bx - ax, by - ay) * 0.5
            c1x, c1y = ax + si.ux * k, ay + si.uy * k
            c2x, c2y = bx + sj.ux * k, by + sj.uy * k
            d = (f'M {ax:.3f} {ay:.3f} '
                 f'C {c1x:.3f} {c1y:.3f} {c2x:.3f} {c2y:.3f} '
                 f'{bx:.3f} {by:.3f}')
        elems.append(_path_elem(d))

    # --- unmatched sockets: extend outward along the channel ---------------
    for sock in sockets:
        if sock.matched:
            continue
        gx = sock.cx + sock.gap * sock.ux
        gy = sock.cy + sock.gap * sock.uy
        core = _trace_outward(sock, nav) if nav is not None else None
        if core is not None and len(core) >= 2:
            ctrl = _curve_control_points(core, nav)
            full = [(gx, gy)] + ctrl
            d = _smooth_path_d(full)
        elif nav is None:
            # No water field: legacy straight full-length stub.
            ex = sock.cx + sock.length * sock.ux
            ey = sock.cy + sock.length * sock.uy
            d = f'M {gx:.3f} {gy:.3f} L {ex:.3f} {ey:.3f}'
        else:
            continue  # no reachable water near this bridge: draw nothing
        elems.append(_path_elem(d))

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{TILE_SIZE}" height="{TILE_SIZE}" '
        f'viewBox="0 0 {TILE_SIZE} {TILE_SIZE}">'
        f'{"".join(elems)}'
        f'</svg>'
    )
    return svg, len(elems)


def render_bridges_aim_layer(
    region_name: str,
    water_dist: Optional[np.ndarray] = None,
) -> bool:
    """Render the procedural ``bridges_aim`` layer into
    BRIDGES_AIM_DIR/<region_name>.png.

    ``water_dist`` is an optional float distance-to-shore field (0 on
    non-water) used to route the aim lines through navigable water. When None
    the lines degrade to straight full-length stubs (no clearance routing)."""
    import cairosvg

    json_path = JSON_DIR / f"{region_name}.json"
    if not json_path.is_file():
        warn(f"  bridges_aim: no JSON at {json_path}; skipped")
        return False
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    sockets = _collect_bridge_sockets(data)
    nav = (_Nav(water_dist, BRIDGES_AIM_MIN_CLEARANCE_PX)
           if water_dist is not None else None)

    out_dir = BRIDGES_AIM_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{region_name}.png"

    if not sockets:
        # Still emit a blank tile so the stitch step has a consistent input.
        svg, n = _bridges_aim_svg([], [], nav)
        pairs: List[Tuple[_Socket, _Socket]] = []
    else:
        pairs = _match_sockets(sockets)
        svg, n = _bridges_aim_svg(sockets, pairs, nav)

    try:
        cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            write_to=str(out_path),
            output_width=TILE_SIZE,
            output_height=TILE_SIZE,
        )
    except Exception as exc:
        warn(f"  bridges_aim rasterize failed: {exc}")
        return False

    n_pairs = len(pairs) if sockets else 0
    log(f"  [bridges_aim] {len(sockets)} socket(s), {n_pairs} snapped "
          f"pair(s) -> {out_path.name}")
    return True
