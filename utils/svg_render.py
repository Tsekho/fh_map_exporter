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
    BRIDGES_AIM_DIRECT_MAX_SPAN_PX,
    BRIDGES_AIM_ENTRY_MAX_ANGLE_DEG,
    BRIDGES_AIM_FACE_MIN_DOT,
    BRIDGES_AIM_END_DEPTH_BIAS,
    BRIDGES_AIM_END_TRIM_PX,
    BRIDGES_AIM_GAP_PX,
    BRIDGES_AIM_LATERAL_BIAS,
    BRIDGES_AIM_LEAD_IN_MAX_SPAN_FRAC,
    BRIDGES_AIM_LEAD_IN_MIN_DEV_DEG,
    BRIDGES_AIM_LEAD_IN_PX,
    BRIDGES_AIM_LENGTH_PX,
    BRIDGES_AIM_MIN_CLEARANCE_PX,
    BRIDGES_AIM_MIN_CTRL_SPACING_PX,
    BRIDGES_AIM_PAIR_MAX_DETOUR,
    BRIDGES_AIM_REFINE_PASSES,
    BRIDGES_AIM_RELAX_CLEARANCE_PX,
    BRIDGES_AIM_RELAXED_MAX_BOW_FRAC,
    BRIDGES_AIM_RELAXED_MAX_BOW_PX,
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

# The "bridges_aim" layer is rendered procedurally (see
# render_bridges_aim_layer) into its own BRIDGES_AIM_DIR rather than by
# stamping static symbols, so it is not part of SVG_LAYERS. This set is kept
# as a defensive guard for the generic _build_layer_svg loop.
PROCEDURAL_LAYERS = {"bridges_aim"}

# The layers render_svg_layers() actually rasterizes into
# SVG_LAYERS_DIR/<layer>/ -- SVG_LAYERS minus anything procedural. Callers
# sizing a progress bar or an expected-output count use this, not SVG_LAYERS,
# so adding a procedural layer to SVG_LAYERS never leaves a bar one short.
SVG_FILE_LAYERS = [l for l in SVG_LAYERS if l not in PROCEDURAL_LAYERS]


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

    ok = True
    # Procedural layers are rendered separately (and after the ID bake) via
    # render_bridges_aim_layer, so they are not counted here either.
    total_layers = len(SVG_FILE_LAYERS)
    w = len(str(max(total_layers, 1)))
    for i, layer in enumerate(SVG_FILE_LAYERS, 1):
        svg_text, n = _build_layer_svg(data, SVG_LAYERS[layer])
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
            # Report it: a missing layer PNG silently shorts the caller's
            # expected-output count, so the region must not read as clean.
            print(f"  [WARN] svg layer '{layer}' rasterize failed: {exc}")
            ok = False
            continue
        print(f"    [{i:>{w}}/{total_layers}] {layer}: "
              f"{n} placement(s) -> {out_path.name}")
    return ok


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
                 "bias", "pref", "water_mask")

    def __init__(self, water_dist: np.ndarray, clearance: float,
                 water_mask: Optional[np.ndarray] = None):
        self.water_dist = water_dist
        self.clearance = clearance
        self.mask = water_dist >= clearance
        # Raw water coverage, before the navigability depth gate. Routing
        # asks "can a freighter sail here"; on_water() asks the much weaker
        # "is this water rather than land", and the two differ wherever a
        # channel is real but shallow. Falls back to the routing field when
        # the caller has no coverage mask to give.
        self.water_mask = water_mask
        self.H, self.W = self.mask.shape
        self.bias = float(BRIDGES_AIM_CENTER_BIAS)
        self.pref = max(float(BRIDGES_AIM_CHANNEL_PREF_PX), 1.0)

    def clear_at(self, x: float, y: float) -> bool:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= iy < self.H and 0 <= ix < self.W:
            return bool(self.mask[iy, ix])
        return False

    def relaxed(self, clearance: float) -> "_Nav":
        """A view of the same water with a smaller ship radius."""
        return _Nav(self.water_dist, clearance, self.water_mask)

    def on_water(self, x: float, y: float) -> bool:
        """True where the pixel is water at all -- not land, and not a deck.

        Deliberately the loosest test available: no clearance erosion and no
        depth gate. It answers "would a line drawn here cross terrain", which
        is a different and much weaker question than "can a ship sail here".
        A shallow passage is unroutable but still water, and a line along it
        is correct even though no freighter could use it."""
        ix, iy = int(round(x)), int(round(y))
        if not (0 <= iy < self.H and 0 <= ix < self.W):
            return False
        if self.water_mask is not None:
            return bool(self.water_mask[iy, ix])
        return bool(self.water_dist[iy, ix] > 0.0)

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
    return _refine_curve(pts, nav)


def _refine_curve(
    pts: List[Tuple[float, float]], nav: _Nav,
    skip_first: bool = False, skip_last: bool = False,
) -> List[Tuple[float, float]]:
    """Subdivide spans whose rounded curve bulges out of the navigable band.

    A span is split at its chord midpoint -- itself navigable, because every
    taut chord is line-of-sight clear -- until the curve is clear or the span
    drops below MIN_CTRL_SPACING_PX (the close-control-point exception where
    smoothness wins over clearance).

    ``skip_first``/``skip_last`` exempt the spans anchored on a bridge gap.
    Those deliberately cross the deck, which is non-navigable by
    construction, so testing them would subdivide forever. Everything between
    them is enforced -- and must be re-enforced *after* the gap anchors are
    attached, since prepending a point changes the spline's tangents and can
    push a span that was clear on its own out over land."""
    min_spacing = BRIDGES_AIM_MIN_CTRL_SPACING_PX
    step = BRIDGES_AIM_CURVE_CHECK_STEP_PX

    for _ in range(max(int(BRIDGES_AIM_REFINE_PASSES), 0)):
        inserts: Dict[int, Tuple[float, float]] = {}
        last = len(pts) - 2
        for k in range(len(pts) - 1):
            if (skip_first and k == 0) or (skip_last and k == last):
                continue
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


def _bow_off_chord(
    pts: List[Tuple[float, float]],
    a: Tuple[float, float], b: Tuple[float, float],
) -> float:
    """Largest perpendicular distance from ``pts`` to the chord a->b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return 0.0
    ux, uy = dx / L, dy / L
    return max(abs((x - a[0]) * -uy + (y - a[1]) * ux) for x, y in pts)


def _detour_is_contrived(
    core: List[Tuple[float, float]], rnav: _Nav, nav: _Nav,
    a: Tuple[float, float], b: Tuple[float, float],
) -> bool:
    """True when a route should be discarded in favour of a direct line.

    Only ever true for a route that needed a relaxed clearance. Those are
    found by scraping through whatever pockets survive in an almost-empty
    navigable mask -- a shallow passage the depth gate has all but erased --
    and the path can swing far off the line the two bridges plainly want,
    which draws as an L or a hook between two crossings that face each other.
    A route at the full clearance is trusted however much it bends, because
    there the bend is a real channel going round something."""
    if rnav.clearance >= nav.clearance:
        return False
    span = math.hypot(b[0] - a[0], b[1] - a[1])
    bow = _bow_off_chord(core, a, b)
    # Both thresholds must trip; the fraction alone fires on near-zero spans.
    return (bow > float(BRIDGES_AIM_RELAXED_MAX_BOW_PX)
            and bow > float(BRIDGES_AIM_RELAXED_MAX_BOW_FRAC) * span)


def _polyline_len(pts: List[Tuple[float, float]]) -> float:
    return sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
               for i in range(len(pts) - 1))


def _departure_dir(
    gx: float, gy: float, ux: float, uy: float,
    ctrl: List[Tuple[float, float]], min_dist: float = 3.0,
) -> Optional[float]:
    """cos(angle) between the socket axis and the curve's opening tangent.

    The spline leaves the gap along the chord to its first control point, so
    that chord is the departure. Points closer than ``min_dist`` are skipped:
    a control point one or two pixels off the gap gives a direction dominated
    by grid rounding, not by where the line is actually going. None when no
    control point is far enough out to give a stable answer."""
    for (x, y) in ctrl:
        dx, dy = x - gx, y - gy
        d = math.hypot(dx, dy)
        if d >= min_dist:
            return (dx * ux + dy * uy) / d
    return None


def _lead_in(
    gx: float, gy: float, ux: float, uy: float,
    ctrl: List[Tuple[float, float]], nav: _Nav,
    route_len: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Force a straight run along the socket axis before the route may bend.

    A ship leaving a bridge is committed to the passage axis for at least a
    hull length; it cannot turn the moment it clears the deck. This replaces
    every control point within the lead distance of the gap (measured along
    the axis) with a single on-axis point at the end of the longest straight
    navigable run, so the curve's opening tangent is the passage axis itself.

    It is a correction for an implausible departure, not a rule applied to
    every line, and it only engages when the route actually departs more than
    LEAD_IN_MIN_DEV_DEG off the axis. Bridges facing each other across a
    channel are rarely exactly collinear; straightening each end onto its own
    slightly-off axis bends the line one way and then back the other, which
    is what turns a smooth link between two near-aligned crossings into an S.
    Within the tolerance the departure is already plausible -- the entry cone
    bounds it -- so the curve is left alone.

    ``route_len`` (the length of the route this lead-in attaches to) caps the
    run at LEAD_IN_MAX_SPAN_FRAC of it, so a fixed lead can't swallow a short
    connector and leave no room to blend back into the channel.

    The substitution is dropped rather than forced when the run can't be
    found or the join to the remaining route isn't clear."""
    lead = float(BRIDGES_AIM_LEAD_IN_PX)
    if route_len is not None:
        lead = min(lead, float(BRIDGES_AIM_LEAD_IN_MAX_SPAN_FRAC) * route_len)
    if lead <= 0.0 or len(ctrl) < 2:
        return ctrl

    # Longest uninterrupted navigable run along the axis, within the lead.
    first: Optional[float] = None
    run = 0.0
    k = 0.0
    while k <= lead + 1e-9:
        if nav.clear_at(gx + ux * k, gy + uy * k):
            if first is None:
                first = k
            run = k
        elif first is not None:
            break  # left the water again; keep the run up to here
        k += 1.0
    if first is None or run <= first:
        return ctrl

    px, py = gx + ux * run, gy + uy * run
    # Strip only the *leading* run of swallowed points. Filtering the whole
    # list would also drop a later point that happens to sit back near the
    # bridge -- a route that doubles back on itself -- and silently cut the
    # middle out of the line.
    i = 0
    while (i < len(ctrl)
           and (ctrl[i][0] - gx) * ux + (ctrl[i][1] - gy) * uy <= run):
        i += 1
    rest = ctrl[i:]
    if not rest:
        return ctrl

    # Leave a departure that is already plausible alone (see above). The
    # angle to measure is the one the curve actually leaves with -- the
    # opening tangent, i.e. the chord from the gap to the first control
    # point -- NOT the angle to rest[0]. rest[0] sits further down the route,
    # past everything the run would strip, and on any curving channel it
    # reads far more off-axis than the departure really is. Judging by it
    # makes the lead-in fire on lines that already leave straight and bend
    # them, which is the sharp S right at a bridge head.
    dep = _departure_dir(gx, gy, ux, uy, ctrl)
    min_dev = math.cos(math.radians(max(0.0, min(
        89.0, float(BRIDGES_AIM_LEAD_IN_MIN_DEV_DEG)))))
    if dep is not None and dep >= min_dev:
        return ctrl

    if not _los_clear((px, py), rest[0], nav):
        return ctrl
    return [(px, py)] + rest


def _axis_entry(
    nav: _Nav, sx: float, sy: float, ux: float, uy: float, max_march: float
) -> Optional[Tuple[int, int]]:
    """Navigable entry cell in front of a bridge gap.

    First marches straight along the socket axis (sx, sy)+(ux, uy), so the
    aim line emanates straight out of the bridge instead of hooking sideways.
    If the axis ray misses the water (e.g. the bridge is slightly misaligned
    with the channel) it falls back to the nearest navigable cell inside the
    exit cone: within ENTRY_MAX_ANGLE_DEG of the socket axis, ties broken
    toward the axis.

    The cone is what keeps the line plausible as a ship track. Accepting any
    cell merely *forward* of the gap (projection >= 0) admits angles up to 90
    degrees, which draws the line squirting sideways off the deck -- a turn
    no vessel leaving a bridge can make. It also subsumes the older forward
    restriction, so an anchor can still never land behind the *other*
    bridge's deck and force the U-shaped loop back over it. Returns None when
    no water lies ahead inside the cone; the caller then draws nothing, which
    beats drawing a line at an impossible angle."""
    n = int(max_march)
    for k in range(n + 1):
        ix = int(round(sx + ux * k))
        iy = int(round(sy + uy * k))
        if 0 <= iy < nav.H and 0 <= ix < nav.W and nav.mask[iy, ix]:
            return (iy, ix)

    ci, ri = int(round(sx)), int(round(sy))
    min_cos = math.cos(math.radians(max(0.0, min(
        89.0, float(BRIDGES_AIM_ENTRY_MAX_ANGLE_DEG)))))
    best: Optional[Tuple[int, int]] = None
    best_key = (1e18, 1e18)
    for dr in range(-n, n + 1):
        for dc in range(-n, n + 1):
            along = dc * ux + dr * uy
            if along <= 0.0:
                continue  # never anchor behind the bridge
            d = math.hypot(dc, dr)
            if along < min_cos * d:
                continue  # outside the exit cone
            r, c = ri + dr, ci + dc
            if 0 <= r < nav.H and 0 <= c < nav.W and nav.mask[r, c]:
                # Nearest wins; among equals, the one closest to the axis.
                key = (d, abs(dc * -uy + dr * ux))
                if key < best_key:
                    best_key, best = key, (r, c)
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
) -> Optional[Tuple[List[Tuple[float, float]], _Nav]]:
    """Route the navigable channel linking two snapped bridge gaps.

    Runs A* over the navigable mask inside a padded bounding box around the
    two gaps, so the route keeps the ship radius away from land while
    connecting both crossings -- bending around headlands/islands and never
    crossing another bridge deck (which reads as non-navigable).

    The full MIN_CLEARANCE_PX is tried first, then the RELAX_CLEARANCE_PX
    fallbacks in turn. Channels genuinely pinch below the nominal radius --
    under the bridge itself, in a narrow cut -- and there the strict mask has
    no route at all. Failing over to the unrouted connector there is the
    worse answer by far: it is a straight line drawn across whatever lies
    between the gaps, terrain included, while a route at a tighter clearance
    is still a route that stays on the water.

    Returns (dense route between the two snapped anchors, the _Nav it is
    clear in -- the caller must fit the curve against that same one), or None
    when no clearance yields a route."""
    clearances = [nav.clearance]
    clearances += [float(c) for c in BRIDGES_AIM_RELAX_CLEARANCE_PX
                   if 0.0 < float(c) < nav.clearance]
    for i, clearance in enumerate(clearances):
        attempt = nav if i == 0 else nav.relaxed(clearance)
        got = _trace_pair_path_at(si, sj, attempt)
        if got is not None:
            return got, attempt
    return None


def _trace_pair_path_at(
    si: "_Socket", sj: "_Socket", nav: _Nav
) -> Optional[List[Tuple[float, float]]]:
    """One pair-routing attempt at ``nav``'s clearance (see _trace_pair_path)."""
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


def _direct_connector(
    ax: float, ay: float, si: "_Socket",
    bx: float, by: float, sj: "_Socket", nav: Optional[_Nav],
) -> Optional[str]:
    """Last-resort connector between two gaps no route could link.

    A facing-tangent bezier, drawn only when it is safe to: either the gaps
    are within DIRECT_MAX_SPAN_PX (near-coincident placements of what is
    really one crossing, where the connector hides under the decks), or the
    curve is verified to stay on water. Anything longer that fails the check
    returns None and is not drawn -- an unverified connector across a wide
    gap is exactly the line seen running over terrain, and no line is a
    better answer than a wrong one.

    The check uses raw water coverage rather than the eroded navigable band:
    the connector exists precisely because the band had no route, so judging
    it by that standard would reject every one of them."""
    k = math.hypot(bx - ax, by - ay) * 0.5
    c1x, c1y = ax + si.ux * k, ay + si.uy * k
    c2x, c2y = bx + sj.ux * k, by + sj.uy * k
    d = (f'M {ax:.3f} {ay:.3f} '
         f'C {c1x:.3f} {c1y:.3f} {c2x:.3f} {c2y:.3f} '
         f'{bx:.3f} {by:.3f}')

    span = k * 2.0
    if nav is None or span <= float(BRIDGES_AIM_DIRECT_MAX_SPAN_PX):
        return d

    p0, p1 = (ax, ay), (bx, by)
    nsamp = max(8, int(span / max(BRIDGES_AIM_CURVE_CHECK_STEP_PX, 0.5)))
    for i in range(nsamp + 1):
        x, y = _bezier_at(p0, (c1x, c1y), (c2x, c2y), p1, i / nsamp)
        if not nav.on_water(x, y):
            return None
    return d


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
        routed = _trace_pair_path(si, sj, nav) if nav is not None else None
        if (routed is not None and nav is not None
                and _detour_is_contrived(routed[0], routed[1], nav,
                                         (ax, ay), (bx, by))
                and _direct_connector(ax, ay, si, bx, by, sj, nav) is not None):
            # The straight line is on water and the route is not; prefer it.
            routed = None
        if routed is not None and len(routed[0]) >= 2:
            core, rnav = routed
            ctrl = _curve_control_points(core, rnav)
            # Straight out of both decks before the curve is free to bend --
            # but only where the departure needs correcting, and never more
            # than a fraction of the link (a short bridge-to-bridge hop has
            # no room to spare).
            rlen = _polyline_len(core)
            ctrl = _lead_in(ax, ay, si.ux, si.uy, ctrl, rnav, rlen)
            ctrl = _lead_in(bx, by, sj.ux, sj.uy, ctrl[::-1], rnav, rlen)[::-1]
            full = [(ax, ay)] + ctrl + [(bx, by)]
            # Re-enforce clearance now that the gap anchors have bent the
            # spline; their own spans cross the decks and are exempt.
            full = _refine_curve(full, rnav, skip_first=True, skip_last=True)
            d = _smooth_path_d(full)
        else:
            d = _direct_connector(ax, ay, si, bx, by, sj, nav)
            if d is None:
                continue
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
            ctrl = _lead_in(gx, gy, sock.ux, sock.uy, ctrl, nav,
                            _polyline_len(core))
            full = [(gx, gy)] + ctrl
            full = _refine_curve(full, nav, skip_first=True)
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
    water_mask: Optional[np.ndarray] = None,
) -> bool:
    """Render the procedural ``bridges_aim`` layer into
    BRIDGES_AIM_DIR/<region_name>.png.

    ``water_dist`` is an optional float distance-to-shore field (0 on
    non-water) used to route the aim lines through navigable water. When None
    the lines degrade to straight full-length stubs (no clearance routing).

    ``water_mask`` is the raw water coverage behind that field, before the
    navigability depth gate. It is what decides whether an unroutable direct
    connector still lies on water; without it a shallow channel looks like
    land and its connector is dropped."""
    import cairosvg

    json_path = JSON_DIR / f"{region_name}.json"
    if not json_path.is_file():
        print(f"  [WARN] bridges_aim: no JSON at {json_path}; skipped")
        return False
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    sockets = _collect_bridge_sockets(data)
    nav = (_Nav(water_dist, BRIDGES_AIM_MIN_CLEARANCE_PX, water_mask)
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
        print(f"  [WARN] bridges_aim rasterize failed: {exc}")
        return False

    n_pairs = len(pairs) if sockets else 0
    print(f"  [bridges_aim] {len(sockets)} socket(s), {n_pairs} snapped "
          f"pair(s) -> {out_path.name}")
    return True
