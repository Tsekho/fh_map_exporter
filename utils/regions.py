"""Region geometry, deep-water spawning, and per-region spill builder."""

import json
import os
from typing import Dict, List, Optional, Tuple

import bpy

from utils.blender import (
    make_color_material,
    place_spline_mesh,
    transform_to_blender_xy,
)
from utils.config import (
    CATEGORY_COLORS,
    DEEP_WATER_DEPTH,
    SPLINE_CATEGORIES,
    short_path,
)
from utils.map import Map
from utils.psk import clear_caches, mesh_cache
from utils.tui import error, log, warn


REGION_UNIT_TO_METER = 1890.0 / 1776.0

# Foxhole hex grid: vertical spacing 1776 units, diagonals (±1540, ±888).
REGION_NEIGHBOR_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (0, -1776),
    (0, 1776),
    (-1540, -888),
    (1540, -888),
    (-1540, 888),
    (1540, 888),
)

RESERVED_CATEGORIES: Tuple[str, ...] = ("terrain", "water", "deep_water")


def region_center_to_blender(center: List[float]) -> Tuple[float, float]:
    """Region-center JSON (x, y) -> Blender meters. Y is flipped."""
    return (center[0] * REGION_UNIT_TO_METER, -center[1] * REGION_UNIT_TO_METER)


def find_region_neighbors(
    region_centers: Dict[str, List[float]],
    region_key: str,
    tol: float = 1.0,
) -> List[str]:
    """Return 0-6 neighbor keys matching REGION_NEIGHBOR_OFFSETS."""
    cx, cy = region_centers[region_key]
    expected = [(cx + dx, cy + dy) for dx, dy in REGION_NEIGHBOR_OFFSETS]
    neighbors: List[str] = []
    for name, (nx, ny) in region_centers.items():
        if name == region_key:
            continue
        for ex, ey in expected:
            if abs(nx - ex) < tol and abs(ny - ey) < tol:
                neighbors.append(name)
                break
    return neighbors


def signed_distance_past_midline(
    point_xy: Tuple[float, float],
    own_center_xy: Tuple[float, float],
    neighbor_center_xy: Tuple[float, float],
) -> float:
    """Signed distance from point to the midline between the two centers,
    positive toward neighbor."""
    mx = 0.5 * (own_center_xy[0] + neighbor_center_xy[0])
    my = 0.5 * (own_center_xy[1] + neighbor_center_xy[1])
    dx = neighbor_center_xy[0] - own_center_xy[0]
    dy = neighbor_center_xy[1] - own_center_xy[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0.0:
        return 0.0
    return ((point_xy[0] - mx) * dx + (point_xy[1] - my) * dy) / length


# ------------------------------------------------------------------------------
#  Deep-water clones
# ------------------------------------------------------------------------------


def spawn_deep_water_tree(
    water_root: bpy.types.Collection,
    parent_collection: bpy.types.Collection,
    depth: float = DEEP_WATER_DEPTH,
    color: str = "#000000",
    coll_name: str = "deep_water",
) -> List[bpy.types.Object]:
    """Mirror water_root into parent_collection/<coll_name>, each clone
    placed depth metres below with a black object-level material."""
    if water_root is None:
        return []

    deep_root = bpy.data.collections.new(coll_name)
    parent_collection.children.link(deep_root)
    black_mat = make_color_material(color)

    mapping: Dict[bpy.types.Collection, bpy.types.Collection] = {
        water_root: deep_root
    }

    def _mirror(src, dst) -> None:
        for child in src.children:
            new_child = bpy.data.collections.new(child.name)
            dst.children.link(new_child)
            mapping[child] = new_child
            _mirror(child, new_child)

    _mirror(water_root, deep_root)
    clones: List[bpy.types.Object] = []

    def _clone(src) -> None:
        dst = mapping[src]
        for obj in list(src.objects):
            if obj.data is None:
                continue
            clone = bpy.data.objects.new(f"{obj.name}_deep", obj.data)
            dst.objects.link(clone)
            clone.location = (obj.location.x, obj.location.y,
                              obj.location.z - depth)
            clone.rotation_mode = obj.rotation_mode
            clone.rotation_euler = obj.rotation_euler
            clone.scale = obj.scale
            if not clone.material_slots:
                clone.data.materials.append(None)
            clone.material_slots[0].link = "OBJECT"
            clone.material_slots[0].material = black_mat
            clones.append(clone)
        for child in src.children:
            _clone(child)

    _clone(water_root)
    log(f"  Deep water: {len(clones)} clone(s) placed {depth:.0f} m below")
    return clones


def spawn_deep_water(
    surface_objects: List[bpy.types.Object],
    parent_collection: bpy.types.Collection,
    depth: float = DEEP_WATER_DEPTH,
    color: str = "#000000",
    coll_name: str = "Deep_Water",
) -> int:
    """Clone each surface object depth metres below with a black override."""
    if not surface_objects:
        return 0

    deep_coll = bpy.data.collections.new(coll_name)
    parent_collection.children.link(deep_coll)
    black_mat = make_color_material(color)

    placed = 0
    for surf in surface_objects:
        if surf.data is None:
            continue
        clone = bpy.data.objects.new(f"{surf.name}_deep", surf.data)
        deep_coll.objects.link(clone)
        clone.location = (surf.location.x, surf.location.y,
                          surf.location.z - depth)
        clone.rotation_mode = surf.rotation_mode
        clone.rotation_euler = surf.rotation_euler
        clone.scale = surf.scale
        if not clone.material_slots:
            clone.data.materials.append(None)
        clone.material_slots[0].link = "OBJECT"
        clone.material_slots[0].material = black_mat
        placed += 1

    log(f"  Deep water: {placed} clone(s) placed {depth:.0f} m below")
    return placed


# ------------------------------------------------------------------------------
#  Region spill builder
# ------------------------------------------------------------------------------


def _build_category_lookup(whitelist: Dict[str, List[str]], cats: List[str]) -> set:
    out: set = set()
    for cat in cats:
        out.update(whitelist.get(cat, []))
    return out


def build_region_with_spill(
    region_key: str,
    export_dir: str,
    region_centers: Dict[str, List[float]],
    catalogue: Dict[str, List[str]],
    json_name_map: Dict[str, str],
    spill_meters: float = 200.0,
    terrain_stride: int = 1,
) -> None:
    """Build a region .blend with neighbor spill into blend_spill/.

    Focus region: terrain + every whitelist category. Neighbors contribute
    only non-water spill categories within spill_meters of the shared border.
    """
    from utils.blender import create_terrain  # local to avoid import cycle at top

    spill_categories: List[str] = [
        c for c in catalogue if c not in RESERVED_CATEGORIES
    ]

    missing_color = [
        c for c in (["terrain", "water"] + spill_categories)
        if c not in CATEGORY_COLORS
    ]
    if missing_color:
        warn(f"  No CATEGORY_COLORS entry for: "
             f"{', '.join(missing_color)} - those categories will be skipped")
    spill_categories = [c for c in spill_categories if c in CATEGORY_COLORS]

    allowed_neighbor_meshes = _build_category_lookup(
        catalogue, spill_categories
    )
    water_meshes = set(catalogue.get("water", []))

    palette: Dict[str, str] = {}
    focus_categories = [c for c in ("water",) + tuple(spill_categories)
                        if c in CATEGORY_COLORS]
    for cat in focus_categories:
        color = CATEGORY_COLORS[cat]
        for mesh in catalogue.get(cat, []):
            palette[mesh] = color

    mesh_to_category: Dict[str, str] = {}
    for cat in focus_categories:
        for m in catalogue.get(cat, []):
            mesh_to_category[m] = cat

    own_name = json_name_map[region_key]
    own_json = os.path.join(export_dir, "_json", f"{own_name}.json")
    if not os.path.exists(own_json):
        error(f"JSON not found: {own_json}")
        return

    own_center = region_center_to_blender(region_centers[region_key])
    neighbors = find_region_neighbors(region_centers, region_key)

    log(f"=== {own_name} (spill) ===")
    log(f"  Center (Blender m): ({own_center[0]:.1f}, {own_center[1]:.1f})")
    log(f"  Neighbors ({len(neighbors)}): "
          f"{', '.join(json_name_map.get(n, n) for n in neighbors) or 'none'}")

    focus_include = sorted(allowed_neighbor_meshes | water_meshes)
    neighbor_include = sorted(allowed_neighbor_meshes)
    neighbor_exclude = sorted(water_meshes)

    own_map = Map(own_json, export_dir, include=focus_include, palette=palette)

    clear_caches()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = own_name

    root = bpy.data.collections.new(own_name)
    scene.collection.children.link(root)

    coll_cache: Dict[Tuple[str, ...], bpy.types.Collection] = {}
    total = 0

    terrain_mat = make_color_material(CATEGORY_COLORS["terrain"])
    if os.path.exists(own_map.heightmap_path):
        log("[terrain] Building focus terrain ...")
        t_coll = bpy.data.collections.new("terrain")
        root.children.link(t_coll)
        create_terrain(
            own_map.heightmap_path, t_coll,
            stride=terrain_stride, material=terrain_mat,
        )
    else:
        log(f"[terrain] Missing heightmap: {short_path(own_map.heightmap_path)}")

    focus_category_objs: Dict[str, List[bpy.types.Object]] = {}
    total += own_map._populate_by_category(
        root, coll_cache,
        mesh_to_category=mesh_to_category,
        category_objects=focus_category_objs,
    )

    # Dedup near-identical placements:
    #   - same source (JSON places the same mesh twice): 0.1 m
    #   - cross source (shared-border assets in both regions): 1.5 m
    # Match on (mesh, scale, rotation to 3 dp) + XY distance.
    DEDUP_CATS: Tuple[str, ...] = tuple(spill_categories)
    SELF_DEDUP_XY_TOL = 0.1
    CROSS_DEDUP_XY_TOL = 1.5
    DEDUP_BUCKET = 2.0

    def _dedup_shape_key(o: bpy.types.Object) -> Tuple:
        return (
            o.data.name,
            round(o.scale.x, 3), round(o.scale.y, 3), round(o.scale.z, 3),
            round(o.rotation_euler.x, 3),
            round(o.rotation_euler.y, 3),
            round(o.rotation_euler.z, 3),
        )

    def _bucket_index(o: bpy.types.Object) -> Tuple[int, int]:
        return (int(o.location.x // DEDUP_BUCKET),
                int(o.location.y // DEDUP_BUCKET))

    def _has_hit_within(
        sk, ox, oy, ix, iy,
        buckets: Dict[Tuple, Dict[Tuple[int, int], List[Tuple[float, float]]]],
        tol_sq: float,
    ) -> bool:
        cells = buckets.get(sk)
        if not cells:
            return False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                pts = cells.get((ix + dx, iy + dy))
                if not pts:
                    continue
                for px, py in pts:
                    if (px - ox) ** 2 + (py - oy) ** 2 <= tol_sq:
                        return True
        return False

    def _dedup_within_source(cat_objs):
        tol_sq = SELF_DEDUP_XY_TOL * SELF_DEDUP_XY_TOL
        buckets: Dict = {}
        removed = 0
        for cat in DEDUP_CATS:
            kept: List[bpy.types.Object] = []
            for o in cat_objs.get(cat, []):
                sk = _dedup_shape_key(o)
                ix, iy = _bucket_index(o)
                ox, oy = o.location.x, o.location.y
                if _has_hit_within(sk, ox, oy, ix, iy, buckets, tol_sq):
                    bpy.data.objects.remove(o, do_unlink=True)
                    removed += 1
                    continue
                buckets.setdefault(sk, {}).setdefault(
                    (ix, iy), []
                ).append((ox, oy))
                kept.append(o)
            if cat in cat_objs:
                cat_objs[cat] = kept
        return removed, buckets

    focus_self_removed, focus_buckets = _dedup_within_source(focus_category_objs)
    if focus_self_removed:
        log(f"  dedup (focus self): {focus_self_removed} "
              f"duplicate placement(s) removed")

    _cross_tol_sq = CROSS_DEDUP_XY_TOL * CROSS_DEDUP_XY_TOL

    def _collides_with_focus(o: bpy.types.Object) -> bool:
        sk = _dedup_shape_key(o)
        ix, iy = _bucket_index(o)
        return _has_hit_within(
            sk, o.location.x, o.location.y, ix, iy,
            focus_buckets, _cross_tol_sq,
        )

    for neigh_key in neighbors:
        neigh_name = json_name_map.get(neigh_key)
        if neigh_name is None:
            warn(f"  No JSON for neighbor '{neigh_key}', skipped")
            continue
        neigh_json = os.path.join(export_dir, "_json", f"{neigh_name}.json")
        if not os.path.exists(neigh_json):
            warn(f"  Missing neighbor JSON: {neigh_json}")
            continue

        neigh_center = region_center_to_blender(region_centers[neigh_key])
        shift = (neigh_center[0] - own_center[0],
                 neigh_center[1] - own_center[1], 0.0)
        log(f"[neighbor] {neigh_name}  "
              f"shift=({shift[0]:+.1f}, {shift[1]:+.1f}) m")

        neigh_root = bpy.data.collections.new(neigh_name)
        scene.collection.children.link(neigh_root)

        # Border filter: neighbor-local (lx, ly) passes when
        #   (lx, ly) · v_hat <= spill_meters - v_len/2
        vx = neigh_center[0] - own_center[0]
        vy = neigh_center[1] - own_center[1]
        v_len = (vx * vx + vy * vy) ** 0.5 or 1.0
        vx_hat, vy_hat = vx / v_len, vy / v_len
        max_local_dot = spill_meters - 0.5 * v_len

        def _in_spill(t: list, _xh=vx_hat, _yh=vy_hat, _m=max_local_dot) -> bool:
            lx, ly = transform_to_blender_xy(t)
            return (lx * _xh + ly * _yh) <= _m

        neigh_map = Map(
            neigh_json, export_dir,
            include=neighbor_include,
            exclude=neighbor_exclude,
        )
        neigh_cat_objs: Dict[str, List[bpy.types.Object]] = {}
        placed_n = neigh_map._populate_by_category(
            neigh_root, coll_cache,
            mesh_to_category=mesh_to_category,
            root_key_prefix=(neigh_name,),
            shift=shift,
            border_filter=_in_spill,
            announce=False,
            category_objects=neigh_cat_objs,
        )

        self_removed, _ = _dedup_within_source(neigh_cat_objs)
        cross_removed = 0
        for _cat in DEDUP_CATS:
            kept: List[bpy.types.Object] = []
            for _o in neigh_cat_objs.get(_cat, []):
                if _collides_with_focus(_o):
                    bpy.data.objects.remove(_o, do_unlink=True)
                    cross_removed += 1
                else:
                    kept.append(_o)
            if _cat in neigh_cat_objs:
                neigh_cat_objs[_cat] = kept

        removed = self_removed + cross_removed
        placed_n -= removed
        if removed:
            log(f"  placed {placed_n:,} spill object(s) from {neigh_name} "
                  f"({self_removed} self-dup, {cross_removed} focus-dup removed)")
        else:
            log(f"  placed {placed_n:,} spill object(s) from {neigh_name}")
        total += placed_n

    own_map._apply_palette()

    # Splines (focus region only)
    spline_color = CATEGORY_COLORS.get("splines")
    if spline_color and SPLINE_CATEGORIES:
        mesh_to_spline_cat: Dict[str, str] = {}
        for cat, names in SPLINE_CATEGORIES.items():
            for n in names:
                mesh_to_spline_cat[n] = cat

        with open(own_json, "r", encoding="utf-8") as _f:
            _raw = json.load(_f)
        raw_splines: Dict[str, list] = _raw.get("splines", {}) or {}

        spline_mat = make_color_material(spline_color)
        splines_root: Optional[bpy.types.Collection] = None
        cat_colls: Dict[str, bpy.types.Collection] = {}
        placed_splines = 0

        for mesh_name, entries in raw_splines.items():
            cat = mesh_to_spline_cat.get(mesh_name)
            if cat is None:
                continue
            if splines_root is None:
                splines_root = bpy.data.collections.new("splines")
                root.children.link(splines_root)
            if cat not in cat_colls:
                c = bpy.data.collections.new(cat)
                splines_root.children.link(c)
                cat_colls[cat] = c
            coll = cat_colls[cat]
            leaf_name = mesh_name.split("__")[-1]
            for i, entry in enumerate(entries, 1):
                obj = place_spline_mesh(
                    mesh_name, entry, coll, own_map.meshes_dir,
                    f"{leaf_name}.{i}",
                )
                if obj is None:
                    continue
                if obj.data.materials:
                    obj.data.materials[0] = spline_mat
                else:
                    obj.data.materials.append(spline_mat)
                placed_splines += 1

        if placed_splines:
            log(f"[splines] placed {placed_splines:,} spline object(s)")
            total += placed_splines

    water_root: Optional[bpy.types.Collection] = None
    for ch in root.children:
        if ch.name == "water":
            water_root = ch
            break
    if water_root:
        spawn_deep_water_tree(water_root, root)

    loaded_ok = sum(1 for v in mesh_cache.values() if v is not None)
    loaded_err = sum(1 for v in mesh_cache.values() if v is None)
    log(f"\n  Objects placed : {total:,}\n"
          f"  Unique meshes  : {loaded_ok} loaded, "
          f"{loaded_err} missing/errored")

    out_dir = os.path.join(export_dir, "blend_spill")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(out_dir, f"{own_name}.blend"))
    if os.path.exists(out_path):
        os.remove(out_path)
    log(f"\nSaving -> {short_path(out_path)} ...")
    bpy.ops.wm.save_as_mainfile(filepath=out_path, compress=True)
    log("Done.\n")
