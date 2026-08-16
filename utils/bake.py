"""Top-down bake renderers (AO / heightmap / ID / coverage) + BVH raycasting."""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

import bpy
import cv2
import numpy as np
from mathutils import Vector

from utils.config import (
    AO_NEAR_WHITE_CUTOFF,
    AO_SLOPE_C_STRONG,
    AO_SLOPE_C_WEAK,
    AO_SLOPE_C_SL,
    AO_SLOPE_POWER_STRONG,
    AO_SLOPE_POWER_WEAK,
    AO_SLOPE_POWER_SL,
    MIN_SPLIT_LAYER_VALUE,
    SPLIT_LAYER_EDGE_SHADER_POWER,
    SPLIT_LAYER_EDGE_SHADER_RADIUS_PX,
    SPLIT_LAYER_EDGE_SHADER_STRENGTH,
    short_path,
)
from utils.png import write_png16_gray, write_png8_gray, write_png8_rgb, write_png8_rgba
from utils.tui import log, warn


BAKE_IMG_SIZE = 2048
BAKE_PIXEL_SIZE_M = 1890.0 / 1776.0
BAKE_CAM_Z = 5000.0


# ------------------------------------------------------------------------------
#  BVH / raycast plumbing
# ------------------------------------------------------------------------------


def load_hex_mask(path: str, size: int = BAKE_IMG_SIZE) -> np.ndarray:
    """Load path as a boolean mask (True = inside hex), top-down."""
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    px = px[::-1]  # Blender pixels are bottom-up
    mask = px[..., 0] > 0.5
    bpy.data.images.remove(img)
    if mask.shape != (size, size):
        ys = (np.arange(size) * mask.shape[0] // size)
        xs = (np.arange(size) * mask.shape[1] // size)
        mask = mask[ys][:, xs]
    return mask


def _bake_pixel_world_xy(size: int = BAKE_IMG_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    c = size / 2.0
    j = np.arange(size, dtype=np.float32)
    X = (j + 0.5 - c) * BAKE_PIXEL_SIZE_M
    Y = (c - j - 0.5) * BAKE_PIXEL_SIZE_M
    return X, Y


def _set_hide(objs: List[bpy.types.Object], hide: bool, key: str) -> Dict:
    prev: Dict[str, bool] = {}
    for o in objs:
        prev[o.name] = getattr(o, key)
        setattr(o, key, hide)
    return prev


def _restore_hide(prev: Dict[str, bool], key: str) -> None:
    for name, v in prev.items():
        o = bpy.data.objects.get(name)
        if o is not None:
            setattr(o, key, v)


def _rasterize_targets_footlog(
    target_objs: List[bpy.types.Object],
    size: int = BAKE_IMG_SIZE,
    margin_px: int = 4,
) -> np.ndarray:
    """Rasterize the top-down XY footprint of target_objs' evaluated
    triangles into a boolean image of `size` x `size`, dilated by
    margin_px. Returns all-False if no usable geometry."""
    raster = np.zeros((size, size), dtype=np.uint8)
    if not target_objs:
        return raster.astype(bool)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    c = size / 2.0
    px = BAKE_PIXEL_SIZE_M

    all_polys: List[np.ndarray] = []

    for obj in target_objs:
        if obj is None or obj.type != "MESH":
            continue
        ev = obj.evaluated_get(depsgraph)
        try:
            me = ev.to_mesh()
        except RuntimeError:
            continue
        if me is None:
            continue
        try:
            nv = len(me.vertices)
            if nv == 0:
                continue
            co = np.empty(nv * 3, dtype=np.float32)
            me.vertices.foreach_get("co", co)
            co = co.reshape(nv, 3)
            # World-transform via matrix_world (4x4). mathutils stores
            # row-major; np.array gives us the same orientation as M @ v.
            M = np.array(obj.matrix_world, dtype=np.float32)
            hv = np.empty((nv, 4), dtype=np.float32)
            hv[:, :3] = co
            hv[:, 3] = 1.0
            w = hv @ M.T
            wx = w[:, 0]
            wy = w[:, 1]
            # Pixel indices: j = x/px + c - 0.5, i = c - 0.5 - y/px.
            j_px = wx / px + c - 0.5
            i_px = c - 0.5 - wy / px

            me.calc_loop_triangles()
            nt = len(me.loop_triangles)
            if nt == 0:
                continue
            tri = np.empty(nt * 3, dtype=np.int32)
            me.loop_triangles.foreach_get("vertices", tri)
            tri = tri.reshape(nt, 3)

            # (nt, 3, 2) polygons as (x=j, y=i) int32 for cv2.fillPoly.
            poly = np.empty((nt, 3, 2), dtype=np.int32)
            poly[..., 0] = np.rint(j_px[tri])
            poly[..., 1] = np.rint(i_px[tri])
            all_polys.append(poly)
        finally:
            ev.to_mesh_clear()

    if all_polys:
        # Single batched fillPoly call is much faster than one per tri.
        polys = np.concatenate(all_polys, axis=0)
        # Cheap culling: drop triangles whose whole AABB is offscreen.
        xmn = polys[..., 0].min(axis=1)
        xmx = polys[..., 0].max(axis=1)
        ymn = polys[..., 1].min(axis=1)
        ymx = polys[..., 1].max(axis=1)
        keep = (xmx >= 0) & (xmn < size) & (ymx >= 0) & (ymn < size)
        polys = polys[keep]
        if polys.size:
            cv2.fillPoly(raster, polys, 255)

    if margin_px > 0:
        k = 2 * int(margin_px) + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        raster = cv2.dilate(raster, kernel)

    return raster > 0


def _mask_restricted_to_targets(
    mask: np.ndarray,
    target_objs: List[bpy.types.Object],
    margin_px: int = 4,
) -> np.ndarray:
    """Return `mask` AND'd with a tight rasterized footprint of
    target_objs (triangles projected to XY, dilated by margin_px)."""
    footprint = _rasterize_targets_footlog(
        target_objs, size=mask.shape[0], margin_px=margin_px,
    )
    out = mask & footprint
    if mask.any():
        kept = int(out.sum())
        total = int(mask.sum())
        if total:
            log(f"  [footprint] restricted bake area to "
                  f"{kept}/{total} px ({100.0 * kept / total:.1f}%)")
    return out


def _build_bvh_from_objs(
    objs: List[bpy.types.Object],
) -> Tuple[Optional[object], List[int]]:
    """Build a BVHTree from world-space triangles of objs.
    Returns (bvh, tri_to_obj_idx)."""
    from mathutils.bvhtree import BVHTree
    if not objs:
        return None, []

    depsgraph = bpy.context.evaluated_depsgraph_get()
    verts: List[Tuple[float, float, float]] = []
    tris: List[Tuple[int, int, int]] = []
    tri_to_obj_idx: List[int] = []

    for oi, obj in enumerate(objs):
        if obj is None or obj.type != 'MESH':
            continue
        ev = obj.evaluated_get(depsgraph)
        try:
            me = ev.to_mesh()
        except RuntimeError:
            continue
        if me is None:
            continue
        try:
            mw = obj.matrix_world.copy()
            base = len(verts)
            for v in me.vertices:
                wv = mw @ v.co
                verts.append((wv.x, wv.y, wv.z))
            me.calc_loop_triangles()
            for lt in me.loop_triangles:
                vs = lt.vertices
                tris.append((base + vs[0], base + vs[1], base + vs[2]))
                tri_to_obj_idx.append(oi)
        finally:
            ev.to_mesh_clear()

    if not tris:
        return None, []
    return BVHTree.FromPolygons(verts, tris), tri_to_obj_idx


def _bake_rows_parallel(mask: np.ndarray, row_fn: Callable[[int], None]) -> None:
    """Run row_fn(i) on every row with any True pixel via a thread pool.
    BVHTree.ray_cast releases the GIL, giving near-linear speedup."""
    rows = [i for i in range(mask.shape[0]) if mask[i].any()]
    workers = max(1, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(row_fn, rows))


# ------------------------------------------------------------------------------
#  Raycast bakes
# ------------------------------------------------------------------------------


def raycast_heightmap(
    output_path: str,
    mask: np.ndarray,
    visible_objs: List[bpy.types.Object],
    occluders: Optional[List[bpy.types.Object]] = None,
) -> None:
    """Top-down heightmap bake: value = z_m * 100 + 32768.
    Occluder hits produce void (0)."""
    occluders = occluders or []
    seen = set()
    all_objs: List[bpy.types.Object] = []
    for o in list(visible_objs) + list(occluders):
        if o is None or o.name in seen:
            continue
        seen.add(o.name)
        all_objs.append(o)

    bvh, tri_to_obj_idx = _build_bvh_from_objs(all_objs)
    out = np.zeros((BAKE_IMG_SIZE, BAKE_IMG_SIZE), dtype=np.uint16)

    if bvh is not None and tri_to_obj_idx:
        occluder_names = {o.name for o in occluders}
        is_occ = np.asarray(
            [all_objs[oi].name in occluder_names for oi in range(len(all_objs))],
            dtype=bool,
        )
        tri_to_obj_np = np.asarray(tri_to_obj_idx, dtype=np.int32)
        X, Y = _bake_pixel_world_xy()
        direction = Vector((0.0, 0.0, -1.0))

        def _row(i: int) -> None:
            yv = float(Y[i])
            mrow = mask[i]
            row = out[i]
            origin = Vector((0.0, yv, BAKE_CAM_Z))
            cols = np.where(mrow)[0]
            for j in cols:
                origin.x = float(X[int(j)])
                loc, _n, tri_idx, _d = bvh.ray_cast(origin, direction)
                if loc is None or tri_idx is None:
                    continue
                oi = int(tri_to_obj_np[tri_idx])
                if is_occ[oi]:
                    continue
                v = int(round(loc.z * 100.0 + 32768.0))
                row[int(j)] = 0 if v < 0 else (65535 if v > 65535 else v)

        _bake_rows_parallel(mask, _row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_png16_gray(output_path, out)
    log(f"  Heightmap saved -> {short_path(output_path)}")


def raycast_id_map(
    output_path: str,
    mask: np.ndarray,
    category_objects: Dict[str, List[bpy.types.Object]],
    category_colors: Dict[str, str],
    occluders: Optional[List[bpy.types.Object]] = None,
) -> None:
    """Top-down ID bake; occluder hits stay black."""
    flat_objs: List[bpy.types.Object] = []
    obj_color: List[Tuple[int, int, int]] = []
    for cat, objs in category_objects.items():
        hc = category_colors.get(cat, "#000000").lstrip("#")
        col = (int(hc[0:2], 16), int(hc[2:4], 16), int(hc[4:6], 16))
        for o in objs:
            flat_objs.append(o)
            obj_color.append(col)

    occluders = occluders or []
    seen = {o.name for o in flat_objs}
    for o in occluders:
        if o is None or o.name in seen:
            continue
        seen.add(o.name)
        flat_objs.append(o)
        obj_color.append((0, 0, 0))

    bvh, tri_to_obj_idx = _build_bvh_from_objs(flat_objs)
    out = np.zeros((BAKE_IMG_SIZE, BAKE_IMG_SIZE, 3), dtype=np.uint8)

    if bvh is not None and tri_to_obj_idx:
        tri_to_obj_np = np.asarray(tri_to_obj_idx, dtype=np.int32)
        color_np = np.asarray(obj_color, dtype=np.uint8)
        X, Y = _bake_pixel_world_xy()
        direction = Vector((0.0, 0.0, -1.0))

        def _row(i: int) -> None:
            yv = float(Y[i])
            mrow = mask[i]
            row = out[i]
            origin = Vector((0.0, yv, BAKE_CAM_Z))
            cols = np.where(mrow)[0]
            for j in cols:
                origin.x = float(X[int(j)])
                loc, _n, tri_idx, _d = bvh.ray_cast(origin, direction)
                if loc is not None and tri_idx is not None:
                    oi = tri_to_obj_np[tri_idx]
                    row[int(j)] = color_np[oi]

        _bake_rows_parallel(mask, _row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_png8_rgb(output_path, out)
    log(f"  ID map saved -> {short_path(output_path)}")


def raycast_binary_mask(
    output_path: str,
    mask: np.ndarray,
    occluders: List[bpy.types.Object],
    target_objs: List[bpy.types.Object],
) -> None:
    """Top-down binary bake: white where the first hit is in target_objs."""
    seen = set()
    all_objs: List[bpy.types.Object] = []
    for o in list(target_objs) + list(occluders):
        if o is None or o.name in seen:
            continue
        seen.add(o.name)
        all_objs.append(o)

    bvh, tri_to_obj_idx = _build_bvh_from_objs(all_objs)
    out = np.zeros((BAKE_IMG_SIZE, BAKE_IMG_SIZE, 3), dtype=np.uint8)

    if bvh is not None and tri_to_obj_idx:
        target_names = {o.name for o in target_objs}
        is_target = np.asarray(
            [all_objs[oi].name in target_names for oi in range(len(all_objs))],
            dtype=bool,
        )
        tri_to_obj_np = np.asarray(tri_to_obj_idx, dtype=np.int32)
        X, Y = _bake_pixel_world_xy()
        direction = Vector((0.0, 0.0, -1.0))

        def _row(i: int) -> None:
            yv = float(Y[i])
            mrow = mask[i]
            row = out[i]
            origin = Vector((0.0, yv, BAKE_CAM_Z))
            cols = np.where(mrow)[0]
            for j in cols:
                origin.x = float(X[int(j)])
                loc, _n, tri_idx, _d = bvh.ray_cast(origin, direction)
                if loc is not None and tri_idx is not None:
                    oi = int(tri_to_obj_np[tri_idx])
                    if is_target[oi]:
                        row[int(j)] = (255, 255, 255)

        _bake_rows_parallel(mask, _row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_png8_rgb(output_path, out)
    log(f"  Binary mask saved -> {short_path(output_path)}")


def raycast_id_ssaa_per_category(
    output_paths: Dict[str, str],
    mask: np.ndarray,
    category_objects: Dict[str, List[bpy.types.Object]],
    occluders: Optional[List[bpy.types.Object]] = None,
    samples_per_side: int = 2,
) -> None:
    """Per-category SSAA top-down coverage bake.

    Fires samples_per_side^2 rays per pixel. For each category in
    output_paths, writes a grayscale PNG with pixel value
    round(hits / total_samples * 255). Occluders block rays but claim
    no category.
    """
    S = max(1, int(samples_per_side))
    SS = S * S

    categories = list(output_paths.keys())
    cat_index: Dict[str, int] = {c: i for i, c in enumerate(categories)}
    n_cats = len(categories)

    flat_objs: List[bpy.types.Object] = []
    obj_cat_idx: List[int] = []
    for cat in categories:
        for o in category_objects.get(cat, []):
            flat_objs.append(o)
            obj_cat_idx.append(cat_index[cat])

    occluders = occluders or []
    seen = {o.name for o in flat_objs}
    for o in occluders:
        if o is None or o.name in seen:
            continue
        seen.add(o.name)
        flat_objs.append(o)
        obj_cat_idx.append(-1)  # blocker

    bvh, tri_to_obj_idx = _build_bvh_from_objs(flat_objs)
    acc = np.zeros((n_cats, BAKE_IMG_SIZE, BAKE_IMG_SIZE), dtype=np.uint16)

    if bvh is not None and tri_to_obj_idx:
        tri_to_obj_np = np.asarray(tri_to_obj_idx, dtype=np.int32)
        cat_idx_np = np.asarray(obj_cat_idx, dtype=np.int32)
        px = BAKE_PIXEL_SIZE_M
        sub = (np.arange(S, dtype=np.float32) + 0.5) / S - 0.5
        sub_m = sub * px
        X, Y = _bake_pixel_world_xy()
        direction = Vector((0.0, 0.0, -1.0))

        def _row(i: int) -> None:
            yv_base = float(Y[i])
            mrow = mask[i]
            origin = Vector((0.0, 0.0, BAKE_CAM_Z))
            cols = np.where(mrow)[0]
            counts = [0] * n_cats
            for j in cols:
                xv_base = float(X[int(j)])
                for ci in range(n_cats):
                    counts[ci] = 0
                for si in range(S):
                    origin.y = yv_base + float(sub_m[si])
                    for sj in range(S):
                        origin.x = xv_base + float(sub_m[sj])
                        loc, _n, tri_idx, _d = bvh.ray_cast(origin, direction)
                        if loc is None or tri_idx is None:
                            continue
                        oi = int(tri_to_obj_np[tri_idx])
                        ci = int(cat_idx_np[oi])
                        if ci >= 0:
                            counts[ci] += 1
                jj = int(j)
                for ci in range(n_cats):
                    c = counts[ci]
                    if c:
                        acc[ci, i, jj] = c

        _bake_rows_parallel(mask, _row)

    for cat, path in output_paths.items():
        ci = cat_index[cat]
        img = ((acc[ci].astype(np.uint32) * 255 + SS // 2) // SS)
        img = np.clip(img, 0, 255).astype(np.uint8)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_png8_gray(path, img)
        log(f"  ID coverage [{cat}] saved -> {short_path(path)}  (SSAA {S}x{S})")


def raycast_split_layer_rgba(
    output_path: str,
    mask: np.ndarray,
    target_objs: List[bpy.types.Object],
    color_hex: str,
    occluders: Optional[List[bpy.types.Object]] = None,
    samples_per_side: int = 4,
    slope_c: float = 0.6,
    slope_power: float = 5.0,
) -> None:
    """Top-down SSAA RGBA bake for a single split-layer category.

    rgb = native color darkened by slope shade (1 + C*(max(0,Nz)^P - 1)),
    averaged across target-hit subsamples.
    alpha = target_hits / total_samples * 255.
    Pixels that only hit occluders stay transparent.
    """
    S = max(1, int(samples_per_side))
    SS = S * S

    hc = color_hex.lstrip("#")
    base_col = np.array(
        (int(hc[0:2], 16), int(hc[2:4], 16), int(hc[4:6], 16)),
        dtype=np.float32,
    )

    flat_objs: List[bpy.types.Object] = []
    is_target: List[bool] = []
    seen: set = set()
    for o in target_objs or []:
        if o is None or o.name in seen:
            continue
        seen.add(o.name)
        flat_objs.append(o)
        is_target.append(True)
    for o in occluders or []:
        if o is None or o.name in seen:
            continue
        seen.add(o.name)
        flat_objs.append(o)
        is_target.append(False)

    bvh, tri_to_obj_idx = _build_bvh_from_objs(flat_objs)
    out = np.zeros((BAKE_IMG_SIZE, BAKE_IMG_SIZE, 4), dtype=np.uint8)

    if bvh is not None and tri_to_obj_idx:
        tri_to_obj_np = np.asarray(tri_to_obj_idx, dtype=np.int32)
        is_target_np = np.asarray(is_target, dtype=bool)
        px = BAKE_PIXEL_SIZE_M
        sub = (np.arange(S, dtype=np.float32) + 0.5) / S - 0.5
        sub_m = sub * px
        X, Y = _bake_pixel_world_xy()
        direction = Vector((0.0, 0.0, -1.0))
        C = float(slope_c)
        P = float(slope_power)

        def _row(i: int) -> None:
            yv_base = float(Y[i])
            mrow = mask[i]
            row_out = out[i]
            origin = Vector((0.0, 0.0, BAKE_CAM_Z))
            cols = np.where(mrow)[0]
            for j in cols:
                xv_base = float(X[int(j)])
                r_sum = g_sum = b_sum = 0.0
                t_hits = 0
                for si in range(S):
                    origin.y = yv_base + float(sub_m[si])
                    for sj in range(S):
                        origin.x = xv_base + float(sub_m[sj])
                        loc, normal, tri_idx, _d = bvh.ray_cast(
                            origin, direction,
                        )
                        if loc is None or tri_idx is None:
                            continue
                        oi = int(tri_to_obj_np[tri_idx])
                        if not is_target_np[oi]:
                            continue
                        nz = float(normal.z) if normal is not None else 1.0
                        if nz < 0.0:
                            nz = -nz  # flipped winding; top-down face
                        if nz > 1.0:
                            nz = 1.0
                        shade = 1.0 + C * (pow(nz, P) - 1.0)
                        if shade < 0.0:
                            shade = 0.0
                        r_sum += base_col[0] * shade
                        g_sum += base_col[1] * shade
                        b_sum += base_col[2] * shade
                        t_hits += 1
                if t_hits == 0:
                    continue
                a = int(round(t_hits * 255.0 / SS))
                if a > 255:
                    a = 255
                jj = int(j)
                inv = 1.0 / t_hits
                row_out[jj, 0] = int(min(255.0, max(0.0, r_sum * inv)))
                row_out[jj, 1] = int(min(255.0, max(0.0, g_sum * inv)))
                row_out[jj, 2] = int(min(255.0, max(0.0, b_sum * inv)))
                row_out[jj, 3] = a

        _bake_rows_parallel(mask, _row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_png8_rgba(output_path, out)
    log(f"  Split layer RGBA saved -> {short_path(output_path)}  (SSAA {S}x{S})")


def raycast_coverage_rgba(
    output_path: str,
    mask: np.ndarray,
    category_objects: Dict[str, List[bpy.types.Object]],
    category_colors: Dict[str, str],
    occluders: Optional[List[bpy.types.Object]] = None,
    samples_per_side: int = 4,
) -> None:
    """Top-down supersampled coverage bake to RGBA.

    alpha = target_samples / total_samples; rgb = alpha-weighted mean
    of target-sample colors. Pixels outside mask stay transparent.
    """
    S = max(1, int(samples_per_side))
    SS = S * S

    flat_objs: List[bpy.types.Object] = []
    obj_color: List[Tuple[int, int, int]] = []
    is_target: List[bool] = []
    for cat, objs_in in category_objects.items():
        hc = category_colors.get(cat, "#FFFFFF").lstrip("#")
        col = (int(hc[0:2], 16), int(hc[2:4], 16), int(hc[4:6], 16))
        for o in objs_in:
            flat_objs.append(o)
            obj_color.append(col)
            is_target.append(True)

    occluders = occluders or []
    seen = {o.name for o in flat_objs}
    for o in occluders:
        if o is None or o.name in seen:
            continue
        seen.add(o.name)
        flat_objs.append(o)
        obj_color.append((0, 0, 0))
        is_target.append(False)

    # Restrict work to pixels within the targets' XY bounding box:
    # the coverage bake only writes where a target is hit, so pixels
    # outside the targets' AABB would all be raycast for nothing.
    target_only_objs = [
        o for lst in category_objects.values() for o in lst if o is not None
    ]
    mask = _mask_restricted_to_targets(mask, target_only_objs, margin_px=4)

    bvh, tri_to_obj_idx = _build_bvh_from_objs(flat_objs)
    out = np.zeros((BAKE_IMG_SIZE, BAKE_IMG_SIZE, 4), dtype=np.uint8)

    if bvh is not None and tri_to_obj_idx and mask.any():
        tri_to_obj_np = np.asarray(tri_to_obj_idx, dtype=np.int32)
        color_np = np.asarray(obj_color, dtype=np.uint16)
        is_target_np = np.asarray(is_target, dtype=bool)
        px = BAKE_PIXEL_SIZE_M
        sub = (np.arange(S, dtype=np.float32) + 0.5) / S - 0.5
        sub_m = sub * px
        X, Y = _bake_pixel_world_xy()
        direction = Vector((0.0, 0.0, -1.0))

        def _row(i: int) -> None:
            yv_base = float(Y[i])
            mrow = mask[i]
            row_out = out[i]
            origin = Vector((0.0, 0.0, BAKE_CAM_Z))
            cols = np.where(mrow)[0]
            for j in cols:
                xv_base = float(X[int(j)])
                r_sum = g_sum = b_sum = 0
                t_hits = 0
                for si in range(S):
                    origin.y = yv_base + float(sub_m[si])
                    for sj in range(S):
                        origin.x = xv_base + float(sub_m[sj])
                        loc, _n, tri_idx, _d = bvh.ray_cast(origin, direction)
                        if loc is None or tri_idx is None:
                            continue
                        oi = int(tri_to_obj_np[tri_idx])
                        if not is_target_np[oi]:
                            continue
                        c = color_np[oi]
                        r_sum += int(c[0])
                        g_sum += int(c[1])
                        b_sum += int(c[2])
                        t_hits += 1
                if t_hits == 0:
                    continue
                a = int(round(t_hits * 255.0 / SS))
                if a > 255:
                    a = 255
                row_out[int(j), 0] = r_sum // t_hits
                row_out[int(j), 1] = g_sum // t_hits
                row_out[int(j), 2] = b_sum // t_hits
                row_out[int(j), 3] = a

        _bake_rows_parallel(mask, _row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_png8_rgba(output_path, out)
    log(f"  Coverage RGBA saved -> {short_path(output_path)}  (SSAA {S}x{S})")


def bake_spline_layer(
    output_path: str,
    mask: np.ndarray,
    targets: Dict[str, List[bpy.types.Object]],
    palette: Dict[str, str],
    occluders: List[bpy.types.Object],
    terrain_occluders: Optional[List[bpy.types.Object]] = None,
    terrain_drop: float = 0.0,
    samples_per_side: int = 1,
    label: str = "spline layer",
) -> bool:
    """Render a top-down coloured coverage layer for a set of spline
    categories via :func:`raycast_coverage_rgba`.

    ``targets`` is ``{category: [spline_objs]}``; ``palette`` is
    ``{category: "#RRGGBB"}``; ``occluders`` is the caller-built list
    of non-terrain occluders (spill, water, deep_water, non-target
    non-terrain splines, etc.). When ``terrain_occluders`` is
    provided, each terrain object is temporarily lowered by
    ``terrain_drop`` metres before the bake and restored after, so the
    drop lets surface splines read through while burying any
    far-underground artefact splines below the lowered terrain. When
    ``terrain_occluders`` is None/empty, terrain is not included at
    all (used e.g. for beaches, which should never be occluded by
    terrain).

    Writes a blank-but-empty-friendly bake if ``targets`` is empty
    (``raycast_coverage_rgba`` handles the empty-BVH case). Returns
    True on success, False if the bake raised.
    """
    if not targets:
        log(f"  [{label}] no source splines; writing empty image")

    full_occluders: List[bpy.types.Object] = list(occluders)

    prev_z: List[Tuple[bpy.types.Object, float]] = []
    if terrain_occluders:
        for o in terrain_occluders:
            if o is None:
                continue
            prev_z.append((o, o.location.z))
            o.location.z -= terrain_drop
        full_occluders.extend(terrain_occluders)

    try:
        raycast_coverage_rgba(
            output_path, mask, targets, palette,
            occluders=full_occluders,
            samples_per_side=samples_per_side,
        )
        return True
    except Exception as exc:
        warn(f"  [WARN] {label} bake failed: {exc}")
        return False
    finally:
        for o, z in prev_z:
            try:
                o.location.z = z
            except ReferenceError:
                pass


# ------------------------------------------------------------------------------
#  AO Cycles bake
# ------------------------------------------------------------------------------


def _clip_near_white(path: str, cutoff: int) -> None:
    """Snap PNG pixels with any RGB channel >= cutoff to 255."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return
    if img.ndim == 2:
        img[img >= cutoff] = 255
    else:
        rgb = img[..., :3]
        hit = (rgb >= cutoff).any(axis=2)
        rgb[hit] = 255
    cv2.imwrite(path, img)


def _apply_mask_to_image_file(path: str, mask: np.ndarray,
                              preserve_alpha: bool = False) -> None:
    """Zero RGB in path wherever mask is False.
    When preserve_alpha is False the alpha is forced to 1.0 (legacy AO
    behavior). When True the existing alpha is masked too (outside hex
    becomes fully transparent) - used for transparent-film renders."""
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    m = mask
    if m.shape != (h, w):
        ys = (np.arange(h) * m.shape[0] // h)
        xs = (np.arange(w) * m.shape[1] // w)
        m = m[ys][:, xs]
    m_bu = m[::-1]  # Blender pixels are bottom-up; mask is top-down
    inv = ~m_bu
    px[inv, 0] = 0.0
    px[inv, 1] = 0.0
    px[inv, 2] = 0.0
    if preserve_alpha:
        px[inv, 3] = 0.0
    else:
        px[..., 3] = 1.0
    img.pixels = px.ravel().tolist()
    img.filepath_raw = path
    img.file_format = 'PNG'
    img.save()
    bpy.data.images.remove(img)


_GPU_CONFIGURED = False


def _enable_cycles_gpu(scene) -> None:
    """Configure Cycles GPU (OPTIX/CUDA/HIP/ONEAPI/METAL); CPU fallback."""
    global _GPU_CONFIGURED
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except (KeyError, AttributeError):
        return

    if not _GPU_CONFIGURED:
        chosen = None
        for backend in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
            try:
                prefs.compute_device_type = backend
            except TypeError:
                continue
            try:
                devs = prefs.get_devices_for_type(backend)
            except Exception:
                devs = []
            if devs:
                chosen = backend
                for d in devs:
                    d.use = True
                try:
                    for d in prefs.get_devices_for_type("CPU"):
                        d.use = True
                except Exception:
                    pass
                break
        if chosen is None:
            log("  [info] no Cycles GPU backend available; using CPU")
        else:
            names = [d.name for d in prefs.get_devices_for_type(chosen) if d.use]
            log(f"  [info] Cycles GPU backend: {chosen} ({', '.join(names)})")
        _GPU_CONFIGURED = True

    try:
        scene.cycles.device = 'GPU'
    except (AttributeError, TypeError):
        pass


def render_ao(
    output_path: str,
    mask: np.ndarray,
    hidden_objs: List[bpy.types.Object],
    samples: int = 32,
    ao_distance: float = 10.0,
    strong_slope_objs: Optional[List[bpy.types.Object]] = None,
) -> None:
    """Top-down AO bake (Cycles, ortho, 2048x2048) with a white AO+slope override.

    Slope shading tier is picked per-object via pass_index: 0 = weak,
    1 = strong (strong_slope_objs).
    """
    output_path = os.path.abspath(output_path)

    scene = bpy.context.scene
    prev_hidden = _set_hide(hidden_objs, True, "hide_render")

    strong_slope_objs = strong_slope_objs or []
    prev_pass_index: List[Tuple[bpy.types.Object, int]] = []
    for o in strong_slope_objs:
        if o is None:
            continue
        prev_pass_index.append((o, o.pass_index))
        o.pass_index = 1

    cam_data = bpy.data.cameras.new("AO_Cam")
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = BAKE_IMG_SIZE * BAKE_PIXEL_SIZE_M
    cam_data.clip_start = 0.1
    cam_data.clip_end = BAKE_CAM_Z * 2.0
    cam_obj = bpy.data.objects.new("AO_Cam", cam_data)
    cam_obj.location = (0.0, 0.0, BAKE_CAM_Z)
    cam_obj.rotation_euler = (0.0, 0.0, 0.0)
    scene.collection.objects.link(cam_obj)

    ao_mat = bpy.data.materials.new("AO_White")
    ao_mat.use_nodes = True
    nt = ao_mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out_node = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    ao_node = nt.nodes.new("ShaderNodeAmbientOcclusion")
    try:
        ao_node.samples = 16
    except AttributeError:
        pass
    ao_node.inputs["Distance"].default_value = ao_distance
    ao_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)

    geom = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geom.outputs["Normal"], sep.inputs["Vector"])

    # Clamp cos(tilt) to [0, 1]; POWER is undefined on negatives.
    z_pos = nt.nodes.new("ShaderNodeMath")
    z_pos.operation = 'MAXIMUM'
    z_pos.inputs[1].default_value = 0.0
    nt.links.new(sep.outputs["Z"], z_pos.inputs[0])

    def _slope_chain(c: float, power: float):
        pz = nt.nodes.new("ShaderNodeMath")
        pz.operation = 'POWER'
        pz.inputs[1].default_value = power
        nt.links.new(z_pos.outputs["Value"], pz.inputs[0])
        sub = nt.nodes.new("ShaderNodeMath")
        sub.operation = 'SUBTRACT'
        sub.inputs[1].default_value = 1.0
        nt.links.new(pz.outputs["Value"], sub.inputs[0])
        scl = nt.nodes.new("ShaderNodeMath")
        scl.operation = 'MULTIPLY'
        scl.inputs[1].default_value = c
        nt.links.new(sub.outputs["Value"], scl.inputs[0])
        add1 = nt.nodes.new("ShaderNodeMath")
        add1.operation = 'ADD'
        add1.inputs[1].default_value = 1.0
        nt.links.new(scl.outputs["Value"], add1.inputs[0])
        return add1

    slope_weak = _slope_chain(AO_SLOPE_C_WEAK, AO_SLOPE_POWER_WEAK)
    slope_strong = _slope_chain(AO_SLOPE_C_STRONG, AO_SLOPE_POWER_STRONG)

    # Lerp tiers by clamped object_index (0 -> weak, 1 -> strong).
    obj_info = nt.nodes.new("ShaderNodeObjectInfo")
    idx_clamp = nt.nodes.new("ShaderNodeMath")
    idx_clamp.operation = 'MINIMUM'
    idx_clamp.inputs[1].default_value = 1.0
    idx_clamp.use_clamp = True
    nt.links.new(obj_info.outputs["Object Index"], idx_clamp.inputs[0])

    diff = nt.nodes.new("ShaderNodeMath")
    diff.operation = 'SUBTRACT'
    nt.links.new(slope_strong.outputs["Value"], diff.inputs[0])
    nt.links.new(slope_weak.outputs["Value"], diff.inputs[1])

    scaled = nt.nodes.new("ShaderNodeMath")
    scaled.operation = 'MULTIPLY'
    nt.links.new(diff.outputs["Value"], scaled.inputs[0])
    nt.links.new(idx_clamp.outputs["Value"], scaled.inputs[1])

    add = nt.nodes.new("ShaderNodeMath")
    add.operation = 'ADD'
    nt.links.new(slope_weak.outputs["Value"], add.inputs[0])
    nt.links.new(scaled.outputs["Value"], add.inputs[1])

    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = 'MULTIPLY'
    nt.links.new(ao_node.outputs["Color"], mul.inputs[0])
    nt.links.new(add.outputs["Value"], mul.inputs[1])
    nt.links.new(mul.outputs["Value"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out_node.inputs["Surface"])

    prev = {
        "engine": scene.render.engine,
        "resx": scene.render.resolution_x,
        "resy": scene.render.resolution_y,
        "pct": scene.render.resolution_percentage,
        "override": scene.view_layers[0].material_override,
        "fp": scene.render.filepath,
        "fmt": scene.render.image_settings.file_format,
        "cmode": scene.render.image_settings.color_mode,
        "cdepth": scene.render.image_settings.color_depth,
        "cam": scene.camera,
        "film_transparent": scene.render.film_transparent,
        "view_transform": scene.view_settings.view_transform,
        "look": scene.view_settings.look,
        "exposure": scene.view_settings.exposure,
        "gamma": scene.view_settings.gamma,
    }

    # Linear Standard view transform so 1.0 emission stays 1.0 in PNG.
    try:
        scene.view_settings.view_transform = 'Standard'
    except TypeError:
        pass
    try:
        scene.view_settings.look = 'None'
    except TypeError:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    scene.render.engine = 'CYCLES'
    try:
        scene.cycles.samples = samples
    except AttributeError:
        pass
    _enable_cycles_gpu(scene)
    scene.render.resolution_x = BAKE_IMG_SIZE
    scene.render.resolution_y = BAKE_IMG_SIZE
    scene.render.resolution_percentage = 100
    scene.view_layers[0].material_override = ao_mat
    scene.camera = cam_obj
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'BW'
    scene.render.image_settings.color_depth = '8'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

    _restore_hide(prev_hidden, "hide_render")
    scene.render.engine = prev["engine"]
    scene.render.resolution_x = prev["resx"]
    scene.render.resolution_y = prev["resy"]
    scene.render.resolution_percentage = prev["pct"]
    scene.view_layers[0].material_override = prev["override"]
    scene.render.filepath = prev["fp"]
    scene.render.image_settings.file_format = prev["fmt"]
    scene.render.image_settings.color_mode = prev["cmode"]
    scene.render.image_settings.color_depth = prev["cdepth"]
    scene.camera = prev["cam"]
    scene.render.film_transparent = prev["film_transparent"]
    try:
        scene.view_settings.view_transform = prev["view_transform"]
    except TypeError:
        pass
    try:
        scene.view_settings.look = prev["look"]
    except TypeError:
        pass
    scene.view_settings.exposure = prev["exposure"]
    scene.view_settings.gamma = prev["gamma"]

    for o, pi in prev_pass_index:
        try:
            o.pass_index = pi
        except ReferenceError:
            pass

    bpy.data.objects.remove(cam_obj)
    bpy.data.cameras.remove(cam_data)
    bpy.data.materials.remove(ao_mat)

    _apply_mask_to_image_file(output_path, mask)
    _clip_near_white(output_path, AO_NEAR_WHITE_CUTOFF)
    log(f"  AO saved -> {short_path(output_path)}")


def _add_split_layer_edge_darken(nt, shade_socket):
    """Append a footprint-distance rim darken to an existing split-
    layer shader graph.

    Returns (new_shade_socket, edge_tex_node). The caller must assign
    ``edge_tex_node.image`` to a per-layer rim multiplier PNG (see
    :func:`_build_rim_multiplier_png`) before each render, and clear /
    remove it afterwards. Returns (shade_socket, None) when the
    feature is disabled (STRENGTH <= 0), in which case no nodes are
    added and no per-layer setup is required.

    Shader math (all per shading sample, so Cycles' pixel AA applies
    directly on the mesh silhouette):

        edge_mult = RGBToBW( edge_tex.sample(window_xy_v_flipped) )
        shade'    = shade * edge_mult
    """
    if SPLIT_LAYER_EDGE_SHADER_STRENGTH <= 0.0:
        return shade_socket, None

    # Window output is bottom-up (UV=(0,0) = screen bottom-left), and
    # Blender's TexImage sampling is also bottom-up (UV=(0,0) = image
    # bottom-left, i.e. the LAST row of the numpy array we wrote to
    # the PNG file). The two conventions match, so we can wire
    # Window -> TexImage.Vector directly without any V flip.
    tc = nt.nodes.new("ShaderNodeTexCoord")
    edge_tex = nt.nodes.new("ShaderNodeTexImage")
    # Linear keeps the ring edges smooth at sub-pixel scale; EXTEND
    # stops sampling from wrapping at the borders.
    edge_tex.interpolation = 'Linear'
    edge_tex.extension = 'EXTEND'
    nt.links.new(tc.outputs["Window"], edge_tex.inputs["Vector"])

    # Gray PNG -> scalar multiplier.
    rgb2bw = nt.nodes.new("ShaderNodeRGBToBW")
    nt.links.new(edge_tex.outputs["Color"], rgb2bw.inputs["Color"])

    shaded = nt.nodes.new("ShaderNodeMath")
    shaded.operation = 'MULTIPLY'
    nt.links.new(shade_socket, shaded.inputs[0])
    nt.links.new(rgb2bw.outputs["Val"], shaded.inputs[1])
    return shaded.outputs["Value"], edge_tex


def _build_rim_multiplier_png(
    target_objs: List[bpy.types.Object],
    tmp_path: str,
    size: int = BAKE_IMG_SIZE,
) -> bool:
    """Write a grayscale rim-multiplier PNG for the given targets.

    Rasterizes their tight XY footprint, runs cv2.distanceTransform,
    then maps distance -> multiplier in [1 - STRENGTH, 1]:

        t        = clamp(dist_px / RADIUS_PX, 0, 1)
        factor   = (1 - t)**POWER * STRENGTH
        mult     = 1 - factor                  # inside footprint
                 = 1                           # outside (unused: alpha 0)

    Returns True on success, False if the footprint is empty. The PNG
    is saved top-down (row 0 at the top) and intended to be loaded
    with colorspace 'Non-Color' so the stored byte value is used as-is
    as a linear multiplier.
    """
    footprint = _rasterize_targets_footlog(
        target_objs, size=size, margin_px=0,
    )
    if not footprint.any():
        return False
    inside = footprint.astype(np.uint8) * 255
    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
    R = max(float(SPLIT_LAYER_EDGE_SHADER_RADIUS_PX), 1e-6)
    P = float(SPLIT_LAYER_EDGE_SHADER_POWER)
    S = float(SPLIT_LAYER_EDGE_SHADER_STRENGTH)
    t = np.clip(dist / R, 0.0, 1.0)
    factor = ((1.0 - t) ** P) * S
    mult = 1.0 - factor
    mult = np.where(footprint, mult, 1.0)
    img = np.clip(mult * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    cv2.imwrite(tmp_path, img)
    return True


def render_split_layers_ao(
    layers: List[Tuple[str, List[Tuple[List[bpy.types.Object], str]], str]],
    mask: np.ndarray,
    samples: int = 32,
    ao_distance: float = 10.0,
    slope_c: float = AO_SLOPE_C_SL,
    slope_power: float = AO_SLOPE_POWER_SL,
    announce: Optional[Callable[[str], None]] = None,
    skip_teardown: bool = False,
) -> List[str]:
    """Renders
    multiple split layers in a single setup/teardown. ``layers`` is a
    list of ``(output_path, groups, label)``. Camera,
    material, scene settings, and global hide/colour bookkeeping are
    done ONCE; only the per-layer hide_render flips and border
    rectangle change between renders. Returns the list of output paths
    written.

    ``announce`` (optional) is called with each layer's ``label``
    immediately before that layer's render starts, so the caller can
    emit a progress line at the right moment (e.g. a ``[bake i/N]``
    counter). When ``None`` the labels are printed as-is.
    """
    from time import time
    nnn = time()

    if not layers:
        return []

    scene = bpy.context.scene

    # Union of every target object across every layer. These are the
    # only meshes that may ever be visible during this batch; all
    # other meshes get hide_render=True once, up front.
    union_objs: List[bpy.types.Object] = []
    union_names: set = set()
    per_layer_sets: List[set] = []
    for _out, grps, _label in layers:
        lset: set = set()
        for grp_objs, _color in grps:
            for o in grp_objs:
                if o is None:
                    continue
                if o.name not in union_names:
                    union_names.add(o.name)
                    union_objs.append(o)
                lset.add(o.name)
        per_layer_sets.append(lset)

    other_meshes = [
        o for o in bpy.data.objects
        if o.type == 'MESH' and o.name not in union_names
    ]
    prev_hidden = _set_hide(other_meshes, True, "hide_render")

    # Hide every union mesh too; each layer will un-hide its own
    # subset inside the loop. Save prior hide_render so we can fully
    # restore at teardown.
    prev_union_hide = _set_hide(union_objs, True, "hide_render")

    # Save object.color for every union object once. Per-layer we just
    # stamp the appropriate tint; restore happens once at the end.
    prev_obj_color: List[Tuple[bpy.types.Object, Tuple[float, float, float, float]]] = [
        (o, tuple(o.color)) for o in union_objs
    ]

    cam_data = bpy.data.cameras.new("AO_Cam_SL")
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = BAKE_IMG_SIZE * BAKE_PIXEL_SIZE_M
    cam_data.clip_start = 0.1
    cam_data.clip_end = BAKE_CAM_Z * 2.0
    cam_obj = bpy.data.objects.new("AO_Cam_SL", cam_data)
    cam_obj.location = (0.0, 0.0, BAKE_CAM_Z)
    cam_obj.rotation_euler = (0.0, 0.0, 0.0)
    scene.collection.objects.link(cam_obj)

    ao_mat = bpy.data.materials.new("AO_SplitLayer")
    ao_mat.use_nodes = True
    nt = ao_mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out_node = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    ao_node = nt.nodes.new("ShaderNodeAmbientOcclusion")
    try:
        ao_node.samples = 16
    except AttributeError:
        pass
    ao_node.inputs["Distance"].default_value = ao_distance
    ao_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)

    geom = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geom.outputs["Normal"], sep.inputs["Vector"])

    z_pos = nt.nodes.new("ShaderNodeMath")
    z_pos.operation = 'MAXIMUM'
    z_pos.inputs[1].default_value = 0.0
    nt.links.new(sep.outputs["Z"], z_pos.inputs[0])

    pz = nt.nodes.new("ShaderNodeMath")
    pz.operation = 'POWER'
    pz.inputs[1].default_value = float(slope_power)
    nt.links.new(z_pos.outputs["Value"], pz.inputs[0])
    sub = nt.nodes.new("ShaderNodeMath")
    sub.operation = 'SUBTRACT'
    sub.inputs[1].default_value = 1.0
    nt.links.new(pz.outputs["Value"], sub.inputs[0])
    scl = nt.nodes.new("ShaderNodeMath")
    scl.operation = 'MULTIPLY'
    scl.inputs[1].default_value = float(slope_c)
    nt.links.new(sub.outputs["Value"], scl.inputs[0])
    slope = nt.nodes.new("ShaderNodeMath")
    slope.operation = 'ADD'
    slope.inputs[1].default_value = 1.0
    nt.links.new(scl.outputs["Value"], slope.inputs[0])

    shade = nt.nodes.new("ShaderNodeMath")
    shade.operation = 'MULTIPLY'
    nt.links.new(ao_node.outputs["Color"], shade.inputs[0])
    nt.links.new(slope.outputs["Value"], shade.inputs[1])

    shade_out, edge_tex_node = _add_split_layer_edge_darken(
        nt, shade.outputs["Value"],
    )

    # Shader-level minimum on the final shade scalar. This caps how
    # dark AO / slope / edge darkening can drive the tint (so e.g.
    # tightly-walled ghouse interiors never render near-black). The
    # cap is applied to the 0..1 multiplier BEFORE the tint, so the
    # output color is clamped to `min_v * tint` per pixel while all
    # anti-aliasing and gradients above the floor are preserved.
    min_v = max(0.0, min(1.0, float(MIN_SPLIT_LAYER_VALUE) / 255.0))
    if min_v > 0.0:
        shade_min = nt.nodes.new("ShaderNodeMath")
        shade_min.operation = 'MAXIMUM'
        shade_min.inputs[1].default_value = min_v
        nt.links.new(shade_out, shade_min.inputs[0])
        shade_out = shade_min.outputs["Value"]

    tint = nt.nodes.new("ShaderNodeObjectInfo")
    mix = nt.nodes.new("ShaderNodeVectorMath")
    mix.operation = 'MULTIPLY'
    nt.links.new(tint.outputs["Color"], mix.inputs[0])
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(shade_out, comb.inputs["X"])
    nt.links.new(shade_out, comb.inputs["Y"])
    nt.links.new(shade_out, comb.inputs["Z"])
    nt.links.new(comb.outputs["Vector"], mix.inputs[1])
    nt.links.new(mix.outputs["Vector"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out_node.inputs["Surface"])

    prev = {
        "engine": scene.render.engine,
        "resx": scene.render.resolution_x,
        "resy": scene.render.resolution_y,
        "pct": scene.render.resolution_percentage,
        "override": scene.view_layers[0].material_override,
        "fp": scene.render.filepath,
        "fmt": scene.render.image_settings.file_format,
        "cmode": scene.render.image_settings.color_mode,
        "cdepth": scene.render.image_settings.color_depth,
        "cam": scene.camera,
        "film_transparent": scene.render.film_transparent,
        "view_transform": scene.view_settings.view_transform,
        "look": scene.view_settings.look,
        "exposure": scene.view_settings.exposure,
        "gamma": scene.view_settings.gamma,
        "use_border": scene.render.use_border,
        "use_crop_to_border": scene.render.use_crop_to_border,
        "bmin_x": scene.render.border_min_x,
        "bmax_x": scene.render.border_max_x,
        "bmin_y": scene.render.border_min_y,
        "bmax_y": scene.render.border_max_y,
    }

    try:
        scene.view_settings.view_transform = 'Standard'
    except TypeError:
        pass
    try:
        scene.view_settings.look = 'None'
    except TypeError:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    scene.render.engine = 'CYCLES'
    try:
        scene.cycles.samples = samples
    except AttributeError:
        pass
    _enable_cycles_gpu(scene)
    scene.render.resolution_x = BAKE_IMG_SIZE
    scene.render.resolution_y = BAKE_IMG_SIZE
    scene.render.resolution_percentage = 100
    scene.view_layers[0].material_override = ao_mat
    scene.camera = cam_obj
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'
    scene.render.use_border = True
    scene.render.use_crop_to_border = False

    size = BAKE_IMG_SIZE
    written: List[str] = []

    # Temp dir for per-layer rim multiplier PNGs (one per layer).
    # The PNGs are loaded via bpy.data.images.load so Cycles can
    # sample them in the shader; removed immediately after each
    # render completes.
    import tempfile
    rim_tmp_dir: Optional[str] = None
    if edge_tex_node is not None:
        rim_tmp_dir = tempfile.mkdtemp(prefix="fh_split_rim_")

    nn = time()
    log(f"  [split-layers] setup in {nn - nnn:.2f}s "
          f"({len(layers)} layer(s), {len(union_objs)} target mesh(es), "
          f"{len(other_meshes)} other mesh(es) hidden)")
    nnn = nn

    try:
        for (output_path, groups, label), lset in zip(layers, per_layer_sets):
            output_path = os.path.abspath(output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            if announce is not None:
                announce(label)
            else:
                log(label)

            # Stamp tints for this layer's groups. We overwrite color
            # unconditionally; it'll be restored globally at teardown.
            layer_target_objs: List[bpy.types.Object] = []
            for grp_objs, color_hex in groups:
                hc = color_hex.lstrip("#")
                r = int(hc[0:2], 16) / 255.0
                g = int(hc[2:4], 16) / 255.0
                b = int(hc[4:6], 16) / 255.0
                for o in grp_objs:
                    if o is None:
                        continue
                    o.color = (r, g, b, 1.0)
                    layer_target_objs.append(o)

            # Un-hide just this layer's targets for the render.
            for o in layer_target_objs:
                o.hide_render = False

            try:
                footprint = _rasterize_targets_footlog(
                    layer_target_objs, size=size, margin_px=16,
                )
                if not footprint.any():
                    blank = np.zeros((size, size, 4), dtype=np.uint8)
                    write_png8_rgba(output_path, blank)
                    log(f"  [split-layer] empty footprint; wrote blank "
                          f"-> {short_path(output_path)}")
                else:
                    rows = np.where(footprint.any(axis=1))[0]
                    cols = np.where(footprint.any(axis=0))[0]
                    i0, i1 = int(rows[0]), int(rows[-1])
                    j0, j1 = int(cols[0]), int(cols[-1])
                    scene.render.border_min_x = j0 / size
                    scene.render.border_max_x = (j1 + 1) / size
                    scene.render.border_min_y = 1.0 - (i1 + 1) / size
                    scene.render.border_max_y = 1.0 - i0 / size
                    tw, th = j1 - j0 + 1, i1 - i0 + 1

                    log(f"  [split-layer] tile {tw}x{th} "
                          f"of {size}x{size} "
                          f"({100.0 * tw * th / (size * size):.2f}%) "
                          f"-> {short_path(output_path)}")

                    # Build + load the per-layer rim multiplier image
                    # and hook it to the shader's TexImage node. We
                    # use the TIGHT footprint (margin=0) here so the
                    # rim rides right up against the real silhouette.
                    rim_img_blender = None
                    if edge_tex_node is not None and rim_tmp_dir is not None:
                        layer_stem = os.path.splitext(
                            os.path.basename(output_path),
                        )[0]
                        rim_png = os.path.join(
                            rim_tmp_dir,
                            f"rim_{layer_stem}.png",
                        )
                        if _build_rim_multiplier_png(
                            layer_target_objs, rim_png, size=size,
                        ):
                            rim_img_blender = bpy.data.images.load(
                                rim_png, check_existing=False,
                            )
                            try:
                                rim_img_blender.colorspace_settings.name = (
                                    'Non-Color'
                                )
                            except (AttributeError, TypeError):
                                pass
                            edge_tex_node.image = rim_img_blender
                        else:
                            # No targets rasterized; skip rim.
                            edge_tex_node.image = None

                    try:
                        scene.render.filepath = output_path
                        bpy.ops.render.render(write_still=True)
                    finally:
                        if edge_tex_node is not None:
                            edge_tex_node.image = None
                        if rim_img_blender is not None:
                            try:
                                bpy.data.images.remove(rim_img_blender)
                            except (RuntimeError, ReferenceError):
                                pass

                written.append(output_path)
            finally:
                # Re-hide this layer's targets before moving on.
                for o in layer_target_objs:
                    try:
                        o.hide_render = True
                    except ReferenceError:
                        pass

            _apply_mask_to_image_file(output_path, mask, preserve_alpha=True)
    finally:
        nn = time()
        log(f"  [split-layers] all renders in {nn - nnn:.2f}s")
        nnn = nn

        if rim_tmp_dir is not None:
            import shutil
            try:
                shutil.rmtree(rim_tmp_dir, ignore_errors=True)
            except OSError:
                pass

        if skip_teardown:
            log("  [split-layers] skipping teardown "
                  "(process about to exit)")
        else:
            _restore_hide(prev_hidden, "hide_render")
            _restore_hide(prev_union_hide, "hide_render")
            scene.render.engine = prev["engine"]
            scene.render.resolution_x = prev["resx"]
            scene.render.resolution_y = prev["resy"]
            scene.render.resolution_percentage = prev["pct"]
            scene.view_layers[0].material_override = prev["override"]
            scene.render.filepath = prev["fp"]
            scene.render.image_settings.file_format = prev["fmt"]
            scene.render.image_settings.color_mode = prev["cmode"]
            scene.render.image_settings.color_depth = prev["cdepth"]
            scene.camera = prev["cam"]
            scene.render.film_transparent = prev["film_transparent"]
            try:
                scene.view_settings.view_transform = prev["view_transform"]
            except TypeError:
                pass
            try:
                scene.view_settings.look = prev["look"]
            except TypeError:
                pass
            scene.view_settings.exposure = prev["exposure"]
            scene.view_settings.gamma = prev["gamma"]
            scene.render.use_border = prev["use_border"]
            scene.render.use_crop_to_border = prev["use_crop_to_border"]
            scene.render.border_min_x = prev["bmin_x"]
            scene.render.border_max_x = prev["bmax_x"]
            scene.render.border_min_y = prev["bmin_y"]
            scene.render.border_max_y = prev["bmax_y"]

            bpy.data.objects.remove(cam_obj)
            bpy.data.cameras.remove(cam_data)
            bpy.data.materials.remove(ao_mat)

            for o, c in prev_obj_color:
                try:
                    o.color = c
                except ReferenceError:
                    pass

            nn = time()
            log(f"  [split-layers] teardown in {nn - nnn:.2f}s")

    return written
