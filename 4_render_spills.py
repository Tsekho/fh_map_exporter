"""Render top-down bakes for every region .blend.

Outputs (each flag gates its own set of per-region PNGs):
    -svg  ->  export/svg_layers/<layer>/<Region>.png
              + export/bridges_aim/<Region>.png (procedural aim lines)
    -ao   ->  export/ao/<Region>.png
    -hm   ->  export/heightmap_landscape/<Region>.png
              + export/heightmap_water/<Region>.png
    -id   ->  export/id/<category>/<Region>.png (incl. id/water/)
    -r    ->  export/roads/<Region>.png
    -b    ->  export/beaches/<Region>.png
    -sl   ->  export/split_layers/<layer>/<Region>.png

Usage:
    python 4_render_spills.py [RegionName] [-a]
        [-svg] [-ao] [-hm] [-id] [-r] [-b] [-sl]
    (no bake flags = all bakes)
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import bpy

from utils.config import (
    AO_DIR,
    BEACHES_CATS,
    BEACHES_DIR,
    BRIDGES_AIM_DIR,
    BRIDGES_AIM_MIN_DEPTH_M,
    BRIDGES_AIM_WATER_THRESH,
    CATEGORY_COLORS,
    HM_LANDSCAPE_DIR,
    HM_WATER_DIR,
    ID_DIR,
    ID_RECOLOR,
    ID_SSAA,
    NUM_WORKERS_SPILLS,
    NUM_WORKERS_SVG,
    MASK_FILE,
    RESERVED_CATS,
    ROADS_CATS,
    ROADS_DIR,
    SPILL_DIR,
    SPLINE_COLORS,
    SPLINE_LAYER_SSAA,
    SPLINE_LAYER_TERRAIN_DROP,
    SPLIT_LAYERS,
    SPLIT_LAYERS_DIR,
    SVG_LAYERS,
    SVG_LAYERS_DIR,
    TERRAIN_CULL_MIN_Z,
    TERRAIN_SPLINE_CATS,
    TERRAIN_WHITELIST,
    Z_FIX_REGIONS,
)

from utils.bake import (
    bake_spline_layer,
    clear_bvh_cache,
    clear_mesh_cache,
    raycast_heightmap,
    raycast_id_ssaa_per_category,
    render_ao,
    render_split_layers_ao,
)
from utils import progress, tui
from utils.parallel import run_parallel_subprocesses
from utils.svg_render import render_bridges_aim_layer, render_svg_layers


def _load_region_water_dist(region_name: str) -> Optional[np.ndarray]:
    """Distance-to-shore field (float32, 0 on non-water) from the per-region
    water ID coverage PNG, or None when it's not on disk yet.

    A pixel counts as water when its coverage is at least
    BRIDGES_AIM_WATER_THRESH AND, when the region's landscape/water
    heightmap bakes are present, it is submerged by at least
    BRIDGES_AIM_MIN_DEPTH_M metres (freighters ground in shallower water).
    Bridge decks read as non-water in the water bake (they occlude the
    surface), so aim lines stop before crossing another bridge. The
    distance transform lets the aim tracer steer toward the more open
    side of the channel."""
    water_path = ID_DIR / "water" / f"{region_name}.png"
    if not water_path.is_file():
        return None
    wc = cv2.imread(str(water_path), cv2.IMREAD_GRAYSCALE)
    if wc is None:
        return None
    water = wc >= BRIDGES_AIM_WATER_THRESH
    if BRIDGES_AIM_MIN_DEPTH_M > 0:
        hl = cv2.imread(
            str(HM_LANDSCAPE_DIR / f"{region_name}.png"), cv2.IMREAD_UNCHANGED)
        hw = cv2.imread(
            str(HM_WATER_DIR / f"{region_name}.png"), cv2.IMREAD_UNCHANGED)
        if hl is None or hw is None:
            print("  [WARN] bridges_aim: heightmap bake(s) missing; "
                  "navigability depth gate skipped")
        else:
            depth_cm = hw.astype(np.int32) - hl.astype(np.int32)
            deep = depth_cm >= int(round(BRIDGES_AIM_MIN_DEPTH_M * 100))
            water &= (hl != 0) & (hw != 0) & deep
    water = water.astype(np.uint8) * 255
    return cv2.distanceTransform(water, cv2.DIST_L2, 3)


def _collect_focus_terrain_objects(region_name: str) -> list:
    root = bpy.data.collections.get(region_name)
    if root is None:
        return []
    for child in root.children:
        if child.name.lower() == "terrain":
            return [o for o in child.all_objects if o.type == "MESH"]
    return []


def _cull_terrain_below(objs: list, min_z: float) -> int:
    """Delete verts with world-space Z < min_z. Returns count deleted."""
    import bmesh

    total = 0
    seen_meshes: set = set()

    for obj in objs:
        if obj is None or obj.type != "MESH" or obj.data is None:
            continue
        if obj.data.users > 1 or obj.data.name in seen_meshes:
            obj.data = obj.data.copy()
        seen_meshes.add(obj.data.name)

        mw = obj.matrix_world.copy()
        bm = bmesh.new()
        try:
            bm.from_mesh(obj.data)
            doomed = [v for v in bm.verts if (mw @ v.co).z < min_z]
            if doomed:
                bmesh.ops.delete(bm, geom=doomed, context="VERTS")
                bm.to_mesh(obj.data)
                obj.data.update()
                total += len(doomed)
        finally:
            bm.free()

    return total


_DATA_SUFFIX_RE = re.compile(r"\.\d{3}$")


def _base_name(name: str) -> str:
    return _DATA_SUFFIX_RE.sub("", name)


def _collect_splines(
    region_name: str,
) -> Dict[str, List[bpy.types.Object]]:
    """Return {spline_category: [objects]} for the focus region's
    'splines' collection. Empty dict if no splines present."""
    root = bpy.data.collections.get(region_name)
    if root is None:
        return {}
    splines_root = None
    for child in root.children:
        if _base_name(child.name).lower() == "splines":
            splines_root = child
            break
    if splines_root is None:
        return {}
    out: Dict[str, List[bpy.types.Object]] = {}
    for cat_coll in splines_root.children:
        base = _base_name(cat_coll.name)
        out[base] = [o for o in cat_coll.all_objects if o.type == "MESH"]
    return out


def _collect_focus_objects(
    region_name: str,
) -> Optional[Dict[str, List[bpy.types.Object]]]:
    """Return {category: [objects]} for the focus region, folding neighbor
    spill into the same non-reserved buckets. None if root collection
    missing."""
    root = bpy.data.collections.get(region_name)
    if root is None:
        return None

    buckets: Dict[str, List[bpy.types.Object]] = {}
    for child in root.children:
        buckets.setdefault(child.name, []).extend(child.all_objects)

    spill_added: Dict[str, int] = {
        c: 0 for c in buckets if c not in RESERVED_CATS
    }
    scene = bpy.context.scene
    for top in scene.collection.children:
        if top.name == region_name:
            continue
        for cat_coll in top.children:
            base = _base_name(cat_coll.name)
            if base in RESERVED_CATS:
                continue
            extra = list(cat_coll.all_objects)
            buckets.setdefault(base, []).extend(extra)
            spill_added[base] = spill_added.get(base, 0) + len(extra)

    if any(spill_added.values()):
        summary = ", ".join(
            f"{c}+{n}" for c, n in spill_added.items() if n
        )
        print(f"  Neighbor spill folded in: {summary}")

    for r in RESERVED_CATS:
        buckets.setdefault(r, [])

    return buckets


def _spill_categories(objs: Dict[str, List[bpy.types.Object]]) -> List[str]:
    """Sorted non-reserved category names that participate in the
    ao/hm/id bakes. A spill category is included only if it appears in
    TERRAIN_WHITELIST; every other category present in the .blend is
    ignored by those bakes (it may still drive split-layer renders)."""
    whitelist = set(TERRAIN_WHITELIST)
    return sorted(c for c in objs if c not in RESERVED_CATS
                  and c in whitelist)


def _split_layer_objs(
    objs: Dict[str, List[bpy.types.Object]],
) -> List[bpy.types.Object]:
    """Flat list of all objects belonging to any split-layer category."""
    out: List[bpy.types.Object] = []
    for cats in SPLIT_LAYERS.values():
        for c in cats:
            out.extend(objs.get(c, []))
    return out


def render_one(
    blend_path: Path,
    mask: np.ndarray,
    do_ao: bool,
    do_hm: bool,
    do_id: bool,
    do_roads: bool,
    do_beaches: bool,
    do_split_layers: bool,
    do_svg: bool,
) -> bool:
    region_name = blend_path.stem
    print(f"=== {region_name} ===")

    # SVG layers are driven purely by the region JSON, so render them
    # before opening the .blend (cheap, no Blender state required).
    ok = True
    if do_svg:
        print(f"  [svg] rasterizing {region_name} svg layers")
        try:
            if not render_svg_layers(region_name):
                ok = False
        except Exception as exc:
            print(f"  [WARN] SVG layer render failed: {exc}")
            ok = False

    needs_blend = (do_ao or do_hm or do_id or do_roads or do_beaches
                   or do_split_layers)
    if not needs_blend:
        # SVG-only run: skip the multi-GB scene load.
        return ok

    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    # Neither cache may survive a file load: both are keyed on object
    # names/matrices, which can repeat across regions, so a serial
    # in-process run would reuse the previous region's data.
    clear_mesh_cache()
    clear_bvh_cache()

    bake_total = (
        (2 if do_hm else 0)
        + (2 if do_id else 0)
        + (1 if do_ao else 0)
        + (len(SPLIT_LAYERS) if do_split_layers else 0)
        + (1 if do_roads else 0)
        + (1 if do_beaches else 0)
    )
    bake_i = [0]
    bw = len(str(max(bake_total, 1)))

    def _tag(label: str) -> str:
        bake_i[0] += 1
        return f"[bake {bake_i[0]:>{bw}}/{bake_total}] {label}"

    objs = _collect_focus_objects(region_name)
    if objs is None:
        print(f"  [WARN] root collection '{region_name}' not found; skipped")
        return False

    spill_cats = _spill_categories(objs)
    split_objs = _split_layer_objs(objs)

    # Whatever ID categories this region lacks can never be written, so
    # credit them rather than let the bar end short of full.
    if do_id:
        progress.announce_credit(ID_FILES_MAX - (len(spill_cats) + 3))

    excluded = sorted(
        c for c, lst in objs.items()
        if lst and c not in RESERVED_CATS and c not in set(TERRAIN_WHITELIST)
    )
    if excluded:
        print(f"  Excluded from ao/hm/id (not in TERRAIN_WHITELIST): "
              f"{', '.join(excluded)}")

    if split_objs:
        present = []
        for layer, cats in SPLIT_LAYERS.items():
            cats_with_objs = [c for c in cats if objs.get(c)]
            if cats_with_objs:
                present.append(f"{layer}({'+'.join(cats_with_objs)})")
        if present:
            print(f"  Split layers: {', '.join(present)}")

    splines = _collect_splines(region_name)
    terrain_spline_objs: List[bpy.types.Object] = []
    for c in TERRAIN_SPLINE_CATS:
        terrain_spline_objs.extend(splines.get(c, []))
    all_spline_objs: List[bpy.types.Object] = []
    for v in splines.values():
        all_spline_objs.extend(v)
    nonterrain_spline_objs = [
        o for o in all_spline_objs if o not in set(terrain_spline_objs)
    ]
    if splines:
        summary = ", ".join(f"{k}:{len(v)}" for k, v in splines.items() if v)
        if summary:
            print(f"  Splines: {summary}")

    if region_name in Z_FIX_REGIONS:
        focus_terrain = _collect_focus_terrain_objects(region_name)
        if focus_terrain:
            n_culled = _cull_terrain_below(focus_terrain, TERRAIN_CULL_MIN_Z)
            print(f"  [terrain-cull] {region_name}: deleted {n_culled} "
                  f"vertex(es) with z < {TERRAIN_CULL_MIN_Z} m")
            # Vertex edits aren't covered by either cache key.
            clear_mesh_cache()
            clear_bvh_cache()

    def _out(sub: Path) -> str:
        return str((sub / f"{region_name}.png").resolve())

    def _spill_objs() -> List[bpy.types.Object]:
        acc: List[bpy.types.Object] = []
        for c in spill_cats:
            acc.extend(objs.get(c, []))
        return acc

    # Bake order is deliberate: heightmap(landscape)+ID share an object
    # union, as do heightmap(water)+water coverage. Running each pair
    # back-to-back lets the single-entry BVH cache hit. Keep interleaved.
    if do_hm:
        try:
            print(_tag("heightmap (landscape)"))
            hm_objs = objs["terrain"] + _spill_objs() + terrain_spline_objs
            raycast_heightmap(
                _out(HM_LANDSCAPE_DIR),
                mask, hm_objs,
                occluders=objs["deep_water"],
            )
        except Exception as exc:
            print(f"  [WARN] heightmap bake failed: {exc}")
            ok = False

    if do_id:
        try:
            print(_tag(
                f"ID per-category coverage (SSAA {ID_SSAA}x{ID_SSAA})"
            ))
            id_cats: Dict[str, List[bpy.types.Object]] = {
                "terrain":    objs["terrain"] + terrain_spline_objs,
                "deep_water": objs["deep_water"],
            }
            for c in spill_cats:
                id_cats[c] = objs[c]

            out_paths: Dict[str, str] = {
                cat: str(
                    (ID_DIR / cat / f"{region_name}.png").resolve()
                )
                for cat in id_cats
            }
            raycast_id_ssaa_per_category(
                out_paths, mask, id_cats,
                occluders=None,
                samples_per_side=ID_SSAA,
            )
        except Exception as exc:
            print(f"  [WARN] ID coverage bake failed: {exc}")
            ok = False

    if do_hm:
        try:
            print(_tag("heightmap (water surface)"))
            hm_water_objs = (
                objs["terrain"] + _spill_objs()
                + terrain_spline_objs + objs["water"]
            )
            raycast_heightmap(
                _out(HM_WATER_DIR),
                mask, hm_water_objs,
                occluders=objs["deep_water"],
            )
        except Exception as exc:
            print(f"  [WARN] heightmap_water bake failed: {exc}")
            ok = False

    if do_id:
        try:
            print(_tag(f"water coverage (SSAA {ID_SSAA}x{ID_SSAA})"))
            water_occluders = (
                objs["terrain"] + _spill_objs() + terrain_spline_objs
                + objs["deep_water"]
            )
            raycast_id_ssaa_per_category(
                {"water": str(
                    (ID_DIR / "water" / f"{region_name}.png").resolve()
                )},
                mask,
                {"water": objs["water"]},
                occluders=water_occluders,
                samples_per_side=ID_SSAA,
            )
        except Exception as exc:
            print(f"  [WARN] water coverage bake failed: {exc}")
            ok = False

    if do_ao:
        try:
            print(_tag("AO"))
            _ray_keys = (
                "diffuse", "glossy", "transmission",
                "volume_scatter", "shadow",
            )
            prev_vis = []
            for o in objs["deep_water"]:
                saved = {k: getattr(o, f"visible_{k}", True) for k in _ray_keys}
                prev_vis.append((o, saved))
                for k in _ray_keys:
                    if hasattr(o, f"visible_{k}"):
                        setattr(o, f"visible_{k}", False)
            non_whitelist_objs: List[bpy.types.Object] = []
            for c, lst in objs.items():
                if c in RESERVED_CATS or c in set(TERRAIN_WHITELIST):
                    continue
                non_whitelist_objs.extend(lst)
            try:
                render_ao(
                    _out(AO_DIR), mask,
                    hidden_objs=(
                        objs["water"] + nonterrain_spline_objs
                        + non_whitelist_objs
                    ),
                    strong_slope_objs=(
                        objs["terrain"] + terrain_spline_objs
                    ),
                )
            finally:
                for o, saved in prev_vis:
                    for k, v in saved.items():
                        if hasattr(o, f"visible_{k}"):
                            setattr(o, f"visible_{k}", v)
        except Exception as exc:
            print(f"  [WARN] AO bake failed: {exc}")
            ok = False

    def _run_spline_layer(out_dir: Path, target_cats: tuple, label: str,
                          include_terrain: bool = True) -> bool:
        """Build arguments for bake_spline_layer from region-local state
        (splines, objs, nonterrain_spline_objs) and dispatch."""
        targets: Dict[str, List[bpy.types.Object]] = {}
        for c in target_cats:
            tl = splines.get(c, [])
            if tl:
                targets[c] = tl

        target_set = set()
        for lst in targets.values():
            target_set.update(lst)

        occluders: List[bpy.types.Object] = []
        occluders.extend(_spill_objs())
        occluders.extend(objs["water"])
        occluders.extend(objs["deep_water"])
        for o in nonterrain_spline_objs:
            if o not in target_set:
                occluders.append(o)

        palette = {c: SPLINE_COLORS.get(c, CATEGORY_COLORS.get("splines",
                                                                "#FFFFFF"))
                   for c in target_cats}

        if include_terrain:
            print(_tag(
                f"{label} (terrain dropped "
                f"{SPLINE_LAYER_TERRAIN_DROP:.2f} m)"
            ))
            terrain_occluders: Optional[List[bpy.types.Object]] = list(
                objs["terrain"]
            )
            terrain_drop = SPLINE_LAYER_TERRAIN_DROP
        else:
            print(_tag(f"{label} (terrain excluded)"))
            terrain_occluders = None
            terrain_drop = 0.0

        return bake_spline_layer(
            _out(out_dir), mask,
            targets=targets,
            palette=palette,
            occluders=occluders,
            terrain_occluders=terrain_occluders,
            terrain_drop=terrain_drop,
            samples_per_side=SPLINE_LAYER_SSAA,
            label=label,
        )

    if do_roads:
        if not _run_spline_layer(ROADS_DIR, ROADS_CATS, "roads"):
            ok = False
        else:
            # Soften crossroads (T1/T2/T3 transitions) with a 3x3 blur on
            # the RGB channels only. To prevent black bleeding from
            # transparent pixels, do an alpha-weighted (premultiplied)
            # blur: blur premultiplied RGB and alpha separately, then
            # unpremultiply. The on-disk alpha channel is restored
            # unchanged so the layer's silhouette doesn't widen.
            roads_path = ROADS_DIR / f"{region_name}.png"
            roads_img = cv2.imread(str(roads_path), cv2.IMREAD_UNCHANGED)
            if (roads_img is None or roads_img.ndim != 3
                    or roads_img.shape[2] != 4):
                print("  [WARN] roads blur skipped "
                      "(unreadable or non-RGBA output)")
            else:
                bgr = roads_img[..., :3].astype(np.float32)
                a = roads_img[..., 3].astype(np.float32) / 255.0
                a3 = a[..., None]
                premul = bgr * a3
                premul_b = cv2.blur(premul, (3, 3))
                a_b = cv2.blur(a, (3, 3))
                denom = np.maximum(a_b, 1e-6)[..., None]
                new_bgr = np.clip(premul_b / denom, 0, 255).astype(np.uint8)
                visible = roads_img[..., 3] > 0
                roads_img[..., :3][visible] = new_bgr[visible]
                cv2.imwrite(str(roads_path), roads_img)
                print("  [roads] 3x3 alpha-weighted RGB blur applied "
                      "(alpha preserved)")

    if do_beaches:
        if not _run_spline_layer(BEACHES_DIR, BEACHES_CATS, "beaches",
                                 include_terrain=False):
            ok = False
        else:
            beach_path = BEACHES_DIR / f"{region_name}.png"
            terrain_path = ID_DIR / "terrain" / f"{region_name}.png"
            water_path = ID_DIR / "water" / f"{region_name}.png"
            if not (terrain_path.is_file() and water_path.is_file()):
                print("  [WARN] beaches land-mask skipped "
                      "(missing per-region terrain/water ID PNG)")
            else:
                beach_img = cv2.imread(str(beach_path), cv2.IMREAD_UNCHANGED)
                terrain_cov = cv2.imread(str(terrain_path),
                                         cv2.IMREAD_GRAYSCALE)
                water_cov = cv2.imread(str(water_path),
                                       cv2.IMREAD_GRAYSCALE)
                if (beach_img is None or terrain_cov is None
                        or water_cov is None or beach_img.ndim != 3
                        or beach_img.shape[2] != 4):
                    print("  [WARN] beaches land-mask skipped "
                          "(unreadable or non-RGBA inputs)")
                else:
                    non_water = (255 - water_cov).astype(np.uint16)
                    land = (
                        (terrain_cov.astype(np.uint16) * non_water + 127)
                        // 255
                    )
                    new_alpha = (
                        (beach_img[..., 3].astype(np.uint16) * land + 127)
                        // 255
                    ).astype(np.uint8)
                    beach_img[..., 3] = new_alpha
                    cv2.imwrite(str(beach_path), beach_img)
                    print("  [beaches] alpha masked to land "
                          "(terrain * non-water)")

    if do_split_layers:
        batch: List[Tuple[str, List[Tuple[List[bpy.types.Object], str]], str]] = []
        for layer, cat_colors in SPLIT_LAYERS.items():
            out_path = str(
                (SPLIT_LAYERS_DIR / layer / f"{region_name}.png").resolve()
            )
            groups: List[Tuple[List[bpy.types.Object], str]] = []
            group_cats: List[str] = []
            for cat, color_override in cat_colors.items():
                cat_objs = objs.get(cat, [])
                if not cat_objs:
                    continue
                color = (color_override
                         or ID_RECOLOR.get(cat)
                         or CATEGORY_COLORS.get(cat, "#FFFFFF"))
                groups.append((cat_objs, color))
                group_cats.append(cat)

            if not groups:
                print(_tag(
                    f"split layer '{layer}': no meshes; writing empty image"
                ))
                blank = np.zeros(
                    (2048, 2048, 4), dtype=np.uint8
                )
                import os as _os
                _os.makedirs(_os.path.dirname(out_path), exist_ok=True)
                from utils.png import write_png8_rgba
                write_png8_rgba(out_path, blank)
                continue

            summary = ", ".join(
                f"{c}:{len(o)}" for c, (o, _) in zip(group_cats, groups)
            )
            label = f"split layer '{layer}' (Cycles AO, {summary})"
            batch.append((out_path, groups, label))

        if batch:
            try:
                render_split_layers_ao(
                    batch, mask,
                    announce=lambda lbl: print(_tag(lbl)),
                    skip_teardown=True,
                )
            except Exception as exc:
                print(f"  [WARN] split layers bake failed: {exc}")
                ok = False

    if do_svg:
        # bridges_aim is rendered procedurally here (not by render_svg_layers)
        # so it can snap nearby bridges into curves and truncate straight
        # aim lines at the shoreline using this region's land mask.
        water_dist = _load_region_water_dist(region_name)
        if water_dist is None:
            print("  [bridges_aim] no water ID PNG; "
                  "aim lines drawn at full length (no steering/cut)")
        try:
            if not render_bridges_aim_layer(region_name, water_dist):
                ok = False
        except Exception as exc:
            print(f"  [WARN] bridges_aim render failed: {exc}")
            ok = False

    return ok


def _list_blends() -> List[Path]:
    if not SPILL_DIR.is_dir():
        return []
    return sorted(SPILL_DIR.glob("*.blend"))


# Bake flags, in BAKE_CHOICES order:
# (svg, ao, hm, id, roads, beaches, split_layers).
Flags = Tuple[bool, bool, bool, bool, bool, bool, bool]


def _output_candidates(region: str, flags: Flags) -> List[Path]:
    """Every PNG this run could write for ``region``. Re-evaluated on each
    poll, so ID category dirs created mid-bake are picked up."""
    svg, ao, hm, ids, roads, beaches, split = flags
    out: List[Path] = []
    if svg:
        out.extend(SVG_LAYERS_DIR / layer / f"{region}.png"
                   for layer in SVG_LAYERS)
        out.append(BRIDGES_AIM_DIR / f"{region}.png")
    if ao:
        out.append(AO_DIR / f"{region}.png")
    if hm:
        out.append(HM_LANDSCAPE_DIR / f"{region}.png")
        out.append(HM_WATER_DIR / f"{region}.png")
    if ids and ID_DIR.is_dir():
        out.extend(d / f"{region}.png"
                   for d in ID_DIR.iterdir() if d.is_dir())
    if roads:
        out.append(ROADS_DIR / f"{region}.png")
    if beaches:
        out.append(BEACHES_DIR / f"{region}.png")
    if split:
        out.extend(SPLIT_LAYERS_DIR / layer / f"{region}.png"
                   for layer in SPLIT_LAYERS)
    return out


# ID coverage is the only bake whose file count varies by region: terrain,
# deep_water and water plus one per spill category present. Bars are sized
# for a region that has all of them; render_one() credits the rest.
ID_SPILL_CATS = [c for c in TERRAIN_WHITELIST if c not in RESERVED_CATS]
ID_FILES_MAX = len(ID_SPILL_CATS) + 3


def _expected_outputs(flags: Flags) -> int:
    """Total the bar is sized for: every file a region could write."""
    svg, ao, hm, ids, roads, beaches, split = flags
    return (
        (len(SVG_LAYERS) + 1) * svg
        + 1 * ao
        + 2 * hm
        + ID_FILES_MAX * ids
        + 1 * roads
        + 1 * beaches
        + len(SPLIT_LAYERS) * split
    )


def pick_region_interactive(blends: List[Path]) -> Optional[List[Path]]:
    return tui.select_many(
        blends, "Regions to render",
        label_fn=lambda p: p.stem,
        noun="region",
    )


# Token, and what that bake writes. Order matches ask_bakes()'s return tuple.
BAKE_CHOICES: List[Tuple[str, str]] = [
    ("svg", "svg_layers/<layer> + bridges_aim"),
    ("ao", "ao"),
    ("hm", "heightmap_landscape + heightmap_water"),
    ("id", "id/<category> (incl. id/water)"),
    ("r", "roads"),
    ("b", "beaches"),
    ("sl", "split_layers/<layer>"),
]


def ask_bakes() -> Optional[Tuple[bool, bool, bool, bool, bool, bool, bool]]:
    picked = tui.select_many(
        [tok for tok, _ in BAKE_CHOICES],
        "Bakes to render",
        label_fn=lambda tok: f"{tok:<3}  ->  {dict(BAKE_CHOICES)[tok]}",
        short_fn=lambda tok: tok,
        noun="bake",
    )
    if picked is None:
        return None
    chosen = set(picked)
    return tuple(tok in chosen for tok, _ in BAKE_CHOICES)  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render bakes for every region .blend in export/blend_spill. "
            "If no bake flags are given, all bakes are produced."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("region_name", nargs="?",
                        help="Region stem; omit for interactive prompt")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Render every .blend in export/blend_spill")
    parser.add_argument("-ao", dest="do_ao", action="store_true",
                        help="Render the AO bake")
    parser.add_argument("-hm", dest="do_hm", action="store_true",
                        help="Render heightmap_landscape + heightmap_water")
    parser.add_argument("-id", dest="do_id", action="store_true",
                        help="Render per-category ID coverage "
                             "(incl. id/water)")
    parser.add_argument("-svg", dest="do_svg", action="store_true",
                        help="Rasterize SVG_LAYERS into "
                             "export/svg_layers/<layer>/<region>.png "
                             "and render the procedural bridges_aim layer")
    parser.add_argument("-r", dest="do_roads", action="store_true",
                        help="Render the roads layer")
    parser.add_argument("-b", dest="do_beaches", action="store_true",
                        help="Render the beaches layer")
    parser.add_argument("-sl", dest="do_split_layers", action="store_true",
                        help="Render the SPLIT_LAYERS bakes")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print every bake log line instead of just "
                             "progress bars and warnings")
    args = parser.parse_args()

    blends = _list_blends()
    if not blends:
        print(f"ERROR: no .blend files found in {SPILL_DIR}")
        return 1

    interactive_bakes = False
    if args.all:
        targets = blends
    elif args.region_name:
        match = next(
            (p for p in blends if p.stem.lower() == args.region_name.lower()),
            None,
        )
        if match is None:
            print(f"ERROR: '{args.region_name}' not in {SPILL_DIR}")
            return 1
        targets = [match]
    else:
        picked = pick_region_interactive(blends)
        if picked is None:
            return 1
        targets = picked
        interactive_bakes = True

    any_flag = (args.do_ao or args.do_hm or args.do_id
                or args.do_roads or args.do_beaches
                or args.do_split_layers or args.do_svg)
    if any_flag:
        do_ao, do_hm, do_id = args.do_ao, args.do_hm, args.do_id
        do_roads, do_beaches = args.do_roads, args.do_beaches
        do_split_layers = args.do_split_layers
        do_svg = args.do_svg
    elif interactive_bakes:
        bakes = ask_bakes()
        if bakes is None:
            return 1
        (do_svg, do_ao, do_hm, do_id,
         do_roads, do_beaches, do_split_layers) = bakes
    else:
        do_ao = do_hm = do_id = do_roads = do_beaches = True
        do_split_layers = True
        do_svg = True

    if not (do_ao or do_hm or do_id or do_roads or do_beaches
            or do_split_layers or do_svg):
        print("ERROR: every bake was disabled; nothing to do")
        return 1

    if not MASK_FILE.is_file():
        print(f"ERROR: mask not found at {MASK_FILE}")
        return 1
    raw = cv2.imread(str(MASK_FILE), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        print(f"ERROR: cv2 failed to read {MASK_FILE}")
        return 1
    mask = raw > 127

    # An SVG-only run never loads a .blend, so it is not bound by the
    # per-worker memory ceiling that sizes NUM_WORKERS_SPILLS.
    needs_blend = (do_ao or do_hm or do_id or do_roads or do_beaches
                   or do_split_layers)
    n_workers = NUM_WORKERS_SPILLS if needs_blend else NUM_WORKERS_SVG

    parallel = len(targets) > 1 and n_workers > 1
    print(f"=== Rendering {len(targets)} region(s) "
          f"(svg={do_svg}, ao={do_ao}, hm={do_hm}, id={do_id}, "
          f"roads={do_roads}, beaches={do_beaches}, "
          f"split_layers={do_split_layers}, "
          f"workers={n_workers if parallel else 1}) ===")

    flags: Flags = (do_svg, do_ao, do_hm, do_id,
                    do_roads, do_beaches, do_split_layers)
    tracker = progress.FileTracker(
        candidates=lambda blend: _output_candidates(blend.stem, flags),
        expected=lambda _blend: _expected_outputs(flags),
        # "[bake 3/11] AO" -> the bar's status column.
        status_re=r"\[bake\s+\d+/\d+\]\s+(.+)",
    )

    if parallel:
        def _cmd(blend: Path) -> List[str]:
            argv = [sys.executable, str(Path(__file__).resolve()), blend.stem]
            if do_svg:
                argv.append("-svg")
            if do_ao:
                argv.append("-ao")
            if do_hm:
                argv.append("-hm")
            if do_id:
                argv.append("-id")
            if do_roads:
                argv.append("-r")
            if do_beaches:
                argv.append("-b")
            if do_split_layers:
                argv.append("-sl")
            return argv

        # Split cores across workers rather than os.cpu_count() each.
        import os as _os
        cores = _os.cpu_count() or 4
        per_worker = max(2, cores // n_workers)
        print(f"    (row-pool threads per worker: {per_worker} "
              f"of {cores} cores)")

        failed_items = run_parallel_subprocesses(
            targets, _cmd,
            workers=n_workers,
            label_fn=lambda b: b.stem,
            env_extra={"FH_BAKE_THREADS": per_worker},
            tracker=tracker,
            title="Rendering bakes",
            unit="region",
            step_unit="file",
            verbose=args.verbose,
        )
        if failed_items:
            names = [b.stem for b in failed_items]
            print(f"\n{len(names)} region(s) had issues: {', '.join(names)}")
            return 1
        print(f"\n=== SUCCESS ===")
        return 0

    failed_targets = progress.run_serial(
        targets,
        lambda blend: render_one(blend, mask, do_ao, do_hm, do_id,
                                 do_roads, do_beaches, do_split_layers,
                                 do_svg),
        title="Rendering bakes",
        tracker=tracker,
        label_fn=lambda b: b.stem,
        unit="region",
        step_unit="file",
        verbose=args.verbose,
    )

    if failed_targets:
        failed = [b.stem for b in failed_targets]
        print(f"\n{len(failed)} region(s) had issues: {', '.join(failed)}")
        return 1

    print(f"\n=== SUCCESS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
