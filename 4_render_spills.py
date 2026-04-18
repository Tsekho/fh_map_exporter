"""
4_render_spills.py
========
Render top-down 2048x2048 bakes for every region .blend produced by
3_blend_spills.py.

For each export/blend_spill/<Region>.blend the script opens the file and,
for each bake group enabled on the command line (all enabled by default),
writes PNGs:

    -ao   ->  export/ao/<Region>.png                   AO (Cycles, hex-masked)
    -hm   ->  export/heightmap_landscape/<R>.png       16-bit heightmap (raycast)
    -id   ->  export/id/<Region>.png                   category ID map
              export/water/<Region>.png                binary water mask

Heightmap rules:
    Rays see terrain (focus region only) + rocks + glaciers + landscape_meshes
    from both focus and neighbor spill regions. Water and deep water are not
    part of the BVH, so rays passing over water miss and the pixel stays at 0.

ID map rules:
    Rays see terrain (focus region only) + rocks + glaciers + landscape_meshes
    (focus + neighbor spill) + deep_water (colored with the "water" color).
    Surface water is excluded so it never occludes meshes below it.

Usage:
    python 4_render_spills.py [RegionName] [-a] [-ao] [-hm] [-id]
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
    CATEGORY_COLORS,
    CATEGORY_NAMES,
    HM_LANDSCAPE_DIR,
    ID_DIR,
    NUM_WORKERS,
    MASK_FILE,
    SPILL_DIR,
    WATER_DIR,
)
from utils.helpers import (
    raycast_heightmap,
    raycast_id_map,
    raycast_binary_mask,
    render_ao,
)
from utils.parallel import run_parallel_subprocesses


# Categories spilled in from neighbor regions (neighbor terrain/water are
# excluded by 3_blend_spills.py).
_SPILL_CATEGORIES = ("rocks", "glaciers", "landscape_meshes")

# Blender auto-renames duplicate collection names to "<name>.001"; strip
# that when matching neighbor-region spill buckets to focus buckets.
_DATA_SUFFIX_RE = re.compile(r"\.\d{3}$")


def _base_name(name: str) -> str:
    return _DATA_SUFFIX_RE.sub("", name)


def _collect_focus_objects(
    region_name: str,
) -> Optional[Dict[str, List[bpy.types.Object]]]:
    """
    Return ``{category: [objects, ...]}`` for the focus region, folding in
    neighbor-region spill objects for ``_SPILL_CATEGORIES``. Returns None
    when the region root collection is missing.
    """
    root = bpy.data.collections.get(region_name)
    if root is None:
        return None

    buckets: Dict[str, List[bpy.types.Object]] = {c: [] for c in CATEGORY_NAMES}
    for child in root.children:
        if child.name in buckets:
            buckets[child.name] = list(child.all_objects)

    spill_added = {c: 0 for c in _SPILL_CATEGORIES}
    scene = bpy.context.scene
    for top in scene.collection.children:
        if top.name == region_name:
            continue
        for cat_coll in top.children:
            base = _base_name(cat_coll.name)
            if base not in _SPILL_CATEGORIES:
                continue
            extra = list(cat_coll.all_objects)
            buckets[base].extend(extra)
            spill_added[base] += len(extra)
    if any(spill_added.values()):
        summary = ", ".join(
            f"{c}+{n}" for c, n in spill_added.items() if n
        )
        print(f"  Neighbor spill folded in: {summary}")
    return buckets


# ------------------------------------------------------------------------------
#  Per-region render
# ------------------------------------------------------------------------------


def render_one(
    blend_path: Path,
    mask: np.ndarray,
    do_ao: bool,
    do_hm: bool,
    do_id: bool,
) -> bool:
    """Render the enabled bakes for *blend_path*. Returns True on success."""
    region_name = blend_path.stem
    print(f"\n=== {region_name} ===")
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))

    objs = _collect_focus_objects(region_name)
    if objs is None:
        print(f"  [WARN] root collection '{region_name}' not found; skipped")
        return False

    ok = True

    def _out(sub: Path) -> str:
        return str((sub / f"{region_name}.png").resolve())

    if do_hm:
        try:
            print("[bake] heightmap ...")
            hm_objs = (
                objs["terrain"]
                + objs["rocks"]
                + objs["glaciers"]
                + objs["landscape_meshes"]
            )
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
            print("[bake] ID map ...")
            id_cats: Dict[str, List[bpy.types.Object]] = {
                "terrain":          objs["terrain"],
                "rocks":            objs["rocks"],
                "glaciers":         objs["glaciers"],
                "landscape_meshes": objs["landscape_meshes"],
            }
            raycast_id_map(
                _out(ID_DIR), mask, id_cats, CATEGORY_COLORS,
                occluders=objs["deep_water"],
            )
            print("[bake] water mask ...")
            water_occluders = (
                objs["terrain"]
                + objs["rocks"]
                + objs["glaciers"]
                + objs["landscape_meshes"]
            )
            raycast_binary_mask(
                _out(WATER_DIR), mask, water_occluders, objs["water"],
            )
        except Exception as exc:
            print(f"  [WARN] ID/water bake failed: {exc}")
            ok = False

    if do_ao:
        try:
            print("[bake] AO ...")
            # Surface water fully hidden; deep water visible to camera
            # (bakes as white) but invisible to secondary rays so it
            # doesn't blacken the terrain below.
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
            try:
                render_ao(
                    _out(AO_DIR), mask,
                    hidden_objs=objs["water"],
                )
            finally:
                for o, saved in prev_vis:
                    for k, v in saved.items():
                        if hasattr(o, f"visible_{k}"):
                            setattr(o, f"visible_{k}", v)
        except Exception as exc:
            print(f"  [WARN] AO bake failed: {exc}")
            ok = False

    return ok


# ------------------------------------------------------------------------------
#  Region picking
# ------------------------------------------------------------------------------


def _list_blends() -> List[Path]:
    if not SPILL_DIR.is_dir():
        return []
    return sorted(SPILL_DIR.glob("*.blend"))


def pick_region_interactive(blends: List[Path]) -> Optional[List[Path]]:
    names = [p.stem for p in blends]
    print("Available .blend spills:")
    print("    0. All regions")
    for i, name in enumerate(names, 1):
        print(f"  {i:3}. {name}")
    while True:
        raw = input("\nSelect region (0 for all, number or name): ").strip()
        if raw == "0":
            return blends
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(blends):
                return [blends[idx]]
        elif raw in names:
            return [blends[names.index(raw)]]
        print("  Invalid selection, try again.")


def ask_bakes() -> Tuple[bool, bool, bool]:
    """Prompt for bake selection. Empty = all; else any of {ao, hm, id}."""
    while True:
        raw = input(
            "Select bakes (space-separated from: ao hm id; empty = all): "
        ).strip().lower()
        if raw == "":
            return True, True, True
        tokens = raw.replace(",", " ").split()
        valid = {"ao", "hm", "id"}
        if any(t not in valid for t in tokens):
            print("  Invalid token, try again.")
            continue
        return ("ao" in tokens, "hm" in tokens, "id" in tokens)


# ------------------------------------------------------------------------------
#  CLI
# ------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render bakes for every region .blend in export/blend_spill. "
            "If none of -ao / -hm / -id is given, all bakes are produced."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "region_name", nargs="?",
        help="Region stem (e.g. OarbreakerHex); omit for interactive prompt",
    )
    parser.add_argument(
        "-a", "--all", action="store_true",
        help="Render every .blend in export/blend_spill",
    )
    parser.add_argument("-ao", dest="do_ao", action="store_true",
                        help="Render the AO bake")
    parser.add_argument("-hm", dest="do_hm", action="store_true",
                        help="Render the heightmap")
    parser.add_argument("-id", dest="do_id", action="store_true",
                        help="Render the ID map and the binary water mask")
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

    if args.do_ao or args.do_hm or args.do_id:
        do_ao, do_hm, do_id = args.do_ao, args.do_hm, args.do_id
    elif interactive_bakes:
        do_ao, do_hm, do_id = ask_bakes()
    else:
        do_ao = do_hm = do_id = True

    if not (do_ao or do_hm or do_id):
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

    parallel = len(targets) > 1 and NUM_WORKERS > 1
    print(f"=== Rendering {len(targets)} region(s) "
          f"(ao={do_ao}, hm={do_hm}, id={do_id}, "
          f"workers={NUM_WORKERS if parallel else 1}) ===")

    if parallel:
        def _cmd(blend: Path) -> List[str]:
            argv = [sys.executable, str(Path(__file__).resolve()), blend.stem]
            if do_ao:
                argv.append("-ao")
            if do_hm:
                argv.append("-hm")
            if do_id:
                argv.append("-id")
            return argv

        failed_items = run_parallel_subprocesses(
            targets, _cmd,
            workers=NUM_WORKERS,
            label_fn=lambda b: b.stem,
        )
        if failed_items:
            names = [b.stem for b in failed_items]
            print(f"\n{len(names)} region(s) had issues: {', '.join(names)}")
            return 1
        print(f"\n=== SUCCESS ===")
        return 0

    failed: List[str] = []
    for blend in targets:
        try:
            if not render_one(blend, mask, do_ao, do_hm, do_id):
                failed.append(blend.stem)
        except Exception as exc:
            print(f"ERROR while rendering {blend.stem}: {exc}")
            failed.append(blend.stem)

    if failed:
        print(f"\n{len(failed)} region(s) had issues: {', '.join(failed)}")
        return 1

    print(f"\n=== SUCCESS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
