"""
2_blend_all.py
========
Generate a .blend file from Foxhole map exports produced by Exporter.exe.

Reads per-map JSON from export/_json and meshes from export/_meshes, then
writes a Blender scene to export/blend/<MapName>.blend.

Usage:
    python 2_blend_all.py [MapName] [-nt] [-a]

Examples:
    python 2_blend_all.py OarbreakerHex
    python 2_blend_all.py OarbreakerHex -nt
    python 2_blend_all.py -a -nt
    python 2_blend_all.py                   # interactive map/terrain selection

Options:
    MapName            Map name (e.g. OarbreakerHex); omit for interactive prompt
    -nt, --no-terrain  Exclude heightmap terrain (terrain included by default)
    -a, --all          Process every map found in export/_json
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from utils.config import EXPORT_DIR, JSON_DIR, NUM_WORKERS
from utils.helpers import Map
from utils.parallel import run_parallel_subprocesses


# ------------------------------------------------------------------------------
#  Interactive helpers
# ------------------------------------------------------------------------------


def pick_map_interactive() -> Optional[List[str]]:
    """Prompt the user to select a map from those found in export/_json."""
    if not JSON_DIR.is_dir():
        print(f"ERROR: {JSON_DIR} not found")
        return None

    maps = sorted(p.stem for p in JSON_DIR.glob("*.json"))
    if not maps:
        print(f"ERROR: no JSON files found in {JSON_DIR}")
        return None

    print("Available maps:")
    print("    0. All maps")
    for i, name in enumerate(maps, 1):
        print(f"  {i:3}. {name}")

    while True:
        raw = input("\nSelect map (0 for all, number or name): ").strip()
        if raw == "0":
            return maps
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(maps):
                return [maps[idx]]
        elif raw in maps:
            return [raw]
        print("  Invalid selection, try again.")


def ask_terrain() -> bool:
    """Prompt the user whether to include heightmap terrain."""
    while True:
        raw = input("Include terrain? [Y/n]: ").strip().lower()
        if raw in ("", "y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please enter y or n.")


# ------------------------------------------------------------------------------
#  CLI entry point
# ------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a .blend from Foxhole map exports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "map_name",
        nargs="?",
        help="Map name (e.g. OarbreakerHex); omit for interactive selection",
    )
    parser.add_argument(
        "-nt", "--no-terrain",
        action="store_true",
        help="Exclude heightmap terrain (terrain included by default)",
    )
    parser.add_argument(
        "-a", "--all",
        action="store_true",
        help="Process every map found in export/_json",
    )
    args = parser.parse_args()

    # Resolve map list and terrain flag --------------------------------------
    if args.all:
        if not JSON_DIR.is_dir():
            print(f"ERROR: {JSON_DIR} not found")
            return 1
        map_names = sorted(p.stem for p in JSON_DIR.glob("*.json"))
        if not map_names:
            print(f"ERROR: no maps found in {JSON_DIR}")
            return 1
        terrain = not args.no_terrain
    elif args.map_name:
        map_names = [args.map_name]
        terrain = not args.no_terrain
    else:
        picked = pick_map_interactive()
        if picked is None:
            return 1
        map_names = picked
        terrain = ask_terrain()

    parallel = len(map_names) > 1 and NUM_WORKERS > 1
    print(f"=== Building {len(map_names)} map(s) "
          f"(terrain={terrain}, workers={NUM_WORKERS if parallel else 1}) ===")

    # Parallel fan-out when running more than one map --------------------------
    if parallel:
        def _cmd(name: str) -> List[str]:
            argv = [sys.executable, str(Path(__file__).resolve()), name]
            if not terrain:
                argv.append("-nt")
            return argv

        failed = run_parallel_subprocesses(
            map_names, _cmd, workers=NUM_WORKERS,
        )
        if failed:
            print(f"\n{len(failed)} map(s) failed: {', '.join(failed)}")
            return 1
        print(f"\n=== SUCCESS ===")
        return 0

    # Serial path -------------------------------------------------------------
    errors: List[str] = []
    for name in map_names:
        json_path = JSON_DIR / f"{name}.json"
        if not json_path.exists():
            print(f"ERROR: JSON not found: {json_path}")
            errors.append(name)
            continue
        print(f"\n=== {name} ===")
        try:
            Map(str(json_path), str(EXPORT_DIR)).blend(terrain=terrain)
        except Exception as exc:
            print(f"ERROR while processing {name}: {exc}")
            errors.append(name)

    if errors:
        print(f"\n{len(errors)} map(s) failed: {', '.join(errors)}")
        return 1

    print(f"\n=== SUCCESS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
