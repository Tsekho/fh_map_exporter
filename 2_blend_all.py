"""Generate .blend files from Foxhole map exports.

Usage:
    python 2_blend_all.py [MapName] [-nt] [-a]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Set

from utils.config import CENTRES_FILE, EXPORT_DIR, JSON_DIR, NUM_WORKERS
from utils.map import Map
from utils.parallel import run_parallel_subprocesses
from utils.tui import ScriptTUI


def _load_region_keys() -> Set[str]:
    """Set of lowercase region keys from region_centers.json. Used to
    filter out non-region JSONs (e.g. MainMenu) that have no centre."""
    if not CENTRES_FILE.is_file():
        return set()
    with CENTRES_FILE.open("r", encoding="utf-8") as f:
        return {k.lower() for k in json.load(f).keys()}


def _list_maps() -> List[str]:
    """JSON stems that correspond to real regions (present in
    region_centers.json)."""
    if not JSON_DIR.is_dir():
        return []
    keys = _load_region_keys()
    if not keys:
        return sorted(p.stem for p in JSON_DIR.glob("*.json"))
    return sorted(
        p.stem for p in JSON_DIR.glob("*.json") if p.stem.lower() in keys
    )


def pick_map_interactive() -> Optional[List[str]]:
    if not JSON_DIR.is_dir():
        print(f"ERROR: {JSON_DIR} not found")
        return None

    maps = _list_maps()
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a .blend from Foxhole map exports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("map_name", nargs="?",
                        help="Map name (e.g. OarbreakerHex); omit for interactive")
    parser.add_argument("-nt", "--no-terrain", action="store_true",
                        help="Exclude heightmap terrain")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Process every map in export/_json")
    args = parser.parse_args()

    if args.all:
        if not JSON_DIR.is_dir():
            print(f"ERROR: {JSON_DIR} not found")
            return 1
        map_names = _list_maps()
        if not map_names:
            print(f"ERROR: no maps found in {JSON_DIR}")
            return 1
        terrain = not args.no_terrain
    elif args.map_name:
        keys = _load_region_keys()
        if keys and args.map_name.lower() not in keys:
            print(f"ERROR: '{args.map_name}' is not a region "
                  f"(missing from {CENTRES_FILE.name}); skipping")
            return 1
        map_names = [args.map_name]
        terrain = not args.no_terrain
    else:
        picked = pick_map_interactive()
        if picked is None:
            return 1
        map_names = picked
        terrain = not args.no_terrain

    parallel = len(map_names) > 1 and NUM_WORKERS > 1
    total = len(map_names)
    title = f"Building {total} map(s) (terrain={terrain})"

    with ScriptTUI(title=title, total=total) as tui:
        if parallel:
            def _cmd(name: str) -> List[str]:
                argv = [sys.executable, str(Path(__file__).resolve()), name]
                if not terrain:
                    argv.append("-nt")
                return argv

            failed = run_parallel_subprocesses(
                map_names, _cmd, workers=NUM_WORKERS, tui=tui,
            )
            if failed:
                tui.error(f"{len(failed)} map(s) failed: {', '.join(failed)}")
                return 1
            tui.log("=== SUCCESS ===")
            return 0

        errors: List[str] = []
        for name in map_names:
            json_path = JSON_DIR / f"{name}.json"
            if not json_path.exists():
                tui.error(f"ERROR: JSON not found: {json_path}")
                errors.append(name)
                tui.advance()
                continue
            tui.set_description(name)
            tui.log(f"--- {name} ---")
            try:
                Map(str(json_path), str(EXPORT_DIR)).blend(terrain=terrain)
            except Exception as exc:
                tui.error(f"ERROR while processing {name}: {exc}")
                errors.append(name)
            tui.advance()

        if errors:
            tui.error(f"{len(errors)} map(s) failed: {', '.join(errors)}")
            return 1

        tui.log("=== SUCCESS ===")

    return 0


if __name__ == "__main__":
    sys.exit(main())
