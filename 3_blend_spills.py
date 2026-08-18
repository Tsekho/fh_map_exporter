"""Generate a per-region .blend with neighbor spill.

Neighbor terrain/water are excluded; only focus-region terrain is included.
Output: export/blend_spill/<Region>.blend.

Usage:
    python 3_blend_spills.py [RegionName] [-a]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from utils.config import (
    CATEGORY_COLORS, CENTRES_FILE, EXPORT_DIR, JSON_DIR, NUM_WORKERS,
    CATALOGUE_FILE,
)
from utils.regions import build_region_with_spill
from utils.parallel import run_parallel_subprocesses
from utils.prompt import is_interactive, select
from utils.tui import ScriptTUI


def load_json_name_map() -> Dict[str, str]:
    if not JSON_DIR.is_dir():
        return {}
    return {p.stem.lower(): p.stem for p in JSON_DIR.glob("*.json")}


def _pick_region_fallback(
    keys: List[str],
    names: List[str],
    region_centers: Dict[str, List[float]],
    json_name_map: Dict[str, str],
) -> Optional[List[str]]:
    print("Available regions:")
    print("    0. All regions")
    for i, name in enumerate(names, 1):
        print(f"  {i:3}. {name}")

    while True:
        raw = input("\nSelect region (0 for all, number or name): ").strip()
        if raw == "0":
            return keys
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(keys):
                return [keys[idx]]
        elif raw in names:
            return [keys[names.index(raw)]]
        elif raw.lower() in region_centers and raw.lower() in json_name_map:
            return [raw.lower()]
        print("  Invalid selection, try again.")


def pick_region_interactive(
    region_centers: Dict[str, List[float]],
    json_name_map: Dict[str, str],
) -> Optional[List[str]]:
    keys = sorted(k for k in region_centers if k in json_name_map)
    if not keys:
        print(f"ERROR: no regions available (check {JSON_DIR} and {CENTRES_FILE})")
        return None

    names = [json_name_map[k] for k in keys]

    if not is_interactive():
        return _pick_region_fallback(keys, names, region_centers, json_name_map)

    idx = select("Select region", ["All regions"] + names)
    if idx is None:
        return None
    return keys if idx == 0 else [keys[idx - 1]]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a region .blend with neighbor spill.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("region_name", nargs="?",
                        help="Region name; omit for interactive selection")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Process every region in export/_json")
    args = parser.parse_args()

    if not CENTRES_FILE.is_file():
        print(f"ERROR: {CENTRES_FILE} not found")
        return 1
    if not CATALOGUE_FILE.is_file():
        print(f"ERROR: {CATALOGUE_FILE} not found")
        return 1

    with CENTRES_FILE.open("r", encoding="utf-8") as f:
        region_centers: Dict[str, List[float]] = json.load(f)
    with CATALOGUE_FILE.open("r", encoding="utf-8") as f:
        catalogue: Dict[str, List[str]] = json.load(f)

    dropped = [c for c in catalogue if c not in CATEGORY_COLORS]
    if dropped:
        print(f"[filter] skipping categories not in CATEGORY_COLORS: "
              f"{', '.join(dropped)}")
        catalogue = {
            c: meshes for c, meshes in catalogue.items()
            if c in CATEGORY_COLORS
        }

    json_name_map = load_json_name_map()
    if not json_name_map:
        print(f"ERROR: no JSON files found in {JSON_DIR}")
        return 1

    if args.all:
        region_keys = sorted(k for k in region_centers if k in json_name_map)
        if not region_keys:
            print("ERROR: no regions have both a center entry and a JSON")
            return 1
    elif args.region_name:
        key = args.region_name.lower()
        if key not in region_centers:
            print(f"ERROR: '{args.region_name}' not in {CENTRES_FILE}")
            return 1
        if key not in json_name_map:
            print(f"ERROR: '{args.region_name}' has no JSON in {JSON_DIR}")
            return 1
        region_keys = [key]
    else:
        picked = pick_region_interactive(region_centers, json_name_map)
        if picked is None:
            return 1
        region_keys = picked

    parallel = len(region_keys) > 1 and NUM_WORKERS > 1
    total = len(region_keys)

    with ScriptTUI(title=f"Building {total} region spill(s)", total=total) as tui:
        if parallel:
            def _cmd(key: str) -> List[str]:
                name = json_name_map.get(key, key)
                return [sys.executable, str(Path(__file__).resolve()), name]

            failed = run_parallel_subprocesses(
                region_keys, _cmd,
                workers=NUM_WORKERS, tui=tui,
                label_fn=lambda k: json_name_map.get(k, k),
            )
            if failed:
                names = [json_name_map.get(k, k) for k in failed]
                tui.error(f"{len(failed)} region(s) failed: {', '.join(names)}")
                return 1
            tui.log("=== SUCCESS ===")
            return 0

        errors: List[str] = []
        for key in region_keys:
            name = json_name_map.get(key, key)
            tui.set_description(name)
            tui.log(f"--- {name} ---")
            try:
                build_region_with_spill(
                    region_key=key,
                    export_dir=str(EXPORT_DIR),
                    region_centers=region_centers,
                    catalogue=catalogue,
                    json_name_map=json_name_map,
                )
            except Exception as exc:
                tui.error(f"while processing {name}: {exc}")
                errors.append(name)
            tui.advance()

        if errors:
            tui.error(f"{len(errors)} region(s) failed: {', '.join(errors)}")
            return 1

        tui.log("=== SUCCESS ===")

    return 0


if __name__ == "__main__":
    sys.exit(main())
