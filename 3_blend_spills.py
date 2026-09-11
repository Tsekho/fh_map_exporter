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
from utils import progress, tui
from utils.regions import build_region_with_spill
from utils.parallel import run_parallel_subprocesses


def load_json_name_map() -> Dict[str, str]:
    if not JSON_DIR.is_dir():
        return {}
    return {p.stem.lower(): p.stem for p in JSON_DIR.glob("*.json")}


# Checkpoints build_region_with_spill() prints, in order. One line, one
# checkpoint; the neighbor line comes once per hexagonal neighbor.
SPILL_PHASES = [
    (r"^\[terrain\]", "focus terrain"),
    (r"^\[meshes\]", "focus objects"),
    (r"^\[neighbor\]", "neighbor spill", 6),
    # Not map.py's parse-time "[splines] N unique meshes": every neighbor
    # Map() prints one, and it would skip the bar to the end.
    (r"^\[splines\] placed", "splines"),
    (r"^Saving ->", "saving"),
    (r"^Done\.", "saved"),
]


def pick_region_interactive(
    region_centers: Dict[str, List[float]],
    json_name_map: Dict[str, str],
) -> Optional[List[str]]:
    keys = sorted(k for k in region_centers if k in json_name_map)
    if not keys:
        print(f"ERROR: no regions available (check {JSON_DIR} and {CENTRES_FILE})")
        return None

    return tui.select_many(
        keys, "Regions to build",
        label_fn=lambda k: json_name_map[k],
        noun="region",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a region .blend with neighbor spill.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("region_name", nargs="?",
                        help="Region name; omit for interactive selection")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Process every region in export/_json")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print every build log line instead of just "
                             "progress bars and warnings")
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
    print(f"=== Building {len(region_keys)} region spill(s) "
          f"(workers={NUM_WORKERS if parallel else 1}) ===")

    tracker = progress.PhaseTracker(SPILL_PHASES)

    if parallel:
        def _cmd(key: str) -> List[str]:
            name = json_name_map.get(key, key)
            return [sys.executable, str(Path(__file__).resolve()), name]

        failed = run_parallel_subprocesses(
            region_keys, _cmd,
            workers=NUM_WORKERS,
            label_fn=lambda k: json_name_map.get(k, k),
            tracker=tracker,
            title="Building spills",
            unit="region",
            verbose=args.verbose,
        )
        if failed:
            names = [json_name_map.get(k, k) for k in failed]
            print(f"\n{len(failed)} region(s) failed: {', '.join(names)}")
            return 1
        print(f"\n=== SUCCESS ===")
        return 0

    def _build(key: str) -> bool:
        build_region_with_spill(
            region_key=key,
            export_dir=str(EXPORT_DIR),
            region_centers=region_centers,
            catalogue=catalogue,
            json_name_map=json_name_map,
        )
        return True

    failed_keys = progress.run_serial(
        region_keys, _build,
        title="Building spills",
        tracker=tracker,
        label_fn=lambda k: json_name_map.get(k, k),
        unit="region",
        verbose=args.verbose,
    )

    if failed_keys:
        errors = [json_name_map.get(k, k) for k in failed_keys]
        print(f"\n{len(errors)} region(s) failed: {', '.join(errors)}")
        return 1

    print(f"\n=== SUCCESS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
