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
    CATALOGUE_FILE, PURGE,
)
from utils import progress, tui
from utils.regions import build_region_with_spill, find_region_neighbors
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


def warn_purged(
    region_keys: List[str],
    region_centers: Dict[str, List[float]],
    json_name_map: Dict[str, str],
) -> None:
    """List the config.PURGE entries these builds left out -- both the
    regions built and the neighbours that spill into them."""
    touched = set(region_keys)
    for key in region_keys:
        touched.update(find_region_neighbors(region_centers, key))
    names = {json_name_map.get(k, k) for k in touched}
    hits = [(name, section, key)
            for name, sections in sorted(PURGE.items()) if name in names
            for section, targets in sections.items() for key, _ in targets]
    if not hits:
        return
    tui.heading(f"Purged {len(hits)} instance(s)", "PURGE in utils/config.py")
    mark = tui.yellow(tui.glyph("▲", "!"))
    width = max(len(key) for _, _, key in hits)
    for name, section, key in hits:
        print(f"  {mark} {key:<{width}}  {tui.dim(f'{name} {section}')}")


def pick_region_interactive(
    region_centers: Dict[str, List[float]],
    json_name_map: Dict[str, str],
) -> Optional[List[str]]:
    keys = sorted(k for k in region_centers if k in json_name_map)
    if not keys:
        tui.error(f"no regions available (check {JSON_DIR} and {CENTRES_FILE})")
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
    # Set on the per-region subprocesses of a parallel run, which leave the
    # purge summary to the parent so it prints once.
    parser.add_argument("--worker", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not CENTRES_FILE.is_file():
        tui.error(f"{CENTRES_FILE} not found")
        return 1
    if not CATALOGUE_FILE.is_file():
        tui.error(f"{CATALOGUE_FILE} not found")
        return 1

    with CENTRES_FILE.open("r", encoding="utf-8") as f:
        region_centers: Dict[str, List[float]] = json.load(f)
    with CATALOGUE_FILE.open("r", encoding="utf-8") as f:
        catalogue: Dict[str, List[str]] = json.load(f)

    dropped = [c for c in catalogue if c not in CATEGORY_COLORS]
    if dropped:
        # Expected: catalogue categories without a colour (foliage_*,
        # ignore_spline_meshes, ...) feed svg layers only, not spills.
        print(f"[filter] skipping categories not in CATEGORY_COLORS: "
              f"{', '.join(dropped)}")
        catalogue = {
            c: meshes for c, meshes in catalogue.items()
            if c in CATEGORY_COLORS
        }

    json_name_map = load_json_name_map()
    if not json_name_map:
        tui.error(f"no JSON files found in {JSON_DIR}")
        return 1

    if args.all:
        region_keys = sorted(k for k in region_centers if k in json_name_map)
        if not region_keys:
            tui.error("no regions have both a center entry and a JSON")
            return 1
    elif args.region_name:
        key = args.region_name.lower()
        if key not in region_centers:
            tui.error(f"'{args.region_name}' not in {CENTRES_FILE}")
            return 1
        if key not in json_name_map:
            tui.error(f"'{args.region_name}' has no JSON in {JSON_DIR}")
            return 1
        region_keys = [key]
    else:
        picked = pick_region_interactive(region_centers, json_name_map)
        if picked is None:
            return 1
        region_keys = picked

    parallel = len(region_keys) > 1 and NUM_WORKERS > 1
    tui.heading(f"Building {len(region_keys)} region spill(s)",
                f"workers={NUM_WORKERS if parallel else 1}")

    tracker = progress.PhaseTracker(SPILL_PHASES)

    if parallel:
        def _cmd(key: str) -> List[str]:
            name = json_name_map.get(key, key)
            return [sys.executable, str(Path(__file__).resolve()), name,
                    "--worker"]

        failed = run_parallel_subprocesses(
            region_keys, _cmd,
            workers=NUM_WORKERS,
            label_fn=lambda k: json_name_map.get(k, k),
            tracker=tracker,
            title="Building spills",
            unit="region",
            verbose=args.verbose,
        )
        warn_purged(region_keys, region_centers, json_name_map)
        if failed:
            names = [json_name_map.get(k, k) for k in failed]
            tui.error(f"{len(failed)} region(s) failed: {', '.join(names)}")
            return 1
        tui.done(f"{len(region_keys)} spill(s) built")
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

    if not args.worker:
        warn_purged(region_keys, region_centers, json_name_map)
    if failed_keys:
        errors = [json_name_map.get(k, k) for k in failed_keys]
        tui.error(f"{len(errors)} region(s) failed: {', '.join(errors)}")
        return 1

    tui.done(f"{len(region_keys)} spill(s) built")
    return 0


if __name__ == "__main__":
    tui.run(main)
