"""
1_export.py
========
Export Foxhole game data from the .pak file using Exporter.exe.

Clears the export/ directory, runs Exporter.exe against the game .pak, then
prints a summary of extracted symbols, groups, blueprints, meshes, and JSON.

Usage:
    python 1_export.py
"""

import subprocess
import shutil
import json
import sys

from utils.config import EXPORT_DIR, EXPORTER_EXE, FOXHOLE_PAK, JSON_DIR, MESHES_DIR


def main() -> int:
    if not EXPORTER_EXE.exists():
        print(f"ERROR: Exporter.exe not found: {EXPORTER_EXE}")
        return 1

    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)

    result = subprocess.run(
        [str(EXPORTER_EXE), "-i", str(FOXHOLE_PAK), "-o", str(EXPORT_DIR), "-t"]
    )
    if result.returncode != 0:
        print(f"\nERROR: Exporter.exe failed (exit code {result.returncode})")
        return result.returncode

    json_path = JSON_DIR / "HomeRegionW.json"
    if not json_path.exists():
        print(f"ERROR: JSON not found: {json_path}")
        return 1

    data = json.loads(json_path.read_text(encoding="utf-8"))
    n_pskx = len(list(MESHES_DIR.rglob("*.pskx"))) if MESHES_DIR.exists() else 0
    n_psk  = len(list(MESHES_DIR.rglob("*.psk")))  if MESHES_DIR.exists() else 0

    print()
    print("=== RESULTS ===")
    print(f"  symbols    : {len(data.get('symbols',    [])):>6} mesh types")
    print(f"  groups     : {len(data.get('groups',     [])):>6} mesh types")
    print(f"  blueprints : {len(data.get('blueprints', [])):>6} class types")
    print(f"  meshes     : {n_pskx:>6} .pskx files in {MESHES_DIR}")
    print(f"  meshes     : {n_psk:>6} .psk  files in {MESHES_DIR}")
    print(f"  JSON       : {json_path}  ({json_path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
