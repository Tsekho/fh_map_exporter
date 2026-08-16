"""Run Exporter.exe against the game .pak and print a summary."""

import subprocess
import json
import sys

from utils.config import (
    CATALOGUE_FILE,
    EXPORT_DIR,
    EXPORTER_EXE,
    FOXHOLE_PAK,
    JSON_DIR,
    MESHES_DIR,
)


def main() -> int:
    if not EXPORTER_EXE.exists():
        print(f"ERROR: Exporter.exe not found: {EXPORTER_EXE}")
        return 1

    if FOXHOLE_PAK is None:
        print(
            "ERROR: could not locate War-WindowsNoEditor.pak.\n"
            "  Searched every Steam library folder; Foxhole doesn't seem "
            "to be installed in any of them.\n"
            "  If it's installed somewhere unusual, set the "
            "FOXHOLE_PAK_PATH environment variable to the full path of "
            "War-WindowsNoEditor.pak and try again."
        )
        return 1

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

    # Compare catalogue.json vs exported meshes
    if CATALOGUE_FILE.exists() and MESHES_DIR.exists():
        catalogue = json.loads(CATALOGUE_FILE.read_text(encoding="utf-8"))
        catalogue_entries = set()
        for entries in catalogue.values():
            catalogue_entries.update(entries)

        exported = {
            p.stem
            for p in MESHES_DIR.rglob("*")
            if p.suffix.lower() in (".pskx", ".psk")
        }

        missing = sorted(catalogue_entries - exported)
        disappeared = sorted(exported - catalogue_entries)

        print()
        print("=== CATALOGUE DIFF ===")
        print(f"  catalogue entries : {len(catalogue_entries)}")
        print(f"  exported meshes   : {len(exported)}")
        print(f"  in catalogue, not exported: {len(missing)}")
        for name in missing:
            print(f"    - {name}")
        print(f"  exported, not in catalogue: {len(disappeared)}")
        for name in disappeared:
            print(f"    + {name}")
    else:
        if not CATALOGUE_FILE.exists():
            print(f"WARN: catalogue not found: {CATALOGUE_FILE}")
        if not MESHES_DIR.exists():
            print(f"WARN: meshes dir not found: {MESHES_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
