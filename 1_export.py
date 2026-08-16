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
from utils.tui import ScriptTUI


def main() -> int:
    if not EXPORTER_EXE.exists():
        print(f"ERROR: Exporter.exe not found: {EXPORTER_EXE}")
        return 1

    with ScriptTUI(title="Exporting from .pak", total=1) as tui:
        with tui.suspend():
            result = subprocess.run(
                [str(EXPORTER_EXE), "-i", str(FOXHOLE_PAK), "-o", str(EXPORT_DIR), "-t"]
            )
        if result.returncode != 0:
            tui.error(f"ERROR: Exporter.exe failed (exit code {result.returncode})")
            return result.returncode
        tui.advance()

        json_path = JSON_DIR / "HomeRegionW.json"
        if not json_path.exists():
            tui.error(f"ERROR: JSON not found: {json_path}")
            return 1

        data = json.loads(json_path.read_text(encoding="utf-8"))
        n_pskx = len(list(MESHES_DIR.rglob("*.pskx"))) if MESHES_DIR.exists() else 0
        n_psk  = len(list(MESHES_DIR.rglob("*.psk")))  if MESHES_DIR.exists() else 0

        tui.log("")
        tui.log("=== RESULTS ===")
        tui.log(f"  symbols    : {len(data.get('symbols',    [])):>6} mesh types")
        tui.log(f"  groups     : {len(data.get('groups',     [])):>6} mesh types")
        tui.log(f"  blueprints : {len(data.get('blueprints', [])):>6} class types")
        tui.log(f"  meshes     : {n_pskx:>6} .pskx files in {MESHES_DIR}")
        tui.log(f"  meshes     : {n_psk:>6} .psk  files in {MESHES_DIR}")
        tui.log(f"  JSON       : {json_path}  ({json_path.stat().st_size / 1024:.0f} KB)")

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

            tui.log("")
            tui.log("=== CATALOGUE DIFF ===")
            tui.log(f"  catalogue entries : {len(catalogue_entries)}")
            tui.log(f"  exported meshes   : {len(exported)}")
            tui.log(f"  in catalogue, not exported: {len(missing)}")
            for name in missing:
                tui.log(f"    - {name}")
            tui.log(f"  exported, not in catalogue: {len(disappeared)}")
            for name in disappeared:
                tui.log(f"    + {name}")
        else:
            if not CATALOGUE_FILE.exists():
                tui.warn(f"WARN: catalogue not found: {CATALOGUE_FILE}")
            if not MESHES_DIR.exists():
                tui.warn(f"WARN: meshes dir not found: {MESHES_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
