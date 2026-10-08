"""Run Exporter.exe against the game .pak and print a summary.

Afterwards every region JSON is hashed into export/hashes.txt; when a
previous hashes.txt exists, the regions whose JSON changed, appeared or
disappeared since that export are listed.
"""

import hashlib
import subprocess
import json
from pathlib import Path
from typing import Dict

from utils import tui
from utils.config import (
    CATALOGUE_FILE,
    EXPORT_DIR,
    EXPORTER_EXE,
    FOXHOLE_PAK,
    HASHES_FILE,
    JSON_DIR,
    JSON_HASH_LEN,
    MESHES_DIR,
)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:JSON_HASH_LEN]


def _read_hashes(path: Path) -> Dict[str, str]:
    """{filename: hash} from a hashes.txt; {} when there is none."""
    if not path.is_file():
        return {}
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and not line.startswith("#"):
            out[parts[0]] = parts[1]
    return out


def _write_hashes(path: Path, hashes: Dict[str, str]) -> None:
    """One "filename  hash" line per JSON, names padded to align and sorted
    for stable diffs."""
    width = max(len(name) for name in hashes)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Region json hashes to track single region changes\n")
        for name in sorted(hashes):
            f.write(f"{name:<{width}}  {hashes[name]}\n")


def update_hashes() -> None:
    names = sorted(p for p in JSON_DIR.glob("*.json") if p.is_file())
    if not names:
        tui.warn(f"no JSONs to hash in {JSON_DIR}")
        return
    old = _read_hashes(HASHES_FILE)
    new = {p.name: _file_hash(p) for p in names}
    _write_hashes(HASHES_FILE, new)

    tui.heading("JSON hashes", f"{len(new)} file(s) -> {HASHES_FILE.name}")
    if not old:
        print(tui.dim("  no previous hashes.txt; nothing to compare against"))
        return

    changed = sorted(n for n in new if n in old and old[n] != new[n])
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    if not (changed or added or removed):
        print(tui.dim("  no region JSON changed since the previous export"))
        return
    for label, names_, mark in (("changed", changed, tui.yellow("~")),
                                ("new", added, tui.green("+")),
                                ("gone", removed, tui.red("-"))):
        if names_:
            print(f"  {label}: {len(names_)}")
            for name in names_:
                print(f"    {mark} {name}")


def main() -> int:
    if not EXPORTER_EXE.exists():
        tui.error(f"Exporter.exe not found: {EXPORTER_EXE}")
        return 1

    if FOXHOLE_PAK is None:
        tui.error("could not locate War-WindowsNoEditor.pak.")
        print(tui.dim(
            "  Searched every Steam library folder; Foxhole doesn't seem "
            "to be installed in any of them.\n"
            "  If it's installed somewhere unusual, set the "
            "FOXHOLE_PAK_PATH environment variable to the full path of "
            "War-WindowsNoEditor.pak and try again."
        ))
        return 1

    result = subprocess.run(
        [str(EXPORTER_EXE), "-i", str(FOXHOLE_PAK), "-o", str(EXPORT_DIR), "-t"]
    )
    if result.returncode != 0:
        tui.error(f"Exporter.exe failed (exit code {result.returncode})")
        return result.returncode

    json_paths = sorted(JSON_DIR.glob("*.json")) if JSON_DIR.is_dir() else []
    if not json_paths:
        tui.error(f"no JSONs written to {JSON_DIR}")
        return 1

    # Distinct keys per section across every map, not per-map sums: the
    # same mesh or blueprint class turns up in most regions.
    sections = ("symbols", "groups", "blueprints", "splines")
    types = {s: set() for s in sections}
    for path in json_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for s in sections:
            types[s].update(data.get(s, {}))
    json_mb = sum(p.stat().st_size for p in json_paths) / 2**20
    n_pskx = len(list(MESHES_DIR.rglob("*.pskx"))) if MESHES_DIR.exists() else 0
    n_psk  = len(list(MESHES_DIR.rglob("*.psk")))  if MESHES_DIR.exists() else 0

    tui.heading("Results")
    for label, count, what in (
        ("maps", len(json_paths),
         f"JSONs in {JSON_DIR.name}/  ({json_mb:.0f} MB)"),
        ("symbols", len(types["symbols"]), "mesh types"),
        ("groups", len(types["groups"]), "mesh types"),
        ("blueprints", len(types["blueprints"]), "class types"),
        ("splines", len(types["splines"]), "mesh types"),
        ("meshes", n_pskx, f".pskx files in {MESHES_DIR.name}/"),
        ("meshes", n_psk, f".psk  files in {MESHES_DIR.name}/"),
    ):
        print(f"  {label:<10} {tui.bold(f'{count:>6}')}  {tui.dim(what)}")

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

        tui.heading("Catalogue diff", f"{len(catalogue_entries)} in "
                    f"catalogue, {len(exported)} exported")
        print(f"  in catalogue, not exported: {len(missing)}")
        for name in missing:
            print(f"    {tui.red('-')} {name}")
        print(f"  exported, not in catalogue: {len(disappeared)}")
        for name in disappeared:
            print(f"    {tui.green('+')} {name}")
    else:
        if not CATALOGUE_FILE.exists():
            tui.warn(f"catalogue not found: {CATALOGUE_FILE}")
        if not MESHES_DIR.exists():
            tui.warn(f"meshes dir not found: {MESHES_DIR}")

    update_hashes()
    return 0


if __name__ == "__main__":
    tui.run(main)
