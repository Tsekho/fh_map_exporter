"""Map-mod toolset: everything between a world image and the game.

    python 6_toolset.py                                  # pick interactively
    python 6_toolset.py recipe recipe.json [-n NAME] [-o DIR] [--store]
    python 6_toolset.py mod    World.png|DIR [-n NAME] [-o DIR] [--store]
    python 6_toolset.py unpack Mod.pak [-o DIR]
    python 6_toolset.py break  World.png [--1k] [-o DIR]
    python 6_toolset.py stitch DIR [-o World.png]

  recipe  JSON list of world layers -> War-WindowsNoEditor_<NAME>.pak
          (see toolset/recipe.py; recipe.json rebuilds the release mod)
  mod     world PNG, or a folder of region PNGs -> War-WindowsNoEditor_<NAME>.pak
  unpack  map mod .pak -> one PNG per region
  break   world PNG -> one PNG per region (2048x1776, or 1024x888 with --1k)
  stitch  folder of region PNGs -> world PNG

A world PNG is laid out like the export/_final layers. Region PNGs are named
after their region (AcrithiaHex.png, ...); a folder may hold only some of
them, and a mod built from it leaves the rest of the map untouched. Outputs
land in the current directory unless -o says otherwise.
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from toolset import mapmod, recipe as recipes
from utils import tui
from utils.png import imwrite_atomic


# ------------------------------------------------------------------------------
#  Actions
# ------------------------------------------------------------------------------

def _write_tiles(tiles: Iterable[Tuple[str, np.ndarray]], total: int,
                 out_dir: Path, title: str) -> int:
    """Write (texture name, tile) pairs as <out_dir>/<Region>.png."""
    out_dir.mkdir(parents=True, exist_ok=True)
    failed: List[str] = []
    with tui.Progress(title, unit="tile", step_unit="tile") as bar:
        bar.start("tiles", out_dir.name, total)
        for tex, tile in tiles:
            name = mapmod.short_name(tex)
            try:
                imwrite_atomic(str(out_dir / f"{name}.png"), tile)
            except OSError as exc:
                bar.log(f"  [WARN] {name}: {exc}")
                failed.append(name)
            bar.update("tiles", advance=1, status=name)
        bar.finish("tiles", not failed,
                   note=f"{len(failed)} failed" if failed else "")
    if failed:
        tui.error(f"{len(failed)} tile(s) not written: {', '.join(failed)}")
        return 1
    tui.done(f"{total - len(failed)} tile(s) in {out_dir}")
    return 0


def _load_folder(folder: Path) -> Optional[dict]:
    """Region tiles from a folder, reporting what was skipped or missing."""
    if not folder.is_dir():
        tui.error(f"not a folder: {folder}")
        return None
    try:
        tiles, skipped = mapmod.load_folder(folder)
    except (FileNotFoundError, ValueError) as exc:
        tui.error(str(exc))
        return None
    for name in skipped:
        tui.warn(f"{name}: not named after a region, skipped")
    if not tiles:
        tui.error(f"no region images in {folder}")
        return None
    missing = sorted(mapmod.short_name(t) for t in mapmod.HEADERS
                     if t not in tiles)
    if missing:
        print(tui.dim(f"  {len(missing)} region(s) not in the folder: "
                      f"{', '.join(missing)}"))
    return tiles


def _load_world(path: Path) -> Optional[np.ndarray]:
    print(tui.dim(f"  loading {path.name} ..."))
    try:
        return mapmod.load_image(path)
    except FileNotFoundError as exc:
        tui.error(str(exc))
        return None


def do_break(src: Path, one_k: bool, out_dir: Path) -> int:
    try:
        world = _load_world(src)
        if world is None:
            return 1
        tiles = mapmod.break_world(world, one_k)
    except (FileNotFoundError, ValueError) as exc:
        tui.error(str(exc))
        return 1
    size = "1024x888" if one_k else f"{mapmod.TEX_W}x{mapmod.TEX_H}"
    tui.heading(f"Breaking {src.name}",
                f"{world.shape[1]}x{world.shape[0]} -> "
                f"{len(tiles)} regions @ {size}")
    del world
    return _write_tiles(tiles.items(), len(tiles), out_dir, "Breaking tiles")


def do_mod(src: Path, name: str, out_dir: Path, compress: bool = True) -> int:
    if src.is_dir():
        tiles = _load_folder(src)
    else:
        world = _load_world(src)
        tiles = None if world is None else mapmod.break_world(world)
        del world
    if tiles is None:
        return 1
    return _build_mod(tiles, name, out_dir, compress)


def _load_recipe(path: Path) -> Optional[recipes.Recipe]:
    """A recipe whose files are all there, or None (reported)."""
    try:
        rec = recipes.load(path)
    except (OSError, recipes.RecipeError) as exc:
        tui.error(str(exc))
        return None
    missing = recipes.missing(rec)
    for f in missing:
        tui.error(f"{f}: not found")
    return None if missing else rec


def do_recipe(path: Path, name: Optional[str], out_dir: Path,
              compress: bool = True, rec: Optional[recipes.Recipe] = None,
              ) -> int:
    rec = rec or _load_recipe(path)
    if rec is None:
        return 1
    tui.heading(f"Compositing {path.name}", f"{len(rec.layers)} layer(s)")
    try:
        with tui.Progress("Compositing", unit="layer",
                          step_unit="layer") as bar:
            bar.start("layers", path.stem, len(rec.layers))
            world = recipes.compose(rec, on_layer=lambda lyr: bar.update(
                "layers", advance=1, status=lyr.label))
            bar.finish("layers")
    except (OSError, recipes.RecipeError) as exc:
        tui.error(str(exc))
        return 1
    tiles = mapmod.break_world(world)
    del world
    return _build_mod(tiles, name or rec.name, out_dir, compress)


def _build_mod(tiles: dict, name: str, out_dir: Path,
               compress: bool) -> int:
    out = mapmod.mod_path(out_dir, name)
    tui.heading(f"Building {out.name}", f"{len(tiles)} region(s)")
    with tui.Progress("Encoding BC7", unit="texture",
                      step_unit="texture") as bar:
        bar.start("bc7", name, len(tiles))
        textures = mapmod.encode_all(
            tiles, on_done=lambda tex: bar.update(
                "bc7", advance=1, status=mapmod.short_name(tex)))
        bar.finish("bc7")

    print(tui.dim("  packing ..."))
    out_dir.mkdir(parents=True, exist_ok=True)
    mapmod.write_mod(out, textures, compress)
    tui.done(f"{out}  {tui.dim(f'{out.stat().st_size / 2**20:.1f} MB')}")
    return 0


def do_unpack(pak: Path, out_dir: Path) -> int:
    try:
        textures = mapmod.read_mod(pak)
    except (OSError, ValueError) as exc:
        tui.error(str(exc))
        return 1
    if not textures:
        tui.error(f"no map textures in {pak.name}")
        return 1
    tui.heading(f"Unpacking {pak.name}", f"{len(textures)} region(s)")
    decoded = ((tex, mapmod.decode(bc7)) for tex, bc7 in textures.items())
    return _write_tiles(decoded, len(textures), out_dir, "Decoding tiles")


def do_stitch(folder: Path, out: Path) -> int:
    tiles = _load_folder(folder)
    if tiles is None:
        return 1
    tui.heading(f"Stitching {folder.name}", f"{len(tiles)} region(s)")
    world = mapmod.stitch_world(tiles)
    try:
        imwrite_atomic(str(out), world)
    except OSError as exc:
        tui.error(str(exc))
        return 1
    tui.done(f"{out}  {tui.dim(f'{world.shape[1]}x{world.shape[0]}')}")
    return 0


# ------------------------------------------------------------------------------
#  Interactive
# ------------------------------------------------------------------------------

_PASTE = "Paste a path manually..."

ACTIONS = [
    ("recipe", "recipe.json     ->  mod.pak"),
    ("mod", "World.png       ->  mod.pak"),
    ("mod_dir", "Regions/*.png   ->  mod.pak"),
    ("unpack", "mod.pak         ->  Regions/*.png"),
    tui.SEPARATOR,
    ("break", "World.png       ->  Regions/*.png"),
    ("stitch", "Regions/*.png   ->  World.png"),
]


def _pick(entries: Sequence[Path], title: str, noun: str,
          want_dir: bool) -> Optional[Path]:
    picked = tui.select_one(
        entries, title,
        label_fn=lambda p: p.name + ("/" if want_dir else ""),
        extra=_PASTE,
        noun=noun,
    )
    if picked is None or picked != _PASTE:
        return picked
    while True:
        raw = input(f"Path to {noun}: ").strip().strip('"').strip("'")
        if not raw:
            tui.warn("empty path; try again")
            continue
        path = Path(raw).expanduser()
        if path.is_dir() if want_dir else path.is_file():
            return path
        tui.warn(f"not a {'folder' if want_dir else 'file'}: {path}")


def _pick_file(title: str, suffixes: Sequence[str],
               noun: str) -> Optional[Path]:
    files = sorted(p for p in Path.cwd().iterdir()
                   if p.is_file() and p.suffix.lower() in suffixes)
    return _pick(files, title, noun, want_dir=False)


def _pick_folder(title: str) -> Optional[Path]:
    folders = sorted(p for p in Path.cwd().iterdir()
                     if p.is_dir() and any(p.glob("*.png")))
    return _pick(folders, title, "folder", want_dir=True)


def _ask_name(default: str) -> str:
    raw = input(f"Mod name [{default}]: ").strip()
    return raw or default


def interactive() -> int:
    action = tui.select_one(ACTIONS, "Toolset", label_fn=lambda a: a[1],
                            noun="action")
    if action is None:
        return 1
    key = action[0]
    cwd = Path.cwd()

    if key == "break":
        src = _pick_file("World PNG", (".png",), "PNG")
        if src is None:
            return 1
        one_k = tui.confirm("Downscale output to 1k (1024x888)?",
                            default=False)
        if one_k is None:
            return 1
        return do_break(src, one_k, cwd / src.stem)

    if key in ("mod", "mod_dir"):
        src = (_pick_file("World PNG", (".png",), "PNG") if key == "mod"
               else _pick_folder("Folder of region PNGs"))
        if src is None:
            return 1
        return do_mod(src, _ask_name(src.stem), cwd)

    if key == "recipe":
        path = _pick_file("Recipe", (".json",), "recipe")
        if path is None:
            return 1
        rec = _load_recipe(path)
        if rec is None:
            return 1
        return do_recipe(path, _ask_name(rec.name), cwd, rec=rec)

    if key == "unpack":
        pak = _pick_file("Mod to unpack", (".pak",), "pak")
        if pak is None:
            return 1
        return do_unpack(pak, cwd / pak.stem)

    folder = _pick_folder("Folder of region PNGs")
    if folder is None:
        return 1
    return do_stitch(folder, cwd / f"{folder.name}.png")


# ------------------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------------------

def main() -> int:
    if len(sys.argv) == 1:
        return interactive()

    parser = argparse.ArgumentParser(
        description="Break, stitch, compose, pack and unpack Foxhole "
                    "world-map textures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="action", required=True)

    p = sub.add_parser("recipe", help="layer recipe .json -> .pak")
    p.add_argument("recipe", type=Path)
    p.add_argument("-n", "--name",
                   help="mod name (default: the recipe's \"name\")")
    p.add_argument("-o", "--out", type=Path, default=Path.cwd(),
                   help="output folder (default: .)")
    p.add_argument("--store", action="store_true",
                   help="skip zlib compression")

    p = sub.add_parser("mod", help="world PNG or region PNG folder -> .pak")
    p.add_argument("src", type=Path)
    p.add_argument("-n", "--name",
                   help="mod name (default: <src stem>)")
    p.add_argument("-o", "--out", type=Path, default=Path.cwd(),
                   help="output folder (default: .)")
    p.add_argument("--store", action="store_true",
                   help="skip zlib compression")

    p = sub.add_parser("unpack", help="mod .pak -> region PNGs")
    p.add_argument("pak", type=Path)
    p.add_argument("-o", "--out", type=Path,
                   help="output folder (default: ./<pak stem>)")

    p = sub.add_parser("break", help="world PNG -> region PNGs")
    p.add_argument("src", type=Path)
    p.add_argument("--1k", dest="one_k", action="store_true",
                   help="downscale tiles to 1024x888")
    p.add_argument("-o", "--out", type=Path,
                   help="output folder (default: ./<src stem>)")

    p = sub.add_parser("stitch", help="region PNG folder -> world PNG")
    p.add_argument("folder", type=Path)
    p.add_argument("-o", "--out", type=Path,
                   help="output PNG (default: ./<folder name>.png)")

    args = parser.parse_args()
    cwd = Path.cwd()
    if args.action == "break":
        return do_break(args.src, args.one_k, args.out or cwd / args.src.stem)
    if args.action == "mod":
        return do_mod(args.src, args.name or args.src.stem, args.out,
                      compress=not args.store)
    if args.action == "recipe":
        return do_recipe(args.recipe, args.name, args.out,
                         compress=not args.store)
    if args.action == "unpack":
        return do_unpack(args.pak, args.out or cwd / args.pak.stem)
    return do_stitch(args.folder,
                     args.out or cwd / f"{args.folder.name}.png")


if __name__ == "__main__":
    tui.run(main)
