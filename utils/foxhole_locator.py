"""
foxhole_locator.py
========
Locates Foxhole's War-WindowsNoEditor.pak without relying on a single
hardcoded Steam install path -- the game can be installed under any
Steam library folder, on any drive.

Search order:
  1. The FOXHOLE_PAK_PATH environment variable, if set -- points
     straight at the .pak file, for non-Steam or otherwise unusual
     installs.
  2. Every Steam library folder (read from the registry, then from
     Steam's own libraryfolders.vdf) that has a "Foxhole" folder under
     steamapps/common/.
  3. A couple of conventional default Steam install locations, as a
     last resort for machines where the registry key isn't set (e.g.
     Steam has never actually been launched).

find_foxhole_pak() returns None if nothing is found; it never raises --
callers are responsible for reporting that to the user (see
1_export.py). This is important because utils/config.py calls it at
import time, and utils/config.py is imported by every pipeline script,
including ones that have nothing to do with the .pak file -- a broken
registry/VDF read here must not be able to crash unrelated scripts.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

_PAK_RELATIVE_PATH = Path("War") / "Content" / "Paks" / "War-WindowsNoEditor.pak"
_GAME_FOLDER_NAME = "Foxhole"

_DEFAULT_LIBRARY_PATHS = [
    Path(r"C:\Program Files (x86)\Steam"),
    Path(r"C:\Program Files\Steam"),
]


def _steam_install_path() -> Optional[Path]:
    """Steam's own install directory, read from the registry. Returns
    None if Steam has never been installed/launched on this machine, or
    if winreg isn't available (non-Windows)."""
    try:
        import winreg
    except ImportError:
        return None

    candidates = (
        (winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam", "InstallPath"),
    )
    for hive, key, value_name in candidates:
        try:
            with winreg.OpenKey(hive, key) as opened:
                value, _ = winreg.QueryValueEx(opened, value_name)
        except OSError:
            continue
        path = Path(value)
        if path.is_dir():
            return path
    return None


def _library_folders(steam_path: Path) -> List[Path]:
    """Every Steam library root (the main install plus any additional
    drives), read from libraryfolders.vdf. Falls back to just the main
    install if that file is missing or unreadable."""
    libraries = [steam_path]
    vdf_path = steam_path / "steamapps" / "libraryfolders.vdf"
    try:
        text = vdf_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return libraries

    for match in re.finditer(r'"path"\s+"([^"]+)"', text):
        path = Path(match.group(1).replace("\\\\", "\\"))
        if path.is_dir() and path not in libraries:
            libraries.append(path)
    return libraries


def _pak_in_library(library: Path) -> Optional[Path]:
    pak = library / "steamapps" / "common" / _GAME_FOLDER_NAME / _PAK_RELATIVE_PATH
    return pak if pak.is_file() else None


def find_foxhole_pak() -> Optional[Path]:
    """Locate War-WindowsNoEditor.pak. See module docstring for the
    search order. Never raises -- returns None if it can't be found."""
    override = os.environ.get("FOXHOLE_PAK_PATH")
    if override:
        path = Path(override)
        if path.is_file():
            return path

    try:
        libraries: List[Path] = []
        steam_path = _steam_install_path()
        if steam_path is not None:
            libraries.extend(_library_folders(steam_path))
        for default in _DEFAULT_LIBRARY_PATHS:
            if default.is_dir() and default not in libraries:
                libraries.append(default)

        for library in libraries:
            pak = _pak_in_library(library)
            if pak is not None:
                return pak
    except Exception:
        pass

    return None
