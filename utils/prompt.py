"""
prompt.py
========
Arrow-key-driven interactive menus (select / multiselect / confirm) for the
pipeline scripts' pre-TUI prompts -- Up/Down to move, Space to toggle in a
multiselect, Enter to confirm, type-to-filter to narrow a long list. Stdlib
only (msvcrt) -- no third-party dependency, matching utils/tui.py's
"hand-roll it" precedent. Windows-only, matching the rest of the project.

Usage:
    idx = select("Select map", ["All maps", "MapA", "MapB"])
    checked = multiselect("Select bakes", labels, checked={0, 1, 2})
    yes = confirm("Downscale to 1k?", default=False)
    path_str = text("Path to PNG: ")

select()/multiselect()/confirm() assume a real interactive Windows console
is attached. Callers must check is_interactive() first and fall back to a
plain input()-based prompt when it's False (piped/redirected/non-Windows
invocation) -- see the six call sites in 2_blend_all.py, 3_blend_spills.py,
4_render_spills.py, and 6_breaker.py for the pattern.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import List, Optional, Sequence, Set, TextIO, Tuple

from utils.config import PROMPT_MAX_VISIBLE_OPTIONS

_CSI = "\x1b["
_HIDE_CURSOR = f"{_CSI}?25l"
_SHOW_CURSOR = f"{_CSI}?25h"
_CLEAR_LINE = f"{_CSI}2K"
_RESET = f"{_CSI}0m"
_BOLD = f"{_CSI}1m"
_DIM = f"{_CSI}2m"
_CYAN = f"{_CSI}36m"
_GREEN = f"{_CSI}32m"

_ARROW_CODES = {"H": "UP", "P": "DOWN", "K": "LEFT", "M": "RIGHT"}


def _cursor_up(n: int) -> str:
    return f"{_CSI}{n}A" if n else ""


def is_interactive() -> bool:
    """Whether a real interactive Windows console is attached, i.e.
    whether select()/multiselect()/confirm() can be used at all. msvcrt
    reads the console input buffer directly (bypassing sys.stdin), so
    piped/redirected stdin must be detected here -- otherwise a read
    would block forever waiting for a keypress that will never come."""
    if os.name != "nt":
        return False
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    try:
        import msvcrt  # noqa: F401
    except ImportError:
        return False
    return True


def _supports_unicode(stream: TextIO) -> bool:
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        "❯✓".encode(encoding)  # "❯✓"
        return True
    except (LookupError, UnicodeEncodeError):
        return False


def _read_key() -> str:
    """Block for one keypress. Arrow/nav keys decode to the sentinel
    strings "UP"/"DOWN"/"LEFT"/"RIGHT"; everything else is returned as
    the raw character msvcrt read. Raises KeyboardInterrupt on Ctrl-C --
    Windows normally delivers that via an out-of-band console-control
    thread even while blocked here, but the explicit check covers any
    console host where that signal doesn't fire.
    """
    import msvcrt

    ch = msvcrt.getwch()
    if ch == "\x03":
        raise KeyboardInterrupt
    if ch in ("\x00", "\xe0"):
        code = msvcrt.getwch()
        return _ARROW_CODES.get(code, "")
    return ch


def text(question: str) -> str:
    """Plain input() prompt -- for literal free-text entry (e.g. pasting
    a file path) where there's nothing to navigate."""
    return input(question)


class _Menu:
    """Shared raw-key engine for select()/multiselect(). Renders a
    question line, one row per (filtered) option, and a filter/hint row,
    redrawing the whole block in place after each keypress."""

    def __init__(
        self,
        question: str,
        options: Sequence[str],
        *,
        multi: bool,
        checked: Optional[Set[int]] = None,
        highlighted: int = 0,
    ) -> None:
        self.question = question
        self.options = list(options)
        self.multi = multi
        self.checked: Set[int] = set(checked) if checked else set()
        self.highlighted = highlighted
        self.filter_buf = ""
        self.stream = sys.stdout
        self.unicode = _supports_unicode(self.stream)
        self._prev_lines = 0

    def _filtered(self) -> List[int]:
        if not self.filter_buf:
            return list(range(len(self.options)))
        needle = self.filter_buf.lower()
        return [i for i, opt in enumerate(self.options) if needle in opt.lower()]

    def _snap(self, filtered: List[int]) -> List[int]:
        if self.highlighted not in filtered:
            self.highlighted = filtered[0] if filtered else -1
        return filtered

    def _move(self, filtered: List[int], step: int) -> List[int]:
        if not filtered:
            return filtered
        pos = filtered.index(self.highlighted) if self.highlighted in filtered else 0
        self.highlighted = filtered[(pos + step) % len(filtered)]
        return filtered

    def _max_visible(self) -> int:
        """Cap the option viewport to PROMPT_MAX_VISIBLE_OPTIONS rows, or
        fewer on a short terminal (reserving room for the question, the
        scroll indicators, and the filter/hint row)."""
        rows = shutil.get_terminal_size(fallback=(100, 24)).lines
        reserved = 6
        return max(3, min(PROMPT_MAX_VISIBLE_OPTIONS, rows - reserved))

    def _visible_window(self, filtered: List[int]) -> Tuple[List[int], int, int]:
        """Slice of ``filtered`` to actually render, centred on the
        highlighted row, plus how many matches are hidden above/below."""
        max_visible = self._max_visible()
        if len(filtered) <= max_visible:
            return filtered, 0, 0
        pos = filtered.index(self.highlighted) if self.highlighted in filtered else 0
        start = max(0, min(pos - max_visible // 2, len(filtered) - max_visible))
        end = start + max_visible
        return filtered[start:end], start, len(filtered) - end

    def _build_row(self, i: int, budget: int, cursor_glyph: str, check_glyph: str) -> str:
        is_hi = i == self.highlighted
        marker = cursor_glyph if is_hi else " "
        row = f" {marker} "
        if self.multi:
            box = check_glyph if i in self.checked else " "
            box_color = _GREEN if i in self.checked else ""
            row += f"[{box_color}{box}{_RESET if box_color else ''}] "
        row += self.options[i]
        if len(row) > budget:
            row = row[: max(budget - 3, 0)] + "..."
        if is_hi:
            row = f"{_CYAN}{_BOLD}{row}{_RESET}"
        return row

    def _build_lines(self, filtered: List[int]) -> List[str]:
        columns = shutil.get_terminal_size(fallback=(100, 24)).columns
        budget = max(columns - 1, 1)
        cursor_glyph = "❯" if self.unicode else ">"  # "❯"
        check_glyph = "✓" if self.unicode else "x"   # "✓"
        up_glyph = "↑" if self.unicode else "^"       # "↑"
        down_glyph = "↓" if self.unicode else "v"     # "↓"

        lines = [f"{_BOLD}? {self.question}{_RESET}"]
        if not filtered:
            lines.append(f"  {_DIM}(no matches){_RESET}")
        else:
            window, above, below = self._visible_window(filtered)
            if above:
                lines.append(f"  {_DIM}{up_glyph} {above} more above{_RESET}")
            lines.extend(
                self._build_row(i, budget, cursor_glyph, check_glyph) for i in window
            )
            if below:
                lines.append(f"  {_DIM}{down_glyph} {below} more below{_RESET}")

        match_count = f"{len(filtered)}/{len(self.options)} matches"
        lines.append(f"{_DIM}/{self.filter_buf}  ({match_count}){_RESET}")
        return lines

    def _redraw(self, lines: List[str]) -> None:
        if self._prev_lines:
            self.stream.write(_cursor_up(self._prev_lines))
        for line in lines:
            self.stream.write("\r" + _CLEAR_LINE + line + "\n")
        stale = self._prev_lines - len(lines)
        if stale > 0:
            for _ in range(stale):
                self.stream.write("\r" + _CLEAR_LINE + "\n")
            self.stream.write(_cursor_up(stale))
        self.stream.flush()
        self._prev_lines = len(lines)

    def _handle_key(self, key: str, filtered: List[int]) -> Tuple[List[int], bool]:
        """Apply one keypress. Returns (new filtered list, submitted)."""
        if key == "\r":
            return filtered, bool(filtered)
        if key == " " and self.multi and filtered:
            self.checked ^= {self.highlighted}
        elif key == "UP":
            filtered = self._move(filtered, -1)
        elif key == "DOWN":
            filtered = self._move(filtered, +1)
        elif key == "\x08":
            self.filter_buf = self.filter_buf[:-1]
            filtered = self._snap(self._filtered())
        elif key and len(key) == 1 and key.isprintable():
            self.filter_buf += key
            filtered = self._snap(self._filtered())
        return filtered, False

    def _result(self) -> object:
        return self.checked if self.multi else self.highlighted

    def run(self) -> Optional[object]:
        stream = self.stream
        stream.write(_HIDE_CURSOR)
        try:
            filtered = self._filtered()
            self._redraw(self._build_lines(filtered))
            while True:
                try:
                    key = _read_key()
                except KeyboardInterrupt:
                    return None
                if key == "\x1b":
                    return None
                filtered, submitted = self._handle_key(key, filtered)
                if submitted:
                    return self._result()
                self._redraw(self._build_lines(filtered))
        finally:
            stream.write(_SHOW_CURSOR)
            stream.flush()


def select(question: str, options: Sequence[str], *, default: int = 0) -> Optional[int]:
    """Arrow-key single-select menu. Returns the chosen index into
    ``options`` (Up/Down to move, type to filter, Enter to confirm), or
    None if cancelled (Esc/Ctrl-C). Assumes is_interactive() -- see the
    module docstring."""
    if not options:
        return None
    return _Menu(question, options, multi=False, highlighted=default).run()


def multiselect(
    question: str,
    options: Sequence[str],
    *,
    checked: Optional[Set[int]] = None,
) -> Optional[Set[int]]:
    """Arrow-key checkbox menu. Returns the set of checked indices into
    ``options`` (Space to toggle, Enter to confirm), or None if cancelled
    (Esc/Ctrl-C). ``checked`` seeds the initial checked set. Assumes
    is_interactive() -- see the module docstring."""
    if not options:
        return set()
    return _Menu(question, options, multi=True, checked=checked).run()


def _confirm_line(question: str, yes: bool) -> str:
    yes_label = f"{_CYAN}{_BOLD}Yes{_RESET}" if yes else "Yes"
    no_label = f"{_CYAN}{_BOLD}No{_RESET}" if not yes else "No"
    return f"{_BOLD}? {question}{_RESET}  {yes_label} / {no_label}"


def confirm(question: str, *, default: bool = False) -> bool:
    """Compact yes/no prompt: Left/Right/Up/Down or y/n set the
    highlighted choice, Enter confirms it. Assumes is_interactive() --
    see the module docstring. No cancel value; Ctrl-C propagates as
    KeyboardInterrupt like a plain input() prompt would."""
    stream = sys.stdout
    yes = default
    stream.write(_HIDE_CURSOR)
    try:
        stream.write("\r" + _CLEAR_LINE + _confirm_line(question, yes))
        stream.flush()
        while True:
            key = _read_key()
            if key == "\r":
                stream.write("\n")
                stream.flush()
                return yes
            if key in ("LEFT", "RIGHT", "UP", "DOWN"):
                yes = not yes
            elif key in ("y", "Y"):
                yes = True
            elif key in ("n", "N"):
                yes = False
            stream.write("\r" + _CLEAR_LINE + _confirm_line(question, yes))
            stream.flush()
    finally:
        stream.write(_SHOW_CURSOR)
        stream.flush()
