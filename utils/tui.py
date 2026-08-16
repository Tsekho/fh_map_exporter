"""
tui.py
========
Shared terminal UI for the pipeline scripts (0_make_release.py through
6_breaker.py): a scrolling log region with a progress bar pinned to the
last terminal line, showing x/y completion and elapsed time. Stdlib only
(ctypes/threading/shutil/time) -- no third-party dependency.

Usage:
    with ScriptTUI(title="Building maps", total=len(items)) as tui:
        for item in items:
            tui.set_description(item)
            tui.log(f"processing {item}")
            tui.advance()

Deeply-nested domain code (utils/map.py, utils/bake.py, ...) can't
conveniently take a `tui` parameter, so it should import and call the
module-level free functions (log/warn/error/advance/...) below instead;
they route into whichever ScriptTUI is currently active, or fall back to
plain print() when none is.

SIGTERM is caught and converted into a clean shutdown (cursor restored,
"with" block unwound normally) -- see ScriptTUI._on_sigterm. SIGKILL
cannot be handled here or anywhere: POSIX guarantees it's undeliverable
to any signal handler, and Windows has no equivalent at all (Task
Manager "End Task" / TerminateProcess tears the process down with no
chance to run cleanup code). If a script is hard-killed and a terminal
is left with its cursor hidden, running a further script (its __enter__
re-hides then its __exit__ shows it again) or simply typing `cls`/
`clear` recovers it.
"""

from __future__ import annotations

import atexit
import ctypes
import os
import shutil
import signal
import sys
import threading
import time
from types import FrameType, TracebackType
from typing import Optional, TextIO, Tuple, Type

from utils.config import TUI_BAR_WIDTH, TUI_ETA_SMOOTHING, TUI_TICK_INTERVAL_S

_CSI = "\x1b["
_HIDE_CURSOR = f"{_CSI}?25l"
_SHOW_CURSOR = f"{_CSI}?25h"
_CLEAR_LINE = f"{_CSI}2K"

_RESET = f"{_CSI}0m"
_BOLD = f"{_CSI}1m"
_DIM = f"{_CSI}2m"
_CYAN = f"{_CSI}36m"
_GREEN = f"{_CSI}32m"
_FG_BLACK = f"{_CSI}30m"
_FG_WHITE = f"{_CSI}97m"
_BG_CYAN = f"{_CSI}46m"
_BG_YELLOW = f"{_CSI}43m"
_BG_RED = f"{_CSI}41m"

# Level tags for log()/warn()/error(): text + (background, foreground).
# Padded to a common width so every tag occupies the same on-screen
# width regardless of level.
_LEVEL_TAGS = {"info": "INFO", "warn": "WARN", "error": "ERROR"}
_LEVEL_COLORS = {
    "info": (_BG_CYAN, _FG_WHITE),
    "warn": (_BG_YELLOW, _FG_BLACK),
    "error": (_BG_RED, _FG_WHITE),
}
_LEVEL_TAG_WIDTH = max(len(tag) for tag in _LEVEL_TAGS.values())


def _tag_prefix(level: str, *, colored: bool) -> str:
    """Left-aligned, fixed-width level tag with one space of padding on
    each side, e.g. " INFO  " / " WARN  " / " ERROR ". Colored with a
    background block when the stream supports ANSI, else a plain
    bracketed fallback of the same tag width."""
    tag = _LEVEL_TAGS.get(level, _LEVEL_TAGS["info"]).ljust(_LEVEL_TAG_WIDTH)
    if not colored:
        return f"[{tag}] "
    bg, fg = _LEVEL_COLORS.get(level, _LEVEL_COLORS["info"])
    return f"{bg}{fg}{_BOLD} {tag} {_RESET} "


def _tag_lines(message: str, level: str, *, colored: bool) -> str:
    """Prefix every non-blank line of ``message`` with a level tag; blank
    lines (e.g. the leading "\\n" in banner-style messages) are left
    untouched rather than getting a tagged empty row."""
    prefix = _tag_prefix(level, colored=colored)
    return "\n".join(
        prefix + line if line else line for line in message.split("\n")
    )


def strip_tag(line: str) -> Tuple[str, str]:
    """If ``line`` starts with a plain (uncolored) level tag -- as
    produced by a child process's own non-interactive log()/warn()/
    error() fallback -- return ``(level, remainder)`` with the tag and
    its trailing space removed. Otherwise return ``("info", line)``
    unchanged. Used by utils/parallel.py so a forwarded child-process
    line that was already an ERROR/WARN gets re-tagged at that level
    instead of being wrapped in a second, misleading INFO tag."""
    for level, tag in _LEVEL_TAGS.items():
        prefix = f"[{tag.ljust(_LEVEL_TAG_WIDTH)}] "
        if line.startswith(prefix):
            return level, line[len(prefix):]
    return "info", line

# Minimum bar-graphic width worth drawing; below this it's dropped
# entirely so the percent/count/elapsed suffix -- which is never
# truncated -- still fits on narrow terminals.
_MIN_BAR_GLYPHS = 4

# Bar fill glyphs. Unicode eighth-block characters (index 0..6 = 1/8..7/8
# fill) give the bar sub-character precision; used only when the stream's
# encoding can represent them (see _supports_unicode_bar).
_BLOCK_FULL = "█"     # full block
_BLOCK_EMPTY = "░"    # light shade
_BLOCK_PARTIALS = "▏▎▍▌▋▊▉"  # 1/8..7/8

_ACTIVE: "Optional[ScriptTUI]" = None


def _enable_windows_vt(stream: TextIO) -> bool:
    """Best-effort enable of ANSI escape processing (and a UTF-8 output
    codepage, for the block-character bar) on legacy Windows consoles.
    Returns True if the stream can be treated as ANSI-capable."""
    if os.name != "nt":
        return True
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        enable_vt_flag = 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        ok = bool(kernel32.SetConsoleMode(handle, mode.value | enable_vt_flag))
        kernel32.SetConsoleOutputCP(65001)  # best-effort; ignore failure
        return ok
    except Exception:
        return False


def _supports_unicode_bar(stream: TextIO) -> bool:
    """Whether ``stream`` can encode the block-character bar glyphs."""
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        (_BLOCK_FULL + _BLOCK_EMPTY + _BLOCK_PARTIALS).encode(encoding)
        return True
    except (LookupError, UnicodeEncodeError):
        return False


def _format_elapsed(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class ScriptTUI:
    """Pinned bottom progress bar + scrolling log, stdlib-only.

    The bar is redrawn in place via `\\r` + erase-line rather than a VT
    scroll region -- simpler and self-healing on Windows consoles, where
    scroll-region escapes behave inconsistently.
    """

    def __init__(
        self,
        title: str = "",
        total: int = 0,
        *,
        bar_width: int = TUI_BAR_WIDTH,
        tick_interval: float = TUI_TICK_INTERVAL_S,
        stream: TextIO = sys.stdout,
    ) -> None:
        self._description = title
        self._total = max(total, 0)
        self._completed = 0
        self._bar_width = bar_width
        self._tick_interval = tick_interval
        self._stream = stream
        self._lock = threading.Lock()
        self._start = 0.0
        self._last_advance_time = 0.0
        self._ema_item_seconds: Optional[float] = None
        self._interactive = False
        self._unicode_bar = False
        self._suspended = False
        self._heartbeat: Optional[threading.Thread] = None
        self._stop_heartbeat = threading.Event()
        self._prior_active: Optional["ScriptTUI"] = None
        self._prior_sigterm_handler: Optional[object] = None

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "ScriptTUI":
        global _ACTIVE
        self._interactive = self._stream.isatty() and _enable_windows_vt(self._stream)
        self._unicode_bar = self._interactive and _supports_unicode_bar(self._stream)
        self._start = time.monotonic()
        self._last_advance_time = self._start
        self._prior_active = _ACTIVE
        _ACTIVE = self
        atexit.register(self._restore)
        self._prior_sigterm_handler = self._install_sigterm_handler()
        with self._lock:
            if self._interactive:
                self._stream.write(_HIDE_CURSOR)
                self._redraw_locked()
        if self._interactive:
            self._heartbeat = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        global _ACTIVE
        try:
            self._restore_sigterm_handler()
            self._stop_heartbeat.set()
            if self._heartbeat is not None:
                self._heartbeat.join(timeout=2)
            with self._lock:
                if self._interactive:
                    self._redraw_locked()
                    self._stream.write("\n" + _SHOW_CURSOR)
                    self._stream.flush()
        finally:
            _ACTIVE = self._prior_active

    def _restore(self) -> None:
        """Best-effort cursor restore for the unhandled-exception path."""
        if self._interactive:
            self._stream.write(_SHOW_CURSOR)
            self._stream.flush()

    def _install_sigterm_handler(self) -> Optional[object]:
        """Convert SIGTERM into a SystemExit so it unwinds the ``with``
        block (and thus runs __exit__) instead of killing the process
        outright -- Python's default SIGTERM disposition terminates
        immediately, bypassing all cleanup. Best-effort: signal handlers
        can only be installed from the main thread, so this is a no-op
        (returns None) anywhere that isn't possible or supported.
        """
        try:
            return signal.signal(signal.SIGTERM, self._on_sigterm)
        except (ValueError, OSError, AttributeError):
            return None

    def _restore_sigterm_handler(self) -> None:
        if self._prior_sigterm_handler is None:
            return
        try:
            signal.signal(signal.SIGTERM, self._prior_sigterm_handler)  # type: ignore[arg-type]
        except (ValueError, OSError):
            pass

    @staticmethod
    def _on_sigterm(signum: int, frame: Optional[FrameType]) -> None:
        del frame
        raise SystemExit(128 + signum)

    def _heartbeat_loop(self) -> None:
        while not self._stop_heartbeat.wait(self._tick_interval):
            with self._lock:
                self._redraw_locked()

    # -- bar rendering (caller must hold self._lock) ------------------------

    def _bar_glyphs(self, width: int, frac: float) -> str:
        """Bar graphic, colored green (done) / dim (remaining).

        Uses unicode eighth-block characters for sub-character fill
        precision when the stream's encoding supports them, else falls
        back to a plain ASCII `====>----` bar.
        """
        if not self._unicode_bar:
            filled = int(round(width * frac))
            if filled <= 0:
                body = "-" * width
            elif filled >= width:
                body = "=" * width
            else:
                body = "=" * (filled - 1) + ">" + "-" * (width - filled)
            done, rest = body[:filled], body[filled:]
            return f"{_GREEN}{done}{_RESET}{_DIM}{rest}{_RESET}"

        eighths = max(0, min(int(round(width * 8 * frac)), width * 8))
        full, remainder = divmod(eighths, 8)
        full = min(full, width)
        partial = _BLOCK_PARTIALS[remainder - 1] if remainder and full < width else ""
        empty = max(width - full - len(partial), 0)
        done = _BLOCK_FULL * full + partial
        rest = _BLOCK_EMPTY * empty
        return f"{_GREEN}{done}{_RESET}{_DIM}{rest}{_RESET}"

    def _eta_seconds(self) -> Optional[float]:
        """Smoothed ETA: exponential-moving-average per-item duration
        (see TUI_ETA_SMOOTHING) times items remaining. None until at
        least one item has completed -- there's no rate to extrapolate
        from yet. Smoothing rides out per-item cost variance (e.g. one
        much bigger region) far better than a plain elapsed/completed
        average would."""
        if self._total <= 0 or self._ema_item_seconds is None:
            return None
        if self._completed >= self._total:
            return 0.0
        return self._ema_item_seconds * (self._total - self._completed)

    def _bar_text(self) -> str:
        total = max(self._total, 1)
        frac = min(self._completed / total, 1.0)
        percent = int(round(frac * 100))
        elapsed = _format_elapsed(time.monotonic() - self._start)
        eta_s = self._eta_seconds()
        eta = _format_elapsed(eta_s) if eta_s is not None else "--:--"

        # This suffix (percent, x/y count, elapsed/eta time) must always
        # be fully visible -- the bar graphic shrinks first, then the
        # description, to make room for it on narrow terminals.
        suffix_plain = (
            f"{percent:3d}%  {self._completed}/{self._total}  "
            f"elapsed {elapsed}  eta {eta}"
        )
        suffix = (
            f"{_BOLD}{percent:3d}%{_RESET}  "
            f"{_CYAN}{self._completed}/{self._total}{_RESET}  "
            f"{_DIM}elapsed {elapsed}  eta {eta}{_RESET}"
        )

        columns = shutil.get_terminal_size(fallback=(100, 24)).columns
        budget = max(columns - 1, 1)

        bar_width = min(self._bar_width, budget - len(suffix_plain) - 3)
        if bar_width < _MIN_BAR_GLYPHS:
            bar_part_plain = ""
            bar_part = ""
        else:
            bar_part_plain = f"[{'#' * bar_width}] "
            bar_part = f"[{self._bar_glyphs(bar_width, frac)}] "

        desc_budget = budget - len(bar_part_plain) - len(suffix_plain) - 2
        description = self._description
        if desc_budget <= 0:
            description = ""
        elif len(description) > desc_budget:
            description = (
                description[: max(desc_budget - 3, 0)] + "..."
                if desc_budget > 3 else ""
            )

        desc_part = f"{_BOLD}{description}{_RESET}  " if description else ""
        return f"{desc_part}{bar_part}{suffix}"

    def _redraw_locked(self) -> None:
        if not self._interactive or self._suspended:
            return
        self._stream.write("\r" + _CLEAR_LINE + self._bar_text())
        self._stream.flush()

    # -- public API ----------------------------------------------------------

    def log(self, message: str = "", *, level: str = "info") -> None:
        with self._lock:
            if not self._interactive:
                print(_tag_lines(message, level, colored=False), file=self._stream)
                return
            tagged = _tag_lines(message, level, colored=True)
            if self._suspended:
                self._stream.write(tagged + "\n")
                self._stream.flush()
                return
            # Clear the bar off the current row, write the new log content,
            # then redraw the bar fresh on whatever row the cursor lands
            # on. This deliberately never moves the cursor *up* -- only
            # "\r" + clear on the row we're already on, then natural "\n"
            # scrolling for everything else -- so it stays correct even
            # when a line is longer than the terminal width and the
            # terminal auto-wraps it onto extra visual rows; the terminal
            # itself accounts for that when it processes "\n", but a
            # fixed cursor-up-N-rows guess (an earlier version of this
            # code did that, to keep a blank line pinned above the bar)
            # can't, and ends up clobbering the wrong row.
            self._stream.write("\r" + _CLEAR_LINE)
            for line in tagged.split("\n"):
                self._stream.write(line + "\n")
            self._redraw_locked()

    def warn(self, message: str) -> None:
        self.log(message, level="warn")

    def error(self, message: str) -> None:
        self.log(message, level="error")

    def advance(self, n: int = 1) -> None:
        with self._lock:
            self._completed += n
            now = time.monotonic()
            if n > 0:
                item_seconds = (now - self._last_advance_time) / n
                self._ema_item_seconds = (
                    item_seconds if self._ema_item_seconds is None
                    else TUI_ETA_SMOOTHING * item_seconds
                    + (1 - TUI_ETA_SMOOTHING) * self._ema_item_seconds
                )
            self._last_advance_time = now
            if not self._interactive:
                print(f"[{self._completed}/{self._total}] {self._description}",
                      file=self._stream)
                return
            self._redraw_locked()

    def set_total(self, total: int) -> None:
        with self._lock:
            self._total = max(total, 0)
            self._redraw_locked()

    def set_description(self, text: str) -> None:
        with self._lock:
            self._description = text
            self._redraw_locked()

    def step(self, message: str, *, level: str = "info") -> None:
        """Log a line and advance the counter by one, in that order."""
        self.log(message, level=level)
        self.advance(1)

    def suspend(self) -> "_Suspend":
        """Hide the bar and hand the raw terminal to an external process
        (e.g. `dotnet publish`) whose own console output shouldn't be
        re-piped through our log."""
        return _Suspend(self)


class _Suspend:
    def __init__(self, tui: ScriptTUI) -> None:
        self._tui = tui

    def __enter__(self) -> None:
        with self._tui._lock:
            if self._tui._interactive:
                self._tui._stream.write("\r" + _CLEAR_LINE + _SHOW_CURSOR)
                self._tui._stream.flush()
            self._tui._suspended = True

    def __exit__(self, *exc_info: object) -> None:
        with self._tui._lock:
            self._tui._suspended = False
            if self._tui._interactive:
                self._tui._stream.write(_HIDE_CURSOR)
                self._tui._redraw_locked()


# -- module-level singleton + free functions --------------------------------
#
# Lets domain code several call-levels deep route into whichever ScriptTUI
# is currently active without threading a `tui` object through every
# function signature. Falls back to plain print() when no TUI is active.

def log(message: str = "", *, level: str = "info") -> None:
    if _ACTIVE is not None:
        _ACTIVE.log(message, level=level)
    else:
        print(_tag_lines(message, level, colored=False))


def warn(message: str) -> None:
    log(message, level="warn")


def error(message: str) -> None:
    log(message, level="error")


def advance(n: int = 1) -> None:
    if _ACTIVE is not None:
        _ACTIVE.advance(n)


def set_total(total: int) -> None:
    if _ACTIVE is not None:
        _ACTIVE.set_total(total)


def set_description(text: str) -> None:
    if _ACTIVE is not None:
        _ACTIVE.set_description(text)


def step(message: str, *, level: str = "info") -> None:
    log(message, level=level)
    advance(1)
