"""
tui.py
======
Terminal widgets shared by the ``N_*.py`` pipeline scripts: arrow-key
pickers, a yes/no prompt and progress bars.

Everything degrades to numbered ``input()`` prompts and plain lines when
stdin/stdout is not a terminal, so piped runs and subprocess workers behave
the same. No third-party dependencies: keys come from ``msvcrt`` on Windows
and ``termios``/``tty`` elsewhere, drawing is plain ANSI.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from typing import Any, Callable, List, Optional, Sequence

__all__ = [
    "supports_tui",
    "select_many",
    "select_one",
    "confirm",
    "heading",
    "Progress",
    "bold", "cyan", "green", "yellow", "red", "dim", "glyph",
]


# ------------------------------------------------------------------------------
#  Terminal capabilities
# ------------------------------------------------------------------------------

def _enable_vt() -> bool:
    """Turn on ANSI escape processing on Windows consoles. True if colour
    output is usable."""
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


_COLOR = False
_UNICODE = False


def _probe() -> None:
    global _COLOR, _UNICODE
    _COLOR = (
        os.environ.get("NO_COLOR") is None
        and os.environ.get("TERM") != "dumb"
        and _enable_vt()
    )
    _UNICODE = _can_encode(_GLYPH_PROBE)
    if not _UNICODE and _COLOR:
        # A VT-capable console renders UTF-8 fine even when Python picked up
        # a legacy codepage; switch the stream over to keep the glyphs.
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
            _UNICODE = _can_encode(_GLYPH_PROBE)
        except Exception:
            _UNICODE = False


_GLYPH_PROBE = "❯✓─↑←…·"


def _can_encode(text: str) -> bool:
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        text.encode(enc)
        return True
    except Exception:
        return False


def supports_tui() -> bool:
    """True when full-screen key-driven widgets can be used."""
    if os.environ.get("FH_NO_TUI"):
        return False
    try:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return False
    except Exception:
        return False
    if os.name == "nt":
        try:
            import msvcrt  # noqa: F401
        except ImportError:
            return False
    else:
        try:
            import termios  # noqa: F401
            import tty  # noqa: F401
        except ImportError:
            return False
    _probe()
    # Redrawing in place needs ANSI; without it, use the numbered prompts.
    return _COLOR


# ------------------------------------------------------------------------------
#  Styling
# ------------------------------------------------------------------------------

def _c(code: str, text: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if _COLOR else text


def _dim(t: str) -> str:
    return _c("2", t)


def _bold(t: str) -> str:
    return _c("1", t)


def _cyan(t: str) -> str:
    return _c("96", t)


def _green(t: str) -> str:
    return _c("92", t)


def _yellow(t: str) -> str:
    return _c("93", t)


def _red(t: str) -> str:
    return _c("91", t)


def _grey(t: str) -> str:
    return _c("90", t)


def _glyph(fancy: str, plain: str) -> str:
    return fancy if _UNICODE else plain


# For other modules colouring their own log lines.
bold, cyan, green, yellow, red, dim = _bold, _cyan, _green, _yellow, _red, _grey
glyph = _glyph  # glyph("fancy", "plain"): picks what the console can encode


def _term_width() -> int:
    return max(40, shutil.get_terminal_size((100, 30)).columns)


def _term_height() -> int:
    return max(10, shutil.get_terminal_size((100, 30)).lines)


def heading(text: str) -> None:
    """Print a section heading in the widgets' style."""
    if not _COLOR:
        _probe()
    width = min(60, _term_width() - 2)
    bar = _glyph("─", "-") * max(0, width - len(text) - 1)
    print(f"\n{_bold(_cyan(text))} {_grey(bar)}")


def _hide_cursor() -> None:
    if _COLOR:
        sys.stdout.write("\x1b[?25l")


def _show_cursor() -> None:
    if _COLOR:
        sys.stdout.write("\x1b[?25h")
        sys.stdout.flush()


# ------------------------------------------------------------------------------
#  Key input
# ------------------------------------------------------------------------------

_WIN_SPECIAL = {
    "H": "up", "P": "down", "K": "left", "M": "right",
    "G": "home", "O": "end", "I": "pgup", "Q": "pgdn",
}
_VT_SPECIAL = {
    "A": "up", "B": "down", "C": "right", "D": "left",
    "H": "home", "F": "end", "5~": "pgup", "6~": "pgdn",
}


def _read_key() -> str:
    """Block for one keypress: ``up``/``down``/``left``/``right``/``home``/
    ``end``/``pgup``/``pgdn``/``enter``/``esc``/``space``/``backspace``/
    ``tab``, or the character typed. Ctrl-C raises ``KeyboardInterrupt``."""
    if os.name == "nt":
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            return _WIN_SPECIAL.get(msvcrt.getwch(), "")
    else:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                # Escape alone, or the start of a CSI sequence.
                if not select.select([sys.stdin], [], [], 0.05)[0]:
                    return "esc"
                if sys.stdin.read(1) != "[":
                    return "esc"
                body = ""
                while select.select([sys.stdin], [], [], 0.05)[0]:
                    body += sys.stdin.read(1)
                    if body[-1].isalpha() or body[-1] == "~":
                        break
                return _VT_SPECIAL.get(body, "")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    if ch == "\x03":
        raise KeyboardInterrupt
    if ch in ("\r", "\n"):
        return "enter"
    if ch == "\x1b":
        return "esc"
    if ch == " ":
        return "space"
    if ch in ("\x08", "\x7f"):
        return "backspace"
    if ch == "\t":
        return "tab"
    return ch


# ------------------------------------------------------------------------------
#  Frame rendering
# ------------------------------------------------------------------------------

def _visible_len(s: str) -> int:
    out, i = 0, 0
    while i < len(s):
        if s[i] == "\x1b":
            j = s.find("m", i)
            if j == -1:
                break
            i = j + 1
            continue
        out += 1
        i += 1
    return out


def _clip(s: str, width: int) -> str:
    """Truncate to ``width`` visible characters, keeping escape codes intact."""
    if _visible_len(s) <= width:
        return s
    out, seen, i = [], 0, 0
    while i < len(s) and seen < width - 1:
        if s[i] == "\x1b":
            j = s.find("m", i)
            if j == -1:
                break
            out.append(s[i:j + 1])
            i = j + 1
            continue
        out.append(s[i])
        seen += 1
        i += 1
    out.append(_glyph("…", "~"))
    if _COLOR:
        out.append("\x1b[0m")
    return "".join(out)


class _Frame:
    """Redraws a block of lines in place, without clearing the scrollback."""

    def __init__(self, stream=None) -> None:
        self._lines = 0
        self._stream = stream

    @property
    def _out_stream(self):
        return self._stream if self._stream is not None else sys.stdout

    def clear(self) -> None:
        """Blank the block and park the cursor at its first line, so a caller
        can print there and redraw underneath."""
        if not self._lines or not _COLOR:
            self._lines = 0
            return
        n = self._lines
        up = f"\x1b[{n}A"
        self._out_stream.write(up + "\x1b[2K\n" * n + up)
        self._out_stream.flush()
        self._lines = 0

    def draw(self, lines: Sequence[str]) -> None:
        out: List[str] = []
        if self._lines and _COLOR:
            out.append(f"\x1b[{self._lines}A")
        width = _term_width() - 1
        for line in lines:
            if _COLOR:
                out.append("\x1b[2K")
            out.append(_clip(line, width))
            out.append("\n")
        # Wipe leftovers from a taller previous frame, then come back up.
        extra = max(0, self._lines - len(lines))
        for _ in range(extra):
            out.append("\x1b[2K\n" if _COLOR else "\n")
        if extra and _COLOR:
            out.append(f"\x1b[{extra}A")
        self._out_stream.write("".join(out))
        self._out_stream.flush()
        self._lines = len(lines)


def _hint(pairs: Sequence[Sequence[str]]) -> List[str]:
    """Key/meaning hints, wrapped to the terminal width."""
    sep_txt = _glyph("  ·  ", "  |  ")
    sep = _grey(sep_txt)
    width = _term_width() - 2
    lines: List[str] = []
    cur, cur_len = "", 0
    for k, v in pairs:
        chunk = f"{_yellow(k)} {_grey(v)}"
        size = len(k) + 1 + len(v)
        if cur and cur_len + len(sep_txt) + size > width:
            lines.append(cur)
            cur, cur_len = chunk, size
            continue
        cur = f"{cur}{sep}{chunk}" if cur else chunk
        cur_len += size + (len(sep_txt) if cur_len else 0)
    if cur:
        lines.append(cur)
    return lines


# ------------------------------------------------------------------------------
#  Widgets
# ------------------------------------------------------------------------------

def _labels_for(items: Sequence[Any],
                label_fn: Optional[Callable[[Any], str]]) -> List[str]:
    if label_fn is None:
        return [str(i) for i in items]
    return [label_fn(i) for i in items]

def _window(cursor: int, count: int, rows: int) -> int:
    """First visible row index for a viewport of ``rows`` around ``cursor``."""
    return max(0, min(cursor - rows // 2, count - rows))


def select_many(
    items: Sequence[Any],
    title: str,
    *,
    label_fn: Optional[Callable[[Any], str]] = None,
    short_fn: Optional[Callable[[Any], str]] = None,
    preselected: Optional[Sequence[int]] = None,
    noun: str = "item",
) -> Optional[List[Any]]:
    """Checkbox picker; everything starts ticked unless ``preselected`` says
    otherwise, so Enter straight away runs them all. Returns the chosen items
    in list order, or None if cancelled. ``short_fn`` names an item on the
    summary line left behind, for when row labels are long."""
    if not items:
        return None
    labels = _labels_for(items, label_fn)
    shorts = _labels_for(items, short_fn) if short_fn else labels
    if not supports_tui():
        return _fallback_many(items, labels, title, noun)

    chosen = set(range(len(items)) if preselected is None else preselected)
    cursor = 0
    error = ""
    frame = _Frame()
    num_w = len(str(len(items)))

    _hide_cursor()
    try:
        while True:
            cursor = max(0, min(cursor, len(items) - 1))
            rows = max(3, min(len(items), _term_height() - 7))
            start = _window(cursor, len(items), rows)

            lines = [
                f"{_bold(_cyan(title))}  "
                f"{_grey(f'({len(chosen)}/{len(items)} selected)')}",
                "",
            ]
            for idx in range(start, min(start + rows, len(items))):
                is_cur = idx == cursor
                box = (_green(_glyph("[✓]", "[x]")) if idx in chosen
                       else _grey("[ ]"))
                name = labels[idx]
                arrow = _cyan(_glyph("❯", ">")) if is_cur else " "
                num = _grey(f"{idx + 1:>{num_w}}.")
                lines.append(f" {arrow} {box} {num} "
                             f"{_bold(name) if is_cur else name}")
            if len(items) > rows:
                lines.append(_grey(
                    f"   {start + 1}-{min(start + rows, len(items))}"
                    f" of {len(items)}"))

            lines.append("")
            lines.extend(_hint([
                (_glyph("↑↓", "up/dn"), "move"),
                ("space", "toggle"),
                ("a", "all"),
                ("n", "none"),
                ("enter", "run"),
                ("esc", "cancel"),
            ]))
            if error:
                lines.append(_red(f"   {error}"))
            frame.draw(lines)
            error = ""

            key = _read_key()

            if key == "up":
                cursor = (cursor - 1) % len(items)
            elif key == "down":
                cursor = (cursor + 1) % len(items)
            elif key == "pgup":
                cursor -= rows
            elif key == "pgdn":
                cursor += rows
            elif key == "home":
                cursor = 0
            elif key == "end":
                cursor = len(items) - 1
            elif key == "space":
                chosen.symmetric_difference_update({cursor})
                cursor = min(cursor + 1, len(items) - 1)
            elif key in ("a", "A"):
                chosen = set(range(len(items)))
            elif key in ("n", "N"):
                chosen.clear()
            elif key == "enter":
                if not chosen:
                    error = f"select at least one {noun}"
                    continue
                order = sorted(chosen)
                frame.draw(_summary_lines(title, shorts, order,
                                          noun, len(items)))
                return [items[i] for i in order]
            elif key in ("esc", "q", "Q"):
                frame.draw([_grey(f"{title}: cancelled")])
                return None
    except KeyboardInterrupt:
        frame.draw([_grey(f"{title}: cancelled")])
        return None
    finally:
        _show_cursor()

def _summary_lines(title: str, labels: Sequence[str], chosen: Sequence[int],
                   noun: str, count: int) -> List[str]:
    """One line naming what was picked; two when that needs spelling out."""
    plural = "" if len(chosen) == 1 else "s"
    head = f"{_bold(_cyan(title))}  "
    if len(chosen) == count:
        return [head + _green(f"all {count} {noun}{plural}")]
    names = [labels[i] for i in chosen]
    shown = ", ".join(names[:6])
    if len(names) > 6:
        shown += _grey(f" (+{len(names) - 6} more)")
    if _visible_len(head + shown) <= _term_width() - 2:
        return [head + _green(shown)]
    return [
        head + _green(f"{len(names)} {noun}{plural}"),
        f"   {shown}",
    ]


def select_one(
    items: Sequence[Any],
    title: str,
    *,
    label_fn: Optional[Callable[[Any], str]] = None,
    extra: Optional[str] = None,
    noun: str = "item",
) -> Optional[Any]:
    """Single-choice picker. ``extra`` appends a trailing pseudo-entry (e.g.
    "Paste a path..."); choosing it returns that same string."""
    labels = _labels_for(items, label_fn)
    entries: List[Any] = list(items)
    if extra is not None:
        entries.append(extra)
        labels = labels + [extra]
    if not entries:
        return None
    if not supports_tui():
        return _fallback_one(entries, labels, title, noun)

    cursor = 0
    frame = _Frame()
    num_w = len(str(len(entries)))

    _hide_cursor()
    try:
        while True:
            cursor = max(0, min(cursor, len(entries) - 1))
            rows = max(3, min(len(entries), _term_height() - 7))
            start = _window(cursor, len(entries), rows)

            lines = [_bold(_cyan(title)), ""]
            for idx in range(start, min(start + rows, len(entries))):
                is_cur = idx == cursor
                name = labels[idx]
                if extra is not None and idx == len(entries) - 1:
                    name = _yellow(name)
                arrow = _cyan(_glyph("❯", ">")) if is_cur else " "
                num = _grey(f"{idx + 1:>{num_w}}.")
                lines.append(f" {arrow} {num} "
                             f"{_bold(name) if is_cur else name}")
            if len(entries) > rows:
                lines.append(_grey(
                    f"   {start + 1}-{min(start + rows, len(entries))}"
                    f" of {len(entries)}"))

            lines.append("")
            lines.extend(_hint([
                (_glyph("↑↓", "up/dn"), "move"),
                ("enter", "select"),
                ("esc", "cancel"),
            ]))
            frame.draw(lines)

            key = _read_key()
            if key == "up":
                cursor = (cursor - 1) % len(entries)
            elif key == "down":
                cursor = (cursor + 1) % len(entries)
            elif key == "pgup":
                cursor -= rows
            elif key == "pgdn":
                cursor += rows
            elif key == "home":
                cursor = 0
            elif key == "end":
                cursor = len(entries) - 1
            elif key == "enter":
                frame.draw([f"{_bold(_cyan(title))}  "
                            f"{_green(labels[cursor])}"])
                return entries[cursor]
            elif key in ("esc", "q", "Q"):
                frame.draw([_grey(f"{title}: cancelled")])
                return None
    except KeyboardInterrupt:
        frame.draw([_grey(f"{title}: cancelled")])
        return None
    finally:
        _show_cursor()


def confirm(question: str, default: bool = False) -> Optional[bool]:
    """Yes/no toggle driven by arrows or y/n.  None if cancelled."""
    if not supports_tui():
        return _fallback_confirm(question, default)

    value = default
    frame = _Frame()
    _hide_cursor()
    try:
        while True:
            yes, no = " Yes ", " No "
            if _COLOR:
                yes = f"\x1b[7;92m{yes}\x1b[0m" if value else _grey(yes)
                no = _grey(no) if value else f"\x1b[7;92m{no}\x1b[0m"
            else:
                yes = f"[{yes}]" if value else f" {yes} "
                no = f" {no} " if value else f"[{no}]"
            frame.draw(
                [f"{_bold(_cyan(question))}   {yes}  {no}"]
                + _hint([(_glyph("←→", "left/right"), "switch"),
                         ("y/n", "pick"),
                         ("enter", "confirm")])
            )
            key = _read_key()
            if key in ("left", "right", "tab", "space", "up", "down"):
                value = not value
            elif key in ("y", "Y"):
                value = True
            elif key in ("n", "N"):
                value = False
            elif key == "enter":
                frame.draw([f"{_bold(_cyan(question))}   "
                            f"{_green('yes' if value else 'no')}"])
                return value
            elif key in ("esc", "q", "Q"):
                frame.draw([_grey(f"{question}: cancelled")])
                return None
    except KeyboardInterrupt:
        frame.draw([_grey(f"{question}: cancelled")])
        return None
    finally:
        _show_cursor()


# ------------------------------------------------------------------------------
#  Progress bars
# ------------------------------------------------------------------------------

class _Task:
    __slots__ = ("key", "label", "total", "done", "status", "started", "ended",
                 "ok", "last_report")

    def __init__(self, key: Any, label: str, total: int) -> None:
        self.key = key
        self.label = label
        self.total = max(1, total)
        self.done = 0
        self.status = ""
        self.started = time.monotonic()
        self.ended: Optional[float] = None
        self.ok = True
        self.last_report = -1.0

    @property
    def frac(self) -> float:
        return min(1.0, self.done / self.total)

    @property
    def elapsed(self) -> float:
        return (self.ended or time.monotonic()) - self.started


# Label column bounds for a Progress block (visible characters).
_LABEL_W_MIN = 8
_LABEL_W_MAX = 24


def _fmt_secs(seconds: float) -> str:
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}:{(s % 3600) // 60:02}:{s % 60:02}"
    return f"{s // 60}:{s % 60:02}"


class Progress:
    """A block of live progress bars pinned to the bottom of the terminal.

    One bar per running task plus a total line, redrawn in place; ``log()``
    prints above the block, and a finished task leaves it for one summary
    line. Outside a terminal this degrades to occasional one-line updates.
    """

    def __init__(self, title: str, total: int = 0, unit: str = "item",
                 step_unit: str = "step", refresh: float = 0.15,
                 stream=None, label_width: int = 0) -> None:
        self.title = title
        self.total = total
        self.unit = unit
        # Names what one unit counts; the count itself is always shown.
        self.step_unit = step_unit
        self._live = supports_tui()
        # Callers redirect sys.stdout (and sometimes fd 1) into log(), so
        # the block needs the stream captured here to reach the terminal.
        self._out = stream if stream is not None else sys.stdout
        self._frame = _Frame(self._out)
        self._lock = threading.RLock()
        self._tasks: "dict[Any, _Task]" = {}
        self._order: List[Any] = []
        self._finished = 0
        self._failed = 0
        self._units = 0
        self._unit_total = 0
        self._started_at = time.monotonic()
        self._closed = False
        self._refresh = refresh
        self._ticker: Optional[threading.Thread] = None
        # Column widths only ever grow, per block. Sizing them to the tasks
        # that happen to be live makes the whole bar block slide sideways
        # every time one starts or finishes; a high-water mark lets the
        # layout settle after the first few tasks and then stay put.
        # Callers that know every label up front pass label_width, so the
        # column is right from the first frame instead of widening once.
        self._label_w = max(_LABEL_W_MIN, min(_LABEL_W_MAX, label_width))
        self._count_w = 0

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "Progress":
        if self._live:
            _hide_cursor()
            self._ticker = threading.Thread(target=self._tick, daemon=True)
            self._ticker.start()
        else:
            print(f"=== {self.title} ===", file=self._out, flush=True)
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if self._ticker is not None:
            self._ticker.join(timeout=1.0)
        with self._lock:
            if self._live:
                self._frame.clear()
                _show_cursor()
            total = _fmt_secs(time.monotonic() - self._started_at)
            # Several items: count them. One task: count its own units.
            done = (f"{self._finished}/{self.total}" if self.total
                    else f"{self._units}/{self._unit_total}")
            tail = f" ({self._failed} failed)" if self._failed else ""
            word = self.unit if self.total else self.step_unit
            print(f"{_bold(_cyan(self.title))}  "
                  f"{_green(done + ' ' + word + 's')}{_red(tail)}  "
                  f"{_grey('in ' + total)}", file=self._out, flush=True)

    @property
    def live(self) -> bool:
        """True when bars are being drawn rather than piped away."""
        return self._live

    def passthrough(self, line: str) -> None:
        """Write a line straight to the real stdout, bars or not."""
        print(line, file=self._out, flush=True)

    def _tick(self) -> None:
        failures = 0
        while True:
            with self._lock:
                if self._closed:
                    return
                try:
                    self._render()
                    failures = 0
                except Exception:
                    # A broken stream must not take the thread down with a
                    # traceback across the bars; give up drawing after a
                    # few and let the run finish quietly.
                    failures += 1
                    if failures >= 3:
                        self._live = False
                        return
            time.sleep(self._refresh)

    # -- task API ----------------------------------------------------------

    def start(self, key: Any, label: str, total: int) -> None:
        with self._lock:
            self._tasks[key] = _Task(key, label, total)
            self._order.append(key)
            self._grow_columns(self._tasks[key])
            if not self._live:
                print(f"  [start] {label}", file=self._out, flush=True)
            else:
                self._render()

    def set_total(self, key: Any, total: int) -> None:
        with self._lock:
            task = self._tasks.get(key)
            if task is not None and total > 0:
                task.total = total

    def update(self, key: Any, done: Optional[int] = None, advance: int = 0,
               status: Optional[str] = None) -> None:
        with self._lock:
            task = self._tasks.get(key)
            if task is None:
                return
            if done is not None:
                task.done = max(task.done, done)
            if advance:
                task.done += advance
            if task.done > task.total:
                # More outputs than predicted: believe the files.
                task.total = task.done
            if status is not None:
                task.status = status
            if self._live:
                self._render()
            else:
                self._report_plain(task)

    def finish(self, key: Any, ok: bool = True, note: str = "") -> None:
        with self._lock:
            task = self._tasks.pop(key, None)
            if key in self._order:
                self._order.remove(key)
            self._finished += 1
            if not ok:
                self._failed += 1
            if task is None:
                return
            task.ended = time.monotonic()
            self._units += task.done
            self._unit_total += task.total
            mark = (_green(_glyph("✓", "+")) if ok
                    else _red(_glyph("✗", "x")))
            detail = (f"{task.done}/{task.total} {self.step_unit}s"
                      if ok else note)
            self.log(f"{mark} {_bold(task.label)}  {_grey(detail)}  "
                     f"{_grey(_fmt_secs(task.elapsed))}")

    # -- output ------------------------------------------------------------

    def log(self, message: str) -> None:
        """Print above the bars."""
        with self._lock:
            if self._live:
                self._frame.clear()
                print(message, file=self._out, flush=True)
                self._render()
            else:
                print(message, file=self._out, flush=True)

    def _report_plain(self, task: _Task) -> None:
        """Non-TTY: one line per 10% of a task, at most."""
        step = int(task.frac * 10)
        if step <= task.last_report:
            return
        task.last_report = step
        print(f"  [{task.label}] {int(task.frac * 100):>3}% "
              f"({task.done}/{task.total}) {task.status}".rstrip(),
              file=self._out, flush=True)

    # -- rendering ---------------------------------------------------------

    def _counts(self, done: int, total: int) -> str:
        """``done/total`` in a column that never changes width: ``done`` is
        padded to ``total``'s digits, the field to the block's high-water."""
        return f"{done:>{len(str(total))}}/{total}".rjust(self._count_w)

    def _grow_columns(self, task: "_Task") -> None:
        self._label_w = max(self._label_w,
                            min(_LABEL_W_MAX, len(task.label)))
        self._count_w = max(self._count_w,
                            len(f"{task.total}/{task.total}"))

    def _bar(self, frac: float, width: int, color=None) -> str:
        filled = int(round(frac * width))
        full, empty = _glyph("█", "#"), _glyph("░", ".")
        paint = color or _green
        return paint(full * filled) + _grey(empty * (width - filled))

    def _render(self) -> None:
        if not self._live:
            return
        width = _term_width() - 1
        for task in list(self._tasks.values()):
            # update() can push total past its prediction; widen for it.
            self._grow_columns(task)
        if self.total:
            self._count_w = max(self._count_w,
                                len(f"{self.total}/{self.total}"))
        label_w = self._label_w
        # "  " + label + " " + bar + " 100% " + counts + "  " + status
        bar_w = max(10, min(34, width - label_w - self._count_w - 12))

        lines = []
        header = f"{_bold(_cyan(self.title))}"
        if self.total:
            header += _grey(f"  {self._finished}/{self.total} {self.unit}s"
                            f" done")
        header += _grey(f"  {_fmt_secs(time.monotonic() - self._started_at)}")
        lines.append(header)

        for key in self._order:
            task = self._tasks.get(key)
            if task is None:
                continue
            label = task.label[:label_w].ljust(label_w)
            pct = f"{int(task.frac * 100):>3}%"
            line = (f"  {_cyan(label)} {self._bar(task.frac, bar_w)} "
                    f"{_bold(pct)} {_grey(self._counts(task.done, task.total))}")
            if task.status:
                line += _grey(f"  {task.status}")
            lines.append(line)

        if self.total:
            frac = self._finished / self.total
            label = "total".ljust(label_w)
            lines.append(
                f"  {_yellow(label)} "
                f"{self._bar(frac, bar_w, _yellow)} "
                f"{_bold(f'{int(frac * 100):>3}%')} "
                f"{_grey(self._counts(self._finished, self.total))}")
        self._frame.draw(lines)


# ------------------------------------------------------------------------------
#  Non-interactive fallbacks (piped stdin, subprocess workers)
# ------------------------------------------------------------------------------

def _fallback_many(items: Sequence[Any], labels: Sequence[str],
                   title: str, noun: str) -> Optional[List[Any]]:
    print(f"{title}:")
    print(f"    0. All {noun}s")
    for i, name in enumerate(labels, 1):
        print(f"  {i:3}. {name}")
    while True:
        raw = input(f"\nSelect {noun} "
                    f"(0 for all, numbers or names): ").strip()
        if raw == "0":
            return list(items)
        picked = _parse_tokens(raw, items, labels)
        if picked:
            return picked
        print("  Invalid selection, try again.")


def _fallback_one(entries: Sequence[Any], labels: Sequence[str],
                  title: str, noun: str) -> Optional[Any]:
    print(f"{title}:")
    for i, name in enumerate(labels, 1):
        print(f"  {i:3}. {name}")
    while True:
        raw = input(f"\nSelect {noun} (number or name): ").strip()
        picked = _parse_tokens(raw, entries, labels)
        if picked and len(picked) == 1:
            return picked[0]
        print("  Invalid selection, try again.")


def _parse_tokens(raw: str, items: Sequence[Any],
                  labels: Sequence[str]) -> Optional[List[Any]]:
    """Map a whitespace/comma separated list of numbers or names to items."""
    if not raw:
        return None
    lower = [lb.lower() for lb in labels]
    out: List[int] = []
    for tok in raw.replace(",", " ").split():
        if tok.isdigit():
            idx = int(tok) - 1
            if not (0 <= idx < len(items)):
                return None
            out.append(idx)
        elif tok.lower() in lower:
            out.append(lower.index(tok.lower()))
        else:
            return None
    return [items[i] for i in sorted(set(out))]


def _fallback_confirm(question: str, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"{question} {suffix}: ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  please answer y or n")
