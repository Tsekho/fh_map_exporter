"""
progress.py
===========
How the pipeline scripts measure work for the progress bars in ``utils.tui``.

``FileTracker`` counts output files on disk (step 4); ``PhaseTracker``
counts log-line checkpoints (steps 2 and 3, whose only file appears at the
end). Both are driven by ``run_serial()`` here and by
``utils.parallel.run_parallel_subprocesses()``: one bar per running item,
worker chatter swallowed, warnings printed above the bars.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple,
)

from utils import tui

__all__ = [
    "FileTracker",
    "PhaseTracker",
    "Tracker",
    "is_noteworthy",
    "run_serial",
    "capture_output",
    "console_stream",
    "CREDIT_PREFIX",
    "WORKER_ENV",
    "announce_credit",
]


# Bars are sized for a full set of outputs. A worker that cannot write some
# of them (a region with no glaciers) says so on this line and they count as
# done, so every item's total matches.
CREDIT_PREFIX = "##FH-CREDIT "

# Set by the parallel runner in every child. A worker must not draw bars or
# swallow its own log lines: the parent reads them to move the real ones.
WORKER_ENV = "FH_WORKER"

_NOTEWORTHY = re.compile(
    r"\b(WARN|WARNING|ERROR|Traceback|Exception|FAILED)\b", re.IGNORECASE
)

# Matches _NOTEWORTHY but says nothing about the result: bpy's allocator
# prints a "not freed memory blocks" tally while exiting successfully.
_BENIGN = re.compile(
    r"(not freed memory blocks|total unfreed memory)", re.IGNORECASE
)

POLL_INTERVAL = 0.25


def announce_credit(count: int) -> None:
    """Say that ``count`` of this item's expected outputs don't apply here,
    so the bar can credit them instead of stalling short of full."""
    if count > 0:
        print(f"{CREDIT_PREFIX}{count}", flush=True)


def is_noteworthy(line: str) -> bool:
    """True for lines worth showing above the bars."""
    return bool(_NOTEWORTHY.search(line)) and not _BENIGN.search(line)


# ------------------------------------------------------------------------------
#  Trackers
# ------------------------------------------------------------------------------

class Tracker:
    """Turns a worker's activity into (done, status) for one item."""

    #: Whether the driver should run a polling thread for this tracker.
    polls = False

    def total(self, item: Any) -> int:
        return 1

    def begin(self, item: Any) -> None:
        pass

    def poll(self, item: Any) -> Optional[int]:
        return None

    def on_line(self, item: Any, line: str) -> Tuple[Optional[int],
                                                     Optional[str]]:
        return None, None

    def credit(self, item: Any, count: int) -> None:
        """Count ``count`` outputs that will never be written as done."""


class FileTracker(Tracker):
    """Counts output files written since the item started.

    ``candidates`` returns every path the item could write, re-evaluated on
    each poll; ``expected`` is the full set's size, the same for every item.
    An item that legitimately writes fewer calls ``announce_credit()``.
    """

    polls = True

    def __init__(
        self,
        candidates: Callable[[Any], Iterable[Path]],
        expected: Callable[[Any], int],
        status: Optional[Callable[[Path], str]] = None,
        status_re: Optional[str] = None,
    ) -> None:
        self._candidates = candidates
        self._expected = expected
        self._status = status or (lambda p: p.parent.name)
        # First group names the step the worker is on, for the bar's status.
        self._status_re = re.compile(status_re) if status_re else None
        self._since: Dict[Any, float] = {}
        self._credit: Dict[Any, int] = {}
        self._last_file: Dict[Any, str] = {}
        self._last_step: Dict[Any, str] = {}

    def total(self, item: Any) -> int:
        return max(1, self._expected(item))

    def begin(self, item: Any) -> None:
        # A second of slack: filesystem timestamps are coarse.
        self._since[item] = time.time() - 1.0
        self._credit[item] = 0

    def credit(self, item: Any, count: int) -> None:
        self._credit[item] = max(self._credit.get(item, 0), count)

    def poll(self, item: Any) -> Optional[int]:
        since = self._since.get(item)
        if since is None:
            return None
        done = 0
        newest: Optional[Tuple[float, Path]] = None
        for path in self._candidates(item):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime >= since:
                done += 1
                if newest is None or mtime > newest[0]:
                    newest = (mtime, path)
        if newest is not None:
            self._last_file[item] = self._status(newest[1])
        return min(self.total(item), done + self._credit.get(item, 0))

    def status(self, item: Any) -> str:
        """What the worker is doing: the step it announced if it announces
        any, else where the last file landed."""
        return (self._last_step.get(item)
                or self._last_file.get(item, ""))

    def on_line(self, item: Any, line: str) -> Tuple[Optional[int],
                                                     Optional[str]]:
        if self._status_re is not None:
            m = self._status_re.search(line)
            if m:
                self._last_step[item] = m.group(1).strip()
                return None, self._last_step[item]
        return None, None


class PhaseTracker(Tracker):
    """Counts checkpoints: how many known log milestones a worker has passed.

    ``phases`` is an ordered list of ``(pattern, label)``, or
    ``(pattern, label, repeat)`` when one line recurs (the six neighbor
    regions) and each occurrence is its own checkpoint. Monotonic: a
    duplicate or out-of-order line can never move the bar backwards.
    """

    def __init__(self, phases: Sequence[Sequence]) -> None:
        self._phases: List[Tuple[Any, str, int]] = []
        self._offsets: List[int] = []
        offset = 0
        for p in phases:
            repeat = p[2] if len(p) > 2 else 1
            self._phases.append((re.compile(p[0]), p[1], repeat))
            self._offsets.append(offset)
            offset += repeat
        self._total = max(1, offset)
        self._state: Dict[Any, Dict[str, Any]] = {}

    def total(self, item: Any) -> int:
        return self._total

    def begin(self, item: Any) -> None:
        self._state[item] = {"done": 0, "hits": {}, "label": ""}

    def _st(self, item: Any) -> Dict[str, Any]:
        return self._state.setdefault(
            item, {"done": 0, "hits": {}, "label": ""})

    def on_line(self, item: Any, line: str) -> Tuple[Optional[int],
                                                     Optional[str]]:
        st = self._st(item)
        for i, (pattern, label, repeat) in enumerate(self._phases):
            if not pattern.search(line):
                continue
            hit = min(repeat, st["hits"].get(i, 0) + 1)
            st["hits"][i] = hit
            reached = self._offsets[i] + hit
            if reached >= st["done"]:
                st["done"] = reached
                st["label"] = (label if repeat == 1
                               else f"{label} {hit}/{repeat}")
            return st["done"], st["label"]
        return None, None


# ------------------------------------------------------------------------------
#  Driving a worker that prints to stdout
# ------------------------------------------------------------------------------

class LineSink(io.TextIOBase):
    """Stands in for ``sys.stdout`` while an item is being worked on.

    Complete lines go to the tracker (to move the bar) and, when they look
    like a warning or the caller asked for everything, to the display's log
    area above the bars.
    """

    def __init__(self, disp: tui.Progress, item: Any, tracker: Tracker,
                 label: str, verbose: bool) -> None:
        self._disp = disp
        self._item = item
        self._tracker = tracker
        self._label = label
        self._verbose = verbose
        self._buf = ""

    def write(self, text: str) -> int:  # type: ignore[override]
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.handle(line)
        return len(text)

    def handle(self, line: str) -> None:
        stripped = line.rstrip()
        if not stripped:
            return
        if stripped.startswith(CREDIT_PREFIX):
            try:
                self._tracker.credit(self._item,
                                     int(stripped[len(CREDIT_PREFIX):]))
            except ValueError:
                pass
            if not self._disp.live:
                # A worker piping to a parent: pass the credit on.
                self._disp.passthrough(stripped)
            return
        done, status = self._tracker.on_line(self._item, stripped)
        if done is not None or status is not None:
            self._disp.update(self._item, done=done, status=status)
        if self._verbose or is_noteworthy(stripped):
            self._disp.log(f"  {tui.dim(self._label + ' |')} {stripped}")

    def flush(self) -> None:  # type: ignore[override]
        pass

    def close_line(self) -> None:
        if self._buf.strip():
            self.handle(self._buf)
        self._buf = ""

    def writable(self) -> bool:  # type: ignore[override]
        return True


class Poller:
    """Background thread that asks a tracker where an item has got to."""

    def __init__(self, disp: tui.Progress, tracker: Tracker) -> None:
        self._disp = disp
        self._tracker = tracker
        self._items: List[Any] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def add(self, item: Any) -> None:
        with self._lock:
            self._items.append(item)

    def remove(self, item: Any) -> None:
        self.sample(item)
        with self._lock:
            if item in self._items:
                self._items.remove(item)

    def sample(self, item: Any) -> None:
        done = self._tracker.poll(item)
        if done is None:
            return
        status = getattr(self._tracker, "status", lambda _i: "")(item)
        self._disp.update(item, done=done, status=status)

    def _run(self) -> None:
        while not self._stop.wait(POLL_INTERVAL):
            with self._lock:
                items = list(self._items)
            for item in items:
                self.sample(item)

    def __enter__(self) -> "Poller":
        if self._tracker.polls:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class NativeCapture:
    """Routes OS-level stdout/stderr into a sink for the duration of a block.

    Blender writes straight to file descriptor 1, under Python's
    ``sys.stdout``; uncaught, those lines scramble the progress block. Best
    effort -- if the descriptor dance fails, capture is skipped.
    """

    def __init__(self, sink: LineSink, fds: Sequence[int] = (1, 2)) -> None:
        self._sink = sink
        self._fds = list(fds)
        self._saved: Dict[int, int] = {}
        self._reader: Optional[threading.Thread] = None
        self._read_fd: Optional[int] = None

    def __enter__(self) -> "NativeCapture":
        try:
            for stream in (sys.__stdout__, sys.__stderr__):
                if stream is not None:
                    stream.flush()
            read_fd, write_fd = os.pipe()
            for fd in self._fds:
                self._saved[fd] = os.dup(fd)
                os.dup2(write_fd, fd)
            os.close(write_fd)
            self._read_fd = read_fd
            self._reader = threading.Thread(target=self._pump, daemon=True)
            self._reader.start()
        except Exception:
            self._restore()
        return self

    def _pump(self) -> None:
        assert self._read_fd is not None
        with os.fdopen(self._read_fd, "rb", 0) as pipe:
            for chunk in iter(pipe.readline, b""):
                self._sink.write(chunk.decode("utf-8", "replace"))

    def _restore(self) -> None:
        for fd, saved in self._saved.items():
            try:
                os.dup2(saved, fd)
                os.close(saved)
            except OSError:
                pass
        self._saved.clear()

    def __exit__(self, *exc) -> None:
        try:
            for stream in (sys.__stdout__, sys.__stderr__):
                if stream is not None:
                    stream.flush()
        except Exception:
            pass
        # Dropping the last write end is what lets the reader see EOF.
        self._restore()
        if self._reader is not None:
            self._reader.join(timeout=2.0)


@contextlib.contextmanager
def capture_output(disp: tui.Progress, key: Any, tracker: Tracker,
                   label: str, verbose: bool = False):
    """Route everything a block prints into the progress display, including
    writes straight to the stdout/stderr descriptors (Blender, OpenCV)."""
    sink = LineSink(disp, key, tracker, label, verbose)
    real_stdout = sys.stdout
    sys.stdout = sink
    try:
        with NativeCapture(sink):
            yield sink
    finally:
        sys.stdout = real_stdout
        sink.close_line()


def run_serial(
    items: Sequence[Any],
    work_fn: Callable[[Any], bool],
    *,
    title: str,
    tracker: Tracker,
    label_fn: Callable[[Any], str] = str,
    unit: str = "item",
    step_unit: str = "step",
    verbose: bool = False,
) -> List[Any]:
    """Run ``work_fn`` over ``items`` in this process, one bar at a time.

    ``work_fn`` returns False (or raises) to mark the item failed; its
    ``print()`` output is captured.  Returns the failed items.
    """
    if os.environ.get(WORKER_ENV):
        # The parent owns the bars and reads our log lines to move them.
        return _run_plain(items, work_fn, label_fn)

    failed: List[Any] = []
    # Survives both the sys.stdout swap and the fd-level capture below.
    console = console_stream()
    with tui.Progress(title, total=len(items), unit=unit,
                      step_unit=step_unit, stream=console) as disp:
        with Poller(disp, tracker) as poller:
            for item in items:
                label = label_fn(item)
                tracker.begin(item)
                disp.start(item, label, tracker.total(item))
                poller.add(item)
                note = ""
                try:
                    with capture_output(disp, item, tracker, label, verbose):
                        ok = bool(work_fn(item))
                except Exception as exc:
                    ok = False
                    note = f"{type(exc).__name__}: {exc}"
                    disp.log(tui.red(f"  {label}: {note}"))
                    disp.log(tui.dim(traceback.format_exc().rstrip()))
                finally:
                    poller.remove(item)
                disp.finish(item, ok, note=note or "failed")
                if not ok:
                    failed.append(item)
    if console is not sys.stdout:
        console.flush()
        console.close()
    return failed


def _run_plain(
    items: Sequence[Any],
    work_fn: Callable[[Any], bool],
    label_fn: Callable[[Any], str],
) -> List[Any]:
    """run_serial() for a worker process: plain output, no capture."""
    failed: List[Any] = []
    for item in items:
        try:
            ok = bool(work_fn(item))
        except Exception:
            traceback.print_exc()
            ok = False
        if not ok:
            print(f"ERROR: {label_fn(item)} failed", flush=True)
            failed.append(item)
    return failed


def console_stream():
    """A writable duplicate of the current stdout that keeps working while
    fd 1 is redirected.

    Anything drawing to the terminal around ``capture_output()`` -- the
    progress bars above all -- must write here, not to ``sys.stdout``: that
    object still points at fd 1, which the capture has aimed at a pipe.
    Falls back to ``sys.stdout`` when it has no descriptor to dup.
    """
    try:
        return os.fdopen(os.dup(sys.stdout.fileno()), "w",
                         buffering=1, encoding="utf-8", errors="replace")
    except Exception:
        return sys.stdout
