"""
parallel.py
========
Subprocess fan-out helper used by the "all" modes of the pipeline scripts.

Each work item is processed in its own child process (so each one gets its
own fresh ``bpy`` state). Child stdout/stderr is streamed back line by line,
tagged with the item's name as log() context, into the caller's
``ScriptTUI`` so interleaved output from concurrent workers lands in the
same scrolling log the rest of the script uses.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from typing import Callable, List, Sequence

from utils.config import REPO_ROOT
from utils.tui import ScriptTUI, strip_tag

# On Windows, a child spawned via subprocess.Popen shares the parent's
# console window unless told otherwise. Our own stdout/stderr redirection
# (below) captures everything the child writes through Python, but native
# code inside bpy/Blender can call the Win32 console API directly, which
# writes straight to that shared console handle -- bypassing the pipe
# *and* the parent ScriptTUI's lock entirely, corrupting the parent's
# display with no relation to our own write sequencing. Giving each child
# its own (hidden) console via CREATE_NO_WINDOW isolates any such direct
# writes away from the parent's terminal.
_POPEN_KWARGS = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


# Match any absolute path that starts with REPO_ROOT (both "\" and "/"
# separators, case-insensitive on Windows) and rewrite it to a repo-
# relative POSIX path so Blender's own log lines (e.g.
# `Read blend: "C:\...\export\blend_spill\X.blend"`) don't blow the
# column budget.
def _make_path_shortener() -> Callable[[str], str]:
    root = str(REPO_ROOT)
    root_variants = {root, root.replace("\\", "/"), root.replace("/", "\\")}
    # Sort longest-first so the most specific prefix wins.
    alts = sorted({re.escape(v) for v in root_variants}, key=len, reverse=True)
    pattern = re.compile("(" + "|".join(alts) + r")([\\/][^\s\"')]*)?",
                         re.IGNORECASE)

    def _sub(m: re.Match) -> str:
        tail = m.group(2) or ""
        tail = tail.lstrip("\\/").replace("\\", "/")
        return tail if tail else "."

    def _shorten(line: str) -> str:
        return pattern.sub(_sub, line)

    return _shorten


_shorten_paths = _make_path_shortener()


def _pump(name: str, proc: subprocess.Popen, tui: ScriptTUI) -> None:
    """Forward a child's stdout into the TUI's log, tagged with ``name``
    (the item this child is processing) as log() context.

    Repo-root absolute paths in the line are rewritten to repo-relative
    POSIX paths so Blender's native log output doesn't dump 100+ char
    absolute paths. The child runs with piped (non-interactive) stdout,
    so its own log()/warn()/error() calls already carry a plain "[LEVEL]"
    tag -- that tag is parsed off and reapplied via ``level=`` instead of
    being forwarded as literal text, so a child's ERROR/WARN line isn't
    wrapped in a second, misleading INFO tag here.
    """
    assert proc.stdout is not None
    for line in proc.stdout:
        level, remainder = strip_tag(_shorten_paths(line).rstrip())
        tui.log(remainder, level=level, context=name)


def run_parallel_subprocesses(
    items: Sequence[str],
    build_cmd: Callable[[str], List[str]],
    workers: int,
    tui: ScriptTUI,
    label_fn: Callable[[str], str] = lambda x: x,
) -> List[str]:
    """
    Run one subprocess per item with at most ``workers`` running at once.

    Each forwarded output line is tagged with ``label_fn(item)`` as the
    log's context (rendered as e.g. "(RegionName)" after the level tag).
    ``tui`` advances by one for every *completed* (not launched) child.

    Returns the list of items whose subprocess exited with a non-zero code.
    """
    if workers < 1:
        workers = 1

    pending: List[str] = list(items)
    active: dict = {}  # Popen -> (item, thread, name)
    failed: List[str] = []

    def _launch(item: str) -> None:
        name = label_fn(item)
        cmd = build_cmd(item)
        tui.log("launching", context=name)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            **_POPEN_KWARGS,
        )
        thread = threading.Thread(
            target=_pump, args=(name, proc, tui), daemon=True,
        )
        thread.start()
        active[proc] = (item, thread, name)

    while pending and len(active) < workers:
        _launch(pending.pop(0))

    while active:
        finished = [p for p in active if p.poll() is not None]
        for proc in finished:
            item, thread, name = active.pop(proc)
            thread.join(timeout=5)
            rc = proc.returncode
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            tui.log(f"done: {status}", level="info" if rc == 0 else "error",
                    context=name)
            tui.advance()
            if rc != 0:
                failed.append(item)
            if pending:
                _launch(pending.pop(0))
        if not finished:
            time.sleep(0.05)

    return failed
