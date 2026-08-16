"""
parallel.py
========
Subprocess fan-out helper used by the "all" modes of the pipeline scripts.

Each work item is processed in its own child process (so each one gets its
own fresh ``bpy`` state). Child stdout/stderr is streamed back line by line,
prefixed with ``[label] ``, into the caller's ``ScriptTUI`` so interleaved
output from concurrent workers lands in the same scrolling log the rest of
the script uses.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from typing import Callable, List, Sequence

from utils.config import REPO_ROOT
from utils.tui import ScriptTUI


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


def _pump(label: str, proc: subprocess.Popen, tui: ScriptTUI) -> None:
    """Forward a child's stdout into the TUI's log, prefixed ``[label] ``.

    Repo-root absolute paths in the line are rewritten to repo-relative
    POSIX paths so Blender's native log output doesn't dump 100+ char
    absolute paths.
    """
    assert proc.stdout is not None
    for line in proc.stdout:
        tui.log(f"[{label}] {_shorten_paths(line).rstrip()}")


def run_parallel_subprocesses(
    items: Sequence[str],
    build_cmd: Callable[[str], List[str]],
    workers: int,
    tui: ScriptTUI,
    label_fn: Callable[[str], str] = lambda x: x,
) -> List[str]:
    """
    Run one subprocess per item with at most ``workers`` running at once.

    Each forwarded output line is prefixed with "[i/N name] " where
    ``name`` is ``label_fn(item)`` right-padded to the longest label so
    concurrent workers line up in the log. ``tui`` advances by one for
    every *completed* (not launched) child.

    Returns the list of items whose subprocess exited with a non-zero code.
    """
    if workers < 1:
        workers = 1

    pending: List[str] = list(items)
    active: dict = {}  # Popen -> (item, thread, label)
    failed: List[str] = []

    total = len(items)
    # Precompute aligned prefixes so interleaved output lines up cleanly.
    raw_labels = [label_fn(it) for it in items]
    name_width = max((len(n) for n in raw_labels), default=0)
    idx_width = len(str(total))

    started = 0

    def _prefix(item: str) -> str:
        nonlocal started
        started += 1
        name = label_fn(item).ljust(name_width)
        return f"{started:>{idx_width}}/{total} {name}"

    def _launch(item: str) -> None:
        label = _prefix(item)
        cmd = build_cmd(item)
        tui.log(f"[{label}] launching")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        thread = threading.Thread(
            target=_pump, args=(label, proc, tui), daemon=True,
        )
        thread.start()
        active[proc] = (item, thread, label)

    while pending and len(active) < workers:
        _launch(pending.pop(0))

    while active:
        finished = [p for p in active if p.poll() is not None]
        for proc in finished:
            item, thread, label = active.pop(proc)
            thread.join(timeout=5)
            rc = proc.returncode
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            tui.log(f"[{label}] done: {status}")
            tui.advance()
            if rc != 0:
                failed.append(item)
            if pending:
                _launch(pending.pop(0))
        if not finished:
            time.sleep(0.05)

    return failed
