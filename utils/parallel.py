"""
parallel.py
========
Subprocess fan-out helper used by the "all" modes of the pipeline scripts.

Each work item is processed in its own child process (so each one gets its
own fresh ``bpy`` state). Child stdout/stderr is streamed back line by line
with a ``[label] `` prefix so interleaved output from concurrent workers
stays readable.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence

from utils import progress, tui
from utils.config import REPO_ROOT


_PRINT_LOCK = threading.Lock()


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


def _pump(label: str, proc: subprocess.Popen,
          sink: Optional["progress.LineSink"] = None) -> None:
    """Forward a child's stdout to our stdout, prefixed with ``[label] ``,
    or to ``sink`` when the progress display owns the terminal.

    Repo-root absolute paths in the line are rewritten to repo-relative
    POSIX paths so Blender's native log output doesn't dump 100+ char
    absolute paths.
    """
    assert proc.stdout is not None
    for line in proc.stdout:
        short = _shorten_paths(line)
        if sink is not None:
            sink.write(short if short.endswith("\n") else short + "\n")
            continue
        with _PRINT_LOCK:
            sys.stdout.write(f"[{label}] {short}")
            sys.stdout.flush()


def run_parallel_subprocesses(
    items: Sequence[str],
    build_cmd: Callable[[str], List[str]],
    workers: int,
    label_fn: Callable[[str], str] = lambda x: x,
    env_extra: Optional[Dict[str, str]] = None,
    tracker: Optional["progress.Tracker"] = None,
    title: str = "Working",
    unit: str = "item",
    step_unit: str = "step",
    verbose: bool = False,
) -> List[str]:
    """
    Run one subprocess per item with at most ``workers`` running at once.

    With a ``tracker``, each running worker gets a progress bar and its
    chatter is hidden (``verbose`` restores it); warnings still print above
    the bars. Without one, every output line is prefixed with "[i/N name] "
    so concurrent workers line up in the terminal.

    ``env_extra`` is merged into each child's environment (used to hand
    workers their per-process thread budget).

    Returns the list of items whose subprocess exited with a non-zero code.
    """
    if tracker is not None:
        return _run_with_progress(
            items, build_cmd, workers, label_fn, env_extra,
            tracker, title, unit, step_unit, verbose,
        )
    if workers < 1:
        workers = 1

    # Writing to a pipe makes the child's stdout block-buffered, so a worker
    # flushes ~8 KB at a time and looks stalled between bursts.
    child_env = dict(os.environ)
    child_env["PYTHONUNBUFFERED"] = "1"
    if env_extra:
        child_env.update({k: str(v) for k, v in env_extra.items()})

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
        with _PRINT_LOCK:
            print(f"[{label}] launching", flush=True)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            env=child_env,
        )
        thread = threading.Thread(
            target=_pump, args=(label, proc), daemon=True,
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
            with _PRINT_LOCK:
                status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                print(f"[{label}] done: {status}", flush=True)
            if rc != 0:
                failed.append(item)
            if pending:
                _launch(pending.pop(0))
        if not finished:
            time.sleep(0.05)

    return failed


def _run_with_progress(
    items: Sequence[str],
    build_cmd: Callable[[str], List[str]],
    workers: int,
    label_fn: Callable[[str], str],
    env_extra: Optional[Dict[str, str]],
    tracker: "progress.Tracker",
    title: str,
    unit: str,
    step_unit: str,
    verbose: bool,
) -> List[str]:
    """run_parallel_subprocesses() with a bar per running worker."""
    if workers < 1:
        workers = 1

    child_env = dict(os.environ)
    child_env["PYTHONUNBUFFERED"] = "1"
    # No widgets or ANSI from a child writing to a pipe, and no capture:
    # their plain log lines are what move our bars.
    child_env["FH_NO_TUI"] = "1"
    child_env[progress.WORKER_ENV] = "1"
    if env_extra:
        child_env.update({k: str(v) for k, v in env_extra.items()})

    pending: List[str] = list(items)
    active: dict = {}  # Popen -> (item, thread)
    failed: List[str] = []

    with tui.Progress(title, total=len(items), unit=unit,
                      step_unit=step_unit,
                      label_width=progress.max_label_width(items, label_fn)) as disp:
        with progress.Poller(disp, tracker) as poller:

            def _launch(item: str) -> None:
                label = label_fn(item)
                tracker.begin(item)
                disp.start(item, label, tracker.total(item))
                poller.add(item)
                sink = progress.LineSink(disp, item, tracker, label, verbose)
                proc = subprocess.Popen(
                    build_cmd(item),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    encoding="utf-8",
                    errors="replace",
                    env=child_env,
                )
                thread = threading.Thread(
                    target=_pump, args=(label, proc, sink), daemon=True,
                )
                thread.start()
                active[proc] = (item, thread)

            while pending and len(active) < workers:
                _launch(pending.pop(0))

            while active:
                finished = [p for p in active if p.poll() is not None]
                for proc in finished:
                    item, thread = active.pop(proc)
                    thread.join(timeout=5)
                    poller.remove(item)
                    rc = proc.returncode
                    if rc != 0:
                        failed.append(item)
                    disp.finish(item, rc == 0, note=f"exit code {rc}")
                    if pending:
                        _launch(pending.pop(0))
                if not finished:
                    time.sleep(0.05)

    return failed
