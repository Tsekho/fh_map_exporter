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

import subprocess
import sys
import threading
import time
from typing import Callable, List, Sequence


_PRINT_LOCK = threading.Lock()


def _pump(label: str, proc: subprocess.Popen) -> None:
    """Forward a child's stdout to our stdout, prefixed with ``[label] ``."""
    assert proc.stdout is not None
    for line in proc.stdout:
        with _PRINT_LOCK:
            sys.stdout.write(f"[{label}] {line}")
            sys.stdout.flush()


def run_parallel_subprocesses(
    items: Sequence[str],
    build_cmd: Callable[[str], List[str]],
    workers: int,
    label_fn: Callable[[str], str] = lambda x: x,
) -> List[str]:
    """
    Run one subprocess per item with at most ``workers`` running at once.

    ``build_cmd(item)`` must return the argv list for that item's subprocess.
    ``label_fn(item)`` produces the short prefix used on every output line.

    Returns the list of items whose subprocess exited with a non-zero code.
    """
    if workers < 1:
        workers = 1

    pending: List[str] = list(items)
    active: dict = {}  # Popen -> (item, thread)
    failed: List[str] = []

    def _launch(item: str) -> None:
        label = label_fn(item)
        cmd = build_cmd(item)
        with _PRINT_LOCK:
            print(f"[{label}] launching: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        thread = threading.Thread(
            target=_pump, args=(label, proc), daemon=True,
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
            label = label_fn(item)
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
