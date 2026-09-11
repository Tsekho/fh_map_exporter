"""
gpu_lock.py
===========
Cross-process gate around Cycles renders.

The render workers are independent processes (see :mod:`utils.parallel`)
sharing one GPU. Concurrent Cycles renders don't overlap usefully -- the
driver time-slices them, both run at about half speed, and the VRAM cost
adds up. Holding a machine-wide lock for each ``render()`` call turns that
into a queue: one process renders while the others run their CPU raycast
bakes, so the phases pipeline instead of colliding.

The lock is advisory and file-based, so it needs no parent coordinating
the processes. Any failure falls through and renders anyway -- a slow
render beats a dead pipeline.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import time
from typing import Iterator, Optional

from utils.config import GPU_SERIALIZE_RENDERS

_LOCK_PATH = os.path.join(tempfile.gettempdir(), "fh_map_exporter_gpu.lock")

# Poll interval while another process holds the GPU.
_RETRY_S = 0.25
# Give up waiting (and render anyway) after this long, so a crashed
# worker that never released the lock cannot wedge the whole run.
_MAX_WAIT_S = 900.0


def _try_lock(fh) -> bool:
    """Attempt a non-blocking exclusive lock. True on success."""
    try:
        import msvcrt

        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        return True
    except ImportError:
        pass
    except OSError:
        return False

    try:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except ImportError:
        return True  # no locking primitive available; don't serialize
    except OSError:
        return False


def _unlock(fh) -> None:
    try:
        import msvcrt

        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        return
    except (ImportError, OSError):
        pass
    try:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except (ImportError, OSError):
        pass


@contextlib.contextmanager
def gpu_slot(label: str = "") -> Iterator[None]:
    """Hold the machine-wide GPU slot for the duration of the block.

    A no-op when GPU_SERIALIZE_RENDERS is False, or when the env var
    ``FH_GPU_LOCK`` is set to ``0`` (used to A/B the gate without
    editing config).
    """
    enabled = GPU_SERIALIZE_RENDERS and os.environ.get("FH_GPU_LOCK", "1") != "0"
    if not enabled:
        yield
        return

    fh: Optional[object] = None
    try:
        fh = open(_LOCK_PATH, "a+b")
        fh.write(b"\0")
        fh.flush()
        fh.seek(0)
    except OSError:
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass
        yield  # cannot lock; render unserialized rather than fail
        return

    held = False
    t0 = time.perf_counter()
    waited = 0.0
    try:
        while True:
            if _try_lock(fh):
                held = True
                break
            waited = time.perf_counter() - t0
            if waited >= _MAX_WAIT_S:
                print(f"  [gpu-lock] timed out after {waited:.0f}s; "
                      f"rendering without the gate{f' ({label})' if label else ''}")
                break
            time.sleep(_RETRY_S)

        if held and waited > 1.0:
            print(f"  [gpu-lock] waited {waited:.1f}s for the GPU"
                  f"{f' ({label})' if label else ''}")
        yield
    finally:
        if held:
            _unlock(fh)
        try:
            fh.close()  # type: ignore[union-attr]
        except OSError:
            pass
