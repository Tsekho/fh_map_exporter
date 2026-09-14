"""PNG read/write helpers (16-bit gray, 8-bit gray/RGB/RGBA)."""

import os
import struct
import zlib
from contextlib import contextmanager
from typing import Iterator, Tuple

import numpy as np


def partial_path(path: str) -> str:
    """The sibling temp name atomic_png() writes to. Keeps a ``.png`` suffix
    so Blender's use_file_extension does not append one when it is handed to
    scene.render.filepath, and a ``.partial`` stem so no stitcher glob or
    per-region lookup ever mistakes a leftover for a finished bake."""
    root, ext = os.path.splitext(path)
    return f"{root}.partial{ext or '.png'}"


def unlink_quiet(path: str) -> None:
    """Delete ``path`` if present, ignoring failures."""
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


@contextmanager
def atomic_png(path: str) -> Iterator[str]:
    """Yield a sibling ``<name>.partial.png`` to write, then rename it over
    ``path``.

    A bake PNG takes seconds to encode (and a Cycles render far longer);
    written straight to its final name, a Ctrl+C mid-write leaves a
    truncated file carrying a fresh mtime, which every downstream
    existence/mtime check -- the step-4 progress tracker, step-5's
    Stage.up_to_date(), the stitcher -- reads as a finished bake. The
    rename is atomic, so an interrupt leaves either the previous output
    or none, both of which read as "not done".

    """
    tmp = partial_path(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        unlink_quiet(tmp)


def imwrite_atomic(path: str, img: np.ndarray) -> None:
    """cv2.imwrite through atomic_png(). Raises OSError if the encode fails."""
    import cv2

    with atomic_png(path) as tmp:
        # cv2 signals a bad extension by raising cv2.error and an encode
        # failure by returning False; callers only have to catch OSError.
        try:
            wrote = cv2.imwrite(tmp, img)
        except cv2.error as exc:
            raise OSError(f"cv2 failed to write {tmp}: {exc}") from exc
        if not wrote:
            raise OSError(f"cv2 failed to write {tmp}")


def read_png16_gray(path: str) -> Tuple[int, int, np.ndarray]:
    """Read a 16-bit grayscale PNG. Returns (width, height, uint16 array).

    Foxhole heightmap encoding: height_m = (pixel - 32768) / 100.
    """
    with open(path, "rb") as f:
        sig = f.read(8)
    assert sig == b"\x89PNG\r\n\x1a\n", "Not a valid PNG file"

    width = height = 0
    bit_depth = color_type = 0
    idat_chunks: list[bytes] = []

    with open(path, "rb") as f:
        f.read(8)
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            length = struct.unpack(">I", hdr[:4])[0]
            tag = hdr[4:8]
            data = f.read(length)
            f.read(4)  # CRC

            if tag == b"IHDR":
                width, height = struct.unpack(">II", data[:8])
                bit_depth = data[8]
                color_type = data[9]
                assert bit_depth == 16 and color_type == 0, (
                    f"Expected 16-bit grayscale, got depth={bit_depth} "
                    f"ctype={color_type}"
                )
            elif tag == b"IDAT":
                idat_chunks.append(data)
            elif tag == b"IEND":
                break

    raw = zlib.decompress(b"".join(idat_chunks))

    bpp = 2
    row_len = width * bpp
    out = np.zeros((height, width), dtype=np.uint16)
    prev = np.zeros(row_len, dtype=np.uint8)

    pos = 0
    for y in range(height):
        filt = raw[pos]
        pos += 1
        row = np.frombuffer(
            raw, dtype=np.uint8, count=row_len, offset=pos
        ).copy()
        pos += row_len

        if filt == 0:
            pass
        elif filt == 1:
            for i in range(bpp, row_len):
                row[i] = (row[i] + row[i - bpp]) & 0xFF
        elif filt == 2:
            row = (row.astype(np.int16) + prev).astype(np.uint8)
        elif filt == 3:
            row = row.astype(np.int16)
            for i in range(row_len):
                a = row[i - bpp] if i >= bpp else 0
                b = int(prev[i])
                row[i] = (row[i] + (a + b) // 2) & 0xFF
            row = row.astype(np.uint8)
        elif filt == 4:
            row = row.astype(np.int16)
            for i in range(row_len):
                a = int(row[i - bpp]) if i >= bpp else 0
                b = int(prev[i])
                c = int(prev[i - bpp]) if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                row[i] = (row[i] + pr) & 0xFF
            row = row.astype(np.uint8)

        prev = row.astype(np.uint8)
        out[y] = (
            row.reshape(width, 2).astype(np.uint16)[:, 0] * 256
            + row.reshape(width, 2).astype(np.uint16)[:, 1]
        )

    return width, height, out


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data)) + tag + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _write_png(path: str, w: int, h: int, depth: int, ctype: int,
               row_bytes: bytes, row_len: int) -> None:
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        raw += row_bytes[y * row_len:(y + 1) * row_len]
    idat = zlib.compress(bytes(raw), level=6)
    with atomic_png(path) as tmp:
        with open(tmp, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
            f.write(_png_chunk(
                b"IHDR", struct.pack(">IIBBBBB", w, h, depth, ctype, 0, 0, 0)
            ))
            f.write(_png_chunk(b"IDAT", idat))
            f.write(_png_chunk(b"IEND", b""))


def write_png16_gray(path: str, arr: np.ndarray) -> None:
    h, w = arr.shape
    be = np.ascontiguousarray(arr.astype(">u2")).tobytes()
    _write_png(path, w, h, 16, 0, be, w * 2)


def write_png8_rgb(path: str, arr: np.ndarray) -> None:
    h, w, _ = arr.shape
    data = np.ascontiguousarray(arr.astype(np.uint8)).tobytes()
    _write_png(path, w, h, 8, 2, data, w * 3)


def write_png8_gray(path: str, arr: np.ndarray) -> None:
    h, w = arr.shape
    data = np.ascontiguousarray(arr.astype(np.uint8)).tobytes()
    _write_png(path, w, h, 8, 0, data, w)


def write_png8_rgba(path: str, arr: np.ndarray) -> None:
    h, w, _ = arr.shape
    data = np.ascontiguousarray(arr.astype(np.uint8)).tobytes()
    _write_png(path, w, h, 8, 6, data, w * 4)
