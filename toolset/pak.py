"""Unreal .pak (version 3) container: write and read.

Just enough of the format for map mods: a flat file list, optional zlib
compression in 64 KiB blocks, no encryption. Foxhole mounts a v3 pak
dropped next to its own War-WindowsNoEditor.pak.

The reader only understands what the writer produces (v3, zlib or stored),
which covers every mod built here or with the old fh_map_mod_generator.
"""

import hashlib
import zlib
from struct import pack, unpack_from
from typing import Dict, List, Mapping, Tuple

MAGIC = 0x5A6F12E1
VERSION = 3
BLOCK_SIZE = 65536
MOUNT_POINT = "../../../"

# magic, version, index offset, index size, index sha1
_FOOTER = "<IIQQ20s"
_FOOTER_SIZE = 44
# In-data copy of the entry header: offset, compressed size, size, method,
# sha1 (48 bytes); the block table and "<BI" (encrypted, block size) follow.
_DATA_HEADER_SIZE = 48


def _pack_string(text: str) -> bytes:
    encoded = text.replace("\\", "/").encode("utf-8") + b"\0"
    return pack("<I", len(encoded)) + encoded


def _write_stored(stream, data: bytes) -> Tuple[int, bytes]:
    stream.write(data)
    return len(data), hashlib.sha1(data).digest()


def _write_zlib(stream, data: bytes) -> Tuple[int, bytes, int, List[int]]:
    """Write data as zlib blocks behind a block table of absolute offsets."""
    block_count = (len(data) + BLOCK_SIZE - 1) // BLOCK_SIZE
    base_offset = stream.tell()

    stream.write(pack("<I", block_count))
    stream.seek(block_count * 16, 1)
    stream.write(pack("<BI", 0, BLOCK_SIZE))

    cur_offset = base_offset + 4 + block_count * 16 + 5
    blocks = [0] * block_count * 2
    compressed_size = 0
    hasher = hashlib.sha1()

    for i in range(block_count):
        chunk = zlib.compress(data[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE])
        compressed_size += len(chunk)
        blocks[i * 2] = cur_offset
        cur_offset += len(chunk)
        blocks[i * 2 + 1] = cur_offset
        hasher.update(chunk)
        stream.write(chunk)

    end = stream.tell()
    stream.seek(base_offset + 4, 0)
    stream.write(pack(f"<{block_count * 2}Q", *blocks))
    stream.seek(end, 0)
    return compressed_size, hasher.digest(), block_count, blocks


def _write_entry(stream, data: bytes, compress: bool) -> bytes:
    """Write one file's data; return its index record."""
    offset = stream.tell()
    stream.write(pack("<16xQI20x", len(data), int(compress)))

    if compress:
        compressed_size, sha1, block_count, blocks = _write_zlib(stream, data)
    else:
        stream.write(pack("<BI", 0, 0))
        compressed_size, sha1 = _write_stored(stream, data)

    end = stream.tell()
    stream.seek(offset + 8, 0)
    stream.write(pack("<Q", compressed_size))
    stream.seek(offset + 28, 0)
    stream.write(sha1)
    stream.seek(end, 0)

    if compress:
        return (pack("<QQQI20s", offset, compressed_size, len(data), 1, sha1)
                + pack(f"<I{block_count * 2}Q", block_count, *blocks)
                + pack("<BI", 0, BLOCK_SIZE))
    return pack("<QQQI20sBI", offset, compressed_size, len(data), 0, sha1,
                0, 0)


def write_pak(path, files: Mapping[str, bytes], compress: bool = True) -> None:
    """Write ``files`` ({in-game path: bytes}) as a v3 pak at ``path``."""
    with open(path, "wb") as stream:
        records = [(name, _write_entry(stream, files[name], compress))
                   for name in sorted(files)]

        hasher = hashlib.sha1()
        index_offset = stream.tell()
        index = _pack_string(MOUNT_POINT) + pack("<I", len(records))
        for name, record in records:
            index += _pack_string(name) + record
        hasher.update(index)
        stream.write(index)
        stream.write(pack(_FOOTER, MAGIC, VERSION, index_offset, len(index),
                          hasher.digest()))


def _read_string(buf: bytes, pos: int) -> Tuple[str, int]:
    (length,) = unpack_from("<I", buf, pos)
    pos += 4
    return buf[pos:pos + length - 1].decode("utf-8"), pos + length


def read_pak(path) -> Dict[str, bytes]:
    """{in-game path: bytes} for every file in a v3 pak."""
    with open(path, "rb") as f:
        buf = f.read()
    if len(buf) < _FOOTER_SIZE:
        raise ValueError(f"{path}: too small to be a .pak")
    magic, version, index_offset, _, _ = unpack_from(
        _FOOTER, buf, len(buf) - _FOOTER_SIZE)
    if magic != MAGIC or version != VERSION:
        raise ValueError(f"{path}: not a version {VERSION} .pak "
                         f"(magic {magic:#x}, version {version}); only "
                         f"map mods built by this toolset can be read")

    _, pos = _read_string(buf, index_offset)
    (count,) = unpack_from("<I", buf, pos)
    pos += 4

    files: Dict[str, bytes] = {}
    for _ in range(count):
        name, pos = _read_string(buf, pos)
        offset, _, size, method, _ = unpack_from(
            "<QQQI20s", buf, pos)
        pos += 48
        blocks: List[Tuple[int, int]] = []
        if method:
            (block_count,) = unpack_from("<I", buf, pos)
            pos += 4
            raw = unpack_from(f"<{block_count * 2}Q", buf, pos)
            pos += block_count * 16
            blocks = list(zip(raw[0::2], raw[1::2]))
        encrypted, _ = unpack_from("<BI", buf, pos)
        pos += 5
        if encrypted:
            raise ValueError(f"{path}: {name} is encrypted")

        if method == 0:
            start = offset + _DATA_HEADER_SIZE + 5
            data = buf[start:start + size]
        elif method == 1:
            data = b"".join(zlib.decompress(buf[a:b]) for a, b in blocks)
        else:
            raise ValueError(f"{path}: {name} uses compression method "
                             f"{method}; only zlib is supported")
        if len(data) != size:
            raise ValueError(f"{path}: {name} is truncated")
        files[name] = data
    return files
