"""PSK/PSKX reader and Blender mesh loader."""

import ctypes
import os
from typing import Dict, List, Optional, Tuple

import bpy
import numpy as np

from utils.tui import warn


class _Section(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 20),
        ("type_flags", ctypes.c_int32),
        ("data_size", ctypes.c_int32),
        ("data_count", ctypes.c_int32),
    ]


class _Vec3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float),
                ("z", ctypes.c_float)]


class _Wedge16(ctypes.Structure):
    _fields_ = [
        ("point_index", ctypes.c_uint16),
        ("_pad1", ctypes.c_int16),
        ("u", ctypes.c_float),
        ("v", ctypes.c_float),
        ("material_index", ctypes.c_uint8),
        ("_reserved", ctypes.c_int8),
        ("_pad2", ctypes.c_int16),
    ]


class _Wedge32(ctypes.Structure):
    _fields_ = [
        ("point_index", ctypes.c_uint32),
        ("u", ctypes.c_float),
        ("v", ctypes.c_float),
        ("material_index", ctypes.c_uint32),
    ]


class _Face16(ctypes.Structure):
    _fields_ = [
        ("wedge_indices", ctypes.c_uint16 * 3),
        ("material_index", ctypes.c_uint8),
        ("aux_material_index", ctypes.c_uint8),
        ("smoothing_groups", ctypes.c_int32),
    ]


class _Face32(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("wedge_indices", ctypes.c_uint32 * 3),
        ("material_index", ctypes.c_uint8),
        ("aux_material_index", ctypes.c_uint8),
        ("smoothing_groups", ctypes.c_int32),
    ]


class _Material(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 64),
        ("texture_index", ctypes.c_int32),
        ("poly_flags", ctypes.c_int32),
        ("aux_material", ctypes.c_int32),
        ("aux_flags", ctypes.c_int32),
        ("lod_bias", ctypes.c_int32),
        ("lod_style", ctypes.c_int32),
    ]


def _read_chunk(fp, cls, section: _Section):
    nb = section.data_size * section.data_count
    buf = bytearray(fp.read(nb))
    return tuple((cls * section.data_count).from_buffer(buf))


def read_psk(path: str):
    """Parse a .psk / .pskx file. Returns (points, wedges, faces)."""
    points = wedges = faces = ()
    with open(path, "rb") as fp:
        sec_size = ctypes.sizeof(_Section)
        while True:
            raw = fp.read(sec_size)
            if len(raw) < sec_size:
                break
            sec = _Section.from_buffer_copy(raw)
            if sec.name == b"ACTRHEAD":
                pass
            elif sec.name == b"PNTS0000":
                points = _read_chunk(fp, _Vec3, sec)
            elif sec.name == b"VTXW0000":
                cls = _Wedge32 if len(points) > 0xFFFF else _Wedge16
                wedges = _read_chunk(fp, cls, sec)
            elif sec.name == b"FACE0000":
                faces = _read_chunk(fp, _Face16, sec)
            elif sec.name == b"FACE3200":
                faces = _read_chunk(fp, _Face32, sec)
            elif sec.name == b"MATT0000":
                _read_chunk(fp, _Material, sec)
            else:
                fp.read(sec.data_size * sec.data_count)
    return points, wedges, faces


# Cached bpy.types.Mesh per source name; cleared each Map.blend().
mesh_cache: Dict[str, Optional[bpy.types.Material]] = {}

# Raw PSK cache for spline deformation: (verts_np UE cm Nx3, tris).
# Faces are pre-reversed (2,1,0) to compensate for the Y-flip on convert.
_psk_raw_cache: Dict[str, Optional[Tuple[np.ndarray, List[Tuple[int, int, int]]]]] = {}


def psk_to_blender_mesh(points, wedges, faces, name: str) -> bpy.types.Mesh:
    """Convert PSK data to a bpy mesh (UE cm -> Blender m)."""
    mesh = bpy.data.meshes.new(name)
    verts = [(p.x * 0.01, p.y * 0.01, p.z * 0.01) for p in points]
    tris = [
        [
            wedges[face.wedge_indices[2]].point_index,
            wedges[face.wedge_indices[1]].point_index,
            wedges[face.wedge_indices[0]].point_index,
        ]
        for face in faces
    ]
    mesh.from_pydata(verts, [], tris)
    mesh.update()
    return mesh


def get_mesh(mesh_name: str, meshes_dir: str) -> Optional[bpy.types.Mesh]:
    """Load (and cache) a .pskx/.psk mesh."""
    if mesh_name in mesh_cache:
        return mesh_cache[mesh_name]

    for ext in (".pskx", ".psk"):
        path = os.path.join(meshes_dir, mesh_name + ext)
        if os.path.exists(path):
            try:
                pts, wdgs, fcs = read_psk(path)
                if pts and fcs:
                    mesh = psk_to_blender_mesh(pts, wdgs, fcs, mesh_name)
                    mesh_cache[mesh_name] = mesh
                    return mesh
                warn(f"  [WARN] Empty PSK: {path}")
            except Exception as exc:
                warn(f"  [WARN] PSK read error ({mesh_name}): {exc}")
            break

    mesh_cache[mesh_name] = None
    return None


def get_raw_psk(mesh_name: str, meshes_dir: str):
    """Return cached raw (verts_ue, tris) for spline deformation."""
    if mesh_name in _psk_raw_cache:
        return _psk_raw_cache[mesh_name]

    for ext in (".pskx", ".psk"):
        path = os.path.join(meshes_dir, mesh_name + ext)
        if not os.path.exists(path):
            continue
        try:
            pts, wdgs, fcs = read_psk(path)
            if not pts or not fcs:
                break
            verts = np.array([(p.x, p.y, p.z) for p in pts], dtype=np.float64)
            tris = [
                (
                    wdgs[f.wedge_indices[2]].point_index,
                    wdgs[f.wedge_indices[1]].point_index,
                    wdgs[f.wedge_indices[0]].point_index,
                )
                for f in fcs
            ]
            _psk_raw_cache[mesh_name] = (verts, tris)
            return _psk_raw_cache[mesh_name]
        except Exception as exc:
            warn(f"  [WARN] PSK raw read error ({mesh_name}): {exc}")
        break

    _psk_raw_cache[mesh_name] = None
    return None


def clear_caches() -> None:
    mesh_cache.clear()
    _psk_raw_cache.clear()
