"""
helpers.py
========
Shared helpers for the Foxhole map -> Blender pipeline.

    - Map class  (with include / exclude / palette filtering)
    - PSK / PSKX parsing
    - 16-bit grayscale PNG reader
    - Blender mesh / collection / transform utilities
    - Heightmap terrain builder
    - Region-spill .blend builder
    - Raycast / AO bakers used by the top-down renderer

Filtering / palette parameters (all on Map):
    include - optional list of fnmatch patterns; only mesh names that match
              at least one pattern are placed.
    exclude - optional list of fnmatch patterns; mesh names matching any
              pattern are skipped (applied after include).
    palette - optional dict mapping an fnmatch pattern to a '#RRGGBB' hex
              string.  After all meshes are loaded the first matching pattern
              wins and a Principled-BSDF material is assigned to the mesh
              data-block (shared by all instances).  Unmatched meshes keep
              the Blender default material.
"""

import ctypes
import fnmatch
import json
import os
import struct
import zlib
from math import radians
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import bpy
from mathutils import Vector


# ------------------------------------------------------------------------------
#  PSK data structures
# ------------------------------------------------------------------------------


class _Section(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 20),
        ("type_flags", ctypes.c_int32),
        ("data_size", ctypes.c_int32),
        ("data_count", ctypes.c_int32),
    ]


class _Vec3(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
    ]


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
    """
    Parse a .psk / .pskx file.
    Returns (points, wedges, faces) tuples or raises on error.
    """
    points = ()
    wedges = ()
    faces = ()

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


# ------------------------------------------------------------------------------
#  16-bit grayscale PNG reader
# ------------------------------------------------------------------------------


def read_png16_gray(path: str) -> Tuple[int, int, np.ndarray]:
    """
    Read a 16-bit grayscale PNG without Pillow/cv2.
    Returns (width, height, array[uint16, shape=(height, width)]).

    Height encoding in Foxhole heightmaps:
        height_m = (pixel_value - 32768) / 100.0
    Canvas: 2200×2200, center pixel (1100, 1100) = world origin (0, 0).
    """
    with open(path, "rb") as f:
        sig = f.read(8)
    assert sig == b"\x89PNG\r\n\x1a\n", "Not a valid PNG file"

    width = height = 0
    bit_depth = color_type = 0
    idat_chunks: list[bytes] = []

    with open(path, "rb") as f:
        f.read(8)  # signature
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


# ------------------------------------------------------------------------------
#  Blender helpers
# ------------------------------------------------------------------------------

# Module-level mesh cache; callers can clear() between maps.
mesh_cache: Dict[str, Optional[bpy.types.Material]] = {}


def _hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """
    Convert a '#RRGGBB' or 'RRGGBB' string to an (r, g, b) tuple in [0, 1].
    """
    h = hex_color.lstrip("#")
    return (
        int(h[0:2], 16) / 255.0,
        int(h[2:4], 16) / 255.0,
        int(h[4:6], 16) / 255.0,
    )


def _make_color_material(hex_color: str) -> bpy.types.Material:
    """
    Return a Principled-BSDF material whose Base Color matches *hex_color*.

    Materials are named ``Color_RRGGBB`` and reused if they already exist in
    the current blend file, so repeated calls with the same color are cheap.
    """
    name = f"Color_{hex_color.lstrip('#').upper()}"
    if name in bpy.data.materials:
        return bpy.data.materials[name]

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        r, g, b = _hex_to_rgb(hex_color)
        bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    return mat


def psk_to_blender_mesh(points, wedges, faces, name: str) -> bpy.types.Mesh:
    """Convert parsed PSK data into a bpy.data.Mesh (UE cm → Blender m)."""
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
    """Load (and cache) a .pskx/.psk mesh by its key name."""
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
                print(f"  [WARN] Empty PSK: {path}")
            except Exception as exc:
                print(f"  [WARN] PSK read error ({mesh_name}): {exc}")
            break

    mesh_cache[mesh_name] = None
    return None


def apply_ue_transform(
    obj: bpy.types.Object,
    t: list,
    shift: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """
    Apply a UE transform [x,y,z, sx,sy,sz, pitch,yaw,roll] to a Blender object.

    Coordinate mapping (matches BlenderUmap2):
        Blender X =  UE X / 100
        Blender Y = -UE Y / 100
        Blender Z =  UE Z / 100
        Euler XYZ = (roll, -pitch, -yaw)

    An optional *shift* tuple (in Blender meters) is added to the final
    object location after the UE→Blender conversion.  This is used by the
    spill builder to translate neighbor-region meshes onto their correct
    world-space position relative to the focus region.
    """
    x, y, z = t[0], t[1], t[2]
    sx, sy, sz = t[3], t[4], t[5]
    pitch, yaw, roll = t[6], t[7], t[8]

    obj.location = (
        x * 0.01 + shift[0],
        y * -0.01 + shift[1],
        z * 0.01 + shift[2],
    )
    obj.scale = (sx, sy, sz)
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = (radians(roll), radians(-pitch), radians(-yaw))


def transform_to_blender_xy(t: list) -> Tuple[float, float]:
    """Return (x, y) in Blender meters for a UE transform list."""
    return t[0] * 0.01, t[1] * -0.01


def place_mesh(
    mesh_name: str,
    transform: list,
    collection: bpy.types.Collection,
    meshes_dir: str,
    obj_name: Optional[str] = None,
    shift: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """
    Instantiate *mesh_name* into *collection* with *transform* applied.

    *shift* is an optional (x, y, z) translation in Blender meters, applied
    after the UE→Blender conversion.  Used by the spill builder to place
    neighbor-region meshes at their correct world-space offset.
    """
    mesh = get_mesh(mesh_name, meshes_dir)
    if mesh is None:
        return
    obj = bpy.data.objects.new(obj_name or mesh_name, mesh)
    collection.objects.link(obj)
    apply_ue_transform(obj, transform, shift)


def ensure_collection(
    segments: List[str],
    parent: bpy.types.Collection,
    cache: Dict[Tuple[str, ...], bpy.types.Collection],
    root_key: Tuple[str, ...],
) -> bpy.types.Collection:
    """
    Walk *segments*, creating sub-collections as needed under *parent*.
    *root_key* is prepended to each cache key so sibling trees with identical
    segment names (e.g. Symbols vs Groups) remain distinct.
    """
    current = parent
    for i, seg in enumerate(segments):
        key = root_key + tuple(segments[: i + 1])
        if key not in cache:
            coll = bpy.data.collections.new(seg)
            current.children.link(coll)
            cache[key] = coll
        current = cache[key]
    return current


def create_terrain(
    heightmap_path: str,
    collection: bpy.types.Collection,
    stride: int = 1,
    min_height: float = -100.0,
    shift: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    name: str = "Terrain",
    material: Optional[bpy.types.Material] = None,
) -> bpy.types.Object:
    """
    Build a grid mesh from a 16-bit PNG heightmap and add it to *collection*.

    *shift* is an optional (x, y, z) translation in Blender meters applied to
    the resulting object location, used when placing neighbor-region terrain
    in the spill build.  *material*, if given, is assigned to the terrain
    mesh data-block.
    """
    print(f"  Loading heightmap: {heightmap_path}")
    w, h, pixels = read_png16_gray(heightmap_path)
    print(f"  Heightmap size: {w}x{h}")

    cx, cy = w // 2, h // 2
    sub = pixels[::stride, ::stride]
    rows, cols = sub.shape

    valid = sub != 0
    hm = sub.astype(np.float32)
    hm[~valid] = 32768.0
    hm = (hm - 32768.0) / 100.0

    valid = valid & (hm >= min_height)

    # Level the N/S edge quads to match the quad just inside them.  Without
    # this, the outermost valid row along a region border often sits at a
    # markedly different height than its inward neighbor (steep fall-off or
    # a thin "skirt"), and the 2-row expansion below would then propagate
    # that artefact outward, producing quads that plunge well under the true
    # surface.  For every column, copy the second-from-edge row's height
    # onto the edge row, so the edge quad becomes flat and consistent with
    # the neighboring interior quad.
    any_valid_col = valid.any(axis=0)
    first_row = np.argmax(valid, axis=0)
    last_row = rows - 1 - np.argmax(valid[::-1], axis=0)
    lvl_cols = np.where(any_valid_col & (first_row + 1 <= last_row))[0]
    if lvl_cols.size:
        fr = first_row[lvl_cols]
        lr = last_row[lvl_cols]
        hm[fr, lvl_cols] = hm[fr + 1, lvl_cols]
        hm[lr, lvl_cols] = hm[lr - 1, lvl_cols]

    # Patch up-to-2-pixel holes at the N/S region border: for every void
    # cell that lies directly north or south of a valid cell (within 2
    # steps), copy the nearest valid neighbor's height and mark it valid.
    # Each iteration uses a snapshot of the previous pass's valid mask so
    # the expansion advances exactly one row per step (no runaway chaining
    # across the full column).
    for _ in range(2):
        v_orig = valid.copy()
        hm_orig = hm.copy()

        # South-facing: valid[i] -> fill row i+1 where row i+1 is void.
        src_s = v_orig[:-1, :] & ~v_orig[1:, :]
        if src_s.any():
            hm_dst_s = hm[1:, :]
            hm_src_s = hm_orig[:-1, :]
            hm_dst_s[src_s] = hm_src_s[src_s]
            valid[1:, :][src_s] = True

        # North-facing: valid[i] -> fill row i-1 where row i-1 is void.
        src_n = v_orig[1:, :] & ~v_orig[:-1, :]
        if src_n.any():
            hm_dst_n = hm[:-1, :]
            hm_src_n = hm_orig[1:, :]
            hm_dst_n[src_n] = hm_src_n[src_n]
            valid[:-1, :][src_n] = True

    px_coords = np.arange(0, w, stride, dtype=np.float32)[:cols]
    py_coords = np.arange(0, h, stride, dtype=np.float32)[:rows]
    VX, VY = np.meshgrid(px_coords - cx, -(py_coords - cy))
    verts_np = np.stack([VX.ravel(), VY.ravel(), hm.ravel()], axis=1)

    valid_flat = valid.ravel()
    old_to_new = np.full(rows * cols, -1, dtype=np.int32)
    old_to_new[valid_flat] = np.arange(valid_flat.sum(), dtype=np.int32)

    verts_np = verts_np[valid_flat]

    gy_idx, gx_idx = np.arange(rows - 1), np.arange(cols - 1)
    GY, GX = np.meshgrid(gy_idx, gx_idx, indexing="ij")
    GY, GX = GY.ravel(), GX.ravel()

    keep = (
        valid[GY, GX]
        & valid[GY, GX + 1]
        & valid[GY + 1, GX]
        & valid[GY + 1, GX + 1]
    )
    GY, GX = GY[keep], GX[keep]

    faces_np = np.empty((keep.sum(), 4), dtype=np.int32)
    faces_np[:, 0] = old_to_new[GY * cols + GX]
    faces_np[:, 1] = old_to_new[GY * cols + GX + 1]
    faces_np[:, 2] = old_to_new[(GY + 1) * cols + GX + 1]
    faces_np[:, 3] = old_to_new[(GY + 1) * cols + GX]

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(
        [tuple(v) for v in verts_np.tolist()],
        [],
        [tuple(f) for f in faces_np.tolist()],
    )
    mesh.update()

    if material is not None:
        mesh.materials.append(material)

    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj.location = shift

    print(
        f"  Terrain: {cols}x{rows} grid, "
        f"{len(verts_np):,} verts, {keep.sum():,} faces, "
        f"stride={stride} (1 vert = {stride} m)"
    )
    return obj


# ------------------------------------------------------------------------------
#  Map
# ------------------------------------------------------------------------------


class Map:
    """
    Represents one Foxhole map and drives the full Blender build.

    Parameters
    ----------
    json_path : str
        Path to the map JSON file produced by the extraction pipeline.
        Expected top-level keys: ``symbols``, ``groups``, ``blueprints``.
    export_dir : str
        Root export directory.  Sub-directories ``_heightmap``, ``_meshes``,
        and ``blend`` are resolved relative to this path.
    include : list[str] | None
        Optional allowlist of fnmatch patterns matched against mesh names.
        When provided, only mesh names matching **at least one** pattern are
        placed.  Blueprint ``_self`` keys are never affected by filtering.
    exclude : list[str] | None
        Optional denylist of fnmatch patterns matched against mesh names.
        Mesh names matching **any** pattern are skipped.  Applied *after*
        ``include``, so a name must pass both to be placed.
    palette : dict[str, str] | None
        Optional mapping of ``{fnmatch_pattern: '#RRGGBB'}`` hex color
        strings.  After all meshes have been loaded the first pattern (in
        dict insertion order) that matches a mesh name wins, and a
        Principled-BSDF material of that color is assigned to the shared
        mesh data-block.  Unmatched meshes keep the Blender default material.

    Examples
    --------
    ::

        m = Map(
            "exports/KalkStop.json",
            "exports",
            include=["*Building*", "*Wall*"],
            exclude=["*LOD*"],
            palette={
                "*Building*": "#C8A87A",
                "*Wall*":     "#8B8B8B",
            },
        )
        m.blend()
    """

    def __init__(
        self,
        json_path: str,
        export_dir: str,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        palette: Optional[Dict[str, str]] = None,
    ) -> None:
        self.json_path = json_path
        self.export_dir = export_dir
        self.name = os.path.splitext(os.path.basename(json_path))[0]
        self.include = include or []
        self.exclude = exclude or []
        self.palette = palette or {}

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if self.include or self.exclude:
            inc = (
                f"{len(self.include)} pattern(s)"
                if len(self.include) > 8 else (self.include or "none")
            )
            exc = (
                f"{len(self.exclude)} pattern(s)"
                if len(self.exclude) > 8 else (self.exclude or "none")
            )
            print(f"  Filters — include: {inc}, exclude: {exc}")

        raw_symbols: Dict[str, list] = data.get("symbols", {})
        raw_groups: Dict[str, list] = data.get("groups", {})
        raw_blueprints: Dict[str, list] = data.get("blueprints", {})

        self.symbols: Dict[str, list] = {
            k: v for k, v in raw_symbols.items() if self._should_place(k)
        }
        self.groups: Dict[str, list] = {
            k: v for k, v in raw_groups.items() if self._should_place(k)
        }

        filtered_bps: Dict[str, list] = {}
        for bp_class, instances in raw_blueprints.items():
            kept: list = []
            for inst in instances:
                filtered = {
                    k: v
                    for k, v in inst.items()
                    if k == "_self" or self._should_place(k)
                }
                if any(k != "_self" for k in filtered):
                    kept.append(filtered)
            if kept:
                filtered_bps[bp_class] = kept
        self.blueprints: Dict[str, list] = filtered_bps

    # -- Derived paths --------------------------------------------------------

    @property
    def heightmap_path(self) -> str:
        return os.path.join(self.export_dir, "_heightmap", f"{self.name}.png")

    @property
    def meshes_dir(self) -> str:
        return os.path.join(self.export_dir, "_meshes")

    @property
    def blend_path(self) -> str:
        return os.path.abspath(
            os.path.join(self.export_dir, "blend", f"{self.name}.blend")
        )

    # -- Filtering ------------------------------------------------------------

    def _should_place(self, mesh_name: str) -> bool:
        """
        Return ``True`` when *mesh_name* passes the include / exclude filters.

        Logic:
          1. If ``include`` patterns are set, the name must match at least one.
          2. If ``exclude`` patterns are set, the name must not match any.
        Both conditions must hold simultaneously.
        """
        if self.include and not any(
            fnmatch.fnmatch(mesh_name, p) for p in self.include
        ):
            return False
        if self.exclude and any(
            fnmatch.fnmatch(mesh_name, p) for p in self.exclude
        ):
            return False
        return True

    # -- Palette application --------------------------------------------------

    def _apply_palette(self) -> None:
        """
        Assign Principled-BSDF materials to loaded meshes based on
        ``self.palette``.

        Iterates ``mesh_cache`` after all geometry has been placed.  For each
        cached mesh the palette patterns are tested in insertion order; the
        first match wins.  The material is written directly to the mesh
        data-block so every object instance sharing that mesh inherits the
        color automatically.  Meshes with no matching pattern are left
        untouched (Blender default material).
        """
        if not self.palette:
            return

        mat_cache: Dict[str, bpy.types.Material] = {}
        colored = 0

        for mesh_name, mesh in mesh_cache.items():
            if mesh is None:
                continue
            for pattern, hex_color in self.palette.items():
                if fnmatch.fnmatch(mesh_name, pattern):
                    if hex_color not in mat_cache:
                        mat_cache[hex_color] = _make_color_material(hex_color)
                    mat = mat_cache[hex_color]
                    if mesh.materials:
                        mesh.materials[0] = mat
                    else:
                        mesh.materials.append(mat)
                    colored += 1
                    break  # first matching pattern wins

        print(f"  Palette applied: {colored} mesh(es) colored")

    # -- Placement ------------------------------------------------------------

    def _flatten(self) -> Dict[str, List[list]]:
        """
        Merge ``symbols``, ``groups`` and every blueprint mesh entry into a
        single ``{mesh_name: [transform, ...]}`` dict.  Transforms for the
        same mesh across different sources are concatenated; ``_self``
        blueprint entries are skipped.  Used by :meth:`_populate_flat` when
        the caller wants a single undifferentiated collection tree keyed
        purely on mesh name.
        """
        flat: Dict[str, List[list]] = {}
        for src in (self.symbols, self.groups):
            for k, v in src.items():
                flat.setdefault(k, []).extend(v)
        for instances in self.blueprints.values():
            for inst in instances:
                for k, v in inst.items():
                    if k == "_self":
                        continue
                    flat.setdefault(k, []).extend(v)
        return flat

    def _populate_flat(
        self,
        root: bpy.types.Collection,
        coll_cache: Dict[Tuple[str, ...], bpy.types.Collection],
        root_key_prefix: Tuple[str, ...] = (),
        shift: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        border_filter: Optional[Callable[[list], bool]] = None,
        announce: bool = True,
        track_meshes: Optional[set] = None,
        tracked_objects: Optional[List[bpy.types.Object]] = None,
    ) -> int:
        """
        Place every mesh (symbols + groups + blueprint meshes merged) into a
        single flat tree under *root*, with hierarchy derived from splitting
        each mesh name on ``__``::

            <root>/<seg_1>/<seg_2>/.../<seg_N>/<seg_N>.<i>

        No ``Symbols`` / ``Groups`` / ``Blueprints`` partition — the source
        dict is irrelevant, only the mesh name drives the tree.

        Parameters
        ----------
        root, coll_cache, root_key_prefix, shift, border_filter
            Same semantics as :meth:`_populate`.
        announce : bool
            Print a ``[meshes]`` summary line before placement.
        track_meshes : set[str], optional
            When given, every placed object whose mesh name is in this set is
            appended to *tracked_objects*.  Used by the deep-water pass to
            locate water objects without re-walking the scene.
        tracked_objects : list, optional
            Sink list for *track_meshes* (required when *track_meshes* is set).

        Returns
        -------
        int
            Number of objects created.
        """
        flat = self._flatten()

        if announce:
            inst_count = sum(len(v) for v in flat.values())
            print(
                f"[meshes] {len(flat):,} unique meshes, "
                f"{inst_count:,} instances ..."
            )

        total = 0
        for mesh_name, transforms in flat.items():
            sp = mesh_name.split("__")
            leaf = ensure_collection(sp, root, coll_cache, root_key_prefix)
            tracked_here = (
                track_meshes is not None
                and tracked_objects is not None
                and mesh_name in track_meshes
            )
            i = 0
            for t in transforms:
                if border_filter is not None and not border_filter(t):
                    continue
                mesh = get_mesh(mesh_name, self.meshes_dir)
                if mesh is None:
                    continue
                i += 1
                obj = bpy.data.objects.new(f"{sp[-1]}.{i}", mesh)
                leaf.objects.link(obj)
                apply_ue_transform(obj, t, shift)
                if tracked_here:
                    tracked_objects.append(obj)
                total += 1
        return total

    def _populate_by_category(
        self,
        root: bpy.types.Collection,
        coll_cache: Dict[Tuple[str, ...], bpy.types.Collection],
        mesh_to_category: Dict[str, str],
        root_key_prefix: Tuple[str, ...] = (),
        shift: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        border_filter: Optional[Callable[[list], bool]] = None,
        announce: bool = True,
        category_objects: Optional[Dict[str, List[bpy.types.Object]]] = None,
    ) -> int:
        """
        Place meshes under *root* grouped by category.  Tree layout::

            <root>/<category>/<seg_1>/.../<seg_N>/<seg_N>.<i>

        Only meshes present in *mesh_to_category* are placed; meshes with no
        category are silently skipped.  If *category_objects* is provided,
        every placed object is appended to the list for its category.
        """
        flat = self._flatten()

        if announce:
            total_inst = sum(
                len(v) for k, v in flat.items() if k in mesh_to_category
            )
            placed_meshes = sum(1 for k in flat if k in mesh_to_category)
            print(
                f"[meshes] {placed_meshes:,} unique meshes, "
                f"{total_inst:,} instances ..."
            )

        cat_roots: Dict[str, bpy.types.Collection] = {}
        total = 0
        for mesh_name, transforms in flat.items():
            cat = mesh_to_category.get(mesh_name)
            if cat is None:
                continue
            if cat not in cat_roots:
                c = bpy.data.collections.new(cat)
                root.children.link(c)
                cat_roots[cat] = c
            cat_root = cat_roots[cat]
            sp = mesh_name.split("__")
            leaf = ensure_collection(
                sp, cat_root, coll_cache, root_key_prefix + (cat,)
            )
            i = 0
            for t in transforms:
                if border_filter is not None and not border_filter(t):
                    continue
                mesh = get_mesh(mesh_name, self.meshes_dir)
                if mesh is None:
                    continue
                i += 1
                obj = bpy.data.objects.new(f"{sp[-1]}.{i}", mesh)
                leaf.objects.link(obj)
                apply_ue_transform(obj, t, shift)
                if category_objects is not None:
                    category_objects.setdefault(cat, []).append(obj)
                total += 1
        return total

    def _populate(
        self,
        root: bpy.types.Collection,
        coll_cache: Dict[Tuple[str, ...], bpy.types.Collection],
        root_key_prefix: Tuple[str, ...] = (),
        shift: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        border_filter: Optional[Callable[[list], bool]] = None,
        announce: bool = True,
    ) -> int:
        """
        Place this map's symbols, groups, and blueprint instances under *root*.

        Uses the exact collection layout of ``Map.blend()``:

            <root>/
                Symbols/     <a>/<b>/.../<leaf>/<leaf>.<i>
                Groups/      <a>/<b>/.../<leaf>/<leaf>.<i>
                Blueprints/  <bp_class>/<bp_class>.<i>/<mesh_name>/<mesh_name>.<j>

        Parameters
        ----------
        root : bpy.types.Collection
            Parent collection under which ``Symbols`` / ``Groups`` /
            ``Blueprints`` sub-collections are created.
        coll_cache : dict
            Shared collection cache keyed by ``root_key_prefix + path_segments``.
            Pass the *same* cache object across multiple ``_populate`` calls to
            keep the focus and neighbor trees distinct while reusing sub-paths
            inside each.
        root_key_prefix : tuple[str, ...]
            Tuple prepended to every cache key so parallel populations (e.g.
            one per neighbor region) don't collide.  The focus region uses
            ``()``; a neighbor uses ``(neighbor_name,)``.
        shift : (x, y, z)
            World-space translation in Blender meters applied to every
            placed object.  Used by the spill builder.
        border_filter : callable(transform) -> bool, optional
            When given, only transforms for which the filter returns ``True``
            are placed.  Applied uniformly to symbols, groups, and every
            mesh transform inside blueprints.
        announce : bool
            When ``True`` print the standard ``[symbols]`` / ``[groups]`` /
            ``[blueprints]`` summary lines, matching ``Map.blend()``.

        Returns
        -------
        int
            Number of objects placed.
        """
        total = 0

        def _keep(t: list) -> bool:
            return border_filter is None or border_filter(t)

        # -- Symbols ----------------------------------------------------------
        if self.symbols:
            if announce:
                s_inst = sum(len(v) for v in self.symbols.values())
                print(
                    f"[symbols] {len(self.symbols):,} unique meshes, "
                    f"{s_inst:,} instances ..."
                )
            s_root = bpy.data.collections.new("Symbols")
            root.children.link(s_root)
            cache_root = root_key_prefix + ("Symbols",)
            for mesh_name, instances in self.symbols.items():
                sp = mesh_name.split("__")
                leaf = ensure_collection(sp, s_root, coll_cache, cache_root)
                i = 0
                for t in instances:
                    if not _keep(t):
                        continue
                    i += 1
                    place_mesh(
                        mesh_name, t, leaf, self.meshes_dir,
                        f"{sp[-1]}.{i}", shift=shift,
                    )
                    total += 1

        # -- Groups -----------------------------------------------------------
        if self.groups:
            if announce:
                g_inst = sum(len(v) for v in self.groups.values())
                print(
                    f"[groups]  {len(self.groups):,} unique meshes, "
                    f"{g_inst:,} instances ..."
                )
            g_root = bpy.data.collections.new("Groups")
            root.children.link(g_root)
            cache_root = root_key_prefix + ("Groups",)
            for mesh_name, instances in self.groups.items():
                sp = mesh_name.split("__")
                leaf = ensure_collection(sp, g_root, coll_cache, cache_root)
                i = 0
                for t in instances:
                    if not _keep(t):
                        continue
                    i += 1
                    place_mesh(
                        mesh_name, t, leaf, self.meshes_dir,
                        f"{sp[-1]}.{i}", shift=shift,
                    )
                    total += 1

        # -- Blueprints -------------------------------------------------------
        if self.blueprints:
            if announce:
                bp_inst = sum(
                    sum(len(v) for k, v in inst.items() if k != "_self")
                    for lst in self.blueprints.values()
                    for inst in lst
                )
                print(
                    f"[blueprints] {len(self.blueprints):,} classes, "
                    f"{bp_inst:,} mesh instances ..."
                )
            b_root = bpy.data.collections.new("Blueprints")
            root.children.link(b_root)

            for bp_class, instances in self.blueprints.items():
                bp_coll = bpy.data.collections.new(bp_class)
                b_root.children.link(bp_coll)

                for i, instance in enumerate(instances, 1):
                    inst_coll = bpy.data.collections.new(f"{bp_class}.{i}")
                    bp_coll.children.link(inst_coll)

                    for mesh_name, transforms in instance.items():
                        if mesh_name == "_self":
                            continue
                        mesh_coll = bpy.data.collections.new(mesh_name)
                        inst_coll.children.link(mesh_coll)
                        j = 0
                        for t in transforms:
                            if not _keep(t):
                                continue
                            j += 1
                            place_mesh(
                                mesh_name, t, mesh_coll, self.meshes_dir,
                                f"{mesh_name}.{j}", shift=shift,
                            )
                            total += 1

        return total

    # -- Main build -----------------------------------------------------------

    def blend(self, terrain: bool = True) -> None:
        """
        Build a complete Blender scene from this map and save it as a
        compressed ``.blend`` file under ``<export_dir>/blend/``.

        Steps
        -----
        1. Reset Blender to an empty state.
        2. Optionally build the heightmap terrain mesh.
        3. Place all symbols, groups, and blueprint mesh instances that pass
           the ``include`` / ``exclude`` filters.
        4. Apply ``palette`` colors to loaded meshes.
        5. Save the ``.blend`` file.

        Parameters
        ----------
        terrain : bool
            When ``True`` (default) the heightmap PNG is loaded and a terrain
            mesh is created.  Pass ``False`` to skip terrain (useful when
            iterating quickly on props only).
        """
        mesh_cache.clear()
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        scene.name = self.name

        root = bpy.data.collections.new(self.name)
        scene.collection.children.link(root)

        # Shared collection cache; keyed by (root_name, *path_segments).
        coll_cache: Dict[Tuple[str, ...], bpy.types.Collection] = {}
        total = 0

        # -- Terrain ----------------------------------------------------------
        if terrain:
            if os.path.exists(self.heightmap_path):
                print("[terrain] Building terrain ...")
                t_coll = bpy.data.collections.new("Terrain")
                root.children.link(t_coll)
                create_terrain(self.heightmap_path, t_coll)
            else:
                print(
                    f"[terrain] Heightmap not found, skipping: "
                    f"{self.heightmap_path}"
                )

        # -- Symbols / Groups / Blueprints ------------------------------------
        total += self._populate(root, coll_cache)

        # -- Palette ----------------------------------------------------------
        self._apply_palette()

        loaded_ok = sum(1 for v in mesh_cache.values() if v is not None)
        loaded_err = sum(1 for v in mesh_cache.values() if v is None)
        print(
            f"\n  Objects placed : {total:,}\n"
            f"  Unique meshes  : {loaded_ok} loaded, "
            f"{loaded_err} missing/errored"
        )

        # -- Save -------------------------------------------------------------
        os.makedirs(os.path.dirname(self.blend_path), exist_ok=True)
        if os.path.exists(self.blend_path):
            os.remove(self.blend_path)
        print(f"\nSaving -> {self.blend_path} ...")
        bpy.ops.wm.save_as_mainfile(filepath=self.blend_path, compress=True)
        print("Done.\n")


# ------------------------------------------------------------------------------
#  Region / spill helpers
# ------------------------------------------------------------------------------

# World-space conversion between region-center JSON units and Blender meters.
# 1890 Blender meters correspond to 1776 region-center units along one hex edge.
REGION_UNIT_TO_METER = 1890.0 / 1776.0

# Hexagonal neighbor offsets in region-center units. Foxhole's map uses a
# "pointy-top-rotated" hex grid where the vertical neighbor spacing is
# 1776 units and the diagonal offsets are (±1540, ±888).
REGION_NEIGHBOR_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (0, -1776),
    (0, 1776),
    (-1540, -888),
    (1540, -888),
    (-1540, 888),
    (1540, 888),
)


def region_center_to_blender(center: List[float]) -> Tuple[float, float]:
    """
    Convert a (x, y) region-center JSON coordinate to Blender meters.

    Region-center Y grows southward (image convention).  Blender Y+ is
    northward (matches the UE→Blender Y flip in ``apply_ue_transform``), so
    Y is negated here to keep the two frames consistent.
    """
    return (
        center[0] * REGION_UNIT_TO_METER,
        -center[1] * REGION_UNIT_TO_METER,
    )


def find_region_neighbors(
    region_centers: Dict[str, List[float]],
    region_key: str,
    tol: float = 1.0,
) -> List[str]:
    """
    Return the 0–6 neighbor keys of *region_key* in *region_centers*.

    Two regions are neighbors when their centers differ by one of
    ``REGION_NEIGHBOR_OFFSETS`` (within *tol* units).  *region_key* must be
    a key of *region_centers* (lowercase, as stored in ``region_centers.json``).
    """
    cx, cy = region_centers[region_key]
    expected = [(cx + dx, cy + dy) for dx, dy in REGION_NEIGHBOR_OFFSETS]
    neighbors: List[str] = []
    for name, (nx, ny) in region_centers.items():
        if name == region_key:
            continue
        for ex, ey in expected:
            if abs(nx - ex) < tol and abs(ny - ey) < tol:
                neighbors.append(name)
                break
    return neighbors


def signed_distance_past_midline(
    point_xy: Tuple[float, float],
    own_center_xy: Tuple[float, float],
    neighbor_center_xy: Tuple[float, float],
) -> float:
    """
    Signed perpendicular distance from *point_xy* to the midline between
    *own_center_xy* and *neighbor_center_xy*, in the direction of the
    neighbor center.

    For hexagonal maps the border between two adjacent cells is the
    perpendicular bisector of the segment joining their centers, so this
    is the "distance past the shared border into the other cell".  The
    value is positive when *point_xy* lies on the neighbor side of the
    midline and negative when it lies on the own side.
    """
    mx = 0.5 * (own_center_xy[0] + neighbor_center_xy[0])
    my = 0.5 * (own_center_xy[1] + neighbor_center_xy[1])
    dx = neighbor_center_xy[0] - own_center_xy[0]
    dy = neighbor_center_xy[1] - own_center_xy[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0.0:
        return 0.0
    return ((point_xy[0] - mx) * dx + (point_xy[1] - my) * dy) / length


DEEP_WATER_DEPTH = 25.0


def spawn_deep_water_tree(
    water_root: bpy.types.Collection,
    parent_collection: bpy.types.Collection,
    depth: float = DEEP_WATER_DEPTH,
    color: str = "#000000",
    coll_name: str = "deep_water",
) -> List[bpy.types.Object]:
    """
    Mirror the *water_root* collection tree into a new sub-collection of
    *parent_collection* named *coll_name*.  Every water object is cloned
    *depth* meters below its surface counterpart and rendered with a black
    object-level material override.  Returns the list of deep-water clones.
    """
    if water_root is None:
        return []

    deep_root = bpy.data.collections.new(coll_name)
    parent_collection.children.link(deep_root)
    black_mat = _make_color_material(color)

    mapping: Dict[bpy.types.Collection, bpy.types.Collection] = {
        water_root: deep_root
    }

    def _mirror(src: bpy.types.Collection,
                dst: bpy.types.Collection) -> None:
        for child in src.children:
            new_child = bpy.data.collections.new(child.name)
            dst.children.link(new_child)
            mapping[child] = new_child
            _mirror(child, new_child)

    _mirror(water_root, deep_root)

    clones: List[bpy.types.Object] = []

    def _clone(src: bpy.types.Collection) -> None:
        dst = mapping[src]
        for obj in list(src.objects):
            if obj.data is None:
                continue
            clone = bpy.data.objects.new(f"{obj.name}_deep", obj.data)
            dst.objects.link(clone)
            clone.location = (
                obj.location.x,
                obj.location.y,
                obj.location.z - depth,
            )
            clone.rotation_mode = obj.rotation_mode
            clone.rotation_euler = obj.rotation_euler
            clone.scale = obj.scale
            if not clone.material_slots:
                clone.data.materials.append(None)
            clone.material_slots[0].link = "OBJECT"
            clone.material_slots[0].material = black_mat
            clones.append(clone)
        for child in src.children:
            _clone(child)

    _clone(water_root)
    print(f"  Deep water: {len(clones)} clone(s) placed {depth:.0f} m below")
    return clones


def spawn_deep_water(
    surface_objects: List[bpy.types.Object],
    parent_collection: bpy.types.Collection,
    depth: float = DEEP_WATER_DEPTH,
    color: str = "#000000",
    coll_name: str = "Deep_Water",
) -> int:
    """
    For each object in *surface_objects*, add a clone *depth* meters below it
    with a pure-black object-level material override.

    The clone shares mesh data with the surface object (cheap), lives in a
    new sub-collection named *coll_name* linked under *parent_collection*,
    and names itself ``<surface_name>_deep``.  The black material is applied
    as an object-linked slot override so the surface water keeps its blue
    mesh-level color.  Returns the number of clones placed.
    """
    if not surface_objects:
        return 0

    deep_coll = bpy.data.collections.new(coll_name)
    parent_collection.children.link(deep_coll)
    black_mat = _make_color_material(color)

    placed = 0
    for surf in surface_objects:
        if surf.data is None:
            continue
        clone = bpy.data.objects.new(f"{surf.name}_deep", surf.data)
        deep_coll.objects.link(clone)
        clone.location = (
            surf.location.x,
            surf.location.y,
            surf.location.z - depth,
        )
        clone.rotation_mode = surf.rotation_mode
        clone.rotation_euler = surf.rotation_euler
        clone.scale = surf.scale

        # Object-level material override: the mesh data-block keeps whatever
        # material the palette assigned (blue), but this specific object
        # renders with *black_mat*.  Requires the mesh to have at least one
        # material slot; create an empty one if the palette never matched.
        if not clone.material_slots:
            clone.data.materials.append(None)
        clone.material_slots[0].link = "OBJECT"
        clone.material_slots[0].material = black_mat
        placed += 1

    print(f"  Deep water: {placed} clone(s) placed {depth:.0f} m below")
    return placed


# ------------------------------------------------------------------------------
#  Top-down bake renderers (AO / heightmap / ID)
# ------------------------------------------------------------------------------

BAKE_IMG_SIZE = 2048
BAKE_PIXEL_SIZE_M = 1890.0 / 1776.0  # Blender meters per pixel
BAKE_CAM_Z = 5000.0


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data)) + tag + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png16_gray(path: str, arr: np.ndarray) -> None:
    """Write a 16-bit grayscale PNG from a uint16 array shape (h, w)."""
    h, w = arr.shape
    be = np.ascontiguousarray(arr.astype(">u2")).tobytes()
    row = w * 2
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        raw += be[y * row:(y + 1) * row]
    idat = zlib.compress(bytes(raw), level=6)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_png_chunk(
            b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 0, 0, 0, 0)
        ))
        f.write(_png_chunk(b"IDAT", idat))
        f.write(_png_chunk(b"IEND", b""))


def write_png8_rgb(path: str, arr: np.ndarray) -> None:
    """Write an 8-bit RGB PNG from a uint8 array shape (h, w, 3)."""
    h, w, _ = arr.shape
    data = np.ascontiguousarray(arr.astype(np.uint8)).tobytes()
    row = w * 3
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        raw += data[y * row:(y + 1) * row]
    idat = zlib.compress(bytes(raw), level=6)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_png_chunk(
            b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        ))
        f.write(_png_chunk(b"IDAT", idat))
        f.write(_png_chunk(b"IEND", b""))


def load_hex_mask(path: str, size: int = BAKE_IMG_SIZE) -> np.ndarray:
    """
    Load *path* as a boolean mask (True = inside hex).  Returned shape is
    (size, size) with row 0 at the top (image convention).
    """
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    px = px[::-1]  # Blender pixels are bottom-up; flip to top-down.
    mask = px[..., 0] > 0.5
    bpy.data.images.remove(img)
    if mask.shape != (size, size):
        ys = (np.arange(size) * mask.shape[0] // size)
        xs = (np.arange(size) * mask.shape[1] // size)
        mask = mask[ys][:, xs]
    return mask


def _bake_pixel_world_xy(size: int = BAKE_IMG_SIZE) -> Tuple[np.ndarray,
                                                             np.ndarray]:
    """Return (X, Y) 1D arrays of world meters for each pixel column / row."""
    c = size / 2.0
    j = np.arange(size, dtype=np.float32)
    X = (j + 0.5 - c) * BAKE_PIXEL_SIZE_M
    Y = (c - j - 0.5) * BAKE_PIXEL_SIZE_M  # row 0 = north
    return X, Y


def _set_hide(objs: List[bpy.types.Object], hide: bool, key: str) -> Dict:
    prev: Dict[str, bool] = {}
    for o in objs:
        prev[o.name] = getattr(o, key)
        setattr(o, key, hide)
    return prev


def _restore_hide(prev: Dict[str, bool], key: str) -> None:
    for name, v in prev.items():
        o = bpy.data.objects.get(name)
        if o is not None:
            setattr(o, key, v)


def _build_bvh_from_objs(
    objs: List[bpy.types.Object],
) -> Tuple[Optional[object], List[int]]:
    """
    Build a single ``BVHTree`` from the triangulated world-space geometry of
    *objs*.  Returns ``(bvh, tri_to_obj_idx)`` where ``tri_to_obj_idx[k]`` is
    the index into *objs* for the k-th triangle (aligned with the BVH's
    polygon ``index`` return value).  Returns ``(None, [])`` if *objs* is
    empty or yields no geometry.
    """
    from mathutils.bvhtree import BVHTree  # local import (module load cost)

    if not objs:
        return None, []

    depsgraph = bpy.context.evaluated_depsgraph_get()
    verts: List[Tuple[float, float, float]] = []
    tris: List[Tuple[int, int, int]] = []
    tri_to_obj_idx: List[int] = []

    for oi, obj in enumerate(objs):
        if obj is None or obj.type != 'MESH':
            continue
        ev = obj.evaluated_get(depsgraph)
        try:
            me = ev.to_mesh()
        except RuntimeError:
            continue
        if me is None:
            continue
        try:
            mw = obj.matrix_world.copy()
            base = len(verts)
            for v in me.vertices:
                wv = mw @ v.co
                verts.append((wv.x, wv.y, wv.z))
            me.calc_loop_triangles()
            for lt in me.loop_triangles:
                vs = lt.vertices
                tris.append((base + vs[0], base + vs[1], base + vs[2]))
                tri_to_obj_idx.append(oi)
        finally:
            ev.to_mesh_clear()

    if not tris:
        return None, []
    bvh = BVHTree.FromPolygons(verts, tris)
    return bvh, tri_to_obj_idx


def _bake_rows_parallel(
    mask: np.ndarray,
    row_fn: Callable[[int], None],
) -> None:
    """
    Run *row_fn(i)* for every row of *mask* that has at least one True pixel,
    using a thread pool.  ``mathutils.bvhtree.BVHTree.ray_cast`` releases the
    GIL internally so this gives a near-linear speedup.
    """
    from concurrent.futures import ThreadPoolExecutor
    rows = [i for i in range(mask.shape[0]) if mask[i].any()]
    workers = max(1, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(row_fn, rows))


def raycast_heightmap(
    output_path: str,
    mask: np.ndarray,
    visible_objs: List[bpy.types.Object],
    occluders: Optional[List[bpy.types.Object]] = None,
) -> None:
    """
    Cast one ray straight down per masked pixel and save the hit Z as a
    16-bit PNG (``value = z_m * 100 + 32768``).

    The BVH is built from *visible_objs* plus *occluders*.  When the topmost
    hit is an occluder the pixel stays 0 (void) - this models "deep water
    eats anything below it": anything under the deep-water plane is as good
    as gone from existence.
    """
    occluders = occluders or []
    seen = set()
    all_objs: List[bpy.types.Object] = []
    for o in list(visible_objs) + list(occluders):
        if o is None or o.name in seen:
            continue
        seen.add(o.name)
        all_objs.append(o)

    bvh, tri_to_obj_idx = _build_bvh_from_objs(all_objs)
    out = np.zeros((BAKE_IMG_SIZE, BAKE_IMG_SIZE), dtype=np.uint16)

    if bvh is not None and tri_to_obj_idx:
        occluder_names = {o.name for o in occluders}
        is_occ = np.asarray(
            [all_objs[oi].name in occluder_names for oi in range(len(all_objs))],
            dtype=bool,
        )
        tri_to_obj_np = np.asarray(tri_to_obj_idx, dtype=np.int32)
        X, Y = _bake_pixel_world_xy()
        direction = Vector((0.0, 0.0, -1.0))

        def _row(i: int) -> None:
            yv = float(Y[i])
            mrow = mask[i]
            row = out[i]
            origin = Vector((0.0, yv, BAKE_CAM_Z))
            cols = np.where(mrow)[0]
            for j in cols:
                origin.x = float(X[int(j)])
                loc, _n, tri_idx, _d = bvh.ray_cast(origin, direction)
                if loc is None or tri_idx is None:
                    continue
                oi = int(tri_to_obj_np[tri_idx])
                if is_occ[oi]:
                    continue  # hit deep water -> void
                v = int(round(loc.z * 100.0 + 32768.0))
                if v < 0:
                    v = 0
                elif v > 65535:
                    v = 65535
                row[int(j)] = v

        _bake_rows_parallel(mask, _row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_png16_gray(output_path, out)
    print(f"  Heightmap saved -> {output_path}")


def raycast_id_map(
    output_path: str,
    mask: np.ndarray,
    category_objects: Dict[str, List[bpy.types.Object]],
    category_colors: Dict[str, str],
    occluders: Optional[List[bpy.types.Object]] = None,
) -> None:
    """
    Cast one ray straight down per masked pixel; color each pixel by the
    category of the topmost hit object using *category_colors*.

    *occluders* are included in the BVH but have no color entry: when a ray
    hits an occluder first the pixel stays black (0, 0, 0).  This is how
    deep water hides everything beneath it.
    """
    flat_objs: List[bpy.types.Object] = []
    obj_color: List[Tuple[int, int, int]] = []
    for cat, objs in category_objects.items():
        hc = category_colors.get(cat, "#000000").lstrip("#")
        col = (int(hc[0:2], 16), int(hc[2:4], 16), int(hc[4:6], 16))
        for o in objs:
            flat_objs.append(o)
            obj_color.append(col)

    occluders = occluders or []
    seen = {o.name for o in flat_objs}
    occ_start = len(flat_objs)
    for o in occluders:
        if o is None or o.name in seen:
            continue
        seen.add(o.name)
        flat_objs.append(o)
        obj_color.append((0, 0, 0))

    bvh, tri_to_obj_idx = _build_bvh_from_objs(flat_objs)
    out = np.zeros((BAKE_IMG_SIZE, BAKE_IMG_SIZE, 3), dtype=np.uint8)

    if bvh is not None and tri_to_obj_idx:
        tri_to_obj_np = np.asarray(tri_to_obj_idx, dtype=np.int32)
        color_np = np.asarray(obj_color, dtype=np.uint8)  # (N, 3)
        X, Y = _bake_pixel_world_xy()
        direction = Vector((0.0, 0.0, -1.0))

        def _row(i: int) -> None:
            yv = float(Y[i])
            mrow = mask[i]
            row = out[i]
            origin = Vector((0.0, yv, BAKE_CAM_Z))
            cols = np.where(mrow)[0]
            for j in cols:
                origin.x = float(X[int(j)])
                loc, _n, tri_idx, _d = bvh.ray_cast(origin, direction)
                if loc is not None and tri_idx is not None:
                    oi = tri_to_obj_np[tri_idx]
                    row[int(j)] = color_np[oi]

        _bake_rows_parallel(mask, _row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_png8_rgb(output_path, out)
    print(f"  ID map saved -> {output_path}")


def raycast_binary_mask(
    output_path: str,
    mask: np.ndarray,
    occluders: List[bpy.types.Object],
    target_objs: List[bpy.types.Object],
) -> None:
    """
    Cast one ray straight down per masked pixel against a BVH built from the
    union of *occluders* and *target_objs*; write an 8-bit black/white PNG
    where white = the topmost hit is one of *target_objs*.

    This is the correct way to produce a "water" mask: non-water meshes
    (rocks / landscape / terrain / ...) correctly occlude water beneath
    them, so only pixels where water is the first thing a downward ray
    meets come out white.
    """
    seen = set()
    all_objs: List[bpy.types.Object] = []
    for o in list(target_objs) + list(occluders):
        if o is None or o.name in seen:
            continue
        seen.add(o.name)
        all_objs.append(o)

    bvh, tri_to_obj_idx = _build_bvh_from_objs(all_objs)
    out = np.zeros((BAKE_IMG_SIZE, BAKE_IMG_SIZE, 3), dtype=np.uint8)

    if bvh is not None and tri_to_obj_idx:
        target_names = {o.name for o in target_objs}
        is_target = np.asarray(
            [all_objs[oi].name in target_names for oi in range(len(all_objs))],
            dtype=bool,
        )
        tri_to_obj_np = np.asarray(tri_to_obj_idx, dtype=np.int32)
        X, Y = _bake_pixel_world_xy()
        direction = Vector((0.0, 0.0, -1.0))

        def _row(i: int) -> None:
            yv = float(Y[i])
            mrow = mask[i]
            row = out[i]
            origin = Vector((0.0, yv, BAKE_CAM_Z))
            cols = np.where(mrow)[0]
            for j in cols:
                origin.x = float(X[int(j)])
                loc, _n, tri_idx, _d = bvh.ray_cast(origin, direction)
                if loc is not None and tri_idx is not None:
                    oi = int(tri_to_obj_np[tri_idx])
                    if is_target[oi]:
                        row[int(j)] = (255, 255, 255)

        _bake_rows_parallel(mask, _row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_png8_rgb(output_path, out)
    print(f"  Binary mask saved -> {output_path}")


def _apply_mask_to_image_file(path: str, mask: np.ndarray) -> None:
    """Zero the RGB channels of *path* wherever *mask* is False."""
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    # Blender pixels are bottom-up; our mask is top-down.
    m = mask
    if m.shape != (h, w):
        ys = (np.arange(h) * m.shape[0] // h)
        xs = (np.arange(w) * m.shape[1] // w)
        m = m[ys][:, xs]
    m_bu = m[::-1]
    inv = ~m_bu
    px[inv, 0] = 0.0
    px[inv, 1] = 0.0
    px[inv, 2] = 0.0
    px[..., 3] = 1.0
    img.pixels = px.ravel().tolist()
    img.filepath_raw = path
    img.file_format = 'PNG'
    img.save()
    bpy.data.images.remove(img)


_GPU_CONFIGURED = False


def _enable_cycles_gpu(scene) -> None:
    """
    Configure Cycles to render on the GPU.  Tries OPTIX -> CUDA -> HIP ->
    ONEAPI -> METAL in order and enables every compatible device plus the
    CPU (Cycles happily uses both together).  Silently falls back to CPU
    when no GPU backend is available.  The preferences query runs once
    per process; the per-scene device flag is set on every call so newly
    opened .blend files inherit it.
    """
    global _GPU_CONFIGURED
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except (KeyError, AttributeError):
        return

    if not _GPU_CONFIGURED:
        chosen = None
        for backend in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
            try:
                prefs.compute_device_type = backend
            except TypeError:
                continue
            try:
                devs = prefs.get_devices_for_type(backend)
            except Exception:
                devs = []
            if devs:
                chosen = backend
                for d in devs:
                    d.use = True
                # Also enable CPU alongside the GPU for extra throughput.
                try:
                    for d in prefs.get_devices_for_type("CPU"):
                        d.use = True
                except Exception:
                    pass
                break
        if chosen is None:
            print("  [info] no Cycles GPU backend available; using CPU")
        else:
            names = [d.name for d in prefs.get_devices_for_type(chosen) if d.use]
            print(f"  [info] Cycles GPU backend: {chosen} ({', '.join(names)})")
        _GPU_CONFIGURED = True

    try:
        scene.cycles.device = 'GPU'
    except (AttributeError, TypeError):
        pass


def render_ao(
    output_path: str,
    mask: np.ndarray,
    hidden_objs: List[bpy.types.Object],
    samples: int = 32,
    ao_distance: float = 10.0,
) -> None:
    """
    Render a top-down AO bake (Cycles, ortho camera, 2048x2048) with every
    visible mesh forced white via a view-layer material override.  Every
    object in *hidden_objs* is hidden from rendering (pass surface water
    plus deep water here to exclude both).  *mask* is composited over the
    final PNG so everything outside the hex becomes black.
    """
    # Blender interprets scene.render.filepath relative to the currently
    # opened .blend; resolve up front so relative paths stay relative to CWD.
    output_path = os.path.abspath(output_path)

    scene = bpy.context.scene
    prev_hidden = _set_hide(hidden_objs, True, "hide_render")

    cam_data = bpy.data.cameras.new("AO_Cam")
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = BAKE_IMG_SIZE * BAKE_PIXEL_SIZE_M
    cam_data.clip_start = 0.1
    cam_data.clip_end = BAKE_CAM_Z * 2.0
    cam_obj = bpy.data.objects.new("AO_Cam", cam_data)
    cam_obj.location = (0.0, 0.0, BAKE_CAM_Z)
    cam_obj.rotation_euler = (0.0, 0.0, 0.0)
    scene.collection.objects.link(cam_obj)

    ao_mat = bpy.data.materials.new("AO_White")
    ao_mat.use_nodes = True
    nt = ao_mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out_node = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    ao_node = nt.nodes.new("ShaderNodeAmbientOcclusion")
    try:
        ao_node.samples = 16
    except AttributeError:
        pass
    ao_node.inputs["Distance"].default_value = ao_distance
    ao_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)

    # Slope shading: multiply AO by max(N.z, 0)^slope_power so that
    # perfectly flat (upward-facing) surfaces stay at 1.0 (pure white)
    # while any tilt away from vertical darkens the result. This gives
    # the image more three-dimensional relief than pure AO, which only
    # reacts to nearby occluders and leaves open slopes flat-white.
    slope_power = 0.5  # higher = steeper falloff on slopes

    geom = nt.nodes.new("ShaderNodeNewGeometry")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    clamp_z = nt.nodes.new("ShaderNodeMath")
    clamp_z.operation = 'MAXIMUM'
    clamp_z.inputs[1].default_value = 0.0
    pow_z = nt.nodes.new("ShaderNodeMath")
    pow_z.operation = 'POWER'
    pow_z.inputs[1].default_value = slope_power
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = 'MULTIPLY'

    nt.links.new(geom.outputs["Normal"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], clamp_z.inputs[0])
    nt.links.new(clamp_z.outputs["Value"], pow_z.inputs[0])
    # AO color (grayscale 0..1) feeds the first multiply socket; Blender
    # auto-converts color -> value using luminance, which is fine since
    # AO Color stays on the achromatic axis here.
    nt.links.new(ao_node.outputs["Color"], mul.inputs[0])
    nt.links.new(pow_z.outputs["Value"], mul.inputs[1])
    nt.links.new(mul.outputs["Value"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out_node.inputs["Surface"])

    prev = {
        "engine": scene.render.engine,
        "resx": scene.render.resolution_x,
        "resy": scene.render.resolution_y,
        "pct": scene.render.resolution_percentage,
        "override": scene.view_layers[0].material_override,
        "fp": scene.render.filepath,
        "fmt": scene.render.image_settings.file_format,
        "cmode": scene.render.image_settings.color_mode,
        "cdepth": scene.render.image_settings.color_depth,
        "cam": scene.camera,
        "film_transparent": scene.render.film_transparent,
        "view_transform": scene.view_settings.view_transform,
        "look": scene.view_settings.look,
        "exposure": scene.view_settings.exposure,
        "gamma": scene.view_settings.gamma,
    }

    # Filmic/AgX compresses highlights (emission 1.0 bakes to ~0.75).
    # Force a linear Standard view transform so pure-white emission
    # stays pure white in the PNG.
    try:
        scene.view_settings.view_transform = 'Standard'
    except TypeError:
        pass
    try:
        scene.view_settings.look = 'None'
    except TypeError:
        pass
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    scene.render.engine = 'CYCLES'
    try:
        scene.cycles.samples = samples
    except AttributeError:
        pass
    _enable_cycles_gpu(scene)
    scene.render.resolution_x = BAKE_IMG_SIZE
    scene.render.resolution_y = BAKE_IMG_SIZE
    scene.render.resolution_percentage = 100
    scene.view_layers[0].material_override = ao_mat
    scene.camera = cam_obj
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'BW'
    scene.render.image_settings.color_depth = '8'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

    # Restore
    _restore_hide(prev_hidden, "hide_render")
    scene.render.engine = prev["engine"]
    scene.render.resolution_x = prev["resx"]
    scene.render.resolution_y = prev["resy"]
    scene.render.resolution_percentage = prev["pct"]
    scene.view_layers[0].material_override = prev["override"]
    scene.render.filepath = prev["fp"]
    scene.render.image_settings.file_format = prev["fmt"]
    scene.render.image_settings.color_mode = prev["cmode"]
    scene.render.image_settings.color_depth = prev["cdepth"]
    scene.camera = prev["cam"]
    scene.render.film_transparent = prev["film_transparent"]
    try:
        scene.view_settings.view_transform = prev["view_transform"]
    except TypeError:
        pass
    try:
        scene.view_settings.look = prev["look"]
    except TypeError:
        pass
    scene.view_settings.exposure = prev["exposure"]
    scene.view_settings.gamma = prev["gamma"]

    bpy.data.objects.remove(cam_obj)
    bpy.data.cameras.remove(cam_data)
    bpy.data.materials.remove(ao_mat)

    _apply_mask_to_image_file(output_path, mask)
    print(f"  AO saved -> {output_path}")


# ------------------------------------------------------------------------------

def _build_category_lookup(
    catalogue: Dict[str, List[str]],
    categories: List[str],
) -> set:
    """Return a flat set of mesh names belonging to any of *categories*."""
    out: set = set()
    for cat in categories:
        out.update(catalogue.get(cat, []))
    return out


def build_region_with_spill(
    region_key: str,
    export_dir: str,
    region_centers: Dict[str, List[float]],
    catalogue: Dict[str, List[str]],
    json_name_map: Dict[str, str],
    spill_meters: float = 200.0,
    terrain_stride: int = 1,
) -> None:
    """
    Build a region with neighbor spill and save it under
    ``<export_dir>/blend_spill/<JsonName>.blend``.

    The focus region is loaded in full (terrain + water + rocks + glaciers +
    landscape_meshes + every symbol / group / blueprint mesh in its JSON)
    and colored using a fixed category palette.  For every hexagonal
    neighbor, only meshes in the ``landscape_meshes``, ``rocks`` and
    ``glaciers`` categories are placed; terrain and water meshes are
    explicitly excluded from neighbors.  Neighbor assets are kept only when
    their Blender-space position lies within *spill_meters* of the shared
    border (the perpendicular bisector of the center-to-center segment).

    Parameters
    ----------
    region_key : str
        Lowercase key from ``region_centers``.
    export_dir : str
        Root export directory.
    region_centers : dict[str, list[float]]
        ``region_centers.json`` contents (keys lowercase).
    catalogue : dict[str, list[str]]
        ``catalogue.json`` contents; must include ``water``, ``rocks``,
        ``glaciers`` and ``landscape_meshes``.
    json_name_map : dict[str, str]
        Mapping from lowercase region key to the original case-sensitive
        JSON filename stem (e.g. ``"oarbreakerhex" -> "OarbreakerHex"``).
    spill_meters : float
        Distance past the shared border, in Blender meters, within which
        neighbor assets are kept.
    terrain_stride : int
        Terrain grid stride (1 = 1 vertex per meter).
    """
    category_colors = {
        "water":            "#0000FF",
        "terrain":          "#00FF00",
        "rocks":            "#FF0000",
        "glaciers":         "#FFFFFF",
        "landscape_meshes": "#FF00FF",
    }
    neighbor_categories = ["landscape_meshes", "rocks", "glaciers"]

    allowed_neighbor_meshes = _build_category_lookup(
        catalogue, neighbor_categories
    )
    water_meshes = set(catalogue.get("water", []))

    # Per-mesh-name palette for Map._apply_palette.
    palette: Dict[str, str] = {}
    for cat, color in category_colors.items():
        if cat == "terrain":
            continue
        for mesh in catalogue.get(cat, []):
            palette[mesh] = color

    # Category lookup used by _populate_by_category.
    mesh_to_category: Dict[str, str] = {}
    for cat in ("water", "rocks", "glaciers", "landscape_meshes"):
        for m in catalogue.get(cat, []):
            mesh_to_category[m] = cat

    own_name = json_name_map[region_key]
    own_json = os.path.join(export_dir, "_json", f"{own_name}.json")
    if not os.path.exists(own_json):
        print(f"ERROR: JSON not found: {own_json}")
        return

    own_center = region_center_to_blender(region_centers[region_key])
    neighbors = find_region_neighbors(region_centers, region_key)

    print(f"=== {own_name} (spill) ===")
    print(f"  Center (Blender m): ({own_center[0]:.1f}, {own_center[1]:.1f})")
    print(f"  Neighbors ({len(neighbors)}): "
          f"{', '.join(json_name_map.get(n, n) for n in neighbors) or 'none'}")

    # Mesh-name allowlists for ``Map.__init__`` (exact names are literal
    # fnmatch patterns).  ``Map.__init__`` applies these filters globally,
    # dropping disallowed meshes inside symbols, groups, AND blueprints;
    # blueprint instances with no surviving children are removed entirely.
    focus_include = sorted(
        allowed_neighbor_meshes | water_meshes  # landscape + rocks + glaciers + water
    )
    neighbor_include = sorted(allowed_neighbor_meshes)
    neighbor_exclude = sorted(water_meshes)

    # Load focus region (filtered: water + glaciers + rocks + landscape) ----
    own_map = Map(own_json, export_dir, include=focus_include, palette=palette)

    mesh_cache.clear()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = own_name

    root = bpy.data.collections.new(own_name)
    scene.collection.children.link(root)

    coll_cache: Dict[Tuple[str, ...], bpy.types.Collection] = {}
    total = 0

    # Focus terrain (colored green) -------------------------------------------
    terrain_mat = _make_color_material(category_colors["terrain"])
    focus_terrain_objs: List[bpy.types.Object] = []
    if os.path.exists(own_map.heightmap_path):
        print("[terrain] Building focus terrain ...")
        t_coll = bpy.data.collections.new("terrain")
        root.children.link(t_coll)
        focus_terrain_objs.append(
            create_terrain(
                own_map.heightmap_path,
                t_coll,
                stride=terrain_stride,
                material=terrain_mat,
            )
        )
    else:
        print(f"[terrain] Missing heightmap: {own_map.heightmap_path}")

    # Focus meshes (categorized tree: water/rocks/glaciers/landscape_meshes) -
    focus_category_objs: Dict[str, List[bpy.types.Object]] = {}
    total += own_map._populate_by_category(
        root, coll_cache,
        mesh_to_category=mesh_to_category,
        category_objects=focus_category_objs,
    )
    surface_water_objs: List[bpy.types.Object] = focus_category_objs.get(
        "water", []
    )

    # Dedup lookup: some rocks / glaciers / landscape meshes are placed by
    # the game in both adjacent regions' data, producing two near-identical
    # copies at the border.  The duplicates z-fight under the AO bake and
    # render as heavy artifacts.  Two placements are considered duplicates
    # when they share the same mesh + scale + rotation (rounded to 3 dp)
    # and lie within ``DEDUP_XY_TOL`` meters on the XY plane.  Inter-region
    # coordinate rounding can drift by > 1 m, so a strict match isn't
    # enough: we use a 2 m spatial bucket grid and check the 3x3 neighbor
    # cells around each neighbor placement.
    DEDUP_CATS = ("rocks", "glaciers", "landscape_meshes")
    DEDUP_XY_TOL = 1.5
    DEDUP_BUCKET = 2.0  # cell size must be >= tolerance

    def _dedup_shape_key(o: bpy.types.Object) -> Tuple:
        return (
            o.data.name,
            round(o.scale.x, 3),
            round(o.scale.y, 3),
            round(o.scale.z, 3),
            round(o.rotation_euler.x, 3),
            round(o.rotation_euler.y, 3),
            round(o.rotation_euler.z, 3),
        )

    # shape_key -> {(ix, iy): [(x, y), ...]}
    focus_buckets: Dict[Tuple, Dict[Tuple[int, int],
                                    List[Tuple[float, float]]]] = {}
    for _cat in DEDUP_CATS:
        for _o in focus_category_objs.get(_cat, []):
            _sk = _dedup_shape_key(_o)
            _ix = int(_o.location.x // DEDUP_BUCKET)
            _iy = int(_o.location.y // DEDUP_BUCKET)
            focus_buckets.setdefault(_sk, {}).setdefault(
                (_ix, _iy), []
            ).append((_o.location.x, _o.location.y))

    _tol_sq = DEDUP_XY_TOL * DEDUP_XY_TOL

    def _is_duplicate(o: bpy.types.Object) -> bool:
        sk = _dedup_shape_key(o)
        cells = focus_buckets.get(sk)
        if not cells:
            return False
        ix = int(o.location.x // DEDUP_BUCKET)
        iy = int(o.location.y // DEDUP_BUCKET)
        ox, oy = o.location.x, o.location.y
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                pts = cells.get((ix + dx, iy + dy))
                if not pts:
                    continue
                for px, py in pts:
                    if (px - ox) ** 2 + (py - oy) ** 2 <= _tol_sq:
                        return True
        return False

    # -- Neighbors (spill) ----------------------------------------------------
    # Each neighbor is linked as its own top-level collection sibling of the
    # focus root.  This keeps the focus tree clean (terrain / water /
    # deep_water / rocks / glaciers / landscape_meshes) and lets the renderer
    # identify neighbor spill simply by scanning scene-level siblings.
    for neigh_key in neighbors:
        neigh_name = json_name_map.get(neigh_key)
        if neigh_name is None:
            print(f"  [WARN] No JSON for neighbor '{neigh_key}', skipped")
            continue
        neigh_json = os.path.join(export_dir, "_json", f"{neigh_name}.json")
        if not os.path.exists(neigh_json):
            print(f"  [WARN] Missing neighbor JSON: {neigh_json}")
            continue

        neigh_center = region_center_to_blender(region_centers[neigh_key])
        shift = (
            neigh_center[0] - own_center[0],
            neigh_center[1] - own_center[1],
            0.0,
        )
        print(
            f"[neighbor] {neigh_name}  "
            f"shift=({shift[0]:+.1f}, {shift[1]:+.1f}) m"
        )

        neigh_root = bpy.data.collections.new(neigh_name)
        scene.collection.children.link(neigh_root)

        # Border filter: keep when the world-space (post-shift) position
        # lies within *spill_meters* of the shared border on the neighbor
        # side.  Neighbor transforms are in the neighbor's local Blender
        # frame (its own center at 0,0); adding *shift* moves them into
        # the focus region's world frame, where the border is the
        # perpendicular bisector of own_center → neigh_center.
        #
        # For a neighbor-local point (lx, ly):
        #     d_world = (lx, ly) · v_hat + v_len / 2
        # with v = neigh_center - own_center.  Keep points with
        # ``d_world ≤ spill_meters`` (spill band just past the border).
        vx = neigh_center[0] - own_center[0]
        vy = neigh_center[1] - own_center[1]
        v_len = (vx * vx + vy * vy) ** 0.5 or 1.0
        vx_hat, vy_hat = vx / v_len, vy / v_len
        max_local_dot = spill_meters - 0.5 * v_len

        def _in_spill(t: list, _xh=vx_hat, _yh=vy_hat, _m=max_local_dot) -> bool:
            lx, ly = transform_to_blender_xy(t)
            return (lx * _xh + ly * _yh) <= _m

        # Load neighbor as a Map with global include/exclude filters.
        # This drops disallowed meshes everywhere (symbols, groups, AND
        # inside blueprints) — empty blueprint instances are removed by
        # ``Map.__init__`` so the collection tree stays clean.
        neigh_map = Map(
            neigh_json, export_dir,
            include=neighbor_include,
            exclude=neighbor_exclude,
        )
        neigh_cat_objs: Dict[str, List[bpy.types.Object]] = {}
        placed_n = neigh_map._populate_by_category(
            neigh_root,
            coll_cache,
            mesh_to_category=mesh_to_category,
            root_key_prefix=(neigh_name,),
            shift=shift,
            border_filter=_in_spill,
            announce=False,
            category_objects=neigh_cat_objs,
        )

        # Drop neighbor placements that collide with a focus placement of
        # the same mesh at the same world-space XY (duplicate rocks at the
        # shared border).  Keeping only one copy removes the z-fighting
        # that produces AO bake artifacts.
        removed = 0
        for _cat in DEDUP_CATS:
            for _o in neigh_cat_objs.get(_cat, []):
                if _is_duplicate(_o):
                    bpy.data.objects.remove(_o, do_unlink=True)
                    removed += 1
        placed_n -= removed
        if removed:
            print(
                f"  placed {placed_n:,} spill object(s) from {neigh_name} "
                f"({removed} duplicate(s) removed)"
            )
        else:
            print(f"  placed {placed_n:,} spill object(s) from {neigh_name}")
        total += placed_n

    # -- Palette on all loaded meshes (focus + neighbors) --------------------
    own_map._apply_palette()

    # -- Deep water: mirror the water tree 30 m below with a black override --
    water_root: Optional[bpy.types.Collection] = None
    for ch in root.children:
        if ch.name == "water":
            water_root = ch
            break
    spawn_deep_water_tree(water_root, root) if water_root else []

    loaded_ok = sum(1 for v in mesh_cache.values() if v is not None)
    loaded_err = sum(1 for v in mesh_cache.values() if v is None)
    print(
        f"\n  Objects placed : {total:,}\n"
        f"  Unique meshes  : {loaded_ok} loaded, "
        f"{loaded_err} missing/errored"
    )

    # Bakes (AO / heightmap / ID / water) live in 4_render_spills.py, which
    # reopens each saved .blend and runs them via the raycast_* / render_ao
    # helpers in this module.

    # -- Save ----------------------------------------------------------------
    out_dir = os.path.join(export_dir, "blend_spill")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(out_dir, f"{own_name}.blend"))
    if os.path.exists(out_path):
        os.remove(out_path)
    print(f"\nSaving -> {out_path} ...")
    bpy.ops.wm.save_as_mainfile(filepath=out_path, compress=True)
    print("Done.\n")
