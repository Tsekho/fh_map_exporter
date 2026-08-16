"""Blender scene helpers: materials, transforms, placement, terrain, splines, GN instancing."""

from math import radians
from typing import Callable, Dict, List, Optional, Tuple

import bpy
import numpy as np
from mathutils import Vector

from utils.png import read_png16_gray
from utils.psk import get_mesh, get_raw_psk
from utils.config import short_path
from utils.tui import log, warn


# ------------------------------------------------------------------------------
#  Materials
# ------------------------------------------------------------------------------


def _hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return (
        int(h[0:2], 16) / 255.0,
        int(h[2:4], 16) / 255.0,
        int(h[4:6], 16) / 255.0,
    )


def make_color_material(hex_color: str) -> bpy.types.Material:
    """Principled-BSDF with Base Color = hex_color (cached by name)."""
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


# ------------------------------------------------------------------------------
#  Transforms / collections
# ------------------------------------------------------------------------------


def apply_ue_transform(
    obj: bpy.types.Object,
    t: list,
    shift: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Apply UE [x,y,z, sx,sy,sz, pitch,yaw,roll] (cm, degrees) to obj.
    Y is flipped; Euler XYZ = (roll, -pitch, -yaw)."""
    x, y, z = t[0], t[1], t[2]
    sx, sy, sz = t[3], t[4], t[5]
    pitch, yaw, roll = t[6], t[7], t[8]
    obj.location = (x * 0.01 + shift[0], y * -0.01 + shift[1], z * 0.01 + shift[2])
    obj.scale = (sx, sy, sz)
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = (radians(roll), radians(-pitch), radians(-yaw))


def transform_to_blender_xy(t: list) -> Tuple[float, float]:
    return t[0] * 0.01, t[1] * -0.01


def ensure_collection(
    segments: List[str],
    parent: bpy.types.Collection,
    cache: Dict[Tuple[str, ...], bpy.types.Collection],
    root_key: Tuple[str, ...],
) -> bpy.types.Collection:
    """Walk segments, creating sub-collections as needed under parent."""
    current = parent
    for i, seg in enumerate(segments):
        key = root_key + tuple(segments[: i + 1])
        if key not in cache:
            coll = bpy.data.collections.new(seg)
            current.children.link(coll)
            cache[key] = coll
        current = cache[key]
    return current


def place_mesh(
    mesh_name: str,
    transform: list,
    collection: bpy.types.Collection,
    meshes_dir: str,
    obj_name: Optional[str] = None,
    shift: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    mesh = get_mesh(mesh_name, meshes_dir)
    if mesh is None:
        return
    obj = bpy.data.objects.new(obj_name or mesh_name, mesh)
    collection.objects.link(obj)
    apply_ue_transform(obj, transform, shift)


# ------------------------------------------------------------------------------
#  Terrain from 16-bit heightmap
# ------------------------------------------------------------------------------


def create_terrain(
    heightmap_path: str,
    collection: bpy.types.Collection,
    stride: int = 1,
    min_height: float = -100.0,
    shift: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    name: str = "Terrain",
    material: Optional[bpy.types.Material] = None,
) -> bpy.types.Object:
    """Build a grid mesh from a 16-bit PNG heightmap."""
    log(f"  Loading heightmap: {short_path(heightmap_path)}")
    w, h, pixels = read_png16_gray(heightmap_path)
    log(f"  Heightmap size: {w}x{h}")

    cx, cy = w // 2, h // 2
    sub = pixels[::stride, ::stride]
    rows, cols = sub.shape

    valid = sub != 0
    hm = sub.astype(np.float32)
    hm[~valid] = 32768.0
    hm = (hm - 32768.0) / 100.0
    valid = valid & (hm >= min_height)

    # Level N/S edge rows to their inward neighbor so hole-patching doesn't
    # propagate edge artefacts outward.
    any_valid_col = valid.any(axis=0)
    first_row = np.argmax(valid, axis=0)
    last_row = rows - 1 - np.argmax(valid[::-1], axis=0)
    lvl_cols = np.where(any_valid_col & (first_row + 1 <= last_row))[0]
    if lvl_cols.size:
        fr = first_row[lvl_cols]
        lr = last_row[lvl_cols]
        hm[fr, lvl_cols] = hm[fr + 1, lvl_cols]
        hm[lr, lvl_cols] = hm[lr - 1, lvl_cols]

    # Patch up-to-2-pixel N/S border holes, one row per pass.
    for _ in range(2):
        v_orig = valid.copy()
        hm_orig = hm.copy()
        src_s = v_orig[:-1, :] & ~v_orig[1:, :]
        if src_s.any():
            hm[1:, :][src_s] = hm_orig[:-1, :][src_s]
            valid[1:, :][src_s] = True
        src_n = v_orig[1:, :] & ~v_orig[:-1, :]
        if src_n.any():
            hm[:-1, :][src_n] = hm_orig[1:, :][src_n]
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

    log(f"  Terrain: {cols}x{rows} grid, {len(verts_np):,} verts, "
          f"{keep.sum():,} faces, stride={stride}")
    return obj


# ------------------------------------------------------------------------------
#  Spline-mesh deformation
# ------------------------------------------------------------------------------


def _deform_spline_verts(verts_ue: np.ndarray, entry: list) -> np.ndarray:
    """Deform local mesh verts (UE cm) along a cubic hermite spline.

    Replicates USplineMeshComponent::CalcSliceTransform. `entry` is the
    23-float layout from MapExporter.SerializeSpline (see the full layout
    comment in the exporter). Returns Nx3 UE cm world-space verts.
    """
    p0 = np.array(entry[0:3], dtype=np.float64)
    t0 = np.array(entry[3:6], dtype=np.float64)
    p1 = np.array(entry[6:9], dtype=np.float64)
    t1 = np.array(entry[9:12], dtype=np.float64)
    sr, so, ss = float(entry[12]), np.array(entry[13:15]), np.array(entry[15:17])
    er, eo, es = float(entry[17]), np.array(entry[18:20]), np.array(entry[20:22])
    axis = int(entry[22])

    axis_vals = verts_ue[:, axis]
    amin = float(axis_vals.min())
    amax = float(axis_vals.max())
    arange = amax - amin if amax - amin > 1e-9 else 1.0
    alpha = np.clip((axis_vals - amin) / arange, 0.0, 1.0)

    a2 = alpha * alpha
    a3 = a2 * alpha
    hp0 = (2 * a3) - (3 * a2) + 1
    hp1 = a3 - (2 * a2) + alpha
    hp2 = (-2 * a3) + (3 * a2)
    hp3 = a3 - a2
    ht0 = 6 * a2 - 6 * alpha
    ht1 = 3 * a2 - 4 * alpha + 1
    ht2 = -6 * a2 + 6 * alpha
    ht3 = 3 * a2 - 2 * alpha

    pos = (hp0[:, None] * p0 + hp1[:, None] * t0
           + hp2[:, None] * p1 + hp3[:, None] * t1)
    tan = (ht0[:, None] * p0 + ht1[:, None] * t0
           + ht2[:, None] * p1 + ht3[:, None] * t1)

    tn = np.linalg.norm(tan, axis=1, keepdims=True)
    tn = np.where(tn < 1e-9, 1.0, tn)
    splineDir = tan / tn

    up = np.array([0.0, 0.0, 1.0])
    baseX = np.cross(up, splineDir)
    bxn = np.linalg.norm(baseX, axis=1, keepdims=True)
    bxn = np.where(bxn < 1e-9, 1.0, bxn)
    baseX = baseX / bxn
    baseY = np.cross(splineDir, baseX)
    byn = np.linalg.norm(baseY, axis=1, keepdims=True)
    byn = np.where(byn < 1e-9, 1.0, byn)
    baseY = baseY / byn

    off = so + alpha[:, None] * (eo - so)
    pos = pos + off[:, 0:1] * baseX + off[:, 1:2] * baseY

    roll = sr + alpha * (er - sr)
    c = np.cos(roll)[:, None]
    s = np.sin(roll)[:, None]
    xVec = c * baseX - s * baseY
    yVec = c * baseY + s * baseX

    scale = ss + alpha[:, None] * (es - ss)
    scx = scale[:, 0:1]
    scy = scale[:, 1:2]

    local = verts_ue.copy()
    local[:, axis] = 0.0

    if axis == 0:
        world = (pos + splineDir * local[:, 0:1]
                 + xVec * local[:, 1:2] * scx
                 + yVec * local[:, 2:3] * scy)
    elif axis == 1:
        world = (pos + yVec * local[:, 0:1] * scy
                 + splineDir * local[:, 1:2]
                 + xVec * local[:, 2:3] * scx)
    else:
        world = (pos + xVec * local[:, 0:1] * scx
                 + yVec * local[:, 1:2] * scy
                 + splineDir * local[:, 2:3])
    return world


def place_spline_mesh(
    mesh_name: str,
    entry: list,
    collection: bpy.types.Collection,
    meshes_dir: str,
    obj_name: str,
    shift: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Optional[bpy.types.Object]:
    """Rebuild a SplineMeshComponent instance as a standalone Blender object.
    Each placement yields a unique mesh data-block (deformation is per-instance)."""
    raw = get_raw_psk(mesh_name, meshes_dir)
    if raw is None:
        return None
    verts_ue, tris = raw

    try:
        world_ue = _deform_spline_verts(verts_ue, entry)
    except Exception as exc:
        warn(f"  [WARN] Spline deform failed ({mesh_name}): {exc}")
        return None

    verts_bl = np.empty_like(world_ue)
    verts_bl[:, 0] = world_ue[:, 0] * 0.01 + shift[0]
    verts_bl[:, 1] = world_ue[:, 1] * -0.01 + shift[1]
    verts_bl[:, 2] = world_ue[:, 2] * 0.01 + shift[2]

    mesh = bpy.data.meshes.new(obj_name)
    mesh.from_pydata([tuple(v) for v in verts_bl.tolist()], [], tris)
    mesh.update()
    obj = bpy.data.objects.new(obj_name, mesh)
    collection.objects.link(obj)
    return obj


# ------------------------------------------------------------------------------
#  Geometry-Nodes point-cloud instancing
# ------------------------------------------------------------------------------
#
#  For large N, one bpy.data.objects per instance is super-linear in depsgraph
#  overhead. Instead we emit one point-cloud mesh per placement site with
#  per-point 'rotation' and 'scale' attributes, driven by a shared GN group
#  that calls Instance-on-Points against a hidden template object.


_instancer_templates: Dict[str, bpy.types.Object] = {}
_INSTANCER_NG_NAME = "FH_InstancerOnPoints"
_INSTANCER_TEMPLATES_COLL = "_Templates"


def _get_instancer_node_group() -> bpy.types.NodeTree:
    ng = bpy.data.node_groups.get(_INSTANCER_NG_NAME)
    if ng is not None:
        return ng

    ng = bpy.data.node_groups.new(_INSTANCER_NG_NAME, "GeometryNodeTree")

    # Socket declaration - 4.x uses interface, 3.x uses inputs/outputs.
    if hasattr(ng, "interface"):
        ng.interface.new_socket("Geometry", in_out='INPUT',
                                socket_type='NodeSocketGeometry')
        ng.interface.new_socket("Object", in_out='INPUT',
                                socket_type='NodeSocketObject')
        ng.interface.new_socket("Geometry", in_out='OUTPUT',
                                socket_type='NodeSocketGeometry')
    else:
        ng.inputs.new("NodeSocketGeometry", "Geometry")
        ng.inputs.new("NodeSocketObject", "Object")
        ng.outputs.new("NodeSocketGeometry", "Geometry")

    nodes, links = ng.nodes, ng.links
    gin = nodes.new("NodeGroupInput"); gin.location = (-600, 0)
    gout = nodes.new("NodeGroupOutput"); gout.location = (600, 0)

    obj_info = nodes.new("GeometryNodeObjectInfo"); obj_info.location = (-300, -100)
    try:
        obj_info.inputs["As Instance"].default_value = True
    except KeyError:
        pass

    na_rot = nodes.new("GeometryNodeInputNamedAttribute")
    na_rot.location = (-300, -280)
    na_rot.data_type = 'FLOAT_VECTOR'
    na_rot.inputs["Name"].default_value = "rotation"

    na_scl = nodes.new("GeometryNodeInputNamedAttribute")
    na_scl.location = (-300, -420)
    na_scl.data_type = 'FLOAT_VECTOR'
    na_scl.inputs["Name"].default_value = "scale"

    iop = nodes.new("GeometryNodeInstanceOnPoints")
    iop.location = (100, -50)

    links.new(gin.outputs["Geometry"], iop.inputs["Points"])
    links.new(gin.outputs["Object"], obj_info.inputs["Object"])
    geom_out = obj_info.outputs.get("Geometry") or obj_info.outputs.get("Instances")
    links.new(geom_out, iop.inputs["Instance"])
    links.new(na_scl.outputs["Attribute"], iop.inputs["Scale"])

    rot_socket = iop.inputs["Rotation"]
    if rot_socket.type == 'VECTOR':
        links.new(na_rot.outputs["Attribute"], rot_socket)
    else:
        try:
            e2r = nodes.new("FunctionNodeEulerToRotation")
            e2r.location = (-60, -320)
            links.new(na_rot.outputs["Attribute"], e2r.inputs["Euler"])
            links.new(e2r.outputs["Rotation"], rot_socket)
        except RuntimeError:
            links.new(na_rot.outputs["Attribute"], rot_socket)

    links.new(iop.outputs["Instances"], gout.inputs["Geometry"])
    return ng


def _ensure_templates_collection() -> bpy.types.Collection:
    c = bpy.data.collections.get(_INSTANCER_TEMPLATES_COLL)
    if c is not None:
        return c
    c = bpy.data.collections.new(_INSTANCER_TEMPLATES_COLL)
    bpy.context.scene.collection.children.link(c)
    return c


def _get_instancer_template(
    mesh_name: str, mesh: bpy.types.Mesh
) -> bpy.types.Object:
    obj = _instancer_templates.get(mesh_name)
    if obj is not None and obj.name in bpy.data.objects:
        return obj
    tpl_coll = _ensure_templates_collection()
    obj = bpy.data.objects.new(f"_tpl__{mesh_name}", mesh)
    obj.hide_render = True
    obj.hide_select = True
    tpl_coll.objects.link(obj)
    # hide_viewport on templates disables depsgraph eval and breaks Object-Info;
    # hide_set() is view-layer only and safe.
    try:
        obj.hide_set(True)
    except RuntimeError:
        pass
    _instancer_templates[mesh_name] = obj
    return obj


def _set_modifier_object_input(mod, obj) -> None:
    """Modifier inputs keyed by socket identifier - differs between 3.x / 4.x."""
    ng = mod.node_group
    if hasattr(ng, "interface"):
        for item in ng.interface.items_tree:
            if (getattr(item, "in_out", None) == 'INPUT'
                    and getattr(item, "socket_type", None) == 'NodeSocketObject'):
                mod[item.identifier] = obj
                return
    for s in ng.inputs:
        if s.type == 'OBJECT':
            mod[s.identifier] = obj
            return


def place_instanced(
    mesh_name: str,
    transforms: List[list],
    collection: bpy.types.Collection,
    meshes_dir: str,
    obj_name: str,
    shift: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    border_filter: Optional[Callable[[list], bool]] = None,
) -> Optional[bpy.types.Object]:
    """Place N copies of mesh_name as a single Geometry-Nodes instancer."""
    if border_filter is not None:
        transforms = [t for t in transforms if border_filter(t)]
    n = len(transforms)
    if n == 0:
        return None

    mesh = get_mesh(mesh_name, meshes_dir)
    if mesh is None:
        return None

    tarr = np.asarray(transforms, dtype=np.float64)
    positions = np.empty((n, 3), dtype=np.float32)
    positions[:, 0] = tarr[:, 0] * 0.01 + shift[0]
    positions[:, 1] = tarr[:, 1] * -0.01 + shift[1]
    positions[:, 2] = tarr[:, 2] * 0.01 + shift[2]

    scales = tarr[:, 3:6].astype(np.float32)

    rots = np.empty((n, 3), dtype=np.float32)
    rots[:, 0] = np.radians(tarr[:, 8])   # X = roll
    rots[:, 1] = np.radians(-tarr[:, 6])  # Y = -pitch
    rots[:, 2] = np.radians(-tarr[:, 7])  # Z = -yaw

    pm = bpy.data.meshes.new(obj_name)
    pm.vertices.add(n)
    pm.vertices.foreach_set("co", positions.ravel())
    pm.update()

    rot_attr = pm.attributes.new("rotation", type='FLOAT_VECTOR', domain='POINT')
    rot_attr.data.foreach_set("vector", rots.ravel())
    scl_attr = pm.attributes.new("scale", type='FLOAT_VECTOR', domain='POINT')
    scl_attr.data.foreach_set("vector", scales.ravel())

    display_name = obj_name if n == 1 else f"{obj_name} (×{n})"
    obj = bpy.data.objects.new(display_name, pm)
    collection.objects.link(obj)

    template = _get_instancer_template(mesh_name, mesh)
    mod = obj.modifiers.new("Instancer", type='NODES')
    mod.node_group = _get_instancer_node_group()
    _set_modifier_object_input(mod, template)
    return obj


def clear_instancer_templates() -> None:
    _instancer_templates.clear()
