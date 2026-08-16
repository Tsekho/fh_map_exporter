"""Map: a single Foxhole map, drives Blender scene construction."""

import fnmatch
import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import bpy

from utils.blender import (
    apply_ue_transform,
    clear_instancer_templates,
    create_terrain,
    ensure_collection,
    make_color_material,
    place_instanced,
    place_mesh,
    place_spline_mesh,
)
from utils.psk import clear_caches, get_mesh, mesh_cache
from utils.config import short_path
from utils.tui import log


class Map:
    """One Foxhole map.

    include/exclude: fnmatch patterns against mesh names (include first,
    then exclude). Blueprint _self keys are never filtered.
    palette: {fnmatch_pattern: '#RRGGBB'}; first matching pattern wins.
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
            inc = (f"{len(self.include)} pattern(s)"
                   if len(self.include) > 8 else (self.include or "none"))
            exc = (f"{len(self.exclude)} pattern(s)"
                   if len(self.exclude) > 8 else (self.exclude or "none"))
            log(f"  Filters - include: {inc}, exclude: {exc}")

        raw_symbols: Dict[str, list] = data.get("symbols", {})
        raw_groups: Dict[str, list] = data.get("groups", {})
        raw_blueprints: Dict[str, list] = data.get("blueprints", {})
        raw_splines: Dict[str, list] = data.get("splines", {})

        self.symbols = {k: v for k, v in raw_symbols.items() if self._should_place(k)}
        self.groups = {k: v for k, v in raw_groups.items() if self._should_place(k)}
        self.splines = {k: v for k, v in raw_splines.items() if self._should_place(k)}

        filtered_bps: Dict[str, list] = {}
        for bp_class, instances in raw_blueprints.items():
            kept: list = []
            for inst in instances:
                filtered = {
                    k: v for k, v in inst.items()
                    if k == "_self" or self._should_place(k)
                }
                if any(k != "_self" for k in filtered):
                    kept.append(filtered)
            if kept:
                filtered_bps[bp_class] = kept
        self.blueprints = filtered_bps

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

    def _should_place(self, mesh_name: str) -> bool:
        if self.include and not any(fnmatch.fnmatch(mesh_name, p) for p in self.include):
            return False
        if self.exclude and any(fnmatch.fnmatch(mesh_name, p) for p in self.exclude):
            return False
        return True

    def _apply_palette(self) -> None:
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
                        mat_cache[hex_color] = make_color_material(hex_color)
                    mat = mat_cache[hex_color]
                    if mesh.materials:
                        mesh.materials[0] = mat
                    else:
                        mesh.materials.append(mat)
                    colored += 1
                    break
        log(f"  Palette applied: {colored} mesh(es) colored")

    def _flatten(self) -> Dict[str, List[list]]:
        """Merge symbols + groups + blueprint meshes into {mesh: [transforms]}."""
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
        """Flat placement tree keyed by mesh name (split on '__')."""
        flat = self._flatten()
        if announce:
            inst_count = sum(len(v) for v in flat.values())
            log(f"[meshes] {len(flat):,} unique meshes, "
                  f"{inst_count:,} instances ...")

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
        """Placement grouped by category: <root>/<category>/<seg_1>/..."""
        flat = self._flatten()
        if announce:
            total_inst = sum(len(v) for k, v in flat.items() if k in mesh_to_category)
            placed_meshes = sum(1 for k in flat if k in mesh_to_category)
            log(f"[meshes] {placed_meshes:,} unique meshes, "
                  f"{total_inst:,} instances ...")

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
        """Standard Symbols/Groups/Blueprints tree under root."""
        total = 0

        def _keep(t: list) -> bool:
            return border_filter is None or border_filter(t)

        if self.symbols:
            if announce:
                s_inst = sum(len(v) for v in self.symbols.values())
                log(f"[symbols] {len(self.symbols):,} unique meshes, "
                      f"{s_inst:,} instances ...")
            s_root = bpy.data.collections.new("Symbols")
            root.children.link(s_root)
            cache_root = root_key_prefix + ("Symbols",)
            for mesh_name, instances in self.symbols.items():
                sp = mesh_name.split("__")
                leaf = ensure_collection(sp, s_root, coll_cache, cache_root)
                obj = place_instanced(
                    mesh_name, instances, leaf, self.meshes_dir,
                    sp[-1], shift=shift, border_filter=border_filter,
                )
                if obj is not None:
                    total += len(obj.data.vertices)

        if self.groups:
            if announce:
                g_inst = sum(len(v) for v in self.groups.values())
                log(f"[groups]  {len(self.groups):,} unique meshes, "
                      f"{g_inst:,} instances ...")
            g_root = bpy.data.collections.new("Groups")
            root.children.link(g_root)
            cache_root = root_key_prefix + ("Groups",)
            for mesh_name, instances in self.groups.items():
                sp = mesh_name.split("__")
                leaf = ensure_collection(sp, g_root, coll_cache, cache_root)
                obj = place_instanced(
                    mesh_name, instances, leaf, self.meshes_dir,
                    sp[-1], shift=shift, border_filter=border_filter,
                )
                if obj is not None:
                    total += len(obj.data.vertices)

        if self.splines:
            if announce:
                sp_inst = sum(len(v) for v in self.splines.values())
                log(f"[splines] {len(self.splines):,} unique meshes, "
                      f"{sp_inst:,} instances ...")
            sp_root = bpy.data.collections.new("Splines")
            root.children.link(sp_root)
            cache_root = root_key_prefix + ("Splines",)
            for mesh_name, instances in self.splines.items():
                sp = mesh_name.split("__")
                leaf = ensure_collection(sp, sp_root, coll_cache, cache_root)
                i = 0
                for entry in instances:
                    i += 1
                    place_spline_mesh(
                        mesh_name, entry, leaf, self.meshes_dir,
                        f"{sp[-1]}.{i}", shift=shift,
                    )
                    total += 1

        if self.blueprints:
            if announce:
                bp_inst = sum(
                    sum(len(v) for k, v in inst.items() if k != "_self")
                    for lst in self.blueprints.values()
                    for inst in lst
                )
                log(f"[blueprints] {len(self.blueprints):,} classes, "
                      f"{bp_inst:,} mesh instances ...")
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

    def blend(self, terrain: bool = True) -> None:
        """Build and save a complete Blender scene for this map."""
        clear_caches()
        clear_instancer_templates()
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        scene.name = self.name

        root = bpy.data.collections.new(self.name)
        scene.collection.children.link(root)

        coll_cache: Dict[Tuple[str, ...], bpy.types.Collection] = {}
        total = 0

        if terrain:
            if os.path.exists(self.heightmap_path):
                log("[terrain] Building terrain ...")
                t_coll = bpy.data.collections.new("Terrain")
                root.children.link(t_coll)
                create_terrain(self.heightmap_path, t_coll)
            else:
                log(f"[terrain] Heightmap not found, skipping: "
                      f"{short_path(self.heightmap_path)}")

        total += self._populate(root, coll_cache)
        self._apply_palette()

        loaded_ok = sum(1 for v in mesh_cache.values() if v is not None)
        loaded_err = sum(1 for v in mesh_cache.values() if v is None)
        log(
            f"\n  Objects placed : {total:,}\n"
            f"  Unique meshes  : {loaded_ok} loaded, "
            f"{loaded_err} missing/errored"
        )

        os.makedirs(os.path.dirname(self.blend_path), exist_ok=True)
        if os.path.exists(self.blend_path):
            os.remove(self.blend_path)
        log(f"\nSaving -> {short_path(self.blend_path)} ...")
        bpy.ops.wm.save_as_mainfile(filepath=self.blend_path, compress=True)
        log("Done.\n")
