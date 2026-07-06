using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CUE4Parse.FileProvider;
using CUE4Parse.UE4.Assets;
using CUE4Parse.UE4.Assets.Exports;
using CUE4Parse.UE4.Assets.Exports.Component.SplineMesh;
using CUE4Parse.UE4.Assets.Exports.Component.StaticMesh;
using CUE4Parse.UE4.Assets.Exports.SkeletalMesh;
using CUE4Parse.UE4.Assets.Exports.StaticMesh;
using CUE4Parse.UE4.Assets.Objects;
using CUE4Parse.UE4.Objects.Core.Math;
using CUE4Parse.UE4.Objects.UObject;
using CUE4Parse.UE4.Objects.Engine;
using CUE4Parse.UE4.Assets.Exports.Animation;
using CUE4Parse_Conversion;
using CUE4Parse_Conversion.Meshes;
using Newtonsoft.Json.Linq;
using Serilog;
using Microsoft.VisualBasic;
using System.Buffers.Binary;
using System.Reflection.Metadata;

namespace Exporter
{
    internal sealed class TemplateData
    {
        public FPackageIndex? MeshIndex;
        public double[] RelLoc   = [0, 0, 0];
        public double[] RelScale = [1, 1, 1];
        public double[] RelRot   = [0, 0, 0];
        public string?  SocketName;
    }

    internal sealed class MapItem
    {
        public int         Id;
        public UObject     Export      = null!;
        public string      ExportType  = string.Empty;
        public bool        IsBlueprint;
        public string?     BlueprintPath;

        public double[] RelLoc   = [0, 0, 0];
        public double[] RelScale = [1, 1, 1];
        public double[] RelRot   = [0, 0, 0];

        public readonly List<double[]> RelInstances = [];

        public string?        Mesh;
        public FPackageIndex? MeshIndex;

        public MapItem?       Parent;
        public MapItem?       CreatedBy;
        public List<MapItem>  Created = [];

        // Socket on Parent's skeletal mesh this component is attached to (via SCS AttachToName).
        public string?        SocketName;

        // Global (world-space) – null until Globalize() runs
        public double[]?   GlobLoc;
        public double[]?   GlobScale;
        public double[,]?  GlobRot;
        public readonly List<double[]> GlobInstances = [];
    }

    internal sealed class MapExporter
    {
        private readonly DefaultFileProvider _provider;
        private readonly string              _exportFolder;

        private readonly Dictionary<UObject, MapItem> _objToItem
            = new(ReferenceEqualityComparer.Instance);

        private readonly Dictionary<string, Dictionary<string, TemplateData>> _bpTemplateCache = [];

        // Cache: mesh package path → (socket name → socket transform in mesh-local space).
        // Null value means the mesh failed to load or had no sockets.
        private readonly Dictionary<string, Dictionary<string, (double[] loc, double[,] rot, double[] scl)>?> _socketCache = [];

        private readonly bool _exportTextures;

        public MapExporter(DefaultFileProvider provider, string exportFolder, bool exportTextures = false)
        {
            _provider       = provider;
            _exportFolder   = exportFolder;
            _exportTextures = exportTextures;
        }

        public void Export(string assetPath)
        {
            if (assetPath.EndsWith(".umap", StringComparison.OrdinalIgnoreCase))
                assetPath = assetPath[..^5];

            Log.Information("Loading package: {0}", assetPath);

            if (!_provider.TryLoadPackage(assetPath, out var pkg))
            {
                Log.Error("Package not found: {0}", assetPath);
                return;
            }

            if (pkg is not Package p)
            {
                Log.Error("Package is not a standard UE4 Package – aborting.");
                return;
            }

            string mapName = assetPath.Split('/', '\\').Last();
            Log.Information("Processing '{0}'  ({1} exports)", mapName, p.ExportMap.Length);

            // 1. Build flat item list
            var allItems = new List<MapItem>(p.ExportMap.Length);
            _objToItem.Clear();

            for (int i = 0; i < p.ExportMap.Length; i++)
            {
                string exportTypeName = p.ExportMap[i].ClassName;

                UObject? obj = null;
                try { obj = p.ExportsLazy[i].Value; }
                catch (Exception ex) { Log.Verbose("Skip export {0}: {1}", i, ex.Message); }

                if (obj == null)
                {
                    allItems.Add(new MapItem { Id = i, ExportType = exportTypeName });
                    continue;
                }

                bool isBp  = exportTypeName.EndsWith("_C", StringComparison.Ordinal);
                string? bpPath = isBp ? GetBlueprintPath(obj.Class) : null;

                var item = new MapItem
                {
                    Id            = i,
                    Export        = obj,
                    ExportType    = exportTypeName,
                    IsBlueprint   = isBp,
                    BlueprintPath = bpPath,
                };
                allItems.Add(item);
                _objToItem[obj] = item;
            }

            // 2. Init
            Log.Information("Initialising items…");
            foreach (var item in allItems)
                if (item.Export != null) InitItem(item);

            // 3. Populate blueprint defaults (transforms from SCS/CDO)
            Log.Information("Populating blueprint defaults…");
            foreach (var item in allItems)
                if (item.Export != null) PopulateBlueprintDefaults(item);

            // 4. Globalize
            Log.Information("Globalizing transforms…");
            foreach (var item in allItems)
                if (item.Export != null) Globalize(item);

            // 4b. Apply per-region Z offset to all globalised positions.
            double zOffsetCm = Constants.GetHeightOffset(mapName);
            if (zOffsetCm != 0.0)
            {
                Log.Information("Applying Z offset of {0} cm to map '{1}'.", zOffsetCm, mapName);
                foreach (var item in allItems)
                {
                    if (item.GlobLoc != null)
                        item.GlobLoc[2] += zOffsetCm;

                    foreach (var inst in item.GlobInstances)
                        inst[2] += zOffsetCm;
                }
            }

            // 5. Serialize
            Log.Information("Serializing…");
            var symbols    = new Dictionary<string, List<double[]>>();
            var groups     = new Dictionary<string, List<double[]>>();
            var blueprints = new Dictionary<string, List<JObject>>();
            var splines    = new Dictionary<string, List<double[]>>();

            Directory.CreateDirectory(Path.Combine(_exportFolder, "_meshes"));

            foreach (var item in allItems)
                if (item.Export != null) SerializeItem(item, symbols, groups, blueprints, splines);

            Log.Information("symbols={0}  groups={1}  blueprints={2}  splines={3}",
                symbols.Count, groups.Count, blueprints.Count, splines.Count);

            // 6. Export textures (optional)
            if (_exportTextures)
            {
                Log.Information("Stitching landscape textures…");
                LandscapeStitcher.StitchFromPackage(p, _exportFolder, mapName);
            }

            // 7. Write JSON
            var output = new JObject
            {
                ["symbols"]    = ToJObject(symbols),
                ["blueprints"] = BlueprintsToJObject(blueprints),
                ["groups"]     = ToJObject(groups),
                ["splines"]    = ToJObject(splines),
            };

            // Proxy maps are lightweight blending regions; keep their JSON in a
            // dedicated subfolder so they don't mix with real region exports.
            string jsonDir = Path.Combine(_exportFolder, "_json");
            if (mapName.StartsWith("Proxy", StringComparison.OrdinalIgnoreCase))
                jsonDir = Path.Combine(jsonDir, "proxies");
            Directory.CreateDirectory(jsonDir);
            string outPath = Path.Combine(jsonDir, mapName + ".json");
            File.WriteAllText(outPath, JsonOutput.Serialize(output));
            Log.Information("Wrote {0}", outPath);
        }

        private void InitItem(MapItem item)
        {
            var exp = item.Export;

            var rl = exp.GetOrDefault<FVector>("RelativeLocation",  FVector.ZeroVector);
            var rs = exp.GetOrDefault<FVector>("RelativeScale3D",   FVector.OneVector);
            var rr = exp.GetOrDefault<FRotator>("RelativeRotation", FRotator.ZeroRotator);

            item.RelLoc   = [rl.X, rl.Y, rl.Z];
            item.RelScale = [rs.X, rs.Y, rs.Z];
            item.RelRot   = [rr.Pitch, rr.Yaw, rr.Roll];

            var ap = exp.GetOrDefault<FPackageIndex>("AttachParent");
            var rc = exp.GetOrDefault<FPackageIndex>("RootComponent");
            var parentIdx = (ap != null && !ap.IsNull) ? ap
                          : (rc != null && !rc.IsNull) ? rc
                          : null;
            if (parentIdx != null)
                item.Parent = ResolveToItem(parentIdx);

            // HISM / foliage instances
            if (exp is UInstancedStaticMeshComponent hism && hism.PerInstanceSMData != null)
            {
                foreach (var inst in hism.PerInstanceSMData)
                {
                    var t = inst.TransformData;
                    item.RelInstances.Add([
                        t.Translation.X, t.Translation.Y, t.Translation.Z,
                        t.Scale3D.X,     t.Scale3D.Y,     t.Scale3D.Z,
                        t.Rotation.X,    t.Rotation.Y,    t.Rotation.Z, t.Rotation.W,
                    ]);
                }
            }

            bool isFoliage = item.ExportType.Contains("FoliageInstancedStaticMesh",
                                                       StringComparison.OrdinalIgnoreCase);
            if (!isFoliage || item.RelInstances.Count > 0)
            {
                var mi = exp.GetOrDefault<FPackageIndex>("StaticMesh");
                if (mi == null || mi.IsNull)
                    mi = exp.GetOrDefault<FPackageIndex>("SkeletalMesh");
                if (mi != null && !mi.IsNull)
                {
                    item.Mesh      = ResolveMeshPath(mi);
                    item.MeshIndex = mi;
                }
            }

            if (!item.IsBlueprint &&
                exp.GetOrDefault<FPackageIndex[]>("BlueprintCreatedComponents") == null)
                return;

            var seenIds = new HashSet<int>();

            var bcc = exp.GetOrDefault<FPackageIndex[]>("BlueprintCreatedComponents");
            if (bcc != null)
                foreach (var ci in bcc)
                    ClaimComponent(item, ci, seenIds);

            if (item.IsBlueprint)
            {
                foreach (var prop in exp.Properties)
                {
                    if (prop.Name.Text == "BlueprintCreatedComponents") continue;
                    if (prop.Tag == null) continue;
                    FPackageIndex? ci = null;
                    try { ci = prop.Tag.GetValue(typeof(FPackageIndex)) as FPackageIndex; }
                    catch { /* not an object ref */ }
                    if (ci != null) ClaimComponent(item, ci, seenIds);
                }
            }
        }

        private void ClaimComponent(MapItem owner, FPackageIndex? ci, HashSet<int> seenIds)
        {
            if (ci == null || ci.IsNull) return;
            var comp = ResolveToItem(ci);
            if (comp == null || seenIds.Contains(comp.Id)) return;

            comp.CreatedBy = owner;
            seenIds.Add(comp.Id);

            if (Constants.SkipTypes.Contains(comp.ExportType)) return;
            if (Constants.SkipComponentNames.Contains(comp.Export.Name)) return;

            owner.Created.Add(comp);
        }

        private void PopulateBlueprintDefaults(MapItem item)
        {
            if (!item.IsBlueprint || item.BlueprintPath == null) return;

            if (!_bpTemplateCache.TryGetValue(item.BlueprintPath, out var cache))
            {
                cache = BuildTemplateCache(item.BlueprintPath);
                _bpTemplateCache[item.BlueprintPath] = cache;
            }

            // Strip "_GEN_VARIABLE" suffix to match SCS InternalVariableName.
            foreach (var comp in item.Created)
            {
                var lookupName = comp.Export.Name;
                if (lookupName.EndsWith("_GEN_VARIABLE", StringComparison.Ordinal))
                    lookupName = lookupName[..^"_GEN_VARIABLE".Length];

                var templateData = GetBlueprintTemplateData(item.BlueprintPath, lookupName);
                if (templateData == null) continue;

                if (comp.Mesh == null && templateData.MeshIndex != null && !templateData.MeshIndex.IsNull)
                {
                    comp.Mesh      = ResolveMeshPath(templateData.MeshIndex);
                    comp.MeshIndex = templateData.MeshIndex;
                }

                // Only apply template transforms when the property is not explicitly stored in the umap.
                var compProps = comp.Export.Properties;
                if (!compProps.Any(p => p.Name.Text == "RelativeLocation"))
                    comp.RelLoc   = templateData.RelLoc;
                if (!compProps.Any(p => p.Name.Text == "RelativeScale3D"))
                    comp.RelScale = templateData.RelScale;
                if (!compProps.Any(p => p.Name.Text == "RelativeRotation"))
                    comp.RelRot   = templateData.RelRot;

                if (comp.SocketName == null && templateData.SocketName != null)
                    comp.SocketName = templateData.SocketName;
            }

            var exp = item.Export;
            foreach (var prop in exp.Properties)
            {
                if (prop.Name.Text == "BlueprintCreatedComponents") continue;
                if (prop.Tag == null) continue;
                FPackageIndex? ci = null;
                try { ci = prop.Tag.GetValue(typeof(FPackageIndex)) as FPackageIndex; }
                catch { /* not an object ref */ }
                if (ci == null || ci.IsNull) continue;

                var comp = ResolveToItem(ci);
                if (comp == null || comp.CreatedBy != item) continue;

                var templateData = GetBlueprintTemplateData(item.BlueprintPath, prop.Name.Text);
                if (templateData == null) continue;

                if (comp.Mesh == null && templateData.MeshIndex != null && !templateData.MeshIndex.IsNull)
                {
                    comp.Mesh      = ResolveMeshPath(templateData.MeshIndex);
                    comp.MeshIndex = templateData.MeshIndex;
                }

                var compProps = comp.Export.Properties;
                if (!compProps.Any(p => p.Name.Text == "RelativeLocation"))
                    comp.RelLoc   = templateData.RelLoc;
                if (!compProps.Any(p => p.Name.Text == "RelativeScale3D"))
                    comp.RelScale = templateData.RelScale;
                if (!compProps.Any(p => p.Name.Text == "RelativeRotation"))
                    comp.RelRot   = templateData.RelRot;

                if (comp.SocketName == null && templateData.SocketName != null)
                    comp.SocketName = templateData.SocketName;
            }
        }

        private void Globalize(MapItem item)
        {
            if (item.GlobLoc != null) return;
            if (item.Parent != null) Globalize(item.Parent);

            double[]   pLoc   = item.Parent?.GlobLoc   ?? [0, 0, 0];
            double[]   pScale = item.Parent?.GlobScale  ?? [1, 1, 1];
            double[,]  pRot   = item.Parent?.GlobRot    ?? TransformMath.Identity3x3;

            // If attached to a socket on the parent skeletal mesh, offset the parent
            // transform by the socket's mesh-local transform before applying item's relative.
            if (item.SocketName != null)
            {
                if (item.Parent == null)
                {
                    Log.Warning("[Socket] item '{0}' has SocketName='{1}' but no Parent", item.Export?.Name, item.SocketName);
                }
                else
                {
                    var sock = GetSocketMeshLocal(item.Parent.MeshIndex, item.SocketName);
                    if (sock == null)
                    {
                        Log.Warning("[Socket] item '{0}': socket '{1}' not found on parent '{2}' mesh '{3}'",
                            item.Export?.Name, item.SocketName, item.Parent.Export?.Name,
                            ResolveMeshPath(item.Parent.MeshIndex) ?? "<null>");
                    }
                    else
                    {
                        var (sLoc, sRot, sScl) = sock.Value;
                        var sScaled = new double[] { sLoc[0]*pScale[0], sLoc[1]*pScale[1], sLoc[2]*pScale[2] };
                        var rv = TransformMath.MatVecMul(pRot, sScaled);
                        pLoc   = [pLoc[0]+rv[0], pLoc[1]+rv[1], pLoc[2]+rv[2]];
                        pRot   = TransformMath.MatMul(pRot, sRot);
                        pScale = [pScale[0]*sScl[0], pScale[1]*sScl[1], pScale[2]*sScl[2]];
                    }
                }
            }

            var localRot  = TransformMath.RotationMatrix(item.RelRot[0], item.RelRot[1], item.RelRot[2]);
            item.GlobRot  = TransformMath.MatMul(pRot, localRot);
            item.GlobScale = [pScale[0]*item.RelScale[0], pScale[1]*item.RelScale[1], pScale[2]*item.RelScale[2]];

            var sr  = new double[] { item.RelLoc[0]*pScale[0], item.RelLoc[1]*pScale[1], item.RelLoc[2]*pScale[2] };
            var rt  = TransformMath.MatVecMul(pRot, sr);
            item.GlobLoc = [pLoc[0]+rt[0], pLoc[1]+rt[1], pLoc[2]+rt[2]];

            foreach (var ri in item.RelInstances)
            {
                double ix=ri[0], iy=ri[1], iz=ri[2], isx=ri[3], isy=ri[4], isz=ri[5];
                double qx=ri[6], qy=ri[7], qz=ri[8], qw=ri[9];

                var iRot = TransformMath.QuaternionToMatrix(qx,qy,qz,qw);
                var wRot = TransformMath.MatMul(item.GlobRot, iRot);

                var sr2  = new double[] { ix*item.GlobScale[0], iy*item.GlobScale[1], iz*item.GlobScale[2] };
                var rv   = TransformMath.MatVecMul(item.GlobRot, sr2);
                var wPos = new double[] { item.GlobLoc[0]+rv[0], item.GlobLoc[1]+rv[1], item.GlobLoc[2]+rv[2] };
                var wScl = new double[]
                {
                    Math.Round(isx*item.GlobScale[0], 2),
                    Math.Round(isy*item.GlobScale[1], 2),
                    Math.Round(isz*item.GlobScale[2], 2),
                };
                item.GlobInstances.Add(TransformMath.MakeEntry(wPos, wScl, wRot));
            }
        }

        private void SerializeItem(
            MapItem item,
            Dictionary<string, List<double[]>> symbols,
            Dictionary<string, List<double[]>> groups,
            Dictionary<string, List<JObject>>  blueprints,
            Dictionary<string, List<double[]>> splines)
        {
            if (item.CreatedBy != null) return;

            // Spline mesh components deform a mesh along a cubic hermite spline; emit
            // their spline parameters (in world space) so the mesh can be reconstructed
            // downstream. Handled before the generic SkipTypes filter.
            if (item.ExportType == "SplineMeshComponent")
            {
                SerializeSpline(item, splines);
                return;
            }

            if (Constants.SkipTypes.Contains(item.ExportType)) return;
            if (item.GlobLoc == null) return;

            if (item.BlueprintPath != null &&
                Constants.Patches.TryGetValue(item.BlueprintPath, out var replacePath))
            {
                var meshDict = BuildPatchMeshDict(replacePath, item);
                if (meshDict != null)
                {
                    string bpn = Constants.NormaliseBPName(replacePath.Split('/').Last() + "_C");
                    if (!blueprints.ContainsKey(bpn)) blueprints[bpn] = [];
                    blueprints[bpn].Add(meshDict);
                }
                return;
            }

            if (item.Created.Count > 0)
            {
                var selfEntry = TransformMath.MakeEntry(item.GlobLoc!, item.GlobScale!, item.GlobRot!);
                var meshDict  = new JObject { ["_self"] = ToJArray(selfEntry) };
                bool hasMesh  = false;

                foreach (var child in item.Created)
                {
                    if (child.Mesh == null || child.GlobLoc == null) continue;
                    var mn = Unpath(child.Mesh);
                    if (mn == null || Constants.IsHardbanned(mn)) continue;
                    mn = Constants.NormaliseMeshName(mn);
                    ExportMesh(child.MeshIndex);

                    var entry = TransformMath.MakeEntry(child.GlobLoc!, child.GlobScale!, child.GlobRot!);
                    if (!meshDict.ContainsKey(mn)) meshDict[mn] = new JArray();
                    ((JArray)meshDict[mn]!).Add(ToJArray(entry));
                    hasMesh = true;
                }

                if (!hasMesh) return;

                string bpn = Constants.NormaliseBPName(item.ExportType);
                if (!blueprints.ContainsKey(bpn)) blueprints[bpn] = [];
                blueprints[bpn].Add(meshDict);
                return;
            }

            if (item.Mesh == null) return;
            var meshName = Unpath(item.Mesh);
            if (meshName == null || Constants.IsHardbanned(meshName)) return;

            ExportMesh(item.MeshIndex);

            if (item.GlobInstances.Count > 0)
            {
                if (!groups.ContainsKey(meshName)) groups[meshName] = [];
                groups[meshName].AddRange(item.GlobInstances);
            }
            else
            {
                var entry = TransformMath.MakeEntry(item.GlobLoc!, item.GlobScale!, item.GlobRot!);
                if (!symbols.ContainsKey(meshName)) symbols[meshName] = [];
                symbols[meshName].Add(entry);
            }
        }

        private void SerializeSpline(MapItem item, Dictionary<string, List<double[]>> splines)
        {
            if (item.GlobLoc == null || item.GlobRot == null || item.GlobScale == null) return;
            if (item.Mesh == null || item.MeshIndex == null) return;

            var meshName = Unpath(item.Mesh);
            if (meshName == null || Constants.IsHardbanned(meshName)) return;
            meshName = Constants.NormaliseMeshName(meshName);

            FSplineMeshParams? sp = null;
            try { sp = item.Export.GetOrDefault<FSplineMeshParams>("SplineParams"); }
            catch { return; }
            if (sp == null) return;

            var axis = item.Export.GetOrDefault<ESplineMeshAxis>("ForwardAxis", ESplineMeshAxis.X);

            var loc   = item.GlobLoc;
            var rot   = item.GlobRot;
            var scale = item.GlobScale;

            double[] txPoint(double x, double y, double z)
            {
                var sr = new double[] { x*scale[0], y*scale[1], z*scale[2] };
                var rt = TransformMath.MatVecMul(rot, sr);
                return [loc[0]+rt[0], loc[1]+rt[1], loc[2]+rt[2]];
            }
            double[] txDir(double x, double y, double z)
            {
                var sr = new double[] { x*scale[0], y*scale[1], z*scale[2] };
                return TransformMath.MatVecMul(rot, sr);
            }

            var sPos = txPoint(sp.StartPos.X, sp.StartPos.Y, sp.StartPos.Z);
            var ePos = txPoint(sp.EndPos.X,   sp.EndPos.Y,   sp.EndPos.Z);
            var sTan = txDir  (sp.StartTangent.X, sp.StartTangent.Y, sp.StartTangent.Z);
            var eTan = txDir  (sp.EndTangent.X,   sp.EndTangent.Y,   sp.EndTangent.Z);

            // Flat layout (23 values) - sufficient to rebuild the deformed mesh:
            //  [0..2]   world start position
            //  [3..5]   world start tangent
            //  [6..8]   world end position
            //  [9..11]  world end tangent
            //  [12]     start roll (radians)
            //  [13..14] start offset (x, y, component-space 2D)
            //  [15..16] start scale  (x, y)
            //  [17]     end roll (radians)
            //  [18..19] end offset (x, y)
            //  [20..21] end scale  (x, y)
            //  [22]     forward axis: 0=X, 1=Y, 2=Z
            var entry = new double[]
            {
                sPos[0], sPos[1], sPos[2],
                sTan[0], sTan[1], sTan[2],
                ePos[0], ePos[1], ePos[2],
                eTan[0], eTan[1], eTan[2],
                sp.StartRoll, sp.StartOffset.X, sp.StartOffset.Y, sp.StartScale.X, sp.StartScale.Y,
                sp.EndRoll,   sp.EndOffset.X,   sp.EndOffset.Y,   sp.EndScale.X,   sp.EndScale.Y,
                (int)axis,
            };

            ExportMesh(item.MeshIndex);

            if (!splines.ContainsKey(meshName)) splines[meshName] = [];
            splines[meshName].Add(entry);
        }

        private JObject? BuildPatchMeshDict(string replacePath, MapItem actor)
        {
            if (!_provider.TryLoadPackage(replacePath, out var bpPkg)) return null;

            UBlueprintGeneratedClass? bgc = null;
            foreach (var e in bpPkg.GetExports())
                if (e is UBlueprintGeneratedClass c) { bgc = c; break; }
            if (bgc == null) return null;

            var scsIdx = bgc.SimpleConstructionScript;
            UObject? scs = null;
            if (scsIdx != null && !scsIdx.IsNull)
                try { scs = scsIdx.Load(); } catch { }

            double[] rPos = actor.GlobLoc!;
            double[,] rRot = actor.GlobRot!;
            double[] rScl = actor.GlobScale!;

            var meshEntries = new Dictionary<string, (List<double[]> entries, FPackageIndex? idx)>();

            if (scs != null)
            {
                var rootNodes = scs.GetOrDefault<FPackageIndex[]>("RootNodes");
                if (rootNodes != null)
                    foreach (var n in rootNodes)
                        ProcessSCSNode(n, rPos, rRot, rScl, meshEntries);
            }

            // Also scan CDO for native C++ components not in any SCS node.
            var cdoIdx = bgc.ClassDefaultObject;
            if (cdoIdx != null && !cdoIdx.IsNull)
            {
                UObject? cdo = null;
                try { cdo = cdoIdx.Load(); } catch { }
                if (cdo != null)
                {
                    foreach (var cdoProp in cdo.Properties)
                    {
                        if (cdoProp.Tag == null) continue;
                        FPackageIndex? compIdx = null;
                        try { compIdx = cdoProp.Tag.GetValue(typeof(FPackageIndex)) as FPackageIndex; }
                        catch { }
                        if (compIdx == null || compIdx.IsNull) continue;

                        UObject? comp = null;
                        try { comp = compIdx.Load(); } catch { }
                        if (comp == null) continue;
                        if (Constants.SkipTypes.Contains(comp.ExportType)) continue;

                        var mi = comp.GetOrDefault<FPackageIndex>("StaticMesh");
                        if (mi == null || mi.IsNull)
                            mi = comp.GetOrDefault<FPackageIndex>("SkeletalMesh");
                        if (mi == null || mi.IsNull) continue;

                        var mn = Unpath(ResolveMeshPath(mi));
                        if (mn == null || Constants.IsHardbanned(mn)) continue;
                        mn = Constants.NormaliseMeshName(mn);

                        var rl  = comp.GetOrDefault<FVector>("RelativeLocation",  FVector.ZeroVector);
                        var rs  = comp.GetOrDefault<FVector>("RelativeScale3D",   FVector.OneVector);
                        var rr2 = comp.GetOrDefault<FRotator>("RelativeRotation", FRotator.ZeroRotator);

                        var lRot = TransformMath.RotationMatrix(rr2.Pitch, rr2.Yaw, rr2.Roll);
                        var wRot = TransformMath.MatMul(rRot, lRot);
                        var wScl = new double[] { rScl[0]*rs.X, rScl[1]*rs.Y, rScl[2]*rs.Z };
                        var sr   = new double[] { rl.X*rScl[0], rl.Y*rScl[1], rl.Z*rScl[2] };
                        var rv   = TransformMath.MatVecMul(rRot, sr);
                        var wPos = new double[] { rPos[0]+rv[0], rPos[1]+rv[1], rPos[2]+rv[2] };

                        ExportMesh(mi);
                        if (!meshEntries.ContainsKey(mn))
                            meshEntries[mn] = ([], mi);
                        meshEntries[mn].Item1.Add(TransformMath.MakeEntry(wPos, wScl, wRot));
                    }
                }
            }

            if (meshEntries.Count == 0) return null;

            var selfEntry = TransformMath.MakeEntry(rPos, rScl, rRot);
            var meshDict  = new JObject { ["_self"] = ToJArray(selfEntry) };

            foreach (var (mn, (entries, midx)) in meshEntries)
            {
                ExportMesh(midx);
                var arr = new JArray();
                foreach (var e in entries) arr.Add(ToJArray(e));
                meshDict[mn] = arr;
            }

            return meshDict;
        }

        private void ProcessSCSNode(
            FPackageIndex nodeIdx,
            double[] parentPos, double[,] parentRot, double[] parentScale,
            Dictionary<string, (List<double[]>, FPackageIndex?)> result)
        {
            if (nodeIdx == null || nodeIdx.IsNull) return;
            UObject? node = null;
            try { node = nodeIdx.Load(); } catch { return; }
            if (node == null) return;

            double[] wPos = parentPos; double[,] wRot = parentRot; double[] wScl = parentScale;

            var tplIdx = node.GetOrDefault<FPackageIndex>("ComponentTemplate");
            if (tplIdx != null && !tplIdx.IsNull)
            {
                UObject? tpl = null;
                try { tpl = tplIdx.Load(); } catch { }

                if (tpl != null && !Constants.SkipTypes.Contains(tpl.ExportType))
                {
                    var rl  = tpl.GetOrDefault<FVector>("RelativeLocation",  FVector.ZeroVector);
                    var rs  = tpl.GetOrDefault<FVector>("RelativeScale3D",   FVector.OneVector);
                    var rr  = tpl.GetOrDefault<FRotator>("RelativeRotation", FRotator.ZeroRotator);

                    var lRot = TransformMath.RotationMatrix(rr.Pitch, rr.Yaw, rr.Roll);
                    wRot  = TransformMath.MatMul(parentRot, lRot);
                    wScl  = [parentScale[0]*rs.X, parentScale[1]*rs.Y, parentScale[2]*rs.Z];
                    var sr = new double[] { rl.X*parentScale[0], rl.Y*parentScale[1], rl.Z*parentScale[2] };
                    var rv = TransformMath.MatVecMul(parentRot, sr);
                    wPos  = [parentPos[0]+rv[0], parentPos[1]+rv[1], parentPos[2]+rv[2]];

                    var mi = tpl.GetOrDefault<FPackageIndex>("StaticMesh");
                    if (mi == null || mi.IsNull)
                        mi = tpl.GetOrDefault<FPackageIndex>("SkeletalMesh");
                    if (mi != null && !mi.IsNull)
                    {
                        var mn = Unpath(ResolveMeshPath(mi));
                        if (mn != null && !Constants.IsHardbanned(mn))
                        {
                            mn = Constants.NormaliseMeshName(mn);
                            ExportMesh(mi);
                            if (!result.ContainsKey(mn))
                                result[mn] = ([], mi);
                            result[mn].Item1.Add(TransformMath.MakeEntry(wPos, wScl, wRot));
                        }
                    }
                }
            }

            var childNodes = node.GetOrDefault<FPackageIndex[]>("ChildNodes");
            if (childNodes != null)
                foreach (var c in childNodes)
                    ProcessSCSNode(c, wPos, wRot, wScl, result);
        }

        private TemplateData? GetBlueprintTemplateData(string bpPath, string compName)
        {
            if (!_bpTemplateCache.TryGetValue(bpPath, out var cache))
            {
                cache = BuildTemplateCache(bpPath);
                _bpTemplateCache[bpPath] = cache;
            }
            return cache.GetValueOrDefault(compName);
        }

        private Dictionary<string, TemplateData> BuildTemplateCache(string bpPath)
        {
            var result = new Dictionary<string, TemplateData>(StringComparer.OrdinalIgnoreCase);
            if (!_provider.TryLoadPackage(bpPath, out var pkg)) return result;

            UBlueprintGeneratedClass? bpc = null;
            foreach (var e in pkg.GetExports())
                if (e is UBlueprintGeneratedClass c) { bpc = c; break; }

            if (bpc != null) ExtractTemplatesFromClass(bpc, result);
            return result;
        }

        private void ExtractTemplatesFromClass(UBlueprintGeneratedClass bpc,
                                               Dictionary<string, TemplateData> result)
        {
            if (bpc.Owner == null) return;

            // Recurse into superclass first so that child-class data takes precedence.
            try
            {
                if (bpc.SuperStruct?.Load() is UBlueprintGeneratedClass super)
                    ExtractTemplatesFromClass(super, result);
            }
            catch { }

            // Pass 1: SCS_Node (blueprint-defined components via SimpleConstructionScript)
            // Pass 2: InheritableComponentHandler (child-blueprint overrides)
            var ownScsKeys = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            foreach (var exp in bpc.Owner.GetExports())
            {
                if (exp.ExportType == "SCS_Node")
                {
                    var ivn = exp.GetOrDefault<FName>("InternalVariableName");
                    if (ivn.IsNone) continue;
                    var ti = exp.GetOrDefault<FPackageIndex>("ComponentTemplate");
                    if (ti == null || ti.IsNull) continue;
                    UObject? t = null;
                    try { t = ti.Load(); } catch { }
                    if (t == null) continue;

                    var meshIdx = t.GetOrDefault<FPackageIndex>("StaticMesh");
                    if (meshIdx == null || meshIdx.IsNull)
                        meshIdx = t.GetOrDefault<FPackageIndex>("SkeletalMesh");

                    var tdata = new TemplateData { MeshIndex = meshIdx };

                    var rl = t.GetOrDefault<FVector>("RelativeLocation",  FVector.ZeroVector);
                    var rs = t.GetOrDefault<FVector>("RelativeScale3D",   FVector.OneVector);
                    var rr = t.GetOrDefault<FRotator>("RelativeRotation", FRotator.ZeroRotator);
                    tdata.RelLoc   = [rl.X, rl.Y, rl.Z];
                    tdata.RelScale = [rs.X, rs.Y, rs.Z];
                    tdata.RelRot   = [rr.Pitch, rr.Yaw, rr.Roll];

                    // SCS-level socket attachment (e.g. "S_cabin" on a parent skeletal mesh).
                    var atn = exp.GetOrDefault<FName>("AttachToName");
                    if (!atn.IsNone && !string.IsNullOrEmpty(atn.Text) && atn.Text != "None")
                        tdata.SocketName = atn.Text;

                    result[ivn.Text] = tdata;
                    ownScsKeys.Add(ivn.Text);
                }

                if (exp.ExportType == "InheritableComponentHandler")
                {
                    var records = exp.GetOrDefault<FStructFallback[]>("Records");
                    if (records == null) continue;
                    foreach (var rec in records)
                    {
                        var ck  = rec.GetOrDefault<FStructFallback>("ComponentKey");
                        if (ck == null) continue;
                        var svn = ck.GetOrDefault<FName>("SCSVariableName");
                        if (svn.IsNone) continue;
                        var ti  = rec.GetOrDefault<FPackageIndex>("ComponentTemplate");
                        if (ti == null || ti.IsNull) continue;
                        UObject? t = null;
                        try { t = ti.Load(); } catch { }
                        if (t == null) continue;

                        if (!result.TryGetValue(svn.Text, out var tdata))
                        {
                            tdata = new TemplateData();
                            result[svn.Text] = tdata;
                        }

                        var mesh = t.GetOrDefault<FPackageIndex>("StaticMesh");
                        if (mesh == null || mesh.IsNull)
                            mesh = t.GetOrDefault<FPackageIndex>("SkeletalMesh");
                        if (mesh != null && !mesh.IsNull)
                            tdata.MeshIndex = mesh;

                        // Only overwrite inherited transforms when the ICH template explicitly declares the property.
                        if (t.Properties.Any(p => p.Name.Text == "RelativeLocation"))
                        {
                            var rl = t.GetOrDefault<FVector>("RelativeLocation", FVector.ZeroVector);
                            tdata.RelLoc = [rl.X, rl.Y, rl.Z];
                        }
                        if (t.Properties.Any(p => p.Name.Text == "RelativeScale3D"))
                        {
                            var rs = t.GetOrDefault<FVector>("RelativeScale3D", FVector.OneVector);
                            tdata.RelScale = [rs.X, rs.Y, rs.Z];
                        }
                        if (t.Properties.Any(p => p.Name.Text == "RelativeRotation"))
                        {
                            var rr = t.GetOrDefault<FRotator>("RelativeRotation", FRotator.ZeroRotator);
                            tdata.RelRot = [rr.Pitch, rr.Yaw, rr.Roll];
                        }
                    }
                }
            }

            // Pass 3: CDO – native C++ components not in the SCS. SCS-defined entries take precedence.
            try
            {
                if (bpc.ClassDefaultObject != null && !bpc.ClassDefaultObject.IsNull)
                {
                    var cdo = bpc.ClassDefaultObject.Load();
                    if (cdo != null)
                    {
                        foreach (var prop in cdo.Properties)
                        {
                            // Skip if the current class's own SCS already defines this component
                            // (keyed either by property name or by component-object name below).
                            if (ownScsKeys.Contains(prop.Name.Text)) continue;
                            if (prop.Tag == null) continue;

                            FPackageIndex? compIdx = null;
                            try { compIdx = prop.Tag.GetValue(typeof(FPackageIndex)) as FPackageIndex; }
                            catch { }
                            if (compIdx == null || compIdx.IsNull) continue;

                            UObject? comp = null;
                            try { comp = compIdx.Load(); } catch { }
                            if (comp == null) continue;
                            if (Constants.SkipTypes.Contains(comp.ExportType)) continue;
                            if (ownScsKeys.Contains(comp.Name)) continue;

                            // Merge with any existing template (populated by parent class's CDO/SCS).
                            // Child CDOs often re-list inherited components without re-declaring the
                            // mesh or transform; those inherited values must be preserved.
                            if (!result.TryGetValue(comp.Name, out var tdata))
                            {
                                tdata = new TemplateData();
                                result[comp.Name] = tdata;
                            }

                            var compProps = comp.Properties;

                            FPackageIndex? cdoMeshIdx = null;
                            if (compProps.Any(p => p.Name.Text == "StaticMesh"))
                                cdoMeshIdx = comp.GetOrDefault<FPackageIndex>("StaticMesh");
                            if ((cdoMeshIdx == null || cdoMeshIdx.IsNull)
                                && compProps.Any(p => p.Name.Text == "SkeletalMesh"))
                                cdoMeshIdx = comp.GetOrDefault<FPackageIndex>("SkeletalMesh");
                            if (cdoMeshIdx != null && !cdoMeshIdx.IsNull)
                                tdata.MeshIndex = cdoMeshIdx;

                            if (compProps.Any(p => p.Name.Text == "RelativeLocation"))
                            {
                                var rl = comp.GetOrDefault<FVector>("RelativeLocation", FVector.ZeroVector);
                                tdata.RelLoc = [rl.X, rl.Y, rl.Z];
                            }
                            if (compProps.Any(p => p.Name.Text == "RelativeScale3D"))
                            {
                                var rs = comp.GetOrDefault<FVector>("RelativeScale3D", FVector.OneVector);
                                tdata.RelScale = [rs.X, rs.Y, rs.Z];
                            }
                            if (compProps.Any(p => p.Name.Text == "RelativeRotation"))
                            {
                                var rr = comp.GetOrDefault<FRotator>("RelativeRotation", FRotator.ZeroRotator);
                                tdata.RelRot = [rr.Pitch, rr.Yaw, rr.Roll];
                            }

                            // Alias by the CDO property name (e.g. "Mesh") for the prop-iteration
                            // path in PopulateBlueprintDefaults.
                            if (!string.Equals(comp.Name, prop.Name.Text, StringComparison.Ordinal)
                                && !result.ContainsKey(prop.Name.Text))
                                result[prop.Name.Text] = tdata;
                        }
                    }
                }
            }
            catch { }
        }

        // Resolve a named socket on a skeletal mesh to its transform in mesh-local space.
        // Combines the bone's mesh-local ref-pose transform with the socket's bone-relative transform.
        private (double[] loc, double[,] rot, double[] scl)? GetSocketMeshLocal(FPackageIndex? meshIdx, string socketName)
        {
            if (meshIdx == null || meshIdx.IsNull) return null;
            var path = ResolveMeshPath(meshIdx);
            if (path == null) return null;

            if (!_socketCache.TryGetValue(path, out var cache))
            {
                cache = BuildSocketCache(meshIdx);
                _socketCache[path] = cache;
            }
            if (cache == null) return null;
            return cache.TryGetValue(socketName, out var xf) ? xf : null;
        }

        private static Dictionary<string, (double[] loc, double[,] rot, double[] scl)>? BuildSocketCache(FPackageIndex meshIdx)
        {
            USkeletalMesh? mesh;
            try { mesh = meshIdx.Load<UObject>() as USkeletalMesh; }
            catch { return null; }
            if (mesh == null) return null;

            var refSkel = mesh.ReferenceSkeleton;
            if (refSkel == null) return null;

            // Build mesh-local bone transforms by walking parent chain.
            int boneCount = refSkel.FinalRefBoneInfo.Length;
            var boneLoc   = new double[boneCount][];
            var boneRot   = new double[boneCount][,];
            var boneScl   = new double[boneCount][];
            var nameToIdx = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

            for (int i = 0; i < boneCount; i++)
            {
                var info = refSkel.FinalRefBoneInfo[i];
                var pose = refSkel.FinalRefBonePose[i];

                double[,] localRot = TransformMath.QuaternionToMatrix(pose.Rotation.X, pose.Rotation.Y, pose.Rotation.Z, pose.Rotation.W);
                double[]  localLoc = [pose.Translation.X, pose.Translation.Y, pose.Translation.Z];
                double[]  localScl = [pose.Scale3D.X, pose.Scale3D.Y, pose.Scale3D.Z];

                int parent = info.ParentIndex;
                if (parent < 0)
                {
                    boneLoc[i] = localLoc;
                    boneRot[i] = localRot;
                    boneScl[i] = localScl;
                }
                else
                {
                    var pL = boneLoc[parent]; var pR = boneRot[parent]; var pS = boneScl[parent];
                    var sr = new double[] { localLoc[0]*pS[0], localLoc[1]*pS[1], localLoc[2]*pS[2] };
                    var rv = TransformMath.MatVecMul(pR, sr);
                    boneLoc[i] = [pL[0]+rv[0], pL[1]+rv[1], pL[2]+rv[2]];
                    boneRot[i] = TransformMath.MatMul(pR, localRot);
                    boneScl[i] = [pS[0]*localScl[0], pS[1]*localScl[1], pS[2]*localScl[2]];
                }

                nameToIdx[info.Name.Text] = i;
            }

            var result = new Dictionary<string, (double[] loc, double[,] rot, double[] scl)>(StringComparer.OrdinalIgnoreCase);

            void ProcessSockets(FPackageIndex[]? sockets)
            {
                if (sockets == null) return;
                foreach (var sIdx in sockets)
                {
                    if (sIdx == null || sIdx.IsNull) continue;
                    USkeletalMeshSocket? sock;
                    try { sock = sIdx.Load<UObject>() as USkeletalMeshSocket; }
                    catch { continue; }
                    if (sock == null) continue;

                    string socketName = sock.SocketName.Text;
                    if (string.IsNullOrEmpty(socketName)) continue;

                    // Default: socket is in mesh-local space (bone ignored if not found / root).
                    double[]  bL = [0, 0, 0];
                    double[,] bR = TransformMath.Identity3x3;
                    double[]  bS = [1, 1, 1];
                    if (!sock.BoneName.IsNone && nameToIdx.TryGetValue(sock.BoneName.Text, out var bIdx))
                    {
                        bL = boneLoc[bIdx]; bR = boneRot[bIdx]; bS = boneScl[bIdx];
                    }

                    var sLocRel = new double[] { sock.RelativeLocation.X, sock.RelativeLocation.Y, sock.RelativeLocation.Z };
                    var sRotRel = TransformMath.RotationMatrix(sock.RelativeRotation.Pitch, sock.RelativeRotation.Yaw, sock.RelativeRotation.Roll);
                    var sSclRel = new double[] { sock.RelativeScale.X, sock.RelativeScale.Y, sock.RelativeScale.Z };

                    var sr = new double[] { sLocRel[0]*bS[0], sLocRel[1]*bS[1], sLocRel[2]*bS[2] };
                    var rv = TransformMath.MatVecMul(bR, sr);
                    var outLoc = new double[] { bL[0]+rv[0], bL[1]+rv[1], bL[2]+rv[2] };
                    var outRot = TransformMath.MatMul(bR, sRotRel);
                    var outScl = new double[] { bS[0]*sSclRel[0], bS[1]*sSclRel[1], bS[2]*sSclRel[2] };

                    result[socketName] = (outLoc, outRot, outScl);
                }
            }

            ProcessSockets(mesh.Sockets);

            // Also merge sockets defined on the USkeleton asset.
            try
            {
                if (mesh.Skeleton != null && !mesh.Skeleton.IsNull &&
                    mesh.Skeleton.Load<UObject>() is USkeleton skel)
                {
                    ProcessSockets(skel.Sockets);
                }
            }
            catch { }

            // AttachToName on a SkeletalMeshComponent may also refer directly to a bone
            // (not just a named socket). Expose each bone's mesh-local transform as a
            // fallback entry; existing socket entries take precedence.
            for (int i = 0; i < boneCount; i++)
            {
                var bname = refSkel.FinalRefBoneInfo[i].Name.Text;
                if (string.IsNullOrEmpty(bname)) continue;
                if (result.ContainsKey(bname)) continue;
                result[bname] = (boneLoc[i], boneRot[i], boneScl[i]);
            }

            return result.Count > 0 ? result : null;
        }

        private void ExportMesh(FPackageIndex? meshIdx)
        {
            if (meshIdx == null || meshIdx.IsNull) return;
            var fullPath = ResolveMeshPath(meshIdx);
            if (fullPath == null) return;
            var name = Unpath(fullPath);
            if (name == null) return;
            name = Constants.NormaliseMeshName(name);

            var outDir = Path.Combine(_exportFolder, "_meshes");

            // Skeletal meshes use .psk (≤65536 verts) or .pskx (>65536 verts).
            if (File.Exists(Path.Combine(outDir, name + ".pskx"))) return;
            if (File.Exists(Path.Combine(outDir, name + ".psk")))  return;

            try
            {
                var obj = meshIdx.Load<UObject>();
                if (obj == null) return;

                var exporterOptions = new ExporterOptions
                {
                    MeshFormat   = EMeshFormat.ActorX,
                    LodFormat    = ELodFormat.FirstLod,
                    SocketFormat = ESocketFormat.None,
                };

                if (obj is UStaticMesh sm)
                {
                    var exp = new MeshExporter(sm, exporterOptions);
                    if (exp.MeshLods.Count == 0) return;
                    Directory.CreateDirectory(outDir);
                    File.WriteAllBytes(Path.Combine(outDir, name + ".pskx"), exp.MeshLods[0].FileData);
                    Log.Debug("Exported static mesh → {0}", name);
                }
                else if (obj is USkeletalMesh skm)
                {
                    Log.Debug("Attempting skeletal mesh export → {0}", name);
                    var exp = new MeshExporter(skm, exporterOptions);
                    if (exp.MeshLods.Count == 0)
                    {
                        Log.Warning("No LODs found for skeletal mesh '{0}'", name);
                        return;
                    }
                    var lodExt = Path.GetExtension(exp.MeshLods[0].FileName);
                    if (string.IsNullOrEmpty(lodExt)) lodExt = ".psk";
                    Directory.CreateDirectory(outDir);
                    File.WriteAllBytes(Path.Combine(outDir, name + lodExt), exp.MeshLods[0].FileData);
                    Log.Debug("Exported skeletal mesh → {0} ({1})", name, lodExt.TrimStart('.'));
                }
            }
            catch (Exception ex)
            {
                Log.Warning("Cannot export '{0}': {1}", name, ex.Message);
            }
        }

        private MapItem? ResolveToItem(FPackageIndex? fpi)
        {
            if (fpi == null || fpi.IsNull || !fpi.IsExport) return null;
            try
            {
                var obj = fpi.ResolvedObject?.Object?.Value;
                return obj != null && _objToItem.TryGetValue(obj, out var item) ? item : null;
            }
            catch { return null; }
        }

        private static string? ResolveMeshPath(FPackageIndex? fpi)
        {
            if (fpi == null || fpi.IsNull) return null;
            var ro = fpi.ResolvedObject;
            if (ro == null) return null;
            var pkg  = ro.Package?.Name;
            if (pkg == null) return null;
            var obj  = ro.Name.Text;
            var last = pkg.Split('/').LastOrDefault() ?? string.Empty;
            return string.Equals(last, obj, StringComparison.OrdinalIgnoreCase) ? pkg : pkg + "." + obj;
        }

        // "War/Content/Meshes/Foo/Bar.Bar" → "Meshes__Foo__Bar"
        internal static string? Unpath(string? s)
        {
            if (s == null) return null;
            const string prefix = "War/Content/";
            if (s.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                s = s[prefix.Length..];
            s = s.Replace("/", "__").Replace("\\", "__");
            int dot = s.IndexOf('.');
            if (dot >= 0) s = s[..dot];
            return s;
        }

        // Traverse an imported class's outer chain to rebuild its package path.
        private static string? GetBlueprintPath(ResolvedObject? classResolved)
        {
            if (classResolved == null) return null;
            try
            {
                var outer = classResolved.Outer;
                if (outer == null) return null;

                var parts = new List<string>();
                var cur   = outer;
                while (cur != null)
                {
                    var name = cur.Name.Text;
                    if (string.IsNullOrEmpty(name) || name == "None") break;
                    parts.Insert(0, name);
                    cur = cur.Outer;
                }

                return parts.Count > 0 ? string.Join("/", parts) : null;
            }
            catch { return null; }
        }

        private static JArray ToJArray(double[] entry)
        {
            var a = new JArray();
            foreach (var v in entry) a.Add(new JValue(v));
            return a;
        }

        private static JObject ToJObject(Dictionary<string, List<double[]>> dict)
        {
            var obj = new JObject();
            foreach (var (k, vs) in dict)
            {
                var arr = new JArray();
                foreach (var e in vs) arr.Add(ToJArray(e));
                obj[k] = arr;
            }
            return obj;
        }

        private static JObject BlueprintsToJObject(Dictionary<string, List<JObject>> dict)
        {
            var obj = new JObject();
            foreach (var (k, vs) in dict)
            {
                var arr = new JArray();
                var seen = new HashSet<string>();
                foreach (var e in vs)
                {
                    var key = e.ToString(Newtonsoft.Json.Formatting.None);
                    if (seen.Add(key)) arr.Add(e);
                }
                obj[k] = arr;
            }
            return obj;
        }
    }
}
