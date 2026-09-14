using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using CUE4Parse.UE4.Assets;
using CUE4Parse.UE4.Assets.Exports;
using CUE4Parse.UE4.Assets.Exports.Actor;
using CUE4Parse.UE4.Assets.Exports.Component.Landscape;
using CUE4Parse.UE4.Objects.Core.Math;
using CUE4Parse.UE4.Objects.UObject;
using CUE4Parse_Conversion.Landscape;
using CUE4Parse_Conversion.Meshes;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using SkiaSharp;
using Serilog;

namespace Exporter
{
    // Stitches all landscape heightmaps and weightmap layers from an entire map into single output images.
    // Output: {exportFolder}/_heightmap/{mapName}.png and {exportFolder}/{PhysMatName}/{mapName}.png
    // NormalMap_DX is always skipped.
    internal static class LandscapeStitcher
    {
        // Heightmap: 1 metre = 1 pixel
        private const int    HmOutputSize   = 2200;
        private const double HmPixelsPerCm  = 1.0 / 100.0;
        private const double HmCmPerPixel   = 100.0;
        private const int    HmOutputCenter = HmOutputSize / 2;

        // Weightmap: 1890 m → 1776 px
        private const int    WmOutputSize   = 2048;
        private const double MapCm          = 189_000.0;
        private const double MapPixels      = 1_776.0;
        private const double WmPixelsPerCm  = MapPixels / MapCm;
        private const double WmCmPerPixel   = MapCm     / MapPixels;
        private const int    WmOutputCenter = WmOutputSize / 2;

        private sealed class LandscapeInfo
        {
            public ALandscapeProxy           Proxy      = null!;
            public string                    Name       = string.Empty;
            public ULandscapeComponent[]     Components = [];

            public double LocX, LocY, LocZ;
            public double ScaleX, ScaleY;
            public double ScaleZ;
            // The rotation matrix, in full. Scattering forward only ever multiplies by it,
            // so unlike the inverse mapping it needs no special case for a tilted proxy:
            // the third column (world XY <- local Z) and third row (world Z <- local XY)
            // simply carry their terms like any other.
            public double R00, R01, R02;
            public double R10, R11, R12;
            public double R20, R21, R22;

            // Local-space height of every source heightmap texel, in cm, kept so the
            // weightmap grid - which has its own resolution - can find the surface it sits
            // on. Null when the proxy's heightmap failed to convert.
            public float[]? LocalZCm;
            public bool[]?  LocalZValid;
            public int      LzW, LzH;

            public int SectionOffsetX, SectionOffsetY;
            public int MinX, MinY, MaxX, MaxY;
            public int SrcW, SrcH;

            public Dictionary<string, string> LayerToPhys =
                new(StringComparer.OrdinalIgnoreCase);
        }

        public static void StitchFromPackage(Package pkg, string exportFolder, string mapName)
        {
            var infos = new List<LandscapeInfo>();

            foreach (var lazy in pkg.ExportsLazy)
            {
                UObject? obj = null;
                try { obj = lazy.Value; } catch { continue; }

                if (obj is not ALandscapeProxy proxy) continue;

                var info = BuildInfo(proxy);
                if (info != null) infos.Add(info);
            }

            if (infos.Count == 0)
            {
                Log.Debug("No ALandscapeProxy exports found.");
                return;
            }

            Log.Information("Found {0} landscape actor(s).", infos.Count);

            var allPhysNames = infos
                .SelectMany(i => i.LayerToPhys.Values)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();

            Log.Information("Unique layers: {0}", string.Join(", ", allPhysNames));

            // Catalogue any pre-existing layer PNGs for this region so we can
            // delete the ones that don't get overwritten by the current run.
            // _meshes/, _heightmap/, _json/ are intentionally left alone.
            var preexistingLayerPngs = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            string layersRoot = Path.Combine(exportFolder, "_layers");
            if (Directory.Exists(layersRoot))
            {
                foreach (var sub in Directory.EnumerateDirectories(layersRoot))
                {
                    string p = Path.Combine(sub, mapName + ".png");
                    if (File.Exists(p)) preexistingLayerPngs.Add(p);
                }
            }
            var writtenLayerPngs = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            double zOffsetCm = Constants.GetHeightOffset(mapName);
            if (zOffsetCm != 0.0)
                Log.Information("Applying Z offset of {0} cm to '{1}'.", zOffsetCm, mapName);

            // One depth buffer per output grid, holding the world Z in cm of whichever
            // surface owns each pixel (NoDepth until something claims it). Every surface is
            // projected forward into these and the highest sample wins the pixel outright -
            // its height and all of its materials - which is what the game's own depth test
            // does when it draws the landscapes from above.
            var heightZ = new float[HmOutputSize * HmOutputSize];
            var layerZ  = new float[WmOutputSize * WmOutputSize];
            Array.Fill(heightZ, NoDepth);
            Array.Fill(layerZ,  NoDepth);

            using var heightDst = new Image<L16>(HmOutputSize, HmOutputSize);
            var weightDsts = allPhysNames.ToDictionary(
                n => n,
                _ => new SKBitmap(WmOutputSize, WmOutputSize, SKColorType.Gray8, SKAlphaType.Unpremul),
                StringComparer.OrdinalIgnoreCase);

            try
            {
                foreach (var info in infos)
                {
                    Log.Information(
                        "  Processing '{0}' ({1} components, loc=[{2:F0},{3:F0}], " +
                        "scale=[{4},{5}], yaw={6:F2}°, sectionOffset=[{7},{8}])…",
                        info.Name, info.Components.Length,
                        info.LocX, info.LocY,
                        info.ScaleX, info.ScaleY,
                        Math.Atan2(info.R10, info.R00) * 180.0 / Math.PI,
                        info.SectionOffsetX, info.SectionOffsetY);

                    // A tilted proxy pivots about local quad (0,0); surface it because the
                    // resulting height contribution is easy to mistake for a bad LocZ.
                    double TiltAt(int qx, int qy) =>
                        info.R20 * (qx - info.SectionOffsetX) * info.ScaleX +
                        info.R21 * (qy - info.SectionOffsetY) * info.ScaleY;

                    // Sample all four corners: with opposite-signed pitch and roll the
                    // extremes sit on the off-diagonal corners, not on min/min and max/max.
                    double[] tilts =
                    [
                        TiltAt(info.MinX, info.MinY), TiltAt(info.MaxX, info.MinY),
                        TiltAt(info.MinX, info.MaxY), TiltAt(info.MaxX, info.MaxY),
                    ];
                    double tiltMin = tilts.Min(), tiltMax = tilts.Max();

                    if (Math.Abs(tiltMin) > 1.0 || Math.Abs(tiltMax) > 1.0)
                    {
                        Log.Information(
                            "    '{0}' is tilted (pitch={1:F4}°, roll={2:F4}°); " +
                            "tilt adds {3:F0} to {4:F0} cm of height across its extent.",
                            info.Name,
                            Math.Asin(Math.Clamp(info.R20, -1.0, 1.0)) * 180.0 / Math.PI,
                            Math.Atan2(-info.R21, info.R22) * 180.0 / Math.PI,
                            tiltMin, tiltMax);
                    }

                    bool ok;
                    Dictionary<string, Image>    heightMaps;
                    Dictionary<string, SKBitmap> weightMaps;

                    try
                    {
                        ok = info.Proxy.TryConvert(
                            info.Components,
                            ELandscapeExportFlags.Heightmap | ELandscapeExportFlags.Weightmap,
                            out _,
                            out heightMaps,
                            out weightMaps);
                    }
                    catch (Exception ex)
                    {
                        Log.Warning("    TryConvert failed for '{0}': {1}", info.Name, ex.Message);
                        continue;
                    }

                    if (!ok)
                    {
                        Log.Warning("    TryConvert returned false for '{0}'.", info.Name);
                        continue;
                    }

                    if (heightMaps.TryGetValue("heightmap", out var hImgBase))
                    {
                        info.SrcW = hImgBase.Width;
                        info.SrcH = hImgBase.Height;

                        if (hImgBase is Image<L16> hImg)
                            ReadHeightSource(info, hImg);
                        else
                            Log.Warning("    Unexpected heightmap pixel format for '{0}'.", info.Name);

                        hImgBase.Dispose();
                    }

                    if (info.LocalZCm == null)
                    {
                        Log.Warning("    '{0}' has no usable heightmap; nothing to project.",
                            info.Name);
                        foreach (var (_, bm) in weightMaps) bm.Dispose();
                        continue;
                    }

                    // The height grid drives the heightmap output directly.
                    var heightSurface = BuildSurface(
                        info, info.LzW, info.LzH, info.LocalZCm, info.LocalZValid!, zOffsetCm);
                    ScatterHeights(heightSurface, heightZ);

                    // Gather the proxy's layers before projecting any of them. The depth test
                    // decides per pixel which surface is visible, and that verdict has to
                    // apply to every material at once - projecting layers one at a time would
                    // let different layers of one pixel come from different surfaces.
                    var srcLayers = new Dictionary<string, byte[]>(StringComparer.OrdinalIgnoreCase);
                    int wmW = 0, wmH = 0;

                    foreach (var (rawName, srcBm) in weightMaps)
                    {
                        if (rawName.Equals("NormalMap_DX", StringComparison.OrdinalIgnoreCase))
                        { srcBm.Dispose(); continue; }

                        if (info.SrcW == 0) { info.SrcW = srcBm.Width; info.SrcH = srcBm.Height; }
                        if (wmW == 0) { wmW = srcBm.Width; wmH = srcBm.Height; }

                        if (srcBm.Width != wmW || srcBm.Height != wmH)
                        {
                            Log.Warning("    '{0}' layer '{1}' is {2}x{3}, expected {4}x{5}; skipping.",
                                info.Name, rawName, srcBm.Width, srcBm.Height, wmW, wmH);
                            srcBm.Dispose();
                            continue;
                        }

                        if (!info.LayerToPhys.TryGetValue(rawName, out var physName))
                            physName = rawName;

                        if (weightDsts.ContainsKey(physName))
                        {
                            var arr = CopyGray8(srcBm);
                            // Several raw layers can resolve to one physical material;
                            // combine them within the proxy before the depth test sees them.
                            if (srcLayers.TryGetValue(physName, out var existing))
                                for (int i = 0; i < existing.Length; i++)
                                    existing[i] = Math.Max(existing[i], arr[i]);
                            else
                                srcLayers[physName] = arr;
                        }

                        srcBm.Dispose();
                    }

                    // Unpainted texels are not holes to the game - see FillUnpainted - so
                    // resolve them here, while the proxy's own component grid is still in
                    // hand and every consumer downstream can just read the layers.
                    int gridW = info.MaxX - info.MinX + 1;
                    int gridH = info.MaxY - info.MinY + 1;

                    if (wmW == 0 && wmH == 0) { wmW = gridW; wmH = gridH; }

                    if (wmW == gridW && wmH == gridH)
                        FillUnpainted(info, srcLayers, weightDsts, wmW, wmH);
                    else
                        Log.Warning(
                            "    '{0}' weightmap grid is {1}×{2}, expected {3}×{4}; " +
                            "leaving unpainted texels alone.",
                            info.Name, wmW, wmH, gridW, gridH);

                    if (srcLayers.Count > 0)
                    {
                        // The weightmap grid has its own resolution, so it gets its own
                        // projected surface, sampled off the height grid underneath it.
                        var layerSurface = (wmW == info.LzW && wmH == info.LzH)
                            ? heightSurface
                            : BuildSurfaceFromHeightGrid(info, wmW, wmH, zOffsetCm);
                        ScatterLayers(layerSurface, srcLayers, weightDsts, layerZ);
                    }

                    Log.Information("    Done '{0}': src {1}×{2}.", info.Name, info.SrcW, info.SrcH);
                }

                string hmDir = Path.Combine(exportFolder, "_heightmap");
                Directory.CreateDirectory(hmDir);

                EncodeHeights(heightZ, heightDst);

                string hPath = Path.Combine(hmDir, mapName + ".png");
                using (var fs = File.OpenWrite(hPath))
                    heightDst.Save(fs, new PngEncoder());
                Log.Information("  → {0}", hPath);

                foreach (var (name, bm) in weightDsts)
                {
                    string layerDir = Path.Combine(exportFolder, "_layers", SanitizeFileName(name));
                    Directory.CreateDirectory(layerDir);
                    string wPath = Path.Combine(layerDir, mapName + ".png");
                    using var enc = bm.Encode(SKEncodedImageFormat.Png, 100);
                    File.WriteAllBytes(wPath, enc.ToArray());
                    writtenLayerPngs.Add(wPath);
                    Log.Information("  → {0}", wPath);
                }

                // Delete survivors: layer PNGs from previous runs of this region
                // that no longer correspond to a layer present in the current run.
                int deleted = 0;
                foreach (var stale in preexistingLayerPngs)
                {
                    if (writtenLayerPngs.Contains(stale)) continue;
                    try
                    {
                        File.Delete(stale);
                        deleted++;
                        Log.Information("  ✗ {0} (stale, removed)", stale);

                        // Tidy up: if the layer dir is now empty, drop it too.
                        string? dir = Path.GetDirectoryName(stale);
                        if (dir != null && Directory.Exists(dir) &&
                            !Directory.EnumerateFileSystemEntries(dir).Any())
                        {
                            Directory.Delete(dir);
                        }
                    }
                    catch (Exception ex)
                    {
                        Log.Warning("  Failed to delete stale '{0}': {1}", stale, ex.Message);
                    }
                }
                if (deleted > 0)
                    Log.Information("Removed {0} stale layer file(s) for '{1}'.", deleted, mapName);

                Log.Information("Landscape stitching complete ({0} layer(s)).", allPhysNames.Count);
            }
            finally
            {
                foreach (var bm in weightDsts.Values) bm.Dispose();
            }
        }

        private static LandscapeInfo? BuildInfo(ALandscapeProxy proxy)
        {
            if (proxy.LandscapeComponents.Length == 0)
            {
                Log.Debug("  '{0}' has no component refs, skipping.", proxy.Name);
                return null;
            }

            FVector  loc   = FVector.ZeroVector;
            FVector  scale = FVector.OneVector;
            FRotator rot   = FRotator.ZeroRotator;

            var rcIdx = proxy.GetOrDefault<FPackageIndex>("RootComponent");
            if (rcIdx != null && !rcIdx.IsNull)
            {
                try
                {
                    var rc = rcIdx.Load<UObject>();
                    if (rc != null)
                    {
                        loc   = rc.GetOrDefault<FVector>  ("RelativeLocation", FVector.ZeroVector);
                        scale = rc.GetOrDefault<FVector>  ("RelativeScale3D",  FVector.OneVector);
                        rot   = rc.GetOrDefault<FRotator> ("RelativeRotation", FRotator.ZeroRotator);
                    }
                    else Log.Warning("  RootComponent loaded as null for '{0}'.", proxy.Name);
                }
                catch (Exception ex)
                {
                    Log.Warning("  RootComponent load failed for '{0}': {1}", proxy.Name, ex.Message);
                }
            }
            else
            {
                loc   = proxy.GetOrDefault<FVector>  ("RelativeLocation", FVector.ZeroVector);
                scale = proxy.GetOrDefault<FVector>  ("RelativeScale3D",  FVector.OneVector);
                rot   = proxy.GetOrDefault<FRotator> ("RelativeRotation", FRotator.ZeroRotator);
            }

            var R = TransformMath.RotationMatrix(rot.Pitch, rot.Yaw, rot.Roll);

            int minX = int.MaxValue, minY = int.MaxValue;
            int maxX = int.MinValue, maxY = int.MinValue;

            var components  = new List<ULandscapeComponent>(proxy.LandscapeComponents.Length);
            var layerToPhys = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

            foreach (var ci in proxy.LandscapeComponents)
            {
                ULandscapeComponent? comp = null;
                try { comp = ci.Load<ULandscapeComponent>(); } catch { }
                if (comp == null) continue;

                components.Add(comp);
                comp.GetComponentExtent(ref minX, ref minY, ref maxX, ref maxY);

                foreach (var alloc in comp.WeightmapLayerAllocations)
                {
                    string raw = alloc.GetLayerName();
                    if (raw.Equals("NormalMap_DX", StringComparison.OrdinalIgnoreCase)) continue;
                    if (!layerToPhys.ContainsKey(raw))
                        layerToPhys[raw] = ResolvePhysName(alloc);
                }
            }

            if (components.Count == 0)
            {
                Log.Warning("  '{0}' has no loadable components, skipping.", proxy.Name);
                return null;
            }

            return new LandscapeInfo
            {
                Proxy           = proxy,
                Name            = proxy.Name,
                Components      = components.ToArray(),
                LocX            = loc.X,
                LocY            = loc.Y,
                LocZ            = loc.Z,
                ScaleX          = scale.X,
                ScaleY          = scale.Y,
                ScaleZ          = scale.Z,
                R00             = R[0, 0], R01 = R[0, 1], R02 = R[0, 2],
                R10             = R[1, 0], R11 = R[1, 1], R12 = R[1, 2],
                R20             = R[2, 0], R21 = R[2, 1], R22 = R[2, 2],
                SectionOffsetX  = proxy.LandscapeSectionOffset.X,
                SectionOffsetY  = proxy.LandscapeSectionOffset.Y,
                MinX            = minX, MinY = minY,
                MaxX            = maxX, MaxY = maxY,
                LayerToPhys     = layerToPhys,
            };
        }

        // Depth sentinel for quads UE marks as having no height. Far below any real
        // terrain, and far below the -10000 cm floor the heightmap output clamps at, so a
        // genuine surface always outranks it. NoDepthTest is the threshold to compare
        // against, loose enough that a bilinear tap partly over the sentinel still reads
        // as no-data.
        // Claimed by nothing yet. Far below any real terrain and below the -10000 cm
        // floor the heightmap output clamps at, so any genuine surface outranks it.
        private const float  NoDepth     = -1e9f;
        private const double NoDepthTest = -1e8;

        // Heights below this are dropped from the heightmap output rather than written.
        private const double HeightFloorCm = -10_000.0;

        private const double LandscapeZScale = 1.0 / 128.0;

        // A surface projected into world space: one world position per source texel, plus
        // which texels UE actually has data for.
        private sealed class Surface
        {
            public int      W, H;
            public double[] Wx = [];     // world X, cm
            public double[] Wy = [];     // world Y, cm
            public double[] Wz = [];     // world Z, cm - unclamped, so depth tests stay
                                         // meaningful below the heightmap's output floor
            public bool[]   Valid = [];
        }

        // Decodes the raw heightmap into local-space height, and records which texels carry
        // data. Nothing is projected here - that is BuildSurface's job - because the
        // weightmap grid needs these local heights too, at its own resolution.
        private static void ReadHeightSource(LandscapeInfo info, Image<L16> src)
        {
            int srcW = src.Width, srcH = src.Height;
            var localZs = new float[srcW * srcH];
            var valid   = new bool[srcW * srcH];

            src.ProcessPixelRows(acc =>
            {
                for (int y = 0; y < srcH; y++)
                {
                    var row = acc.GetRowSpan(y);
                    for (int x = 0; x < srcW; x++)
                    {
                        ushort raw = row[x].PackedValue;
                        // 0 is the UE4 no-data sentinel.
                        if (raw == 0) continue;

                        localZs[y * srcW + x] =
                            (float)((raw - 32768) * LandscapeZScale * info.ScaleZ);
                        valid[y * srcW + x] = true;
                    }
                }
            });

            info.LocalZCm    = localZs;
            info.LocalZValid = valid;
            info.LzW         = srcW;
            info.LzH         = srcH;
        }

        // Projects a grid of local heights through the proxy transform. This is the whole
        // of the geometry: a source texel at local (lx, ly, localZ) lands at
        //     world = Loc + R * (lx, ly, localZ)
        // with every term carried, tilt included. There is no inverse to solve and nothing
        // to iterate, so a pitch/roll tilt is not a special case here - and two surfaces
        // over one world XY are simply two samples that both land there, which is exactly
        // what the depth test needs to see.
        private static Surface BuildSurface(
            LandscapeInfo info, int w, int h, float[] localZ, bool[] valid, double zOffsetCm)
        {
            double qSpanX = info.MaxX - info.MinX;
            double qSpanY = info.MaxY - info.MinY;
            double quadsPerPxX = w > 1 ? qSpanX / (w - 1) : 0.0;
            double quadsPerPxY = h > 1 ? qSpanY / (h - 1) : 0.0;

            var surf = new Surface
            {
                W = w, H = h,
                Wx = new double[w * h],
                Wy = new double[w * h],
                Wz = new double[w * h],
                Valid = valid,
            };

            double minZ = double.PositiveInfinity, maxZ = double.NegativeInfinity;

            for (int y = 0; y < h; y++)
            {
                double lqy = info.MinY + y * quadsPerPxY;
                double ly  = (lqy - info.SectionOffsetY) * info.ScaleY;

                double rowX = info.LocX + info.R01 * ly;
                double rowY = info.LocY + info.R11 * ly;
                double rowZ = info.LocZ + info.R21 * ly + zOffsetCm;

                for (int x = 0; x < w; x++)
                {
                    int i = y * w + x;
                    if (!valid[i]) continue;

                    double lqx = info.MinX + x * quadsPerPxX;
                    double lx  = (lqx - info.SectionOffsetX) * info.ScaleX;
                    double lz  = localZ[i];

                    surf.Wx[i] = rowX + info.R00 * lx + info.R02 * lz;
                    surf.Wy[i] = rowY + info.R10 * lx + info.R12 * lz;
                    surf.Wz[i] = rowZ + info.R20 * lx + info.R22 * lz;

                    if (surf.Wz[i] < minZ) minZ = surf.Wz[i];
                    if (surf.Wz[i] > maxZ) maxZ = surf.Wz[i];
                }
            }

            if (!double.IsInfinity(minZ))
                Log.Information("    '{0}' height range: [{1:F0}, {2:F0}] cm (LocZ={3:F0})",
                    info.Name, minZ, maxZ, info.LocZ);

            return surf;
        }

        // The weightmap grid at its own resolution, riding on the height grid's surface.
        private static Surface BuildSurfaceFromHeightGrid(
            LandscapeInfo info, int w, int h, double zOffsetCm)
        {
            var hz = info.LocalZCm!;
            var hv = info.LocalZValid!;
            int lw = info.LzW, lh = info.LzH;

            var localZ = new float[w * h];
            var valid  = new bool[w * h];

            for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                // Both grids span the same local quad extent, so position maps by ratio.
                double fx = w > 1 ? x * (lw - 1.0) / (w - 1.0) : 0.0;
                double fy = h > 1 ? y * (lh - 1.0) / (h - 1.0) : 0.0;

                int x0 = Math.Clamp((int)fx, 0, lw - 1);
                int y0 = Math.Clamp((int)fy, 0, lh - 1);
                int x1 = Math.Min(x0 + 1, lw - 1);
                int y1 = Math.Min(y0 + 1, lh - 1);

                // A texel is only usable where the whole cell under it has data; blending
                // through the sentinel would invent a surface that is not there.
                if (!hv[y0 * lw + x0] || !hv[y0 * lw + x1] ||
                    !hv[y1 * lw + x0] || !hv[y1 * lw + x1]) continue;

                double tx = fx - x0, ty = fy - y0;
                localZ[y * w + x] = (float)(
                    (1 - tx) * (1 - ty) * hz[y0 * lw + x0] +
                          tx  * (1 - ty) * hz[y0 * lw + x1] +
                    (1 - tx) *       ty  * hz[y1 * lw + x0] +
                          tx  *       ty  * hz[y1 * lw + x1]);
                valid[y * w + x] = true;
            }

            return BuildSurface(info, w, h, localZ, valid, zOffsetCm);
        }

        // One triangle of the projected surface, set up in destination-pixel space.
        private readonly struct Tri
        {
            public readonly double X0, Y0, X1, Y1, X2, Y2, InvDet;
            public readonly int MinX, MinY, MaxX, MaxY;
            public readonly bool Ok;

            public Tri(double x0, double y0, double x1, double y1, double x2, double y2,
                       int outputSize)
            {
                X0 = x0; Y0 = y0; X1 = x1; Y1 = y1; X2 = x2; Y2 = y2;

                double det = (Y1 - Y2) * (X0 - X2) + (X2 - X1) * (Y0 - Y2);
                // Degenerate once projected - a sliver seen edge-on contributes nothing,
                // and its neighbours cover the same ground.
                if (Math.Abs(det) < 1e-12)
                {
                    InvDet = 0; MinX = MinY = 0; MaxX = MaxY = -1; Ok = false;
                    return;
                }
                InvDet = 1.0 / det;

                MinX = (int)Math.Ceiling (Math.Min(x0, Math.Min(x1, x2)));
                MaxX = (int)Math.Floor   (Math.Max(x0, Math.Max(x1, x2)));
                MinY = (int)Math.Ceiling (Math.Min(y0, Math.Min(y1, y2)));
                MaxY = (int)Math.Floor   (Math.Max(y0, Math.Max(y1, y2)));

                if (MinX < 0) MinX = 0;
                if (MinY < 0) MinY = 0;
                if (MaxX > outputSize - 1) MaxX = outputSize - 1;
                if (MaxY > outputSize - 1) MaxY = outputSize - 1;

                Ok = MinX <= MaxX && MinY <= MaxY;
            }

            // Barycentric weights of a destination pixel centre, or false if outside.
            public bool Weights(double px, double py, out double b0, out double b1, out double b2)
            {
                b0 = ((Y1 - Y2) * (px - X2) + (X2 - X1) * (py - Y2)) * InvDet;
                b1 = ((Y2 - Y0) * (px - X2) + (X0 - X2) * (py - Y2)) * InvDet;
                b2 = 1.0 - b0 - b1;
                return b0 >= 0.0 && b1 >= 0.0 && b2 >= 0.0;
            }
        }

        // Walks the source grid cell by cell, splitting each into the two triangles UE
        // renders it as, and hands them to the caller in destination-pixel space. Triangles
        // tile the surface, so the projection is gap-free however the proxy is rotated, and
        // a folded surface simply delivers both of its sheets to the same pixels.
        private delegate void TriSink(in Tri tri, int i0, int i1, int i2);

        private static void ForEachTriangle(
            Surface surf, double pixelsPerCm, int outputCenter, int outputSize, TriSink sink)
        {
            int w = surf.W, h = surf.H;

            for (int y = 0; y < h - 1; y++)
            for (int x = 0; x < w - 1; x++)
            {
                int a = y * w + x, b = a + 1, c = a + w, d = c + 1;
                if (!surf.Valid[a] || !surf.Valid[b] ||
                    !surf.Valid[c] || !surf.Valid[d]) continue;

                double ax = outputCenter + surf.Wx[a] * pixelsPerCm;
                double ay = outputCenter + surf.Wy[a] * pixelsPerCm;
                double bx = outputCenter + surf.Wx[b] * pixelsPerCm;
                double by = outputCenter + surf.Wy[b] * pixelsPerCm;
                double cx = outputCenter + surf.Wx[c] * pixelsPerCm;
                double cy = outputCenter + surf.Wy[c] * pixelsPerCm;
                double dx = outputCenter + surf.Wx[d] * pixelsPerCm;
                double dy = outputCenter + surf.Wy[d] * pixelsPerCm;

                var t0 = new Tri(ax, ay, bx, by, dx, dy, outputSize);
                if (t0.Ok) sink(in t0, a, b, d);

                var t1 = new Tri(ax, ay, dx, dy, cx, cy, outputSize);
                if (t1.Ok) sink(in t1, a, d, c);
            }
        }

        private static void ScatterHeights(Surface surf, float[] depth)
        {
            var wz = surf.Wz;

            ForEachTriangle(surf, HmPixelsPerCm, HmOutputCenter, HmOutputSize,
                (in Tri t, int i0, int i1, int i2) =>
            {
                double z0 = wz[i0], z1 = wz[i1], z2 = wz[i2];

                for (int py = t.MinY; py <= t.MaxY; py++)
                for (int px = t.MinX; px <= t.MaxX; px++)
                {
                    if (!t.Weights(px, py, out double b0, out double b1, out double b2))
                        continue;

                    double z = b0 * z0 + b1 * z1 + b2 * z2;
                    // Below the output floor this surface is not written at all, and
                    // nothing lower than it could win the pixel either.
                    if (z < HeightFloorCm) continue;

                    int di = py * HmOutputSize + px;
                    if (z > depth[di]) depth[di] = (float)z;
                }
            });
        }

        // Every material of one proxy resolved in a single pass: whichever sample is
        // highest at a pixel wins it outright and writes all of the materials, including a
        // zero for the ones this proxy does not paint. A lower surface writes nothing, so a
        // pixel's materials always describe one surface - the visible one - rather than a
        // mixture of a surface and whatever is buried beneath it.
        private static unsafe void ScatterLayers(
            Surface surf, Dictionary<string, byte[]> srcLayers,
            Dictionary<string, SKBitmap> dsts, float[] depth)
        {
            int n = dsts.Count;
            var dstPtr = new IntPtr[n];
            var dstRb  = new int[n];
            var srcArr = new byte[]?[n];
            {
                int k = 0;
                foreach (var (name, bm) in dsts)
                {
                    dstPtr[k] = bm.GetPixels();
                    dstRb[k]  = bm.RowBytes;
                    srcArr[k] = srcLayers.TryGetValue(name, out var a) ? a : null;
                    k++;
                }
            }

            var wz = surf.Wz;

            ForEachTriangle(surf, WmPixelsPerCm, WmOutputCenter, WmOutputSize,
                (in Tri t, int i0, int i1, int i2) =>
            {
                double z0 = wz[i0], z1 = wz[i1], z2 = wz[i2];

                for (int py = t.MinY; py <= t.MaxY; py++)
                for (int px = t.MinX; px <= t.MaxX; px++)
                {
                    if (!t.Weights(px, py, out double b0, out double b1, out double b2))
                        continue;

                    double z  = b0 * z0 + b1 * z1 + b2 * z2;
                    int    di = py * WmOutputSize + px;
                    if (z <= depth[di]) continue;
                    depth[di] = (float)z;

                    for (int k = 0; k < n; k++)
                    {
                        var src = srcArr[k];
                        byte v = 0;
                        if (src != null)
                        {
                            double a = b0 * src[i0] + b1 * src[i1] + b2 * src[i2];
                            v = (byte)Math.Clamp((int)Math.Round(a), 0, 255);
                        }
                        ((byte*)dstPtr[k])[py * dstRb[k] + px] = v;
                    }
                }
            });
        }

        private static unsafe byte[] CopyGray8(SKBitmap src)
        {
            int w = src.Width, h = src.Height;
            var arr = new byte[w * h];
            byte* sp = (byte*)src.GetPixels();
            int rb   = src.RowBytes;
            for (int y = 0; y < h; y++)
                Marshal.Copy((IntPtr)(sp + y * rb), arr, y * w, w);
            return arr;
        }

        // The depth buffer is the heightmap: encode what survived the depth test.
        // Stored as output_raw = worldZ + 32768 (32768 = zero height), 0 = no data.
        private static void EncodeHeights(float[] depth, Image<L16> dst)
        {
            dst.ProcessPixelRows(acc =>
            {
                for (int y = 0; y < HmOutputSize; y++)
                {
                    var row = acc.GetRowSpan(y);
                    for (int x = 0; x < HmOutputSize; x++)
                    {
                        float z = depth[y * HmOutputSize + x];
                        if (z < NoDepthTest) continue;
                        row[x] = new L16((ushort)Math.Clamp(
                            (int)Math.Round(z + 32768.0), 0, 65535));
                    }
                }
            });
        }

        // UE's visibility mask. It punches holes in the landscape rather than painting it,
        // so it never takes a share of the layer blend.
        private const string VisibilityLayer = "DataLayer__";

        // A landscape texel with every weight at zero is not a hole - the game still draws
        // ground there. Every layer of the Revamp masters is LB_HeightBlend, and
        // UMaterialExpressionLandscapeLayerBlend::Compile floors each layer's
        // height-modified weight before dividing the lot through by their sum:
        //
        //     w_i = clamp(2*W_i - 1 + H_i, 0.0001, 1),   out = Σ(layer_i * w_i) / Σ(w)
        //
        // The height inputs are mask textures in 0..1, so where every painted weight W_i is
        // zero every layer lands on that same 0.0001 floor and the division hands them back
        // in equal shares - a component with one allocated layer renders it at full
        // strength. Only the layers the component actually allocates take part:
        // FHLSLMaterialTranslator::StaticTerrainLayerWeight compiles the others out of that
        // component's shader permutation. (The one layer that survives not being allocated
        // is 'a', the biome's base ground, which carries PreviewWeight 1.0 and so falls back
        // to a constant weight - but every component allocates it in practice.)
        //
        // So the fallback is per component, not global: Callahan's Passage falls back to
        // 'a', Fisherman's Row to 'c'. Reproduce that in the proxy's vertex grid, which is
        // the same grid TryConvert lays its weightmaps out on.
        private static void FillUnpainted(
            LandscapeInfo info,
            Dictionary<string, byte[]> srcLayers,
            Dictionary<string, SKBitmap> weightDsts,
            int wmW, int wmH)
        {
            // Summed over every layer, not just the component's own, so a texel a
            // neighbour has already painted - the two share a vertex row along their seam -
            // reads as painted here too. Claimed texels are marked in the same array, which
            // hands a contested seam to whichever component reaches it first.
            var claimed = new int[wmW * wmH];
            foreach (var (_, arr) in srcLayers)
                for (int i = 0; i < claimed.Length; i++)
                    claimed[i] += arr[i];

            long filled = 0;
            var physNames = new List<string>();

            foreach (var comp in info.Components)
            {
                physNames.Clear();

                foreach (var alloc in comp.WeightmapLayerAllocations)
                {
                    string raw = alloc.GetLayerName();
                    if (raw.Equals("NormalMap_DX", StringComparison.OrdinalIgnoreCase)) continue;

                    if (!info.LayerToPhys.TryGetValue(raw, out var phys)) phys = raw;
                    if (phys.Equals(VisibilityLayer, StringComparison.OrdinalIgnoreCase)) continue;
                    if (!weightDsts.ContainsKey(phys)) continue;

                    // Several raw layers of one component can resolve to a single physical
                    // material. The game averages them, which - their inputs being the same
                    // material - comes to that material at full strength, so counting the
                    // material once is what reproduces it.
                    if (!physNames.Contains(phys, StringComparer.OrdinalIgnoreCase))
                        physNames.Add(phys);
                }

                if (physNames.Count == 0) continue;

                byte share = (byte)Math.Clamp((int)Math.Round(255.0 / physNames.Count), 1, 255);

                var arrays = new byte[physNames.Count][];
                for (int k = 0; k < physNames.Count; k++)
                {
                    if (!srcLayers.TryGetValue(physNames[k], out var arr))
                    {
                        // Allocated but zero across the whole proxy, so TryConvert never
                        // made it a bitmap. It still takes its share of the fallback.
                        arr = new byte[wmW * wmH];
                        srcLayers[physNames[k]] = arr;
                    }
                    arrays[k] = arr;
                }

                int x0   = comp.SectionBaseX - info.MinX;
                int y0   = comp.SectionBaseY - info.MinY;
                int span = comp.ComponentSizeQuads + 1;

                for (int y = y0; y < y0 + span; y++)
                {
                    if ((uint)y >= (uint)wmH) continue;
                    int row = y * wmW;

                    for (int x = x0; x < x0 + span; x++)
                    {
                        if ((uint)x >= (uint)wmW) continue;

                        int i = row + x;
                        if (claimed[i] != 0) continue;

                        foreach (var arr in arrays) arr[i] = share;
                        claimed[i] = share;
                        filled++;
                    }
                }
            }

            if (filled > 0)
                Log.Information(
                    "    Unpainted texels: {0} of {1} filled from their component's layers.",
                    filled, (long)wmW * wmH);
        }

        private static string ResolvePhysName(FWeightmapLayerAllocationInfo alloc)
        {
            try
            {
                if (alloc.LayerInfo != null && !alloc.LayerInfo.IsNull)
                {
                    var infoObj = alloc.LayerInfo.Load<ULandscapeLayerInfoObject>();
                    if (infoObj != null)
                    {
                        if (infoObj.PhysMaterial != null && !infoObj.PhysMaterial.IsNull)
                        {
                            string pn = infoObj.PhysMaterial.Name;
                            if (!string.IsNullOrEmpty(pn) && pn != "None")
                                return StripPhysSuffix(pn);
                        }

                        if (!infoObj.LayerName.IsNone)
                            return infoObj.LayerName.Text;
                    }
                }
            }
            catch (Exception ex)
            {
                Log.Verbose("ResolvePhysName '{0}': {1}", alloc.GetLayerName(), ex.Message);
            }

            string raw = alloc.GetLayerName();
            int idx    = raw.IndexOf("_LayerInfo", StringComparison.OrdinalIgnoreCase);
            return idx > 0 ? raw[..idx] : raw;
        }

        private static string StripPhysSuffix(string name)
        {
            const string suffix = "Phys";
            if (name.EndsWith(suffix, StringComparison.OrdinalIgnoreCase) &&
                name.Length > suffix.Length)
                return name[..^suffix.Length];
            return name;
        }

        private static string SanitizeFileName(string name)
        {
            foreach (char c in Path.GetInvalidFileNameChars())
                name = name.Replace(c, '_');
            return name;
        }
    }
}
