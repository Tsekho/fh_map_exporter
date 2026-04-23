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
            public double R00, R01, R10, R11;

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

            double zOffsetCm = Constants.GetHeightOffset(mapName);
            if (zOffsetCm != 0.0)
                Log.Information("Applying Z offset of {0} cm to '{1}'.", zOffsetCm, mapName);

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
                            BlitHeightmap(info, hImg, heightDst, zOffsetCm);
                        else
                            Log.Warning("    Unexpected heightmap pixel format for '{0}'.", info.Name);

                        hImgBase.Dispose();
                    }

                    foreach (var (rawName, srcBm) in weightMaps)
                    {
                        if (rawName.Equals("NormalMap_DX", StringComparison.OrdinalIgnoreCase))
                        { srcBm.Dispose(); continue; }

                        if (info.SrcW == 0) { info.SrcW = srcBm.Width; info.SrcH = srcBm.Height; }

                        if (!info.LayerToPhys.TryGetValue(rawName, out var physName))
                            physName = rawName;

                        if (weightDsts.TryGetValue(physName, out var dstBm))
                            BlitWeightmap(info, srcBm, dstBm);

                        srcBm.Dispose();
                    }

                    Log.Information("    Done '{0}': src {1}×{2}.", info.Name, info.SrcW, info.SrcH);
                }

                string hmDir = Path.Combine(exportFolder, "_heightmap");
                Directory.CreateDirectory(hmDir);

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
                    Log.Information("  → {0}", wPath);
                }

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
                R00             = R[0, 0], R01 = R[0, 1],
                R10             = R[1, 0], R11 = R[1, 1],
                SectionOffsetX  = proxy.LandscapeSectionOffset.X,
                SectionOffsetY  = proxy.LandscapeSectionOffset.Y,
                MinX            = minX, MinY = minY,
                MaxX            = maxX, MaxY = maxY,
                LayerToPhys     = layerToPhys,
            };
        }

        private const double LandscapeZScale = 1.0 / 128.0;

        private static void BlitHeightmap(LandscapeInfo info, Image<L16> src, Image<L16> dst,
                                           double zOffsetCm = 0.0)
        {
            int srcW = src.Width, srcH = src.Height;

            // UE4 formula: height_cm = (raw - 32768) * LANDSCAPE_ZSCALE * scaleZ
            // Stored as: output_raw = height_cm + 32768 (32768 = zero height)
            var srcPixels = new ushort[srcW * srcH];
            src.ProcessPixelRows(acc =>
            {
                for (int y = 0; y < srcH; y++)
                {
                    var row = acc.GetRowSpan(y);
                    for (int x = 0; x < srcW; x++)
                    {
                        ushort raw = row[x].PackedValue;
                        // 0 is the UE4 "no-data" sentinel – preserve it so it never overwrites valid height data.
                        if (raw == 0) { srcPixels[y * srcW + x] = 0; continue; }
                        double heightCm = (raw - 32768) * LandscapeZScale * info.ScaleZ + info.LocZ
                                          + zOffsetCm;

                        if (heightCm < -10_000.0) { srcPixels[y * srcW + x] = 0; continue; }

                        double encoded  = heightCm + 32768.0;
                        srcPixels[y * srcW + x] =
                            (ushort)Math.Clamp((int)Math.Round(encoded), 0, 65535);
                    }
                }
            });

            GetOutputBounds(info, srcW, srcH,
                HmPixelsPerCm, HmOutputCenter, HmOutputSize,
                out int dxMin, out int dyMin, out int dxMax, out int dyMax);

            if (dxMin > dxMax || dyMin > dyMax) return;

            dst.ProcessPixelRows(acc =>
            {
                for (int dy = dyMin; dy <= dyMax; dy++)
                {
                    var outRow = acc.GetRowSpan(dy);
                    double wyCm = (dy - HmOutputCenter) * HmCmPerPixel;

                    for (int dx = dxMin; dx <= dxMax; dx++)
                    {
                        double wxCm = (dx - HmOutputCenter) * HmCmPerPixel;

                        if (!WorldToSourceFrac(info, srcW, srcH, wxCm, wyCm,
                                out double fracX, out double fracY))
                            continue;

                        ushort newVal = BilinearSampleU16(srcPixels, srcW, srcH, fracX, fracY);

                        // Keep the highest elevation at each output pixel; skip 0 (no-data sentinel).
                        if (newVal > 0 && newVal > outRow[dx].PackedValue)
                            outRow[dx] = new L16(newVal);
                    }
                }
            });
        }

        private static unsafe void BlitWeightmap(LandscapeInfo info, SKBitmap src, SKBitmap dst)
        {
            int srcW = src.Width, srcH = src.Height;

            var srcArr = new byte[srcW * srcH];
            byte* sp   = (byte*)src.GetPixels();
            int srcRB  = src.RowBytes;
            for (int y = 0; y < srcH; y++)
                Marshal.Copy((IntPtr)(sp + y * srcRB), srcArr, y * srcW, srcW);

            GetOutputBounds(info, srcW, srcH,
                WmPixelsPerCm, WmOutputCenter, WmOutputSize,
                out int dxMin, out int dyMin, out int dxMax, out int dyMax);

            if (dxMin > dxMax || dyMin > dyMax) return;

            byte* dp  = (byte*)dst.GetPixels();
            int dstRB = dst.RowBytes;

            for (int dy = dyMin; dy <= dyMax; dy++)
            {
                double wyCm  = (dy - WmOutputCenter) * WmCmPerPixel;
                byte*  dstRow = dp + dy * dstRB;

                for (int dx = dxMin; dx <= dxMax; dx++)
                {
                    double wxCm = (dx - WmOutputCenter) * WmCmPerPixel;

                    if (!WorldToSourceFrac(info, srcW, srcH, wxCm, wyCm,
                            out double fracX, out double fracY))
                        continue;

                    byte newVal = BilinearSampleU8(srcArr, srcW, srcH, fracX, fracY);

                    if (newVal > dstRow[dx])
                        dstRow[dx] = newVal;
                }
            }
        }

        // fracX/fracY are continuous pixel coordinates in [0, srcW-1] × [0, srcH-1].
        private static ushort BilinearSampleU16(
            ushort[] pixels, int srcW, int srcH, double fracX, double fracY)
        {
            int x0 = (int)Math.Floor(fracX);
            int y0 = (int)Math.Floor(fracY);
            int x1 = Math.Min(x0 + 1, srcW - 1);
            int y1 = Math.Min(y0 + 1, srcH - 1);
            x0 = Math.Max(x0, 0);
            y0 = Math.Max(y0, 0);

            double tx = fracX - x0;
            double ty = fracY - y0;

            double v =
                (1 - tx) * (1 - ty) * pixels[y0 * srcW + x0] +
                      tx  * (1 - ty) * pixels[y0 * srcW + x1] +
                (1 - tx) *       ty  * pixels[y1 * srcW + x0] +
                      tx  *       ty  * pixels[y1 * srcW + x1];

            return (ushort)Math.Clamp((int)Math.Round(v), 0, 65535);
        }

        private static byte BilinearSampleU8(
            byte[] pixels, int srcW, int srcH, double fracX, double fracY)
        {
            int x0 = (int)Math.Floor(fracX);
            int y0 = (int)Math.Floor(fracY);
            int x1 = Math.Min(x0 + 1, srcW - 1);
            int y1 = Math.Min(y0 + 1, srcH - 1);
            x0 = Math.Max(x0, 0);
            y0 = Math.Max(y0, 0);

            double tx = fracX - x0;
            double ty = fracY - y0;

            double v =
                (1 - tx) * (1 - ty) * pixels[y0 * srcW + x0] +
                      tx  * (1 - ty) * pixels[y0 * srcW + x1] +
                (1 - tx) *       ty  * pixels[y1 * srcW + x0] +
                      tx  *       ty  * pixels[y1 * srcW + x1];

            return (byte)Math.Clamp((int)Math.Round(v), 0, 255);
        }

        // Maps a world-space cm position to a fractional source pixel coordinate.
        // Returns false when the position falls outside the source image.
        private static bool WorldToSourceFrac(
            LandscapeInfo info, int srcW, int srcH,
            double wxCm, double wyCm,
            out double fracX, out double fracY)
        {
            double dx = wxCm - info.LocX;
            double dy = wyCm - info.LocY;

            // Inverse rotation (transpose of 2×2 rotation sub-matrix)
            double lx = info.R00 * dx + info.R10 * dy;
            double ly = info.R01 * dx + info.R11 * dy;

            double lqx = lx / info.ScaleX + info.SectionOffsetX;
            double lqy = ly / info.ScaleY + info.SectionOffsetY;

            fracX = (lqx - info.MinX) / (info.MaxX - info.MinX) * (srcW - 1);
            fracY = (lqy - info.MinY) / (info.MaxY - info.MinY) * (srcH - 1);

            return fracX >= 0.0 && fracX <= srcW - 1 &&
                   fracY >= 0.0 && fracY <= srcH - 1;
        }

        // Computes the bounding box in output-pixel space covered by this landscape,
        // clamped to [0, outputSize-1].
        private static void GetOutputBounds(
            LandscapeInfo info, int srcW, int srcH,
            double pixelsPerCm, int outputCenter, int outputSize,
            out int dxMin, out int dyMin, out int dxMax, out int dyMax)
        {
            int[] cqx = { info.MinX, info.MaxX, info.MinX, info.MaxX };
            int[] cqy = { info.MinY, info.MinY, info.MaxY, info.MaxY };

            int oMinX = int.MaxValue, oMinY = int.MaxValue;
            int oMaxX = int.MinValue, oMaxY = int.MinValue;

            for (int i = 0; i < 4; i++)
            {
                double lx = (cqx[i] - info.SectionOffsetX) * info.ScaleX;
                double ly = (cqy[i] - info.SectionOffsetY) * info.ScaleY;
                double wx = info.LocX + info.R00 * lx + info.R01 * ly;
                double wy = info.LocY + info.R10 * lx + info.R11 * ly;

                int ox = (int)Math.Round(outputCenter + wx * pixelsPerCm);
                int oy = (int)Math.Round(outputCenter + wy * pixelsPerCm);

                oMinX = Math.Min(oMinX, ox);
                oMinY = Math.Min(oMinY, oy);
                oMaxX = Math.Max(oMaxX, ox);
                oMaxY = Math.Max(oMaxY, oy);
            }

            // 1-pixel margin to avoid border gaps from rounding
            dxMin = Math.Max(0,              oMinX - 1);
            dyMin = Math.Max(0,              oMinY - 1);
            dxMax = Math.Min(outputSize - 1, oMaxX + 1);
            dyMax = Math.Min(outputSize - 1, oMaxY + 1);
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
