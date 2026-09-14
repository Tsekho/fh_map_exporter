using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace Exporter
{
    internal static class Constants
    {
        /// <summary>Maps "destroyed" blueprint paths → replacement blueprint paths.</summary>
        public static readonly Dictionary<string, string> Patches = new()
        {
            ["War/Content/Blueprints/Structures/Bases/BPDestroyedRelicBase1"] = "War/Content/Blueprints/Structures/Bases/BPRelicBase1",
            ["War/Content/Blueprints/Structures/Bases/BPDestroyedRelicBase2"] = "War/Content/Blueprints/Structures/Bases/BPRelicBase2",
            ["War/Content/Blueprints/Structures/Bases/BPDestroyedRelicBase3"] = "War/Content/Blueprints/Structures/Bases/BPRelicBase3",
            ["War/Content/Blueprints/Structures/BPDestroyedKeep"]             = "War/Content/Blueprints/Structures/BPKeep",
            ["War/Content/Blueprints/Structures/BPDestroyedObservationTower"] = "War/Content/Blueprints/Structures/BPObservationTower",
        };

        /// <summary>Component ExportTypes that are always skipped.</summary>
        public static readonly HashSet<string> SkipTypes = new(System.StringComparer.OrdinalIgnoreCase)
        {
            "DecalComponent",
            "SplineComponent",
            "SplineMeshComponent",
        };

        /// <summary>Component object names that are always skipped (e.g. "DestroyedMesh"
        /// - a native StaticMeshComponent on vehicle BPs that is only shown when the
        /// vehicle is wrecked; without this filter both live and destroyed hulls emit).</summary>
        public static readonly HashSet<string> SkipComponentNames = new(System.StringComparer.OrdinalIgnoreCase)
        {
            "DestroyedMesh",
        };

        public static string NormaliseMeshName(string meshName)
        {
            if (meshName == "Meshes__Structures__ConstructionYardCritical")
                return "Meshes__Structures__ConstructionYard";
            if (meshName == "Meshes__Structures__FortTrenches__AIBunkers__husks__FortT3WallBreach")
                return "Meshes__Structures__FortTrenches__FortT3Wall01";
            return meshName;
        }

        public static string NormaliseBPName(string bpName)
        {
            if (bpName == "BPConcreteBridgeNoBlocker_C")
                return "BPConcreteBridge_C";
            return bpName;
        }

        private static readonly string[] _patternStrings =
        {
            "Engine__Content__BasicShapes__.*",
            "Meshes__Measurement__Plane",
            "FX__Mesh__.*",
            "Meshes__SM_SkySphere"
        };

        private static readonly Regex[] _hardbanPatterns =
            System.Array.ConvertAll(_patternStrings, p => new Regex(p, RegexOptions.IgnoreCase | RegexOptions.Compiled));

        public static bool IsHardbanned(string unpathName)
        {
            foreach (var rx in _hardbanPatterns)
                if (rx.IsMatch(unpathName)) return true;
            return false;
        }

        /// <summary>
        /// Per-region Z offset in UE4 units (cm) added to all heightmap pixel heights
        /// and all JSON Z coordinates when exporting that region.
        /// Keys are lower-cased map names as they appear in the asset path.
        /// </summary>
        public static readonly Dictionary<string, double> HeightOffsets =
            new(System.StringComparer.OrdinalIgnoreCase)
        {
            ["shackledchasmhex"]    = 50,
            ["clahstrahex"]         = 42,
            ["drownedvalehex"]      = 40,
            ["endlessshorehex"]     = 50,
            ["reaverspasshex"]      = 42,
            ["sableporthex"]        = 50,
            ["mooringcountyhex"]    = 30,
            ["stonecradlehex"]      = 50,
            ["weatheredexpansehex"] = 50,
            ["stlicanshelfhex"]     = 42,
            ["tempestislandhex"]    = 50,
            ["wrestahex"]           = 50,
            ["farranaccoasthex"]    = 50,
            ["gutterhex"]           = 50,
            ["kingscagehex"]        = 50,
            ["westgatehex"]         = 50,
            ["fishermansrowhex"]    = 50,
            ["palantinebermhex"]    = 50,
            ["stemalandinghex"]     = 42,
            ["oarbreakerhex"]       = 70,
            ["lykosislehex"]        = 50,
            ["kuurastrandhex"]      = 50,
            ["paripeakhex"]         = 50,
            ["thefingershex"]       = 50,
            ["olaviswakehex"]       = 50,
            ["onyxhex"]             = 50,
            ["tyrantfoothillshex"]  = 50,
            ["pipersenclavehex"]    = 50,
            ["homeregionc"]         = 450,
            ["homeregionw"]         = 450,
        };

        /// <summary>Returns the Z offset (cm) for the given map name, or 0 if not found.</summary>
        public static double GetHeightOffset(string mapName) =>
            HeightOffsets.TryGetValue(mapName, out var v) ? v : 0.0;
    }
}
