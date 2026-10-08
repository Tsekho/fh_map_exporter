"""Build Exporter.exe as a self-contained win-x64 single-file binary."""

import subprocess
import shutil

from utils import tui
from utils.config import (
    EXPORTER_EXE,
    EXPORTER_PROJECT,
    EXPORTER_PUBLISH_DIR,
    EXPORTER_RID,
)


def run(cmd: list[str], **kwargs) -> int:
    print(tui.dim(f"  > {' '.join(str(c) for c in cmd)}"))
    return subprocess.run(cmd, **kwargs).returncode


def main() -> int:
    if not EXPORTER_PROJECT.exists():
        tui.error(f"project file not found: {EXPORTER_PROJECT}")
        return 1

    tui.heading("Building Exporter", "Release / win-x64 / single-file")

    rc = run([
        "dotnet", "publish",
        str(EXPORTER_PROJECT),
        "-c", "Release",
        "-r", EXPORTER_RID,
        "--self-contained", "true",
        "-p:PublishSingleFile=true",
        "-p:EnableCompressionInSingleFile=true",
        "-p:IncludeNativeLibrariesForSelfExtract=true",
        "-p:DebugType=None",
        "-p:DebugSymbols=false",
        # Silence NuGet vulnerability warnings (ImageSharp 3.x via CUE4Parse;
        # the only fix is 4.x, which needs a paid license key to build).
        "-p:NuGetAudit=false",
    ])

    if rc != 0:
        tui.error(f"dotnet publish failed (exit code {rc})")
        return rc

    published = EXPORTER_PUBLISH_DIR / "Exporter.exe"
    if not published.exists():
        tui.error(f"expected output not found: {published}")
        return 1

    shutil.copy2(published, EXPORTER_EXE)
    size_mb = EXPORTER_EXE.stat().st_size / (1024 * 1024)
    tui.done(f"{EXPORTER_EXE}  {tui.dim(f'{size_mb:.1f} MB')}")
    return 0


if __name__ == "__main__":
    tui.run(main)
