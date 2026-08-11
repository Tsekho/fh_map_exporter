"""Write short hashes of the exported map JSONs to export/hashes.txt.

Only the top-level (non-proxy) files in export/_json are hashed; the
proxies subdirectory is skipped. One "filename  hash" line per file
with names padded to align, sorted by filename for stable diffs. The
hash is a truncated SHA-256 -- just enough to see when a JSON changes.
"""

import hashlib
import os

HASH_LEN = 8

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
JSON_DIR = os.path.join(ROOT, "export", "_json")
OUT_PATH = os.path.join(ROOT, "export", "hashes.txt")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:HASH_LEN]


def main():
    names = sorted(
        name for name in os.listdir(JSON_DIR)
        if name.endswith(".json") and os.path.isfile(os.path.join(JSON_DIR, name))
    )
    width = max(len(name) for name in names)
    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Region json hashes to track single region changes\n")
        for name in names:
            digest = file_hash(os.path.join(JSON_DIR, name))
            line = f"{name:<{width}}  {digest}"
            f.write(line + "\n")
            print(f"  {line}")
    print(f"done, {len(names)} file(s) hashed -> {OUT_PATH}")


if __name__ == "__main__":
    main()
