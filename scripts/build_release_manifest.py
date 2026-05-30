"""Build a release manifest.json for RemoteUpdateWorker.

Usage:
    python3 scripts/build_release_manifest.py [--dir releases/latest]

The script computes SHA256 + size for each known asset in `releases/latest/`
and writes a manifest.json next to them. Version defaults to UTC timestamp
'YYYYMMDD-HHMM' but can be overridden with --version.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

ALLOWED_FILES = (
    "spam_numbers.csv",
    "prefix_risk.json",
    "prefix_histogram.json",
    "prefix_histogram_3.json",
    "prefix_histogram_7.json",
    "def_code_risk.json",
    "def_code_operator_risk.json",
    "operator_bucket.json",
    "spam_model.tflite",
    "model_card.json",
    "app_category_model.tflite",
    "app_category_vocab.txt",
    "app_category_card.json",
)

DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "..", "releases", "latest")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=DEFAULT_DIR)
    parser.add_argument("--version", default=None)
    parser.add_argument("--min-app-db-version", type=int, default=6)
    args = parser.parse_args(argv)

    release_dir = os.path.abspath(args.dir)
    if not os.path.isdir(release_dir):
        print(f"release dir not found: {release_dir}", file=sys.stderr)
        return 1

    files = {}
    for name in ALLOWED_FILES:
        path = os.path.join(release_dir, name)
        if not os.path.exists(path):
            print(f"missing: {name}", file=sys.stderr)
            continue
        files[name] = {
            "sha256": sha256(path),
            "size": os.path.getsize(path),
            "url": name,
        }

    if not files:
        print("no files to include", file=sys.stderr)
        return 1

    version = args.version or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    manifest = {
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "min_app_db_version": args.min_app_db_version,
        "files": files,
    }

    manifest_path = os.path.join(release_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"wrote {manifest_path} (version={version}, files={len(files)})")
    for name, entry in files.items():
        print(f"  {name}: sha256={entry['sha256'][:12]}\u2026 size={entry['size']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
