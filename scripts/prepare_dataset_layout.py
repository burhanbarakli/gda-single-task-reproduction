#!/usr/bin/env python3
"""Map the official downloaded folder into the exact runtime layout by hash."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--copy", action="store_true", help="Hardlink yerine dosyaları kopyala")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "manifests/datasets.json").read_text())
    expected_by_size = {}
    for row in manifest["files"]:
        expected_by_size.setdefault(row["bytes"], []).append(row)
    found = {}
    for source in args.download_root.expanduser().resolve().rglob("*"):
        if not source.is_file() or source.stat().st_size not in expected_by_size:
            continue
        digest = sha256(source)
        for row in expected_by_size[source.stat().st_size]:
            if digest == row["sha256"]:
                found[row["path"]] = source
    missing = [row["path"] for row in manifest["files"] if row["path"] not in found]
    if missing:
        raise SystemExit("Eksik resmî veri dosyaları: " + ", ".join(missing))
    data_root = args.data_root.expanduser().resolve()
    for relative, source in found.items():
        destination = data_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.stat().st_size != source.stat().st_size or sha256(destination) != sha256(source):
                raise RuntimeError(f"Hedefte farklı dosya var: {destination}")
            continue
        if args.copy:
            shutil.copy2(source, destination)
        else:
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        print(f"{relative} <- {source}")
    print("Hazır veri kökü:", data_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

