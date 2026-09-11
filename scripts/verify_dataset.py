#!/usr/bin/env python3
"""Verify every frozen GDA data file by size and SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify(data_root: Path) -> dict:
    manifest = json.loads((ROOT / "manifests/datasets.json").read_text(encoding="utf-8"))
    rows = []
    for expected in manifest["files"]:
        path = data_root / expected["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        row = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
        if row["bytes"] != expected["bytes"] or row["sha256"] != expected["sha256"]:
            raise RuntimeError(f"Veri doğrulaması başarısız: {row}")
        rows.append(row)
    return {"ok": True, "data_root": str(data_root.resolve()), "files": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.data_root.expanduser().resolve()), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

