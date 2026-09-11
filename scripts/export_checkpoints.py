#!/usr/bin/env python3
"""Copy final checkpoints into a portable, hashed bundle."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

METHODS = ["ot-sim2real", "MMD", "cotrain", "source_only", "target_only"]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in METHODS:
        candidates = sorted(args.runs_root.glob(f"g5_{method}_seed1_*/*/models/model_epoch_500.pth"))
        if not candidates:
            raise FileNotFoundError(f"{method} epoch 500 checkpoint bulunamadı")
        source = candidates[-1]
        destination = args.output / f"{method}_seed1_epoch500.pth"
        shutil.copy2(source, destination)
        rows.append({"method": method, "file": destination.name, "bytes": destination.stat().st_size, "sha256": sha256(destination), "source": str(source.resolve())})
    (args.output / "manifest.json").write_text(json.dumps({"checkpoints": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

