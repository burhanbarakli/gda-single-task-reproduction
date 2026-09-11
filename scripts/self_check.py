#!/usr/bin/env python3
"""Fast repository checks; optional runtime checks after bootstrap."""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    protocol = json.loads((ROOT / "manifests/protocol.json").read_text())
    datasets = json.loads((ROOT / "manifests/datasets.json").read_text())
    assert len(protocol["methods"]) == 5
    assert protocol["training"]["epochs"] == 500
    assert protocol["training"]["updates_per_method"] == 150000
    assert len(datasets["files"]) == 6
    assert sum(row["bytes"] for row in datasets["files"]) == datasets["total_bytes"]
    for method in protocol["methods"]:
        assert (ROOT / "config/reference" / f"{method}_seed1.json").is_file()
    report = {"ok": True, "python": platform.python_version(), "patch_sha256": sha256(ROOT / "patches/0001-demo-rights-and-seeds.patch")}
    if args.runtime:
        import torch
        report.update(torch=torch.__version__, cuda=torch.cuda.is_available())
        for name, commit in {"ot-sim2real": protocol["official_commit"], **protocol["dependency_commits"]}.items():
            actual = subprocess.check_output(["git", "-C", str(ROOT / "vendor" / name), "rev-parse", "HEAD"], text=True).strip()
            assert actual == commit, (name, actual, commit)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

