#!/usr/bin/env python3
"""Aggregate completed final evaluations without changing denominators."""
import argparse
import json
from pathlib import Path

METHODS = ["ot-sim2real", "MMD", "cotrain", "source_only", "target_only"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for method in METHODS:
        path = args.results_root / method / "evaluation.json"
        value = json.loads(path.read_text())
        if value.get("status") != "completed" or len(value.get("episodes", [])) != 100:
            raise RuntimeError(f"Eksik final test: {method}")
        down, up = value["domains"]["down"], value["domains"]["up"]
        successes = down["successes"] + up["successes"]
        rows.append({"method": method, "down": down, "up": up, "successes": successes, "rollouts": 100, "success_rate": successes / 100, "checkpoint_sha256": value["checkpoint_sha256"]})
    summary = {"protocol": "50 down + 50 up per method", "methods": rows}
    output = args.results_root / "summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

