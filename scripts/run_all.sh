#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:?Kullanım: run_all.sh DATA_ROOT [OUTPUT_ROOT]}"
OUTPUT_ROOT="${2:-$ROOT/runs}"
for method in ot-sim2real MMD cotrain source_only target_only; do
  python "$ROOT/scripts/train_method.py" --method "$method" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT"
done

