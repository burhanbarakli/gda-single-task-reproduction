#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINTS="${1:?Kullanım: evaluate_all.sh CHECKPOINT_DIR DATA_ROOT [RESULTS_ROOT]}"
DATA_ROOT="${2:?Kullanım: evaluate_all.sh CHECKPOINT_DIR DATA_ROOT [RESULTS_ROOT]}"
RESULTS_ROOT="${3:-$ROOT/results/final}"
for method in ot-sim2real MMD cotrain source_only target_only; do
  python "$ROOT/scripts/evaluate_method.py" --method "$method" --checkpoint "$CHECKPOINTS/${method}_seed1_epoch500.pth" --data-root "$DATA_ROOT" --results-root "$RESULTS_ROOT"
done
python "$ROOT/scripts/aggregate_final.py" --results-root "$RESULTS_ROOT"

