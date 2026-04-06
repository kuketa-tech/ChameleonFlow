#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <train.csv|parquet> <eval.csv|parquet> <output.json> [extra args...]" >&2
  exit 1
fi

TRAIN_INPUT=$1
EVAL_INPUT=$2
OUTPUT_SUMMARY=$3
shift 3

.venv/bin/python main.py evaluate-sensor-transfer \
  "$TRAIN_INPUT" \
  "$EVAL_INPUT" \
  "$OUTPUT_SUMMARY" \
  "$@"
