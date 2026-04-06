#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 <input-a.csv|parquet> <input-b.csv|parquet> <output-model> <output-metadata> [extra args...]" >&2
  exit 1
fi

INPUT_A=$1
INPUT_B=$2
OUTPUT_MODEL=$3
OUTPUT_METADATA=$4
shift 4

.venv/bin/python main.py train-sensor-multidomain \
  --input "$INPUT_A" \
  --input "$INPUT_B" \
  "$OUTPUT_MODEL" \
  "$OUTPUT_METADATA" \
  "$@"
