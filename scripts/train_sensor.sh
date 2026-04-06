#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <input.csv|parquet> <output_model.txt> <output_metadata.json> [extra args...]"
  exit 1
fi

INPUT_PATH=$1
OUTPUT_MODEL=$2
OUTPUT_METADATA=$3
shift 3

.venv/bin/python -m ml.training.train_sensor \
  --input "$INPUT_PATH" \
  --output-model "$OUTPUT_MODEL" \
  --output-metadata "$OUTPUT_METADATA" \
  "$@"
