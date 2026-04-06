#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <output.csv|parquet> [extra args...]"
  exit 1
fi

OUTPUT_PATH=$1
shift

.venv/bin/python -m ml.training.generate_synthetic_sensor_metrics \
  --output "$OUTPUT_PATH" \
  "$@"
