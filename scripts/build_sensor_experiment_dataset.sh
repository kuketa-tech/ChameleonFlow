#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <run-dir-a> <output.csv|parquet> [run-dir-b ...]" >&2
  exit 1
fi

OUTPUT_PATH=${@: -1}
INPUT_DIRS=("${@:1:$#-1}")

CMD=(.venv/bin/python main.py build-sensor-experiment-dataset)
for INPUT_DIR in "${INPUT_DIRS[@]}"; do
  CMD+=(--input "$INPUT_DIR")
done
CMD+=("$OUTPUT_PATH")

"${CMD[@]}"
