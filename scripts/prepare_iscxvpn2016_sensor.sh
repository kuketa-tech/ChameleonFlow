#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <input.{csv|arff}> <output.{csv|parquet}> [extra args...]" >&2
  exit 1
fi

INPUT_PATH=$1
OUTPUT_PATH=$2
shift 2

.venv/bin/python main.py prepare-iscxvpn2016-sensor \
  --input "$INPUT_PATH" \
  --output "$OUTPUT_PATH" \
  "$@"
