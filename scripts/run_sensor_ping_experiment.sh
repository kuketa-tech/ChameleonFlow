#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <run-dir> <target> <interface> [extra args...]" >&2
  exit 1
fi

RUN_DIR=$1
TARGET=$2
INTERFACE=$3
shift 3

.venv/bin/python main.py run-sensor-ping-experiment \
  "$RUN_DIR" \
  "$TARGET" \
  "$INTERFACE" \
  "$@"
