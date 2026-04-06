#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 <run-dir> <run-id> <domain> <application> <scenario> [extra args...]" >&2
  exit 1
fi

RUN_DIR=$1
RUN_ID=$2
DOMAIN=$3
APPLICATION=$4
SCENARIO=$5
shift 5

.venv/bin/python main.py init-sensor-experiment \
  "$RUN_DIR" \
  "$RUN_ID" \
  "$DOMAIN" \
  "$APPLICATION" \
  "$SCENARIO" \
  "$@"
