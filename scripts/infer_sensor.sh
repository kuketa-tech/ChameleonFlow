#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python main.py infer-sensor "$@"
