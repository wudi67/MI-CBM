#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/environment.sh"
exec "$VQC_PYTHON" -m experiments.grouped_dynamic_vqc.train "$@"
