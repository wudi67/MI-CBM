#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../grouped_dynamic_vqc/scripts/environment.sh"
exec "$VQC_PYTHON" -m experiments.grouped_independent_feedback_ablation.operations launch "$@"
