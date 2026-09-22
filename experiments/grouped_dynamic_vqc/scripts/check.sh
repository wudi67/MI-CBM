#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/environment.sh"
GROUPED_VQC_RUFF=${GROUPED_VQC_RUFF:-}
GROUPED_VQC_TY=${GROUPED_VQC_TY:-}
if [[ -z "$GROUPED_VQC_RUFF" ]]; then
  for candidate in /root/.vscode-server/extensions/charliermarsh.ruff-*/bundled/libs/bin/ruff; do
    if [[ -x "$candidate" ]]; then GROUPED_VQC_RUFF="$candidate"; fi
  done
fi
if [[ -z "$GROUPED_VQC_TY" ]]; then
  for candidate in /root/.vscode-server/extensions/astral-sh.ty-*/bundled/libs/bin/ty; do
    if [[ -x "$candidate" ]]; then GROUPED_VQC_TY="$candidate"; fi
  done
fi
"$VQC_PYTHON" -c 'from rich.console import Console; Console().print("[cyan]Grouped VQC: Ruff, ty, Pylint, pytest (including CUDA)[/cyan]")'
"$GROUPED_VQC_RUFF" check experiments/grouped_dynamic_vqc
"$GROUPED_VQC_RUFF" format --check experiments/grouped_dynamic_vqc
"$GROUPED_VQC_TY" check --python "$VQC_PYTHON" experiments/grouped_dynamic_vqc
bash scripts/run_pylint_quality.sh experiments/grouped_dynamic_vqc
"$VQC_PYTHON" -m pytest experiments/grouped_dynamic_vqc/tests -q --disable-warnings
