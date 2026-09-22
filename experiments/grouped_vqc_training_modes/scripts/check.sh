#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../grouped_dynamic_vqc/scripts/environment.sh"
GROUPED_MODES_RUFF=${GROUPED_MODES_RUFF:-}
GROUPED_MODES_TY=${GROUPED_MODES_TY:-}
if [[ -z "$GROUPED_MODES_RUFF" ]]; then
  for candidate in /root/.vscode-server/extensions/charliermarsh.ruff-*/bundled/libs/bin/ruff; do
    if [[ -x "$candidate" ]]; then GROUPED_MODES_RUFF="$candidate"; fi
  done
fi
if [[ -z "$GROUPED_MODES_TY" ]]; then
  for candidate in /root/.vscode-server/extensions/astral-sh.ty-*/bundled/libs/bin/ty; do
    if [[ -x "$candidate" ]]; then GROUPED_MODES_TY="$candidate"; fi
  done
fi
"$VQC_PYTHON" -c 'from rich.console import Console; Console().print("[cyan]Training modes: Ruff, ty, Pylint, pytest including CUDA[/cyan]")'
"$GROUPED_MODES_RUFF" check experiments/grouped_vqc_training_modes
"$GROUPED_MODES_RUFF" format --check experiments/grouped_vqc_training_modes
"$GROUPED_MODES_TY" check --python "$VQC_PYTHON" experiments/grouped_vqc_training_modes
bash scripts/run_pylint_quality.sh experiments/grouped_vqc_training_modes
"$VQC_PYTHON" -m pytest experiments/grouped_vqc_training_modes/tests experiments/grouped_dynamic_vqc/tests -q --disable-warnings
