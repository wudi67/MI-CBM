#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../grouped_dynamic_vqc/scripts/environment.sh"
GROUPED_ROBOT_MLP_DIAGNOSTIC_RUFF=${GROUPED_ROBOT_MLP_DIAGNOSTIC_RUFF:-}
GROUPED_ROBOT_MLP_DIAGNOSTIC_TY=${GROUPED_ROBOT_MLP_DIAGNOSTIC_TY:-}
if [[ -z "$GROUPED_ROBOT_MLP_DIAGNOSTIC_RUFF" ]]; then
  for candidate in /root/.vscode-server/extensions/charliermarsh.ruff-*/bundled/libs/bin/ruff; do
    if [[ -x "$candidate" ]]; then GROUPED_ROBOT_MLP_DIAGNOSTIC_RUFF="$candidate"; fi
  done
fi
if [[ -z "$GROUPED_ROBOT_MLP_DIAGNOSTIC_TY" ]]; then
  for candidate in /root/.vscode-server/extensions/astral-sh.ty-*/bundled/libs/bin/ty; do
    if [[ -x "$candidate" ]]; then GROUPED_ROBOT_MLP_DIAGNOSTIC_TY="$candidate"; fi
  done
fi
"$VQC_PYTHON" -c 'from rich.console import Console; Console().print("[cyan]Robot concept MLP diagnostic: Ruff, ty, Pylint, pytest including CUDA[/cyan]")'
"$GROUPED_ROBOT_MLP_DIAGNOSTIC_RUFF" check experiments/grouped_robot_mlp_diagnostic
"$GROUPED_ROBOT_MLP_DIAGNOSTIC_RUFF" format --check experiments/grouped_robot_mlp_diagnostic
"$GROUPED_ROBOT_MLP_DIAGNOSTIC_TY" check --python "$VQC_PYTHON" experiments/grouped_robot_mlp_diagnostic
bash scripts/run_pylint_quality.sh experiments/grouped_robot_mlp_diagnostic
"$VQC_PYTHON" -m pytest experiments/grouped_robot_mlp_diagnostic/tests -q --disable-warnings
