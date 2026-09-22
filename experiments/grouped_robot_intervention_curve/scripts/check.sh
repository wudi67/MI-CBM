#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../grouped_dynamic_vqc/scripts/environment.sh"
GROUPED_ROBOT_INTERVENTION_CURVE_RUFF=${GROUPED_ROBOT_INTERVENTION_CURVE_RUFF:-}
GROUPED_ROBOT_INTERVENTION_CURVE_TY=${GROUPED_ROBOT_INTERVENTION_CURVE_TY:-}
if [[ -z "$GROUPED_ROBOT_INTERVENTION_CURVE_RUFF" ]]; then
  for candidate in /root/.vscode-server/extensions/charliermarsh.ruff-*/bundled/libs/bin/ruff; do
    if [[ -x "$candidate" ]]; then GROUPED_ROBOT_INTERVENTION_CURVE_RUFF="$candidate"; fi
  done
fi
if [[ -z "$GROUPED_ROBOT_INTERVENTION_CURVE_TY" ]]; then
  for candidate in /root/.vscode-server/extensions/astral-sh.ty-*/bundled/libs/bin/ty; do
    if [[ -x "$candidate" ]]; then GROUPED_ROBOT_INTERVENTION_CURVE_TY="$candidate"; fi
  done
fi
"$VQC_PYTHON" -c 'from rich.console import Console; Console().print("[cyan]Robot correction count curves: Ruff, ty, Pylint, pytest including CUDA[/cyan]")'
"$GROUPED_ROBOT_INTERVENTION_CURVE_RUFF" check experiments/grouped_robot_intervention_curve
"$GROUPED_ROBOT_INTERVENTION_CURVE_RUFF" format --check experiments/grouped_robot_intervention_curve
"$GROUPED_ROBOT_INTERVENTION_CURVE_TY" check --python "$VQC_PYTHON" experiments/grouped_robot_intervention_curve
bash scripts/run_pylint_quality.sh experiments/grouped_robot_intervention_curve
"$VQC_PYTHON" -m pytest experiments/grouped_robot_intervention_curve/tests -q --disable-warnings
