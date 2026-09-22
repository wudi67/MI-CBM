#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../grouped_dynamic_vqc/scripts/environment.sh"
GROUPED_ROBOT_PILOT_RUFF=${GROUPED_ROBOT_PILOT_RUFF:-}
GROUPED_ROBOT_PILOT_TY=${GROUPED_ROBOT_PILOT_TY:-}
if [[ -z "$GROUPED_ROBOT_PILOT_RUFF" ]]; then
  for candidate in /root/.vscode-server/extensions/charliermarsh.ruff-*/bundled/libs/bin/ruff; do
    if [[ -x "$candidate" ]]; then GROUPED_ROBOT_PILOT_RUFF="$candidate"; fi
  done
fi
if [[ -z "$GROUPED_ROBOT_PILOT_TY" ]]; then
  for candidate in /root/.vscode-server/extensions/astral-sh.ty-*/bundled/libs/bin/ty; do
    if [[ -x "$candidate" ]]; then GROUPED_ROBOT_PILOT_TY="$candidate"; fi
  done
fi
"$VQC_PYTHON" -c 'from rich.console import Console; Console().print("[cyan]Robot P0 pilot: Ruff, ty, Pylint, pytest including CUDA[/cyan]")'
"$GROUPED_ROBOT_PILOT_RUFF" check experiments/grouped_robot_pilot
"$GROUPED_ROBOT_PILOT_RUFF" format --check experiments/grouped_robot_pilot
"$GROUPED_ROBOT_PILOT_TY" check --python "$VQC_PYTHON" experiments/grouped_robot_pilot
bash scripts/run_pylint_quality.sh experiments/grouped_robot_pilot
"$VQC_PYTHON" -m pytest experiments/grouped_robot_pilot/tests -q --disable-warnings
