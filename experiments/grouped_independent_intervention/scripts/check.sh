#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../grouped_dynamic_vqc/scripts/environment.sh"
INDEPENDENT_INTERVENTION_RUFF=${INDEPENDENT_INTERVENTION_RUFF:-}
INDEPENDENT_INTERVENTION_TY=${INDEPENDENT_INTERVENTION_TY:-}
if [[ -z "$INDEPENDENT_INTERVENTION_RUFF" ]]; then
  for candidate in /root/.vscode-server/extensions/charliermarsh.ruff-*/bundled/libs/bin/ruff; do
    if [[ -x "$candidate" ]]; then INDEPENDENT_INTERVENTION_RUFF="$candidate"; fi
  done
fi
if [[ -z "$INDEPENDENT_INTERVENTION_TY" ]]; then
  for candidate in /root/.vscode-server/extensions/astral-sh.ty-*/bundled/libs/bin/ty; do
    if [[ -x "$candidate" ]]; then INDEPENDENT_INTERVENTION_TY="$candidate"; fi
  done
fi
"$VQC_PYTHON" -c 'from rich.console import Console; Console().print("[cyan]Independent training and interventions: Ruff, ty, Pylint, pytest including CUDA[/cyan]")'
"$INDEPENDENT_INTERVENTION_RUFF" check experiments/grouped_independent_intervention
"$INDEPENDENT_INTERVENTION_RUFF" format --check experiments/grouped_independent_intervention
"$INDEPENDENT_INTERVENTION_TY" check --python "$VQC_PYTHON" experiments/grouped_independent_intervention
bash scripts/run_pylint_quality.sh experiments/grouped_independent_intervention
"$VQC_PYTHON" -m pytest experiments/grouped_independent_intervention/tests -q --disable-warnings
