#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../grouped_dynamic_vqc/scripts/environment.sh"
CONTROL_DIAGNOSTICS_RUFF=${CONTROL_DIAGNOSTICS_RUFF:-}
CONTROL_DIAGNOSTICS_TY=${CONTROL_DIAGNOSTICS_TY:-}
if [[ -z "$CONTROL_DIAGNOSTICS_RUFF" ]]; then
  for candidate in /root/.vscode-server/extensions/charliermarsh.ruff-*/bundled/libs/bin/ruff; do
    if [[ -x "$candidate" ]]; then CONTROL_DIAGNOSTICS_RUFF="$candidate"; fi
  done
fi
if [[ -z "$CONTROL_DIAGNOSTICS_TY" ]]; then
  for candidate in /root/.vscode-server/extensions/astral-sh.ty-*/bundled/libs/bin/ty; do
    if [[ -x "$candidate" ]]; then CONTROL_DIAGNOSTICS_TY="$candidate"; fi
  done
fi
"$VQC_PYTHON" -c 'from rich.console import Console; Console().print("[cyan]VQC control diagnostics: Ruff, ty, Pylint, pytest including CUDA[/cyan]")'
"$CONTROL_DIAGNOSTICS_RUFF" check experiments/grouped_control_diagnostics
"$CONTROL_DIAGNOSTICS_RUFF" format --check experiments/grouped_control_diagnostics
"$CONTROL_DIAGNOSTICS_TY" check --python "$VQC_PYTHON" experiments/grouped_control_diagnostics
bash scripts/run_pylint_quality.sh experiments/grouped_control_diagnostics
"$VQC_PYTHON" -m pytest experiments/grouped_control_diagnostics/tests -q --disable-warnings
