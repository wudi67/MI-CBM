#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../grouped_dynamic_vqc/scripts/environment.sh"
GROUPED_SHOTS_FINAL_RUFF=${GROUPED_SHOTS_FINAL_RUFF:-}
GROUPED_SHOTS_FINAL_TY=${GROUPED_SHOTS_FINAL_TY:-}
if [[ -z "$GROUPED_SHOTS_FINAL_RUFF" ]]; then
  for candidate in /root/.vscode-server/extensions/charliermarsh.ruff-*/bundled/libs/bin/ruff; do
    if [[ -x "$candidate" ]]; then GROUPED_SHOTS_FINAL_RUFF="$candidate"; fi
  done
fi
if [[ -z "$GROUPED_SHOTS_FINAL_TY" ]]; then
  for candidate in /root/.vscode-server/extensions/astral-sh.ty-*/bundled/libs/bin/ty; do
    if [[ -x "$candidate" ]]; then GROUPED_SHOTS_FINAL_TY="$candidate"; fi
  done
fi
"$VQC_PYTHON" -c 'from rich.console import Console; Console().print("[cyan]Validation shots then final test: Ruff, ty, Pylint, pytest including CUDA[/cyan]")'
"$GROUPED_SHOTS_FINAL_RUFF" check experiments/grouped_shots_final
"$GROUPED_SHOTS_FINAL_RUFF" format --check experiments/grouped_shots_final
"$GROUPED_SHOTS_FINAL_TY" check --python "$VQC_PYTHON" experiments/grouped_shots_final
bash scripts/run_pylint_quality.sh experiments/grouped_shots_final
"$VQC_PYTHON" -m pytest experiments/grouped_shots_final/tests -q --disable-warnings
