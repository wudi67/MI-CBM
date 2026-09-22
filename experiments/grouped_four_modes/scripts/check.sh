#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../grouped_dynamic_vqc/scripts/environment.sh"
GROUPED_FOUR_MODES_RUFF=${GROUPED_FOUR_MODES_RUFF:-}
GROUPED_FOUR_MODES_TY=${GROUPED_FOUR_MODES_TY:-}
if [[ -z "$GROUPED_FOUR_MODES_RUFF" ]]; then
  for candidate in /root/.vscode-server/extensions/charliermarsh.ruff-*/bundled/libs/bin/ruff; do
    if [[ -x "$candidate" ]]; then GROUPED_FOUR_MODES_RUFF="$candidate"; fi
  done
fi
if [[ -z "$GROUPED_FOUR_MODES_TY" ]]; then
  for candidate in /root/.vscode-server/extensions/astral-sh.ty-*/bundled/libs/bin/ty; do
    if [[ -x "$candidate" ]]; then GROUPED_FOUR_MODES_TY="$candidate"; fi
  done
fi
"$VQC_PYTHON" -c 'from rich.console import Console; Console().print("[cyan]dSprites four modes: Ruff, ty, Pylint, pytest including CUDA[/cyan]")'
"$GROUPED_FOUR_MODES_RUFF" check experiments/grouped_four_modes
"$GROUPED_FOUR_MODES_RUFF" format --check experiments/grouped_four_modes
"$GROUPED_FOUR_MODES_TY" check --python "$VQC_PYTHON" experiments/grouped_four_modes
bash scripts/run_pylint_quality.sh experiments/grouped_four_modes
"$VQC_PYTHON" -m pytest experiments/grouped_four_modes/tests -q --disable-warnings
