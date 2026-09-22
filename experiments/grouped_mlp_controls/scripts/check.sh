#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../grouped_dynamic_vqc/scripts/environment.sh"
GROUPED_MLP_RUFF=${GROUPED_MLP_RUFF:-}
GROUPED_MLP_TY=${GROUPED_MLP_TY:-}
if [[ -z "$GROUPED_MLP_RUFF" ]]; then
  for candidate in /root/.vscode-server/extensions/charliermarsh.ruff-*/bundled/libs/bin/ruff; do
    if [[ -x "$candidate" ]]; then GROUPED_MLP_RUFF="$candidate"; fi
  done
fi
if [[ -z "$GROUPED_MLP_TY" ]]; then
  for candidate in /root/.vscode-server/extensions/astral-sh.ty-*/bundled/libs/bin/ty; do
    if [[ -x "$candidate" ]]; then GROUPED_MLP_TY="$candidate"; fi
  done
fi
"$VQC_PYTHON" -c 'from rich.console import Console; Console().print("[cyan]MLP controls: Ruff, ty, Pylint, pytest including CUDA[/cyan]")'
"$GROUPED_MLP_RUFF" check experiments/grouped_mlp_controls
"$GROUPED_MLP_RUFF" format --check experiments/grouped_mlp_controls
"$GROUPED_MLP_TY" check --python "$VQC_PYTHON" experiments/grouped_mlp_controls
bash scripts/run_pylint_quality.sh experiments/grouped_mlp_controls
"$VQC_PYTHON" -m pytest experiments/grouped_mlp_controls/tests experiments/grouped_dynamic_vqc/tests/test_data.py -q --disable-warnings
