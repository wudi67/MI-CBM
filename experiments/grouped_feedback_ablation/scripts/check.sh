#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../grouped_dynamic_vqc/scripts/environment.sh"
FEEDBACK_ABLATION_RUFF=${FEEDBACK_ABLATION_RUFF:-}
FEEDBACK_ABLATION_TY=${FEEDBACK_ABLATION_TY:-}
if [[ -z "$FEEDBACK_ABLATION_RUFF" ]]; then
  for candidate in /root/.vscode-server/extensions/charliermarsh.ruff-*/bundled/libs/bin/ruff; do
    if [[ -x "$candidate" ]]; then FEEDBACK_ABLATION_RUFF="$candidate"; fi
  done
fi
if [[ -z "$FEEDBACK_ABLATION_TY" ]]; then
  for candidate in /root/.vscode-server/extensions/astral-sh.ty-*/bundled/libs/bin/ty; do
    if [[ -x "$candidate" ]]; then FEEDBACK_ABLATION_TY="$candidate"; fi
  done
fi
"$VQC_PYTHON" -c 'from rich.console import Console; Console().print("[cyan]Paired feedback ablation: Ruff, ty, Pylint, pytest including CUDA[/cyan]")'
"$FEEDBACK_ABLATION_RUFF" check experiments/grouped_feedback_ablation
"$FEEDBACK_ABLATION_RUFF" format --check experiments/grouped_feedback_ablation
"$FEEDBACK_ABLATION_TY" check --python "$VQC_PYTHON" experiments/grouped_feedback_ablation
bash scripts/run_pylint_quality.sh experiments/grouped_feedback_ablation
"$VQC_PYTHON" -m pytest experiments/grouped_feedback_ablation/tests -q --disable-warnings
