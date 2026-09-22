#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../grouped_dynamic_vqc/scripts/environment.sh"
INDEPENDENT_FEEDBACK_RUFF=${INDEPENDENT_FEEDBACK_RUFF:-}
INDEPENDENT_FEEDBACK_TY=${INDEPENDENT_FEEDBACK_TY:-}
if [[ -z "$INDEPENDENT_FEEDBACK_RUFF" ]]; then
  for candidate in /root/.vscode-server/extensions/charliermarsh.ruff-*/bundled/libs/bin/ruff; do
    if [[ -x "$candidate" ]]; then INDEPENDENT_FEEDBACK_RUFF="$candidate"; fi
  done
fi
if [[ -z "$INDEPENDENT_FEEDBACK_TY" ]]; then
  for candidate in /root/.vscode-server/extensions/astral-sh.ty-*/bundled/libs/bin/ty; do
    if [[ -x "$candidate" ]]; then INDEPENDENT_FEEDBACK_TY="$candidate"; fi
  done
fi
"$VQC_PYTHON" -c 'from rich.console import Console; Console().print("[cyan]Independent feedback ablation: Ruff, ty, Pylint, pytest including CUDA[/cyan]")'
"$INDEPENDENT_FEEDBACK_RUFF" check experiments/grouped_independent_feedback_ablation
"$INDEPENDENT_FEEDBACK_RUFF" format --check experiments/grouped_independent_feedback_ablation
"$INDEPENDENT_FEEDBACK_TY" check --python "$VQC_PYTHON" experiments/grouped_independent_feedback_ablation
bash scripts/run_pylint_quality.sh experiments/grouped_independent_feedback_ablation
"$VQC_PYTHON" -m pytest experiments/grouped_independent_feedback_ablation/tests -q --disable-warnings
