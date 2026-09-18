#!/usr/bin/env bash
# Runs the rest of the pipeline unattended, once base and sft_only are
# trained and a smoke check (adbench.evaluation.diagnose) shows clean
# <tool_call> output:
#
#   1. train the distilled student (needs the 14B teacher, so it is slowest)
#   2. evaluate all three conditions (harness + perplexity) -> results/eval_*
#   3. layer analysis: teacher, then each student condition, each in its own
#      fresh process (Unsloth must not be imported when the student loads),
#      then compare -> results/layer_analysis/divergence.jsonl
#
# Start it detached so it survives a closed browser tab, and follow run.log:
#
#   subprocess.Popen(["bash", "scripts/finish_run.sh"], stdout=open("run.log", "w"),
#                    stderr=subprocess.STDOUT, start_new_session=True,
#                    env={**os.environ, "PYTHON": sys.executable})
#
# Stops at the first failing step (set -e); "ALL DONE" is printed only when
# every step succeeded.

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
PY="${PYTHON:-python}"

step() { echo; echo "===== $(date '+%H:%M:%S')  $* ====="; }

step "ensure dependencies"
$PY -c "import unsloth" >/dev/null 2>&1 || $PY -m pip install -q -r requirements-colab.txt

step "preflight: base perplexity (expect ~18.3) before spending an hour on eval"
$PY - <<'EOF' 2>&1 | grep -v -i "warn\|Loading weights"
import gc
import torch
from adbench.evaluation.run_eval import load_condition_model, run_perplexity_for_condition
from adbench.training.train import load_experiment_config

experiment_config = load_experiment_config("configs/experiment.yaml")
models_config = load_experiment_config("configs/models.yaml")
model, tokenizer = load_condition_model("base", experiment_config, models_config)
print("base perplexity:", run_perplexity_for_condition(model, tokenizer, experiment_config))
del model
gc.collect()
torch.cuda.empty_cache()
EOF

step "train distilled"
$PY -m adbench.training.train --condition distilled

step "evaluate base, sft_only, distilled"
$PY -m adbench.evaluation.run_eval

step "layer analysis: teacher"
$PY -m adbench.analysis.layer_analysis --role teacher
for condition in base sft_only distilled; do
    step "layer analysis: student $condition"
    $PY -m adbench.analysis.layer_analysis --role student --condition "$condition"
done

step "layer analysis: compare"
$PY -m adbench.analysis.layer_analysis --role compare

step "ALL DONE"
ls -la results results/layer_analysis
