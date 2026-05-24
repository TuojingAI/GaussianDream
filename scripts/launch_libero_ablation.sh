#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <preset> [extra run_libero_ablation.py args...]"
  echo "Example: GPUS=0,1,2,3,4,5,6,7 $0 no_flow --overwrite"
  exit 2
fi

PRESET="$1"
shift

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-$(echo "$GPUS" | tr ',' '\n' | wc -l)}"
BUDGET="${BUDGET:-paper_30k}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$PROJECT_ROOT/checkpoints}"
EXP_NAME="${EXP_NAME:-gaussiandream_${PRESET}}"
LOG_DIR="${LOG_DIR:-$CHECKPOINT_BASE_DIR/logs/libero_ablation}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${EXP_NAME}.log}"

mkdir -p "$LOG_DIR"

export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,expandable_segments:True}"

echo "[launch_libero_ablation] preset=$PRESET exp=$EXP_NAME gpus=$GPUS nproc=$NPROC budget=$BUDGET"
echo "[launch_libero_ablation] log=$LOG_FILE"

CUDA_VISIBLE_DEVICES="$GPUS" nohup uv_venv/bin/torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC" \
  scripts/run_libero_ablation.py \
  --preset "$PRESET" \
  --budget "$BUDGET" \
  --exp-name "$EXP_NAME" \
  --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" \
  "$@" > "$LOG_FILE" 2>&1 &

PID=$!
echo "[launch_libero_ablation] pid=$PID"
echo "[launch_libero_ablation] tail -f $LOG_FILE"
