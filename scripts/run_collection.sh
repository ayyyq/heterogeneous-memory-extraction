#!/usr/bin/env bash

export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="empty"

set -euo pipefail

MODEL_NAME="openai/Qwen3-32B"
OUT_DIR="data_collection/train"
MAX_JOBS=8

DATASETS=("alfworld" "fever" "pddl" "babyai" "scienceworld" "toolbench" "GameOf24" "AIME_2024" "AIME_2025" "AIME_2020_2024" "GPQA_Diamond" "MMLU_Pro_Physics" "MMLU_Pro_Engineering" "MathEquationBalancer" "bigcodebench")

run_one () {
  local d="$1"
  python -m src.pipelines.run_collection \
    --dataset_name "$d" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUT_DIR" \
    --num_workers 1 \
    --refresh
}

for d in "${DATASETS[@]}"; do
  run_one "$d" &
  while [ "$(jobs -pr | wc -l)" -ge "$MAX_JOBS" ]; do
    sleep 2
  done
done
wait