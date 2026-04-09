#!/usr/bin/env bash

export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="empty"

set -euo pipefail

MODEL_NAME="openai/Qwen3-32B"
OUT_DIR="data_collection/train"

python -m src.pipelines.run_generation \
  --dataset_name membench-high \
  --model_name "${MODEL_NAME}" \
  --output_dir "${OUT_DIR}" \
  --refresh \
  --source_file data/membench/FirstAgentDataHighLevel_multiple_0.json

python -m src.pipelines.run_generation \
  --dataset_name membench-low \
  --model_name "${MODEL_NAME}" \
  --output_dir "${OUT_DIR}" \
  --refresh \
  --source_file data/membench/FirstAgentDataLowLevel_multiple_0.json

python -m src.pipelines.run_generation \
  --dataset_name personamem-v2-val \
  --model_name "${MODEL_NAME}" \
  --output_dir "${OUT_DIR}" \
  --refresh \
  --source_file data/PersonaMem-v2/benchmark/text/val.csv

python -m src.pipelines.run_generation \
  --dataset_name prefeval-mcq \
  --model_name "${MODEL_NAME}" \
  --output_dir "${OUT_DIR}" \
  --refresh \
  --source_file data/PrefEval/implicit_preference/persona-driven

python -m src.pipelines.run_generation \
  --dataset_name longmemeval \
  --model_name "${MODEL_NAME}" \
  --output_dir "${OUT_DIR}" \
  --refresh \
  --source_file data/longmemeval/longmemeval_oracle.json

DATASETS=(
  "alfworld" "fever" "pddl"
  "babyai" "scienceworld"
  "GameOf24" "AIME_2024" "AIME_2025" "AIME_2020_2024" "GPQA_Diamond" "MMLU_Pro_Physics" "MMLU_Pro_Engineering" "MathEquationBalancer"
  "babyai" "scienceworld"
  "bigcodebench"
  "toolbench"
)

for DATASET in ${DATASETS[@]}; do
  python -m src.pipelines.run_generation \
    --dataset_name "${DATASET}" \
    --model_name "${MODEL_NAME}" \
    --output_dir "${OUT_DIR}" \
    --refresh \
    --openai_api_key "${TRUE_OPENAI_API_KEY}" \
    --openai_api_base "${TRUE_OPENAI_API_BASE}"
done