#!/usr/bin/env bash

export OPENAI_API_BASE=""
export OPENAI_API_KEY=""

set -euo pipefail

PROMPTS=("no_memory")
for PROMPT in "${PROMPTS[@]}"; do
  python -m src.pipelines.run_llm_judge_evaluation \
    --dataset_name longmemeval \
    --output_file evolution_runs/evaluation/openai/Qwen3-32B/${PROMPT}/longmemeval/run_1/source_target_output.jsonl \
    --metric_model gpt-4o \
    --quiet
done

for PROMPT in "${PROMPTS[@]}"; do
  python -m src.pipelines.run_llm_judge_evaluation \
    --dataset_name toolbench \
    --output_file evolution_runs/evaluation/openai/Qwen3-32B/${PROMPT}/toolbench/run_1/source_target_output.jsonl \
    --metric_model gpt-4o \
    --concurrency 5 \
    --quiet \
    --evaluate_times 3
done