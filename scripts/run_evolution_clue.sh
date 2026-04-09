#!/usr/bin/env bash

export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="empty"

set -euo pipefail

# Linux-only example launcher for MemEvolve-style prompt tournament evolution.
# Fill in paths/models before running on remote server.

export PYTHONPATH="$PWD/MemEvolve/Flash-Searcher-main${PYTHONPATH:+:$PYTHONPATH}"

SPLIT_FILE="data_collection/train/global_split_per20.json"

ANALYSIS_MODEL="Qwen3-32B"
GENERATION_MODEL="Qwen3-32B"
EXTRACTION_MODEL="openai/Qwen3-32B"
INFERENCE_MODEL="openai/Qwen3-32B"

WORK_DIR="evolution_runs/clue/simple"
LOG_DIR="${WORK_DIR}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"
echo "Logging to: ${LOG_FILE}"

python -m src.evolution_clue.run_evolution_clue \
  --split_file "${SPLIT_FILE}" \
  --num_rounds 5 \
  --num_systems 3 \
  --task_batch_x 35 \
  --top_t 2 \
  --extra_sample_y 10 \
  --analysis_model "${ANALYSIS_MODEL}" \
  --generation_model "${GENERATION_MODEL}" \
  --extraction_model "${EXTRACTION_MODEL}" \
  --inference_model "${INFERENCE_MODEL}" \
  --work_dir "${WORK_DIR}" \
  --extraction_use_async --extraction_async_max_concurrency 12 \
  --resume \
  --prompt_type simple \
  2>&1 | tee "${LOG_FILE}"

# For faster extraction-only parallelism (inference remains serial), you can use:
#   --extraction_num_workers 4
# or
#   --extraction_use_async --extraction_async_max_concurrency 8

python -m src.pipelines.run_gepa_evaluation \
  --split_file ${SPLIT_FILE} \
  --output_dir evolution_runs/evaluation \
  --model_name openai/Qwen3-32B \
  --max_targets_per_source 1 \
  --num_sources_per_dataset 200 \
  --run_times 3 \
  --use_async \
  --async_max_concurrency 4 \
  --prompt_type memevolve \
  --evolution_run_dir evolution_runs/clue/simple \
  --resume

