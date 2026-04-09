export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="empty"

DATASETS=(
  "membench-high"
  "membench-low"
  "personamem-v2-val"
  "prefeval-mcq"
  "alfworld" "fever" "pddl"
  "babyai" "scienceworld"
  "GameOf24"
  "MMLU_Pro_Physics" "MMLU_Pro_Engineering" "MathEquationBalancer"
  "bigcodebench"
)

PROMPT_TYPE=""

for DATASET in ${DATASETS[@]}; do
    python -m src.pipelines.print_evaluation --mode single \
        evolution_runs/evaluation/openai/Qwen3-32B/${PROMPT_TYPE}/${DATASET}
done

python -m src.pipelines.print_evaluation --mode single \
    evolution_runs/evaluation/openai/Qwen3-32B/${PROMPT_TYPE}/AIME_2024 evolution_runs/evaluation/openai/Qwen3-32B/${PROMPT_TYPE}/AIME_2025 evolution_runs/evaluation/openai/Qwen3-32B/${PROMPT_TYPE}/AIME_2020_2024
