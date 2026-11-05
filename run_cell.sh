#!/usr/bin/env bash
# One cell of the grid: sample -> Jev pairs -> score.
# Usage: ./run_cell.sh <dataset> <model> <n_questions>
#   e.g. ./run_cell.sh nq_open Qwen/Qwen3-4B 1000
# Needs a vLLM server for <model> on localhost:8000 and TYPESAFE_API_KEY set.
# Add --chat-kwargs '{}' to the sample.py line below for models without a thinking switch.
set -euo pipefail
ds=$1; model=$2; n=${3:-1000}
tag="${ds}_$(basename "$model" | tr 'A-Z' 'a-z')"
mkdir -p data results
uv run sample.py --dataset "$ds" --model "$model" --questions "$n" --out "data/${tag}_samples.jsonl"
uv run jev_pairs.py --samples "data/${tag}_samples.jsonl" --out "data/${tag}_pairs.jsonl" --workers 16
uv run score.py --samples "data/${tag}_samples.jsonl" --pairs "data/${tag}_pairs.jsonl" \
  --out "data/${tag}_scores.jsonl" | tee "results/${tag}_score_table.txt"
