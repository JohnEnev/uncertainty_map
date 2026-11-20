#!/usr/bin/env bash
# Judge labels + rescored tables for each cell. Needs ANTHROPIC_API_KEY.
# Usage: ./run_judge.sh [tag ...]   (default: all seven cells)
set -euo pipefail
tags=("$@")
if [ ${#tags[@]} -eq 0 ]; then
  tags=(triviaqa_qwen3-4b triviaqa_qwen3-14b triviaqa_gemma-3-12b-it popqa_qwen3-4b popqa_qwen3-14b nq_open_qwen3-4b nq_open_qwen3-14b)
fi
mkdir -p results
for tag in "${tags[@]}"; do
  uv run judge.py --samples "data/${tag}_samples.jsonl" --out "data/${tag}_judge.jsonl"
  uv run score_judged.py --scores "data/${tag}_scores_all.jsonl" --judge "data/${tag}_judge.jsonl" \
    | tee "results/${tag}_score_table_judged.txt"
done
