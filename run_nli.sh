#!/usr/bin/env bash
# NLI rows for finished cells. Stop vLLM first so the GPU is free.
# Usage: ./run_nli.sh <tag> [<tag> ...]     e.g. ./run_nli.sh triviaqa_qwen3-4b nq_open_qwen3-4b
set -euo pipefail
for tag in "$@"; do
  uv run nli_pairs.py --samples "data/${tag}_samples.jsonl" --out "data/${tag}_pairs_nli.jsonl"
  uv run score_nli.py --samples "data/${tag}_samples.jsonl" --scores "data/${tag}_scores.jsonl" \
    --nli "data/${tag}_pairs_nli.jsonl" --out "data/${tag}_scores_all.jsonl" \
    | tee "results/${tag}_score_table_all.txt"
done
