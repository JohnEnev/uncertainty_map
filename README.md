# Jev uncertainty map

Does Jev's soft "same meaning?" probability make semantic entropy better at predicting wrong answers than hard clusters or token log-probs?

## Run

```bash
# 1. Serve a model (any OpenAI-compatible endpoint works)
vllm serve Qwen/Qwen3-4B --port 8000

# 2. Sample: greedy answer + 10 samples per question
uv run sample.py --questions 20          # smoke test first
uv run sample.py --questions 200

# 3. Jev pairs: check the response shape once, then run
export TYPESAFE_API_KEY=...
uv run jev_pairs.py --probe
uv run jev_pairs.py

# 4. Scores + AUROC table
uv run score.py
```

All outputs land in `data/`. Steps 2 and 3 are resumable.

## Notes

- Correctness uses TriviaQA alias matching, never Jev.
- 200 questions x 45 pairs is at most 9,000 Jev calls; identical answers skip the call.
- `data/scores.jsonl` has per-question scores for the demo.
- The paper's NLI baseline (hard clusters from a DeBERTa NLI model) is the next row to add.
