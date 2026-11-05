# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scikit-learn", "typesafe-sdk", "tqdm"]
# ///
"""Extra analyses on top of score.py output."""
import argparse, json
import numpy as np
from sklearn.metrics import roc_auc_score
from score import se_hard, se_soft, sim_matrix

p = argparse.ArgumentParser()
p.add_argument("--samples", default="data/samples.jsonl")
p.add_argument("--pairs", default="data/pairs.jsonl")
p.add_argument("--scores", default="data/scores.jsonl")
args = p.parse_args()

scores = [json.loads(l) for l in open(args.scores)]
samples = {r["id"]: r for r in map(json.loads, open(args.samples))}
pairs = {r["id"]: r["pairs"] for r in map(json.loads, open(args.pairs))}
y = np.array([r["wrong"] for r in scores])
keys = ["neg_mean_logprob", "lexical_entropy", "se_hard_jev", "se_soft_jev"]
X = {k: np.array([r[k] for r in scores]) for k in keys}

# 1. Paired bootstrap on AUROC differences (same resampled questions for both scores)
rng = np.random.default_rng(0)
diffs = {k: [] for k in keys if k != "se_soft_jev"}
for _ in range(2000):
    idx = rng.integers(0, len(y), len(y))
    if y[idx].min() == y[idx].max():
        continue
    a = roc_auc_score(y[idx], X["se_soft_jev"][idx])
    for k in diffs:
        diffs[k].append(a - roc_auc_score(y[idx], X[k][idx]))
print("Soft Jev minus other score, AUROC difference (paired bootstrap, 95% CI)")
for k, v in diffs.items():
    lo, hi = np.percentile(v, [2.5, 97.5])
    print(f"  vs {k:<17} {np.mean(v):+.3f}  [{lo:+.3f}, {hi:+.3f}]")

# 2. AUROC when using only the first k samples (no new Jev calls needed)
print("\nAUROC using only the first k samples")
print("   k   hard   soft")
for k in (3, 4, 6, 8, 10):
    h, s = [], []
    for r in scores:
        n = len(samples[r["id"]]["samples"])
        S = sim_matrix(n, pairs[r["id"]])[:k, :k]
        h.append(se_hard(S))
        s.append(se_soft(S))
    print(f"  {k:>2}  {roc_auc_score(y, h):.3f}  {roc_auc_score(y, s):.3f}")

# 3. Confidently wrong: all samples agree, but the greedy answer is wrong
conf = X["se_hard_jev"] < 1e-9
print(f"\nQuestions where all samples agree (hard entropy 0): {int(conf.sum())}")
if conf.sum():
    print(f"  of those, wrong: {int(y[conf].sum())} ({y[conf].mean():.0%})")
print(f"  share of all wrong answers that look confident: {conf[y == 1].mean():.0%}")
