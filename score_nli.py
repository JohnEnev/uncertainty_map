# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scikit-learn", "typesafe-sdk", "tqdm"]
# ///
"""Adds the NLI rows to the AUROC table and runs paired comparisons.

Reads data/scores.jsonl (labels, log-prob, lexical, Jev scores from score.py) and
data/pairs_nli.jsonl (from nli_pairs.py). Writes data/scores_all.jsonl.
"""
import argparse, json
import numpy as np
from sklearn.metrics import roc_auc_score
from score import auroc_ci, se_hard, se_soft, sim_matrix

p = argparse.ArgumentParser()
p.add_argument("--samples", default="data/samples.jsonl")
p.add_argument("--scores", default="data/scores.jsonl")
p.add_argument("--nli", default="data/pairs_nli.jsonl")
p.add_argument("--out", default="data/scores_all.jsonl")
args = p.parse_args()

samples = {r["id"]: r for r in map(json.loads, open(args.samples))}
nli = {r["id"]: r for r in map(json.loads, open(args.nli))}
rows = []
for r in map(json.loads, open(args.scores)):
    n = len(samples[r["id"]]["samples"])
    r["se_hard_nli"] = se_hard(sim_matrix(n, nli[r["id"]]["hard"]))
    r["se_soft_nli"] = se_soft(sim_matrix(n, nli[r["id"]]["pairs"]))
    rows.append(r)

with open(args.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

keys = ["neg_mean_logprob", "lexical_entropy", "se_hard_nli", "se_soft_nli", "se_hard_jev", "se_soft_jev"]
y = np.array([r["wrong"] for r in rows])
X = {k: np.array([r[k] for r in rows]) for k in keys}
print(f"{len(rows)} questions, {y.sum()} wrong ({y.mean():.0%})\n")
print(f"{'score':<18} {'AUROC':>6}   95% CI")
for k in keys:
    a, lo, hi = auroc_ci(y, X[k])
    print(f"{k:<18} {a:6.3f}   [{lo:.3f}, {hi:.3f}]")

comparisons = [
    ("se_hard_jev", "se_hard_nli"),   # same rule, different judge
    ("se_soft_nli", "se_hard_nli"),   # does keeping probabilities help with NLI too?
    ("se_soft_jev", "se_soft_nli"),   # same soft rule, different judge
    ("se_soft_jev", "se_hard_nli"),   # new method vs the paper's method
]
rng = np.random.default_rng(0)
print("\nPaired AUROC differences (first minus second), 95% bootstrap CI")
for a_key, b_key in comparisons:
    diffs = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], X[a_key][idx]) - roc_auc_score(y[idx], X[b_key][idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  {a_key} - {b_key:<12} {np.mean(diffs):+.3f}  [{lo:+.3f}, {hi:+.3f}]")
