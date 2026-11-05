# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scikit-learn", "typesafe-sdk", "tqdm"]
# ///
"""Does soft beat hard when only k samples are used? Random subsets, paired bootstrap."""
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
mats = [sim_matrix(len(samples[r["id"]]["samples"]), pairs[r["id"]]) for r in scores]
logp = np.array([r["neg_mean_logprob"] for r in scores])
rng = np.random.default_rng(0)
R, B = 30, 2000

print(f"token log-prob baseline (all samples not needed): {roc_auc_score(y, logp):.3f}\n")
print("  k   hard   soft   soft-hard  95% CI (questions + subsets)")
for k in (3, 4, 5, 6, 8, 12, 16):
    H = np.zeros((R, len(y)))
    Sf = np.zeros((R, len(y)))
    for d in range(R):
        for q, S in enumerate(mats):
            sub = rng.choice(len(S), k, replace=False)
            M = S[np.ix_(sub, sub)]
            H[d, q] = se_hard(M)
            Sf[d, q] = se_soft(M)
    ah = np.mean([roc_auc_score(y, H[d]) for d in range(R)])
    asf = np.mean([roc_auc_score(y, Sf[d]) for d in range(R)])
    diffs = []
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        d = rng.integers(0, R)
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], Sf[d, idx]) - roc_auc_score(y[idx], H[d, idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  {k}  {ah:.3f}  {asf:.3f}   {asf - ah:+.3f}    [{lo:+.3f}, {hi:+.3f}]")
