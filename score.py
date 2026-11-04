# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scikit-learn", "typesafe-sdk", "tqdm"]
# ///
"""Step 3: uncertainty scores and AUROC for predicting a wrong answer.

Correctness comes from TriviaQA alias matching, not from Jev, so the label and the
score are never judged by the same model.

Scores (higher = more uncertain):
  neg_mean_logprob  baseline, free
  lexical_entropy   entropy over distinct normalized strings, no model
  se_hard_jev       Jev P>=0.5 -> greedy clusters -> entropy of cluster sizes
  se_soft_jev       mass_i = mean_j P(i,j); H = -mean_i log(mass_i)
                    (0 when all samples agree, log n when all differ)
"""
import argparse, json, math
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score
from jev_pairs import norm


def correct(pred, aliases, question=""):
    p = f" {norm(pred)} "
    q = f" {norm(question)} "
    return any(
        a and (norm(pred) == a or (f" {a} " in p and f" {a} " not in q))
        for a in aliases
    )


def sim_matrix(n, pairs):
    S = np.eye(n)
    for k, v in pairs.items():
        i, j = map(int, k.split(","))
        S[i, j] = S[j, i] = v
    return S


def entropy(counts):
    tot = sum(counts)
    return -sum(c / tot * math.log(c / tot) for c in counts if c)


def se_hard(S, thr=0.5):
    reps, sizes = [], []
    for i in range(len(S)):
        for c, r in enumerate(reps):
            if S[i, r] >= thr:
                sizes[c] += 1
                break
        else:
            reps.append(i)
            sizes.append(1)
    return entropy(sizes)


def se_soft(S):
    mass = S.mean(axis=1)
    return float(-np.mean(np.log(mass)))


def auroc_ci(y, s, n_boot=1000, seed=0):
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            boots.append(roc_auc_score(y[idx], s[idx]))
    return roc_auc_score(y, s), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="data/samples.jsonl")
    p.add_argument("--pairs", default="data/pairs.jsonl")
    p.add_argument("--out", default="data/scores.jsonl")
    args = p.parse_args()

    samples = {r["id"]: r for r in map(json.loads, open(args.samples))}
    pairs = {r["id"]: r["pairs"] for r in map(json.loads, open(args.pairs))}

    rows = []
    for qid, pr in pairs.items():
        r = samples[qid]
        S = sim_matrix(len(r["samples"]), pr)
        rows.append({
            "id": qid,
            "question": r["question"],
            "greedy": r["greedy"],
            "wrong": int(not correct(r["greedy"], r["aliases"], r["question"])),
            "neg_mean_logprob": -(r["mean_logprob"] or 0.0),
            "lexical_entropy": entropy(list(Counter(norm(x) for x in r["samples"]).values())),
            "se_hard_jev": se_hard(S),
            "se_soft_jev": se_soft(S),
        })

    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    y = [r["wrong"] for r in rows]
    print(f"{len(rows)} questions, {sum(y)} wrong ({sum(y) / len(y):.0%})\n")
    print(f"{'score':<18} {'AUROC':>6}   95% CI")
    for k in ["neg_mean_logprob", "lexical_entropy", "se_hard_jev", "se_soft_jev"]:
        a, lo, hi = auroc_ci(y, [r[k] for r in rows])
        print(f"{k:<18} {a:6.3f}   [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
