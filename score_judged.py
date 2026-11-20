# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scikit-learn", "typesafe-sdk", "tqdm"]
# ///
"""AUROC table and paired comparisons with judge labels instead of alias matching.

Main analysis drops questions the judge called ambiguous; a sensitivity check counts them
as right, then as wrong. Also reports how often the alias matcher agreed with the judge.
"""
import argparse, json
import numpy as np
from sklearn.metrics import roc_auc_score
from score import auroc_ci

KEYS = ["neg_mean_logprob", "lexical_entropy", "se_hard_nli", "se_soft_nli", "se_hard_jev", "se_soft_jev"]
COMPARISONS = [
    ("se_soft_jev", "se_hard_jev"),
    ("se_soft_nli", "se_hard_nli"),
    ("se_soft_jev", "se_soft_nli"),
    ("se_soft_jev", "se_hard_nli"),
    ("se_soft_jev", "neg_mean_logprob"),
    ("se_soft_jev", "lexical_entropy"),
    ("se_soft_nli", "neg_mean_logprob"),
]

p = argparse.ArgumentParser()
p.add_argument("--scores", required=True, help="*_scores_all.jsonl from score_nli.py")
p.add_argument("--judge", required=True, help="*_judge.jsonl from judge.py")
args = p.parse_args()

verdict = {r["id"]: r["verdict"] for r in map(json.loads, open(args.judge))}
rows = [r for r in map(json.loads, open(args.scores)) if verdict.get(r["id"])]
for r in rows:
    r["verdict"] = verdict[r["id"]]
missing = sum(1 for r in map(json.loads, open(args.scores)) if not verdict.get(r["id"]))
counts = {v: sum(r["verdict"] == v for r in rows) for v in ("correct", "incorrect", "ambiguous")}
print(f"judged {len(rows)} questions ({missing} without a usable verdict): {counts}")

clear = [r for r in rows if r["verdict"] != "ambiguous"]
tp = sum(r["wrong"] == 1 and r["verdict"] == "incorrect" for r in clear)
tn = sum(r["wrong"] == 0 and r["verdict"] == "correct" for r in clear)
m_wrong_j_right = sum(r["wrong"] == 1 and r["verdict"] == "correct" for r in clear)
m_right_j_wrong = sum(r["wrong"] == 0 and r["verdict"] == "incorrect" for r in clear)
print(f"matcher vs judge (non-ambiguous): agree {tp + tn}/{len(clear)} "
      f"({(tp + tn) / max(len(clear), 1):.0%}); matcher said wrong but judge correct: {m_wrong_j_right}; "
      f"matcher said right but judge incorrect: {m_right_j_wrong}\n")


def arrays(mode):
    if mode == "drop":
        sel = clear
        y = np.array([r["verdict"] == "incorrect" for r in sel], dtype=int)
    else:
        sel = rows
        amb_wrong = mode == "wrong"
        y = np.array([(r["verdict"] == "incorrect") or (r["verdict"] == "ambiguous" and amb_wrong) for r in sel], dtype=int)
    X = {k: np.array([r[k] for r in sel]) for k in KEYS}
    return y, X


def paired(y, a, b, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        d.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return np.mean(d), lo, hi


y, X = arrays("drop")
print(f"MAIN (ambiguous dropped): {len(y)} questions, {y.sum()} wrong ({y.mean():.0%})\n")
print(f"{'score':<18} {'AUROC':>6}   95% CI")
for k in KEYS:
    a, lo, hi = auroc_ci(y, X[k])
    print(f"{k:<18} {a:6.3f}   [{lo:.3f}, {hi:.3f}]")

print("\nPaired AUROC differences (first minus second), 95% bootstrap CI")
for a_key, b_key in COMPARISONS:
    m, lo, hi = paired(y, X[a_key], X[b_key])
    mark = " *" if lo > 0 or hi < 0 else ""
    print(f"  {a_key} - {b_key:<17} {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]{mark}")

conf = X["se_hard_jev"] < 1e-9
print(f"\nAll samples agree: {int(conf.sum())} questions, {int(y[conf].sum())} of them wrong; "
      f"share of all wrong answers that look confident: {conf[y == 1].mean():.0%}")

print("\nSENSITIVITY: AUROC with ambiguous questions counted as right / as wrong")
print(f"{'score':<18} {'drop':>6} {'right':>6} {'wrong':>6}")
yr, Xr = arrays("right")
yw, Xw = arrays("wrong")
for k in KEYS:
    print(f"{k:<18} {roc_auc_score(y, X[k]):6.3f} {roc_auc_score(yr, Xr[k]):6.3f} {roc_auc_score(yw, Xw[k]):6.3f}")
