# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scikit-learn", "typesafe-sdk", "tqdm"]
# ///
"""Export everything the demo page needs into demo/data.json.

Per cell: AUROC with 95% CI for the six scores, key paired differences, the sample-count
curve and the share of errors where all samples agree. Labels come from the judge
(ambiguous questions dropped); a cell without a judge file falls back to the alias matcher.

Plus a gallery of example questions, picked automatically across cells in six categories,
each with its samples and the Jev and NLI similarity matrices.
Feature examples with --include tag:id or tag:phrase, drop with --exclude tag:id (repeatable).
"""
import argparse, json, os
from datetime import date
import numpy as np
from sklearn.metrics import roc_auc_score
from score import auroc_ci, se_hard, se_soft, sim_matrix

TAGS = ["triviaqa_qwen3-4b", "triviaqa_qwen3-14b", "triviaqa_gemma-3-12b-it",
        "popqa_qwen3-4b", "popqa_qwen3-14b", "nq_open_qwen3-4b", "nq_open_qwen3-14b"]
DATASETS = {"triviaqa": "TriviaQA", "nq_open": "NQ-Open", "popqa": "PopQA"}
KEYS = ["neg_mean_logprob", "lexical_entropy", "se_hard_nli", "se_soft_nli", "se_hard_jev", "se_soft_jev"]
PAIRED = [("se_soft_jev", "se_hard_jev"), ("se_soft_nli", "se_hard_nli"),
          ("se_soft_jev", "se_soft_nli"), ("se_soft_jev", "neg_mean_logprob"),
          ("se_soft_jev", "lexical_entropy")]
KS = [3, 4, 5, 6, 8, 10]
CATEGORIES = ["confident_right", "confident_wrong", "scattered_right", "scattered_wrong",
              "partial_overlap", "judges_disagree"]


def load(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else None


def split_tag(tag):
    for k, name in DATASETS.items():
        if tag.startswith(k + "_"):
            return name, tag[len(k) + 1:]
    return tag, ""


def paired(y, a, b, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            d.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return [round(float(np.mean(d)), 4), round(float(lo), 4), round(float(hi), 4)]


def r3(M):
    return [[round(float(v), 3) for v in row] for row in M]


def cell(tag, d):
    samples = {r["id"]: r for r in load(f"{d}/{tag}_samples.jsonl")}
    jev = {r["id"]: r["pairs"] for r in load(f"{d}/{tag}_pairs.jsonl")}
    nli = {r["id"]: r for r in load(f"{d}/{tag}_pairs_nli.jsonl")}
    scores = load(f"{d}/{tag}_scores_all.jsonl")
    judge = load(f"{d}/{tag}_judge.jsonl")
    verdict = {r["id"]: r["verdict"] for r in judge} if judge else None

    rows = []
    for r in scores:
        if verdict is not None:
            v = verdict.get(r["id"])
            if v not in ("correct", "incorrect"):
                continue
            r["is_wrong"] = int(v == "incorrect")
            r["verdict"] = v
        else:
            r["is_wrong"] = r["wrong"]
            r["verdict"] = "incorrect" if r["wrong"] else "correct"
        n = len(samples[r["id"]]["samples"])
        r["_J"] = sim_matrix(n, jev[r["id"]])
        r["_Ns"] = sim_matrix(n, nli[r["id"]]["pairs"])
        r["_Nh"] = sim_matrix(n, nli[r["id"]]["hard"])
        rows.append(r)

    y = np.array([r["is_wrong"] for r in rows])
    X = {k: np.array([r[k] for r in rows]) for k in KEYS}
    auroc = {k: [round(float(v), 4) for v in auroc_ci(y, X[k])] for k in KEYS}
    pairs = {f"{a}-{b}": paired(y, X[a], X[b]) for a, b in PAIRED}

    kcurve = {}
    for k in KS:
        sj = [se_soft(r["_J"][:k, :k]) for r in rows]
        hj = [se_hard(r["_J"][:k, :k]) for r in rows]
        sn = [se_soft(r["_Ns"][:k, :k]) for r in rows]
        hn = [se_hard(r["_Nh"][:k, :k]) for r in rows]
        kcurve[k] = {name: round(float(roc_auc_score(y, v)), 4) for name, v in
                     [("se_soft_jev", sj), ("se_hard_jev", hj), ("se_soft_nli", sn), ("se_hard_nli", hn)]}

    conf = X["se_hard_jev"] < 1e-9
    dataset, model = split_tag(tag)
    summary = {
        "tag": tag, "dataset": dataset, "model": model,
        "labels": "judge" if verdict is not None else "alias matcher",
        "n": len(rows), "n_dropped": len(scores) - len(rows),
        "wrong_rate": round(float(y.mean()), 4),
        "auroc": auroc, "paired": pairs, "kcurve": kcurve,
        "all_agree": int(conf.sum()),
        "all_agree_wrong": int(y[conf].sum()),
        "confident_error_share": round(float(conf[y == 1].mean()), 4) if y.sum() else None,
    }
    for r in rows:
        r["_tag"], r["_samples"] = tag, samples[r["id"]]
    return summary, rows


def example(r, category):
    s = r["_samples"]
    return {
        "tag": r["_tag"], "id": r["id"], "category": category,
        "question": s["question"], "greedy": s["greedy"], "aliases": s["aliases"][:5],
        "verdict": r["verdict"], "samples": s["samples"],
        "jev": r2(r["_J"]), "nli_soft": r2(r["_Ns"]), "nli_hard": r2(r["_Nh"]),
        "scores": {k: round(float(r[k]), 4) for k in KEYS},
    }


def r2(M):
    return r3(M)


def auto_category(r):
    kind = "confident" if r["se_hard_jev"] < 1e-9 else "scattered"
    return f'{kind}_{"wrong" if r["is_wrong"] else "right"}'


def pick_gallery(all_rows, tags, per_cat, include, exclude):
    def key(r):
        return f'{r["_tag"]}:{r["id"]}'

    def qnorm(r):
        return " ".join(r["_samples"]["question"].lower().split())

    def ok(r):
        s = r["_samples"]
        return (key(r) not in exclude and len(s["greedy"]) <= 90
                and max(len(x) for x in s["samples"]) <= 140)

    def mid_share(r):
        J = r["_J"]
        n = len(J)
        vals = [J[i, j] for i in range(n) for j in range(i + 1, n)]
        return sum(0.15 < v < 0.85 for v in vals) / len(vals)

    rules = {
        "confident_right": (lambda r: not r["is_wrong"], lambda r: r["se_soft_jev"]),
        "confident_wrong": (lambda r: r["is_wrong"] and r["se_hard_jev"] < 1e-9, lambda r: r["se_soft_jev"]),
        "scattered_right": (lambda r: not r["is_wrong"], lambda r: -r["se_soft_jev"]),
        "scattered_wrong": (lambda r: r["is_wrong"], lambda r: -r["se_soft_jev"]),
        "partial_overlap": (lambda r: True, lambda r: -mid_share(r)),
        "judges_disagree": (lambda r: True, lambda r: -abs(r["se_soft_jev"] - r["se_soft_nli"])),
    }
    used_keys, used_q, out = set(), set(), []

    # 1. featured examples: --include tag:id, or tag:phrase from the question
    for inc in include:
        tag, _, what = inc.partition(":")
        hits = [r for r in all_rows if r["_tag"] == tag
                and (str(r["id"]) == what or what.lower() in r["_samples"]["question"].lower())]
        if not hits:
            print(f"  ! --include {inc!r}: no match (wrong tag, or the question was dropped as ambiguous)")
            continue
        r = hits[0]
        if key(r) in used_keys:
            continue
        ex = example(r, auto_category(r))
        ex["featured"] = True
        out.append(ex)
        used_keys.add(key(r))
        used_q.add(qnorm(r))

    # 2. automatic picks, round-robin over cells with a rotating start so every cell appears
    by_tag = {t: [r for r in all_rows if r["_tag"] == t] for t in tags}
    for ci, cat in enumerate(CATEGORIES):
        cond, order = rules[cat]
        ranked = {t: sorted([r for r in rs if ok(r) and cond(r)], key=order) for t, rs in by_tag.items()}
        start = (ci * per_cat) % len(tags)
        rotation = tags[start:] + tags[:start]
        got, progress = 0, True
        while got < per_cat and progress:
            progress = False
            for t in rotation:
                if got >= per_cat:
                    break
                for r in ranked[t]:
                    if key(r) not in used_keys and qnorm(r) not in used_q:
                        ex = example(r, cat)
                        ex["featured"] = False
                        out.append(ex)
                        used_keys.add(key(r))
                        used_q.add(qnorm(r))
                        got += 1
                        progress = True
                        break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", default=TAGS)
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="demo/data.json")
    p.add_argument("--per-category", type=int, default=4)
    p.add_argument("--include", action="append", default=[], help="tag:id or tag:phrase-from-the-question to feature (repeatable)")
    p.add_argument("--exclude", action="append", default=[], help="tag:id to keep out of the gallery")
    args = p.parse_args()

    cells, all_rows = [], []
    for tag in args.tags:
        summary, rows = cell(tag, args.data)
        cells.append(summary)
        all_rows += rows
        a = summary["auroc"]
        print(f"{tag:<26} n={summary['n']:<5} labels={summary['labels']:<13} "
              f"soft Jev {a['se_soft_jev'][0]:.3f}  hard Jev {a['se_hard_jev'][0]:.3f}  log-prob {a['neg_mean_logprob'][0]:.3f}")

    gallery = pick_gallery(all_rows, args.tags, args.per_category, args.include, set(args.exclude))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"generated": date.today().isoformat(), "keys": KEYS, "ks": KS,
                   "cells": cells, "gallery": gallery}, f, ensure_ascii=False)
    print(f"\n{len(gallery)} gallery examples -> {args.out} ({os.path.getsize(args.out) // 1024} KB)")
    for g in gallery:
        star = "*" if g.get("featured") else " "
        print(f" {star}{g['category']:<16} {g['tag']:<26} {g['question'][:60]}")


if __name__ == "__main__":
    main()