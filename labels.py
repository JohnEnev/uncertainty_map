# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Validate the judge against your own labels.

  export: write a CSV of questions to label by hand (half the matcher marked wrong, half right)
  agree:  compare your labels with the judge (agreement, Cohen's kappa, confusion table)

In the my_label column write c (correct), i (incorrect) or a (ambiguous).
"""
import argparse, csv, json, random

ALL_TAGS = ["triviaqa_qwen3-4b", "triviaqa_qwen3-14b", "triviaqa_gemma-3-12b-it",
            "popqa_qwen3-4b", "popqa_qwen3-14b", "nq_open_qwen3-4b", "nq_open_qwen3-14b"]
SHORT = {"c": "correct", "i": "incorrect", "a": "ambiguous"}


def export(args):
    rng = random.Random(0)
    out = []
    for tag in args.tags:
        samples = {r["id"]: r for r in map(json.loads, open(f"data/{tag}_samples.jsonl"))}
        scores = [json.loads(l) for l in open(f"data/{tag}_scores.jsonl")]
        wrong = [r for r in scores if r["wrong"]]
        right = [r for r in scores if not r["wrong"]]
        half = args.per_tag // 2
        pick = rng.sample(wrong, min(half, len(wrong))) + rng.sample(right, min(args.per_tag - half, len(right)))
        for r in pick:
            s = samples[r["id"]]
            out.append({"tag": tag, "id": r["id"], "question": s["question"], "greedy": s["greedy"],
                        "aliases": " | ".join(s["aliases"][:8]), "matcher": "wrong" if r["wrong"] else "right",
                        "my_label": ""})
    rng.shuffle(out)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {len(out)} rows to {args.out}; fill my_label with c / i / a")


def agree(args):
    rows = [r for r in csv.DictReader(open(args.sheet)) if r["my_label"].strip()]
    judge = {}
    for tag in {r["tag"] for r in rows}:
        for l in open(f"data/{tag}_judge.jsonl"):
            j = json.loads(l)
            judge[(tag, str(j["id"]))] = j["verdict"]
    pairs = []
    for r in rows:
        me = SHORT.get(r["my_label"].strip().lower()[:1])
        jv = judge.get((r["tag"], str(r["id"])))
        if me and jv:
            pairs.append((me, jv, r))
    n = len(pairs)
    if not n:
        print("no labelled rows with a judge verdict yet")
        return
    classes = ["correct", "incorrect", "ambiguous"]
    po = sum(a == b for a, b, _ in pairs) / n
    pe = sum((sum(a == c for a, _, _ in pairs) / n) * (sum(b == c for _, b, _ in pairs) / n) for c in classes)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    print(f"{n} labelled rows: judge agrees with you on {po:.0%}, Cohen's kappa {kappa:.2f}\n")
    print("rows = you, columns = judge")
    print(f"{'':<11}" + "".join(f"{c:>11}" for c in classes))
    for a in classes:
        print(f"{a:<11}" + "".join(f"{sum(x == a and y == b for x, y, _ in pairs):>11}" for b in classes))
    clear = [(a, r) for a, _, r in pairs if a != "ambiguous"]
    m_ok = sum((a == "incorrect") == (r["matcher"] == "wrong") for a, r in clear)
    print(f"\nfor comparison, the alias matcher agrees with you on {m_ok}/{len(clear)} non-ambiguous rows "
          f"({m_ok / max(len(clear), 1):.0%})")
    print("\ndisagreements:")
    for a, b, r in pairs:
        if a != b:
            print(f"  [{r['tag']}] {r['question'][:70]} -> {r['greedy'][:50]}  (you: {a}, judge: {b})")


p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="cmd", required=True)
e = sub.add_parser("export")
e.add_argument("--tags", nargs="+", default=ALL_TAGS)
e.add_argument("--per-tag", type=int, default=14)
e.add_argument("--out", default="labels_sheet.csv")
g = sub.add_parser("agree")
g.add_argument("--sheet", default="labels_sheet.csv")
args = p.parse_args()
export(args) if args.cmd == "export" else agree(args)
