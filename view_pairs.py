import argparse, json

p = argparse.ArgumentParser()
p.add_argument("--q", default="", help="only questions containing this text")
p.add_argument("--n", type=int, default=5, help="how many questions to show")
p.add_argument("--borderline", action="store_true",
               help="only questions with at least one pair between 0.05 and 0.95")
args = p.parse_args()

samples = {r["id"]: r for r in map(json.loads, open("data/samples.jsonl"))}
shown = 0
for line in open("data/pairs.jsonl"):
    r = json.loads(line)
    s = samples[r["id"]]
    if args.q.lower() not in s["question"].lower():
        continue
    vals = list(r["pairs"].values())
    mid = [v for v in vals if 0.05 < v < 0.95]
    if args.borderline and not mid:
        continue
    n = len(s["samples"])
    print("=" * 70)
    print(s["question"])
    print("greedy:", s["greedy"])
    for i, a in enumerate(s["samples"]):
        print(f"  {i}: {a}")
    print(f"pairs: {len(vals)} | >=0.95: {sum(v >= 0.95 for v in vals)}"
          f" | <=0.05: {sum(v <= 0.05 for v in vals)} | in between: {len(mid)}")
    print("     " + " ".join(f"{j:>4}" for j in range(n)))
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append("   -")
            else:
                row.append(f"{r['pairs'][f'{min(i, j)},{max(i, j)}']:4.2f}")
        print(f"{i:>3}: " + " ".join(row))
    shown += 1
    if shown >= args.n:
        break