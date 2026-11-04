# /// script
# requires-python = ">=3.11"
# dependencies = ["typesafe-sdk", "tqdm"]
# ///
"""Step 2: ask Jev whether each pair of sampled answers means the same thing.

n=10 samples -> 45 pairs per question. Identical answers (after normalization)
get P=1 without a call. Run with --probe first to print one raw Jev response and
check the field names before a full run.
"""
import argparse, json, os, re, string
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from tqdm import tqdm
from typesafe_sdk import Noul, TypeSafeClient

INSTRUCTIONS = (
    "Both answers respond to the question in the state. Do answer_a and answer_b give the "
    "same answer to that question? Say yes only if they refer to the same thing. "
    "Ignore wording, spelling, formatting and extra detail."
)


def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def get_answer(resp, key):
    # The exact response shape is not confirmed; --probe shows it. Fix this once.
    for get in (lambda: resp[key], lambda: resp.answers[key], lambda: getattr(resp, key)):
        try:
            return get()
        except (KeyError, AttributeError, TypeError):
            pass
    raise RuntimeError(f"Unknown response shape, run with --probe: {resp!r}")


def jev_same(client, question, a, b):
    resp = client.system_one(
        state={"question": question, "answer_a": a, "answer_b": b},
        questions={"same": Noul(instructions=INSTRUCTIONS)},
    )
    return float(get_answer(resp, "same").noul)


def pair_prob(client, question, a, b):
    if norm(a) == norm(b):
        return 1.0
    return jev_same(client, question, a, b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="data/samples.jsonl")
    p.add_argument("--out", default="data/pairs.jsonl")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--probe", action="store_true")
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.samples)]
    with TypeSafeClient() as client:
        if args.probe:
            r = rows[0]
            resp = client.system_one(
                state={"question": r["question"], "answer_a": r["samples"][0], "answer_b": r["samples"][1]},
                questions={"same": Noul(instructions=INSTRUCTIONS)},
            )
            print(repr(resp))
            print(dir(resp))
            return

        done = set()
        if os.path.exists(args.out):
            done = {json.loads(l)["id"] for l in open(args.out)}
        with open(args.out, "a") as f, ThreadPoolExecutor(args.workers) as ex:
            for r in tqdm([r for r in rows if r["id"] not in done]):
                s = r["samples"]
                idx = list(combinations(range(len(s)), 2))
                probs = ex.map(lambda ij: pair_prob(client, r["question"], s[ij[0]], s[ij[1]]), idx)
                pairs = {f"{i},{j}": pr for (i, j), pr in zip(idx, probs)}
                f.write(json.dumps({"id": r["id"], "pairs": pairs}) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
