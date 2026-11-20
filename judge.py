# /// script
# requires-python = ">=3.11"
# dependencies = ["anthropic", "tqdm"]
# ///
"""Grade each greedy answer with an LLM judge: correct / incorrect / ambiguous.

The judge sees only the question, the dataset's accepted answers and the model's greedy
answer, never the samples or any uncertainty score. Resumable: ids already in --out are
skipped. Needs ANTHROPIC_API_KEY.
"""
import argparse, json, os, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

DEFAULT_MODEL = "claude-haiku-5-5"
SYSTEM = "You grade answers to short factual questions against the dataset's accepted answers. Reply with JSON only."
PROMPT = """Question: {question}
Accepted answers (from the dataset, lowercased; the list may be incomplete or contain odd variants): {aliases}
Model answer: {answer}

Choose one verdict:
- "correct": the model answer gives the same entity, value or fact as at least one accepted answer. Accept synonyms, a shorter or longer form of the same name, other formats (digits or words, date styles), other languages or scripts, and extra detail that does not change the answer.
- "incorrect": the model answer gives a different answer, an answer too vague to identify an accepted one, several conflicting answers, or no answer (for example "unknown" or a request for clarification).
- "ambiguous": the question can reasonably refer to something else (for example a title shared by several works) and the model answer is right for that other reading, or the accepted answers look wrong or out of date.
Use your own knowledge only to recognise equivalent answers or to flag ambiguity. Do not mark an answer correct when it contradicts the accepted answers on a clear question.

Reply with JSON only: {{"verdict": "correct" | "incorrect" | "ambiguous", "reason": "<one short sentence>"}}"""

VERDICTS = ("correct", "incorrect", "ambiguous")


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            v = str(d.get("verdict", "")).strip().lower()
            if v in VERDICTS:
                return v, str(d.get("reason", ""))[:300]
        except json.JSONDecodeError:
            pass
    low = text.lower()
    for v in ("incorrect", "ambiguous", "correct"):  # "incorrect" contains "correct": check it first
        if re.search(rf"\b{v}\b", low):
            return v, "unparsed: " + text[:200]
    return None, "unparsed: " + text[:200]


def judge_one(client, model, row):
    aliases = list(dict.fromkeys(row["aliases"]))[:15]
    prompt = PROMPT.format(
        question=row["question"],
        aliases=json.dumps(aliases, ensure_ascii=False),
        answer=row["greedy"],
    )
    msg = client.messages.create(
        model=model, max_tokens=1500, system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(getattr(b, "text", "") for b in msg.content)
    verdict, reason = parse(text)
    return {"id": row["id"], "verdict": verdict, "reason": reason}


def main():
    from anthropic import Anthropic

    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="only the first N questions (pilot)")
    args = p.parse_args()

    rows = {}
    for l in open(args.samples):
        r = json.loads(l)
        rows[r["id"]] = r  # same de-duplication as score.py: last row per id
    rows = list(rows.values())[: args.limit]

    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out) if json.loads(l)["verdict"]}
    todo = [r for r in rows if r["id"] not in done]

    client = Anthropic(max_retries=6)
    with open(args.out, "a") as f, ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(judge_one, client, args.model, r) for r in todo]
        for fut in tqdm(as_completed(futs), total=len(futs)):
            f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
            f.flush()

    counts = {}
    for l in open(args.out):
        v = json.loads(l)["verdict"]
        counts[v] = counts.get(v, 0) + 1
    print("verdicts:", counts)


if __name__ == "__main__":
    main()
