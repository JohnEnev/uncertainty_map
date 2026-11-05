# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.40", "datasets>=2.20", "tqdm"]
# ///
"""Step 1: sample answers.

For each question: one greedy answer (with token logprobs, for the baseline and the
correctness label) and N sampled answers at temperature 1 (for semantic entropy).
Datasets: triviaqa (TriviaQA rc.nocontext validation) or nq_open (Natural Questions Open
validation). Works with any OpenAI-compatible endpoint (vLLM on a pod, or a hosted provider).
Resumable and de-duplicated by question id.
"""
import argparse, json, os, random, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

PROMPT = "Answer the following question in as few words as possible.\nQuestion: {q}\nAnswer:"


def norm(s):
    s = re.sub(r"[^\w\s]", " ", s.lower())
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def load_items(name, n):
    from datasets import load_dataset

    if name == "triviaqa":
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        ds = ds.shuffle(seed=0).select(range(n))  # same order as the first runs
        return [
            {"id": x["question_id"], "question": x["question"], "aliases": x["answer"]["normalized_aliases"]}
            for x in ds
        ]
    if name == "nq_open":
        ds = load_dataset("google-research-datasets/nq_open", split="validation")
        items = [
            {"id": f"nq_{i}", "question": x["question"], "aliases": sorted({norm(a) for a in x["answer"]} - {""})}
            for i, x in enumerate(ds)
        ]
        random.Random(0).shuffle(items)
        return items[:n]
    raise ValueError(name)


def ask(client, args, q, temperature, n, logprobs=False):
    kwargs = {}
    chat_kwargs = json.loads(args.chat_kwargs)
    if chat_kwargs:
        kwargs["extra_body"] = {"chat_template_kwargs": chat_kwargs}
    r = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": PROMPT.format(q=q)}],
        temperature=temperature,
        n=n,
        max_tokens=32,
        logprobs=logprobs,
        **kwargs,
    )
    return r.choices


def run_one(client, args, item):
    q = item["question"]
    g = ask(client, args, q, 0.0, 1, logprobs=True)[0]
    lps = [t.logprob for t in (g.logprobs.content or [])] if g.logprobs else []
    samples = [c.message.content.strip() for c in ask(client, args, q, 1.0, args.n)]
    return {
        "id": item["id"],
        "question": q,
        "aliases": item["aliases"],
        "greedy": g.message.content.strip(),
        "mean_logprob": sum(lps) / len(lps) if lps else None,
        "samples": samples,
    }


def main():
    from openai import OpenAI

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="triviaqa", choices=["triviaqa", "nq_open"])
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--questions", type=int, default=200)
    p.add_argument("--n", type=int, default=10, help="samples per question")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", default="data/samples.jsonl")
    p.add_argument(
        "--chat-kwargs", default='{"enable_thinking": false}',
        help="JSON passed to the chat template; use '{}' for models without a thinking switch",
    )
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    seen = set()
    if os.path.exists(args.out):
        seen = {json.loads(l)["id"] for l in open(args.out)}

    todo = []
    for x in load_items(args.dataset, args.questions):
        if x["id"] not in seen:  # also drops duplicate ids inside the dataset
            seen.add(x["id"])
            todo.append(x)

    client = OpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    with open(args.out, "a") as f, ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_one, client, args, x) for x in todo]
        for fut in tqdm(as_completed(futs), total=len(futs)):
            f.write(json.dumps(fut.result()) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
