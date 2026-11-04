# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.40", "datasets>=2.20", "tqdm"]
# ///
"""Step 1: sample answers on TriviaQA.

For each question: one greedy answer (with token logprobs, for the baseline and the
correctness label) and N sampled answers at temperature 1 (for semantic entropy).
Works with any OpenAI-compatible endpoint (vLLM on a pod, or a hosted provider).
Resumable: questions already in the output file are skipped.
"""
import argparse, json, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

PROMPT = "Answer the following question in as few words as possible.\nQuestion: {q}\nAnswer:"


def ask(client, model, q, temperature, n, logprobs=False):
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT.format(q=q)}],
        temperature=temperature,
        n=n,
        max_tokens=32,
        logprobs=logprobs,
        # Qwen3: turn off thinking so answers stay short
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return r.choices


def run_one(client, args, item):
    q = item["question"]
    g = ask(client, args.model, q, 0.0, 1, logprobs=True)[0]
    lps = [t.logprob for t in (g.logprobs.content or [])] if g.logprobs else []
    samples = [c.message.content.strip() for c in ask(client, args.model, q, 1.0, args.n)]
    return {
        "id": item["question_id"],
        "question": q,
        "aliases": item["answer"]["normalized_aliases"],
        "greedy": g.message.content.strip(),
        "mean_logprob": sum(lps) / len(lps) if lps else None,
        "samples": samples,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--questions", type=int, default=200)
    p.add_argument("--n", type=int, default=10, help="samples per question")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", default="data/samples.jsonl")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out)}

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    ds = ds.shuffle(seed=0).select(range(args.questions))
    todo = [x for x in ds if x["question_id"] not in done]

    client = OpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    with open(args.out, "a") as f, ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_one, client, args, x) for x in todo]
        for fut in tqdm(as_completed(futs), total=len(futs)):
            f.write(json.dumps(fut.result()) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
