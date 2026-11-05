# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers", "tqdm"]
# ///
"""NLI baseline: the pair similarities used by the semantic entropy paper.

For every pair of sampled answers (i, j) of a question, an NLI model (DeBERTa-large
fine-tuned on MNLI) is asked whether answer i entails answer j and the other way round,
with the question prepended to both answers so the model has context.

Writes, per question, two similarity dicts in the same "i,j" format as pairs.jsonl:
  pairs: soft similarity = min(P(entail i->j), P(entail j->i))
  hard:  1 if BOTH directions have 'entailment' as the top class, else 0 (the paper's rule)
Identical normalized answers get 1.0 without a model call (same shortcut as jev_pairs.py).
Both directional probabilities are not stored; change SOFT below to try other combinations.
"""
import argparse, json, re
from itertools import combinations

MODEL = "microsoft/deberta-large-mnli"


def norm(s):
    s = re.sub(r"[^\w\s]", " ", s.lower())
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def soft(p_ij, p_ji):
    return min(p_ij, p_ji)


def build_tasks(rows):
    """Unique (premise, hypothesis) texts that need the model, in a stable order."""
    index, texts = {}, []
    for r in rows:
        q, s = r["question"], r["samples"]
        for i, j in combinations(range(len(s)), 2):
            if norm(s[i]) == norm(s[j]):
                continue
            for a, b in ((s[i], s[j]), (s[j], s[i])):
                key = (f"{q} {a}", f"{q} {b}")
                if key not in index:
                    index[key] = len(texts)
                    texts.append(key)
    return index, texts


def score_texts(texts, model_name, batch):
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32).eval()
    ent = [int(i) for i, l in model.config.id2label.items() if l.upper().startswith("ENTAIL")]
    assert len(ent) == 1, model.config.id2label
    ent = ent[0]
    print("device:", device, "| labels:", model.config.id2label, "| tasks:", len(texts))

    order = sorted(range(len(texts)), key=lambda i: len(texts[i][0]) + len(texts[i][1]))
    out = [None] * len(texts)
    with torch.no_grad():
        for s in tqdm(range(0, len(order), batch)):
            idx = order[s : s + batch]
            enc = tok(
                [texts[i][0] for i in idx],
                [texts[i][1] for i in idx],
                padding=True, truncation=True, max_length=256, return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits.float()
            pe = logits.softmax(-1)[:, ent].tolist()
            top = (logits.argmax(-1) == ent).tolist()
            for i, p, t in zip(idx, pe, top):
                out[i] = (p, t)
    return out


def assemble(rows, index, results):
    for r in rows:
        q, s = r["question"], r["samples"]
        soft_d, hard_d = {}, {}
        for i, j in combinations(range(len(s)), 2):
            k = f"{i},{j}"
            if norm(s[i]) == norm(s[j]):
                soft_d[k], hard_d[k] = 1.0, 1
                continue
            p_ij, t_ij = results[index[(f"{q} {s[i]}", f"{q} {s[j]}")]]
            p_ji, t_ji = results[index[(f"{q} {s[j]}", f"{q} {s[i]}")]]
            soft_d[k] = round(soft(p_ij, p_ji), 4)
            hard_d[k] = int(t_ij and t_ji)
        yield {"id": r["id"], "pairs": soft_d, "hard": hard_d}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="data/samples.jsonl")
    p.add_argument("--out", default="data/pairs_nli.jsonl")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--batch", type=int, default=128)
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.samples)]
    index, texts = build_tasks(rows)
    results = score_texts(texts, args.model, args.batch)
    with open(args.out, "w") as f:
        for rec in assemble(rows, index, results):
            f.write(json.dumps(rec) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
