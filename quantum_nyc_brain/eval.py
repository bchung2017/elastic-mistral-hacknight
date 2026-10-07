"""leak@10 / keep@10 for the five methods in MATH.md §9, Mistral as judge.

python eval.py                 # all queries in QUERIES
python eval.py --limit 1       # smoke run
python eval.py --no-paraphrase # skip method 5 (saves ~2 Mistral calls per NOT concept)

Every method retrieves the top-K from Elasticsearch (one _msearch per query). Each distinct doc in the union
of the result lists is judged once: is it about the positive concept, and which negated concepts is it about.
Judgments are cached in judgments.json so re-runs only pay for new docs.
"""
import argparse
import json
import os

import numpy as np
from pydantic import BaseModel

from engine import INDEX, Parsed, concept_subspace, concept_vectors, es, keyword_query, load_mu, mistral, quantum_query
from qlogic import compile_query, script_params

JUDGE_MODEL = "mistral-small-latest"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "judgments.json")
K = 10

# Hand-written parses, not Mistral parses, so the eval is reproducible. Each needs visible q·r (MATH.md §3).
QUERIES = [
    Parsed(positive="squirrels eating food", not_=["pizza", "bagels"], any_of=[]),
    Parsed(positive="squirrels", not_=["dogs"], any_of=["fighting", "chasing"]),
    Parsed(positive="squirrels near people", not_=["children"], any_of=[]),
    Parsed(positive="birds", not_=["squirrels"], any_of=[]),
    Parsed(positive="squirrels in trees", not_=["nuts", "acorns"], any_of=[]),
]


class Judgment(BaseModel):
    about_positive: bool
    about_negated: list[str]


def unit(v):
    return v / np.linalg.norm(v)


def judge(note, positive, negated, cache):
    key = json.dumps([note, positive, negated])
    if key in cache:
        return cache[key]
    r = mistral.chat.parse(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": (
                "You judge a short field note from a squirrel census. Answer two things. "
                f"about_positive: is the note substantially about '{positive}'? "
                f"about_negated: which of these concepts is the note about, even in passing: {negated}. "
                "Return only names from that list.")},
            {"role": "user", "content": note},
        ],
        response_format=Judgment,
    )
    j = r.choices[0].message.parsed
    cache[key] = {"pos": j.about_positive, "neg": [n for n in j.about_negated if n in negated]}
    return cache[key]


def methods(p, mu, paraphrase_k):
    """name -> ES search body. All quantum variants go through the same SCRIPT."""
    pos, nots, ors = concept_vectors(p, mu)
    out = {}
    out["baseline (no NOT)"] = quantum_query(script_params(*compile_query(pos, [], ors)), K)
    out["keyword must_not"] = keyword_query(p, K)
    naive = pos - sum(nots) if pos is not None else None  # q - 1.0*r per r (MATH.md §9 method 3)
    if naive is not None:
        out["naive subtraction"] = quantum_query(script_params(*compile_query(unit(naive), [], ors)), K)
    out["projection (1 vec)"] = quantum_query(script_params(*compile_query(pos, nots, ors)), K)
    if paraphrase_k:
        sub = [v for name in p.not_ for v in concept_subspace(name, mu, paraphrase_k, verbose=False)]
        out[f"projection (k={paraphrase_k} subspace)"] = quantum_query(script_params(*compile_query(pos, sub, ors)), K)
    return out


def run(limit, paraphrase_k):
    mu = load_mu()
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    rows = []
    for p in QUERIES[:limit]:
        label = f"{p.positive or ''} | NOT {p.not_} | ANY {p.any_of}"
        print(f"\n== {label}")
        m = methods(p, mu, paraphrase_k)
        searches = []
        for body in m.values():
            searches += [{"index": INDEX}, body]
        resp = es.msearch(searches=searches)["responses"]
        pos_name = p.positive or " or ".join(p.any_of)
        for name, r in zip(m, resp):
            hits = r["hits"]["hits"]
            js = [judge(h["_source"]["note"], pos_name, p.not_, cache) for h in hits]
            json.dump(cache, open(CACHE, "w"))
            n = max(len(hits), 1)
            leak = sum(bool(j["neg"]) for j in js) / n
            keep = sum(j["pos"] for j in js) / n
            rows.append((label, name, leak, keep, len(hits)))
            print(f"  {name:26s} leak@{K} {leak:.2f}  keep@{K} {keep:.2f}  (n={len(hits)})")
    print("\n== mean over queries")
    for name in dict.fromkeys(r[1] for r in rows):
        sel = [r for r in rows if r[1] == name]
        print(f"  {name:26s} leak@{K} {np.mean([r[2] for r in sel]):.2f}  keep@{K} {np.mean([r[3] for r in sel]):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=len(QUERIES))
    ap.add_argument("--no-paraphrase", action="store_true")
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    run(a.limit, 0 if a.no_paraphrase else a.k)
