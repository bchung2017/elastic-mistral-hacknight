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
import time

import numpy as np
from pydantic import BaseModel

from engine import INDEX, Parsed, concept_subspace, concept_vectors, es, keyword_query, load_mu, mistral, quantum_query
from qlogic import compile_query, script_params

JUDGE_MODEL = "mistral-small-latest"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "judgments.json")
K = 10

# Hand-written parses, not Mistral parses, so the eval is reproducible. Survivors of `eval.py --screen`
# (q·r >= 0.2 and baseline leak@10 >= 0.2, i.e. the no-NOT query already pulls in the negated concept).
QUERIES = [
    Parsed(positive="squirrels eating", not_=["peanuts"]),                      # q·r 0.23, base leak 0.20
    Parsed(positive="squirrels eating", not_=["nuts", "acorns"]),               # q·r 0.27/0.28, base leak 0.50
    Parsed(positive="birds", not_=["pigeons"]),                                 # q·r 0.62, base leak 0.30
    Parsed(positive="squirrels", not_=["rats"]),                                # q·r 0.50, base leak 0.30
    Parsed(positive="squirrels", not_=["pigeons", "rats"]),                     # q·r 0.45/0.50, base leak 0.20
    Parsed(positive="squirrels", not_=["dogs"], any_of=["fighting", "chasing"]),  # q·r 0.38, base leak 0.20
]

# Screening pool. --screen runs only the no-NOT baseline on these and prints q·r and baseline leak@10.
CANDIDATES = [
    Parsed(positive="squirrels being chased", not_=["dogs"]),
    Parsed(positive="squirrels running", not_=["dogs"]),
    Parsed(positive="squirrels interacting with people", not_=["children"]),
    Parsed(positive="squirrels near the playground", not_=["children"]),
    Parsed(positive="people feeding squirrels", not_=["tourists"]),
    Parsed(positive="squirrels eating", not_=["peanuts"]),
    Parsed(positive="squirrels eating", not_=["nuts", "acorns"]),
    Parsed(positive="squirrels climbing", not_=["trees"]),
    Parsed(positive="birds", not_=["pigeons"]),
    Parsed(positive="animals in the park", not_=["squirrels"]),
    Parsed(positive="squirrels", not_=["rats"]),
    Parsed(positive="squirrels", not_=["pigeons", "rats"]),
    Parsed(positive="squirrels", not_=["dogs"], any_of=["fighting", "chasing"]),
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
    for attempt in range(6):
        try:
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
            break
        except Exception as e:  # 429 from the judge; free-tier rate limit
            if "429" not in str(e) or attempt == 5:
                raise
            time.sleep(2 ** attempt)
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


def overlaps(p, mu):
    pos, nots, _ = concept_vectors(p, mu)
    return {n: float(pos @ r) for n, r in zip(p.not_, nots)} if pos is not None else {}


def screen():
    """Baseline only: which candidates have something to negate? Keep q·r >= 0.2 and baseline leak >= 0.2."""
    mu = load_mu()
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    print(f"{'query':58s} {'q·r':>22s}  base leak  base keep")
    for p in CANDIDATES:
        pos, nots, ors = concept_vectors(p, mu)
        qr = " ".join(f"{pos @ r:.2f}" for r in nots) if pos is not None else "-"
        body = quantum_query(script_params(*compile_query(pos, [], ors)), K)
        hits = es.search(index=INDEX, **body)["hits"]["hits"]
        js = [judge(h["_source"]["note"], p.positive or " or ".join(p.any_of), p.not_, cache) for h in hits]
        json.dump(cache, open(CACHE, "w"))
        leak = sum(bool(j["neg"]) for j in js) / len(hits)
        keep = sum(j["pos"] for j in js) / len(hits)
        flag = "  <-- keep" if leak >= 0.2 and (pos is None or min(pos @ r for r in nots) >= 0.2) else ""
        print(f"{p.positive or '':30s} NOT {str(p.not_):26s} {qr:>22s}  {leak:9.2f}  {keep:9.2f}{flag}")


def run(limit, paraphrase_k):
    mu = load_mu()
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    rows = []
    for p in QUERIES[:limit]:
        label = f"{p.positive or ''} | NOT {p.not_} | ANY {p.any_of}"
        print(f"\n== {label}  q·r {overlaps(p, mu)}")
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
    ap.add_argument("--screen", action="store_true", help="baseline-only pass over CANDIDATES")
    a = ap.parse_args()
    screen() if a.screen else run(a.limit, 0 if a.no_paraphrase else a.k)
