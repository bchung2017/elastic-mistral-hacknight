"""leak@10 / keep@10 evaluation (MATH.md §9).

python eval.py            # all queries
python eval.py --size 10 --judge-model mistral-large-4

Methods, all scored by Elasticsearch on the live index:
  baseline    q itself (no NOT)                       script_score
  keyword     bool match + must_not on the words      BM25
  naive       q - sum(r_i), renormalized              script_score (overshoots, MATH.md §3)
  projection  q' = (I - N N^T) q  (qlogic.compile_query)
  subspace    same, N from Mistral paraphrases per concept (MATH.md §4), top-k SVD
Metrics over top-k: leak = fraction judged about a negated concept, keep = fraction judged about the positive.
Mistral is the judge (structured yes/no per document).
"""
import argparse
import json

import numpy as np
from pydantic import BaseModel, Field

from engine import INDEX, META_ID, PARSE_MODEL, Parsed, center, embed, es, keyword_query, mistral, quantum_query
from qlogic import compile_query, script_params

# positive / not pairs chosen so q·r is visibly nonzero (MATH.md §3)
QUERIES = [
    Parsed(positive="squirrels eating food", **{"not": ["pizza", "bagels"]}),
    Parsed(positive="squirrels approaching people", **{"not": ["dogs"]}),
    Parsed(positive="squirrels chasing each other", **{"not": ["birds"]}),
    Parsed(positive="squirrels climbing", **{"not": ["trees"]}),
    Parsed(positive="squirrels being fed", **{"not": ["children", "kids"]}),
]
PARAPHRASES, RANK = 8, 3


class Paraphrases(BaseModel):
    items: list[str] = Field(description="short paraphrases, synonyms or concrete instances of the concept")


class Verdict(BaseModel):
    about_positive: bool
    about_negated: bool


class Verdicts(BaseModel):
    items: list[Verdict]


def paraphrase(concept):
    r = mistral.chat.parse(
        model=PARSE_MODEL,
        messages=[
            {"role": "system", "content": f"Give {PARAPHRASES} short paraphrases, synonyms or concrete instances of the concept, 1-4 words each, as they might appear in a note about squirrels in a park."},
            {"role": "user", "content": concept},
        ],
        response_format=Paraphrases,
    )
    return [concept] + r.choices[0].message.parsed.items[:PARAPHRASES]


def subspace_basis(vecs, k=RANK):
    """Top-k left singular vectors of the centered paraphrase cloud (MATH.md §4)."""
    U, s, _ = np.linalg.svd(np.stack(vecs, axis=1), full_matrices=False)
    return [U[:, i] for i in range(min(k, U.shape[1]))], s


def judge(p, notes, model):
    out = []
    for i in range(0, len(notes), 10):
        chunk = notes[i:i + 10]
        r = mistral.chat.parse(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "You judge park notes about squirrels. For each note answer two yes/no questions. "
                    f"about_positive: is the note about '{p.positive}'? "
                    f"about_negated: does the note involve any of {p.not_}? "
                    "Return one verdict per note, in order.")},
                {"role": "user", "content": "\n".join(f"{j + 1}. {n}" for j, n in enumerate(chunk))},
            ],
            response_format=Verdicts,
        )
        v = r.choices[0].message.parsed.items
        assert len(v) == len(chunk), (len(v), len(chunk))
        out.extend(v)
    return out


def run(p, size, judge_model):
    mu = np.array(es.get(index=INDEX, id=META_ID)["_source"]["mu"])
    V = center(embed([p.positive] + p.not_), mu)
    q, nots = V[0], list(V[1:])
    overlap = {n: float(q @ r) for n, r in zip(p.not_, nots)}

    naive = q - sum(nots)
    naive /= np.linalg.norm(naive)

    para = {n: paraphrase(n) for n in p.not_}
    para_vecs = {n: list(center(embed(para[n]), mu)) for n in p.not_}
    sub = []
    sv = {}
    for n in p.not_:
        basis, s = subspace_basis(para_vecs[n])
        sub.extend(basis)
        sv[n] = s[:RANK + 2].round(3).tolist()

    methods = {
        "baseline": quantum_query(script_params(q, [], []), size),
        "keyword": keyword_query(p, size),
        "naive": quantum_query(script_params(naive, [], []), size),
        "projection": quantum_query(script_params(*compile_query(q, nots, [])), size),
        "subspace": quantum_query(script_params(*compile_query(q, sub, [])), size),
    }
    body = []
    for m in methods.values():
        body += [{"index": INDEX}, m]
    rs = es.msearch(searches=body)["responses"]
    hits = {k: [h for h in r["hits"]["hits"]] for k, r in zip(methods, rs)}

    ids = {}
    for hs in hits.values():
        for h in hs:
            ids[h["_id"]] = " ".join(h["_source"]["note"].split())
    order = list(ids)
    verdicts = dict(zip(order, judge(p, [ids[i] for i in order], judge_model)))

    rows = {}
    for k, hs in hits.items():
        n = len(hs)
        leak = sum(verdicts[h["_id"]].about_negated for h in hs) / n if n else float("nan")
        keep = sum(verdicts[h["_id"]].about_positive for h in hs) / n if n else float("nan")
        rows[k] = {"n": n, "leak": leak, "keep": keep}
    return {"query": {"positive": p.positive, "not": p.not_}, "overlap": overlap, "paraphrases": para,
            "singular_values": sv, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--judge-model", default=PARSE_MODEL)
    ap.add_argument("--json", help="write full results here")
    a = ap.parse_args()
    results = []
    for p in QUERIES:
        r = run(p, a.size, a.judge_model)
        results.append(r)
        print(f"\n{p.positive!r} NOT {p.not_}")
        for n, v in r["overlap"].items():
            print(f"  q·r({n!r}) = {v:.3f}   paraphrases: {r['paraphrases'][n][1:]}   sv: {r['singular_values'][n]}")
        print(f"  {'method':11s} {'n':>2s}  {'leak@k':>7s}  {'keep@k':>7s}")
        for k, row in r["rows"].items():
            print(f"  {k:11s} {row['n']:2d}  {row['leak']:7.2f}  {row['keep']:7.2f}")
    print("\nmean over queries")
    print(f"  {'method':11s}  {'leak@k':>7s}  {'keep@k':>7s}")
    for k in results[0]["rows"]:
        leak = np.nanmean([r["rows"][k]["leak"] for r in results])
        keep = np.nanmean([r["rows"][k]["keep"] for r in results])
        print(f"  {k:11s}  {leak:7.2f}  {keep:7.2f}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
