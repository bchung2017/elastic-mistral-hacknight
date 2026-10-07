"""Quantum-logic query engine over squirrel stories (gfqj-f768). Ranked records only.

python engine.py ingest
python engine.py query "squirrels eating food but not pizza or bagels"
python engine.py query --size 10 --paraphrase "squirrels or chipmunks, not trees"

Ingest embeds every story, stores centered unit vectors (MATH.md §1) and the corpus mean in a meta doc.
Query: Mistral parses to {positive, not, any_of} -> qlogic.compile_query -> one _msearch of
(keyword must_not baseline, quantum script_score) -> side-by-side tables.
--paraphrase builds each NOT concept as a k-D subspace from Mistral paraphrases (MATH.md §4).
"""
import argparse
import os
import sys

import numpy as np
import requests
from elasticsearch import Elasticsearch, helpers
from mistralai.client import Mistral
from pydantic import BaseModel, ConfigDict, Field

from qlogic import SCRIPT, compile_query, script_params

INDEX = "qnb_squirrels"
META_ID = "__mean__"
URL = "https://data.cityofnewyork.us/resource/gfqj-f768.json"
EMBED_MODEL, PARSE_MODEL = "mistral-embed", "mistral-large-4"
D = 1024
SOURCE = ["note", "hectare", "date"]

es = Elasticsearch(os.environ["ELASTIC_ENDPOINT"], api_key=os.environ["ELASTIC_API_KEY"])
mistral = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


class Parsed(BaseModel):
    # JSON key is "not" (schema + wire); attribute is not_ because "not" is a Python keyword
    model_config = ConfigDict(populate_by_name=True)
    positive: str | None = None
    not_: list[str] = Field(default_factory=list, alias="not")
    any_of: list[str] = Field(default_factory=list)


class ParsedLLM(BaseModel):
    """What Mistral fills. "not" is a JSON Schema keyword and the model mis-fills a property by that name, so it's `exclude` here."""
    positive: str | None
    exclude: list[str]
    any_of: list[str]


class Paraphrases(BaseModel):
    paraphrases: list[str]


def embed(texts, batch=32):
    out = []
    for i in range(0, len(texts), batch):
        r = mistral.embeddings.create(model=EMBED_MODEL, inputs=texts[i:i + batch])
        out.extend(d.embedding for d in r.data)
    return np.array(out, dtype=np.float64)


def center(E, mu):
    E = E - mu
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def load_mu():
    return np.array(es.get(index=INDEX, id=META_ID)["_source"]["mu"])


def ingest():
    rows = requests.get(URL, params={"$where": "note_squirrel_park_stories IS NOT NULL", "$limit": 5000}, timeout=60).json()
    rows = [r for r in rows if r["note_squirrel_park_stories"].strip()]
    E = embed([r["note_squirrel_park_stories"] for r in rows])
    mu = E.mean(axis=0)
    V = center(E, mu)
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    off = ~np.eye(len(E), dtype=bool)
    print(f"{len(rows)} stories | mean pairwise cosine raw {(En @ En.T)[off].mean():.3f} -> centered {(V @ V.T)[off].mean():.3f}")
    if es.indices.exists(index=INDEX):
        es.indices.delete(index=INDEX)
    es.indices.create(index=INDEX, mappings={"properties": {
        "vec": {"type": "dense_vector", "dims": D, "index": False},
        "note": {"type": "text"},
        "mu": {"type": "float", "index": False, "doc_values": False},  # ES 9 drops dynamic dense_vector from _source
        "hectare": {"type": "keyword"}, "shift": {"type": "keyword"}, "date": {"type": "keyword"},
    }})
    docs = [{"_index": INDEX, "_id": str(i), "_source": {
        "vec": V[i].tolist(), "note": r["note_squirrel_park_stories"],
        "hectare": r.get("hectare"), "shift": r.get("shift"), "date": r.get("date")}} for i, r in enumerate(rows)]
    docs.append({"_index": INDEX, "_id": META_ID, "_source": {"mu": mu.tolist()}})
    helpers.bulk(es, docs, chunk_size=200)
    es.indices.refresh(index=INDEX)


def parse(text):
    r = mistral.chat.parse(
        model=PARSE_MODEL,
        messages=[
            {"role": "system", "content": (
                "Convert the search request into a flat logic expression over short noun-phrase concepts. "
                "positive: the main thing sought (null if the request is only a set of alternatives). "
                "exclude: concepts to exclude. any_of: alternatives joined by OR. Concepts are 1-4 words, no logic words.")},
            {"role": "user", "content": text},
        ],
        response_format=ParsedLLM,
        temperature=0, top_p=1,
    )
    p = r.choices[0].message.parsed
    return Parsed(positive=p.positive, not_=p.exclude, any_of=p.any_of)


def paraphrases(concept, n=8):
    r = mistral.chat.parse(
        model=PARSE_MODEL,
        messages=[
            {"role": "system", "content": (
                f"Give {n} short phrases (1-3 words each) a park visitor would literally write in a field note when "
                "mentioning this concept: plain synonyms, specific kinds, and closely associated things "
                "(for 'dogs': dog, puppy, off-leash dog, leash, barking). Plain words only, no metaphors, no duplicates.")},
            {"role": "user", "content": concept},
        ],
        response_format=Paraphrases,
    )
    return r.choices[0].message.parsed.paraphrases[:n]


def concept_subspace(concept, mu, k=3, verbose=True):
    """Top-k left singular vectors of the centered paraphrase embeddings (MATH.md §4). Returns list of k unit vectors."""
    phrases = [concept] + paraphrases(concept)
    V = center(embed(phrases), mu)
    U, S, _ = np.linalg.svd(V.T, full_matrices=False)
    if verbose:
        print(f"  subspace({concept!r}): {phrases[1:]}")
        print(f"    singular values {np.round(S, 2).tolist()} -> keeping {k}")
    return [U[:, i] for i in range(k)]


def concept_vectors(p, mu, paraphrase_k=0):
    """(pos, nots, ors) as centered unit vectors. paraphrase_k > 0 expands each NOT concept into k basis vectors."""
    names = ([p.positive] if p.positive else []) + p.not_ + p.any_of
    V = center(embed(names), mu)
    it = iter(V)
    pos = next(it) if p.positive else None
    nots = [next(it) for _ in p.not_]
    ors = [next(it) for _ in p.any_of]
    if paraphrase_k:
        nots = [v for name in p.not_ for v in concept_subspace(name, mu, paraphrase_k)]
    return pos, nots, ors


def keyword_query(p, n):
    b = {"must_not": [{"match": {"note": c}} for c in p.not_]}
    if p.positive:
        b["must"] = [{"match": {"note": p.positive}}]
        b["should"] = [{"match": {"note": c}} for c in p.any_of]
    else:
        b["should"] = [{"match": {"note": c}} for c in p.any_of]
        b["minimum_should_match"] = 1
    return {"size": n, "query": {"bool": b}, "_source": SOURCE}


def quantum_query(params, n, with_vec=False):
    body = {"size": n, "_source": SOURCE, "query": {"script_score": {
        "query": {"bool": {"must_not": [{"ids": {"values": [META_ID]}}]}},
        "script": {"source": SCRIPT, "params": params}}}}
    if with_vec:
        body["fields"] = ["vec"]
    return body


def hit(h, offset, q=None, u=None, c=None):
    s = h["_source"]
    # _source keeps null fields as null, so .get(k, '') still returns None for them
    out = {"id": h["_id"], "score": h["_score"] - offset, "note": " ".join(s["note"].split()),
           "hectare": s.get("hectare") or "", "date": s.get("date") or ""}
    if "fields" in h:  # positive + any_of: recompute s_pos and s_or for display; ranking is still the ES score
        d = np.array(h["fields"]["vec"])
        out["pos"] = float(d @ q)
        out["or"] = float(sum((ui @ d) ** 2 for ui in u)) if max(ci @ d for ci in c) > 0 else 0.0
    return out


def search(p, size, paraphrase_k=0):
    """Embed the parsed concepts, compile (MATH.md §7), run one _msearch. Returns a JSON-able dict."""
    mu = load_mu()
    pos, nots, ors = concept_vectors(p, mu, paraphrase_k)
    overlap = {name: float(pos @ r) for name, r in zip(p.not_, nots)} if pos is not None else {}
    q, u, c = compile_query(pos, nots, ors)
    both = q is not None and bool(u)
    r = es.msearch(searches=[{"index": INDEX}, keyword_query(p, size),
                             {"index": INDEX}, quantum_query(script_params(q, u, c), size, with_vec=both)])
    kw, qn = r["responses"]
    return {"parsed": {"positive": p.positive, "not": p.not_, "any_of": p.any_of}, "overlap": overlap, "took": r["took"],
            "keyword_total": kw["hits"]["total"]["value"],
            "keyword": [hit(h, 0.0) for h in kw["hits"]["hits"]],
            "quantum": [hit(h, 1.0, q, u, c) for h in qn["hits"]["hits"]]}


def table(title, hits):
    print(f"\n{title}")
    for rank, h in enumerate(hits, 1):
        extra = f"  pos {h['pos']:6.3f}  or {h['or']:5.3f}" if "pos" in h else ""
        print(f"{rank:2d}  {h['score']:7.3f}{extra}  {h['hectare']:4s} {h['date']:9s} {h['note'][:100]}")
    if not hits:
        print("  (no results)")


def query(text, size, paraphrase_k=0):
    p = parse(text)
    print(f"parsed: positive={p.positive!r} not={p.not_} any_of={p.any_of}")
    try:
        out = search(p, size, paraphrase_k)
    except ValueError as e:  # nothing to score (no positive and no usable any_of)
        sys.exit(f"cannot run this query: {e}")
    for name, v in out["overlap"].items():
        print(f"  q·r({p.positive!r}, {name!r}) = {v:.3f}")
    print(f"_msearch took {out['took']} ms")
    table(f"keyword (bool + must_not)  [BM25]  total {out['keyword_total']}", out["keyword"])
    both = out["quantum"] and "pos" in out["quantum"][0]
    table("quantum logic (script_score, exact)  [s = _score - 1" + ("; s = max(0,pos)·or]" if both else "]"), out["quantum"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    qp = sub.add_parser("query")
    qp.add_argument("text")
    qp.add_argument("--size", type=int, default=10)
    qp.add_argument("--paraphrase", type=int, nargs="?", const=3, default=0, metavar="K",
                    help="expand each NOT concept into a K-D paraphrase subspace (default K=3)")
    a = ap.parse_args()
    ingest() if a.cmd == "ingest" else query(a.text, a.size, a.paraphrase)
