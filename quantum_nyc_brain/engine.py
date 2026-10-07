"""Quantum-logic query engine over squirrel stories (gfqj-f768). Ranked records only.

python engine.py ingest
python engine.py query "squirrels eating food but not pizza or bagels"
python engine.py query --size 10 "squirrels or chipmunks, not trees"

Ingest embeds every story, stores centered unit vectors (MATH.md §1) and the corpus mean in a meta doc.
Query: Mistral parses to {positive, not, any_of} -> qlogic.compile_query -> one _msearch of
(keyword must_not baseline, quantum script_score) -> side-by-side tables.
"""
import argparse
import os
import sys

import numpy as np
import requests
from elasticsearch import Elasticsearch, helpers
from mistralai.client import Mistral
from pydantic import BaseModel

from qlogic import SCRIPT, compile_query, script_params

INDEX = "qnb_squirrels"
META_ID = "__mean__"
URL = "https://data.cityofnewyork.us/resource/gfqj-f768.json"
EMBED_MODEL, PARSE_MODEL = "mistral-embed", "mistral-large-4"
D = 1024

es = Elasticsearch(os.environ["ELASTIC_ENDPOINT"], api_key=os.environ["ELASTIC_API_KEY"])
mistral = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


class Parsed(BaseModel):
    positive: str | None
    not_: list[str]
    any_of: list[str]


def embed(texts, batch=32):
    out = []
    for i in range(0, len(texts), batch):
        r = mistral.embeddings.create(model=EMBED_MODEL, inputs=texts[i:i + batch])
        out.extend(d.embedding for d in r.data)
    return np.array(out, dtype=np.float64)


def center(E, mu):
    E = E - mu
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def ingest():
    rows = requests.get(URL, params={"$where": "note_squirrel_park_stories IS NOT NULL", "$limit": 5000}, timeout=60).json()
    rows = [r for r in rows if r["note_squirrel_park_stories"].strip()]
    E = embed([r["note_squirrel_park_stories"] for r in rows])
    mu = E.mean(axis=0)
    V = center(E, mu)
    cos = (E / np.linalg.norm(E, axis=1, keepdims=True)) @ (E / np.linalg.norm(E, axis=1, keepdims=True)).T
    off = ~np.eye(len(E), dtype=bool)
    cc = V @ V.T
    print(f"{len(rows)} stories | mean pairwise cosine raw {cos[off].mean():.3f} -> centered {cc[off].mean():.3f}")
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
                "not_: concepts to exclude. any_of: alternatives joined by OR. Concepts are 1-4 words, no logic words.")},
            {"role": "user", "content": text},
        ],
        response_format=Parsed,
    )
    return r.choices[0].message.parsed


def keyword_query(p, n):
    must = {"match": {"note": p.positive}} if p.positive else None
    should = [{"match": {"note": c}} for c in p.any_of]
    b = {"must_not": [{"match": {"note": c}} for c in p.not_]}
    if must:
        b["must"] = [must]
        b["should"] = should
    else:
        b["should"] = should
        b["minimum_should_match"] = 1
    return {"size": n, "query": {"bool": b}, "_source": ["note", "hectare", "date"]}


def quantum_query(params, n):
    return {"size": n, "_source": ["note", "hectare", "date"], "query": {"script_score": {
        "query": {"bool": {"must_not": [{"ids": {"values": [META_ID]}}]}},
        "script": {"source": SCRIPT, "params": params}}}}


def table(title, hits, offset):
    print(f"\n{title}")
    for rank, h in enumerate(hits, 1):
        s = h["_source"]
        note = " ".join(s["note"].split())
        print(f"{rank:2d}  {h['_score'] - offset:7.3f}  {s.get('hectare', ''):4s} {s.get('date', ''):9s} {note[:110]}")
    if not hits:
        print("  (no results)")


def query(text, size):
    mu = np.array(es.get(index=INDEX, id=META_ID)["_source"]["mu"])
    p = parse(text)
    print(f"parsed: positive={p.positive!r} not={p.not_} any_of={p.any_of}")
    names = ([p.positive] if p.positive else []) + p.not_ + p.any_of
    V = center(embed(names), mu)
    it = iter(V)
    pos = next(it) if p.positive else None
    nots = [next(it) for _ in p.not_]
    ors = [next(it) for _ in p.any_of]
    if pos is not None:
        for name, r in zip(p.not_, nots):
            print(f"  q·r({p.positive!r}, {name!r}) = {pos @ r:.3f}")
    params = script_params(*compile_query(pos, nots, ors))
    r = es.msearch(searches=[{"index": INDEX}, keyword_query(p, size), {"index": INDEX}, quantum_query(params, size)])
    kw, qn = r["responses"]
    print(f"_msearch took {r['took']} ms")
    table(f"keyword (bool + must_not)  [BM25]  total {kw['hits']['total']['value']}", kw["hits"]["hits"], 0.0)
    table("quantum logic (script_score, exact)  [s = _score - 1]", qn["hits"]["hits"], 1.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    q = sub.add_parser("query")
    q.add_argument("text")
    q.add_argument("--size", type=int, default=10)
    a = ap.parse_args()
    ingest() if a.cmd == "ingest" else query(a.text, a.size)
