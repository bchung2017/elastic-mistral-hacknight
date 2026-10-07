"""Run against the real cluster before building on MATH.md §8. Creates and deletes a scratch index.

export ELASTIC_ENDPOINT=... ELASTIC_API_KEY=...
python es_checks.py
"""
import os

import numpy as np
from elasticsearch import Elasticsearch, helpers

from qlogic import SCRIPT, compile_query, score, script_params

INDEX = "qnb_es_checks"
N_DOCS, D = 5000, 1024

es = Elasticsearch(os.environ["ELASTIC_ENDPOINT"], api_key=os.environ["ELASTIC_API_KEY"])
rng = np.random.default_rng(0)


def unit(v):
    return v / np.linalg.norm(v)


X = rng.normal(size=(N_DOCS, D)).astype(np.float32)
X /= np.linalg.norm(X, axis=1, keepdims=True)

if es.indices.exists(index=INDEX):
    es.indices.delete(index=INDEX)
es.indices.create(index=INDEX, mappings={"properties": {
    "vec": {"type": "dense_vector", "dims": D, "index": False},
    "note": {"type": "text"},
}})
ok, errors = helpers.bulk(
    es, ({"_index": INDEX, "_id": str(i), "_source": {"vec": X[i].tolist(), "note": f"doc {i}"}} for i in range(N_DOCS)),
    chunk_size=250, raise_on_error=False,
)
es.indices.refresh(index=INDEX)
print(f"indexed {ok}, errors {len(errors)}")

n1 = unit(rng.normal(size=D))
pos = unit(unit(rng.normal(size=D)) + 0.8 * n1)
ors = [unit(unit(rng.normal(size=D)) + 0.5 * X[k]) for k in range(4)]

cases = {
    "positive only": dict(positive=pos, not_=[n1]),
    "any_of only (q = null)": dict(not_=[n1], any_of=ors),
    "positive + not + 4 any_of": dict(positive=pos, not_=[n1], any_of=ors),
}
try:
    for name, kw in cases.items():
        params = script_params(*compile_query(**kw))
        resp = es.search(index=INDEX, size=10, query={"script_score": {
            "query": {"match_all": {}}, "script": {"source": SCRIPT, "params": params}}})
        got = [(h["_id"], h["_score"] - 1.0) for h in resp["hits"]["hits"]]
        ref = np.array([score(X[i].astype(np.float64), **kw) for i in range(N_DOCS)])
        top = np.argsort(-ref)[:10]
        same_ids = {i for i, _ in got} == {str(i) for i in top}
        max_err = max(abs(s - ref[int(i)]) for i, s in got)
        print(f"{name:28s} took {resp['took']:5d} ms | top-10 set matches numpy: {same_ids} | max score err {max_err:.2e}")
finally:
    es.indices.delete(index=INDEX)
