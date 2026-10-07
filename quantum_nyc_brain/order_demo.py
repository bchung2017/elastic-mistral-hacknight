"""Order effect (MATH.md §5): NOT a then NOT b, the reverse, and the joint projection, on the real index.

python order_demo.py "squirrels" "dogs" "children"

Prints the angle between a and b, the commutator norm |cos t| sin t, the residual a- and b-components of each
negated query (the leak), and the three top-10 lists side by side from one _msearch.
"""
import sys

import numpy as np

from engine import INDEX, center, embed, es, load_mu, quantum_query
from qlogic import compile_query, script_params

K = 10


def unit(v):
    return v / np.linalg.norm(v)


def main(positive, a_name, b_name):
    mu = load_mu()
    q, a, b = center(embed([positive, a_name, b_name]), mu)
    cos = float(a @ b)
    theta = np.degrees(np.arccos(cos))
    print(f"q·a {q @ a:.3f}  q·b {q @ b:.3f}  a·b {cos:.3f}  θ {theta:.1f}°  ‖[P_a,P_b]‖ = |cos θ| sin θ = {abs(cos) * np.sqrt(1 - cos**2):.3f}")

    def not_(v, r):
        return v - (v @ r) * r

    variants = {
        f"NOT {a_name} then NOT {b_name}": unit(not_(not_(q, a), b)),
        f"NOT {b_name} then NOT {a_name}": unit(not_(not_(q, b), a)),
        "joint (span{a,b})⊥": compile_query(q, [a, b])[0],
    }
    for name, v in variants.items():
        print(f"  {name:34s} residual a {v @ a:+.4f}  b {v @ b:+.4f}")
    print(f"  cos(seq ab, seq ba) {variants[list(variants)[0]] @ variants[list(variants)[1]]:.5f}")

    searches = []
    for v in variants.values():
        searches += [{"index": INDEX}, quantum_query(script_params(*compile_query(v, [], [])), K)]
    resp = es.msearch(searches=searches)["responses"]
    lists = [[h["_id"] for h in r["hits"]["hits"]] for r in resp]
    notes = {h["_id"]: " ".join(h["_source"]["note"].split()) for r in resp for h in r["hits"]["hits"]}
    for name, r in zip(variants, resp):
        print(f"\n{name}")
        for rank, h in enumerate(r["hits"]["hits"], 1):
            print(f"{rank:2d}  {h['_score'] - 1:6.3f}  {notes[h['_id']][:100]}")
    ab, ba, joint = lists
    print(f"\ntop-{K} overlap: seq ab ∩ seq ba {len(set(ab) & set(ba))}  |  seq ab ∩ joint {len(set(ab) & set(joint))}  |  seq ba ∩ joint {len(set(ba) & set(joint))}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
