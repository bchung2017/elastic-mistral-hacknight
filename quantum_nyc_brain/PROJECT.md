# Quantum NYC Brain — project reference

## Decision

**A quantum-logic query engine over NYC data. Output: ranked records only** (no generated answer paragraph).

Dropped: the autonomous ETL agent (mapping generation, self-repair, catalog). Data is loaded with the repo's existing notebooks plus an embedding step.

## Skeleton

1. **Input:** a natural-language query.
2. **Parse:** Mistral → logic expression of concepts + filters.
3. **Compile:** concepts → subspaces → projectors. Scoring rule: **MATH.md §7 is the source of truth** (NOT applied to every term, gated OR, soft-AND combination).
4. **Retrieve:** Elasticsearch `script_score`, exact brute force, returning 1 + s (MATH.md §8). Sent in one `_msearch` with the `must_not` keyword baseline.
5. **Output:** ranked NYC records.

## Constraints

- Build ~5:45–8:00. Stop ~7:30 to review MATH.md.
- 3-minute demo must show the Elastic part and the Mistral part explicitly.
- Code runs on the user's laptop. This container can't reach Socrata and has no keys.

## Pitch rules

- "Quantum" = quantum logic (Birkhoff–von Neumann 1936). One-line answer at the top of MATH.md, used only when asked.
- Never say unprompted: qubits, quantum computing, entanglement, superposition, collapse, speedup.

## MVP decisions

- **Dataset:** squirrel stories (gfqj-f768) only — the control corpus. CNC observer descriptions are a candidate second corpus, gated on description fill rate (~1–2k substantive descriptions).
- **Retrieval:** `script_score` exact for everything; no kNN/HNSW.
- **Grammar (flat, no nesting):** `{positive, not: [...], any_of: [...]}`, filled by Mistral structured output.
- **Concepts:** one embedding per concept; Mistral paraphrase → SVD subspace only if time allows.
- **Filters:** none.
- **Interface:** one script/notebook printing side-by-side tables (keyword baseline vs. quantum, from one `_msearch`).

## First steps on the laptop (in order)

1. `python smoke_test.py`: Elasticsearch reachable, Mistral import path (`mistralai.client` vs `mistralai`), embed dims, squirrel row count.
2. `python es_checks.py`: `vectorValue` on `index: false`, `null` param, Painless = numpy on the real cluster, latency at 5k docs. If any line fails, stop and fix before building.

## Status

- `smoke_test.py` and `es_checks.py` pass on the cluster (mistralai.client import OK, dims 1024, 809 squirrel stories with a note; Painless = numpy to ~1e-7, top-10 sets match, 0.4–1.2 s at 5k docs).
- `engine.py ingest|query` built: 809 docs, mean pairwise cosine 0.784 raw -> 0.000 centered (MATH.md §1 anisotropy confirmed for mistral-embed). One `_msearch` of keyword `must_not` vs. quantum `script_score`, side-by-side tables.
- ES 9 omits dense_vector from `_source`, so the corpus mean is stored in a non-vector `mu` float field.
- `eval.py`: methods 1–5 of MATH.md §9 from one `_msearch` per query, `mistral-small-latest` as structured judge, judgments cached in `judgments.json`. Smoke-run on 1 query only; real numbers pending.
- `order_demo.py`: sequential NOT both orders vs joint, θ, commutator norm, residuals, overlap. Effect is small as §5 predicts (8–10/10 overlap on four pairs); best pair so far `squirrels / pigeons / rats` (residual −0.16, overlap 8). Sequential residuals come out negative: the second projection overshoots into anti-a.
- `engine.py query --paraphrase [K]`: NOT concepts as K-D SVD subspaces from Mistral paraphrases (§4). Singular values are flat after the first (1.9, 0.98, 0.89…), so K>1 may be noise; eval decides.
- Display: for positive+any_of queries the table prints `pos` and `or` next to the product.
- Not done: full eval run, demo query selection, filling the Open section.

## Open (to fill in while building, not before)

- Squirrel stories row count and which text field gets embedded
- Demo queries (each needs a visible q·r overlap — MATH.md §3)
- Evaluation: leak@10 / keep@10 vs. `must_not` (MATH.md §8)
