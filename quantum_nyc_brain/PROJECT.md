# Quantum NYC Brain — project reference

## Decision

**A quantum-logic query engine over NYC data. Output: ranked records only** (no generated answer paragraph).

Dropped: the autonomous ETL agent (mapping generation, self-repair, catalog). Data is loaded with the repo's existing notebooks plus an embedding step.

## Skeleton

1. **Input:** a natural-language query.
2. **Parse:** Mistral → logic expression of concepts + filters.
3. **Compile:** concepts → subspaces → projectors → one query vector (MATH.md). Filters (geo, date, borough) stay ordinary Elasticsearch filters.
4. **Retrieve:** Elasticsearch kNN on the query vector + filters.
5. **Output:** ranked NYC records.

## Constraints

- Build ~5:45–8:00. Stop ~7:30 to review MATH.md.
- 3-minute demo must show the Elastic part and the Mistral part explicitly.
- Code runs on the user's laptop. This container can't reach Socrata and has no keys.

## Pitch rules

- "Quantum" = quantum logic (Birkhoff–von Neumann 1936). One-line answer at the top of MATH.md, used only when asked.
- Never say unprompted: qubits, quantum computing, entanglement, superposition, collapse, speedup.

## Open (to fill in while building, not before)

- Datasets and which text field gets embedded
- Expression grammar Mistral emits
- Demo queries (each needs a visible q·r overlap — MATH.md §3)
- Evaluation: leak@10 / keep@10 vs. `must_not` (MATH.md §8)
