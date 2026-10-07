# Quantum NYC Brain — project reference

## Pitch

**Title:** Quantum NYC Brain.

**What it is, plainly:** an autonomous ETL agent (Mistral) that onboards unseen NYC Open Data sets into Elasticsearch. It infers mappings, writes ingest pipelines, repairs its own errors, and records what it learned in a catalog. The query layer adds quantum-logic negation (NOT as orthogonal complement) over Mistral embeddings.

**Defensible framings:**
- **Brain:** Complementary Learning Systems (McClelland, McNaughton & O'Reilly 1995). A fast episodic store (raw sample ingest = hippocampus) plus slow structured knowledge (catalog of schemas and join keys = neocortex), linked by consolidation (the agent inferring mappings and rules). The self-repair loop is error-driven learning: bulk error → revise → retry.
- **Quantum:** quantum *logic*, not quantum computing. See MATH.md. Bring it up only when asked; one-line answer at the top of MATH.md.

**Never say unprompted:** qubits, quantum computing, entanglement, superposition, collapse, speedup.

## Hackathon constraints (from the repo README)

- Build ~5:45–8:00. **Stop building ~7:30** to review the math (user's call).
- 3-minute demo; must show the Elastic part (mappings, queries, tools) and the Mistral part explicitly.
- Judged on Novelty, Use of Elastic, Use of Mistral. No weights given. UI worth nothing.
- Elasticsearch Serverless 9.4+. Mistral key from event credits (`MIST-ELAST-NYC`).
- Code runs on the user's laptop. This cloud container's network policy blocks Socrata and has no keys, so nothing here has run against live services.

## Architecture

```
Socrata (NYC Open Data)
   │  metadata + sample rows
   ▼
Mistral agent (function calling, mistral-large-4)
   │  tools: inspect_dataset, create_index, put_pipeline, simulate_pipeline,
   │         bulk_ingest, run_query, record_catalog
   ▼
Elasticsearch
   ├── nyc_<dataset>            one index per onboarded dataset
   ├── nyc_<dataset> pipeline   generated ingest pipeline
   ├── brain_catalog            what the agent learned: fields, types, join keys, notes
   └── squirrels_q (demo)       dense_vector index for quantum-logic negation
```

### ETL agent loop
1. `inspect_dataset(id)`: Socrata column metadata + ~200 sample rows.
2. Mistral emits the mapping and pipeline as **structured output**, restricted to a fixed processor set: `date`, `convert`, `rename`, `lowercase`, `trim`, `remove`, `set`, plus a geo_point builder for lat/lon. Free-form Painless only as a last resort.
3. `simulate_pipeline` on sample docs before any bulk write. This is cheap error detection.
4. Create the index, bulk ingest up to ~2k rows, and collect per-doc errors. **Do not use `raise_on_error=False` silently** like the repo notebooks do; count and return the errors.
5. If errors occur, feed them back to Mistral and retry. **Max 3 attempts** per dataset, then log the failure.
6. `record_catalog`: field roles, detected join keys (zip, BBL, borough, camis, lat/lon), text fields with `semantic_text`.

### Quantum layer
See MATH.md §7–8. Client-side `mistral-embed`, centering, numpy projection, Elasticsearch `knn`.

## Build plan

| Time | Work | Cut if late |
|---|---|---|
| 0:00–0:25 | Quantum operator on squirrel stories (~2k docs) and the 5-method comparison | Drop paraphrase subspace and order effect |
| 0:25–1:20 | ETL agent: tools, loop, repair | — (core) |
| 1:20–1:40 | Catalog + one cross-dataset query; run on 3–5 unseen datasets | Cut first |
| 1:40 | Stop, review MATH.md | — |

Order: quantum first. It's ~20 minutes, settles whether projection helps, and guarantees an opener.

## Demo script (3 min)

1. (20s) `GET _cat/indices` is empty. "This is a brain for New York that's never seen the city."
2. (60s) Give the agent 3 dataset IDs it has never seen. Show the mappings and pipelines it writes, at least one self-repair, and the catalog filling in.
3. (40s) A cross-dataset question that nobody wrote a join for.
4. (40s) Quantum logic: `rats NOT restaurants`, side by side (baseline / `must_not` / projection), with leak@10 numbers. Print q·r.
5. (20s) If time: the order effect, with θ and ‖[P_a,P_b]‖ shown.

## Risks and checks

- **Socrata endpoints (unverified, from memory):** column metadata `https://data.cityofnewyork.us/api/views/<id>.json`, catalog `https://api.us.socrata.com/api/catalog/v1?domains=data.cityofnewyork.us&only=datasets`, rows `https://data.cityofnewyork.us/resource/<id>.json?$limit=N`. The rows endpoint is confirmed by the repo notebooks; curl the other two first.
- **Mistral SDK import:** the repo guide uses `from mistralai.client import Mistral`. Confirm against the installed version.
- **`mistral-large-4`** is a public preview released the day before the event. Rate limits are unknown. Fallback: `mistral-small-latest`.
- **Embedding cost:** cap ~2k docs per dataset. Batch `embeddings.create` calls.
- **Generated pipelines:** the most likely failure point. That's why the processor set is constrained and simulation runs first.
- **Anisotropy:** measure mean pairwise cosine before and after centering on the real corpus.
- **Projection might not beat `must_not`:** then present the exact geometry and order effect as a curiosity, with numbers.

## Datasets known to the repo (ground truth for mappings)

`nyc_restaurant_inspections` (43nn-pn8j), `nyc_squirrel_stories` (gfqj-f768), `nyc_311_feedback` (7ffd-6gs9), 311 service requests (erm2-nwe9), MTA GTFS, iNaturalist CNC, SONYC, tax photos. The hand-written mappings in the repo notebooks are a reference to score the agent's generated mappings against.
