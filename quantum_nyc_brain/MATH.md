# Quantum NYC Brain — the math

Every numbered claim below is checked numerically in `verify_math.py` (check number in brackets).

**What this is:** quantum *logic* (Birkhoff & von Neumann, 1936) applied to retrieval (Widdows & Peters 2003, "Word Vectors and Quantum Logic"; van Rijsbergen 2004, *The Geometry of Information Retrieval*).
**What this is not:** quantum computing. No qubits, no gates, no speedup. The quantum-computing "NOT" is the Pauli-X gate (a unitary bit flip) — unrelated to the negation used here.

One-line answer if asked: *"Quantum logic: negation as orthogonal complement in Hilbert space, Birkhoff–von Neumann 1936, applied to Mistral embeddings in Elasticsearch."*

---

## 1. States

A document or query is embedded by `mistral-embed` into e ∈ ℝ¹⁰²⁴.

State preparation:

    ψ = (e − μ) / ‖e − μ‖        μ = mean embedding over the corpus

- **Normalize** because states are unit vectors, and because Elasticsearch `dot_product` similarity requires unit-length vectors.
- **Center** (subtract μ) because embedding spaces are typically *anisotropic*: every vector shares a large common direction, so all pairwise cosines are high (often 0.5+) and carry little information. With that shared direction left in, projecting out a concept r also strips part of the common direction from the query, which distorts it. Centering removes it [10]. Precedent: Mu & Viswanath 2018, "All-but-the-Top" (remove mean plus top principal components). **Unverified for mistral-embed specifically** — measure mean pairwise cosine on the real corpus before and after centering. Apply the same μ to queries and concept vectors.

The 1024 = 2¹⁰ coincidence: any unit vector in ℝ¹⁰²⁴ *is* a valid real-amplitude 10-qubit state, but nothing below uses the tensor-product (qubit) structure. Everything works identically in 1000 dimensions. Fun fact at most; never present it as mechanism.

## 2. Propositions are subspaces; projectors are their operators

A concept r (unit vector) defines a 1-D subspace with projector

    P_r = r rᵀ          P² = P,  Pᵀ = P

A concept described by several vectors (paraphrases) defines a k-D subspace spanned by them. Orthonormalize (Gram–Schmidt / QR) into columns U, then P = U Uᵀ.

Birkhoff–von Neumann lattice operations:

| Logic | Subspace | Projector |
|---|---|---|
| NOT A | orthogonal complement A⊥ | I − P_A |
| A OR B | closed span of A and B | QR of [A B] → U Uᵀ |
| A AND B | intersection A ∩ B | (limit of alternating projections) |

**Usable for retrieval: NOT and OR.** AND of two 1-D concepts is the zero subspace unless they're identical, so it's useless here. Implement "A and B" with ordinary score combination (Elasticsearch bool/rescoring), not quantum AND.

## 3. Negation: the core operator

    NOT r applied to q:   q′ = (I − r rᵀ) q = q − (r·q) r,   then renormalize

Properties:
- q′ ⊥ r exactly [1].
- Score decomposition [2]: for any document d,

      d · q′ = d · q − (r·q)(d·r) = d_⊥ · q′        where d_⊥ = d − (d·r) r

  **Each document's component along r is ignored, not penalized.** Documents are scored only on what remains after removing r. A doc that is mostly r has a small d_⊥ and drops; a doc that mentions r in passing but is mostly about q keeps its score.
- If the query has no r-component (q·r = 0), NOT r does nothing [3]. Negation acts on the query's *confound direction*. "rats NOT restaurants" works *because* "rats" carries a restaurant component (rodent violations). Choose demo queries where q·r is visibly nonzero, and print q·r.
- Naive subtraction q − λr: with λ > q·r it overshoots into anti-r [6] and retrieves documents that are actively opposite to r. Orthogonal projection is the unique λ = q·r that removes exactly the r-component — that's the principled argument (Widdows).

Contrast with keyword `must_not`:
- misses paraphrases ("mouse droppings in the kitchen" contains no excluded word)
- over-excludes documents that mention the word incidentally

## 4. Negating several concepts

"Neither a nor b" = complement of span{a, b}:

    U = QR([a b]),   q′ = q − U(Uᵀ q)

**Sequential negation is not the same thing.** NOT a then NOT b gives (I−P_b)(I−P_a)q, which reintroduces an a-component whenever a·b ≠ 0 [5]:

    a · (I−P_b)(I−P_a)q = −(b · (I−P_a)q)(a·b)

Use the subspace projector for multi-concept negation.

### Concept subspaces from Mistral paraphrases (practical upgrade)

A single embedding of "restaurant" is a thin target. Instead, ask Mistral (structured output) for 6–10 paraphrases or instances ("deli", "pizza place", "food truck", "bodega with a grill"…), embed and center them, take the top-k left singular vectors (SVD) as U. This removes the concept more completely than projecting out the mean paraphrase vector [12]. It also gives Mistral a real role inside the quantum piece. k = 3–5; inspect the singular values and drop the tail.

## 5. Non-commutativity: the order effect

Projectors for non-orthogonal, non-identical concepts don't commute:

    [P_a, P_b] = P_a P_b − P_b P_a = (a·b)(a bᵀ − b aᵀ)

    ‖[P_a, P_b]‖ = |cos θ| · sin θ        θ = angle between a and b   [4]

- zero at θ = 0 (same concept) and θ = 90° (unrelated concepts)
- maximum ½ at θ = 45°

The two orders of negation differ by exactly this commutator [4]:

    (I−P_a)(I−P_b) − (I−P_b)(I−P_a) = [P_a, P_b]

Demo: pick two related concepts (centered cosine ~0.4–0.7), run NOT a then NOT b vs. the reverse, and show the top-10 changing. **Expect the effect to be small** unless the pair is near 45°. Print θ and the commutator norm next to the results so the size is explained, not hidden. This is the formal reason the quantum-logic lattice is non-distributive. Classic example: three distinct lines a, b, c in a plane. a AND (b OR c) = a AND plane = a, but (a AND b) OR (a AND c) = 0 OR 0 = 0 [9]. Curiosity — don't demo distributivity.

## 6. Born rule — and where it breaks

Interpretation: P(d | q) = ⟨d|q⟩² = cos²(d, q).

- **Sign-blind [7].** An anti-aligned document (cos = −0.9) gets the same "probability" as an aligned one (cos = +0.9). After centering, negative cosines are common. **Rank by signed dot product.** Use cos² only as a labeled interpretation, never for ranking.
- **Interference exists, but it's trivial [8].** Real amplitudes do interfere: ⟨d| αq₁ + βq₂⟩² has the signed cross term 2αβ⟨d|q₁⟩⟨d|q₂⟩. That's just squaring a linear combination. Don't sell it.
- **No complex phase.** Real Hilbert space only. Phenomena that need complex phase have no counterpart here.
- **No measurement dynamics.** Nothing collapses. Projection is applied to the query by choice, not by observation.

Words to avoid unprompted: qubit, quantum computing, entanglement, superposition of answers, collapse, speedup.

## 7. Elasticsearch implementation

Mapping:

```json
"vec": {
  "type": "dense_vector",
  "dims": 1024,
  "similarity": "dot_product",
  "index_options": { "type": "hnsw" }
}
```

- Set `index_options` explicitly. Recent versions default to a quantized index type for large dims (I believe int8 or BBQ, depending on version — unverified). Quantization makes scores approximate, which blurs small effects like the order effect. A corpus of ~2k docs doesn't need it. Alternative: exact brute force with `script_score` and `dotProduct(params.q, 'vec')`.
- Store centered and normalized vectors. Keep μ client-side (`.npy`), or as a doc in a `_meta` index.
- All projection happens client-side in numpy. Elasticsearch receives the final unit `query_vector`:

```json
"knn": { "field": "vec", "query_vector": [...], "k": 10, "num_candidates": 200 }
```

- Score conversion for float vectors: `_score = (1 + dot) / 2`, so dot = 2·_score − 1 [11].

## 8. Evaluation (what makes "untested" into a number)

Methods compared on the same queries:
1. baseline kNN on q
2. kNN on q + keyword `must_not`
3. kNN on q − 1.0·r (naive subtraction)
4. kNN on projected q′ (single vector)
5. kNN on projected q′ (Mistral paraphrase subspace)

Metrics for each, over top-10:
- **leak@10:** fraction of results about the negated concept (Mistral as judge, structured yes/no)
- **keep@10:** fraction still about the positive concept

Projection is worth claiming only if it lowers leak@10 versus `must_not` while holding keep@10. If it doesn't, say so and present it as an exact-geometry curiosity.

## 9. One-slide version

- Documents and queries → Mistral embeddings → centered unit vectors in ℝ¹⁰²⁴ (states).
- A concept is a subspace. NOT is the orthogonal complement: q′ = q − U Uᵀ q (Birkhoff–von Neumann 1936).
- Elasticsearch kNN scores ⟨d|q′⟩. Every document's component along the negated concept is ignored exactly.
- Negations don't commute: ‖[P_a, P_b]‖ = |cos θ| sin θ. Show both orders.
- Result: leak@10 and keep@10 vs. `must_not`.
