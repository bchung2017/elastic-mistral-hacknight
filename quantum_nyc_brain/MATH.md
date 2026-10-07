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

- **Sign-blind [7].** An anti-aligned document (cos = −0.9) gets the same "probability" as an aligned one (cos = +0.9). After centering, negative cosines are common and mean "unlike the concept." **Never rank by a squared overlap without a sign gate.** This applies to OR (§7) as well.
- **Interference exists, but it's trivial [8].** Real amplitudes do interfere: ⟨d| αq₁ + βq₂⟩² has the signed cross term 2αβ⟨d|q₁⟩⟨d|q₂⟩. That's just squaring a linear combination. Don't sell it.
- **No complex phase.** Real Hilbert space only. Phenomena that need complex phase have no counterpart here.
- **No measurement dynamics.** Nothing collapses. Projection is applied to the query by choice, not by observation.

Words to avoid unprompted: qubit, quantum computing, entanglement, superposition of answers, collapse, speedup.

## 7. Query semantics — the scoring rule

Grammar (flat, emitted by Mistral structured output): `{positive?, not: [...], any_of: [...]}`. At least one of `positive` / `any_of` must be present. All vectors are centered and unit-normalized (§1); d is a document state.

**Step 1 — NOT applies to everything.**
N = orth(not concepts) via QR, Q_N = I − N Nᵀ. Both the positive query and every `any_of` concept pass through Q_N. Otherwise an OR concept entangled with a negated one re-admits it [14].

**Step 2 — positive term.**

    q′ = Q_N q / ‖Q_N q‖          s_pos(d) = d · q′ ∈ [−1, 1]

**Step 3 — OR term (gated subspace membership).**

    c̃ᵢ = Q_N cᵢ     (drop any with ‖c̃ᵢ‖ < 0.1: that concept was mostly the negated one)
    U = orth(c̃ᵢ)    (U ⊥ N automatically, so d's negated component is ignored)
    s_or(d) = ‖Uᵀd‖²   if Σᵢ c̃ᵢ · d > 0,   else 0        ∈ [0, 1]

‖Uᵀd‖² is the Born probability that d lies in the OR-subspace. That's the right quantity for OR, but it's sign-blind [15], so the gate zeroes documents on the wrong side of the concepts. Known gap: a doc strongly aligned with c̃₁ and anti-aligned with c̃₂ can pass the gate if the sum is positive. Acceptable for MVP.

Alternative if the subspace story isn't needed: signed soft-OR `max_i (c̃ᵢ · d)`. That's the fuzzy-logic (Gödel) OR, not quantum logic. Simpler, and correct on sign.

**Step 4 — combine.**

| Present | Score s |
|---|---|
| positive only | s_pos ∈ [−1, 1] |
| any_of only | s_or ∈ [0, 1] |
| both | max(0, s_pos) · s_or ∈ [0, 1] (soft AND, a product of probabilities) |

Quantum AND isn't usable (§2), so the conjunction is classical. Say so if asked. [16] checks the ranges. `score()` in `verify_math.py` is the reference implementation.

## 8. Elasticsearch implementation

Retrieval is **exact brute force with `script_score`**, not kNN. At ~5k docs it's cheap; it's exact (no HNSW approximation or quantization blurring small effects), and OR can't be expressed as one kNN vector anyway.

Mapping (no ANN index needed):

```json
"vec":  { "type": "dense_vector", "dims": 1024, "index": false },
"note": { "type": "text" }
```

Client-side, in numpy: centering, Q_N, q′, c̃ᵢ, and U. Elasticsearch receives the final vectors as params.

**Script score must be ≥ 0.** Elasticsearch rejects negative `script_score` results, and s_pos is often negative on centered vectors. Return `1.0 + s`. That's monotone, so ranking is unchanged [13]. Convert back client-side: s = _score − 1.

**Don't loop `dotProduct(params.u[i], 'vec')`.** I believe Painless vector functions are bound per call site to the first query vector they see, so a loop would silently score against u[0] every time. Not verified. The safe pattern is a manual dot product over `doc['vec'].vectorValue`:

```painless
float[] v = doc['vec'].vectorValue;
double pos = 0;
if (params.q != null) { for (int j = 0; j < v.length; j++) pos += v[j] * params.q[j]; }
double orS = 0, gate = 0;
for (int i = 0; i < params.u.size(); i++) {
  double t = 0; for (int j = 0; j < v.length; j++) t += v[j] * params.u[i][j];
  orS += t * t;
}
for (int i = 0; i < params.c.size(); i++) {
  for (int j = 0; j < v.length; j++) gate += v[j] * params.c[i][j];
}
if (params.c.size() > 0 && gate <= 0) orS = 0;
double s;
if (params.q != null && params.u.size() > 0) s = Math.max(0, pos) * orS;
else if (params.q != null) s = pos;
else s = orS;
return 1.0 + s;
```

Params: `q` (q′, or null), `u` (basis U as a list of vectors), `c` (the c̃ᵢ, for the gate). Write the Painless to match `score()` and test it against `score()` on ~20 docs before trusting it.

**Show Elastic explicitly in the demo:** send the keyword baseline (`bool` + `must_not` on `note`) and the quantum `script_score` query in **one `_msearch`**, and display both result lists side by side.

## 9. Evaluation (what makes "untested" into a number)

Methods compared on the same queries, all scored exactly:
1. baseline: s_pos with no NOT (q itself)
2. keyword: `bool` match on the positive text + `must_not` on the negated words
3. naive subtraction: q − 1.0·r
4. projection, single vector per concept (§7)
5. projection, Mistral paraphrase subspace per concept (§4), if time allows

Metrics for each, over top-10:
- **leak@10:** fraction of results about a negated concept (Mistral as judge, structured yes/no)
- **keep@10:** fraction still about the positive concept

Projection is worth claiming only if it lowers leak@10 versus `must_not` while holding keep@10. If it doesn't, say so and present it as an exact-geometry curiosity.

## 10. One-slide version

- Documents and queries → Mistral embeddings → centered unit vectors in ℝ¹⁰²⁴ (states).
- A concept is a subspace. NOT is the orthogonal complement: q′ = q − N Nᵀ q (Birkhoff–von Neumann 1936). It applies to every other term.
- OR is gated subspace membership: ‖Uᵀd‖², zeroed on the wrong side.
- Elasticsearch scores every document exactly (`script_score`), side by side with a `must_not` baseline in one `_msearch`.
- Negations don't commute: ‖[P_a, P_b]‖ = |cos θ| sin θ. Show both orders.
- Result: leak@10 and keep@10 vs. `must_not`.
