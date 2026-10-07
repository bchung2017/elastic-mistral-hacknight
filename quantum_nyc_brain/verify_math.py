"""Numerical checks for every claim in MATH.md. Run: python verify_math.py"""
import numpy as np

from qlogic import orth, score

rng = np.random.default_rng(0)
D = 1024


def unit(v):
    return v / np.linalg.norm(v)


def proj(*vecs):
    """Projector onto span(vecs), via QR (Gram-Schmidt)."""
    Q, _ = np.linalg.qr(np.stack(vecs, axis=1))
    return Q @ Q.T


def at_angle(a, theta):
    """Unit vector at angle theta from unit vector a."""
    w = unit(rng.normal(size=D))
    w = unit(w - (w @ a) * a)
    return np.cos(theta) * a + np.sin(theta) * w


I = np.eye(D)
q, r = unit(rng.normal(size=D)), unit(rng.normal(size=D))
q = unit(q + 0.8 * r)  # query with a real component along r

# 1. NOT r: q' = (I - r r^T) q is orthogonal to r
qp = (I - np.outer(r, r)) @ q
assert abs(qp @ r) < 1e-12

# 2. Score decomposition: d . q' = d_perp . q_perp (the r-component of every doc is ignored)
d = unit(rng.normal(size=D) + 2 * r)
d_perp = d - (d @ r) * r
assert np.isclose(d @ qp, d_perp @ qp)
assert np.isclose(d @ qp, d @ q - (r @ q) * (d @ r))

# 3. If q is already orthogonal to r, NOT r does nothing
q0 = unit(q - (q @ r) * r)
assert np.allclose((I - np.outer(r, r)) @ q0, q0)

# 4. Commutator norm ||[P_a, P_b]|| = |cos t| sin t, max 1/2 at 45 degrees
a = unit(rng.normal(size=D))
for theta in [0.1, np.pi / 6, np.pi / 4, np.pi / 3, 1.4]:
    b = at_angle(a, theta)
    Pa, Pb = np.outer(a, a), np.outer(b, b)
    C = Pa @ Pb - Pb @ Pa
    assert np.isclose(np.linalg.norm(C, 2), abs(np.cos(theta)) * np.sin(theta))
    # and the two orders of negation differ by exactly that commutator
    assert np.allclose((I - Pa) @ (I - Pb) - (I - Pb) @ (I - Pa), C)

# 5. Sequential NOT a, NOT b re-introduces an a-component unless a is orthogonal to b;
#    projecting out span{a,b} removes both
b = at_angle(a, np.pi / 3)
qa = unit(q + 0.5 * a + 0.5 * b)
seq = (I - np.outer(b, b)) @ (I - np.outer(a, a)) @ qa
assert abs(seq @ a) > 1e-3 and abs(seq @ b) < 1e-12
joint = (I - proj(a, b)) @ qa
assert abs(joint @ a) < 1e-12 and abs(joint @ b) < 1e-12
assert np.isclose(seq @ a, -(b @ ((I - np.outer(a, a)) @ qa)) * (a @ b))

# 6. Naive subtraction q - lam*r with lam > q.r overshoots into anti-r
over = q - 2.0 * (q @ r) * r
assert over @ r < 0

# 7. Born: (d.q)^2 is sign-blind -- an anti-aligned doc scores like an aligned one
assert np.isclose((q @ q) ** 2, ((-q) @ q) ** 2)

# 8. Real amplitudes still interfere: |<d|aq1+bq2>|^2 has a signed cross term
q1, q2 = unit(rng.normal(size=D)), unit(rng.normal(size=D))
al, be = 0.6, 0.8
lhs = (d @ (al * q1 + be * q2)) ** 2
rhs = al**2 * (d @ q1) ** 2 + be**2 * (d @ q2) ** 2 + 2 * al * be * (d @ q1) * (d @ q2)
assert np.isclose(lhs, rhs)

# 9. Non-distributivity needs a in span(b, c): a AND (b OR c) = a, (a AND b) OR (a AND c) = 0
#    In R^2 with three distinct lines this is exact; check via ranks.
a2, b2, c2 = unit(np.array([1.0, 0])), unit(np.array([1.0, 1])), unit(np.array([1.0, -1]))
span_bc = np.linalg.matrix_rank(np.stack([b2, c2]))
assert span_bc == 2  # b OR c = whole plane, so a AND (b OR c) = a (nonzero)
assert np.linalg.matrix_rank(np.stack([a2, b2])) == 2  # a, b distinct lines -> a AND b = {0}

# 10. Anisotropy: a shared mean direction inflates all cosines; centering removes it
mu = unit(rng.normal(size=D))
X = np.stack([unit(4 * mu + 3 * rng.normal(size=D) / np.sqrt(D)) for _ in range(500)])
raw_cos = (X @ X.T)[np.triu_indices(500, 1)].mean()
Xc = X - X.mean(0)
Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
cen_cos = (Xc @ Xc.T)[np.triu_indices(500, 1)].mean()
assert raw_cos > 0.5 and abs(cen_cos) < 0.01

# 11. Elasticsearch dot_product score s = (1 + dot) / 2  ->  dot = 2s - 1
s = (1 + d @ q) / 2
assert np.isclose(2 * s - 1, d @ q)

# 12. Concept subspace from paraphrases: rank-k projector removes all of them better than the mean vector
base = unit(rng.normal(size=D))
paras = np.stack([unit(base + 0.7 * unit(rng.normal(size=D))) for _ in range(6)])
U, S, _ = np.linalg.svd(paras.T, full_matrices=False)
Pk = U[:, :4] @ U[:, :4].T
m = unit(paras.mean(0))
qq = unit(rng.normal(size=D) + paras.sum(0))
res_sub = np.abs(paras @ ((I - Pk) @ qq)).mean()
res_mean = np.abs(paras @ ((I - np.outer(m, m)) @ qq)).mean()
assert res_sub < res_mean
# ---- Query semantics (MATH.md §7), reference implementation in qlogic.py ----
n1 = unit(rng.normal(size=D))
c1 = unit(unit(rng.normal(size=D)) + 1.5 * n1)  # OR concept entangled with the negated concept
c2 = unit(rng.normal(size=D))

# 13. Script score 1 + s is non-negative for s in [-1, 1] and order-preserving
ss = np.array([-1.0, -0.3, 0.0, 0.4, 1.0])
assert np.all(1 + ss >= 0) and np.all(np.diff(1 + ss) > 0)

# 14. Without projecting OR concepts through NOT, OR re-admits the negated concept;
#     with it, a doc purely along n1 scores 0
U_raw = orth([c1, c2])
assert np.sum((U_raw.T @ n1) ** 2) > 0.1
assert np.isclose(score(n1, not_=[n1], any_of=[c1, c2]), 0.0)
assert np.allclose(orth([(I - np.outer(n1, n1)) @ c for c in (c1, c2)]).T @ n1, 0)

# 15. Ungated subspace score is sign-blind; gated score zeroes the anti-aligned doc
d_al = unit(c1 + c2)
U2 = orth([c1, c2])
assert np.isclose(np.sum((U2.T @ d_al) ** 2), np.sum((U2.T @ -d_al) ** 2))
assert score(d_al, any_of=[c1, c2]) > 0.5 and score(-d_al, any_of=[c1, c2]) == 0.0
# max-gate rejects a doc aligned with c1 but strongly anti-aligned with c2 only if no concept is positive;
# a doc anti-aligned with every concept is rejected
d_mixed = unit(1.0 * c1 - 1.5 * c2)
assert (c1 @ d_mixed) > 0 and score(d_mixed, any_of=[c1, c2]) > 0  # max-gate passes: it IS aligned with c1
assert score(unit(-c1 - c2), any_of=[c1, c2]) == 0.0

# 15b. All any_of concepts swallowed by NOT: fall back to positive-only, or raise if no positive
assert np.isclose(score(d, positive=q, not_=[n1], any_of=[n1]), score(d, positive=q, not_=[n1]))
try:
    score(d, not_=[n1], any_of=[n1])
    raise AssertionError("expected ValueError")
except ValueError:
    pass

# 16. Combined score lies in [0, 1]; pure-positive score lies in [-1, 1]
for _ in range(200):
    dd = unit(rng.normal(size=D))
    sc = score(dd, positive=q, not_=[n1], any_of=[c1, c2])
    assert 0.0 <= sc <= 1.0
    assert -1.0 <= score(dd, positive=q, not_=[n1]) <= 1.0

print(f"all checks pass | raw mean cos {raw_cos:.2f} -> centered {cen_cos:.3f} | "
      f"paraphrase leakage: mean-vector {res_mean:.3f} vs rank-4 subspace {res_sub:.3f}")
