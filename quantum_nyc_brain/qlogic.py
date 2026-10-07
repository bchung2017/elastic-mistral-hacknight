"""Quantum-logic query compilation and scoring (MATH.md §7–8).

compile_query turns concept vectors (centered, unit) into the params the Painless SCRIPT takes.
score is the numpy reference the Painless must match.
"""
import numpy as np

DROP = 0.1


def orth(vecs, eps=DROP):
    """Orthonormal basis (columns) of span(vecs), dropping vectors with norm < eps."""
    vecs = [v for v in vecs if np.linalg.norm(v) >= eps]
    if not vecs:
        return None
    Q, _ = np.linalg.qr(np.stack(vecs, axis=1))
    return Q


def compile_query(positive=None, not_=(), any_of=()):
    """Return (q, u, c): q′ or None, OR basis vectors, surviving c̃ᵢ for the gate."""
    if positive is None and not any_of:
        raise ValueError("query needs a positive term or any_of")
    N = orth(list(not_))

    def qn(v):
        return v if N is None else v - N @ (N.T @ v)

    q = None
    if positive is not None:
        qp = qn(positive)
        q = qp / np.linalg.norm(qp)
    c = [x for x in (qn(v) for v in any_of) if np.linalg.norm(x) >= DROP]
    if any_of and not c:
        if q is None:
            raise ValueError("every any_of concept lies inside the NOT subspace; nothing left to score")
        print("warning: every any_of concept dropped by NOT; scoring positive only")
    U = orth(c)
    u = [] if U is None else list(U.T)
    return q, u, c


def score(d, positive=None, not_=(), any_of=()):
    q, u, c = compile_query(positive, not_, any_of)
    s_pos = None if q is None else float(d @ q)
    s_or = None
    if u:
        s_or = float(sum((ui @ d) ** 2 for ui in u)) if max(ci @ d for ci in c) > 0 else 0.0
    if s_pos is not None and s_or is not None:
        return max(0.0, s_pos) * s_or
    return s_pos if s_pos is not None else s_or


def script_params(q, u, c):
    return {
        "q": None if q is None else q.tolist(),
        "u": [x.tolist() for x in u],
        "c": [x.tolist() for x in c],
    }


# Returns 1 + s: Elasticsearch rejects negative script scores. Manual dot products on purpose (MATH.md §8).
SCRIPT = """
float[] v = doc['vec'].vectorValue;
double pos = 0;
if (params.q != null) { for (int j = 0; j < v.length; j++) pos += v[j] * params.q[j]; }
double orS = 0;
for (int i = 0; i < params.u.size(); i++) {
  double t = 0; for (int j = 0; j < v.length; j++) t += v[j] * params.u[i][j];
  orS += t * t;
}
double gate = -1e9;
for (int i = 0; i < params.c.size(); i++) {
  double t = 0; for (int j = 0; j < v.length; j++) t += v[j] * params.c[i][j];
  gate = Math.max(gate, t);
}
if (params.c.size() > 0 && gate <= 0) orS = 0;
double s;
if (params.q != null && params.u.size() > 0) s = Math.max(0, pos) * orS;
else if (params.q != null) s = pos;
else s = orS;
return 1.0 + s;
"""
