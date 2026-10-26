"""K-means (numpy, k-means++) over combined vectors; writes cluster_id back."""
from __future__ import annotations
import numpy as np
from .db import DB, _vec

def cluster(db: DB, k: int = 3, seed: int = 0, iters: int = 50):
    rows = db.rows()
    if len(rows) < k: raise SystemExit(f"need >= {k} chains, have {len(rows)}")
    X = np.stack([_vec(r["combined_vec"]) for r in rows]).astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    rng = np.random.default_rng(seed)
    cents = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(cents)[None])**2).sum(-1), axis=1)
        cents.append(X[rng.choice(len(X), p=d2/d2.sum())])
    C = np.array(cents)
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        labels = ((X @ C.T)).argmax(1)  # cosine assignment
        new = np.stack([X[labels == j].mean(0) if (labels == j).any() else C[j] for j in range(k)])
        new /= np.linalg.norm(new, axis=1, keepdims=True) + 1e-9
        if np.allclose(new, C): break
        C = new
    for r, lab in zip(rows, labels):
        db.con.execute("UPDATE chains SET cluster_id=? WHERE uid=?", (int(lab), r["uid"]))
    db.con.commit()
    return {int(j): [rows[i]["uid"] for i in range(len(rows)) if labels[i] == j] for j in range(k)}
