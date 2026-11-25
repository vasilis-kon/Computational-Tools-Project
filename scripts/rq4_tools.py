"""Shared helpers for the RQ4_pipeline notebooks (k-means vs Louvain).

Nothing here is new method code: spherical k-means is the one used in RQ4_kmeans / RQ4_extension,
the Louvain code is in scripts/louvain.py (from notebooks/04_louvain_sound_graph.ipynb), and the
stream rule is scripts/streams.py. This module only removes copy-paste between the notebooks.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from scipy.sparse import csr_matrix


def find_root(start=None):
    """The project folder: the first parent that contains data/raw/yambda."""
    root = Path.cwd() if start is None else Path(start)
    while not (root / "data" / "raw" / "yambda").exists():
        if root.parent == root:
            raise FileNotFoundError("Could not find data/raw/yambda above " + str(Path.cwd()))
        root = root.parent
    return root


# ----------------------------------------------------------------------------- k-means (Method A)
def spherical_kmeans(X, k, seed, n_iter=60):
    """Spherical k-means with k-means++ seeding, identical to RQ4_kmeans / RQ4_extension.

    X must have unit-length rows. Returns (centroids, labels, mean cosine to own centroid).
    """
    r = np.random.default_rng(seed); n = len(X)
    C = np.empty((k, X.shape[1]), dtype=np.float32); C[0] = X[r.integers(n)]
    dist = 1 - X @ C[0]
    for j in range(1, k):
        pr = np.maximum(dist, 0) ** 2; C[j] = X[r.choice(n, p=pr / pr.sum())]
        dist = np.minimum(dist, 1 - X @ C[j])
    labels = np.full(n, -1)
    for _ in range(n_iter):
        new = np.argmax(X @ C.T, axis=1)
        if (new == labels).mean() > 0.9995:
            labels = new; break
        labels = new
        sums = csr_matrix((np.ones(n, np.float32), (labels, np.arange(n))), shape=(k, n)) @ X
        nrm = np.linalg.norm(sums, axis=1, keepdims=True)
        empty = nrm[:, 0] == 0
        sums[empty] = X[r.integers(n, size=empty.sum())]; nrm[empty] = 1
        C = (sums / np.maximum(nrm, 1e-9)).astype(np.float32)
    return C, labels, float((X * C[labels]).sum(1).mean())


def assign_nearest(X, C, block=200_000):
    """Label of the centroid with the highest cosine for every row of X (unit-length rows)."""
    out = np.empty(len(X), dtype=np.int16)
    for s in range(0, len(X), block):
        out[s:s + block] = np.argmax(np.asarray(X[s:s + block]) @ C.T, axis=1)
    return out


def kmeans_grid(EN, k_grid, n_seeds, n_fit, seed):
    """Fit spherical k-means on n_fit random rows for every k and seed, assign every row of EN.

    Same procedure as RQ4_extension: centroids from a 150,000-track sample, starts SEED + 100 * s.
    Returns {(k, s): labels of all rows of EN}.
    """
    rng = np.random.default_rng(seed)
    fit_idx = rng.choice(len(EN), n_fit, replace=False)
    Xf = np.asarray(EN[np.sort(fit_idx)])
    out = {}
    for k in k_grid:
        for s in range(n_seeds):
            C, _, _ = spherical_kmeans(Xf, k, seed + 100 * s)
            out[(k, s)] = assign_nearest(EN, C)
    return out


# ----------------------------------------------------------------------------- comparing partitions
def contingency(a, b):
    """Contingency table of two labelings (rows: labels of a, columns: labels of b)."""
    _, a = np.unique(a, return_inverse=True); _, b = np.unique(b, return_inverse=True)
    t = np.zeros((a.max() + 1, b.max() + 1)); np.add.at(t, (a, b), 1)
    return t


def nmi(a, b):
    """Normalised mutual information (arithmetic mean normalisation, as scikit-learn's default)."""
    t = contingency(a, b); n = t.sum(); p = t / n
    pa, pb = p.sum(1), p.sum(0)
    nz = p > 0
    mi = (p[nz] * np.log(p[nz] / np.outer(pa, pb)[nz])).sum()
    ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum(); hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return float(mi / ((ha + hb) / 2)) if ha + hb > 0 else 1.0


def pair_overlap(a, b):
    """Pair counting: P(same in b | same in a) and P(same in a | same in b).

    High first / low second means b splits a's groups less than a splits b's, i.e. b is coarser.
    """
    t = contingency(a, b); c2 = lambda x: x * (x - 1) / 2
    both = c2(t).sum(); same_a = c2(t.sum(1)).sum(); same_b = c2(t.sum(0)).sum()
    return float(both / same_a), float(both / same_b)


def effective_number(labels):
    """exp(entropy) of the cluster sizes: the number of equal-sized clusters with the same entropy."""
    p = np.bincount(labels) / len(labels); p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


# ----------------------------------------------------------------------------- data
def item_map(path, col, L):
    """item_id -> first artist / album id (-1 if unknown), as an array of length L."""
    mp = pd.read_parquet(path).drop_duplicates("item_id"); arr = np.full(L, -1, dtype=np.int64)
    ids = mp["item_id"].to_numpy(); ok = ids < L; arr[ids[ok]] = mp[col].to_numpy()[ok]
    return arr


def read_listens(data_dir, real_pct, stream_mode, warmup_days, stream_labels):
    """All listens of Yambda-50M with the shared stream label, then real listens only.

    Same code as the RQ4_extension notebooks. Returns uid, item, ts (int32) and the stream label
    (1 organic, 0 recommended, -1 excluded) of every real listen, sorted by user, then time.
    """
    tbl = pq.read_table(data_dir / "flat/50m/multi_event.parquet",
                        columns=["uid", "item_id", "timestamp", "is_organic", "played_ratio_pct"],
                        filters=[("event_type", "=", "listen")])
    U_, I_, T_ = tbl["uid"].to_numpy(), tbl["item_id"].to_numpy(), tbl["timestamp"].to_numpy()
    O_, R_ = tbl["is_organic"].to_numpy().astype(np.int8), tbl["played_ratio_pct"].to_numpy(); del tbl
    if not np.all((U_[1:] > U_[:-1]) | ((U_[1:] == U_[:-1]) & (T_[1:] >= T_[:-1]))):
        order = np.lexsort((T_, U_)); U_, I_, T_, O_, R_ = U_[order], I_[order], T_[order], O_[order], R_[order]; del order
    UIDX_ = np.r_[0, np.cumsum(U_[1:] != U_[:-1])].astype(np.int32)
    LAB = stream_labels(UIDX_, I_, O_, R_, T_, real_pct, stream_mode, warmup_days); del UIDX_, O_
    real = R_ >= real_pct; del R_
    out = U_[real].astype(np.int32), I_[real].astype(np.int32), T_[real].astype(np.int32), LAB[real]
    return out


def graph_from_csr(A):
    """scripts/louvain.Graph from a symmetric scipy sparse matrix (no self-loops expected)."""
    from louvain import Graph
    A = sparse.csr_matrix(A)
    adj = []
    for i in range(A.shape[0]):
        a, b = A.indptr[i], A.indptr[i + 1]
        adj.append(dict(zip(A.indices[a:b].tolist(), A.data[a:b].astype(float).tolist())))
    return Graph(adj)


def modularity_sparse(A, labels):
    """Modularity (gamma = 1) of a partition on a symmetric sparse matrix, as in RQ4_extension/01."""
    _, lab = np.unique(labels, return_inverse=True); k = lab.max() + 1
    M = csr_matrix((np.ones(len(lab)), (np.arange(len(lab)), lab)), shape=(len(lab), k))
    B = (M.T @ A @ M).toarray(); tot = B.sum()
    return float(np.trace(B) / tot - ((B.sum(1) / tot) ** 2).sum())
