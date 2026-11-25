"""Own Louvain community detection on a kNN graph of embeddings (RQ4, Method B).

Copied unchanged from notebooks/04_louvain_sound_graph.ipynb (Peter's notebook, section 1), where every
function is tested against networkx / scikit-learn (section 2 there). Only change: `cluster` takes the
block size `chunk_mb` as an argument instead of reading the notebook's global KNN_CHUNK_MB.
numpy only; no graph or community-detection library.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np


def preprocess(X: np.ndarray, l2: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Validate raw embeddings and optionally scale every row to unit length.

    Rows with NaN/inf values or with zero length are dropped (they have no direction).
    Returns the cleaned matrix (float64) and a boolean mask of the rows that were kept.
    No centring here: PCA centres the data.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-d matrix, got shape {X.shape}")
    finite = np.isfinite(X).all(axis=1)
    norms = np.linalg.norm(np.where(finite[:, None], X, 0.0), axis=1)
    keep = finite & (norms > 0)
    if (~keep).any():
        print(f"preprocess: dropped {(~finite).sum():,} rows with NaN/inf and {(finite & (norms == 0)).sum():,} zero rows")
    X = X[keep]
    if l2:
        X = X / norms[keep][:, None]
    return X, keep


@dataclass
class PCAResult:
    scores: np.ndarray              # (n, d_max) coordinates on the first d_max components
    components: np.ndarray          # (d_max, D) principal directions (rows)
    mean: np.ndarray                # (D,) column means removed before the SVD
    explained_variance: np.ndarray  # (min(n, D),) variance along every component


def fit_pca(X: np.ndarray, d_max: int) -> PCAResult:
    """PCA via the SVD of the centred data, fitted once at the largest dimension.

    Components are nested, so the scores for any d <= d_max are scores[:, :d].
    Memory: the SVD of an n x D matrix needs about 3 copies of it (fine for n <= 50k, D = 128).
    For much larger n one would accumulate Xc.T @ Xc over row blocks and use np.linalg.eigh instead.
    """
    mean = X.mean(axis=0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    d_max = min(d_max, len(S))
    return PCAResult(scores=U[:, :d_max] * S[:d_max], components=Vt[:d_max], mean=mean,
                     explained_variance=S ** 2 / (len(X) - 1))


def parallel_analysis(X: np.ndarray, n_shuffles: int, seed: int, quantile: float = 0.95) -> tuple[np.ndarray, int]:
    """Horn's parallel analysis: which components carry more variance than chance?

    Every column is permuted independently (this keeps each column's distribution but destroys the
    correlations), the eigenvalues are recomputed, and a component counts as real if its eigenvalue is
    above the `quantile` of the shuffled eigenvalues at the same rank. Returns (thresholds, suggested d).
    """
    r = np.random.default_rng(seed)
    real = fit_pca(X, X.shape[1]).explained_variance
    shuffled = np.empty((n_shuffles, len(real)))
    for s in range(n_shuffles):
        Xs = np.empty_like(X)
        for col in range(X.shape[1]):
            Xs[:, col] = r.permutation(X[:, col])
        shuffled[s] = fit_pca(Xs, X.shape[1]).explained_variance
    thresholds = np.quantile(shuffled, quantile, axis=0)
    above = real > thresholds
    suggested = int(np.argmin(above)) if not above.all() else len(real)   # first rank that is not above chance
    return thresholds, suggested


def l2_normalize(Z: np.ndarray) -> np.ndarray:
    """Scale every row to unit length; rows of length 0 stay 0. Afterwards dot product = cosine similarity."""
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    return np.divide(Z, norms, out=np.zeros_like(Z, dtype=np.float64), where=norms > 0)


def knn_cosine(Zn: np.ndarray, k: int, chunk_mb: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Exact k nearest neighbours by cosine similarity (rows of Zn must have unit length), self excluded.

    Brute force: all n x n similarities, O(n^2 * d) time. To bound memory, the similarity matrix is
    computed in blocks of rows, each block at most `chunk_mb` MB. Neighbours are sorted by decreasing
    similarity, ties broken by the smaller index, so the result is deterministic.
    Returns (idx, sims), both of shape (n, k).
    """
    n = len(Zn)
    if k >= n:
        print(f"knn_cosine: k = {k} >= n = {n}, using k = {n - 1}")
        k = n - 1
    rows_per_block = max(1, (chunk_mb * 2**20) // (n * 8))
    idx = np.empty((n, k), dtype=np.int64)
    sims = np.empty((n, k), dtype=np.float64)
    for start in range(0, n, rows_per_block):
        stop = min(start + rows_per_block, n)
        S = Zn[start:stop] @ Zn.T                                   # (block, n) cosine similarities
        S[np.arange(stop - start), np.arange(start, stop)] = -np.inf  # a track is not its own neighbour
        cand = np.argpartition(-S, k - 1, axis=1)[:, :k]            # the k largest, in arbitrary order
        cand_sims = np.take_along_axis(S, cand, axis=1)
        order = np.lexsort((cand, -cand_sims), axis=1)              # by -similarity, then by index
        idx[start:stop] = np.take_along_axis(cand, order, axis=1)
        sims[start:stop] = np.take_along_axis(cand_sims, order, axis=1)
    return idx, sims


def knn_jaccard(idx_a: np.ndarray, idx_b: np.ndarray) -> float:
    """Mean over nodes of the Jaccard overlap between two neighbour lists of the same nodes."""
    total = 0.0
    for a, b in zip(idx_a, idx_b):
        sa, sb = set(a.tolist()), set(b.tolist())
        total += len(sa & sb) / len(sa | sb)
    return total / len(idx_a)


@dataclass
class Graph:
    """Undirected weighted graph as a list of dicts: adj[i][j] = weight of edge i-j (stored in both dicts).

    A self-loop of node i is stored once, as adj[i][i]. It counts twice in the degree of i
    (k_i = sum_{j != i} w_ij + 2 w_ii), which is the convention that keeps modularity unchanged
    when communities are collapsed into single nodes (see `aggregate`).
    """
    adj: list[dict[int, float]]

    @property
    def n(self) -> int:
        return len(self.adj)


def graph_from_edges(n: int, edges: list[tuple[int, int, float]]) -> Graph:
    """Build a Graph from (i, j, w) triples; repeated edges add up, i == j makes a self-loop."""
    adj: list[dict[int, float]] = [dict() for _ in range(n)]
    for i, j, w in edges:
        adj[i][j] = adj[i].get(j, 0.0) + w
        if i != j:
            adj[j][i] = adj[j].get(i, 0.0) + w
    return Graph(adj)


def build_knn_graph(idx: np.ndarray, sims: np.ndarray) -> Graph:
    """Symmetric kNN graph: edge i-j if j is among i's neighbours OR i among j's (union).

    Weight = cosine similarity clipped at 0; edges of weight 0 are not stored; no self-loops.
    If both directions exist the larger weight is kept (for cosine they are equal anyway).
    """
    n = len(idx)
    adj: list[dict[int, float]] = [dict() for _ in range(n)]
    for i in range(n):
        for j, s in zip(idx[i].tolist(), sims[i].tolist()):
            w = max(s, 0.0)
            if w > 0.0 and j != i:
                w = max(w, adj[i].get(j, 0.0))
                adj[i][j] = w
                adj[j][i] = w
    return Graph(adj)


def degrees(g: Graph) -> np.ndarray:
    """Weighted degree k_i = sum of edge weights at i, a self-loop counted twice."""
    k = np.zeros(g.n)
    for i, nbrs in enumerate(g.adj):
        for j, w in nbrs.items():
            k[i] += 2 * w if j == i else w
    return k


def total_weight(g: Graph) -> float:
    """m = total edge weight, every edge (and self-loop) counted once = sum of degrees / 2."""
    return float(degrees(g).sum() / 2)


def connected_components(g: Graph) -> tuple[int, np.ndarray]:
    """Breadth-first search; returns (number of components, component id per node)."""
    comp = np.full(g.n, -1, dtype=np.int64)
    n_comp = 0
    for start in range(g.n):
        if comp[start] >= 0:
            continue
        comp[start] = n_comp
        queue = deque([start])
        while queue:
            i = queue.popleft()
            for j in g.adj[i]:
                if comp[j] < 0:
                    comp[j] = n_comp
                    queue.append(j)
        n_comp += 1
    return n_comp, comp


def modularity(g: Graph, labels: np.ndarray, resolution: float = 1.0) -> float:
    """Weighted modularity with resolution gamma, computed from scratch:

        Q = sum_c [ L_c / m  -  gamma * (Sigma_tot_c / 2m)^2 ]

    L_c = weight of the edges inside community c (each edge once, self-loops included),
    Sigma_tot_c = sum of the degrees of c's nodes. This equals
    (1/2m) sum_ij [A_ij - gamma k_i k_j / 2m] delta(c_i, c_j) with A_ii = 2 w_ii.
    A graph without edges has Q = 0 by definition.
    """
    m = total_weight(g)
    if m == 0:
        return 0.0
    k = degrees(g)
    inside: dict[int, float] = {}
    for i, nbrs in enumerate(g.adj):
        for j, w in nbrs.items():
            if j >= i and labels[i] == labels[j]:          # each edge once (j == i: self-loop)
                inside[labels[i]] = inside.get(labels[i], 0.0) + w
    sigma_tot: dict[int, float] = {}
    for i in range(g.n):
        sigma_tot[labels[i]] = sigma_tot.get(labels[i], 0.0) + k[i]
    return sum(inside.get(c, 0.0) / m - resolution * (s / (2 * m)) ** 2 for c, s in sigma_tot.items())


def insertion_gain(k_i_in: float, sigma_tot_c: float, k_i: float, m: float, resolution: float) -> float:
    """Modularity gain of putting an isolated node i into community C:

        gain = k_i,in / m  -  gamma * Sigma_tot(C) * k_i / (2 m^2)

    k_i,in = weight of the edges from i into C, Sigma_tot(C) = total degree of C without i.
    """
    return k_i_in / m - resolution * sigma_tot_c * k_i / (2 * m * m)


def local_moving(g: Graph, resolution: float, rng: np.random.Generator,
                 tol: float = 1e-10) -> tuple[np.ndarray, int, list[float]]:
    """Louvain phase 1: move single nodes to the neighbouring community with the largest gain.

    Starts from singletons. Nodes are visited in one seeded random order; full sweeps are repeated until
    no node moves or a sweep improves Q by less than `tol`. Returns (community per node, number of
    sweeps, Q after every sweep). Ties: staying is preferred, then the smallest community id.
    """
    n = g.n
    community = np.arange(n)
    m = total_weight(g)
    if m == 0:
        return community, 0, [0.0]
    k = degrees(g)
    sigma_tot = k.copy()                  # total degree per community (community ids = node ids)
    order = rng.permutation(n)
    q_trace = [modularity(g, community, resolution)]
    sweeps = 0
    while True:
        sweeps += 1
        moved, sweep_gain = 0, 0.0
        for i in order:
            own = community[i]
            # weight from i into each neighbouring community (the self-loop of i does not count)
            k_i_to: dict[int, float] = {}
            for j, w in g.adj[i].items():
                if j != i:
                    k_i_to[community[j]] = k_i_to.get(community[j], 0.0) + w
            sigma_tot[own] -= k[i]        # take i out of its community
            stay_gain = insertion_gain(k_i_to.get(own, 0.0), sigma_tot[own], k[i], m, resolution)
            best, best_gain = own, stay_gain
            for c in sorted(k_i_to):
                gain = insertion_gain(k_i_to[c], sigma_tot[c], k[i], m, resolution)
                if gain > best_gain + 1e-12:
                    best, best_gain = c, gain
            sigma_tot[best] += k[i]       # put i into the best community
            community[i] = best
            if best != own:
                moved += 1
                sweep_gain += best_gain - stay_gain
        q_trace.append(q_trace[-1] + sweep_gain)
        if moved == 0 or sweep_gain < tol:
            break
    return community, sweeps, q_trace


def relabel(labels: np.ndarray) -> np.ndarray:
    """Renumber labels to 0..C-1 in order of first appearance."""
    new_id: dict[int, int] = {}
    out = np.empty(len(labels), dtype=np.int64)
    for i, c in enumerate(labels.tolist()):
        if c not in new_id:
            new_id[c] = len(new_id)
        out[i] = new_id[c]
    return out


def aggregate(g: Graph, labels: np.ndarray) -> Graph:
    """Louvain phase 2: one node per community (labels must be 0..C-1).

    Edges between communities are summed. Edges inside a community, and old self-loops, become the
    self-loop of the new node, stored once. With the degree convention of `Graph` (self-loop counted
    twice) every new degree equals the sum of its members' degrees and m is unchanged, so modularity
    of a partition equals modularity of the singleton partition of the aggregated graph.
    """
    C = int(labels.max()) + 1 if g.n else 0
    adj: list[dict[int, float]] = [dict() for _ in range(C)]
    for i, nbrs in enumerate(g.adj):
        for j, w in nbrs.items():
            if j < i:
                continue                              # every undirected edge once
            a, b = int(labels[i]), int(labels[j])
            if a == b:
                adj[a][a] = adj[a].get(a, 0.0) + w
            else:
                adj[a][b] = adj[a].get(b, 0.0) + w
                adj[b][a] = adj[b].get(a, 0.0) + w
    return Graph(adj)


@dataclass
class LouvainResult:
    labels: np.ndarray            # community 0..C-1 of every original node
    modularity: float             # Q of the final partition on the original graph
    n_levels: int                 # number of aggregation levels that changed something
    sweeps_per_level: list[int]   # phase-1 sweeps at every level (including the last, unchanged one)
    q_trace: list[float]          # Q after every sweep, over all levels (never decreases)


def louvain(g: Graph, resolution: float = 1.0, seed: int = 0, tol: float = 1e-10) -> LouvainResult:
    """Louvain: alternate local moving (phase 1) and aggregation (phase 2) until a level changes nothing."""
    rng = np.random.default_rng(seed)
    labels = np.arange(g.n)               # community of every original node
    current = g
    sweeps_per_level: list[int] = []
    q_trace: list[float] = []
    while True:
        community, sweeps, trace = local_moving(current, resolution, rng, tol)
        sweeps_per_level.append(sweeps)
        q_trace.extend(trace if not q_trace else trace[1:])
        if len(np.unique(community)) == current.n:  # nobody moved: done
            break
        community = relabel(community)
        labels = community[labels]
        current = aggregate(current, community)
    labels = relabel(labels)
    q = modularity(g, labels, resolution)
    assert abs(q - q_trace[-1]) < 1e-8, f"bookkeeping error: tracked Q {q_trace[-1]} vs recomputed {q}"
    return LouvainResult(labels, q, len(sweeps_per_level) - 1, sweeps_per_level, q_trace)


@dataclass
class ClusterResult:
    labels: np.ndarray
    n_clusters: int
    sizes: np.ndarray             # tracks per cluster, largest first
    modularity: float
    n_components: int             # connected components of the kNN graph
    n_levels: int
    sweeps_per_level: list[int]


def cluster_from_knn(idx: np.ndarray, sims: np.ndarray, k: int, resolution: float, seed: int) -> ClusterResult:
    """Graph + Louvain from precomputed neighbour lists (lists for a larger k can be sliced: they are nested)."""
    g = build_knn_graph(idx[:, :k], sims[:, :k])
    n_comp, _ = connected_components(g)
    res = louvain(g, resolution, seed)
    sizes = np.sort(np.bincount(res.labels))[::-1]
    return ClusterResult(res.labels, len(sizes), sizes, res.modularity, n_comp, res.n_levels, res.sweeps_per_level)


def cluster(Z_full: np.ndarray, d: int, k: int, resolution: float, seed: int, chunk_mb: int = 256) -> ClusterResult:
    """Steps 2-4: first d PCA columns -> unit length (dot = cosine) -> kNN graph -> Louvain."""
    Zn = l2_normalize(Z_full[:, :d])     # must happen after PCA: PCA re-centres and changes lengths
    idx, sims = knn_cosine(Zn, k, chunk_mb)
    return cluster_from_knn(idx, sims, k, resolution, seed)


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index of two labelings (1 = identical, about 0 = chance), from the contingency table."""
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1)
    comb2 = lambda x: x * (x - 1) / 2
    sum_cells = comb2(table).sum()
    sum_a, sum_b = comb2(table.sum(axis=1)).sum(), comb2(table.sum(axis=0)).sum()
    expected = sum_a * sum_b / comb2(len(a))
    max_index = (sum_a + sum_b) / 2
    if max_index == expected:
        return 1.0
    return float((sum_cells - expected) / (max_index - expected))
