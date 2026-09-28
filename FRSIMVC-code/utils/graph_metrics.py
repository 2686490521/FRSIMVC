"""Graph-level diagnostics for the FSRIMVC pipeline.

The v3 originals are kept unchanged:

* ``graph_metrics``         -> nodes / undirected edges / connected components
                               / algebraic connectivity lambda_2 of one graph;
* ``edge_retention_ratio``  -> how much of a reference graph survives.

Outline-v6 (module I, RADG) adds the aggregation helpers the new adaptive-fusion
diagnostics need.  They are deliberately dependency-free (numpy + scipy only)
so that ``dataset.py`` -- which runs before any torch model exists -- can use
them:

* ``distribution_summary``            -> mean/std/p10/p25/p50/p75/p90
* ``graph_degree_stats``              -> degree mean/std/min/max
* ``graph_relative_frobenius_change`` -> ``||A - B||_F / (||B||_F + eps)``
* ``node_mean_incident_weight``       -> per-node mean incident edge weight
* ``pearson_correlation``             -> Pearson r, ``None`` when undefined
"""

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh


def graph_metrics(adjacency):
    graph = sparse.csr_matrix(adjacency, dtype=np.float32)
    graph = sparse.csr_matrix(graph.maximum(graph.T), dtype=np.float32)
    graph = graph - sparse.diags(graph.diagonal())
    graph.eliminate_zeros()
    n_nodes = graph.shape[0]
    n_components, _ = csgraph.connected_components(graph, directed=False)
    algebraic_connectivity = 0.0
    if n_nodes > 1 and graph.nnz > 0:
        try:
            laplacian = csgraph.laplacian(graph, normed=True)
            eigenvalues = eigsh(
                laplacian,
                k=min(2, n_nodes - 1),
                which='SM',
                return_eigenvectors=False,
            )
            eigenvalues = np.sort(np.real(eigenvalues))
            if eigenvalues.size > 1:
                algebraic_connectivity = float(max(0.0, eigenvalues[1]))
        except Exception:
            algebraic_connectivity = 0.0
    return {
        'nodes': int(n_nodes),
        'undirected_edges': int(graph.nnz // 2),
        'connected_components': int(n_components),
        'algebraic_connectivity': algebraic_connectivity,
    }


def edge_retention_ratio(masked_graph, reference_graph):
    reference = reference_graph.astype(bool).tocsr()
    retained = masked_graph.astype(bool).multiply(reference).nnz
    return float(retained / reference.nnz) if reference.nnz else 1.0


def node_feature_edge_loss(local_graph, reference_graph):
    """Fraction of each node's reference neighbours absent from local graph.

    The reference uses corrupted features without the observed-node filter;
    this measures filter-induced neighbourhood change, not clean-data quality.
    """
    reference = sparse.csr_matrix(reference_graph).copy()
    local = sparse.csr_matrix(local_graph).copy()
    for graph in (reference, local):
        graph.setdiag(0)
        graph.eliminate_zeros()
        graph.data[:] = 1
    counts = np.diff(reference.indptr)
    retained = np.asarray(reference.multiply(local).sum(axis=1)).reshape(-1)
    loss = np.full(counts.size, np.nan)
    valid = counts > 0
    loss[valid] = 1.0 - retained[valid] / counts[valid]
    return loss


# --------------------------------------------------------------------------
# outline-v6 (module I / RADG) helpers
# --------------------------------------------------------------------------

def _finite(values):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return array
    return array[np.isfinite(array)]


def distribution_summary(values, quantiles=(10, 25, 50, 75, 90), prefix=''):
    """Mean / std / percentile summary of a sample, safe for empty inputs.

    Every percentile key is written as ``p10``, ``p25`` ... so the same helper
    can describe ``beta_ij``, ``r_i`` or the graph change.  ``None`` values are
    dropped; if nothing is left every field is ``None`` (never NaN, so the
    payload stays valid JSON and the aggregator can skip it).
    """
    array = _finite(values)
    keys = ['mean', 'std'] + ['p%d' % int(q) for q in quantiles]
    payload = {prefix + key: None for key in keys}
    if array.size == 0:
        return payload
    payload[prefix + 'mean'] = float(array.mean())
    payload[prefix + 'std'] = float(array.std(ddof=1)) if array.size > 1 else 0.0
    percentiles = np.percentile(array, [float(q) for q in quantiles])
    for q, value in zip(quantiles, percentiles):
        payload[prefix + 'p%d' % int(q)] = float(value)
    return payload


def graph_degree_stats(adjacency):
    """Degree mean/std/min/max of an undirected, self-loop-free view of a graph."""
    graph = sparse.csr_matrix(adjacency, dtype=np.float64)
    graph = sparse.csr_matrix(graph.maximum(graph.T), dtype=np.float64)
    graph = graph - sparse.diags(graph.diagonal())
    graph.eliminate_zeros()
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    if degree.size == 0:
        return {'degree_mean': 0.0, 'degree_std': 0.0, 'degree_min': 0.0, 'degree_max': 0.0}
    return {
        'degree_mean': float(degree.mean()),
        'degree_std': float(degree.std(ddof=1)) if degree.size > 1 else 0.0,
        'degree_min': float(degree.min()),
        'degree_max': float(degree.max()),
    }


def graph_relative_frobenius_change(graph_a, graph_b, eps=1e-12):
    """``||A - B||_F / (||B||_F + eps)`` -- outline-v6 Delta_A.

    ``graph_b`` is the reference (the fixed fused graph).  ``None`` is returned
    when the shapes disagree, so a broken comparison can never be mistaken for
    "the graph did not move".
    """
    a = sparse.csr_matrix(graph_a, dtype=np.float64)
    b = sparse.csr_matrix(graph_b, dtype=np.float64)
    if a.shape != b.shape:
        return None
    difference = (a - b).tocsr()
    numerator = np.sqrt(float(difference.multiply(difference).sum()))
    denominator = np.sqrt(float(b.multiply(b).sum()))
    return float(numerator / (denominator + eps))


def node_mean_incident_weight(adjacency):
    """Per-node mean incident edge weight (self-loops ignored).

    Returns ``(mean_weight, n_incident_edges)``; nodes without a single edge
    get ``mean_weight = 0`` and ``n = 0``.
    """
    graph = sparse.csr_matrix(adjacency, dtype=np.float64)
    graph = graph - sparse.diags(graph.diagonal())
    graph.eliminate_zeros()
    total = np.asarray(graph.sum(axis=1)).reshape(-1)
    counts = np.diff(graph.indptr).astype(np.float64)
    mean_weight = np.zeros_like(total)
    valid = counts > 0
    mean_weight[valid] = total[valid] / counts[valid]
    return mean_weight, counts


def node_neighbour_reliability(adjacency, node_reliability):
    """Mean observed ratio of each node's neighbours in ``adjacency``.

    ``None`` for isolated nodes.  ``1 - value`` is the outline's
    ``d_i^feature`` (how corrupted a node's feature neighbourhood is).
    """
    graph = sparse.csr_matrix(adjacency, dtype=np.float64)
    graph = graph - sparse.diags(graph.diagonal())
    graph.eliminate_zeros()
    reliability = np.asarray(node_reliability, dtype=np.float64).reshape(-1)
    if reliability.size != graph.shape[0]:
        raise ValueError(
            'node_reliability length %d does not match graph size %d'
            % (reliability.size, graph.shape[0])
        )
    weighted = graph.copy()
    weighted.data = np.ones_like(weighted.data)
    neighbour_sum = np.asarray(weighted @ reliability).reshape(-1)
    counts = np.diff(weighted.indptr).astype(np.float64)
    result = np.full(graph.shape[0], np.nan, dtype=np.float64)
    valid = counts > 0
    result[valid] = neighbour_sum[valid] / counts[valid]
    return result


def pearson_correlation(x, y, min_samples=3):
    """Pearson r between two equal-length samples; ``None`` when undefined."""
    left = np.asarray(x, dtype=np.float64).reshape(-1)
    right = np.asarray(y, dtype=np.float64).reshape(-1)
    if left.size != right.size:
        return None
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if left.size < min_samples:
        return None
    if left.std() <= 0.0 or right.std() <= 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])
