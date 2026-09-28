import numpy as np
from skimage.measure import regionprops
from skimage.segmentation import slic, mark_boundaries, find_boundaries
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, minmax_scale, normalize, StandardScaler


from matplotlib import cm
import matplotlib as mpl
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import cv2
import typing
from skimage import graph
from pathlib import Path
from scipy import sparse

def HSI_to_superpixels(img, n_superpixels, save_path=None):
    superpixel_label = slic(img, n_segments=n_superpixels, start_label=0)
    if save_path is not None:
        show_superpixel(superpixel_label, img, save_path)
    return superpixel_label


def show_superpixel(label, x=None, save_path='superpixels.pdf'):
    color = (162 / 255, 169 / 255, 175 / 255)
    if x is not None:
        n_row, n_col, n_band = x.shape
        x = minmax_scale(x.reshape(n_row * n_col, n_band)).reshape(n_row, n_col, n_band)
        mask = np.clip(mark_boundaries(x, label, color=(1, 1, 0), mode='subpixel'), 0, 1)
    else:
        n_row, n_col = label.shape
        mask_boundary = find_boundaries(label, mode='subpixel')
        mask = np.ones((n_row, n_col, 3))
        mask[mask_boundary] = color
    fig = plt.figure()
    plt.imshow(mask)
    plt.axis('off')
    plt.tight_layout()
    output_format = Path(save_path).suffix.lstrip('.') or 'png'
    fig.savefig(save_path, format=output_format, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()


def create_association_mat(superpixel_labels):
    labels = np.unique(superpixel_labels)
    n_labels = labels.shape[0]
    n_pixels = superpixel_labels.shape[0] * superpixel_labels.shape[1]
    association_mat = np.zeros((n_pixels, n_labels))
    superpixel_labels_ = superpixel_labels.reshape(-1)
    for i, label in enumerate(labels):
        association_mat[np.where(label == superpixel_labels_), i] = 1
    return association_mat

def _superpixel_means(source_img, superpixel_labels):
    s = source_img.reshape((-1, source_img.shape[-1]))
    labels_flat = superpixel_labels.reshape(-1)
    n_labels = np.unique(superpixel_labels).shape[0]
    counts = np.bincount(labels_flat, minlength=n_labels).clip(min=1)
    return np.column_stack([
        np.bincount(labels_flat, weights=s[:, band], minlength=n_labels) / counts
        for band in range(s.shape[1])
    ]).astype(np.float32)


def _superpixel_centers(superpixel_labels):
    n_labels = np.unique(superpixel_labels).shape[0]
    regions = regionprops(superpixel_labels + 1)
    center_indx = np.zeros((n_labels, 2))
    for i, props in enumerate(regions):
        center_indx[i, :] = props.centroid
    return center_indx.astype(np.float32)


def create_consensus_spatial_graph(superpixel_labels):
    """Build the modality-independent superpixel topology A^c."""
    n_labels = np.unique(superpixel_labels).shape[0]
    g = graph.RAG(superpixel_labels)
    spatial_adj = sparse.csr_matrix(
        nx.linalg.adjacency_matrix(g, nodelist=list(range(n_labels))),
        dtype=np.float32,
    )
    spatial_adj.data[:] = 1.0
    spatial_adj = spatial_adj.maximum(spatial_adj.T).tocsr()
    spatial_adj = spatial_adj - sparse.diags(spatial_adj.diagonal())
    spatial_adj.eliminate_zeros()
    return spatial_adj, _superpixel_centers(superpixel_labels)


def create_spixel_graph(source_img, superpixel_labels, n_neighbors=50, observed_mask=None):
    """Build A_f only from observed nodes so zero filling cannot create fake edges."""
    mean_fea = minmax_scale(_superpixel_means(source_img, superpixel_labels))
    n_labels = mean_fea.shape[0]
    if observed_mask is None:
        observed_mask = np.ones(n_labels, dtype=bool)
    observed_mask = np.asarray(observed_mask, dtype=bool)
    observed_indices = np.flatnonzero(observed_mask)

    if observed_indices.size <= 1:
        feature_adj = sparse.csr_matrix((n_labels, n_labels), dtype=np.float32)
    else:
        observed_features = mean_fea[observed_indices]
        valid_neighbors = min(n_neighbors, observed_indices.size - 1)
        observed_adj = kneighbors_graph(
            observed_features,
            n_neighbors=valid_neighbors,
            mode='distance',
            include_self=False,
        ).tocoo()
        variance = observed_features.var()
        gamma = 1.0 / (observed_features.shape[1] * variance) if variance > 0 else 1.0
        weights = np.exp(-np.square(observed_adj.data) * gamma).astype(np.float32)
        feature_adj = sparse.csr_matrix(
            (
                weights,
                (observed_indices[observed_adj.row], observed_indices[observed_adj.col]),
            ),
            shape=(n_labels, n_labels),
            dtype=np.float32,
        )
        feature_adj = sparse.csr_matrix(
            feature_adj.maximum(feature_adj.T), dtype=np.float32
        )
        feature_adj.eliminate_zeros()

    spatial_adj, centers = create_consensus_spatial_graph(superpixel_labels)
    return feature_adj, spatial_adj, centers


def normalize_sparse_adjacency(adjacency, add_self_loops=True):
    adjacency = sparse.csr_matrix(adjacency, dtype=np.float32)
    adjacency = sparse.csr_matrix(
        adjacency.maximum(adjacency.T), dtype=np.float32
    )
    if add_self_loops:
        adjacency = adjacency + sparse.eye(adjacency.shape[0], dtype=np.float32, format='csr')
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    valid = degree > 0
    inv_sqrt[valid] = np.power(degree[valid], -0.5)
    scale = sparse.diags(inv_sqrt)
    return sparse.csr_matrix(scale @ adjacency @ scale, dtype=np.float32)


def fuse_consensus_local_graph(consensus_adj, local_adj, beta=0.5):
    """Fixed fusion: same beta on every edge (the v3/v5 baseline)."""
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f'graph beta must be in [0, 1], got {beta}')
    fused = beta * consensus_adj + (1.0 - beta) * local_adj
    return normalize_sparse_adjacency(fused, add_self_loops=True)


# --------------------------------------------------------------------------
# outline-v6, module I: Reliability-Adaptive Dual-Graph (RADG-v1)
#
#   c_ij^(v)   = sqrt(r_i^(v) r_j^(v))                                  (edge reliability)
#   beta_ij^(v)= beta_min + (beta_max - beta_min) (1 - c_ij^(v))
#   A_ij^(v)   = beta_ij^(v) A_s,ij + (1 - beta_ij^(v)) A_f,ij^(v)
#
# The weights are built on each graph's own non-zero edges, so nothing is
# densified, and the result goes through the *unchanged* post-processing
# (symmetricisation, self loops, symmetric normalisation).  With
# beta_min == beta_max == 0.5 -- or with a constant c_ij -- this is exactly
# ``fuse_consensus_local_graph(..., beta=0.5)``; ``_test_radg.py`` asserts it.
# --------------------------------------------------------------------------

EDGE_RELIABILITY_MODES = ('geometric_mean', 'arithmetic_mean')


def edge_reliability_values(node_reliability, rows, cols, mode='geometric_mean'):
    """``c_ij`` for the given edge list, from the per-node observed ratios."""
    reliability = np.asarray(node_reliability, dtype=np.float64).reshape(-1)
    rows = np.asarray(rows).reshape(-1)
    cols = np.asarray(cols).reshape(-1)
    if rows.size != cols.size:
        raise ValueError('edge row/col arrays must have the same length')
    left = np.clip(reliability[rows], 0.0, 1.0)
    right = np.clip(reliability[cols], 0.0, 1.0)
    if mode == 'geometric_mean':
        # An edge is only as reliable as its weaker endpoint: sqrt(r_i r_j).
        return np.sqrt(left * right)
    if mode == 'arithmetic_mean':
        return 0.5 * (left + right)
    raise ValueError(
        f'unknown edge_reliability {mode!r}; expected one of {EDGE_RELIABILITY_MODES}'
    )


def reliability_adaptive_edge_weights(adjacency, node_reliability, beta_min,
                                      beta_max, edge_reliability='geometric_mean',
                                      use_complement=False):
    """Reweight one sparse graph edge-wise.

    Returns ``(weighted, beta_values, reliability_values)``, one entry per
    stored edge, in COO order.  ``use_complement=True`` multiplies by
    ``1 - beta_ij`` instead of ``beta_ij`` (the feature-graph side).
    """
    matrix = sparse.csr_matrix(adjacency, dtype=np.float32).tocoo()
    c_ij = edge_reliability_values(
        node_reliability, matrix.row, matrix.col, edge_reliability
    )
    beta_ij = beta_min + (beta_max - beta_min) * (1.0 - c_ij)
    factor = (1.0 - beta_ij) if use_complement else beta_ij
    weighted = sparse.csr_matrix(
        (matrix.data.astype(np.float64) * factor, (matrix.row, matrix.col)),
        shape=matrix.shape,
        dtype=np.float32,
    )
    return weighted, beta_ij.astype(np.float32), c_ij.astype(np.float32)


def fuse_reliability_adaptive_graph(consensus_adj, local_adj, node_reliability,
                                    beta_min=0.2, beta_max=0.8,
                                    edge_reliability='geometric_mean'):
    """Edge-wise adaptive fusion ``A = beta_ij A_s + (1 - beta_ij) A_f``.

    Returns ``(fused, info)``; ``info`` carries the per-edge ``beta_ij`` values
    of both edge sets so the caller can report the distribution -- the lesson
    of BSSAT-v2 is that a modulation nobody can measure is a modulation that
    did not happen.
    """
    if not 0.0 <= beta_min <= 1.0:
        raise ValueError(f'radg_beta_min must be in [0, 1], got {beta_min}')
    if not 0.0 <= beta_max <= 1.0:
        raise ValueError(f'radg_beta_max must be in [0, 1], got {beta_max}')
    if beta_min > beta_max:
        raise ValueError(
            f'radg_beta_min ({beta_min}) must not exceed radg_beta_max ({beta_max})'
        )
    if edge_reliability not in EDGE_RELIABILITY_MODES:
        raise ValueError(
            f'unknown edge_reliability {edge_reliability!r}; '
            f'expected one of {EDGE_RELIABILITY_MODES}'
        )
    n_nodes = sparse.csr_matrix(consensus_adj, dtype=np.float32).shape[0]
    reliability = np.asarray(node_reliability, dtype=np.float64).reshape(-1)
    if reliability.size != n_nodes:
        raise ValueError(
            f'node_reliability has {reliability.size} entries but the graph has '
            f'{n_nodes} nodes'
        )

    spatial_weighted, beta_spatial, c_spatial = reliability_adaptive_edge_weights(
        consensus_adj, reliability, beta_min, beta_max, edge_reliability
    )
    feature_weighted, beta_feature, c_feature = reliability_adaptive_edge_weights(
        local_adj, reliability, beta_min, beta_max, edge_reliability,
        use_complement=True,
    )
    fused = sparse.csr_matrix(spatial_weighted + feature_weighted, dtype=np.float32)
    fused.eliminate_zeros()
    fused = normalize_sparse_adjacency(fused, add_self_loops=True)
    info = {
        'beta_spatial': beta_spatial,
        'beta_feature': beta_feature,
        'edge_reliability_spatial': c_spatial,
        'edge_reliability_feature': c_feature,
        'edges_spatial': int(spatial_weighted.nnz),
        'edges_feature': int(feature_weighted.nnz),
    }
    return fused, info


# ---------------------------------------------------------------------------
# RADG-v1.1 / "centered" RADG  (outline v7 sections 3-8)
#
#   u_ij     = 1 - c_ij = 1 - sqrt(r_i r_j)
#   beta_raw = beta_0 + gamma (u_ij - mean(u))
#   beta     = clip(beta_raw, floor, ceiling)
#   then iterative recentring so that |mean(beta) - beta_0| < tolerance.
#
# The whole point of this mode is that the *average* fusion strength stays at
# the fixed baseline, so any accuracy change can be attributed to the local
# re-allocation and not to "the spatial graph was simply turned down".
# ---------------------------------------------------------------------------

RADG_MODES = ('centered', 'uncentered')
RADG_CENTER_MAX_ITERS = 12


def edge_uncertainty_values(node_reliability, rows, cols,
                            edge_reliability='geometric_mean'):
    """``u_ij = 1 - c_ij`` for the given edge list."""
    c_ij = edge_reliability_values(node_reliability, rows, cols, edge_reliability)
    return 1.0 - c_ij


def _clip(values, floor, ceiling):
    return np.clip(np.asarray(values, dtype=np.float64), floor, ceiling)


def centered_beta_values(u_ij, base_beta=0.5, gamma=0.5, floor=0.2, ceiling=0.8,
                         tolerance=0.005, max_iters=RADG_CENTER_MAX_ITERS):
    """Mean-preserving, clipped edge weights.

    Returns ``(beta, info)``.  ``beta`` is in ``[floor, ceiling]`` and its mean
    is driven back towards ``base_beta``: outline v7 section 6 asks for the
    clip-then-recentre loop to be repeated 2-3 times and to end with
    ``|mean(beta) - base_beta| < 0.005`` (``< 0.001`` preferred).
    """
    u = np.asarray(u_ij, dtype=np.float64).reshape(-1)
    if gamma < 0.0:
        raise ValueError(f'radg_gamma must be >= 0, got {gamma}')
    if not 0.0 <= floor <= ceiling <= 1.0:
        raise ValueError(
            f'expected 0 <= floor <= ceiling <= 1, got floor={floor} ceiling={ceiling}'
        )
    if not 0.0 <= base_beta <= 1.0:
        raise ValueError(f'radg_base_beta must be in [0, 1], got {base_beta}')
    if u.size == 0:
        return np.zeros(0, dtype=np.float64), {
            'edge_uncertainty_mean': None,
            'edge_uncertainty_std': None,
            'beta_mean': None,
            'beta_mean_error_from_base': None,
            'beta_mean_error_before_recentre': None,
            'recentre_iterations': 0,
            'recentre_delta': 0.0,
            'centered': True,
        }
    u_bar = float(u.mean())
    raw = base_beta + float(gamma) * (u - u_bar)
    beta = _clip(raw, floor, ceiling)
    mean_before = float(beta.mean())
    delta_total = 0.0
    iterations = 0
    mean_error = abs(mean_before - base_beta)
    # Only worth recentring when the clip actually bit (or the raw mean is
    # already off, which can happen if u_bar was computed on a different set).
    for _ in range(int(max_iters)):
        mean_error = abs(float(beta.mean()) - base_beta)
        if mean_error < float(tolerance):
            break
        delta = base_beta - float(beta.mean())
        # A shift that cannot move anything (everything pinned at one bound)
        # would loop forever without changing beta; stop instead.
        if abs(delta) < 1e-12:
            break
        shifted = _clip(beta + delta, floor, ceiling)
        delta_total += delta
        iterations += 1
        if np.array_equal(shifted, beta):
            beta = shifted
            break
        beta = shifted
    final_mean = float(beta.mean())
    info = {
        'edge_uncertainty_mean': u_bar,
        'edge_uncertainty_std': float(u.std()),
        'beta_mean': final_mean,
        'beta_mean_error_from_base': abs(final_mean - base_beta),
        'beta_mean_error_before_recentre': abs(mean_before - base_beta),
        'recentre_iterations': int(iterations),
        'recentre_delta': float(delta_total),
        'centered': True,
    }
    return beta, info


def _union_upper_edges(consensus_adj, local_adj):
    """Unique undirected edges of the union of both graphs, no self loops.

    ``beta_ij`` must be a single number per unordered pair, otherwise the
    spatial and the feature side of the same pair would disagree.  Working on
    the union (and only the upper triangle) is also what keeps the mean of
    beta well defined when ``E_s != E_f``.
    """
    union = (consensus_adj.astype(bool) + local_adj.astype(bool)).tocsr()
    union.setdiag(0)
    union.eliminate_zeros()
    upper = sparse.triu(union, k=1).tocoo()
    return upper.row.astype(np.int64), upper.col.astype(np.int64)


def _lookup_beta(rows, cols, union_rows, union_cols, beta_union, n_nodes):
    """Map ``beta`` from the sorted union edge list onto an arbitrary edge list."""
    key = lambda r, c: np.where(r <= c, r, c).astype(np.int64) * np.int64(n_nodes) + \
        np.where(r <= c, c, r).astype(np.int64)
    union_key = key(union_rows, union_cols)
    order = np.argsort(union_key)
    union_key = union_key[order]
    beta_sorted = np.asarray(beta_union, dtype=np.float64)[order]
    target = key(np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))
    pos = np.searchsorted(union_key, target)
    pos = np.clip(pos, 0, union_key.size - 1)
    missing = union_key[pos] != target
    if missing.any():
        raise RuntimeError(
            'internal error: an edge is missing from the RADG union edge list'
        )
    return beta_sorted[pos]


def fuse_centered_graph(consensus_adj, local_adj, node_reliability,
                        base_beta=0.5, gamma=0.5, floor=0.2, ceiling=0.8,
                        edge_reliability='geometric_mean', tolerance=0.005):
    """Mean-preserving edge-wise fusion (outline v7 sections 4-8).

    Returns ``(fused, info)``.  Everything runs on the sparse edge lists; no
    ``N x N`` dense ``beta`` matrix is ever materialised (section 8).
    """
    if edge_reliability not in EDGE_RELIABILITY_MODES:
        raise ValueError(
            f'unknown edge_reliability {edge_reliability!r}; '
            f'expected one of {EDGE_RELIABILITY_MODES}'
        )
    consensus = sparse.csr_matrix(consensus_adj, dtype=np.float32)
    local = sparse.csr_matrix(local_adj, dtype=np.float32)
    n_nodes = consensus.shape[0]
    reliability = np.asarray(node_reliability, dtype=np.float64).reshape(-1)
    if reliability.size != n_nodes:
        raise ValueError(
            f'node_reliability has {reliability.size} entries but the graph has '
            f'{n_nodes} nodes'
        )

    u_rows, u_cols = _union_upper_edges(consensus, local)
    u_union = edge_uncertainty_values(reliability, u_rows, u_cols, edge_reliability)
    beta_union, centre_info = centered_beta_values(
        u_union, base_beta=base_beta, gamma=gamma, floor=floor, ceiling=ceiling,
        tolerance=tolerance,
    )

    coo_s = consensus.tocoo()
    coo_f = local.tocoo()
    beta_s = _lookup_beta(coo_s.row, coo_s.col, u_rows, u_cols, beta_union, n_nodes)
    beta_f = _lookup_beta(coo_f.row, coo_f.col, u_rows, u_cols, beta_union, n_nodes)

    spatial_weighted = sparse.csr_matrix(
        (coo_s.data.astype(np.float64) * beta_s, (coo_s.row, coo_s.col)),
        shape=consensus.shape, dtype=np.float32,
    )
    feature_weighted = sparse.csr_matrix(
        (coo_f.data.astype(np.float64) * (1.0 - beta_f), (coo_f.row, coo_f.col)),
        shape=local.shape, dtype=np.float32,
    )
    fused = sparse.csr_matrix(spatial_weighted + feature_weighted, dtype=np.float32)
    fused.eliminate_zeros()
    fused = normalize_sparse_adjacency(fused, add_self_loops=True)
    info = {
        'beta_spatial': beta_s.astype(np.float32),
        'beta_feature': beta_f.astype(np.float32),
        'edge_reliability_spatial': edge_reliability_values(
            reliability, coo_s.row, coo_s.col, edge_reliability
        ).astype(np.float32),
        'edge_reliability_feature': edge_reliability_values(
            reliability, coo_f.row, coo_f.col, edge_reliability
        ).astype(np.float32),
        'edges_spatial': int(spatial_weighted.nnz),
        'edges_feature': int(feature_weighted.nnz),
        'union_edges': int(u_rows.size),
        'beta_union': beta_union.astype(np.float32),
        'edge_uncertainty_union': u_union.astype(np.float32),
        'center': centre_info,
    }
    return fused, info


def extract_superpixel_features(img, labels, mode:typing.Literal['mean_std_geo', 'mean_std', 'center_patch']='mean_std_geo', patch_size=None):
    def normalization(data):
        _range = np.max(data) - np.min(data)
        if _range == 0:
            return np.zeros_like(data)
        return (data - np.min(data)) / _range

    num_labels = np.unique(labels).shape[0]
    if mode == 'mean_std_geo':
        features = np.zeros((num_labels, img.shape[2] * 2 + 3), dtype='float32')
        for i in range(num_labels):
            mask = np.zeros_like(labels, dtype='uint8')
            mask[labels==i] = 1

            mean = np.mean(img[labels==i], axis=0)
            std = np.std(img[labels==i], axis=0)
            features[i, 0:img.shape[2]] = mean
            features[i, img.shape[2]:img.shape[2] * 2] = std
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = cv2.arcLength(contours[0], True)
            area = cv2.contourArea(contours[0])
            aspect_ratio = float(np.max(contours[0][:, 0, 0]) - np.min(contours[0][:, 0, 0])) / \
                        float(np.max(contours[0][:, 0, 1]) - np.min(contours[0][:, 0, 1]) + 1)
            if aspect_ratio > 1:
                aspect_ratio = 1 / aspect_ratio
            features[i, -3] = area
            features[i, -2] = perimeter
            features[i, -1] = aspect_ratio
        features[:, :-3] = normalization(features[:, :-3])
        features[:, -3] = normalization(features[:, -3])
        features[:, -2] = normalization(features[:, -2])

    elif mode == 'mean_std':
        features = np.zeros((num_labels, img.shape[2] * 2), dtype='float32')
        for i in range(num_labels):

            mean = np.mean(img[labels==i], axis=0)
            std = np.std(img[labels==i], axis=0)
            features[i, 0:img.shape[2]] = mean
            features[i, img.shape[2]:img.shape[2] * 2] = std


        regions = regionprops(labels + 1)
        n_labels = np.unique(labels).shape[0]
        center_indx = np.zeros((n_labels, 2))
        for i, props in enumerate(regions):
            center_indx[i, :] = props.centroid
        features = np.concatenate((center_indx, features), axis=1)
        features[:, :2] = normalization(features[:, :2])

    elif mode == 'center_patch':
        assert patch_size % 2 == 1, "Patch size must be odd"
        features = np.zeros((num_labels, patch_size, patch_size, img.shape[2]), dtype='float32')
        regions = regionprops(labels + 1)
        pad_size = patch_size // 2
        pad_img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
        for i, props in enumerate(regions):
            center = props.centroid
            x, y = center
            x = int(x + pad_size)
            y = int(y + pad_size)
            feat = pad_img[x - pad_size:x + pad_size + 1, y - pad_size:y + pad_size + 1, :]
            features[i, :] = feat
        features = np.transpose(features, (0, 3, 1, 2))
        features = minmax_scale(features.reshape(features.shape[0], -1), axis=1)

    return features
