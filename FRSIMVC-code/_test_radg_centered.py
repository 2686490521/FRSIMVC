"""Regression checks for outline-v7 RADG-v1.1 (centered RADG).

Everything here is pure numpy/scipy -- no dataset, no GPU, no training -- so it
runs in a couple of seconds and can be used as a launcher preflight.

Covered:

  * the centring guarantee itself: |mean(beta) - base| < tolerance;
  * gamma = 0 degenerates to the fixed fusion (beta == base everywhere, so
    A_RADG == A_fixed exactly);
  * the clip bounds are respected and actually attained when gamma is large;
  * beta is monotone in the edge uncertainty u_ij before clipping;
  * E_s != E_f is handled (one beta per unordered pair, never a dense matrix);
  * the fusion stays sparse (no densification) and never changes the
    normalisation pipeline;
  * argument validation.
"""
import sys

import numpy as np
import scipy.sparse as sparse

sys.path.insert(0, '.')
sys.path.insert(0, 'utils')

from utils.superpixel_utils import (  # noqa: E402
    RADG_MODES,
    centered_beta_values,
    edge_uncertainty_values,
    fuse_centered_graph,
    fuse_consensus_local_graph,
    fuse_reliability_adaptive_graph,
    normalize_sparse_adjacency,
)

PASSED = 0
FAILED = 0


def check(name, condition, detail=''):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print('  PASS  %s %s' % (name, detail))
    else:
        FAILED += 1
        print('  FAIL  %s %s' % (name, detail))


def random_reliability(n_nodes, rng, mode='beta'):
    if mode == 'all_observed':
        return np.ones(n_nodes)
    if mode == 'all_missing':
        return np.zeros(n_nodes)
    if mode == 'binary':
        return rng.integers(0, 2, size=n_nodes).astype(float)
    return rng.beta(2.0, 2.0, size=n_nodes)


def random_graph(n_nodes, rng, density=0.01, seed_edges=None):
    rows, cols = [], []
    if seed_edges:
        rows.extend(seed_edges[0])
        cols.extend(seed_edges[1])
    n_extra = max(1, int(density * n_nodes * n_nodes))
    rows.extend(rng.integers(0, n_nodes, size=n_extra).tolist())
    cols.extend(rng.integers(0, n_nodes, size=n_extra).tolist())
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    keep = rows != cols
    rows, cols = rows[keep], cols[keep]
    data = rng.random(rows.size).astype(np.float32) + 0.1
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    matrix = matrix.maximum(matrix.T)
    return matrix.tocsr()


def main():
    rng = np.random.default_rng(20260921)
    print('RADG-v1.1 (centered) regression checks')
    print('=' * 60)

    # ---- 1. centring ----------------------------------------------------
    print('\n[1] centring: |mean(beta) - base| < tolerance')
    for kind in ('all_observed', 'all_missing', 'binary', 'beta'):
        r = random_reliability(500, rng, kind)
        u = 1.0 - np.sqrt(np.clip(r, 0, 1))
        # expand to a fake edge list so the statistics are non-degenerate
        u_edges = np.repeat(u, 3)
        for gamma in (0.25, 0.5, 0.75, 1.0, 2.0):
            beta, info = centered_beta_values(
                u_edges, base_beta=0.5, gamma=gamma,
                floor=0.2, ceiling=0.8, tolerance=0.005,
            )
            err = abs(float(beta.mean()) - 0.5)
            check(
                f'centred kind={kind} gamma={gamma}',
                err < 0.005 and info['beta_mean_error_from_base'] < 0.005,
                '|E[b]-0.5|=%.2e iters=%d' % (err, info['recentre_iterations']),
            )
            check(
                f'in-bounds kind={kind} gamma={gamma}',
                beta.min() >= 0.2 - 1e-9 and beta.max() <= 0.8 + 1e-9,
                '[%.4f, %.4f]' % (beta.min(), beta.max()),
            )

    # ---- 2. gamma = 0 degenerates to the fixed fusion --------------------
    print('\n[2] gamma = 0 is exactly the fixed fusion')
    n = 400
    for kind in ('all_observed', 'all_missing', 'binary', 'beta'):
        r = random_reliability(n, rng, kind)
        consensus = random_graph(n, rng)
        local = random_graph(n, rng)
        fused, info = fuse_centered_graph(
            consensus, local, r, base_beta=0.5, gamma=0.0,
            floor=0.2, ceiling=0.8,
        )
        control = fuse_consensus_local_graph(consensus, local, beta=0.5)
        diff = (fused - control).tocsr()
        max_abs = float(np.abs(diff.data).max()) if diff.nnz else 0.0
        check(
            'gamma=0 reproduces A_fixed kind=%s' % kind,
            max_abs < 1e-6 and diff.nnz == 0,
            'max|diff|=%.3e' % max_abs,
        )

    # ---- 3. base beta other than 0.5 ------------------------------------
    print('\n[3] non-default base beta is honoured')
    r = random_reliability(n, rng, 'beta')
    consensus = random_graph(n, rng)
    local = random_graph(n, rng)
    for base in (0.3, 0.5, 0.7):
        fused, info = fuse_centered_graph(
            consensus, local, r, base_beta=base, gamma=0.6,
            floor=0.1, ceiling=0.9,
        )
        check(
            'base=%.1f mean beta' % base,
            abs(info['center']['beta_mean'] - base) < 0.005,
            'mean=%.6f' % info['center']['beta_mean'],
        )

    # ---- 4. monotone in u before clipping -------------------------------
    print('\n[4] beta is monotone in the edge uncertainty')
    u = np.linspace(0.0, 1.0, 2001)
    # floor/ceiling are validated to [0, 1], so "no clipping" means [0, 1]:
    # with gamma = 0.25 the raw values stay inside 0.375 .. 0.625 anyway.
    beta, _ = centered_beta_values(u, base_beta=0.5, gamma=0.25,
                                   floor=0.0, ceiling=1.0)
    check('monotone non-decreasing', bool(np.all(np.diff(beta) >= -1e-12)))
    beta_big, _ = centered_beta_values(u, base_beta=0.5, gamma=3.0,
                                       floor=0.2, ceiling=0.8)
    check('clip attained at both ends',
          abs(beta_big.min() - 0.2) < 1e-6 and abs(beta_big.max() - 0.8) < 1e-6,
          '[%.4f, %.4f]' % (beta_big.min(), beta_big.max()))
    check('clipped values stay in [floor, ceiling]',
          bool(((beta_big >= 0.2 - 1e-9) & (beta_big <= 0.8 + 1e-9)).all()))

    # ---- 5. E_s != E_f: one beta per unordered pair ---------------------
    print('\n[5] E_s != E_f still yields one beta per unordered pair')
    r = random_reliability(n, rng, 'beta')
    shared = (np.array([0, 1, 2, 3]), np.array([1, 2, 3, 4]))
    consensus = random_graph(n, rng, seed_edges=shared)
    local = random_graph(n, rng, seed_edges=(shared[1], shared[0]))  # transposed
    fused, info = fuse_centered_graph(
        consensus, local, r, base_beta=0.5, gamma=0.6, floor=0.2, ceiling=0.8,
    )
    # (0,1) appears in both graphs; look it up in both beta vectors.
    cs = sparse.csr_matrix(consensus).tocoo()
    cf = sparse.csr_matrix(local).tocoo()
    key = lambda a, b: np.where(a <= b, a, b) * n + np.where(a <= b, b, a)
    ks = dict(zip(key(cs.row, cs.col).tolist(), info['beta_spatial'].tolist()))
    kf = dict(zip(key(cf.row, cf.col).tolist(), info['beta_feature'].tolist()))
    common = [k for k in ks if k in kf]
    check('shared edges get identical beta',
          bool(common) and all(abs(ks[k] - kf[k]) < 1e-6 for k in common),
          '%d shared edges' % len(common))

    # ---- 6. sparsity: no densification ----------------------------------
    print('\n[6] sparsity is preserved (no dense beta matrix)')
    n_big = 4000
    r_big = random_reliability(n_big, rng, 'beta')
    consensus_big = random_graph(n_big, rng, density=0.0005)
    local_big = random_graph(n_big, rng, density=0.0005)
    fused_big, _ = fuse_centered_graph(
        consensus_big, local_big, r_big, base_beta=0.5, gamma=0.6,
        floor=0.2, ceiling=0.8,
    )
    union = (consensus_big.astype(bool) + local_big.astype(bool))
    expected = union.nnz + n_big  # + self loops added by the normaliser
    check('no densification', fused_big.nnz <= expected,
          'nnz=%d expected<=%d' % (fused_big.nnz, expected))
    check('still sparse', sparse.issparse(fused_big))

    # ---- 7. centred vs uncentred differ ---------------------------------
    print('\n[7] centred and uncentred are genuinely different mechanisms')
    r = random_reliability(n, rng, 'beta')
    consensus = random_graph(n, rng)
    local = random_graph(n, rng)
    centred, info_c = fuse_centered_graph(
        consensus, local, r, base_beta=0.5, gamma=0.6, floor=0.2, ceiling=0.8,
    )
    uncentred, _ = fuse_reliability_adaptive_graph(
        consensus, local, r, beta_min=0.2, beta_max=0.8,
    )
    diff = (centred - uncentred).tocsr()
    max_abs = float(np.abs(diff.data).max()) if diff.nnz else 0.0
    check('centred != uncentred', max_abs > 1e-6, 'max|diff|=%.4f' % max_abs)
    check('centred mean beta ~ 0.5',
          abs(info_c['center']['beta_mean'] - 0.5) < 0.005,
          'mean=%.6f' % info_c['center']['beta_mean'])

    # ---- 8. edge uncertainty definition ---------------------------------
    print('\n[8] u_ij = 1 - sqrt(r_i r_j)')
    r = np.array([1.0, 0.25, 0.0, 0.5])
    u = edge_uncertainty_values(r, np.array([0, 0, 1, 2]), np.array([0, 1, 2, 3]))
    expect = np.array([0.0, 0.5, 1.0, 1.0 - np.sqrt(0.0)])
    check('u formula', np.allclose(u[:3], expect[:3]),
          'got %s' % np.round(u, 6).tolist())

    # ---- 9. validation ---------------------------------------------------
    print('\n[9] argument validation')
    u = np.linspace(0, 1, 50)
    for name, kwargs in (
        ('gamma < 0', {'gamma': -0.1}),
        ('floor > ceiling', {'floor': 0.8, 'ceiling': 0.2}),
        ('base out of range', {'base_beta': 1.5}),
        ('ceiling out of range', {'ceiling': 1.5}),
    ):
        try:
            centered_beta_values(u, **kwargs)
            check('rejects %s' % name, False, 'no error raised')
        except ValueError:
            check('rejects %s' % name, True)
    check('empty edge list is tolerated',
          centered_beta_values(np.zeros(0))[0].size == 0)
    check('RADG modes', RADG_MODES == ('centered', 'uncentered'))

    # ---- 10. normalisation is untouched ---------------------------------
    print('\n[10] the post-processing chain is unchanged')
    manual = normalize_sparse_adjacency(
        0.5 * consensus + 0.5 * local, add_self_loops=True
    )
    degenerate, _ = fuse_centered_graph(
        consensus, local, np.ones(n), base_beta=0.5, gamma=0.0,
        floor=0.2, ceiling=0.8,
    )
    diff = (manual - degenerate).tocsr()
    max_abs = float(np.abs(diff.data).max()) if diff.nnz else 0.0
    check('centred(gamma=0) == manual fixed fusion', max_abs < 1e-6,
          'max|diff|=%.3e' % max_abs)

    print('\n' + '=' * 60)
    if FAILED:
        print('%d/%d RADG-v1.1 CHECKS FAILED' % (FAILED, PASSED + FAILED))
        return 1
    print('ALL %d RADG-v1.1 CHECKS PASSED' % PASSED)
    return 0


if __name__ == '__main__':
    sys.exit(main())
