"""Smoke test for the outline-v5 BSSAT-v2 mass tempering + new diagnostics.

Run on the experiment server (needs torch)::

    python _test_bssat_v2.py

Checks, without touching the data or the GPU:
  * b' = (1-alpha) b + alpha/K is exact (alpha=0 no-op, alpha=1 uniform)
  * the dominant anchor is monotonically damped as alpha grows
  * both marginals stay normalised and the conditional stays row-stochastic
  * proportion_concentration / clustering_diagnostics expose the four new keys
  * the MUUFL foreground metrics are produced iff negative labels are present
"""

import numpy as np
import torch

from utils.csata import align_to_global_anchors
from utils.diagnostics import clustering_diagnostics, proportion_concentration

K, D_SPATIAL, D_SEMANTIC, N_VIEWS = 5, 8, 3, 3
ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.3, 1.0)

torch.manual_seed(0)
spatial = [torch.randn(K, D_SPATIAL) for _ in range(N_VIEWS)]
semantic = [torch.randn(K, D_SEMANTIC) for _ in range(N_VIEWS)]
# deliberately skewed cluster masses: cluster 0 owns most of the nodes
masses = [torch.tensor([50.0, 10.0, 5.0, 3.0, 2.0]) for _ in range(N_VIEWS)]


def run(alpha):
    return align_to_global_anchors(
        spatial, semantic, masses, None, K,
        random_state=0, epsilon=0.1, iterations=50,
        balanced=True, mass_uniform_alpha=alpha,
    )


maxima = []
for alpha in ALPHAS:
    anchors, conditionals, costs, loss, raw, tempered = run(alpha)
    assert abs(float(raw.sum()) - 1.0) < 1e-5, ('raw mass not normalised', alpha)
    assert abs(float(tempered.sum()) - 1.0) < 1e-5, ('tempered mass not normalised', alpha)
    for conditional in conditionals:
        row_sums = conditional.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), (
            'conditional is not row-stochastic', alpha,
        )
    assert anchors['spatial'].shape == (K, D_SPATIAL)
    assert anchors['semantic'].shape == (K, D_SEMANTIC)
    maxima.append(float(tempered.max()))
    print('alpha=%-5.2f  raw_max=%.4f  tempered_max=%.4f  tempered_min=%.5f  rowsum_ok'
          % (alpha, float(raw.max()), float(tempered.max()), float(tempered.min())))

_, _, _, _, raw0, tempered0 = run(0.0)
assert torch.allclose(raw0, tempered0), 'alpha=0 must reproduce BSSAT-r1 exactly'
_, _, _, _, _, tempered1 = run(1.0)
assert torch.allclose(tempered1, torch.full_like(tempered1, 1.0 / K), atol=1e-6), (
    'alpha=1 must give a uniform column marginal'
)
assert all(a >= b - 1e-9 for a, b in zip(maxima, maxima[1:])), maxima
print('p_max monotonically damped:', ['%.4f' % value for value in maxima])

concentration = proportion_concentration(
    np.array([630.0, 200.0, 100.0, 1.0, 1.0]), n_clusters=5
)
print('proportion_concentration:', {
    key: (None if value is None else round(value, 4))
    for key, value in concentration.items()
})
assert 0.0 < concentration['cluster_entropy'] < 1.0
assert concentration['effective_num_clusters'] < 5.0

empty = proportion_concentration(np.zeros(5), n_clusters=5)
assert all(value is None for value in empty.values()), empty

clustering = clustering_diagnostics(
    np.array([0] * 5 + [1] * 3 + [2] * 2),
    np.array([0] * 5 + [1] * 5),
    n_clusters=3,
)
for key in ('cluster_max_ratio', 'cluster_min_ratio', 'cluster_entropy',
            'effective_num_clusters', 'num_predicted_clusters'):
    assert key in clustering, key
print('clustering diagnostics:', {
    key: (None if clustering[key] is None else round(clustering[key], 4))
    for key in ('cluster_max_ratio', 'cluster_min_ratio', 'cluster_entropy',
                'effective_num_clusters')
})

from Flsimulator import _foreground_metrics  # noqa: E402  (heavy import, last)

y_with_background = np.array([1, 1, 2, 2, -1, -1, 3, 3])
y_pred = np.array([1, 1, 2, 2, 1, 1, 3, 3])
foreground = _foreground_metrics(y_with_background, y_pred)
assert 'ACC_foreground' in foreground and 'NMI_foreground' in foreground
assert foreground['foreground_nodes'] == 6
assert foreground['background_nodes'] == 2
assert _foreground_metrics(np.array([0, 1, 2]), np.array([0, 1, 2])) == {}, (
    'datasets without negative labels must not produce foreground metrics'
)
print('foreground metrics:', {
    key: (None if value is None else (round(value, 4) if isinstance(value, float) else value))
    for key, value in foreground.items()
})

print('OK: outline-v5 BSSAT-v2 smoke test passed')
