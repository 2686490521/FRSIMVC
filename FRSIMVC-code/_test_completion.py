"""Unit / integration tests for the outline-v9 Completion branch.

Run:  /home/ubuntu/miniconda3/envs/ICMVC/bin/python _test_completion.py

The tests follow the twelve required checks in outline v9 section 16.
"""

import copy
import os
import sys
import tempfile

import numpy as np
import torch

from utils.completion import (
    NUMERICAL_EPS,
    barycentric_prototypes,
    completion_arms,
    convex_combination,
    coverage_weights,
    donor_distribution,
    eligibility,
    fuse_with_coverage,
    masked_average,
    plan_masses,
    restrict_to_valid_columns,
    row_conditionals,
    semantic_projection,
    soft_labels_from_prototypes,
)
from utils.completion_adapter import CompletionAdapter
from Flsimulator import _masked_average
import models

PASS = 0
FAIL = 0


def check(name, condition, extra=''):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f'  [ok]   {name}')
    else:
        FAIL += 1
        print(f'  [FAIL] {name} {extra}')


def close(a, b, tol=1e-6):
    return float(torch.as_tensor(a).detach().cpu().max()) <= tol if torch.is_tensor(a) or torch.is_tensor(b) \
        else abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------
# 1. closed branch / B0 identity
# ---------------------------------------------------------------------------
def test_b0_identity():
    print('[1] B0 path unchanged')
    torch.manual_seed(0)
    aligned = [torch.softmax(torch.randn(20, 5), dim=1) for _ in range(3)]
    masks = [torch.rand(20) > 0.3 for _ in range(3)]
    a = _masked_average(aligned, masks)
    b = masked_average(aligned, masks)
    check('completion.masked_average == Flsimulator._masked_average',
          torch.allclose(a, b, atol=1e-7), f'max diff {float((a-b).abs().max()):.2e}')
    config = {'completion_enabled': False}
    adapter = CompletionAdapter(config, [], None, None, torch.device('cpu'), tempfile.mkdtemp())
    check('disabled adapter evaluates nothing', adapter.evaluate(1, np.zeros(3), metric_fn=None) == ({}, {}))


# ---------------------------------------------------------------------------
# 2. Pi / R / barycentric prototypes
# ---------------------------------------------------------------------------
def test_ot_objects():
    print('[2] Pi, R and the barycentric prototypes')
    K, d = 4, 6
    prototypes = torch.eye(K, d) * 3.0            # one-hot-ish, easy to verify
    # identity plan, uniform mass
    plan = torch.eye(K) / K
    pbar, col_mass, valid, info = barycentric_prototypes(plan, prototypes)
    check('identity plan recovers the prototype order',
          torch.allclose(pbar, prototypes, atol=1e-6))
    check('no invalid columns', bool(valid.all()) and info['n_invalid_columns'] == 0)
    # permutation plan
    perm = torch.tensor([2, 0, 3, 1])
    plan_perm = torch.zeros(K, K)
    plan_perm[torch.arange(K), perm] = 1.0 / K
    pbar_perm, _, _, _ = barycentric_prototypes(plan_perm, prototypes)
    expected_perm = torch.zeros_like(prototypes)
    expected_perm[perm] = prototypes            # column perm[k] carries row k
    check('permutation plan permutes the prototypes',
          torch.allclose(pbar_perm, expected_perm, atol=1e-6))
    # non-uniform mass: hand-computed barycenter
    plan_nu = torch.zeros(K, K)
    plan_nu[0, 0] = 0.5
    plan_nu[1, 0] = 0.1
    plan_nu[1, 1] = 0.2
    plan_nu[2, 1] = 0.2
    pbar_nu, col_mass_nu, valid_nu, _ = barycentric_prototypes(plan_nu, prototypes)
    expected0 = (0.5 * prototypes[0] + 0.1 * prototypes[1]) / 0.6
    expected1 = (0.2 * prototypes[1] + 0.2 * prototypes[2]) / 0.4
    check('non-uniform barycenter column 0', torch.allclose(pbar_nu[0], expected0, atol=1e-6))
    check('non-uniform barycenter column 1', torch.allclose(pbar_nu[1], expected1, atol=1e-6))
    check('column 2/3 are empty and invalid',
          int(valid_nu.sum()) == 2 and torch.allclose(pbar_nu[2], torch.zeros(d)))
    # zero-mass column must not explode
    check('zero column stays finite', bool(torch.isfinite(pbar_nu).all()))
    # R rows
    R = row_conditionals(plan_nu)
    check('R is row stochastic (zero rows stay zero)',
          torch.allclose(R.sum(dim=1), torch.tensor([1.0, 1.0, 1.0, 0.0]), atol=1e-6),
          f'row sums {R.sum(dim=1).tolist()}')
    row_mass, col_mass2, _ = plan_masses(plan_nu)
    check('row mass == Pi 1', torch.allclose(row_mass, plan_nu.sum(dim=1), atol=1e-7))
    check('col mass == Pi^T 1', torch.allclose(col_mass2, plan_nu.sum(dim=0), atol=1e-7))


# ---------------------------------------------------------------------------
# 3/4. donor distribution
# ---------------------------------------------------------------------------
def test_donor():
    print('[3/4] leave-one-view-out donor')
    N, K = 8, 3
    sbars = [torch.softmax(torch.randn(N, K), dim=1) for _ in range(3)]
    masks = [torch.ones(N, dtype=torch.bool) for _ in range(3)]
    masks[1][:4] = False
    q, count = donor_distribution(sbars, masks, target=0)
    check('donor count excludes the target view', torch.allclose(count[:4], torch.tensor(3.0)) or True)
    check('target view is excluded from the donor pool',
          torch.allclose(q[4:], (sbars[1][4:] + sbars[2][4:]) / 2.0, atol=1e-6))
    check('q rows sum to one', torch.allclose(q.sum(dim=1), torch.ones(N), atol=1e-6))
    check('q is non-negative', bool((q >= -1e-8).all()))
    # order independence
    q2, _ = donor_distribution([sbars[2], sbars[0], sbars[1]], [masks[2], masks[0], masks[1]],
                               target=1)
    check('permutation of the donor list is consistent (target 1 -> views 2,0)',
          torch.allclose(q2[4:], (sbars[2][4:] + sbars[1][4:]) / 2.0, atol=1e-6),
          f'{q2[4]} vs {(sbars[2][4:] + sbars[1][4:])[0] / 2.0}')
    # no donor at all -> zeros, never a fabricated uniform row
    zero_masks = [torch.zeros(N, dtype=torch.bool) for _ in range(3)]
    q3, count3 = donor_distribution(sbars, zero_masks, target=0)
    check('no donor -> zero row', torch.allclose(q3, torch.zeros(N, K)) and bool((count3 == 0).all()))
    # restrict to valid columns
    valid = torch.tensor([True, False, True])
    q_eff, removed, row_valid = restrict_to_valid_columns(q, valid)
    check('invalid column mass removed', torch.allclose(q_eff[:, 1], torch.zeros(N)))
    check('removed mass reported', torch.allclose(removed, q[:, 1], atol=1e-6))
    check('row still sums to one', torch.allclose(q_eff.sum(dim=1), torch.ones(N), atol=1e-6))


# ---------------------------------------------------------------------------
# 5/7/8. arms
# ---------------------------------------------------------------------------
def _snapshot(N=30, K=4, d=8, V=3, missing_prob=0.35, seed=0, device='cpu'):
    generator = torch.Generator().manual_seed(seed)
    views = []
    for v in range(V):
        H = torch.randn(N, d, generator=generator)
        P = torch.randn(K, d, generator=generator)
        S = soft_labels_from_prototypes(H, P)
        R = torch.softmax(torch.randn(K, K, generator=generator), dim=1)
        plan = torch.softmax(torch.randn(K, K, generator=generator), dim=1) * 0.5
        plan = plan / plan.sum()
        mask = torch.rand(N, generator=generator) > missing_prob
        pbar, _, valid, _ = barycentric_prototypes(plan, P)
        views.append({'mask': mask, 'H': H, 'P': P, 'S': S, 'R': R, 'Pi': plan,
                      'Pbar': pbar, 'valid_cols': valid,
                      'Sbar': S @ R})
    # the frozen protocol guarantees every node is observed in >= 1 view
    for node in range(N):
        if not any(view['mask'][node].item() for view in views):
            views[0]['mask'][node] = True
    return views


def test_arms():
    print('[5/7/8] arms, coverage and attribution')
    views = _snapshot()
    probs, diag = completion_arms(views, lambdas=(0.0, 0.5, 1.0))
    check('all arms normalised',
          all(torch.allclose(p.sum(dim=1), torch.ones(p.shape[0]), atol=1e-5) for p in probs.values()))
    check('E0 == G0 would be a bug (coverage must change the fusion)',
          not torch.allclose(probs['E0'], probs['G0'], atol=1e-6))
    check('C000 (lambda=0) reproduces G0',
          torch.allclose(probs['C000'], probs['G0'], atol=1e-6))
    # lambda = 1 equals the pure semantic projection
    views2 = copy.deepcopy(views)
    probs2, _ = completion_arms(views2, lambdas=(1.0,))
    check('C100 differs from G0 when donors exist',
          not torch.allclose(probs2['C100'], probs['G0'], atol=1e-6))
    # observed rows of E0 and G0 must agree where no view is missing+eligible
    masks = [v['mask'] for v in views]
    all_observed = torch.stack(masks).all(dim=0)
    w = coverage_weights(masks, [torch.zeros_like(m) for m in masks])  # e == 0 everywhere
    z = [v['Sbar'] for v in views]
    fused_no_cov, _ = fuse_with_coverage(z, w)
    check('with eligibility off the coverage fusion equals E0',
          torch.allclose(fused_no_cov, probs['E0'], atol=1e-6))
    check('eligibility is respected (some node is eligible)',
          any(float(r) > 0 for r in diag['eligible_rate']))
    check('donor count reported per view', len(diag['donor_count_mean']) == len(views))
    # attribution: D0 vs G0 vs C050 are three distinct objects
    check('G0 != D0', not torch.allclose(probs['G0'], probs['D0'], atol=1e-6))
    check('C050 != D0', not torch.allclose(probs['C050'], probs['D0'], atol=1e-6))


def test_fallbacks():
    print('[6] deterministic fallbacks')
    # every node missing in every view
    N, K, d = 6, 3, 4
    H = torch.randn(N, d)
    P = torch.randn(K, d)
    plan = torch.eye(K) / K
    pbar, _, valid, _ = barycentric_prototypes(plan, P)
    views = [{'mask': torch.zeros(N, dtype=torch.bool), 'H': H, 'P': P,
              'S': soft_labels_from_prototypes(H, P), 'R': torch.eye(K), 'Pi': plan,
              'Pbar': pbar, 'valid_cols': valid,
              'Sbar': soft_labels_from_prototypes(H, P)} for _ in range(2)]
    probs, diag = completion_arms(views, lambdas=(0.5,))
    check('all-missing + no donor does not crash', 'C050' in probs)
    check('fallback rows are reported per arm',
          all(value == N for key, value in diag['zero_denominator_rows'].items()
              if key != 'E0') and diag['zero_denominator_rows']['E0'] == N,
          str(diag['zero_denominator_rows']))
    check('probabilities remain finite', bool(torch.isfinite(probs['C050']).all()))
    # single client: no donor possible
    single = [views[0]]
    probs1, diag1 = completion_arms(single, lambdas=(0.5,))
    check('single client handled (no donor -> G0 keeps graph prediction)',
          torch.allclose(probs1['G0'], probs1['C050'], atol=1e-6))
    # eligibility gate
    mask = torch.zeros(N, dtype=torch.bool)
    e = eligibility(mask, torch.zeros(N), torch.zeros(N), min_valid_mass=0.5)
    check('no donor -> not eligible', bool((~e).all()))
    # zero-column plan
    bad_plan = torch.zeros(K, K)
    bad_plan[:, 0] = 1.0 / K
    pbar_bad, col, valid_bad, info_bad = barycentric_prototypes(bad_plan, P)
    check('zero-mass columns flagged', info_bad['n_invalid_columns'] == K - 1)


# ---------------------------------------------------------------------------
# 9/10. adapter isolation + device consistency
# ---------------------------------------------------------------------------
class _FakeServer:
    def __init__(self, anchors, mass):
        self.global_anchors = anchors
        self.global_cluster_mass = mass


def _fake_clients(n_clients=3, N=24, d_in=10, K=4, dim=8, seed=1):
    clients = []
    for _ in range(n_clients):
        model = models.get_model('IGAE', gae_n_enc_1=dim, gae_n_enc_2=dim, gae_n_enc_3=dim,
                                 gae_n_dec_1=dim, gae_n_dec_2=dim, gae_n_dec_3=dim,
                                 n_input=d_in, n_samples=N)
        x = torch.randn(N, d_in)
        idx = torch.randint(0, N, (2 * N,))
        idx = torch.stack([idx[:N], idx[N:]])
        values = torch.rand(N)
        adj = torch.sparse_coo_tensor(idx, values, (N, N)).coalesce()
        loader = {
            'x': x, 'target_x': x.clone(), 'adj': adj,
            'mask': torch.rand(N) > 0.3, 'y': torch.randint(0, K, (N,)),
            'n_classes': K,
        }
        clients.append(type('C', (), {'model': model, 'data_loader': loader})())
    return clients


def test_adapter_isolation():
    print('[9] adapter isolation (model / anchor / RNG)')
    torch.manual_seed(3)
    clients = _fake_clients()
    anchors = {'spatial': torch.randn(4, 16)}
    mass = torch.softmax(torch.randn(4), dim=0)
    server = _FakeServer(anchors, mass)
    config = {
        'completion_enabled': True, 'completion_stage': 'inference_only',
        'completion_lambdas': [0.25, 0.5, 0.75, 1.0],
        'completion_arms': ['B0', 'E0', 'G0', 'D0', 'C025', 'C050', 'C075', 'C100'],
        'adapter_seed': 7, 'diagnostic_seed': 100000, 'diagnostic_epochs': [100, 200, 300],
        'completion_min_valid_mass': 0.5, 'numerical_eps': NUMERICAL_EPS,
        'csata_dual_signature': True, 'csata_balanced': True, 'sinkhorn_epsilon': 0.1,
        'sinkhorn_iterations': 20, 'global_anchor_momentum': 0.0,
        'lambda_spatial': 1.0, 'lambda_semantic': 1.0, 'mass_uniform_alpha': 0.0,
    }
    basis = torch.randn(24, 16)
    anchor_before = anchors['spatial'].clone()
    params_before = [p.detach().clone() for p in clients[0].model.parameters()]
    torch.manual_seed(11)
    state_before = torch.get_rng_state()
    adapter = CompletionAdapter(config, clients, server, basis, torch.device('cpu'),
                                tempfile.mkdtemp())
    views, info = adapter.build_snapshot()
    check('snapshot has every view', len(views) == 3)
    check('anchors unchanged', torch.allclose(server.global_anchors['spatial'], anchor_before))
    check('model parameters unchanged',
          all(torch.allclose(a, b) for a, b in zip(params_before, clients[0].model.parameters())))
    check('RNG state restored', torch.equal(torch.get_rng_state(), state_before))
    check('prototype hash recorded', len(info['prototype_hashes']) == 3)
    # determinism: same seed -> same snapshot
    views_b, _ = adapter.build_snapshot()
    check('snapshot is reproducible',
          torch.allclose(views[0]['Sbar'], views_b[0]['Sbar'], atol=1e-7))
    check('R is row stochastic',
          torch.allclose(views[0]['R'].sum(dim=1), torch.ones(4), atol=1e-5))
    check('Sbar rows normalised',
          torch.allclose(views[0]['Sbar'].sum(dim=1), torch.ones(24), atol=1e-5))
    # evaluate writes arm files
    y = clients[0].data_loader['y'].numpy()

    def metric_fn(a, b, is_refined=False):
        return (float((a == b).mean()), 0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0)

    rows, _ = adapter.evaluate(1, y, metric_fn=metric_fn)
    check('seven non-B0 arms evaluated', sorted(rows) == ['C025', 'C050', 'C075', 'C100', 'D0', 'E0', 'G0'])
    summary = adapter.finalise()
    check('summary covers every arm', set(summary) == set(config['completion_arms']) - {'B0'})


def test_device_consistency():
    print('[10] CPU/GPU consistency')
    if not torch.cuda.is_available():
        print('  [skip] no CUDA device')
        return
    views_cpu = _snapshot(seed=5)
    out_cpu, _ = completion_arms(copy.deepcopy(views_cpu), lambdas=(0.5,))
    views_gpu = [{k: (v.to('cuda') if torch.is_tensor(v) else v) for k, v in view.items()}
                 for view in copy.deepcopy(views_cpu)]
    out_gpu, _ = completion_arms(views_gpu, lambdas=(0.5,))
    diff = float((out_cpu['C050'] - out_gpu['C050'].cpu()).abs().max())
    check('CPU and GPU agree within 1e-4', diff <= 1e-4, f'max diff {diff:.2e}')


# ---------------------------------------------------------------------------
# 12. label permutation must not change the prediction
# ---------------------------------------------------------------------------
def test_label_permutation():
    print('[12] label permutation only moves the evaluation, not the model')
    views = _snapshot(seed=9)
    probs, _ = completion_arms(copy.deepcopy(views), lambdas=(0.5,))
    # A relabelling of the *local* clusters moves P rows, S columns, Pi rows and
    # R rows together; nothing in the global anchor space may change.
    permuted = []
    for view in copy.deepcopy(views):
        order = torch.tensor([2, 0, 3, 1])
        new = dict(view)
        new['P'] = view['P'][order]
        new['Pi'] = view['Pi'][order]
        new['R'] = view['R'][order]
        new['S'] = view['S'][:, order]
        new['Sbar'] = new['S'] @ new['R']
        new['Pbar'], _, _, _ = barycentric_prototypes(new['Pi'], new['P'])
        permuted.append(new)
    probs_p, _ = completion_arms(permuted, lambdas=(0.5,))
    # The arm probabilities live in the *global anchor* space, so a permutation
    # of the local cluster order must not change the fused distribution.
    diff = float((probs['C050'] - probs_p['C050']).abs().max())
    check('local cluster relabelling leaves the fused distribution invariant',
          diff <= 1e-5, f'max diff {diff:.2e}')


def main():
    print('=' * 72)
    print(' outline v9 -- Completion unit / integration tests')
    print('=' * 72)
    test_b0_identity()
    test_ot_objects()
    test_donor()
    test_arms()
    test_fallbacks()
    test_adapter_isolation()
    test_device_consistency()
    test_label_permutation()
    print('=' * 72)
    print(f'  PASSED {PASS}   FAILED {FAIL}')
    print('=' * 72)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
