"""Cross-Client Semantic-Spatial Completion (outline v9, module III).

Everything in this file is a **pure function**: no global config, no ground
truth, no model, no RNG of its own.  Every helper states its tensor contract
explicitly and degrades deterministically on degenerate input.

Notation (outline v9 section 7/8)
--------------------------------
N nodes, V clients (one client == one view), K clusters, d_v latent dim.

* ``m^(v) in {0,1}^N``    node observation mask of view v (1 = observed).
* ``H^(v) in R^{N x d}``  frozen-model embedding (the *graph* representation).
* ``P^(v) in R^{K x d}``  local prototypes (fit on observed nodes only).
* ``S^(v) in R^{N x K}``  local soft assignment, softmax(-d(H, P)).
* ``Pi^(v) in R^{K x K}`` OT **mass coupling** (rows = local clusters of v,
  columns = global anchors).  ``a_k = sum_j Pi_kj``, ``b_j = sum_k Pi_kj``.
* ``R^(v) in R^{K x K}``  **row-conditional** map, ``R = diag(1/a) Pi``.
* ``Sbar^(v) = S^(v) R^(v)``  aligned soft labels (global anchor order).
* ``Pbar^(v)_j = sum_k Pi^(v)_kj P^(v)_k / b^(v)_j``  target-side prototype of
  global anchor j, expressed **in the target client's own latent space**.

Completion
----------
``q_i^(-v) = sum_{u != v} m_i^(u) Sbar_i^(u) / sum_{u != v} m_i^(u)``  (donor
distribution, leave-one-view-out, observed donors only), then

``h_sem = q_eff Pbar`` ,  ``h_comp = (1 - lambda) h_graph + lambda h_sem``,

recompute ``S_comp = softmax(-d(h_comp, P))`` and ``Sbar_comp = S_comp R`` on
the *missing but eligible* nodes only, and fuse with the explicit coverage
weight ``w_i^(v) = m_i^(v) + (1 - m_i^(v)) e_i^(v)``.
"""

from __future__ import annotations

import torch

#: Division guard for mass normalisation (outline v9 section 8.1).  Columns
#: whose mass is below this are *invalid*; we never divide by them.
NUMERICAL_EPS = 1e-12


# ---------------------------------------------------------------------------
# OT plan bookkeeping
# ---------------------------------------------------------------------------
def plan_masses(plan, eps=NUMERICAL_EPS):
    """Row mass ``a``, column mass ``b`` and the valid-column mask of ``Pi``.

    Parameters
    ----------
    plan : (K, K) float tensor, non-negative.

    Returns
    -------
    (row_mass (K,), col_mass (K,), valid_cols (K,) bool)
    """
    if plan.dim() != 2 or plan.shape[0] != plan.shape[1]:
        raise ValueError(f'plan must be square (K, K), got {tuple(plan.shape)}')
    row_mass = plan.sum(dim=1)
    col_mass = plan.sum(dim=0)
    valid_cols = col_mass > eps
    return row_mass, col_mass, valid_cols


def row_conditionals(plan, eps=NUMERICAL_EPS):
    """R = diag(1/a) Pi with rows renormalised to sum to 1 (zero rows -> 0)."""
    row_mass, _, _ = plan_masses(plan, eps=eps)
    safe = row_mass.clamp_min(eps).unsqueeze(1)
    conditionals = plan / safe
    # A zero-mass row would produce 0/eps = 0 already; force exact zeros.
    conditionals = torch.where(
        row_mass.unsqueeze(1) > eps, conditionals, torch.zeros_like(conditionals)
    )
    return conditionals


def marginal_residuals(plan, target_row_mass=None, target_col_mass=None, eps=NUMERICAL_EPS):
    """max |Pi 1 - a| and max |Pi^T 1 - b| (None target -> skipped)."""
    out = {}
    if target_row_mass is not None:
        out['row_marginal_residual'] = float(
            (plan.sum(dim=1) - target_row_mass).abs().max().detach().cpu()
        )
    if target_col_mass is not None:
        out['col_marginal_residual'] = float(
            (plan.sum(dim=0) - target_col_mass).abs().max().detach().cpu()
        )
    return out


# ---------------------------------------------------------------------------
# barycentric prototypes
# ---------------------------------------------------------------------------
def barycentric_prototypes(plan, prototypes, eps=NUMERICAL_EPS):
    """Column-mass barycenter of the *target client's own* prototypes.

    ``Pbar_j = sum_k Pi_kj P_k / b_j`` with ``b_j = sum_k Pi_kj`` (the actual
    column mass of the realised plan, not the requested marginal).  Invalid
    columns (``b_j < eps``) get a zero prototype and are reported.

    Parameters
    ----------
    plan : (K, K) float tensor.
    prototypes : (K, d) float tensor.

    Returns
    -------
    (Pbar (K, d), col_mass (K,), valid_cols (K,) bool, info dict)
    """
    if prototypes.dim() != 2:
        raise ValueError(f'prototypes must be (K, d), got {tuple(prototypes.shape)}')
    if plan.shape[0] != prototypes.shape[0]:
        raise ValueError(
            f'plan rows ({plan.shape[0]}) must match prototype rows ({prototypes.shape[0]})'
        )
    _, col_mass, valid_cols = plan_masses(plan, eps=eps)
    safe = torch.where(valid_cols, col_mass, torch.ones_like(col_mass))
    numerator = plan.t() @ prototypes
    pbar = numerator / safe.unsqueeze(1)
    pbar = torch.where(valid_cols.unsqueeze(1), pbar, torch.zeros_like(pbar))
    info = {
        'n_invalid_columns': int((~valid_cols).sum().detach().cpu()),
        'col_mass_min': float(col_mass.min().detach().cpu()),
        'col_mass_max': float(col_mass.max().detach().cpu()),
    }
    return pbar, col_mass, valid_cols, info


# ---------------------------------------------------------------------------
# leave-one-view-out donor
# ---------------------------------------------------------------------------
def donor_distribution(aligned_list, masks, target, eps=NUMERICAL_EPS):
    """q_i^(-v) from the *other* views' observed aligned soft labels.

    ``q_i = sum_{u != v} m_i^(u) Sbar_i^(u) / sum_{u != v} m_i^(u)``

    Only original observations are donors; completions are never fed back into
    the pool.  Rows with ``donor_count == 0`` are returned as exact zeros
    (never a fabricated uniform row).

    Returns
    -------
    (q_raw (N, K), donor_count (N,))
    """
    if not aligned_list:
        raise ValueError('donor_distribution needs at least one donor view')
    reference = aligned_list[0]
    device, dtype = reference.device, reference.dtype
    n_nodes, n_clusters = reference.shape
    numerator = torch.zeros((n_nodes, n_clusters), device=device, dtype=dtype)
    denominator = torch.zeros((n_nodes, 1), device=device, dtype=dtype)
    for index, (aligned, mask) in enumerate(zip(aligned_list, masks)):
        if index == target:
            continue
        weight = mask.to(dtype=dtype).unsqueeze(1)
        numerator = numerator + aligned * weight
        denominator = denominator + weight
    donor_count = denominator.squeeze(1)
    safe = donor_count.clamp_min(eps).unsqueeze(1)
    q_raw = numerator / safe
    q_raw = torch.where(donor_count.unsqueeze(1) > eps, q_raw, torch.zeros_like(q_raw))
    return q_raw, donor_count


def restrict_to_valid_columns(q_raw, valid_cols, eps=NUMERICAL_EPS):
    """Zero the donor mass on invalid anchors and renormalise.

    Returns ``(q_eff (N, K), removed_mass (N,), row_valid (N,) bool)``.  A row
    whose remaining mass is 0 is flagged invalid so the caller can fall back.
    """
    mask = valid_cols.to(device=q_raw.device, dtype=q_raw.dtype)
    kept = q_raw * mask.unsqueeze(0)
    remaining = kept.sum(dim=1)
    removed_mass = q_raw.sum(dim=1) - remaining
    safe = remaining.clamp_min(eps).unsqueeze(1)
    q_eff = kept / safe
    row_valid = remaining > eps
    q_eff = torch.where(row_valid.unsqueeze(1), q_eff, torch.zeros_like(q_eff))
    return q_eff, removed_mass, row_valid


# ---------------------------------------------------------------------------
# representation combination
# ---------------------------------------------------------------------------
def semantic_projection(q_eff, pbar):
    """h_sem = q_eff @ Pbar -> (N, d) in the target latent space."""
    if q_eff.shape[1] != pbar.shape[0]:
        raise ValueError(
            f'q_eff columns ({q_eff.shape[1]}) must match Pbar rows ({pbar.shape[0]})'
        )
    return q_eff @ pbar


def convex_combination(h_graph, h_sem, lam):
    """h_comp = (1 - lambda) h_graph + lambda h_sem (no normalisation)."""
    lam = float(lam)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f'lambda must be in [0, 1], got {lam}')
    return (1.0 - lam) * h_graph + lam * h_sem


def soft_labels_from_prototypes(hidden, prototypes, tau=1.0):
    """S = softmax(-d(h, P) / tau); tau=1 reproduces the frozen head."""
    if tau <= 0.0:
        raise ValueError(f'tau must be positive, got {tau}')
    distances = torch.cdist(hidden, prototypes, p=2)
    return torch.softmax(-distances / float(tau), dim=1)


# ---------------------------------------------------------------------------
# eligibility / coverage fusion
# ---------------------------------------------------------------------------
def eligibility(mask, donor_count, removed_mass, min_valid_mass=0.5):
    """e_i = 1 iff the node is missing, has a donor and a usable mapping."""
    missing = ~mask.to(dtype=torch.bool)
    has_donor = donor_count > 0
    enough = (1.0 - removed_mass) >= float(min_valid_mass)
    return missing & has_donor & enough


def coverage_weights(masks, eligible_list):
    """w^(v) = m^(v) + (1 - m^(v)) e^(v)  (float, shape (N, 1))."""
    weights = []
    for mask, eligible in zip(masks, eligible_list):
        m = mask.to(dtype=torch.float32)
        e = eligible.to(dtype=torch.float32)
        weights.append((m + (1.0 - m) * e).unsqueeze(1))
    return weights


def fuse_with_coverage(z_list, weights, eps=NUMERICAL_EPS):
    """sum_v w^(v) Z^(v) / sum_v w^(v), with an explicit zero-denominator fallback.

    Rows whose total coverage weight is 0 (a node missing in *every* view and
    never eligible) fall back to the unweighted mean of the available Z; the
    count is reported instead of being silently patched.
    """
    numerator = torch.zeros_like(z_list[0])
    denominator = torch.zeros(
        (z_list[0].shape[0], 1), device=z_list[0].device, dtype=z_list[0].dtype
    )
    for z, w in zip(z_list, weights):
        numerator = numerator + z * w.to(dtype=z.dtype)
        denominator = denominator + w.to(dtype=z.dtype)
    zero_rows = denominator.squeeze(1) <= eps
    n_zero = int(zero_rows.sum().detach().cpu())
    safe = denominator.clone()
    if n_zero:
        fallback = torch.stack(z_list, dim=0).mean(dim=0)
        numerator = torch.where(zero_rows.unsqueeze(1), fallback, numerator)
        safe = torch.where(zero_rows.unsqueeze(1), torch.ones_like(denominator), denominator)
    fused = numerator / safe.clamp_min(eps)
    fused = fused / fused.sum(dim=1, keepdim=True).clamp_min(eps)
    return fused, n_zero


def masked_average(aligned_list, masks, eps=NUMERICAL_EPS):
    """Original B0/E0 fusion: observed views only (used for the E0 control).

    The frozen protocol guarantees ``ensure_node_coverage``, i.e. every node is
    observed in at least one view, so the degenerate branch never fires in the
    main experiment.  Should it ever fire (extreme unit tests) we fall back to
    the unweighted mean instead of raising, and the caller records the count --
    the outline asks for an explicit fallback plus a log, not a silently
    patched mask.
    """
    numerator = torch.zeros_like(aligned_list[0])
    denominator = torch.zeros(
        (aligned_list[0].shape[0], 1),
        device=aligned_list[0].device,
        dtype=aligned_list[0].dtype,
    )
    for aligned, mask in zip(aligned_list, masks):
        weight = mask.to(dtype=aligned.dtype).unsqueeze(1)
        numerator = numerator + aligned * weight
        denominator = denominator + weight
    zero_rows = denominator.squeeze(1) <= eps
    n_zero = int(zero_rows.sum().detach().cpu())
    safe = denominator.clone()
    if n_zero:
        fallback = torch.stack(aligned_list, dim=0).mean(dim=0)
        numerator = torch.where(zero_rows.unsqueeze(1), fallback, numerator)
        safe = torch.where(zero_rows.unsqueeze(1), torch.ones_like(denominator), denominator)
    consensus = numerator / safe.clamp_min(eps)
    consensus = consensus / consensus.sum(dim=1, keepdim=True).clamp_min(eps)
    return consensus


# ---------------------------------------------------------------------------
# arm construction
# ---------------------------------------------------------------------------
def _scatter_rows(base, index, values):
    out = base.clone()
    if index.numel():
        out = out.index_put((index,), values)
    return out


def completion_arms(snapshot, lambdas=(0.25, 0.5, 0.75, 1.0), min_valid_mass=0.5,
                    include=('E0', 'G0', 'D0'), eps=NUMERICAL_EPS):
    """Build every arm's final probability from one frozen snapshot.

    Parameters
    ----------
    snapshot : list of per-view dicts, one per client, each with
        ``{'mask' (N,) bool, 'Sbar' (N,K), 'H' (N,d), 'P' (K,d), 'R' (K,K),
           'Pbar' (K,d), 'valid_cols' (K,) bool}``.
    lambdas : iterable of the tested completion strengths.

    Returns
    -------
    (probabilities dict name -> (N, K), diagnostics dict)
    """
    views = len(snapshot)
    masks = [item['mask'] for item in snapshot]
    sbar_list = [item['Sbar'] for item in snapshot]

    q_raw_by_view = []
    donor_count_by_view = []
    q_eff_by_view = []
    removed_by_view = []
    row_valid_by_view = []
    eligible_by_view = []
    for view in range(views):
        q_raw, donor_count = donor_distribution(sbar_list, masks, target=view, eps=eps)
        q_eff, removed, row_valid = restrict_to_valid_columns(
            q_raw, snapshot[view]['valid_cols'], eps=eps
        )
        eligible = eligibility(
            masks[view], donor_count, removed, min_valid_mass=min_valid_mass
        )
        q_raw_by_view.append(q_raw)
        donor_count_by_view.append(donor_count)
        q_eff_by_view.append(q_eff)
        removed_by_view.append(removed)
        row_valid_by_view.append(row_valid)
        eligible_by_view.append(eligible)

    weights = coverage_weights(masks, eligible_by_view)

    # Z at missing+eligible nodes, per arm family --------------------------
    z_graph = []
    z_donor = []
    z_comp = {lam: [] for lam in lambdas}
    sem_stats = []
    for view, item in enumerate(snapshot):
        index = torch.nonzero(eligible_by_view[view], as_tuple=False).squeeze(1)
        sbar = item['Sbar']
        # G0: the frozen graph representation's own prediction (Sbar as is).
        z_graph.append(sbar)
        # D0: the donor distribution itself (already in anchor space).
        if index.numel():
            donor_rows = q_eff_by_view[view].index_select(0, index)
        else:
            donor_rows = q_eff_by_view[view].new_zeros((0, q_eff_by_view[view].shape[1]))
        z_donor.append(_scatter_rows(sbar, index, donor_rows))

        h_graph_rows = item['H'].index_select(0, index)
        h_sem_rows = semantic_projection(
            donor_rows, item['Pbar']
        ) if index.numel() else item['H'].new_zeros((0, item['H'].shape[1]))
        if index.numel():
            sem_stats.append({
                'view': view,
                'n_eligible': int(index.numel()),
                'h_graph_norm_mean': float(h_graph_rows.norm(dim=1).mean().detach().cpu()),
                'h_sem_norm_mean': float(h_sem_rows.norm(dim=1).mean().detach().cpu()),
                'cosine_mean': float(
                    torch.nn.functional.cosine_similarity(
                        h_graph_rows, h_sem_rows, dim=1
                    ).mean().detach().cpu()
                ),
                'l2_distance_mean': float(
                    (h_graph_rows - h_sem_rows).norm(dim=1).mean().detach().cpu()
                ),
            })
        else:
            sem_stats.append({'view': view, 'n_eligible': 0})
        for lam in lambdas:
            if index.numel():
                h_comp_rows = convex_combination(h_graph_rows, h_sem_rows, lam)
                s_comp = soft_labels_from_prototypes(h_comp_rows, item['P'])
                sbar_comp = s_comp @ item['R']
                sbar_comp = sbar_comp / sbar_comp.sum(dim=1, keepdim=True).clamp_min(eps)
            else:
                sbar_comp = sbar.new_zeros((0, sbar.shape[1]))
            z_comp[lam].append(_scatter_rows(sbar, index, sbar_comp))

    probabilities = {}
    zero_counts = {}
    if 'E0' in include:
        probabilities['E0'] = masked_average(sbar_list, masks, eps=eps)
        zero_counts['E0'] = int(
            (~torch.stack(masks, dim=0).any(dim=0)).sum().detach().cpu()
        )
    if 'G0' in include:
        probabilities['G0'], zero_counts['G0'] = fuse_with_coverage(z_graph, weights, eps=eps)
    if 'D0' in include:
        probabilities['D0'], zero_counts['D0'] = fuse_with_coverage(z_donor, weights, eps=eps)
    for lam in lambdas:
        name = 'C%03d' % int(round(float(lam) * 100))
        probabilities[name], zero_counts[name] = fuse_with_coverage(
            z_comp[lam], weights, eps=eps
        )

    diagnostics = {
        'donor_count_mean': [
            float(t.detach().cpu().to(dtype=torch.float32).mean()) for t in donor_count_by_view
        ],
        'removed_mass_mean': [
            float(t.detach().cpu().mean()) for t in removed_by_view
        ],
        'eligible_rate': [
            float(t.to(dtype=torch.float32).mean().detach().cpu()) for t in eligible_by_view
        ],
        'missing_rate_nodes': [
            float((~m).to(dtype=torch.float32).mean().detach().cpu()) for m in masks
        ],
        'row_valid_rate': [
            float(t.to(dtype=torch.float32).mean().detach().cpu()) for t in row_valid_by_view
        ],
        'zero_denominator_rows': zero_counts,
        'semantic_projection': sem_stats,
        'lambdas': [float(x) for x in lambdas],
        'min_valid_mass': float(min_valid_mass),
    }
    return probabilities, diagnostics
