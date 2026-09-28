"""CSATA -> BSSAT: cluster-level spatial/semantic signature transport.

Outline v4, module 2 -- *Balanced Spatial-Semantic Anchor Transport*.
Outline v5, step 1 -- *Tempered Global Anchor Mass* (BSSAT-v2).

Two upgrades over the v3 CSATA module:

1. **Dual signature.**  A local cluster is described by

       z_k = [ q_spatial_k || q_semantic_k ]

   * ``q_spatial = D_K^-1 S^T R U^c`` is the projection of the cluster onto the
     shared spatial basis (unchanged).
   * ``q_semantic`` is a low-dimensional, privacy-safe statistic vector
     ``[mass, assignment entropy, compactness]`` computed **on the client**, so
     no private feature matrix ever has to leave it -- only a ``K x 3``
     aggregate travels to the server.

   The transport cost becomes
   ``C_kj = lambda_s * d_spatial(k, j) + lambda_c * d_semantic(k, j)``, which
   turns pure geometric matching into spatial-semantic anchor matching.

2. **Mass-preserving (balanced) transport.**  A plain Sinkhorn plan only
   enforces row stochasticity, so several local clusters can be pushed onto the
   same global anchor.  That is the cluster-concentration signature found at A3
   (3/10 seeds concentrate on both Trento and MUUFL while A0-A2 never do).
   BSSAT instead solves the *balanced* entropic OT problem

       T^(v) = argmin_{T >= 0} <T, C^(v)> + eps * H(T)
       subject to   T 1 = a^(v)   and   T^T 1 = b

   with ``a_k^(v) = sum_i m_i^(v) S_ik^(v) / sum_i m_i^(v)`` the local cluster
   mass and ``b`` the global anchor mass.  The column constraint is what keeps
   every global anchor alive.

3. **Tempered global anchor mass (outline v5 / BSSAT-v2).**  The balanced
   column marginal ``b`` is itself estimated from the clients' own cluster
   masses, which creates a positive feedback loop: a large cluster pushes
   ``b_k`` up, which gives that anchor more transport mass, which makes the
   cluster even larger.  The v5 remedy does **not** touch the Sinkhorn solver;
   it only re-shapes the target

       b' = (1 - alpha) * b + alpha / K

   with ``alpha = 0`` recovering BSSAT-r1 exactly.  ``alpha > 0`` damps the
   dominant anchor *without* forcing a uniform assignment, so
   ``alpha in {0.05, 0.1, 0.2, 0.3}`` interpolates between "greedy mass
   matching" and "balanced anchors".  Both ``b`` (raw) and ``b'`` (tempered)
   are returned so the caller can persist ``global_mass_raw.npy`` and
   ``global_mass_tempered.npy``.
"""

import math

import torch

from torch_clustering import PyTorchKMeans


def compute_spatial_signature(soft_labels, observed_mask, spatial_basis, eps=1e-8):
    """Q^(v) = D_K^-1 S^T R U^c and its per-cluster observed mass."""
    mask = observed_mask.to(device=soft_labels.device, dtype=soft_labels.dtype).unsqueeze(1)
    masked_soft = soft_labels * mask
    cluster_mass = masked_soft.sum(dim=0)
    signature = masked_soft.t() @ spatial_basis
    signature = signature / cluster_mass.clamp_min(eps).unsqueeze(1)
    return signature, cluster_mass


def compute_semantic_stats(soft_labels, embedding, observed_mask, eps=1e-8):
    """Client-side low-dimensional cluster statistics, shape ``(K, 3)``.

    Columns:

    * ``mass``               -- observed responsibility mass, normalised to 1.
    * ``assignment entropy`` -- responsibility-weighted mean per-node
      assignment entropy.  Low means the nodes this cluster owns are
      confidently assigned; high means the cluster sits in an ambiguous region.
    * ``compactness``        -- responsibility-weighted mean squared distance of
      the owning nodes to the responsibility-weighted cluster centroid.

    Computing this on the client keeps the contract from the outline: the
    server only ever sees ``K x 3`` cluster-level aggregates, never local
    features.
    """
    device = soft_labels.device
    dtype = soft_labels.dtype
    mask = observed_mask.to(device=device, dtype=dtype).unsqueeze(1)
    masked_soft = soft_labels * mask
    mass = masked_soft.sum(dim=0)
    safe_mass = mass.clamp_min(eps)

    node_entropy = -(soft_labels * torch.log(soft_labels.clamp_min(eps))).sum(dim=1)
    node_entropy = node_entropy * observed_mask.to(device=device, dtype=dtype)
    assignment_entropy = (masked_soft.t() @ node_entropy) / safe_mass

    centroid = (masked_soft.t() @ embedding) / safe_mass.unsqueeze(1)
    sq_distance = (embedding.unsqueeze(1) - centroid.unsqueeze(0)).pow(2).sum(dim=2)
    compactness = (masked_soft * sq_distance).sum(dim=0) / safe_mass

    normalised_mass = mass / mass.sum().clamp_min(eps)
    stats = torch.stack([normalised_mass, assignment_entropy, compactness], dim=1)
    return stats.detach()


def standardize_columns(values, eps=1e-6):
    """z-score every column of a ``(rows, cols)`` tensor.

    The semantic statistics live on wildly different scales (a normalised mass
    vs. a squared embedding distance), so they are standardised before they
    enter the cost.  The global semantic anchors are maintained in this
    standardised space, which keeps the transport comparable across rounds.
    """
    if values.numel() == 0:
        return values
    mean = values.mean(dim=0, keepdim=True)
    std = values.std(dim=0, keepdim=True, unbiased=False)
    return (values - mean) / std.clamp_min(eps)


def sinkhorn_transport(cost, epsilon=0.05, iterations=100, eps=1e-12):
    """Row-stochastic entropic plan (the v3 behaviour, kept for ablation)."""
    if epsilon <= 0:
        raise ValueError('sinkhorn epsilon must be positive')
    scaled_cost = cost / cost.detach().mean().clamp_min(eps)
    n_rows, n_cols = scaled_cost.shape
    log_kernel = -scaled_cost / epsilon
    log_a = torch.full((n_rows,), -math.log(n_rows), device=cost.device, dtype=cost.dtype)
    log_b = torch.full((n_cols,), -math.log(n_cols), device=cost.device, dtype=cost.dtype)
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(iterations):
        log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
    plan = torch.exp(log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0))
    return plan / plan.sum(dim=1, keepdim=True).clamp_min(eps)


def sinkhorn_balanced(cost, row_mass, col_mass, epsilon=0.05, iterations=100, eps=1e-12):
    """Mass-preserving entropic OT plan.

    Returns ``T`` with ``T @ 1 == row_mass`` and ``T^T @ 1`` close to
    ``col_mass``.  Both marginals are renormalised to unit total mass first,
    which is what makes the problem feasible -- the caller passes raw cluster
    mass.
    """
    if epsilon <= 0:
        raise ValueError('sinkhorn epsilon must be positive')
    scaled_cost = cost / cost.detach().mean().clamp_min(eps)
    log_kernel = -scaled_cost / epsilon

    log_a = torch.log(row_mass.clamp_min(eps).to(cost.dtype))
    log_b = torch.log(col_mass.clamp_min(eps).to(cost.dtype))
    log_a = log_a - torch.logsumexp(log_a, dim=0)
    log_b = log_b - torch.logsumexp(log_b, dim=0)

    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(iterations):
        log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)

    plan = torch.exp(log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0))
    # Enforce the row marginal exactly; the column marginal then holds up to
    # the last Sinkhorn sweep, which is the standard convention.
    plan = plan / plan.sum(dim=1, keepdim=True).clamp_min(eps) * row_mass.unsqueeze(1)
    return plan


def initialize_global_anchors(signatures, n_clusters, random_state=0):
    stacked = torch.cat([signature.detach() for signature in signatures], dim=0)
    kmeans = PyTorchKMeans(
        metric='euclidean',
        init='k-means++',
        random_state=random_state,
        n_clusters=n_clusters,
        n_init=10,
        max_iter=100,
        verbose=False,
    )
    kmeans.fit_predict(stacked)
    return kmeans.cluster_centers_.detach()


def _pairwise_cost(local, anchor, eps):
    distance = torch.cdist(local, anchor, p=2).pow(2)
    return distance / distance.detach().mean().clamp_min(eps)


def _cost_matrix(spatial, semantic, anchors, lambda_spatial, lambda_semantic, eps):
    cost = lambda_spatial * _pairwise_cost(spatial, anchors['spatial'], eps)
    if semantic is not None and anchors.get('semantic') is not None and lambda_semantic > 0.0:
        cost = cost + lambda_semantic * _pairwise_cost(semantic, anchors['semantic'], eps)
    return cost


def align_to_global_anchors(
    spatial_signatures,
    semantic_signatures,
    cluster_masses,
    global_anchors,
    n_clusters,
    random_state=0,
    epsilon=0.05,
    iterations=100,
    momentum=0.0,
    lambda_spatial=1.0,
    lambda_semantic=1.0,
    balanced=True,
    global_mass=None,
    mass_uniform_alpha=0.0,
    eps=1e-8,
    return_details=False,
):
    """Client-to-global transport followed by a barycenter anchor update.

    ``global_anchors`` is ``None`` or a dict with a ``'spatial'`` entry (and,
    when the dual signature is enabled, a ``'semantic'`` entry).
    ``semantic_signatures`` may be ``None`` for the spatial-only ablation.

    ``mass_uniform_alpha`` (outline v5) tempers the balanced column marginal:
    ``b' = (1 - alpha) * b + alpha / K``.  ``alpha = 0.0`` is the original
    BSSAT-r1 behaviour; larger values damp the dominant anchor.

    ``return_details`` (outline v9 section 15.1) additionally returns a
    ``details`` dict per view with ``plan`` (the mass coupling Pi, *not* the
    conditional), ``row_mass``, ``col_mass``, ``conditionals``, the marginal
    residuals and the invalid-column mask.  It is opt-in: the default call
    keeps returning the historical six-tuple, so every existing unpack site
    stays valid.
    """
    if return_details:
        details = []
    if global_anchors is None:
        global_anchors = {
            'spatial': initialize_global_anchors(spatial_signatures, n_clusters, random_state),
        }
        if semantic_signatures is not None:
            global_anchors['semantic'] = initialize_global_anchors(
                semantic_signatures, n_clusters, random_state
            )

    if semantic_signatures is None:
        semantic_signatures = [None] * len(spatial_signatures)

    costs = [
        _cost_matrix(spatial, semantic, global_anchors, lambda_spatial, lambda_semantic, eps)
        for spatial, semantic in zip(spatial_signatures, semantic_signatures)
    ]

    normalised_masses = [mass / mass.sum().clamp_min(eps) for mass in cluster_masses]

    if balanced:
        mean_mass = torch.stack(normalised_masses).mean(dim=0)
        mean_mass = mean_mass / mean_mass.sum().clamp_min(eps)
        if global_mass is None:
            raw_global_mass = mean_mass
        else:
            raw_global_mass = momentum * global_mass + (1.0 - momentum) * mean_mass
            raw_global_mass = raw_global_mass / raw_global_mass.sum().clamp_min(eps)
        # ---- outline v5 / BSSAT-v2: tempered global anchor mass -----------
        # b' = (1 - alpha) b + alpha / K.  alpha = 0 reproduces BSSAT-r1.
        alpha = float(min(max(float(mass_uniform_alpha), 0.0), 1.0))
        if alpha > 0.0:
            n_anchors = raw_global_mass.numel()
            uniform = torch.full_like(raw_global_mass, 1.0 / float(n_anchors))
            tempered_global_mass = (1.0 - alpha) * raw_global_mass + alpha * uniform
        else:
            tempered_global_mass = raw_global_mass
        tempered_global_mass = (
            tempered_global_mass / tempered_global_mass.sum().clamp_min(eps)
        )
        plans = [
            sinkhorn_balanced(cost, mass, tempered_global_mass, epsilon, iterations, eps=eps)
            for cost, mass in zip(costs, normalised_masses)
        ]
        # T 1 = a  =>  T / a is row-stochastic, which is the operator we want.
        conditionals = [
            plan / mass.unsqueeze(1).clamp_min(eps)
            for plan, mass in zip(plans, normalised_masses)
        ]
        if return_details:
            for plan, mass in zip(plans, normalised_masses):
                row_mass = plan.sum(dim=1)
                col_mass = plan.sum(dim=0)
                details.append({
                    'plan': plan.detach(),
                    'row_mass': row_mass.detach(),
                    'col_mass': col_mass.detach(),
                    'conditionals': (plan / mass.unsqueeze(1).clamp_min(eps)).detach(),
                    'row_marginal_residual': float(
                        (row_mass - mass).abs().max().detach().cpu()
                    ),
                    'col_marginal_residual': float(
                        (col_mass - tempered_global_mass).abs().max().detach().cpu()
                    ),
                    'invalid_columns': (col_mass <= eps),
                    'target_col_mass': tempered_global_mass.detach(),
                })
    else:
        plans = [sinkhorn_transport(cost, epsilon, iterations, eps=eps) for cost in costs]
        conditionals = plans
        raw_global_mass = global_mass
        tempered_global_mass = global_mass
        if return_details:
            # v3 row-stochastic plan: Pi == R and the row mass is uniform 1/K.
            n_rows = plans[0].shape[0]
            uniform = torch.full(
                (n_rows,), 1.0 / float(n_rows),
                device=plans[0].device, dtype=plans[0].dtype,
            )
            for plan in plans:
                col_mass = plan.sum(dim=0)
                details.append({
                    'plan': plan.detach(),
                    'row_mass': uniform.detach(),
                    'col_mass': col_mass.detach(),
                    'conditionals': plan.detach(),
                    'row_marginal_residual': float(
                        (plan.sum(dim=1) - 1.0).abs().max().detach().cpu()
                    ),
                    'col_marginal_residual': None,
                    'invalid_columns': (col_mass <= eps),
                    'target_col_mass': None,
                })

    aligned_spatial = []
    aligned_semantic = []
    weights = []
    for index, plan in enumerate(plans):
        column_mass = plan.t().sum(dim=1).clamp_min(eps)
        aligned_spatial.append(
            (plan.t() @ spatial_signatures[index]) / column_mass.unsqueeze(1)
        )
        if semantic_signatures[index] is not None and global_anchors.get('semantic') is not None:
            aligned_semantic.append(
                (plan.t() @ semantic_signatures[index]) / column_mass.unsqueeze(1)
            )
        weights.append(plan.sum().clamp_min(eps))

    weights = torch.stack(weights)
    weights = weights / weights.sum().clamp_min(eps)
    barycenter_spatial = sum(
        weight * signature for weight, signature in zip(weights, aligned_spatial)
    )
    updated_anchors = {
        'spatial': momentum * global_anchors['spatial']
        + (1.0 - momentum) * barycenter_spatial,
    }
    if aligned_semantic:
        barycenter_semantic = sum(
            weight * signature for weight, signature in zip(weights, aligned_semantic)
        )
        updated_anchors['semantic'] = (
            momentum * global_anchors['semantic']
            + (1.0 - momentum) * barycenter_semantic
        )

    alignment_loss = torch.stack([
        (plan * cost).sum() / plan.shape[0] for plan, cost in zip(plans, costs)
    ]).mean()

    if return_details:
        return (
            updated_anchors,
            conditionals,
            costs,
            alignment_loss.detach(),
            raw_global_mass,
            tempered_global_mass,
            details,
        )
    return (
        updated_anchors,
        conditionals,
        costs,
        alignment_loss.detach(),
        raw_global_mass,
        tempered_global_mass,
    )


def align_soft_labels(soft_labels, transport, eps=1e-8):
    aligned = soft_labels @ transport
    return aligned / aligned.sum(dim=1, keepdim=True).clamp_min(eps)


def global_semantic_consensus(soft_labels_list, transports, observed_masks, eps=1e-8):
    numerator = torch.zeros_like(soft_labels_list[0])
    denominator = torch.zeros(
        soft_labels_list[0].shape[0], 1,
        device=soft_labels_list[0].device,
        dtype=soft_labels_list[0].dtype,
    )
    aligned_list = []
    for soft_labels, transport, observed_mask in zip(
        soft_labels_list, transports, observed_masks
    ):
        aligned = align_soft_labels(soft_labels, transport)
        aligned_list.append(aligned)
        mask = observed_mask.to(aligned.dtype).unsqueeze(1)
        numerator = numerator + aligned * mask
        denominator = denominator + mask
    if torch.any(denominator == 0):
        raise RuntimeError('At least one sample is missing in every view.')
    consensus = numerator / denominator.clamp_min(eps)
    consensus = consensus / consensus.sum(dim=1, keepdim=True).clamp_min(eps)
    return consensus, aligned_list
