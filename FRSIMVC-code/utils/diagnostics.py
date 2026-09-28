"""CSATA / clustering diagnostics for the ablation study.

Implements the quantities requested by the v2 outline:

* ``C_T`` transport confidence      = (1/K) * sum_k max_j T_kj
* ``H_T`` transport entropy         = -(1/K) * sum_kj T_kj log(T_kj + eps)
* ``D_inter`` signature separation  = mean pairwise L2 distance between the K
  spatial signatures of a view (averaged over views)
* ``D_G`` graph diagnostic          = algebraic connectivity / components /
  edge-retention of the fused vs reference graph (already produced by
  ``utils.graph_metrics``; surfaced here for a single report)

plus the clustering level of detail the outline asks for:
confusion matrix, per-cluster sample counts, and per-class / per-cluster
recall and purity, which is what tells apart "label permutation" from real
cluster mixing / splitting.

Outline-v5 changes
------------------

* ``C_T`` / ``H_T`` are kept for **descriptive** purposes only.  The BSSAT
  results show ``r(C_T, ACC) < 0`` and ``r(H_T, ACC) > 0``, so a sharper, more
  "confident" transport corresponds to *lower* ACC; they must never be used as
  a "higher is better" quality score.
* The quantity formerly called *collapse* is renamed **cluster concentration**
  (``proportion_concentration``), because it correlates *positively* with ACC
  (Trento +0.54, MUUFL +0.45): it describes how skewed the cluster sizes are,
  not how healthy the run is.
"""

import numpy as np
import torch


def transport_confidence(transport, eps=1e-12):
    transport = transport.detach()
    if transport.numel() == 0:
        return 0.0
    return float(transport.max(dim=1).values.mean())


def transport_entropy(transport, eps=1e-12):
    transport = transport.detach()
    if transport.numel() == 0:
        return 0.0
    return float(-(transport * torch.log(transport + eps)).sum(dim=1).mean())


def signature_separation(signatures):
    """D_inter: mean pairwise Euclidean distance between cluster signatures."""
    values = []
    for signature in signatures or []:
        signature = signature.detach().float()
        k = signature.shape[0]
        if k < 2:
            continue
        distances = torch.cdist(signature, signature, p=2)
        off_diagonal = distances.sum() - torch.diagonal(distances).sum()
        values.append(float(off_diagonal / (k * (k - 1))))
    return float(np.mean(values)) if values else 0.0


def alignment_diagnostics(signatures, transports):
    payload = {
        'signature_separation_D_inter': signature_separation(signatures),
        'transport_confidence_C_T': [],
        'transport_entropy_H_T': [],
    }
    for transport in transports or []:
        payload['transport_confidence_C_T'].append(transport_confidence(transport))
        payload['transport_entropy_H_T'].append(transport_entropy(transport))
    if payload['transport_confidence_C_T']:
        payload['transport_confidence_C_T_mean'] = float(np.mean(payload['transport_confidence_C_T']))
        payload['transport_entropy_H_T_mean'] = float(np.mean(payload['transport_entropy_H_T']))
    else:
        payload['transport_confidence_C_T_mean'] = 0.0
        payload['transport_entropy_H_T_mean'] = 0.0
    return payload


def proportion_concentration(proportions, n_clusters=None, eps=1e-12):
    """Cluster-concentration diagnostics of a proportion vector (outline v5).

    The outline renames the old *collapse* statistic to **cluster
    concentration** and turns it into a description rather than a KPI, because
    it correlates positively with ACC on both datasets.  The same four numbers
    are used for two different bases:

    * the best round's predicted **cluster proportions** ``p_k = n_k / N``
      (-> the ``cluster_*`` keys of :func:`clustering_diagnostics`), and
    * the global anchor **mass vector** ``b_k`` / ``b'_k``
      (-> ``mass_concentration_raw`` / ``mass_concentration_tempered``),

    Parameters
    ----------
    proportions : array-like
        Non-negative weights (cluster sizes ``n_k``, or anchor mass ``b_k``).
    n_clusters : int, optional
        Denominator ``K`` of the normalised entropy.  Defaults to the number of
        entries; pass the configured cluster count so the entropy stays
        comparable when some clusters come out empty.

    Returns
    -------
    dict
        ``cluster_max_ratio`` (:math:`p_{max}`), ``cluster_min_ratio``
        (:math:`p_{min}`), ``cluster_entropy`` (:math:`H_{clu}`, normalised to
        ``[0, 1]``) and ``effective_num_clusters`` (:math:`K_{eff}`).  All
        values are ``None`` when the input carries no mass at all.
    """
    values = np.asarray(proportions, dtype=np.float64).reshape(-1)
    # Empty configured clusters have p_k=0 and must count toward p_min.
    if n_clusters is not None and int(n_clusters) > values.size:
        values = np.pad(values, (0, int(n_clusters) - values.size))
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if values.size == 0 or total <= 0.0:
        return {
            'cluster_max_ratio': None,
            'cluster_min_ratio': None,
            'cluster_entropy': None,
            'effective_num_clusters': None,
        }
    p = values / total
    k_denominator = int(n_clusters) if n_clusters else int(p.size)
    k_denominator = max(k_denominator, 2)
    log_p = np.log(p + eps)
    entropy_nats = float(-(p * log_p).sum())
    return {
        'cluster_max_ratio': float(p.max()),
        'cluster_min_ratio': float(p.min()),
        'cluster_entropy': float(entropy_nats / np.log(k_denominator)),
        'effective_num_clusters': float(np.exp(entropy_nats)),
    }


def matched_labels(y_true, cluster_pred):
    """One-to-one Hungarian mapping; unmatched clusters remain incorrect.

    Use the same predicted-row / true-column convention as cal_metric.BestMap.
    Unlike BestMap this also handles more predicted than true classes.
    """
    from scipy.optimize import linear_sum_assignment
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    pred = np.asarray(cluster_pred, dtype=np.int64).reshape(-1)
    if truth.size != pred.size or truth.size == 0:
        raise ValueError('Expected nonempty equal-length labels')
    classes, gt_idx = np.unique(truth, return_inverse=True)
    clusters, pred_idx = np.unique(pred, return_inverse=True)
    size = max(classes.size, clusters.size)
    counts = np.zeros((size, size), dtype=np.int64)
    np.add.at(counts, (pred_idx, gt_idx), 1)
    rows, cols = linear_sum_assignment(-counts)
    # Unique dummy labels also preserve the original partition for NMI/ARI.
    mapping = np.arange(size, dtype=np.int64) + int(classes.max()) + 1
    for row, col in zip(rows, cols):
        if col < classes.size:
            mapping[row] = classes[col]
    return mapping[pred_idx]


def per_class_recall(y_true, cluster_pred):
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    matched = matched_labels(truth, cluster_pred)
    result = []
    for label in np.unique(truth):
        selected = truth == label
        correct = int(np.sum(matched[selected] == label))
        support = int(selected.sum())
        result.append({'gt_class': int(label), 'support': support,
                       'true_positive': correct, 'recall': correct / support})
    return result


def clustering_diagnostics(y_true, cluster_pred, n_clusters=None):
    """Confusion matrix (gt class x predicted cluster), sizes and per-class detail.

    ``cluster_pred`` must be the *raw* cluster ids (argmax of the soft
    assignment), never the Hungarian-mapped labels, so that cluster mixing and
    splitting stay visible.

    ``n_clusters`` includes empty clusters in p_min and in the entropy
    denominator, so concentrated predictions cannot hide unoccupied clusters.
    """
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    cluster_pred = np.asarray(cluster_pred).reshape(-1).astype(int)
    gt_labels = np.unique(y_true)
    cluster_labels = np.unique(cluster_pred)
    gt_index = {int(v): i for i, v in enumerate(gt_labels)}
    cl_index = {int(v): i for i, v in enumerate(cluster_labels)}

    confusion = np.zeros((gt_labels.size, cluster_labels.size), dtype=np.int64)
    for g, c in zip(y_true, cluster_pred):
        confusion[gt_index[int(g)], cl_index[int(c)]] += 1

    per_gt_class = []
    for i, g in enumerate(gt_labels):
        row = confusion[i]
        support = int(row.sum())
        per_gt_class.append({
            'gt_class': int(g),
            'support': support,
            'recall_vs_cluster': float(row.max() / support) if support else 0.0,
            'dominant_cluster': int(cluster_labels[int(row.argmax())]) if support else None,
        })

    per_cluster = []
    cluster_sizes = []
    for j, c in enumerate(cluster_labels):
        column = confusion[:, j]
        size = int(column.sum())
        cluster_sizes.append(size)
        per_cluster.append({
            'cluster': int(c),
            'size': size,
            'size_fraction': float(size / confusion.sum()) if confusion.sum() else 0.0,
            'purity': float(column.max() / size) if size else 0.0,
            'dominant_gt_class': int(gt_labels[int(column.argmax())]) if size else None,
        })

    # outline v5: cluster concentration of the best round (former "collapse").
    concentration = proportion_concentration(
        cluster_sizes, n_clusters=n_clusters or cluster_labels.size
    )

    return {
        'num_gt_classes': int(gt_labels.size),
        'num_predicted_clusters': int(cluster_labels.size),
        'gt_labels': [int(v) for v in gt_labels],
        'cluster_labels': [int(v) for v in cluster_labels],
        'confusion_matrix': confusion.tolist(),
        'cluster_sizes': {int(c): int(cluster_sizes[i]) for i, c in enumerate(cluster_labels)},
        'cluster_sizes_vector': [int(v) for v in cluster_sizes],
        'per_gt_class': per_gt_class,
        'per_class_recall': per_class_recall(y_true, cluster_pred),
        'recall_definition': 'one-to-one Hungarian mapping on this evaluation subset',
        'per_cluster': per_cluster,
        'cluster_max_ratio': concentration['cluster_max_ratio'],
        'cluster_min_ratio': concentration['cluster_min_ratio'],
        'cluster_entropy': concentration['cluster_entropy'],
        'effective_num_clusters': concentration['effective_num_clusters'],
        'concentration_basis': 'best-round predicted cluster proportions n_k / N',
        'concentration_entropy_denominator': int(n_clusters or cluster_labels.size),
    }
