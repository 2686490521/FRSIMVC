from collections import Counter
from pathlib import Path
import copy
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
import torch
from sklearn.decomposition import PCA

from utils.graph_metrics import (
    distribution_summary,
    edge_retention_ratio,
    graph_degree_stats,
    graph_metrics,
    graph_relative_frobenius_change,
    node_mean_incident_weight,
    node_feature_edge_loss,
    node_neighbour_reliability,
    pearson_correlation,
)
from utils.missing_simulator import (
    derive_node_masks,
    generate_pixel_masks,
    node_observed_ratios,
    repair_all_missing_nodes,
    summarize_pixel_masks,
)
from utils.superpixel_utils import (
    HSI_to_superpixels,
    create_consensus_spatial_graph,
    create_spixel_graph,
    extract_superpixel_features,
    fuse_consensus_local_graph,
    fuse_reliability_adaptive_graph,
    fuse_centered_graph,
    RADG_MODES,
    centered_beta_values,
    edge_uncertainty_values,
    normalize_sparse_adjacency,
    reliability_adaptive_edge_weights,
    edge_reliability_values,
    show_superpixel,
)

# Outline-v3 pipeline: missingness is injected on the 2-D image plane *before*
# SLIC, so that the pixel corruption actually reaches the superpixel partition,
# the node features and the graph topology.
#
#   Raw view -> Mask -> Observed-only preprocessing -> SLIC -> Feature aggregation
#            -> Graph -> {Original FedRSMVC | FSRIMVC}
MASK_STAGE = 'pixel_level_before_slic'


def prepare_data(mat_path):
    data = sio.loadmat(str(mat_path))
    keys = [key for key in data if not key.startswith('__')]
    if not keys:
        raise ValueError(f'No data array found in {mat_path}')
    return data[keys[0]]


def get_sp_labels(seg, gt):
    labels = np.zeros(np.unique(seg).shape[0], dtype=np.int32)
    for idx in range(len(labels)):
        label = gt[seg == idx]
        if len(label) == 0:
            continue
        counts = Counter(label).most_common()
        labels[idx] = counts[1][0] if counts[0][0] == 0 and len(counts) > 1 else counts[0][0]
    return labels


def _normalize_image(image):
    image = np.asarray(image, dtype=np.float32)
    result = np.zeros_like(image, dtype=np.float32)
    for band in range(image.shape[2]):
        values = image[:, :, band]
        low, high = np.nanpercentile(values, [1, 99])
        if high > low:
            result[:, :, band] = np.clip((values - low) / (high - low), 0, 1)
    return result


def _preview_image(data, rgb=None, observed=None):
    """3-channel composite preview; ``observed`` punches the missing holes."""
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    n_bands = data.shape[2]
    if rgb is not None and len(rgb) == 3 and max(rgb) < n_bands:
        indices = rgb
    elif n_bands >= 3:
        indices = [min(n_bands - 1, int(n_bands * 0.75)), n_bands // 2, 0]
    elif n_bands == 2:
        indices = [0, 1, 0]
    else:
        indices = [0, 0, 0]
    preview = _normalize_image(data[:, :, indices])
    if observed is not None:
        preview = preview * np.asarray(observed, dtype=np.float32)[:, :, np.newaxis]
    return preview


def observed_only_minmax(rows, observed_flat):
    """Min-max normalise with statistics taken from observed pixels only.

    With ``observed_flat`` all-true this is exactly ``minmax_scale(rows)``,
    which keeps the ``rho = 0`` runs bit-comparable with the older pipeline.
    """
    rows = np.asarray(rows, dtype=np.float64)
    observed_flat = np.asarray(observed_flat, dtype=bool).reshape(-1)
    if not observed_flat.any():
        return np.zeros_like(rows, dtype=np.float32)
    reference = rows[observed_flat]
    low = reference.min(axis=0)
    high = reference.max(axis=0)
    span = np.where(high > low, high - low, 1.0)
    return np.clip((rows - low) / span, 0.0, 1.0).astype(np.float32)


class FedDataset:
    def __init__(
        self,
        dataset_name,
        mm_data_path,
        gt_path,
        n_pc,
        rgb,
        n_neighbors,
        n_superpixels,
        patch_size,
        output_dir=None,
        data_root=None,
        config_dir=None,
        pca_all_views=False,
        missing_rate=0.3,
        missing_mode='random',
        missing_seed=None,
        missing_params=None,
        superpixel_threshold=0.5,
        ensure_node_coverage=True,
        reference_view=0,
        graph_beta=0.5,
        mask_aware=True,
        spatial_graph=True,
        drop_nonpositive_labels=False,
        reliability_adaptive_graph=False,
        radg_mode='centered',
        radg_beta_min=0.2,
        radg_beta_max=0.8,
        radg_base_beta=0.5,
        radg_gamma=0.5,
        radg_beta_floor=0.2,
        radg_beta_ceiling=0.8,
        radg_center_tolerance=0.005,
        radg_edge_reliability='geometric_mean',
        seed=42,
    ):
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir) if output_dir else None
        self.config_dir = Path(config_dir or '.').resolve()
        self.data_root = Path(data_root).expanduser() if data_root else None
        self.missing_rate = float(missing_rate)
        self.missing_mode = str(missing_mode).lower()
        self.missing_seed = int(seed if missing_seed is None else missing_seed)
        self.missing_params = dict(missing_params or {})
        self.superpixel_threshold = float(superpixel_threshold)
        self.ensure_node_coverage = bool(ensure_node_coverage)
        self.reference_view = int(reference_view)
        self.graph_beta = float(graph_beta)
        self.mask_aware = bool(mask_aware)
        self.spatial_graph = bool(spatial_graph)
        # outline-v6, module I: Reliability-Adaptive Dual-Graph Learning.
        # ``False`` keeps the fixed fusion A = beta A_s + (1 - beta) A_f, which
        # is the frozen BSSAT-r1 control.  ``beta_min == beta_max == 0.5``
        # makes the adaptive path degenerate to the very same graph.
        self.reliability_adaptive_graph = bool(reliability_adaptive_graph)
        # outline-v7, module I v1.1.  ``centered`` keeps E[beta] at
        # ``radg_base_beta`` so the arm only re-allocates weight between edges;
        # ``uncentered`` is the (failed) v6 RADG-v1 kept as an ablation.
        self.radg_mode = str(radg_mode).lower()
        if self.radg_mode not in RADG_MODES:
            raise ValueError(
                f'unknown radg_mode {radg_mode!r}; expected one of {RADG_MODES}'
            )
        self.radg_beta_min = float(radg_beta_min)
        self.radg_beta_max = float(radg_beta_max)
        self.radg_base_beta = float(radg_base_beta)
        self.radg_gamma = float(radg_gamma)
        self.radg_beta_floor = float(radg_beta_floor)
        self.radg_beta_ceiling = float(radg_beta_ceiling)
        self.radg_center_tolerance = float(radg_center_tolerance)
        self.radg_edge_reliability = str(radg_edge_reliability)
        self.radg_weights = {}
        self.radg_view_diagnostics = []
        self.radg_diagnostics = None
        # MUUFL's gt.mat marks background with -1 (not 0), so the historical
        # ``y != 0`` rule keeps every background superpixel.  When this switch
        # is on, only strictly positive labels count as real classes.
        self.drop_nonpositive_labels = bool(drop_nonpositive_labels)
        self.seed = int(seed)
        self.mask_stage = MASK_STAGE

        resolved_views = [self._resolve_path(path) for path in mm_data_path]
        resolved_gt = self._resolve_path(gt_path)
        mm_data_raw = [prepare_data(path) for path in resolved_views]
        mm_data_raw = [data[:, :, np.newaxis] if data.ndim == 2 else data for data in mm_data_raw]
        gt = np.asarray(prepare_data(resolved_gt)).squeeze()
        spatial_shapes = {tuple(data.shape[:2]) for data in mm_data_raw}
        if len(spatial_shapes) != 1 or gt.shape != mm_data_raw[0].shape[:2]:
            raise ValueError(
                f'Spatial shape mismatch for {dataset_name}: views={spatial_shapes}, gt={gt.shape}'
            )

        n_row, n_column, _ = mm_data_raw[0].shape
        self.previews = [
            _preview_image(data, rgb if idx == 0 else None)
            for idx, data in enumerate(mm_data_raw)
        ]
        self._save_input_views(self.previews, gt)

        # ---- 1. pixel-level missingness, BEFORE any preprocessing/SLIC ----
        self.pixel_masks_full = generate_pixel_masks(
            n_views=len(mm_data_raw),
            height=n_row,
            width=n_column,
            missing_rate=self.missing_rate,
            seed=self.missing_seed,
            mode=self.missing_mode,
            params=self.missing_params,
        )
        corrupted_raw = [
            data * self.pixel_masks_full[idx][:, :, np.newaxis]
            for idx, data in enumerate(mm_data_raw)
        ]
        self.corrupted_previews = [
            _preview_image(
                corrupted_raw[idx],
                rgb if idx == self.reference_view else None,
                self.pixel_masks_full[idx],
            )
            for idx in range(len(corrupted_raw))
        ]
        self._save_corrupted_views()

        # ---- 2. observed-only PCA / normalisation on the corrupted views ----
        processed_views = []
        for idx, data in enumerate(corrupted_raw):
            observed_flat = self.pixel_masks_full[idx].reshape(-1)
            rows = data.reshape(n_row * n_column, data.shape[2])
            should_reduce = (
                n_pc is not None
                and data.shape[2] > n_pc
                and (idx == 0 or pca_all_views)
            )
            if should_reduce:
                reference_rows = rows[observed_flat]
                if reference_rows.shape[0] > n_pc:
                    # PCA is fitted on observed pixels only: no clean-image
                    # statistics leak into the corrupted representation.
                    pca = PCA(n_components=n_pc, random_state=self.seed)
                    pca.fit(reference_rows)
                    rows = pca.transform(rows)
            rows = observed_only_minmax(rows, observed_flat)
            image = rows.reshape(n_row, n_column, -1).astype(np.float32)
            # Keep the hole explicit so SLIC really sees the corrupted region.
            image[~self.pixel_masks_full[idx]] = 0.0
            processed_views.append(image)

        # ---- 3. one shared partition, from the corrupted reference view ----
        self.sp_labels = HSI_to_superpixels(
            self.corrupted_previews[self.reference_view],
            n_superpixels=n_superpixels,
            save_path=None,
        )
        self.gt = gt.astype(np.int64)
        self.y_full = get_sp_labels(self.sp_labels, self.gt)
        self.y = self.y_full.copy()
        self.active_superpixel_mask = np.ones_like(self.y_full, dtype=bool)
        self.n_classes = int(np.unique(self.y_full[self._valid_label_mask()]).size)
        self.consensus_adj, self.superpixel_centers = create_consensus_spatial_graph(
            self.sp_labels
        )

        # ---- 4. node availability from the pixel mask and the partition ----
        n_superpixel_nodes = self.y_full.size
        self.node_observed_ratios_full = np.stack([
            node_observed_ratios(
                self.pixel_masks_full[idx], self.sp_labels, n_superpixel_nodes
            )
            for idx in range(len(processed_views))
        ])
        node_masks = derive_node_masks(
            self.node_observed_ratios_full, self.superpixel_threshold
        )
        self.coverage_repairs = 0
        if self.ensure_node_coverage:
            node_masks, self.coverage_repairs = repair_all_missing_nodes(
                node_masks, self.node_observed_ratios_full
            )
        self.node_observed_ratios = self.node_observed_ratios_full.copy()
        self.observation_masks_full = node_masks
        self.observation_masks = self.observation_masks_full.copy()
        self.clients = []

        if self.output_dir:
            visualization_dir = self.output_dir / 'visualizations'
            visualization_dir.mkdir(parents=True, exist_ok=True)
            show_superpixel(
                self.sp_labels,
                self.corrupted_previews[self.reference_view],
                str(visualization_dir / 'superpixels.png'),
            )

        for idx, data in enumerate(processed_views):
            original_features = extract_superpixel_features(
                data, self.sp_labels, mode='center_patch', patch_size=patch_size
            ).astype(np.float32)
            observed_mask = self.observation_masks[idx]
            zero_filled_features = original_features.copy()
            zero_filled_features[~observed_mask] = 0.0

            # Graph and KNN features come from the corrupted view only: A0/A1
            # (mask_aware off) let zero-filled nodes create edges, A2+ keeps the
            # observed-only KNN.
            local_adj, _, _ = create_spixel_graph(
                data,
                self.sp_labels,
                n_neighbors,
                observed_mask=observed_mask if self.mask_aware else None,
            )
            reference_adj, _, _ = create_spixel_graph(
                data, self.sp_labels, n_neighbors, observed_mask=None
            )
            fused_adj = self._fuse_graph(local_adj, idx)
            self.clients.append({
                'client_id': idx,
                'modality_name': f'modality_{idx}',
                'n_classes': self.n_classes,
                'data': zero_filled_features,
                'target_data': original_features,
                'mask': observed_mask,
                'adj': fused_adj,
                'local_adj': local_adj,
                'reference_adj': reference_adj,
                'consensus_adj': self.consensus_adj,
                'y': self.y,
                'sp_labels': self.sp_labels,
                'raw_shape': (n_row, n_column),
            })

    def _resolve_path(self, value):
        path = Path(os.path.expandvars(str(value))).expanduser()
        candidates = [path] if path.is_absolute() else [self.config_dir / path]
        if self.data_root and not path.is_absolute():
            candidates.insert(0, self.data_root / path)
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        raise FileNotFoundError(f'Dataset file not found: {value}; checked {candidates}')

    def _save_input_views(self, previews, gt):
        if not self.output_dir:
            return
        view_dir = self.output_dir / 'visualizations' / 'input_views'
        view_dir.mkdir(parents=True, exist_ok=True)
        for idx, image in enumerate(previews):
            plt.imsave(view_dir / f'view_{idx:02d}.png', image)
        self._save_label_map(gt, self.output_dir / 'visualizations' / 'ground_truth.png')

    def _save_corrupted_views(self):
        if not self.output_dir:
            return
        corrupted_dir = self.output_dir / 'visualizations' / 'corrupted_input_views'
        corrupted_dir.mkdir(parents=True, exist_ok=True)
        for idx, image in enumerate(self.corrupted_previews):
            plt.imsave(
                corrupted_dir / f'view_{idx:02d}_corrupted.png',
                np.clip(image, 0, 1),
            )

    @staticmethod
    def _save_label_map(label_map, path):
        label_map = np.asarray(label_map)
        n_classes = max(1, int(label_map.max()) + 1)
        cmap = copy.copy(plt.get_cmap('tab20', n_classes))
        cmap.set_under('black')
        plt.imsave(path, label_map, cmap=cmap, vmin=0.5, vmax=max(1, n_classes - 0.5))

    def _fuse_graph(self, local_adj, view_idx=None):
        """A2+ fuses the consensus spatial graph; A0/A1 keep the local graph only.

        Outline-v6, module I: with ``reliability_adaptive_graph`` the fusion
        becomes edge-wise,

            A_ij = beta_ij A_s,ij + (1 - beta_ij) A_f,ij,
            beta_ij = b_min + (b_max - b_min) (1 - sqrt(r_i r_j)),

        so a node whose local observations are corrupted hands its edges over to
        the shared spatial graph.  ``b_min == b_max`` collapses this back onto
        the fixed fusion; only the post-processing is shared, it is never
        changed (outline-v6 section 5).
        """
        if self.spatial_graph:
            if self.reliability_adaptive_graph:
                if view_idx is None:
                    raise ValueError(
                        'RADG fusion needs the view index: r_i is per view.'
                    )
                reliability = self.node_observed_ratios[view_idx]
                if self.radg_mode == 'centered':
                    # outline-v7: beta_ij = beta_0 + gamma (u_ij - mean(u)),
                    # clipped to [floor, ceiling] and recentred so that
                    # E[beta] stays at beta_0.
                    fused, info = fuse_centered_graph(
                        self.consensus_adj,
                        local_adj,
                        reliability,
                        base_beta=self.radg_base_beta,
                        gamma=self.radg_gamma,
                        floor=self.radg_beta_floor,
                        ceiling=self.radg_beta_ceiling,
                        edge_reliability=self.radg_edge_reliability,
                        tolerance=self.radg_center_tolerance,
                    )
                else:
                    fused, info = fuse_reliability_adaptive_graph(
                        self.consensus_adj,
                        local_adj,
                        reliability,
                        beta_min=self.radg_beta_min,
                        beta_max=self.radg_beta_max,
                        edge_reliability=self.radg_edge_reliability,
                    )
                self.radg_weights[int(view_idx)] = info
                return fused
            return fuse_consensus_local_graph(
                self.consensus_adj, local_adj, beta=self.graph_beta
            )
        return normalize_sparse_adjacency(local_adj, add_self_loops=True)

    def build_radg_diagnostics(self):
        """Outline-v6 sections 11-12: prove that the adaptive fusion moved.

        The BSSAT-v2 post-mortem showed the failure mode to avoid: a modulation
        whose actual effect was never measured.  For every view this records

        * the distribution of ``r_i`` and of ``beta_ij`` (mean/std/p10/p25/p50/
          p75/p90) plus the *actual* attained ``beta_min`` / ``beta_max``;
        * ``Delta_A = ||A_RADG - A_fixed||_F / (||A_fixed||_F + eps)`` and the
          graph statistics of both variants (edges, components, lambda_2);
        * the two correlations that decide whether RADG-v1 has a theoretical
          basis at all: ``r_i`` against the node's mean incident beta, and
          ``r_i`` against the degradation of its own feature neighbourhood.

        The result is cached on ``self.radg_diagnostics`` and written to
        ``artifacts/graph_diagnostics.json``.
        """
        views = []
        node_ratio_samples = []
        beta_samples = []
        for idx, client in enumerate(self.clients):
            reliability = self.node_observed_ratios[idx]
            local_adj = client['local_adj']
            consensus_adj = client['consensus_adj']
            fixed_adj = normalize_sparse_adjacency(
                self.graph_beta * consensus_adj
                + (1.0 - self.graph_beta) * local_adj,
                add_self_loops=True,
            )
            entry = {
                'view': int(idx),
                'radg_enabled': bool(self.reliability_adaptive_graph),
                'observed_ratio': distribution_summary(reliability, prefix='observed_ratio_'),
                'edges_fused': int(client['adj'].nnz // 2),
                'fixed_graph': graph_metrics(fixed_adj),
                'fixed_degree': graph_degree_stats(fixed_adj),
                'graph_relative_frobenius_change': graph_relative_frobenius_change(
                    client['adj'], fixed_adj
                ),
                'radg_graph': graph_metrics(client['adj']),
                'radg_degree': graph_degree_stats(client['adj']),
            }
            if self.reliability_adaptive_graph:
                # One sample per undirected edge in the union (no double
                # counting edges shared by spatial and feature graphs).
                from scipy import sparse
                union = (consensus_adj.astype(bool) + local_adj.astype(bool)).tocsr()
                union.setdiag(0)
                union.eliminate_zeros()
                upper = sparse.triu(union, k=1).tocoo()
                c_union = edge_reliability_values(reliability, upper.row, upper.col,
                                                   self.radg_edge_reliability)
                u_union = 1.0 - c_union
                if self.radg_mode == 'centered':
                    beta_all, centre_info = centered_beta_values(
                        u_union,
                        base_beta=self.radg_base_beta,
                        gamma=self.radg_gamma,
                        floor=self.radg_beta_floor,
                        ceiling=self.radg_beta_ceiling,
                        tolerance=self.radg_center_tolerance,
                    )
                else:
                    beta_all = (
                        self.radg_beta_min
                        + (self.radg_beta_max - self.radg_beta_min) * (1.0 - c_union)
                    )
                    centre_info = {
                        'edge_uncertainty_mean': float(u_union.mean()),
                        'edge_uncertainty_std': float(u_union.std()),
                        'beta_mean': float(beta_all.mean()),
                        'beta_mean_error_from_base': abs(
                            float(beta_all.mean()) - self.radg_base_beta
                        ),
                        'beta_mean_error_before_recentre': abs(
                            float(beta_all.mean()) - self.radg_base_beta
                        ),
                        'recentre_iterations': 0,
                        'recentre_delta': 0.0,
                        'centered': False,
                    }
                # The per-graph beta vectors actually used by the fusion: the
                # caller stored them in radg_weights while fusing.
                stored = self.radg_weights.get(int(idx)) or {}
                beta_spatial = np.asarray(
                    stored.get('beta_spatial', []), dtype=np.float64
                )
                beta_feature = np.asarray(
                    stored.get('beta_feature', []), dtype=np.float64
                )
                c_spatial = edge_reliability_values(
                    reliability,
                    sparse.csr_matrix(consensus_adj).tocoo().row,
                    sparse.csr_matrix(consensus_adj).tocoo().col,
                    self.radg_edge_reliability,
                )
                c_feature = edge_reliability_values(
                    reliability,
                    sparse.csr_matrix(local_adj).tocoo().row,
                    sparse.csr_matrix(local_adj).tocoo().col,
                    self.radg_edge_reliability,
                )
                entry.update({
                    'radg_mode': self.radg_mode,
                    'beta_base': float(self.radg_base_beta),
                    'beta_gamma': float(self.radg_gamma),
                    'beta_floor': float(self.radg_beta_floor),
                    'beta_ceiling': float(self.radg_beta_ceiling),
                    'beta': distribution_summary(beta_all, prefix='beta_'),
                    'beta_spatial': distribution_summary(beta_spatial, prefix='beta_'),
                    'beta_feature': distribution_summary(beta_feature, prefix='beta_'),
                    'beta_min_actual': float(beta_all.min()) if beta_all.size else None,
                    'beta_max_actual': float(beta_all.max()) if beta_all.size else None,
                    'beta_mean_error_from_base': centre_info.get(
                        'beta_mean_error_from_base'
                    ),
                    'beta_mean_error_before_recentre': centre_info.get(
                        'beta_mean_error_before_recentre'
                    ),
                    'recentre_iterations': centre_info.get('recentre_iterations'),
                    'recentre_delta': centre_info.get('recentre_delta'),
                    'center_within_tolerance': (
                        None
                        if centre_info.get('beta_mean_error_from_base') is None
                        else bool(
                            centre_info['beta_mean_error_from_base']
                            < self.radg_center_tolerance
                        )
                    ),
                    'edge_uncertainty': distribution_summary(
                        u_union, prefix='edge_uncertainty_'
                    ),
                    'edge_reliability_spatial': distribution_summary(
                        c_spatial, prefix='edge_reliability_'
                    ),
                    'edge_reliability_feature': distribution_summary(
                        c_feature, prefix='edge_reliability_'
                    ),
                })
                # Per-node mean incident beta over the union of both edge sets.
                beta_node = np.zeros(reliability.size, dtype=np.float64)
                count_node = np.zeros(reliability.size, dtype=np.float64)
                np.add.at(beta_node, upper.row, beta_all)
                np.add.at(count_node, upper.row, 1.0)
                np.add.at(beta_node, upper.col, beta_all)
                np.add.at(count_node, upper.col, 1.0)
                valid = count_node > 0
                beta_node_mean = np.full(reliability.size, np.nan)
                beta_node_mean[valid] = beta_node[valid] / count_node[valid]
                # d_i^feature: how corrupted the node's own feature neighbourhood is.
                neighbour_ratio = node_neighbour_reliability(local_adj, reliability)
                neighbour_missing = 1.0 - neighbour_ratio
                # Feature-edge weight lost by the observed-only KNN restriction,
                # measured against the unrestricted reference feature graph.
                reference_weight, reference_count = node_mean_incident_weight(
                    client['reference_adj']
                )
                kept_weight, kept_count = node_mean_incident_weight(local_adj)
                edge_loss = node_feature_edge_loss(local_adj, client['reference_adj'])
                entry.update({
                    'beta_distribution_basis': 'unique undirected union edges, no self loops',
                    'feature_edge_loss_definition': '1 - retained reference neighbours / reference neighbours; reference uses corrupted features without node masking',
                    'beta_node_mean': distribution_summary(
                        beta_node_mean, prefix='beta_node_'
                    ),
                    'neighbour_missing': distribution_summary(
                        neighbour_missing, prefix='d_feature_'
                    ),
                    'feature_edge_loss': distribution_summary(
                        edge_loss, prefix='d_feature_'
                    ),
                    'corr_r_beta_node': pearson_correlation(reliability, beta_node_mean),
                    'corr_r_neighbour_missing': pearson_correlation(
                        reliability, neighbour_missing
                    ),
                    'corr_r_feature_edge_loss': pearson_correlation(
                        reliability, edge_loss
                    ),
                    'reference_feature_edges': int(client['reference_adj'].nnz // 2),
                    'reference_feature_nodes_with_edges': int(
                        (reference_count > 0).sum()
                    ),
                    'kept_feature_nodes_with_edges': int((kept_count > 0).sum()),
                })
                node_ratio_samples.append(np.asarray(reliability, dtype=np.float64))
                beta_samples.append(beta_all)
            # Real-dataset degeneration gate; no dependence on toy fixtures.
            if self.spatial_graph:
                degenerate, _ = fuse_reliability_adaptive_graph(
                    consensus_adj, local_adj, reliability, beta_min=0.5, beta_max=0.5
                )
                control = fuse_consensus_local_graph(consensus_adj, local_adj, beta=0.5)
                difference = (degenerate - control).tocsr()
                max_abs = float(np.abs(difference.data).max()) if difference.nnz else 0.0
                relative = graph_relative_frobenius_change(degenerate, control)
                nnz_diff = int(degenerate.nnz - control.nnz)
                entry['degeneration_check'] = {'max_abs_diff': max_abs,
                    'frobenius_relative_diff': relative, 'nnz_diff': nnz_diff}
                if max_abs >= 1e-6 or relative >= 1e-6 or nnz_diff != 0:
                    raise RuntimeError('RADG fixed-control degeneration check failed')
            views.append(entry)

        payload = {
            'radg_enabled': bool(self.reliability_adaptive_graph),
            'radg_mode': (
                self.radg_mode if self.reliability_adaptive_graph else None
            ),
            'radg_beta_min_config': float(self.radg_beta_min),
            'radg_beta_max_config': float(self.radg_beta_max),
            'radg_base_beta_config': float(self.radg_base_beta),
            'radg_gamma_config': float(self.radg_gamma),
            'radg_beta_floor_config': float(self.radg_beta_floor),
            'radg_beta_ceiling_config': float(self.radg_beta_ceiling),
            'radg_center_tolerance': float(self.radg_center_tolerance),
            'radg_edge_reliability': self.radg_edge_reliability,
            'fixed_graph_beta': float(self.graph_beta),
            'views': views,
            'flat': self._flatten_radg_views(views),
        }
        if self.reliability_adaptive_graph:
            pooled_ratio = np.concatenate(node_ratio_samples) if node_ratio_samples else []
            pooled_beta = np.concatenate(beta_samples) if beta_samples else []
            mean_error = payload['flat'].get('beta_mean_error_from_base')
            payload.update({
                'node_reliability': distribution_summary(pooled_ratio),
                'beta': distribution_summary(pooled_beta, prefix='beta_'),
                'beta_min_actual': (
                    float(pooled_beta.min()) if pooled_beta.size else None
                ),
                'beta_max_actual': (
                    float(pooled_beta.max()) if pooled_beta.size else None
                ),
                'beta_mean_error_from_base': mean_error,
                'center_within_tolerance': (
                    None if mean_error is None
                    else bool(mean_error < self.radg_center_tolerance)
                ),
                'graph_relative_frobenius_change': payload['flat']['graph_relative_frobenius_change'],
            })
        self.radg_diagnostics = payload
        return payload

    def _flatten_radg_views(self, views):
        """Per-view RADG payload -> the flat keys written into ``metrics.json``.

        The aggregator reads ``metrics.json`` only, so every RADG number that
        has to appear in the result tables must exist here as a scalar.  Values
        are averaged over the views; ``None`` when a view did not produce them.
        """
        def view_mean(nested_key, leaf_key):
            values = []
            for view in views:
                block = view if nested_key is None else view.get(nested_key)
                if not isinstance(block, dict):
                    continue
                value = block.get(leaf_key)
                if value is not None and np.isfinite(float(value)):
                    values.append(float(value))
            return float(np.mean(values)) if values else None

        flat = {
            'radg_mode': self.radg_mode if self.reliability_adaptive_graph else None,
            'radg_beta_min_config': float(self.radg_beta_min),
            'radg_beta_max_config': float(self.radg_beta_max),
            'radg_base_beta_config': float(self.radg_base_beta),
            'radg_gamma_config': float(self.radg_gamma),
            'radg_beta_floor_config': float(self.radg_beta_floor),
            'radg_beta_ceiling_config': float(self.radg_beta_ceiling),
            'radg_edge_reliability': self.radg_edge_reliability,
            'graph_relative_frobenius_change': view_mean(None, 'graph_relative_frobenius_change'),
            'corr_r_beta_node': view_mean(None, 'corr_r_beta_node'),
            'corr_r_neighbour_missing': view_mean(None, 'corr_r_neighbour_missing'),
            'corr_r_feature_edge_loss': view_mean(None, 'corr_r_feature_edge_loss'),
            'beta_mean_error_from_base': view_mean(None, 'beta_mean_error_from_base'),
            'beta_mean_error_before_recentre': view_mean(
                None, 'beta_mean_error_before_recentre'
            ),
            'recentre_iterations': view_mean(None, 'recentre_iterations'),
            'recentre_delta': view_mean(None, 'recentre_delta'),
        }
        for leaf in ('mean', 'std', 'p10', 'p50', 'p90'):
            flat['observed_ratio_%s' % leaf] = view_mean('observed_ratio', 'observed_ratio_%s' % leaf)
        for leaf in ('mean', 'std', 'p10', 'p25', 'p50', 'p75', 'p90'):
            flat['beta_%s' % leaf] = view_mean('beta', 'beta_%s' % leaf)
        for leaf in ('mean', 'std', 'p10', 'p25', 'p50', 'p75', 'p90'):
            flat['edge_uncertainty_%s' % leaf] = view_mean(
                'edge_uncertainty', 'edge_uncertainty_%s' % leaf
            )
        for leaf in ('nodes', 'undirected_edges', 'connected_components', 'algebraic_connectivity'):
            flat['radg_%s' % leaf] = view_mean('radg_graph', leaf)
            flat['fixed_%s' % leaf] = view_mean('fixed_graph', leaf)
        for leaf in ('degree_mean', 'degree_std'):
            flat['radg_%s' % leaf] = view_mean('radg_degree', leaf)
            flat['fixed_%s' % leaf] = view_mean('fixed_degree', leaf)
        flat['beta_min_actual'] = view_mean(None, 'beta_min_actual')
        flat['beta_max_actual'] = view_mean(None, 'beta_max_actual')
        return flat

    def _valid_label_mask(self):
        """Boolean mask over ``y_full`` marking real (non-background) labels."""
        if self.drop_nonpositive_labels:
            return self.y_full > 0
        return self.y_full != 0

    def remove_background(self):
        self.active_superpixel_mask = self._valid_label_mask()
        self.y = self.y_full[self.active_superpixel_mask]
        self.observation_masks = self.observation_masks_full[:, self.active_superpixel_mask]
        self.node_observed_ratios = self.node_observed_ratios_full[:, self.active_superpixel_mask]
        self.consensus_adj = self.consensus_adj[
            self.active_superpixel_mask
        ][:, self.active_superpixel_mask].tocsr()
        for view_idx, client in enumerate(self.clients):
            mask = self.observation_masks[view_idx]
            client['data'] = client['data'][self.active_superpixel_mask]
            client['target_data'] = client['target_data'][self.active_superpixel_mask]
            client['mask'] = mask
            client['local_adj'] = client['local_adj'][
                self.active_superpixel_mask
            ][:, self.active_superpixel_mask].tocsr()
            client['reference_adj'] = client['reference_adj'][
                self.active_superpixel_mask
            ][:, self.active_superpixel_mask].tocsr()
            client['consensus_adj'] = self.consensus_adj
            client['adj'] = self._fuse_graph(client['local_adj'], view_idx)
            client['y'] = self.y

    def compute_spatial_basis(self, requested_dim):
        n_nodes = self.consensus_adj.shape[0]
        if n_nodes < 2:
            raise ValueError('At least two superpixels are required for a spatial basis.')
        dimension = min(int(requested_dim), n_nodes - 1)
        # scipy<1.11 expects a sparse matrix here, while recent NetworkX may
        # produce a sparse array.  Keep the serialized graph representation
        # compatible with both server and workstation environments.
        from scipy import sparse
        adjacency = sparse.csr_matrix(self.consensus_adj, dtype=np.float64)
        laplacian = sparse.csr_matrix(
            csgraph.laplacian(adjacency, normed=True), dtype=np.float64
        )
        n_eigenvectors = min(n_nodes - 1, dimension + 1)
        eigenvalues, eigenvectors = eigsh(
            laplacian, k=n_eigenvectors, which='SM'
        )
        order = np.argsort(np.real(eigenvalues))
        eigenvalues = np.real(eigenvalues[order])
        eigenvectors = np.real(eigenvectors[:, order])
        nontrivial = np.flatnonzero(eigenvalues > 1e-7)
        selected = list(nontrivial[:dimension])
        if len(selected) < dimension:
            for idx in range(1, eigenvectors.shape[1]):
                if idx not in selected:
                    selected.append(idx)
                if len(selected) == dimension:
                    break
        basis = eigenvectors[:, selected[:dimension]].astype(np.float32)
        return basis

    def save_missing_and_graph_artifacts(self):
        if not self.output_dir:
            return
        artifact_dir = self.output_dir / 'artifacts'
        mask_view_dir = self.output_dir / 'visualizations' / 'missing_masks'
        node_mask_dir = self.output_dir / 'visualizations' / 'node_missing_masks'
        masked_input_dir = self.output_dir / 'visualizations' / 'masked_input_views'
        artifact_dir.mkdir(parents=True, exist_ok=True)
        mask_view_dir.mkdir(parents=True, exist_ok=True)
        node_mask_dir.mkdir(parents=True, exist_ok=True)
        masked_input_dir.mkdir(parents=True, exist_ok=True)

        np.save(artifact_dir / 'pixel_masks_full.npy', self.pixel_masks_full)
        np.save(artifact_dir / 'node_observed_ratios.npy', self.node_observed_ratios)
        np.save(artifact_dir / 'node_observed_ratios_full.npy', self.node_observed_ratios_full)
        np.save(artifact_dir / 'observation_masks.npy', self.observation_masks)
        np.save(artifact_dir / 'observation_masks_full.npy', self.observation_masks_full)
        np.save(artifact_dir / 'sp_labels.npy', self.sp_labels)
        np.save(artifact_dir / 'active_superpixel_mask.npy', self.active_superpixel_mask)
        stats = {
            'mask_stage': self.mask_stage,
            'mode': self.missing_mode,
            'requested_missing_rate': self.missing_rate,
            'missing_seed': self.missing_seed,
            'superpixel_observed_threshold': self.superpixel_threshold,
            'reference_view': self.reference_view,
            'missing_params': self.missing_params,
            'ensure_node_coverage': self.ensure_node_coverage,
            'coverage_repairs': int(self.coverage_repairs),
            'n_superpixel_nodes': int(self.y_full.size),
            'n_active_nodes': int(self.active_superpixel_mask.sum()),
            'all_views_missing_count': int((~self.observation_masks.any(axis=0)).sum()),
            'pixel_summary': summarize_pixel_masks(self.pixel_masks_full),
            'views': [],
            'views_full': [],
        }
        graph_diagnostics = {
            'graph_beta': self.graph_beta,
            'mask_aware': self.mask_aware,
            'spatial_graph': self.spatial_graph,
            'consensus_graph': graph_metrics(self.consensus_adj),
            # outline v6, module I: was the adaptive fusion actually adaptive?
            'radg': self.build_radg_diagnostics(),
            'views': [],
        }
        for idx, client in enumerate(self.clients):
            mask = self.observation_masks[idx]
            stats['views'].append({
                'view': idx,
                'observed_nodes': int(mask.sum()),
                'missing_nodes': int((~mask).sum()),
                'actual_missing_rate': float((~mask).mean()),
                'mean_observed_ratio': float(self.node_observed_ratios[idx].mean()),
            })
            mask_full = self.observation_masks_full[idx]
            stats['views_full'].append({
                'view': idx,
                'pixel_missing_rate': float((~self.pixel_masks_full[idx]).mean()),
                'node_missing_rate': float((~mask_full).mean()),
                'mean_observed_ratio': float(self.node_observed_ratios_full[idx].mean()),
            })
            # True pixel-level corruption M^(v) (the new key artifact).
            plt.imsave(
                mask_view_dir / f'view_{idx:02d}_pixel_mask.png',
                self.pixel_masks_full[idx].astype(np.float32),
                cmap='gray', vmin=0, vmax=1,
            )
            # Node availability after SLIC + threshold tau.
            node_pixel_mask = mask_full[self.sp_labels]
            node_pixel_mask[self.gt == 0] = False
            plt.imsave(
                node_mask_dir / f'view_{idx:02d}_node_mask.png',
                node_pixel_mask.astype(np.float32), cmap='gray', vmin=0, vmax=1,
            )
            plt.imsave(
                masked_input_dir / f'view_{idx:02d}_zero_filled.png',
                np.clip(self.corrupted_previews[idx], 0, 1),
            )
            view_graph = {
                'view': idx,
                'local_graph': graph_metrics(client['local_adj']),
                'fused_graph': graph_metrics(client['adj']),
                'edge_retention_ratio': edge_retention_ratio(
                    client['local_adj'], client['reference_adj']
                ),
            }
            graph_diagnostics['views'].append(view_graph)

        with open(artifact_dir / 'missingness.json', 'w', encoding='utf-8') as handle:
            json.dump(stats, handle, indent=2, ensure_ascii=False)
        with open(artifact_dir / 'graph_diagnostics.json', 'w', encoding='utf-8') as handle:
            json.dump(graph_diagnostics, handle, indent=2, ensure_ascii=False)
        # outline v6: the RADG payload also gets its own file so the per-view
        # beta/r_i distributions can be inspected without the graph metrics.
        if self.radg_diagnostics is not None:
            with open(artifact_dir / 'radg_diagnostics.json', 'w', encoding='utf-8') as handle:
                json.dump(self.radg_diagnostics, handle, indent=2, ensure_ascii=False)

    def recover_background(self, sp_pred):
        if isinstance(sp_pred, torch.Tensor):
            sp_pred = sp_pred.detach().cpu().numpy()
        sp_pred = np.asarray(sp_pred)
        if sp_pred.size == self.y_full.size:
            return sp_pred
        if sp_pred.size != int(self.active_superpixel_mask.sum()):
            warnings.warn('Recover background failed due to shape mismatch; returning raw prediction.')
            return sp_pred
        full = np.zeros(self.y_full.size, dtype=sp_pred.dtype)
        full[self.active_superpixel_mask] = sp_pred
        return full

    def save_prediction(self, sp_pred, output_dir):
        output_dir = Path(output_dir)
        artifact_dir = output_dir / 'artifacts'
        visualization_dir = output_dir / 'visualizations'
        artifact_dir.mkdir(parents=True, exist_ok=True)
        visualization_dir.mkdir(parents=True, exist_ok=True)
        full_sp_pred = self.recover_background(sp_pred)
        pixel_pred = full_sp_pred[self.sp_labels]
        pixel_pred[self.gt == 0] = 0
        np.save(artifact_dir / 'best_superpixel_predictions.npy', full_sp_pred)
        np.save(artifact_dir / 'best_pixel_prediction_map.npy', pixel_pred)
        np.save(artifact_dir / 'ground_truth.npy', self.gt)
        self._save_label_map(pixel_pred, visualization_dir / 'best_clustering_map.png')

    def save_clients(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for client in self.clients:
            adjacency = client['adj'].tocoo()
            indices = torch.tensor(np.vstack([adjacency.row, adjacency.col]), dtype=torch.long)
            values = torch.tensor(adjacency.data, dtype=torch.float32)
            torch_adj = torch.sparse_coo_tensor(indices, values, adjacency.shape).coalesce()
            torch.save({
                'x': torch.from_numpy(client['data']).float(),
                'target_x': torch.from_numpy(client['target_data']).float(),
                'mask': torch.from_numpy(client['mask']).bool(),
                'adj': torch_adj,
                'y': torch.from_numpy(client['y']).long(),
                'n_classes': client['n_classes'],
            }, save_dir / f"client_{client['client_id']}.pt")

    def __len__(self):
        return len(self.clients)

    def __getitem__(self, index):
        return self.clients[index]

    def __str__(self):
        return f'Dataset: {self.dataset_name} Clients: {len(self.clients)}'
