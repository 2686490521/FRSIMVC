import csv
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from cal_metric import full_metric
from client.client import Client
from dataset import FedDataset
import dataset as dataset_module
from server.server import Server
from utils.ablation import resolve_ablation, variant_label
from utils.diagnostics import (
    alignment_diagnostics,
    clustering_diagnostics,
    proportion_concentration,
    matched_labels,
    per_class_recall,
)
from utils.completion_adapter import CompletionAdapter


def map_labels(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if y_true.size != y_pred.size:
        return y_pred
    dimension = max(int(y_pred.max()), int(y_true.max())) + 1
    counts = np.zeros((dimension, dimension), dtype=np.int64)
    np.add.at(counts, (y_pred, y_true), 1)
    row_ind, col_ind = linear_sum_assignment(counts.max() - counts)
    mapping = np.arange(dimension, dtype=np.int64)
    mapping[row_ind] = col_ind
    return mapping[y_pred]


def _json_dump(path, payload):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _masked_average(aligned_soft_labels, observed_masks, eps=1e-8):
    numerator = torch.zeros_like(aligned_soft_labels[0])
    denominator = torch.zeros(
        aligned_soft_labels[0].shape[0], 1,
        device=aligned_soft_labels[0].device,
        dtype=aligned_soft_labels[0].dtype,
    )
    for aligned, mask in zip(aligned_soft_labels, observed_masks):
        weight = mask.to(aligned.dtype).unsqueeze(1)
        numerator = numerator + aligned * weight
        denominator = denominator + weight
    if torch.any(denominator == 0):
        raise RuntimeError('Global consensus found a node missing in every view.')
    consensus = numerator / denominator.clamp_min(eps)
    return consensus / consensus.sum(dim=1, keepdim=True).clamp_min(eps)


def _foreground_metrics(y_true, raw_prediction):
    """Foreground-only metrics for datasets with a negative-label background.

    Outline v5 section 4: ``MUUFL/gt.mat`` marks the background with ``-1``
    (every other dataset uses ``0``), and roughly 23% of the reported BSSAT ACC
    gain on MUUFL came from that single background class.  Whenever negative
    labels are present we therefore also score the foreground subset
    ``y_i != -1`` with the same best-map convention as the headline ACC, so
    "the background got easier" can never masquerade as a real improvement.

    Returns ``{}`` when the ground truth has no negative labels (i.e. for every
    dataset except MUUFL), in which case nothing is recorded.
    """
    y_true = np.asarray(y_true).reshape(-1)
    raw_prediction = np.asarray(raw_prediction).reshape(-1)
    foreground_mask = y_true != -1
    n_foreground = int(foreground_mask.sum())
    if n_foreground == 0 or n_foreground == y_true.size:
        return {}
    acc, _, _, nmi, ari, _, _, _, _ = full_metric(
        y_true[foreground_mask].copy(),
        matched_labels(y_true[foreground_mask], raw_prediction[foreground_mask]),
        is_refined=True,
    )
    return {
        'ACC_foreground': float(acc),
        'NMI_foreground': float(nmi),
        'ARI_foreground': float(ari),
        'foreground_nodes': n_foreground,
        'background_nodes': int(y_true.size - n_foreground),
        'foreground_definition': 'y_i != -1',
        'per_class_recall': per_class_recall(
            y_true[foreground_mask], raw_prediction[foreground_mask]
        ),
    }


class FLSimulator:
    def __init__(self, config, device, output_dir):
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.server = None
        self.clients = []
        self.dataset = None
        self.spatial_basis = None
        self._setup_environment()

    def _setup_environment(self):
        dataset_name = self.config['dataset_name']
        dataset_config = self.config[dataset_name]
        self.ablation_variant, self.ablation_flags = resolve_ablation(self.config)
        self.config['ablation_flags'] = dict(self.ablation_flags)
        self.config['mask_aware_loss'] = self.ablation_flags['mask_aware_loss']
        self.config['mask_aware_eval'] = self.ablation_flags['mask_aware_eval']
        print(f'Ablation: {variant_label(self.ablation_variant, self.ablation_flags)}')
        self.dataset = FedDataset(
            dataset_name=dataset_name,
            mm_data_path=dataset_config['mm_data_path'],
            gt_path=dataset_config['gt_path'],
            n_pc=dataset_config.get('n_pc'),
            rgb=dataset_config.get('rgb'),
            n_neighbors=dataset_config['n_neighbors'],
            n_superpixels=dataset_config['n_superpixels'],
            patch_size=dataset_config['patch_size'],
            output_dir=self.output_dir,
            data_root=self.config.get('data_root'),
            config_dir=self.config.get('config_dir'),
            pca_all_views=dataset_config.get('pca_all_views', False),
            missing_rate=self.config.get('missing_rate', 0.3),
            missing_mode=self.config.get('missing_mode', 'random'),
            missing_seed=self.config.get('missing_seed'),
            missing_params=self.config.get('missing_params'),
            superpixel_threshold=self.config.get('superpixel_observed_threshold', 0.5),
            ensure_node_coverage=self.config.get('ensure_node_coverage', True),
            reference_view=self.config.get('reference_view', 0),
            graph_beta=self.config.get('graph_beta', 0.5),
            mask_aware=self.ablation_flags['mask_aware'],
            spatial_graph=self.ablation_flags['spatial_graph'],
            drop_nonpositive_labels=self.config.get('drop_nonpositive_labels', False),
            # outline v6, module I: Reliability-Adaptive Dual-Graph.
            reliability_adaptive_graph=self.config.get('reliability_adaptive_graph', False),
            radg_mode=self.config.get('radg_mode', 'centered'),
            radg_beta_min=self.config.get('radg_beta_min', 0.2),
            radg_beta_max=self.config.get('radg_beta_max', 0.8),
            radg_base_beta=self.config.get('radg_base_beta', 0.5),
            radg_gamma=self.config.get('radg_gamma', 0.5),
            radg_beta_floor=self.config.get('radg_beta_floor', 0.2),
            radg_beta_ceiling=self.config.get('radg_beta_ceiling', 0.8),
            radg_center_tolerance=self.config.get('radg_center_tolerance', 0.005),
            radg_edge_reliability=self.config.get('radg_edge_reliability', 'geometric_mean'),
            seed=self.config.get('seed', 42),
        )
        if self.config.get('remove_bkg', False):
            self.dataset.remove_background()
        self.dataset.save_missing_and_graph_artifacts()
        # outline v6 sections 11-12: the RADG diagnostics are static (the graph
        # is built once), so they are flattened once here and then copied into
        # every metrics.json / diagnostics.json this run writes.
        self.radg_diagnostics = getattr(self.dataset, 'radg_diagnostics', None) or {}
        self.radg_flat_metrics = dict(self.radg_diagnostics.get('flat') or {})
        self.radg_flat_metrics['radg_enabled'] = bool(
            self.config.get('reliability_adaptive_graph', False)
        )
        self.radg_flat_metrics['radg_mode'] = (
            self.radg_diagnostics.get('radg_mode')
            if self.radg_diagnostics.get('radg_enabled')
            else None
        )
        if self.radg_diagnostics.get('radg_enabled'):
            print(
                'RADG: enabled  mode={} beta_min={:.2f} beta_max={:.2f} '
                'base={:.2f} gamma={:.2f} floor={:.2f} ceiling={:.2f} '
                'edge_reliability={}'.format(
                    self.radg_diagnostics.get('radg_mode'),
                    float(self.radg_diagnostics.get('radg_beta_min_config', 0.0)),
                    float(self.radg_diagnostics.get('radg_beta_max_config', 0.0)),
                    float(self.radg_diagnostics.get('radg_base_beta_config', 0.5)),
                    float(self.radg_diagnostics.get('radg_gamma_config', 0.5)),
                    float(self.radg_diagnostics.get('radg_beta_floor_config', 0.2)),
                    float(self.radg_diagnostics.get('radg_beta_ceiling_config', 0.8)),
                    self.radg_diagnostics.get('radg_edge_reliability'),
                )
            )
            print(
                'RADG: beta mean={} std={} |E[b]-base|={} recentre_iters={} '
                'observed |Delta_A|={}'.format(
                    self.radg_flat_metrics.get('beta_mean'),
                    self.radg_flat_metrics.get('beta_std'),
                    self.radg_flat_metrics.get('beta_mean_error_from_base'),
                    self.radg_flat_metrics.get('recentre_iterations'),
                    self.radg_flat_metrics.get('graph_relative_frobenius_change'),
                )
            )
            mean_error = self.radg_flat_metrics.get('beta_mean_error_from_base')
            tolerance = float(self.config.get('radg_center_tolerance', 0.005))
            if (
                self.radg_diagnostics.get('radg_mode') == 'centered'
                and mean_error is not None
                and float(mean_error) >= tolerance
            ):
                print(
                    'WARNING: centered RADG mean beta deviates from the base by '
                    '{:.6f} (>= tolerance {:.6f})'.format(float(mean_error), tolerance)
                )
        else:
            print('RADG: disabled (fixed fusion A = {:.2f} A_s + {:.2f} A_f)'.format(
                float(self.config.get('graph_beta', 0.5)),
                1.0 - float(self.config.get('graph_beta', 0.5)),
            ))

        spatial_basis_np = self.dataset.compute_spatial_basis(
            self.config.get('spatial_signature_dim', 16)
        )
        artifact_dir = self.output_dir / 'artifacts'
        artifact_dir.mkdir(parents=True, exist_ok=True)
        np.save(artifact_dir / 'consensus_spatial_basis.npy', spatial_basis_np)
        self.spatial_basis = torch.from_numpy(spatial_basis_np).float().to(self.device)

        self.config['num_clients'] = len(self.dataset.clients)
        n_samples = self.dataset.y.size
        self.server = Server(
            self.config,
            device=self.device,
            n_input=self.dataset.clients[0]['data'].shape[1],
            n_samples=n_samples,
        )
        self.clients = [
            Client(
                client_id=idx,
                config=self.config,
                raw_data_dict=raw_client,
                device=self.device,
            )
            for idx, raw_client in enumerate(self.dataset.clients)
        ]
        # ---- outline v9, module III: read-only completion adapter ----------
        # Constructed AFTER the baseline objects, and inactive unless
        # completion_enabled=true, in which case B0 stays bit-identical.
        self.completion = CompletionAdapter(
            config=self.config,
            clients=self.clients,
            server=self.server,
            spatial_basis=self.spatial_basis,
            device=self.device,
            output_dir=self.output_dir,
        )
        self.completion_timing = {'adapter_seconds': 0.0, 'diagnostic_seconds': 0.0}
        if self.completion.enabled:
            print(
                'Completion: enabled  stage={} arms={} lambdas={} '
                'adapter_seed={} diagnostic_epochs={}'.format(
                    self.completion.stage,
                    ' '.join(self.completion.arms),
                    ' '.join(str(x) for x in self.completion.lambdas),
                    self.completion.adapter_seed,
                    list(self.completion.diagnostic_epochs),
                )
            )
        else:
            print('Completion: disabled (B0 path untouched)')

    def _write_history(self, history):
        if not history:
            return
        with open(self.output_dir / 'history.csv', 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

    def _save_v5_artifacts(self, global_consensus, transports):
        artifact_dir = self.output_dir / 'artifacts'
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if global_consensus is not None:
            np.save(
                artifact_dir / 'best_global_semantic_consensus.npy',
                global_consensus.detach().cpu().numpy(),
            )
        anchors = self.server.global_anchors
        if anchors is not None:
            np.save(
                artifact_dir / 'global_semantic_anchors.npy',
                anchors['spatial'].detach().cpu().numpy(),
            )
            if anchors.get('semantic') is not None:
                np.save(
                    artifact_dir / 'global_semantic_anchor_stats.npy',
                    anchors['semantic'].detach().cpu().numpy(),
                )
        raw_mass = getattr(self.server, 'global_cluster_mass', None)
        tempered_mass = getattr(self.server, 'global_cluster_mass_tempered', None)
        if raw_mass is not None:
            np.save(
                artifact_dir / 'global_mass_raw.npy',
                raw_mass.detach().cpu().numpy(),
            )
        if tempered_mass is not None:
            np.save(
                artifact_dir / 'global_mass_tempered.npy',
                tempered_mass.detach().cpu().numpy(),
            )
            # ``global_cluster_mass.npy`` keeps its old meaning: the column
            # marginal this round's transport actually consumed (== tempered
            # mass, and identical to the raw mass when alpha == 0).
            np.save(
                artifact_dir / 'global_cluster_mass.npy',
                tempered_mass.detach().cpu().numpy(),
            )
        elif raw_mass is not None:
            np.save(
                artifact_dir / 'global_cluster_mass.npy',
                raw_mass.detach().cpu().numpy(),
            )
        for idx, (signature, mass, transport) in enumerate(zip(
            self.server.latest_signatures,
            self.server.latest_cluster_masses,
            transports,
        )):
            np.save(
                artifact_dir / f'view_{idx:02d}_spatial_signature.npy',
                signature.detach().cpu().numpy(),
            )
            np.save(
                artifact_dir / f'view_{idx:02d}_cluster_mass.npy',
                mass.detach().cpu().numpy(),
            )
            np.save(
                artifact_dir / f'view_{idx:02d}_sinkhorn_transport.npy',
                transport.detach().cpu().numpy(),
            )
        for idx, stats in enumerate(self.server.latest_semantic_signatures or []):
            np.save(
                artifact_dir / f'view_{idx:02d}_semantic_signature.npy',
                stats.detach().cpu().numpy(),
            )

    def _save_diagnostics(self, y_true, raw_prediction, transports,
                          clustering=None, foreground=None):
        """Persist the v2/v5 outline diagnostics for the best round."""
        artifact_dir = self.output_dir / 'artifacts'
        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            'ablation_variant': self.ablation_variant,
            'ablation_flags': self.ablation_flags,
            'graph_beta': float(self.config.get('graph_beta', 0.5)),
            'csata_dual_signature': bool(self.config.get('csata_dual_signature', True)),
            'csata_balanced': bool(self.config.get('csata_balanced', True)),
            'sinkhorn_epsilon': float(self.config.get('sinkhorn_epsilon', 0.05)),
            'lambda_spatial': float(self.config.get('lambda_spatial', 1.0)),
            'lambda_semantic': float(self.config.get('lambda_semantic', 1.0)),
            'mass_uniform_alpha': float(self.config.get('mass_uniform_alpha', 0.0)),
            # outline v6, module I (RADG-v1)
            'reliability_adaptive_graph': bool(
                self.config.get('reliability_adaptive_graph', False)
            ),
            'radg_beta_min': float(self.config.get('radg_beta_min', 0.2)),
            'radg_beta_max': float(self.config.get('radg_beta_max', 0.8)),
            'radg_edge_reliability': str(
                self.config.get('radg_edge_reliability', 'geometric_mean')
            ),
            'radg_mode': str(self.config.get('radg_mode', 'centered')),
            'radg_base_beta': float(self.config.get('radg_base_beta', 0.5)),
            'radg_gamma': float(self.config.get('radg_gamma', 0.5)),
            'radg_beta_floor': float(self.config.get('radg_beta_floor', 0.2)),
            'radg_beta_ceiling': float(self.config.get('radg_beta_ceiling', 0.8)),
            'radg_center_tolerance': float(self.config.get('radg_center_tolerance', 0.005)),
            'deterministic_mode': bool(self.config.get('deterministic_mode', False)),
            'missing_mode': self.config.get('missing_mode', 'random'),
            'missing_seed': self.config.get('missing_seed'),
            'mask_stage': dataset_module.MASK_STAGE,
            'missing_rate': float(self.config.get('missing_rate', 0.3)),
            'drop_nonpositive_labels': bool(
                self.config.get('drop_nonpositive_labels', False)
            ),
        }
        # C_T / H_T / D_inter.  Outline v5: descriptive only -- r(C_T, ACC) < 0
        # and r(H_T, ACC) > 0, so these must never be read as "higher is better".
        payload.update(
            alignment_diagnostics(self.server.latest_signatures, transports)
        )
        # outline v5: raw vs tempered anchor mass on the same concentration basis
        raw_mass = getattr(self.server, 'global_cluster_mass', None)
        tempered_mass = getattr(self.server, 'global_cluster_mass_tempered', None)
        payload['global_mass_raw'] = (
            None if raw_mass is None
            else [float(v) for v in raw_mass.detach().cpu().reshape(-1)]
        )
        payload['global_mass_tempered'] = (
            None if tempered_mass is None
            else [float(v) for v in tempered_mass.detach().cpu().reshape(-1)]
        )
        payload['mass_concentration_raw'] = (
            None if raw_mass is None
            else proportion_concentration(raw_mass.detach().cpu().numpy())
        )
        payload['mass_concentration_tempered'] = (
            None if tempered_mass is None
            else proportion_concentration(tempered_mass.detach().cpu().numpy())
        )
        # D_G (graph diagnostic, produced by dataset.save_missing_and_graph_artifacts)
        graph_path = artifact_dir / 'graph_diagnostics.json'
        if graph_path.exists():
            with open(graph_path, 'r', encoding='utf-8') as handle:
                payload['graph_diagnostics_D_G'] = json.load(handle)
        else:
            payload['graph_diagnostics_D_G'] = None
        # outline v6, module I: the full per-view RADG payload (beta / r_i
        # distributions, actual beta bounds, Delta_A, the r_i-vs-degradation
        # correlations) plus its flattened view averages.
        payload['radg'] = self.radg_diagnostics
        payload['radg_flat'] = self.radg_flat_metrics
        # confusion matrix + cluster sizes + concentration + per-class recall/purity
        if clustering is None:
            clustering = clustering_diagnostics(y_true, raw_prediction)
        payload['clustering'] = clustering
        payload['foreground_metrics'] = (
            foreground if foreground is not None
            else _foreground_metrics(y_true, raw_prediction)
        )
        _json_dump(artifact_dir / 'diagnostics.json', payload)
        np.save(
            artifact_dir / 'confusion_matrix.npy',
            np.asarray(clustering['confusion_matrix'], dtype=np.int64),
        )
        return payload

    def start(self):
        started_at = datetime.now().astimezone()
        start_clock = time.perf_counter()
        total_rounds = int(self.config['n_epoches'])
        dataset_name = self.config['dataset_name']
        history = []
        errors = []
        best_metrics = {
            'ACC': -1.0,
            'Kappa': 0.0,
            'NMI': 0.0,
            'ARI': 0.0,
            'PURITY': 0.0,
        }
        best_round = 0
        best_foreground = {}
        observed_masks = [client.data_loader['mask'] for client in self.clients]
        observed_rates = [float(mask.float().mean().cpu()) for mask in observed_masks]

        progress = tqdm(
            range(total_rounds),
            desc=f'Training FSRIMVC-V5 on {dataset_name} [{self.device}]',
            leave=True,
        )
        for round_idx in progress:
            round_start = time.perf_counter()
            try:
                global_params = self.server.distribute_model()
                prototypes = []
                local_soft_labels = []
                local_semantic_stats = []
                reconstruction_losses = []
                for client in self.clients:
                    (
                        _,
                        metrics,
                        prototype,
                        soft_labels,
                        semantic_stats,
                    ) = client.local_reconstruction(global_params, round_idx + 1)
                    prototypes.append(prototype)
                    local_soft_labels.append(soft_labels)
                    local_semantic_stats.append(semantic_stats)
                    reconstruction_losses.append(metrics['reconstruction_loss'])

                if self.ablation_flags['csata_alignment']:
                    transports, alignment_loss = self.server.align_spatial_signatures(
                        local_soft_labels,
                        local_semantic_stats,
                        observed_masks,
                        self.spatial_basis,
                    )
                else:
                    # A0-A2: no CSATA alignment.  Still record the spatial
                    # signatures so D_inter can be reported, but use an
                    # identity transport (aligned soft labels == raw ones).
                    self.server.compute_signatures(
                        local_soft_labels,
                        local_semantic_stats,
                        observed_masks,
                        self.spatial_basis,
                    )
                    n_clusters = int(local_soft_labels[0].shape[1])
                    identity = torch.eye(n_clusters, device=self.device)
                    transports = [identity for _ in local_soft_labels]
                    alignment_loss = torch.zeros((), device=self.device)

                if self.ablation_flags['global_consensus']:
                    global_consensus, _ = self.server.build_global_consensus(
                        local_soft_labels, transports, observed_masks
                    )
                else:
                    global_consensus = None

                client_updates = []
                aligned_after_training = []
                consensus_losses = []
                total_local_losses = []
                for client, prototype, transport in zip(
                    self.clients, prototypes, transports
                ):
                    update, aligned, consensus_loss, total_loss = client.consensus_training(
                        prototype, transport, global_consensus
                    )
                    client_updates.append(update)
                    aligned_after_training.append(aligned)
                    consensus_losses.append(consensus_loss)
                    total_local_losses.append(total_loss)

                self.server.update_global_model(client_updates)
                final_probability = _masked_average(
                    aligned_after_training, observed_masks
                )
                y_true = self.clients[0].data_loader['y'].detach().cpu().numpy()
                raw_prediction = torch.argmax(
                    final_probability, dim=1
                ).detach().cpu().numpy()
                acc, _, kappa, nmi, ari, _, _, _, purity = full_metric(
                    y_true.copy(), raw_prediction.copy(), is_refined=False
                )
                round_metrics = {
                    'round': round_idx + 1,
                    'ACC': float(acc),
                    'Kappa': float(kappa),
                    'NMI': float(nmi),
                    'ARI': float(ari),
                    'PURITY': float(purity),
                    'alignment_loss': float(alignment_loss.cpu()),
                    'mean_reconstruction_loss': float(np.mean(reconstruction_losses)),
                    'mean_consensus_loss': float(np.mean(consensus_losses)),
                    'mean_local_total_loss': float(np.mean(total_local_losses)),
                    'round_seconds': float(time.perf_counter() - round_start),
                    'observed_rates': json.dumps(observed_rates),
                }
                history.append(round_metrics)
                self._write_history(history)
                # ---- outline v9 module III: by-pass evaluation of the arms --
                # B0 (``final_probability``) is already computed above and is
                # never modified; the adapter is strictly read-only.
                if self.completion.enabled:
                    b0_row = {
                        'round': round_idx + 1,
                        'ACC': float(acc),
                        'Kappa': float(kappa),
                        'NMI': float(nmi),
                        'ARI': float(ari),
                        'PURITY': float(purity),
                    }
                    self.completion.record_b0(b0_row, raw_prediction, y_true)
                    adapter_start = time.perf_counter()
                    arm_rows, completion_info = self.completion.evaluate(
                        round_idx + 1, y_true, metric_fn=full_metric
                    )
                    self.completion_timing['adapter_seconds'] += (
                        time.perf_counter() - adapter_start
                    )
                    round_metrics['completion_eval_seconds'] = round(
                        time.perf_counter() - adapter_start, 4
                    )
                    for arm_name, arm_row in sorted(arm_rows.items()):
                        round_metrics[f'{arm_name}_ACC'] = arm_row['ACC']
                    if (round_idx + 1) in self.completion.diagnostic_epochs:
                        diagnostic_start = time.perf_counter()
                        self.completion.pseudo_missing(round_idx + 1, y_true)
                        self.completion_timing['diagnostic_seconds'] += (
                            time.perf_counter() - diagnostic_start
                        )
                progress.set_postfix(
                    ACC=f'{acc:.4f}',
                    best=f"{max(acc, best_metrics['ACC']):.4f}",
                )

                if acc > best_metrics['ACC']:
                    best_metrics = {
                        'ACC': float(acc),
                        'Kappa': float(kappa),
                        'NMI': float(nmi),
                        'ARI': float(ari),
                        'PURITY': float(purity),
                    }
                    best_round = round_idx + 1
                    best_prediction = matched_labels(y_true, raw_prediction)
                    np.save(self.output_dir / 'artifacts' / 'best_raw_prediction.npy', raw_prediction)
                    np.save(self.output_dir / 'artifacts' / 'evaluation_labels.npy', y_true)
                    # outline v5: the cluster-concentration diagnostics and the
                    # MUUFL foreground metrics describe the *same best round* as
                    # the headline ACC, so they are computed once here.
                    clustering = clustering_diagnostics(
                        y_true,
                        raw_prediction,
                        n_clusters=int(final_probability.shape[1]),
                    )
                    foreground = _foreground_metrics(y_true, raw_prediction)
                    best_foreground = foreground
                    _json_dump(self.output_dir / 'metrics.json', {
                        **best_metrics,
                        'best_round': best_round,
                        'method_stage': self.ablation_variant or 'V5',
                        'ablation_variant': self.ablation_variant,
                        'ablation_flags': self.ablation_flags,
                        'graph_beta': float(self.config.get('graph_beta', 0.5)),
                        'csata_dual_signature': bool(self.config.get('csata_dual_signature', True)),
                        'csata_balanced': bool(self.config.get('csata_balanced', True)),
                        'sinkhorn_epsilon': float(self.config.get('sinkhorn_epsilon', 0.05)),
                        'lambda_spatial': float(self.config.get('lambda_spatial', 1.0)),
                        'lambda_semantic': float(self.config.get('lambda_semantic', 1.0)),
                        'mass_uniform_alpha': float(self.config.get('mass_uniform_alpha', 0.0)),
                        # outline v6, module I: RADG configuration + diagnostics
                        'reliability_adaptive_graph': bool(
                            self.config.get('reliability_adaptive_graph', False)
                        ),
                        'radg_beta_min': float(self.config.get('radg_beta_min', 0.2)),
                        'radg_beta_max': float(self.config.get('radg_beta_max', 0.8)),
                        'radg_edge_reliability': str(
                            self.config.get('radg_edge_reliability', 'geometric_mean')
                        ),
                        'radg_mode': str(self.config.get('radg_mode', 'centered')),
                        'radg_base_beta': float(self.config.get('radg_base_beta', 0.5)),
                        'radg_gamma': float(self.config.get('radg_gamma', 0.5)),
                        'radg_beta_floor': float(self.config.get('radg_beta_floor', 0.2)),
                        'radg_beta_ceiling': float(self.config.get('radg_beta_ceiling', 0.8)),
                        'radg_center_tolerance': float(self.config.get('radg_center_tolerance', 0.005)),
                        'deterministic_mode': bool(
                            self.config.get('deterministic_mode', False)
                        ),
                        **self.radg_flat_metrics,
                        'missing_mode': self.config.get('missing_mode', 'random'),
                        'missing_seed': self.config.get('missing_seed'),
                        'mask_stage': dataset_module.MASK_STAGE,
                        'missing_rate': float(self.config.get('missing_rate', 0.3)),
                        'drop_nonpositive_labels': bool(
                            self.config.get('drop_nonpositive_labels', False)
                        ),
                        # outline v5 concentration diagnostics (former "collapse")
                        'cluster_max_ratio': clustering['cluster_max_ratio'],
                        'cluster_min_ratio': clustering['cluster_min_ratio'],
                        'cluster_entropy': clustering['cluster_entropy'],
                        'effective_num_clusters': clustering['effective_num_clusters'],
                        'num_predicted_clusters': clustering['num_predicted_clusters'],
                        # outline v5 foreground-only MUUFL metrics (y_i != -1)
                        'ACC_foreground': foreground.get('ACC_foreground'),
                        'NMI_foreground': foreground.get('NMI_foreground'),
                        'ARI_foreground': foreground.get('ARI_foreground'),
                        'foreground_nodes': foreground.get('foreground_nodes'),
                        'num_active_nodes': int(y_true.size),
                        'num_gt_classes': int(np.unique(y_true).size),
                    })
                    self.dataset.save_prediction(best_prediction, self.output_dir)
                    self._save_v5_artifacts(final_probability, transports)
                    self._save_diagnostics(
                        y_true, raw_prediction, transports,
                        clustering=clustering, foreground=foreground,
                    )

            except Exception as error:
                error_payload = {
                    'round': round_idx + 1,
                    'error': repr(error),
                    'traceback': traceback.format_exc(),
                }
                errors.append(error_payload)
                _json_dump(self.output_dir / 'errors.json', errors)
                if not self.config.get('continue_on_round_error', False):
                    raise

        completion_summaries = {}
        if self.completion.enabled:
            completion_summaries = self.completion.finalise()
        duration = time.perf_counter() - start_clock
        finished_at = datetime.now().astimezone()
        if best_metrics['ACC'] < 0:
            raise RuntimeError('Training finished without a successful evaluation round.')

        if self.config.get('save_model', True):
            checkpoint_dir = self.output_dir / 'checkpoints'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                self.server.global_parameters,
                checkpoint_dir / 'final_server_model.pth',
            )

        summary = {
            'method_stage': 'V5',
            'dataset': dataset_name,
            'missing_mode': self.config.get('missing_mode', 'random'),
            'missing_seed': self.config.get('missing_seed'),
            'mask_stage': dataset_module.MASK_STAGE,
            'missing_rate': float(self.config.get('missing_rate', 0.3)),
            'mass_uniform_alpha': float(self.config.get('mass_uniform_alpha', 0.0)),
            # outline v6, module I
            'reliability_adaptive_graph': bool(
                self.config.get('reliability_adaptive_graph', False)
            ),
            'radg_beta_min': float(self.config.get('radg_beta_min', 0.2)),
            'radg_beta_max': float(self.config.get('radg_beta_max', 0.8)),
            'radg_edge_reliability': str(
                self.config.get('radg_edge_reliability', 'geometric_mean')
            ),
            'radg_mode': str(self.config.get('radg_mode', 'centered')),
            'radg_base_beta': float(self.config.get('radg_base_beta', 0.5)),
            'radg_gamma': float(self.config.get('radg_gamma', 0.5)),
            'radg_beta_floor': float(self.config.get('radg_beta_floor', 0.2)),
            'radg_beta_ceiling': float(self.config.get('radg_beta_ceiling', 0.8)),
            'radg_center_tolerance': float(self.config.get('radg_center_tolerance', 0.005)),
            'deterministic_mode': bool(self.config.get('deterministic_mode', False)),
            'radg_metrics': self.radg_flat_metrics,
            'device': str(self.device),
            'start_time': started_at.isoformat(),
            'end_time': finished_at.isoformat(),
            'runtime_seconds': float(duration),
            'best_round': best_round,
            'metrics': best_metrics,
            'foreground_metrics': best_foreground,
            'completed_rounds': len(history),
            'failed_rounds': len(errors),
            'completion_enabled': bool(self.completion.enabled),
            'completion_stage': self.completion.stage,
            'completion_arms': list(self.completion.arms) if self.completion.enabled else [],
            'completion_summaries': completion_summaries,
            'completion_timing': self.completion_timing,
        }
        _json_dump(self.output_dir / 'training_summary.json', summary)
        if self.completion.enabled:
            print('\nCompletion arms (best-round ACC over %d rounds)' % len(history))
            print('-' * 56)
            for arm in self.completion.arms:
                entry = completion_summaries.get(arm)
                if not entry:
                    continue
                print('  %-5s best=%.4f (r%3d)  final=%.4f  tail30=%.4f  median=%.4f' % (
                    arm, entry['ACC_best'], entry['ACC_best_round'], entry['ACC_final'],
                    entry['ACC_tail30'], entry['ACC_median'],
                ))
            print('-' * 56)

        print(f'\nFSRIMVC-V5 Performance on {dataset_name}')
        print('=' * 56)
        print(f"Missing: {self.config.get('missing_mode', 'random')} @ {float(self.config.get('missing_rate', 0.3)):.2f}")
        print(f"Alpha  : {float(self.config.get('mass_uniform_alpha', 0.0)):.4f}  (mass_uniform_alpha)")
        if self.radg_flat_metrics.get('radg_enabled'):
            if self.radg_flat_metrics.get('radg_mode') == 'centered':
                print(
                    "RADG   : centered  base={:.2f} gamma={:.2f} clip=[{:.2f},{:.2f}]  "
                    "beta_mean={}  |E[b]-base|={}  |Delta_A|={}".format(
                        float(self.config.get('radg_base_beta', 0.5)),
                        float(self.config.get('radg_gamma', 0.5)),
                        float(self.config.get('radg_beta_floor', 0.2)),
                        float(self.config.get('radg_beta_ceiling', 0.8)),
                        self.radg_flat_metrics.get('beta_mean'),
                        self.radg_flat_metrics.get('beta_mean_error_from_base'),
                        self.radg_flat_metrics.get('graph_relative_frobenius_change'),
                    )
                )
            else:
                print(
                    "RADG   : uncentered  beta in [{:.2f}, {:.2f}]  beta_mean={}  "
                    "|E[b]-0.5|={}  |Delta_A|={}".format(
                        float(self.config.get('radg_beta_min', 0.2)),
                        float(self.config.get('radg_beta_max', 0.8)),
                        self.radg_flat_metrics.get('beta_mean'),
                        self.radg_flat_metrics.get('beta_mean_error_from_base'),
                        self.radg_flat_metrics.get('graph_relative_frobenius_change'),
                    )
                )
        else:
            print("RADG   : off (fixed fusion, beta={:.2f})".format(
                float(self.config.get('graph_beta', 0.5))
            ))
        print(f"ACC    : {best_metrics['ACC']:.8f}")
        print(f"Kappa  : {best_metrics['Kappa']:.8f}")
        print(f"NMI    : {best_metrics['NMI']:.8f}")
        print(f"ARI    : {best_metrics['ARI']:.8f}")
        print(f"PURITY : {best_metrics['PURITY']:.8f}")
        if best_foreground.get('ACC_foreground') is not None:
            print(f"ACC(fg): {best_foreground['ACC_foreground']:.8f}  (y_i != -1)")
        print(f'Best round: {best_round}')
        print(f'Runtime   : {duration:.2f} seconds')
        print(f'Results   : {self.output_dir}')
        print('=' * 56)
        return summary
