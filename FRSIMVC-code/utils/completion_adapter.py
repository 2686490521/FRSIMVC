"""Read-only evaluation adapter for Cross-Client Semantic-Spatial Completion.

Outline v9 sections 7-9.  The adapter answers one question: *given the model as
it stands at the end of this round, what would every arm predict?*  It never

* writes back to a model, optimiser, prototype cache, graph or anchor,
* draws from the global RNG (KMeans is seeded with a private generator and the
  global states are snapshotted/restored anyway),
* touches the ground truth (GT only enters the metric computation),
* changes the training trajectory (every arm is a by-product of one shared run).

``B0`` stays exactly what :func:`Flsimulator._masked_average` produced; the
adapter only adds ``E0/G0/D0/C*``.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from client import FedAvg_client
from utils.completion import (
    NUMERICAL_EPS,
    barycentric_prototypes,
    completion_arms,
)
from utils.csata import (
    align_to_global_anchors,
    compute_semantic_stats,
    compute_spatial_signature,
    align_soft_labels,
    standardize_columns,
)

ARM_ORDER = ('B0', 'E0', 'G0', 'D0', 'C025', 'C050', 'C075', 'C100')
C_ARM_LAMBDA = {'C025': 0.25, 'C050': 0.50, 'C075': 0.75, 'C100': 1.00}


def _tensor_hash(tensor):
    array = tensor.detach().reshape(-1).to('cpu').numpy().astype(np.float64)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


class CompletionAdapter:
    """Frozen snapshot + eight-arm evaluation, attached to one simulator."""

    def __init__(self, config, clients, server, spatial_basis, device, output_dir):
        self.config = config
        self.clients = clients
        self.server = server
        self.spatial_basis = spatial_basis
        self.device = device
        self.output_dir = Path(output_dir)
        self.enabled = bool(config.get('completion_enabled', False))
        self.stage = str(config.get('completion_stage', 'inference_only'))
        self.adapter_seed = int(config.get('adapter_seed', 20260922))
        self.diagnostic_seed = int(config.get('diagnostic_seed', 100000))
        self.numerical_eps = float(config.get('numerical_eps', NUMERICAL_EPS))
        self.min_valid_mass = float(config.get('completion_min_valid_mass', 0.5))
        self.lambdas = tuple(float(x) for x in config.get(
            'completion_lambdas', [0.25, 0.5, 0.75, 1.0]
        ))
        self.arms = tuple(config.get('completion_arms', list(ARM_ORDER)))
        self.diagnostic_epochs = tuple(int(x) for x in config.get(
            'diagnostic_epochs', [100, 200, 300]
        ))
        self.probability_buffer = bool(config.get('completion_save_probabilities', False))
        # per-arm state
        self._history = {}
        self._best = {}
        self._arm_dirs = {}
        self._stats_path = None
        self._diagnostic_path = None
        self._prepare_dirs()

    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------
    def _prepare_dirs(self):
        if not self.enabled:
            return
        self.arm_root = self.output_dir / 'arms'
        self.shared_dir = self.output_dir / 'shared'
        self.diag_dir = self.output_dir / 'diagnostics'
        self.arm_root.mkdir(parents=True, exist_ok=True)
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        self.diag_dir.mkdir(parents=True, exist_ok=True)
        for arm in self.arms:
            arm_dir = self.arm_root / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            self._arm_dirs[arm] = arm_dir
            self._history[arm] = []
            self._best[arm] = {'ACC': -1.0, 'round': 0}
        self._stats_path = self.diag_dir / 'completion_round_stats.jsonl'
        self._diagnostic_path = self.diag_dir / 'pseudo_missing.jsonl'
        self._write_shared()

    def _write_shared(self):
        masks = [client.data_loader['mask'].detach().cpu().numpy() for client in self.clients]
        payload = {
            'n_views': len(self.clients),
            'n_nodes': int(masks[0].shape[0]),
            'n_clusters': int(self.clients[0].data_loader['n_classes']),
            'adapter_seed': self.adapter_seed,
            'diagnostic_seed': self.diagnostic_seed,
            'completion_lambdas': list(self.lambdas),
            'completion_arms': list(self.arms),
            'completion_stage': self.stage,
            'min_valid_mass': self.min_valid_mass,
            'numerical_eps': self.numerical_eps,
        }
        np.savez_compressed(
            self.shared_dir / 'masks_and_nodes.npz',
            **{f'mask_view_{i:02d}': m for i, m in enumerate(masks)},
        )
        with open(self.shared_dir / 'completion_manifest.json', 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)

    # ------------------------------------------------------------------
    # snapshot construction
    # ------------------------------------------------------------------
    def build_snapshot(self, mask_override=None, x_override=None, seed=None):
        """Freeze H / P / S / Pi / R / Pbar for every view at this time point.

        ``mask_override`` / ``x_override`` are per-view dicts used by the
        pseudo-missing probe only; the main arms always run with the true mask.
        """
        random_state = self.adapter_seed if seed is None else int(seed)
        rng = {
            'random': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng['cuda'] = [t.clone() for t in torch.cuda.get_rng_state_all()]
        modes = [client.model.training for client in self.clients]

        views = []
        spatial_signatures = []
        semantic_stats = []
        cluster_masses = []
        try:
            for index, client in enumerate(self.clients):
                loader = client.data_loader
                x = loader['x'] if not x_override else x_override.get(index, loader['x'])
                mask = loader['mask'] if not mask_override else mask_override.get(
                    index, loader['mask']
                )
                hidden = FedAvg_client.snapshot_forward(client.model, x, loader['adj'])
                n_clusters = int(loader['n_classes'])
                prototypes, fit_info = FedAvg_client.fit_snapshot_prototypes(
                    hidden, mask, n_clusters, random_state=random_state
                )
                if fit_info['status'] != 'ok':
                    raise RuntimeError(
                        f'view {index}: only {fit_info["n_observed"]} observed nodes '
                        f'for {n_clusters} clusters'
                    )
                soft = F.softmax(-torch.cdist(hidden, prototypes, p=2), dim=1)
                signature, mass = compute_spatial_signature(
                    soft, mask, self.spatial_basis, eps=self.numerical_eps
                )
                stats = compute_semantic_stats(soft, hidden, mask, eps=self.numerical_eps)
                views.append({
                    'mask': mask,
                    'H': hidden,
                    'P': prototypes,
                    'S': soft,
                })
                spatial_signatures.append(signature)
                semantic_stats.append(stats)
                cluster_masses.append(mass)

            # ---- alignment on a *copy* of the anchors (never written back) --
            anchor_copy = None
            if self.server.global_anchors is not None:
                anchor_copy = {
                    key: value.detach().clone()
                    for key, value in self.server.global_anchors.items()
                }
            dual = bool(self.config.get('csata_dual_signature', True))
            if dual:
                stacked = torch.cat(
                    [stats.to(device=spatial_signatures[0].device) for stats in semantic_stats],
                    dim=0,
                )
                standardized = standardize_columns(stacked)
                sizes = [stats.shape[0] for stats in semantic_stats]
                semantic_signatures = list(torch.split(standardized, sizes, dim=0))
            else:
                semantic_signatures = None

            balanced = bool(self.config.get('csata_balanced', True))
            outputs = align_to_global_anchors(
                spatial_signatures=spatial_signatures,
                semantic_signatures=semantic_signatures,
                cluster_masses=cluster_masses,
                global_anchors=anchor_copy,
                n_clusters=int(spatial_signatures[0].shape[0]),
                random_state=random_state,
                epsilon=float(self.config.get('sinkhorn_epsilon', 0.05)),
                iterations=int(self.config.get('sinkhorn_iterations', 100)),
                momentum=float(self.config.get('global_anchor_momentum', 0.0)),
                lambda_spatial=float(self.config.get('lambda_spatial', 1.0)),
                lambda_semantic=float(self.config.get('lambda_semantic', 1.0)),
                balanced=balanced,
                global_mass=self.server.global_cluster_mass if balanced else None,
                mass_uniform_alpha=float(self.config.get('mass_uniform_alpha', 0.0)),
                return_details=True,
            )
            # outputs[0] (the refreshed anchors) is deliberately discarded.
            conditionals = outputs[1]
            details = outputs[6]

            for index, view in enumerate(views):
                plan = details[index]['plan']
                pbar, col_mass, valid_cols, info = barycentric_prototypes(
                    plan, view['P'], eps=self.numerical_eps
                )
                view['R'] = conditionals[index]
                view['Pi'] = plan
                view['col_mass'] = col_mass
                view['valid_cols'] = valid_cols
                view['Pbar'] = pbar
                view['Sbar'] = align_soft_labels(view['S'], view['R'])
                view['plan_info'] = info
                view['prototype_hash'] = _tensor_hash(view['P'])
        finally:
            random.setstate(rng['random'])
            np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch'])
            if torch.cuda.is_available() and 'cuda' in rng:
                torch.cuda.set_rng_state_all(rng['cuda'])
            for client, mode in zip(self.clients, modes):
                client.model.train(mode)

        anchor_hash = None
        if self.server.global_anchors is not None:
            anchor_hash = _tensor_hash(self.server.global_anchors['spatial'])
        return views, {
            'anchor_hash': anchor_hash,
            'prototype_hashes': [view['prototype_hash'] for view in views],
            'adapter_seed': random_state,
        }

    # ------------------------------------------------------------------
    # per-round evaluation
    # ------------------------------------------------------------------
    def evaluate(self, round_idx, y_true, b0_probability=None, metric_fn=None):
        """Evaluate every non-B0 arm for this round. Returns (rows, info)."""
        if not self.enabled:
            return {}, {}
        views, adapter_info = self.build_snapshot()
        probabilities, diagnostics = completion_arms(
            views,
            lambdas=self.lambdas,
            min_valid_mass=self.min_valid_mass,
            eps=self.numerical_eps,
        )
        rows = {}
        for arm in self.arms:
            if arm == 'B0':
                continue
            prob = probabilities.get(arm)
            if prob is None:
                raise KeyError(f'arm {arm} was not produced by completion_arms')
            prediction = torch.argmax(prob, dim=1).detach().cpu().numpy()
            acc, _, kappa, nmi, ari, _, _, _, purity = metric_fn(
                y_true.copy(), prediction.copy(), is_refined=False
            )
            row = {
                'round': round_idx,
                'ACC': float(acc),
                'Kappa': float(kappa),
                'NMI': float(nmi),
                'ARI': float(ari),
                'PURITY': float(purity),
            }
            rows[arm] = row
            self._append_history(arm, row)
            if float(acc) > self._best[arm]['ACC']:
                self._best[arm] = {'ACC': float(acc), 'round': round_idx}
                self._write_arm_best(arm, row, prediction, y_true)

        diagnostics['round'] = round_idx
        diagnostics.update(adapter_info)
        self._append_jsonl(self._stats_path, diagnostics)
        return rows, diagnostics

    def record_b0(self, row, prediction, y_true):
        """Mirror the untouched B0 trajectory into ``arms/B0`` for side-by-side analysis."""
        if not self.enabled or 'B0' not in self._arm_dirs:
            return
        self._append_history('B0', row)
        if float(row['ACC']) > self._best['B0']['ACC']:
            self._best['B0'] = {'ACC': float(row['ACC']), 'round': int(row['round'])}
            self._write_arm_best('B0', row, prediction, y_true)

    def _append_history(self, arm, row):
        self._history[arm].append(row)
        path = self._arm_dirs[arm] / 'history.jsonl'
        self._append_jsonl(path, row)

    def _write_arm_best(self, arm, row, prediction, y_true):
        arm_dir = self._arm_dirs[arm]
        payload = dict(row)
        payload['method_stage'] = 'completion_v1'
        payload['arm'] = arm
        payload['best_round'] = row['round']
        payload['num_active_nodes'] = int(y_true.size)
        payload['num_gt_classes'] = int(np.unique(y_true).size)
        tmp = arm_dir / 'metrics.json.tmp'
        with open(tmp, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
        tmp.replace(arm_dir / 'metrics.json')
        np.save(arm_dir / 'best_raw_prediction.npy', prediction)
        np.save(arm_dir / 'evaluation_labels.npy', y_true)

    @staticmethod
    def _append_jsonl(path, payload):
        if path is None:
            return
        with open(path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + '\n')

    # ------------------------------------------------------------------
    # pseudo-missing probe
    # ------------------------------------------------------------------
    def pseudo_missing(self, round_idx, y_true):
        """Node-level probe: hide 10% of a view's observed nodes and compare.

        Limitation recorded honestly in the manifest: the kNN graph is built
        once by the frozen dataset pipeline and is *not* rebuilt for the probe,
        so hidden features still influence their neighbours through the graph.
        The probe therefore measures the completion mechanism, not a full
        re-run of the pixel-level missing protocol.
        """
        if not self.enabled or self.stage == 'off':
            return None
        base_views, _ = self.build_snapshot()
        results = []
        for target in range(len(self.clients)):
            mask_np = self.clients[target].data_loader['mask'].detach().cpu().numpy()
            observed_idx = np.nonzero(mask_np)[0]
            other_masks = [
                self.clients[u].data_loader['mask'].detach().cpu().numpy()
                for u in range(len(self.clients)) if u != target
            ]
            other_observed = np.zeros(mask_np.shape[0], dtype=bool)
            for m in other_masks:
                other_observed |= m.astype(bool)
            candidates = observed_idx[other_observed[observed_idx]]
            if candidates.size == 0:
                results.append({'view': target, 'status': 'no_candidates'})
                continue
            rng = np.random.RandomState(self.diagnostic_seed + target)
            n_hide = max(1, int(round(0.1 * candidates.size)))
            hidden_idx = rng.choice(candidates, size=n_hide, replace=False)
            hidden_mask = np.zeros(mask_np.shape[0], dtype=bool)
            hidden_mask[hidden_idx] = True
            diag_mask = torch.from_numpy(mask_np & ~hidden_mask).to(self.device)
            x = self.clients[target].data_loader['x'].clone()
            x[hidden_idx] = 0.0
            try:
                views, _ = self.build_snapshot(
                    mask_override={target: diag_mask},
                    x_override={target: x},
                    seed=self.diagnostic_seed,
                )
            except Exception as error:  # pragma: no cover - defensive
                results.append({'view': target, 'status': f'failed: {error!r}'})
                continue
            h_graph = views[target]['H']
            h_ref = base_views[target]['H']
            from utils.completion import donor_distribution, restrict_to_valid_columns, \
                semantic_projection, convex_combination
            q_raw, donor_count = donor_distribution(
                [v['Sbar'] for v in views],
                [v['mask'] for v in views],
                target=target,
                eps=self.numerical_eps,
            )
            q_eff, _, _ = restrict_to_valid_columns(
                q_raw, views[target]['valid_cols'], eps=self.numerical_eps
            )
            idx = torch.from_numpy(hidden_idx).to(h_graph.device)
            entry = {
                'view': target,
                'round': round_idx,
                'status': 'ok',
                'n_hidden': int(hidden_idx.size),
                'donor_count_mean': float(donor_count[idx].float().mean().detach().cpu()),
            }
            for lam in self.lambdas:
                h_sem = semantic_projection(q_eff, views[target]['Pbar'])
                h_comp = convex_combination(h_graph, h_sem, lam)
                name = 'C%03d' % int(round(float(lam) * 100))
                entry[name] = self._recovery_stats(
                    h_comp[idx], h_graph[idx], h_ref[idx]
                )
            entry['G0'] = self._recovery_stats(h_graph[idx], h_graph[idx], h_ref[idx])
            results.append(entry)
        payload = {'round': round_idx, 'views': results}
        self._append_jsonl(self._diagnostic_path, payload)
        return payload

    @staticmethod
    def _recovery_stats(candidate, graph, reference, eps=1e-12):
        diff = candidate - reference
        reference_energy = float((reference.detach() ** 2).sum().detach().cpu())
        nmse = float((diff.detach() ** 2).sum().detach().cpu()) / (reference_energy + eps)
        cosine = float(
            torch.nn.functional.cosine_similarity(
                candidate.detach(), reference.detach(), dim=1
            ).mean().detach().cpu()
        )
        return {'nmse': nmse, 'cosine_mean': cosine, 'cosine_distance_mean': 1.0 - cosine}

    # ------------------------------------------------------------------
    # finalisation
    # ------------------------------------------------------------------
    def finalise(self):
        """Write per-arm summary (best / final / tail30 / median / std)."""
        if not self.enabled:
            return {}
        summaries = {}
        for arm, history in self._history.items():
            if not history:
                continue
            keys = ['ACC', 'Kappa', 'NMI', 'ARI', 'PURITY']
            best_row = max(history, key=lambda row: row['ACC'])
            final_row = history[-1]
            tail = history[-30:]
            summary = {'arm': arm, 'rounds': len(history)}
            for key in keys:
                values = np.array([row[key] for row in history], dtype=float)
                tail_values = np.array([row[key] for row in tail], dtype=float)
                summary[f'{key}_best'] = float(best_row[key])
                summary[f'{key}_best_round'] = int(best_row['round'])
                summary[f'{key}_final'] = float(final_row[key])
                summary[f'{key}_tail30'] = float(tail_values.mean())
                summary[f'{key}_median'] = float(np.median(values))
                summary[f'{key}_std'] = float(values.std(ddof=1)) if values.size > 1 else 0.0
            summaries[arm] = summary
            path = self._arm_dirs[arm] / 'summary.json'
            tmp = self._arm_dirs[arm] / 'summary.json.tmp'
            with open(tmp, 'w', encoding='utf-8') as handle:
                json.dump(summary, handle, indent=2)
            tmp.replace(path)
            self._write_history_csv(arm, history)
        out = self.output_dir / 'completion_summary.json'
        tmp = self.output_dir / 'completion_summary.json.tmp'
        with open(tmp, 'w', encoding='utf-8') as handle:
            json.dump(summaries, handle, indent=2)
        tmp.replace(out)
        return summaries

    def _write_history_csv(self, arm, history):
        import csv
        path = self._arm_dirs[arm] / 'history.csv'
        tmp = self._arm_dirs[arm] / 'history.csv.tmp'
        with open(tmp, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        tmp.replace(path)
