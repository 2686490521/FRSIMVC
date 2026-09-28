import torch

import models
from . import FedAvg_server
from utils.csata import (
    align_to_global_anchors,
    compute_spatial_signature,
    global_semantic_consensus,
    standardize_columns,
)


class Server:
    def __init__(self, config, device, n_input, n_samples):
        self.config = config
        self.device = device
        self.global_model = models.get_model(
            config['model_name'],
            gae_n_enc_1=config['gae_n_enc_1'],
            gae_n_enc_2=config['gae_n_enc_2'],
            gae_n_enc_3=config['gae_n_enc_3'],
            gae_n_dec_1=config['gae_n_dec_1'],
            gae_n_dec_2=config['gae_n_dec_2'],
            gae_n_dec_3=config['gae_n_dec_3'],
            n_input=n_input,
            n_samples=n_samples,
        ).to(device)
        self.global_parameters = self.global_model.state_dict()
        # global_anchors is a dict {'spatial': (K, d_s), ['semantic': (K, 3)]}
        self.global_anchors = None
        self.global_cluster_mass = None
        # outline v5 / BSSAT-v2: ``global_cluster_mass`` keeps the raw mass (the
        # momentum carrier), ``global_cluster_mass_tempered`` is the b' that the
        # transport actually used this round.
        self.global_cluster_mass_tempered = None
        self.latest_signatures = None
        self.latest_semantic_signatures = None
        self.latest_cluster_masses = None
        self.latest_transports = None
        self.latest_costs = None

    def distribute_model(self):
        parameters = self.global_parameters.copy()
        parameters.pop('encoder.gnn_1.weight', None)
        parameters.pop('decoder.gnn_6.weight', None)
        return parameters

    def update_global_model(self, client_updates_list):
        if self.config['algorithm'] != 'FedAvg':
            raise ValueError(
                f"Algorithm {self.config['algorithm']} implementation not found for server."
            )
        new_parameters = FedAvg_server.aggregate(client_updates_list)
        if new_parameters:
            self.global_parameters = new_parameters
            self.global_model.load_state_dict(self.global_parameters, strict=False)

    def _dual_signature_enabled(self):
        """BSSAT dual signature: spatial + client-side semantic statistics."""
        return bool(self.config.get('csata_dual_signature', True))

    def compute_signatures(
        self, soft_labels_list, semantic_stats_list, observed_masks, spatial_basis
    ):
        """Spatial signatures Q^(v), semantic statistics and per-cluster mass.

        Called for diagnostics even when CSATA alignment is switched off, so
        D_inter can still be reported.
        """
        spatial_signatures = []
        cluster_masses = []
        for soft_labels, mask in zip(soft_labels_list, observed_masks):
            signature, mass = compute_spatial_signature(
                soft_labels, mask, spatial_basis
            )
            spatial_signatures.append(signature)
            cluster_masses.append(mass)

        if self._dual_signature_enabled() and semantic_stats_list is not None:
            # Standardise on the client statistics of *this* round, so the
            # scale-free semantic anchors and the local statistics agree.
            stacked = torch.cat(
                [stats.to(device=spatial_signatures[0].device) for stats in semantic_stats_list],
                dim=0,
            )
            standardized = standardize_columns(stacked)
            sizes = [stats.shape[0] for stats in semantic_stats_list]
            semantic_signatures = list(torch.split(standardized, sizes, dim=0))
        else:
            semantic_signatures = None

        self.latest_signatures = spatial_signatures
        self.latest_semantic_signatures = semantic_signatures
        self.latest_cluster_masses = cluster_masses
        return spatial_signatures, semantic_signatures, cluster_masses

    def align_spatial_signatures(
        self, soft_labels_list, semantic_stats_list, observed_masks, spatial_basis
    ):
        """BSSAT: balanced spatial-semantic anchor transport."""
        spatial_signatures, semantic_signatures, cluster_masses = self.compute_signatures(
            soft_labels_list, semantic_stats_list, observed_masks, spatial_basis
        )

        n_clusters = soft_labels_list[0].shape[1]
        balanced = bool(self.config.get('csata_balanced', True))
        (
            self.global_anchors,
            transports,
            costs,
            alignment_loss,
            self.global_cluster_mass,
            self.global_cluster_mass_tempered,
        ) = align_to_global_anchors(
            spatial_signatures=spatial_signatures,
            semantic_signatures=semantic_signatures,
            cluster_masses=cluster_masses,
            global_anchors=self.global_anchors,
            n_clusters=n_clusters,
            random_state=int(self.config.get('seed', 42)),
            epsilon=float(self.config.get('sinkhorn_epsilon', 0.05)),
            iterations=int(self.config.get('sinkhorn_iterations', 100)),
            momentum=float(self.config.get('global_anchor_momentum', 0.0)),
            lambda_spatial=float(self.config.get('lambda_spatial', 1.0)),
            lambda_semantic=float(self.config.get('lambda_semantic', 1.0)),
            balanced=balanced,
            global_mass=self.global_cluster_mass if balanced else None,
            mass_uniform_alpha=float(self.config.get('mass_uniform_alpha', 0.0)),
        )
        self.latest_transports = transports
        self.latest_costs = costs
        return transports, alignment_loss

    @staticmethod
    def build_global_consensus(soft_labels_list, transports, observed_masks):
        return global_semantic_consensus(
            soft_labels_list, transports, observed_masks
        )
