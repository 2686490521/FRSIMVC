import numpy as np
from scipy import sparse
import torch

import models
from . import FedAvg_client


def _sparse_to_torch(adjacency, device):
    adjacency = adjacency.tocoo()
    indices = torch.tensor(
        np.vstack([adjacency.row, adjacency.col]), dtype=torch.long
    )
    values = torch.tensor(adjacency.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(
        indices, values, adjacency.shape, dtype=torch.float32
    ).coalesce().to(device)


class Client:
    def __init__(self, client_id, config, raw_data_dict, device):
        self.client_id = client_id
        self.config = config
        self.device = device
        self.data_loader = self._load_data(raw_data_dict)
        self.best_acc = -1.0
        self.model = models.get_model(
            config['model_name'],
            gae_n_enc_1=config['gae_n_enc_1'],
            gae_n_enc_2=config['gae_n_enc_2'],
            gae_n_enc_3=config['gae_n_enc_3'],
            gae_n_dec_1=config['gae_n_dec_1'],
            gae_n_dec_2=config['gae_n_dec_2'],
            gae_n_dec_3=config['gae_n_dec_3'],
            n_input=self.data_loader['x'].shape[1],
            n_samples=self.data_loader['x'].shape[0],
        ).to(device)

    def _load_data(self, raw):
        x = torch.from_numpy(raw['data']).float().to(self.device)
        target_x = torch.from_numpy(raw['target_data']).float().to(self.device)
        if sparse.issparse(raw['adj']):
            adj = _sparse_to_torch(raw['adj'], self.device)
        else:
            adj = torch.from_numpy(raw['adj']).float().to(self.device)
        return {
            'x': x,
            'target_x': target_x,
            'adj': adj,
            'mask': torch.from_numpy(raw['mask']).bool().to(self.device),
            'y': torch.from_numpy(raw['y']).long().to(self.device),
            'raw_shape': raw['raw_shape'],
            'n_classes': raw['n_classes'],
        }

    def local_reconstruction(self, global_parameters, round_idx):
        self.model.load_state_dict(global_parameters, strict=False)
        (
            update,
            metrics,
            new_best_acc,
            prototype,
            soft_labels,
            semantic_stats,
        ) = FedAvg_client.recon_train(
            model=self.model,
            data_loader=self.data_loader,
            config=self.config,
            client_id=self.client_id,
            dataset_name=self.config['dataset_name'],
            round_idx=round_idx,
            best_acc=self.best_acc,
        )
        self.best_acc = new_best_acc
        return update, metrics, prototype, soft_labels, semantic_stats

    def consensus_training(self, prototype, transport, global_consensus):
        return FedAvg_client.consensus_train(
            model=self.model,
            data_loader=self.data_loader,
            config=self.config,
            prototype=prototype,
            transport=transport,
            global_consensus=global_consensus,
        )
