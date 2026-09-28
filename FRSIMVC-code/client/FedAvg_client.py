from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils.csata import align_soft_labels, compute_semantic_stats
from utils.evaluation import my_clustering_gpu
from torch_clustering import PyTorchKMeans


def _shared_state_dict(model):
    parameters = model.state_dict().copy()
    parameters.pop('encoder.gnn_1.weight', None)
    parameters.pop('decoder.gnn_6.weight', None)
    return parameters


def snapshot_forward(model, x, adj):
    """Read-only forward: H = encoder(x, adj) under ``inference_mode``.

    Outline v9 section 7: the adapter must never write back.  The caller is
    responsible for restoring ``model.train()/eval()``; this helper neither
    changes ``requires_grad`` nor touches the optimiser.
    """
    with torch.inference_mode():
        embedding, _, _ = model(x, adj)
    return embedding.detach()


def fit_snapshot_prototypes(embedding, observed_mask, n_clusters, random_state=0):
    """KMeans prototypes fit on *observed* nodes only, with a private RNG.

    Uses the same metric/init/n_init convention as ``my_clustering_gpu`` but
    never looks at the ground truth.  ``PyTorchKMeans`` seeds a local
    ``torch.Generator``, so the global RNG is untouched.

    Returns ``(prototypes (K, d), info)``; ``info['status']`` is ``'ok'`` or
    ``'insufficient_observations'`` (in which case the prototypes are zeros and
    the caller must fall back).
    """
    device = embedding.device
    fit_mask = observed_mask.to(device=device, dtype=torch.bool)
    n_observed = int(fit_mask.sum().detach().cpu())
    if n_observed < n_clusters:
        return (
            torch.zeros((n_clusters, embedding.shape[1]), device=device,
                        dtype=embedding.dtype),
            {'status': 'insufficient_observations', 'n_observed': n_observed},
        )
    kmeans = PyTorchKMeans(
        metric='euclidean',
        init='k-means++',
        n_clusters=n_clusters,
        n_init=10,
        random_state=int(random_state),
        verbose=False,
    )
    kmeans.fit_predict(embedding[fit_mask])
    centers = kmeans.cluster_centers_.detach()
    return centers, {'status': 'ok', 'n_observed': n_observed}


def _masked_reconstruction_loss(reconstruction, target, observed_mask, mask_aware=True):
    if mask_aware:
        if not torch.any(observed_mask):
            raise RuntimeError('A client has no observed samples after masking.')
        return F.mse_loss(reconstruction[observed_mask], target[observed_mask])
    # A0: V0 treats the zero-filled rows as ordinary input (no mask awareness).
    return F.mse_loss(reconstruction, target)


def recon_train(model, data_loader, config, client_id, dataset_name, round_idx, best_acc):
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('lr', 1e-3),
        weight_decay=config.get('weight_decay', 1e-5),
    )
    x = data_loader['x']
    target_x = data_loader['target_x']
    adj = data_loader['adj']
    observed_mask = data_loader['mask']

    model.train()
    reconstruction_loss = None
    for _ in range(config.get('local_epochs', 1)):
        optimizer.zero_grad()
        _, x_hat, _ = model(x, adj)
        reconstruction_loss = _masked_reconstruction_loss(
            x_hat, target_x, observed_mask, mask_aware=config.get('mask_aware_loss', True)
        )
        reconstruction_loss.backward()
        optimizer.step()

    metrics, prototype, soft_labels, semantic_stats, new_best_acc = model_eval(
        model=model,
        x=x,
        adj=adj,
        y=data_loader['y'],
        observed_mask=observed_mask,
        best_acc=best_acc,
        n_classes=data_loader['n_classes'],
        client_id=client_id,
        round_idx=round_idx,
        dataset_name=dataset_name,
        config=config,
    )
    metrics['reconstruction_loss'] = float(reconstruction_loss.detach().cpu())
    return (
        _shared_state_dict(model),
        metrics,
        new_best_acc,
        prototype,
        soft_labels,
        semantic_stats,
    )


def model_eval(
    model,
    x,
    adj,
    y,
    observed_mask,
    best_acc,
    n_classes,
    client_id,
    round_idx,
    dataset_name,
    config,
):
    model.eval()
    with torch.no_grad():
        embedding, _, _ = model(x, adj)
        labels_np = y.detach().cpu().numpy()
        # A0 (naive) fits the prototypes on all nodes, zero-filled rows included.
        fit_mask = observed_mask if config.get('mask_aware_eval', True) else None
        metrics_tuple = my_clustering_gpu(
            embedding,
            labels_np,
            n_classes,
            random_state=config.get('seed', 42),
            fit_mask=fit_mask,
        )
        acc = metrics_tuple[0]
        prototype = metrics_tuple[10].detach()
        distances = torch.cdist(embedding, prototype, p=2)
        soft_labels = F.softmax(-distances, dim=1)
        # BSSAT module 2: K x 3 privacy-safe cluster statistics computed locally.
        # [normalised mass, assignment entropy, compactness] -- only this
        # aggregate is ever handed to the server.
        semantic_stats = compute_semantic_stats(soft_labels, embedding, observed_mask)
        metrics = {
            'ACC': acc,
            'AA': metrics_tuple[1],
            'Kappa': metrics_tuple[2],
            'NMI': metrics_tuple[3],
            'ARI': metrics_tuple[4],
            'F1': metrics_tuple[5],
            'Precision': metrics_tuple[6],
            'Recall': metrics_tuple[7],
            'PURITY': metrics_tuple[8],
        }

        new_best_acc = best_acc
        if acc > best_acc:
            new_best_acc = acc
            if config.get('save_model', True):
                run_dir = Path(config.get('run_dir', config.get('save_path', 'save')))
                save_dir = run_dir / 'checkpoints' / f'client_{client_id}'
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_dir / 'best_model_params.pth')
                np.save(save_dir / 'best_features.npy', embedding.detach().cpu().numpy())
                np.save(save_dir / 'labels.npy', labels_np)
        return metrics, prototype, soft_labels, semantic_stats, new_best_acc


def consensus_train(model, data_loader, config, prototype, transport, global_consensus):
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get('lr', 1e-3),
        weight_decay=config.get('weight_decay', 1e-5),
    )
    x = data_loader['x']
    target_x = data_loader['target_x']
    adj = data_loader['adj']
    observed_mask = data_loader['mask']
    prototype = prototype.detach()
    transport = transport.detach()
    has_consensus = global_consensus is not None
    target_consensus = global_consensus.detach() if has_consensus else None
    mask_aware = bool(config.get('mask_aware_loss', True))
    lambda_rec = float(config.get('lambda_reconstruction', 1.0))
    lambda_consensus = float(config.get('lambda_consensus', 1.0))

    model.train()
    total_loss = None
    consensus_loss = torch.zeros((), device=x.device)
    for _ in range(config.get('local_epochs', 1)):
        optimizer.zero_grad()
        embedding, x_hat, _ = model(x, adj)
        soft_labels = F.softmax(-torch.cdist(embedding, prototype, p=2), dim=1)
        aligned = align_soft_labels(soft_labels, transport)
        reconstruction_loss = _masked_reconstruction_loss(
            x_hat, target_x, observed_mask, mask_aware=mask_aware
        )
        if has_consensus:
            aligned_observed = aligned[observed_mask].clamp_min(1e-8)
            consensus_observed = target_consensus[observed_mask].clamp_min(1e-8)
            consensus_loss = torch.sum(
                aligned_observed
                * (torch.log(aligned_observed) - torch.log(consensus_observed)),
                dim=1,
            ).mean()
            total_loss = lambda_rec * reconstruction_loss + lambda_consensus * consensus_loss
        else:
            # A0-A3: no global consensus term, reconstruction only.
            total_loss = lambda_rec * reconstruction_loss
        total_loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        embedding, _, _ = model(x, adj)
        soft_labels = F.softmax(-torch.cdist(embedding, prototype, p=2), dim=1)
        aligned = align_soft_labels(soft_labels, transport)
    return (
        _shared_state_dict(model),
        aligned.detach(),
        float(consensus_loss.detach().cpu()),
        float(total_loss.detach().cpu()),
    )
