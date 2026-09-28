from warnings import simplefilter
import numpy as np
import torch
from torch_clustering import PyTorchKMeans
simplefilter(action='ignore', category=FutureWarning)
from cal_metric import full_metric

def my_clustering_gpu(feature, true_labels, cluster_num, random_state=0, fit_mask=None):
    kmeans = PyTorchKMeans(
        metric='euclidean', init='k-means++', n_clusters=cluster_num,
        n_init=10, random_state=random_state, verbose=False
    )
    if isinstance(feature, torch.Tensor):
        feature = feature.detach().to(dtype=torch.float32)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        feature = torch.as_tensor(feature, dtype=torch.float32, device=device)
    if fit_mask is None:
        fit_mask = torch.ones(feature.shape[0], dtype=torch.bool, device=feature.device)
    else:
        fit_mask = fit_mask.to(device=feature.device, dtype=torch.bool)
    if int(fit_mask.sum()) < cluster_num:
        raise ValueError('Observed samples must be at least the number of clusters.')
    kmeans.fit_predict(feature[fit_mask])
    center = kmeans.cluster_centers_
    predict_labels, _ = kmeans.predict(feature, center)
    mask_np = fit_mask.detach().cpu().numpy()
    predict_np = predict_labels.detach().cpu().numpy()
    OA, AA, KAPPA, NMI, ARI, F1, PRECISION, RECALL, PURITY = full_metric(
        np.asarray(true_labels)[mask_np], predict_np[mask_np], is_refined=False
    )
    return 100*OA, 100*AA, 100*KAPPA, 100*NMI, 100*ARI, 100*F1, 100*PRECISION, 100*RECALL, 100*PURITY, predict_np, center
