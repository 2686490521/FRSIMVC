import argparse
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import torch

from Flsimulator import FLSimulator
from utils.superpixel_utils import EDGE_RELIABILITY_MODES, RADG_MODES


SCRIPT_DIR = Path(__file__).resolve().parent


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, 'isatty', lambda: False)() for stream in self.streams)


def set_seed(seed=42, deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if deterministic:
        # outline v6 section 10: squeeze the run-to-run variance that the
        # BSSAT-v2 cross-batch check exposed (identical node set, per-seed
        # |delta ACC| up to 0.045).  ``warn_only=True`` keeps CUDA kernels that
        # have no deterministic implementation (sparse/scatter/Sinkhorn) alive
        # but reports them, instead of aborting an 80-run sweep.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True, warn_only=True)


def load_config(path):
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def save_json(path, payload):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def resolve_data_path(value, config_dir, data_root=None):
    path = Path(os.path.expandvars(str(value))).expanduser()
    candidates = [path] if path.is_absolute() else [config_dir / path]
    if data_root and not path.is_absolute():
        candidates.insert(0, Path(data_root).expanduser() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f'Dataset file not found: {value}; checked {candidates}')


def validate_dataset_config(config, dataset_name, config_dir):
    if dataset_name not in config or not isinstance(config[dataset_name], dict):
        raise KeyError(f'Dataset {dataset_name!r} is not defined in the config file.')
    dataset_config = config[dataset_name]
    required = ['mm_data_path', 'gt_path', 'n_neighbors', 'n_superpixels', 'patch_size']
    missing = [key for key in required if key not in dataset_config]
    if missing:
        raise KeyError(f'{dataset_name} is missing config fields: {missing}')
    paths = [
        resolve_data_path(path, config_dir, config.get('data_root'))
        for path in dataset_config['mm_data_path']
    ]
    paths.append(resolve_data_path(dataset_config['gt_path'], config_dir, config.get('data_root')))
    return paths


def select_device(gpu_id):
    if str(gpu_id).lower() == 'cpu':
        return torch.device('cpu')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. Use --gpu cpu only if a CPU run is intentional.')
    gpu_index = int(gpu_id)
    if gpu_index < 0 or gpu_index >= torch.cuda.device_count():
        raise ValueError(
            f'GPU {gpu_index} is invalid; available indices: 0..{torch.cuda.device_count() - 1}'
        )
    torch.cuda.set_device(gpu_index)
    return torch.device(f'cuda:{gpu_index}')


def build_parser():
    parser = argparse.ArgumentParser(description='Run one FedRSMVC dataset experiment.')
    parser.add_argument('--config', type=Path, default=SCRIPT_DIR / 'config.json')
    parser.add_argument('--dataset', help='Dataset key in config.json.')
    parser.add_argument('--gpu', help='CUDA index (for example 0), or cpu.')
    parser.add_argument('--epochs', type=int, help='Override n_epoches for this run.')
    parser.add_argument('--seed', type=int, help='Override random seed for this run.')
    parser.add_argument('--missing-rate', type=float, help='Per-view missing rate in [0, 1).')
    parser.add_argument(
        '--missing-mode',
        choices=['random', 'block', 'strip'],
        help='Pixel-level missingness protocol applied BEFORE SLIC.',
    )
    parser.add_argument(
        '--missing-seed',
        type=int,
        help='Seed of the missingness simulator; defaults to --seed.',
    )
    parser.add_argument(
        '--superpixel-threshold',
        type=float,
        help='Superpixel observed-ratio threshold tau (node missing when r_i < tau).',
    )
    parser.add_argument(
        '--ablation-variant',
        help='Ablation rung A0|A1|A2|A3|A4 (A0=naive zero-fill .. A4=full V5).',
    )
    parser.add_argument(
        '--graph-beta',
        type=float,
        help='Spatial-graph fusion weight beta in [0, 1] (0=local feature graph, 1=spatial graph).',
    )
    parser.add_argument(
        '--sinkhorn-epsilon',
        type=float,
        help='Entropic regularisation of the BSSAT transport (outline v4 module 2).',
    )
    parser.add_argument(
        '--bssat-dual',
        dest='bssat_dual',
        action='store_true',
        default=None,
        help='BSSAT: use the spatial + semantic dual signature.',
    )
    parser.add_argument(
        '--no-bssat-dual',
        dest='bssat_dual',
        action='store_false',
        help='Fall back to the v3 spatial-only CSATA signature.',
    )
    parser.add_argument(
        '--bssat-balanced',
        dest='bssat_balanced',
        action='store_true',
        default=None,
        help='BSSAT: use the mass-preserving (balanced) transport.',
    )
    parser.add_argument(
        '--no-bssat-balanced',
        dest='bssat_balanced',
        action='store_false',
        help='Fall back to the v3 row-stochastic Sinkhorn transport.',
    )
    parser.add_argument(
        '--lambda-spatial',
        type=float,
        help='Cost weight of the spatial signature term in BSSAT.',
    )
    parser.add_argument(
        '--lambda-semantic',
        type=float,
        help='Cost weight of the semantic signature term in BSSAT.',
    )
    parser.add_argument(
        '--mass-uniform-alpha',
        type=float,
        help="BSSAT-v2 (outline v5): temper the balanced column marginal with "
             "b' = (1 - alpha) b + alpha / K. 0.0 reproduces BSSAT-r1.",
    )
    parser.add_argument(
        '--reliability-adaptive-graph',
        dest='reliability_adaptive_graph',
        action='store_true',
        default=None,
        help='outline v6 module I: edge-wise adaptive fusion '
             'A_ij = beta_ij A_s,ij + (1 - beta_ij) A_f,ij.',
    )
    parser.add_argument(
        '--no-reliability-adaptive-graph',
        dest='reliability_adaptive_graph',
        action='store_false',
        help='Keep the fixed fusion A = graph_beta A_s + (1 - graph_beta) A_f '
             '(the frozen BSSAT-r1 control).',
    )
    parser.add_argument(
        '--radg-mode',
        choices=['centered', 'uncentered'],
        help='RADG: "centered" keeps E[beta] at radg_base_beta (outline v7 '
             'RADG-v1.1); "uncentered" is the v6 RADG-v1 kept as an ablation.',
    )
    parser.add_argument(
        '--radg-base-beta',
        type=float,
        help='RADG centered: the fixed baseline beta_0 that E[beta_ij] must '
             'reproduce (default 0.5).',
    )
    parser.add_argument(
        '--radg-gamma',
        type=float,
        help='RADG centered: modulation strength gamma in '
             'beta = beta_0 + gamma (u_ij - mean(u)).',
    )
    parser.add_argument(
        '--radg-beta-floor',
        type=float,
        help='RADG centered: lower clip for beta_ij (default 0.2).',
    )
    parser.add_argument(
        '--radg-beta-ceiling',
        type=float,
        help='RADG centered: upper clip for beta_ij (default 0.8).',
    )
    parser.add_argument(
        '--radg-center-tolerance',
        type=float,
        help='RADG centered: tolerated |E[beta] - beta_0| (default 0.005); '
             'a larger value is reported as a warning.',
    )
    parser.add_argument(
        '--radg-beta-min',
        type=float,
        help='RADG: beta at r_i = 1, i.e. weight of the spatial graph on fully '
             'observed edges.',
    )
    parser.add_argument(
        '--radg-beta-max',
        type=float,
        help='RADG: beta at r_i = 0, i.e. weight of the spatial graph on fully '
             'missing edges.',
    )
    parser.add_argument(
        '--radg-edge-reliability',
        choices=['geometric_mean', 'arithmetic_mean'],
        help='RADG: how r_i and r_j are combined into c_ij '
             '(default geometric_mean = sqrt(r_i r_j)).',
    )
    parser.add_argument(
        '--deterministic-mode',
        dest='deterministic_mode',
        action='store_true',
        default=None,
        help='outline v6 section 10: tighten cuBLAS/cuDNN determinism.',
    )
    parser.add_argument(
        '--no-deterministic-mode',
        dest='deterministic_mode',
        action='store_false',
        help='Leave the default (non-deterministic) CUDA kernels enabled.',
    )
    parser.add_argument(
        '--drop-nonpositive-labels',
        action='store_true',
        help='Treat labels <= 0 as background (MUUFL gt uses -1 for background).',
    )
    parser.add_argument('--output-root', type=Path, help='Root directory for experiment results.')
    parser.add_argument('--run-id', help='Shared run id; default is a local timestamp.')
    parser.add_argument('--validate-only', action='store_true', help='Check data paths and CUDA, then exit.')
    # ---- outline v9, module III: Cross-Client Semantic-Spatial Completion --
    parser.add_argument(
        '--completion-enabled',
        dest='completion_enabled',
        action='store_true',
        default=None,
        help='Attach the read-only completion adapter and evaluate the eight arms.',
    )
    parser.add_argument(
        '--no-completion',
        dest='completion_enabled',
        action='store_false',
        help='Keep the frozen B0 path only (default).',
    )
    parser.add_argument(
        '--completion-stage',
        choices=['inference_only', 'off'],
        help='v9 is inference-only; "off" keeps the adapter disabled.',
    )
    parser.add_argument(
        '--completion-lambdas',
        type=float,
        nargs='+',
        help='Completion interpolation strengths, e.g. 0.25 0.5 0.75 1.0.',
    )
    parser.add_argument(
        '--completion-arms',
        nargs='+',
        help='Arms to record, e.g. B0 E0 G0 D0 C025 C050 C075 C100.',
    )
    parser.add_argument(
        '--adapter-seed',
        type=int,
        help='Seed of the read-only evaluation adapter (private RNG).',
    )
    parser.add_argument(
        '--diagnostic-seed',
        type=int,
        help='Seed of the pseudo-missing probe sampling.',
    )
    parser.add_argument(
        '--diagnostic-epochs',
        type=int,
        nargs='+',
        help='Rounds at which the pseudo-missing probe runs (default 100 200 300).',
    )
    parser.add_argument(
        '--completion-min-valid-mass',
        type=float,
        help='Minimum donor mass on valid OT columns for a row to be eligible.',
    )
    parser.add_argument(
        '--no-frozen-baseline-check',
        dest='frozen_baseline_check',
        action='store_false',
        default=None,
        help='Skip the A3 / RADG-off / beta / epsilon baseline assertions.',
    )
    return parser


def main():
    args = build_parser().parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    dataset_name = args.dataset or config.get('dataset_name')
    if not dataset_name:
        raise ValueError('No dataset selected. Use --dataset or set dataset_name in config.json.')

    config['dataset_name'] = dataset_name
    config['config_dir'] = str(config_path.parent)
    if args.epochs is not None:
        config['n_epoches'] = args.epochs
    if args.seed is not None:
        config['seed'] = args.seed
    if args.gpu is not None:
        config['gpu_id'] = args.gpu
    if args.missing_rate is not None:
        config['missing_rate'] = args.missing_rate
    if args.missing_mode is not None:
        config['missing_mode'] = args.missing_mode
    if args.missing_seed is not None:
        config['missing_seed'] = args.missing_seed
    if args.superpixel_threshold is not None:
        threshold = float(args.superpixel_threshold)
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f'superpixel_observed_threshold must be in (0, 1], got {threshold}')
        config['superpixel_observed_threshold'] = threshold
    if args.ablation_variant is not None:
        config['ablation_variant'] = args.ablation_variant
    if args.graph_beta is not None:
        graph_beta = float(args.graph_beta)
        if not 0.0 <= graph_beta <= 1.0:
            raise ValueError(f'graph_beta must be in [0, 1], got {graph_beta}')
        config['graph_beta'] = graph_beta
    if args.sinkhorn_epsilon is not None:
        epsilon = float(args.sinkhorn_epsilon)
        if epsilon <= 0.0:
            raise ValueError(f'sinkhorn_epsilon must be positive, got {epsilon}')
        config['sinkhorn_epsilon'] = epsilon
    if args.bssat_dual is not None:
        config['csata_dual_signature'] = bool(args.bssat_dual)
    if args.bssat_balanced is not None:
        config['csata_balanced'] = bool(args.bssat_balanced)
    if args.lambda_spatial is not None:
        config['lambda_spatial'] = float(args.lambda_spatial)
    if args.lambda_semantic is not None:
        config['lambda_semantic'] = float(args.lambda_semantic)
    if args.mass_uniform_alpha is not None:
        config['mass_uniform_alpha'] = float(args.mass_uniform_alpha)
    for key in ('lambda_spatial', 'lambda_semantic'):
        value = float(config.get(key, 1.0))
        if value < 0.0:
            raise ValueError(f'{key} must be non-negative, got {value}')
        config[key] = value
    mass_uniform_alpha = float(config.get('mass_uniform_alpha', 0.0))
    if not 0.0 <= mass_uniform_alpha <= 1.0:
        raise ValueError(
            f'mass_uniform_alpha must be in [0, 1], got {mass_uniform_alpha}'
        )
    config['mass_uniform_alpha'] = mass_uniform_alpha
    # ---- outline v6, module I: Reliability-Adaptive Dual-Graph (RADG-v1) ----
    if args.reliability_adaptive_graph is not None:
        config['reliability_adaptive_graph'] = bool(args.reliability_adaptive_graph)
    if args.radg_beta_min is not None:
        config['radg_beta_min'] = float(args.radg_beta_min)
    if args.radg_beta_max is not None:
        config['radg_beta_max'] = float(args.radg_beta_max)
    if args.radg_mode is not None:
        config['radg_mode'] = str(args.radg_mode)
    if args.radg_base_beta is not None:
        config['radg_base_beta'] = float(args.radg_base_beta)
    if args.radg_gamma is not None:
        config['radg_gamma'] = float(args.radg_gamma)
    if args.radg_beta_floor is not None:
        config['radg_beta_floor'] = float(args.radg_beta_floor)
    if args.radg_beta_ceiling is not None:
        config['radg_beta_ceiling'] = float(args.radg_beta_ceiling)
    if args.radg_center_tolerance is not None:
        config['radg_center_tolerance'] = float(args.radg_center_tolerance)
    if args.radg_edge_reliability is not None:
        config['radg_edge_reliability'] = str(args.radg_edge_reliability)
    if args.deterministic_mode is not None:
        config['deterministic_mode'] = bool(args.deterministic_mode)
    # ---- outline v7, module I v1.1: centered RADG -------------------------
    radg_mode = str(config.get('radg_mode', 'centered')).lower()
    if radg_mode not in RADG_MODES:
        raise ValueError(f'radg_mode must be one of {RADG_MODES}, got {radg_mode!r}')
    radg_base_beta = float(config.get('radg_base_beta', 0.5))
    if not 0.0 <= radg_base_beta <= 1.0:
        raise ValueError(f'radg_base_beta must be in [0, 1], got {radg_base_beta}')
    radg_gamma = float(config.get('radg_gamma', 0.5))
    if radg_gamma < 0.0:
        raise ValueError(f'radg_gamma must be non-negative, got {radg_gamma}')
    radg_floor = float(config.get('radg_beta_floor', 0.2))
    radg_ceiling = float(config.get('radg_beta_ceiling', 0.8))
    if not 0.0 <= radg_floor <= radg_ceiling <= 1.0:
        raise ValueError(
            f'expected 0 <= radg_beta_floor <= radg_beta_ceiling <= 1, '
            f'got {radg_floor} / {radg_ceiling}'
        )
    radg_tolerance = float(config.get('radg_center_tolerance', 0.005))
    if radg_tolerance <= 0.0:
        raise ValueError(
            f'radg_center_tolerance must be positive, got {radg_tolerance}'
        )
    config['radg_mode'] = radg_mode
    config['radg_base_beta'] = radg_base_beta
    config['radg_gamma'] = radg_gamma
    config['radg_beta_floor'] = radg_floor
    config['radg_beta_ceiling'] = radg_ceiling
    config['radg_center_tolerance'] = radg_tolerance
    radg_beta_min = float(config.get('radg_beta_min', 0.2))
    radg_beta_max = float(config.get('radg_beta_max', 0.8))
    if not 0.0 <= radg_beta_min <= 1.0:
        raise ValueError(f'radg_beta_min must be in [0, 1], got {radg_beta_min}')
    if not 0.0 <= radg_beta_max <= 1.0:
        raise ValueError(f'radg_beta_max must be in [0, 1], got {radg_beta_max}')
    if radg_beta_min > radg_beta_max:
        raise ValueError(
            f'radg_beta_min ({radg_beta_min}) must not exceed '
            f'radg_beta_max ({radg_beta_max})'
        )
    edge_reliability = str(config.get('radg_edge_reliability', 'geometric_mean'))
    if edge_reliability not in EDGE_RELIABILITY_MODES:
        raise ValueError(
            f'radg_edge_reliability must be one of {EDGE_RELIABILITY_MODES}, '
            f'got {edge_reliability!r}'
        )
    config['radg_beta_min'] = radg_beta_min
    config['radg_beta_max'] = radg_beta_max
    config['radg_edge_reliability'] = edge_reliability
    config['reliability_adaptive_graph'] = bool(
        config.get('reliability_adaptive_graph', False)
    )
    config['deterministic_mode'] = bool(config.get('deterministic_mode', False))
    if config['reliability_adaptive_graph']:
        # RADG reweights the *fusion* of the consensus spatial graph with the
        # local feature graph, so it is meaningless without both of them.
        if str(config.get('ablation_variant') or '').upper() in ('A0', 'A1'):
            raise ValueError(
                'reliability_adaptive_graph requires the consensus spatial '
                f'graph, but ablation_variant={config["ablation_variant"]} has '
                'spatial_graph off. Use A2+ (A3 is the frozen BSSAT-r1 arm).'
            )
        if not config.get('spatial_graph', True):
            raise ValueError(
                'reliability_adaptive_graph requires spatial_graph=true.'
            )
    # ---- outline v9, module III: completion adapter -----------------------
    if args.completion_enabled is not None:
        config['completion_enabled'] = bool(args.completion_enabled)
    if args.completion_stage is not None:
        config['completion_stage'] = str(args.completion_stage)
    if args.completion_lambdas is not None:
        config['completion_lambdas'] = [float(x) for x in args.completion_lambdas]
    if args.completion_arms is not None:
        config['completion_arms'] = list(args.completion_arms)
    if args.adapter_seed is not None:
        config['adapter_seed'] = int(args.adapter_seed)
    if args.diagnostic_seed is not None:
        config['diagnostic_seed'] = int(args.diagnostic_seed)
    if args.diagnostic_epochs is not None:
        config['diagnostic_epochs'] = [int(x) for x in args.diagnostic_epochs]
    if args.completion_min_valid_mass is not None:
        config['completion_min_valid_mass'] = float(args.completion_min_valid_mass)
    config['completion_enabled'] = bool(config.get('completion_enabled', False))
    config['completion_stage'] = str(config.get('completion_stage', 'inference_only'))
    if config['completion_stage'] not in ('inference_only', 'off'):
        raise ValueError(
            f"completion_stage must be inference_only|off, got {config['completion_stage']!r}"
        )
    if config['completion_stage'] == 'off':
        config['completion_enabled'] = False
    completion_lambdas = [float(x) for x in config.get(
        'completion_lambdas', [0.25, 0.5, 0.75, 1.0]
    )]
    if not completion_lambdas or any(not 0.0 <= x <= 1.0 for x in completion_lambdas):
        raise ValueError(f'completion_lambdas must lie in [0, 1], got {completion_lambdas}')
    config['completion_lambdas'] = completion_lambdas
    valid_arms = {'B0', 'E0', 'G0', 'D0', 'C025', 'C050', 'C075', 'C100'}
    completion_arms = [str(x) for x in config.get('completion_arms', sorted(valid_arms))]
    unknown = [arm for arm in completion_arms if arm not in valid_arms]
    if unknown:
        raise ValueError(f'unknown completion arms {unknown}; valid: {sorted(valid_arms)}')
    config['completion_arms'] = completion_arms
    config['adapter_seed'] = int(config.get('adapter_seed', 20260922))
    config['diagnostic_seed'] = int(config.get('diagnostic_seed', 100000))
    config['diagnostic_epochs'] = [int(x) for x in config.get(
        'diagnostic_epochs', [100, 200, 300]
    )]
    config['completion_min_valid_mass'] = float(config.get('completion_min_valid_mass', 0.5))
    config['numerical_eps'] = float(config.get('numerical_eps', 1e-12))
    if config['completion_enabled']:
        # The frozen-baseline assertions (outline v9 section 5.2 / 11.2) make it
        # impossible to run the new branch on a silently different baseline.
        from utils.ablation import resolve_ablation
        variant, flags = resolve_ablation(config)
        problems = []
        if variant != 'A3':
            problems.append(f'ablation_variant={variant!r} (expected A3)')
        if flags['global_consensus']:
            problems.append('global_consensus=True (A3 requires False)')
        if not flags['csata_alignment']:
            problems.append('csata_alignment=False (A3 requires True)')
        if config['reliability_adaptive_graph']:
            problems.append('reliability_adaptive_graph=True (RADG must stay off)')
        if abs(float(config.get('graph_beta', 0.5)) - 0.5) > 1e-9:
            problems.append(f"graph_beta={config.get('graph_beta')} (expected 0.5)")
        if abs(float(config.get('sinkhorn_epsilon', 0.05)) - 0.1) > 1e-9:
            problems.append(f"sinkhorn_epsilon={config.get('sinkhorn_epsilon')} (expected 0.1)")
        if abs(float(config.get('mass_uniform_alpha', 0.0))) > 1e-9:
            problems.append(f"mass_uniform_alpha={config.get('mass_uniform_alpha')} (expected 0)")
        if problems:
            raise ValueError(
                'completion_enabled requires the frozen F05+BSSAT-r1 baseline; '
                'violations: ' + '; '.join(problems)
            )
    if args.drop_nonpositive_labels:
        config['drop_nonpositive_labels'] = True
    missing_mode = str(config.get('missing_mode', 'random')).lower()
    config['missing_mode'] = missing_mode
    if missing_mode not in ('random', 'block', 'strip'):
        raise ValueError(f'missing_mode must be random|block|strip, got {missing_mode!r}')
    if config.get('missing_seed') is None:
        config['missing_seed'] = int(config.get('seed', 42))
    missing_rate = float(config.get('missing_rate', 0.3))
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError(f'missing_rate must be in [0, 1), got {missing_rate}')
    superpixel_threshold = float(config.get('superpixel_observed_threshold', 0.5))
    if not 0.0 < superpixel_threshold <= 1.0:
        raise ValueError(
            f'superpixel_observed_threshold must be in (0, 1], got {superpixel_threshold}'
        )

    resolved_paths = validate_dataset_config(config, dataset_name, config_path.parent)
    device = select_device(config.get('gpu_id', '0'))
    if args.validate_only:
        ablation = config.get('ablation_variant') or 'FULL(V5)'
        print(
            f'Validation OK: dataset={dataset_name}, device={device}, '
            f'missing_mode={missing_mode}, missing_rate={missing_rate}, '
            f'missing_seed={config["missing_seed"]}, '
            f'tau={superpixel_threshold}, ablation={ablation}, '
            f'graph_beta={config.get("graph_beta", 0.5)}, '
            f'sinkhorn_epsilon={config.get("sinkhorn_epsilon", 0.05)}, '
            f'bssat_dual={config.get("csata_dual_signature", True)}, '
            f'bssat_balanced={config.get("csata_balanced", True)}, '
            f'mass_uniform_alpha={config.get("mass_uniform_alpha", 0.0)}, '
            f'reliability_adaptive_graph={config.get("reliability_adaptive_graph", False)}, '
            f'radg_beta=({config.get("radg_beta_min", 0.2)}, {config.get("radg_beta_max", 0.8)}), '
            f'radg_edge_reliability={config.get("radg_edge_reliability", "geometric_mean")}, '
            f'deterministic_mode={config.get("deterministic_mode", False)}, '
            f'drop_nonpositive={config.get("drop_nonpositive_labels", False)}, '
            f'completion_enabled={config.get("completion_enabled", False)}, '
            f'completion_stage={config.get("completion_stage", "inference_only")}, '
            f'completion_lambdas={config.get("completion_lambdas", [])}, '
            f'completion_arms={config.get("completion_arms", [])}, '
            f'adapter_seed={config.get("adapter_seed", 20260922)}, '
            f'diagnostic_epochs={config.get("diagnostic_epochs", [])}'
        )
        for path in resolved_paths:
            print(f'  {path}')
        if device.type == 'cuda':
            print(f'  GPU: {torch.cuda.get_device_name(device)}')
        return 0

    output_root = args.output_root or Path(config.get('result_path', SCRIPT_DIR / 'results'))
    if not output_root.is_absolute():
        output_root = (SCRIPT_DIR / output_root).resolve()
    run_id = args.run_id or datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    run_dir = output_root / dataset_name / run_id
    if run_dir.exists():
        suffix = 1
        while (output_root / dataset_name / f'{run_id}_{suffix:02d}').exists():
            suffix += 1
        run_dir = output_root / dataset_name / f'{run_id}_{suffix:02d}'
    run_dir.mkdir(parents=True, exist_ok=False)
    config['run_dir'] = str(run_dir)
    config['gpu_id'] = str(config.get('gpu_id', '0'))
    save_json(run_dir / 'config.json', config)

    started_at = datetime.now().astimezone()
    start_clock = time.perf_counter()
    metadata = {
        'status': 'running',
        'dataset': dataset_name,
        'start_time': started_at.isoformat(),
        'device': str(device),
        'python': sys.version,
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
        'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'command': sys.argv,
        'ablation_variant': config.get('ablation_variant'),
        'graph_beta': float(config.get('graph_beta', 0.5)),
        'sinkhorn_epsilon': float(config.get('sinkhorn_epsilon', 0.05)),
        'csata_dual_signature': bool(config.get('csata_dual_signature', True)),
        'csata_balanced': bool(config.get('csata_balanced', True)),
        'lambda_spatial': float(config.get('lambda_spatial', 1.0)),
        'lambda_semantic': float(config.get('lambda_semantic', 1.0)),
        'mass_uniform_alpha': float(config.get('mass_uniform_alpha', 0.0)),
        # outline v6, module I
        'reliability_adaptive_graph': bool(
            config.get('reliability_adaptive_graph', False)
        ),
        'radg_beta_min': float(config.get('radg_beta_min', 0.2)),
        'radg_beta_max': float(config.get('radg_beta_max', 0.8)),
        'radg_edge_reliability': str(
            config.get('radg_edge_reliability', 'geometric_mean')
        ),
        'deterministic_mode': bool(config.get('deterministic_mode', False)),
        'drop_nonpositive_labels': bool(config.get('drop_nonpositive_labels', False)),
        'missing_mode': missing_mode,
        'missing_rate': missing_rate,
        'missing_seed': int(config['missing_seed']),
        'superpixel_observed_threshold': superpixel_threshold,
        'mask_stage': 'pixel_level_before_slic',
        'seed': int(config.get('seed', 42)),
        # outline v9, module III
        'completion_enabled': bool(config.get('completion_enabled', False)),
        'completion_stage': str(config.get('completion_stage', 'inference_only')),
        'completion_lambdas': list(config.get('completion_lambdas', [])),
        'completion_arms': list(config.get('completion_arms', [])),
        'adapter_seed': int(config.get('adapter_seed', 20260922)),
        'diagnostic_seed': int(config.get('diagnostic_seed', 100000)),
        'diagnostic_epochs': list(config.get('diagnostic_epochs', [])),
        'completion_min_valid_mass': float(config.get('completion_min_valid_mass', 0.5)),
        'resolved_data_paths': [str(path) for path in resolved_paths],
        'result_directory': str(run_dir),
    }
    save_json(run_dir / 'run_metadata.json', metadata)

    set_seed(
        int(config.get('seed', 42)),
        deterministic=bool(config.get('deterministic_mode', False)),
    )
    log_path = run_dir / 'console.log'
    try:
        with open(log_path, 'w', encoding='utf-8', buffering=1) as log_file:
            original_stdout, original_stderr = sys.stdout, sys.stderr
            sys.stdout = Tee(original_stdout, log_file)
            sys.stderr = Tee(original_stderr, log_file)
            try:
                print(f'Starting {dataset_name} on {device}')
                print(f'Results will be saved to {run_dir}')
                simulator = FLSimulator(config, device=device, output_dir=run_dir)
                summary = simulator.start()
            finally:
                sys.stdout, sys.stderr = original_stdout, original_stderr

        metadata.update({
            'status': 'completed',
            'end_time': datetime.now().astimezone().isoformat(),
            'runtime_seconds': float(time.perf_counter() - start_clock),
            'metrics': summary['metrics'],
            'best_round': summary['best_round'],
        })
        save_json(run_dir / 'run_metadata.json', metadata)
        return 0
    except Exception as error:
        metadata.update({
            'status': 'failed',
            'end_time': datetime.now().astimezone().isoformat(),
            'runtime_seconds': float(time.perf_counter() - start_clock),
            'error': repr(error),
            'traceback': traceback.format_exc(),
        })
        save_json(run_dir / 'run_metadata.json', metadata)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
