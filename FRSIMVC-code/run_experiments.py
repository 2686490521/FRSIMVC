import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def load_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def save_json(path, payload):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Run one or more FedRSMVC datasets sequentially on one GPU.'
    )
    parser.add_argument('--config', type=Path, default=SCRIPT_DIR / 'config.json')
    parser.add_argument('--datasets', nargs='+', help='Examples: Trento MUUFL MDAS')
    parser.add_argument('--gpu', default=None, help='CUDA index; defaults to gpu_id in config.')
    parser.add_argument('--epochs', type=int, help='Override training rounds for every dataset.')
    parser.add_argument('--seed', type=int, help='Override random seed for every dataset.')
    parser.add_argument('--missing-rate', type=float, help='Per-view missing rate in [0, 1).')
    parser.add_argument('--missing-mode', choices=['random', 'block', 'strip'],
                        help='Pixel-level missingness protocol applied BEFORE SLIC.')
    parser.add_argument('--missing-seed', type=int,
                        help='Seed of the missingness simulator; defaults to --seed.')
    parser.add_argument('--superpixel-threshold', type=float,
                        help='Superpixel observed-ratio threshold tau.')
    parser.add_argument('--ablation-variant', help='Ablation rung A0|A1|A2|A3|A4 (A4 = full V5).')
    parser.add_argument('--graph-beta', type=float, help='Spatial-graph fusion weight beta in [0, 1].')
    parser.add_argument('--sinkhorn-epsilon', type=float,
                        help='Entropic regularisation of the BSSAT transport.')
    parser.add_argument('--bssat-dual', dest='bssat_dual', action='store_true', default=None,
                        help='BSSAT: spatial + semantic dual signature.')
    parser.add_argument('--no-bssat-dual', dest='bssat_dual', action='store_false',
                        help='Fall back to the v3 spatial-only CSATA signature.')
    parser.add_argument('--bssat-balanced', dest='bssat_balanced', action='store_true', default=None,
                        help='BSSAT: mass-preserving (balanced) transport.')
    parser.add_argument('--no-bssat-balanced', dest='bssat_balanced', action='store_false',
                        help='Fall back to the v3 row-stochastic Sinkhorn transport.')
    parser.add_argument('--lambda-spatial', type=float, help='BSSAT spatial cost weight.')
    parser.add_argument('--lambda-semantic', type=float, help='BSSAT semantic cost weight.')
    parser.add_argument('--mass-uniform-alpha', type=float,
                        help="BSSAT-v2 (outline v5): temper the balanced column "
                             "marginal with b' = (1 - alpha) b + alpha / K.")
    parser.add_argument('--reliability-adaptive-graph', dest='reliability_adaptive_graph',
                        action='store_true', default=None,
                        help='outline v6 module I: edge-wise adaptive graph fusion.')
    parser.add_argument('--no-reliability-adaptive-graph', dest='reliability_adaptive_graph',
                        action='store_false',
                        help='Keep the fixed fusion (frozen BSSAT-r1 control).')
    parser.add_argument('--radg-mode', choices=['centered', 'uncentered'],
                        help='RADG v1.1: centered keeps E[beta] at radg_base_beta.')
    parser.add_argument('--radg-base-beta', type=float,
                        help='RADG centered: baseline beta_0 (default 0.5).')
    parser.add_argument('--radg-gamma', type=float,
                        help='RADG centered: modulation strength gamma.')
    parser.add_argument('--radg-beta-floor', type=float,
                        help='RADG centered: lower clip for beta_ij.')
    parser.add_argument('--radg-beta-ceiling', type=float,
                        help='RADG centered: upper clip for beta_ij.')
    parser.add_argument('--radg-center-tolerance', type=float,
                        help='RADG centered: tolerated |E[beta] - beta_0|.')
    parser.add_argument('--radg-beta-min', type=float, help='RADG beta at r_i = 1.')
    parser.add_argument('--radg-beta-max', type=float, help='RADG beta at r_i = 0.')
    parser.add_argument('--radg-edge-reliability',
                        choices=['geometric_mean', 'arithmetic_mean'],
                        help='RADG: combine r_i and r_j into c_ij.')
    parser.add_argument('--deterministic-mode', dest='deterministic_mode',
                        action='store_true', default=None,
                        help='outline v6 section 10: tighten CUDA determinism.')
    parser.add_argument('--no-deterministic-mode', dest='deterministic_mode',
                        action='store_false',
                        help='Leave the default CUDA kernels enabled.')
    parser.add_argument('--drop-nonpositive-labels', action='store_true',
                        help='Treat labels <= 0 as background (MUUFL gt uses -1).')
    parser.add_argument('--run-id', help='Explicit run id (useful for multi-seed launchers).')
    parser.add_argument('--output-root', type=Path, help='Override the configured result directory.')
    parser.add_argument('--continue-on-error', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    # ---- outline v9, module III: completion adapter -----------------------
    parser.add_argument('--completion-enabled', dest='completion_enabled',
                        action='store_true', default=None,
                        help='Attach the read-only completion adapter (eight arms).')
    parser.add_argument('--no-completion', dest='completion_enabled',
                        action='store_false',
                        help='Keep the frozen B0 path only (default).')
    parser.add_argument('--completion-stage', choices=['inference_only', 'off'],
                        help='v9 is inference-only; "off" disables the adapter.')
    parser.add_argument('--completion-lambdas', type=float, nargs='+',
                        help='Completion interpolation strengths.')
    parser.add_argument('--completion-arms', nargs='+',
                        help='Arms to record, e.g. B0 E0 G0 D0 C025 C050 C075 C100.')
    parser.add_argument('--adapter-seed', type=int,
                        help='Seed of the read-only evaluation adapter.')
    parser.add_argument('--diagnostic-seed', type=int,
                        help='Seed of the pseudo-missing probe sampling.')
    parser.add_argument('--diagnostic-epochs', type=int, nargs='+',
                        help='Rounds at which the pseudo-missing probe runs.')
    parser.add_argument('--completion-min-valid-mass', type=float,
                        help='Minimum donor mass on valid OT columns for eligibility.')
    return parser


def main():
    args = build_parser().parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_json(config_path)
    datasets = args.datasets or config.get('datasets_to_run') or [config.get('dataset_name')]
    datasets = [name for name in datasets if name]
    if not datasets:
        raise ValueError('No datasets selected.')

    output_root = args.output_root or Path(config.get('result_path', SCRIPT_DIR / 'results'))
    if not output_root.is_absolute():
        output_root = (SCRIPT_DIR / output_root).resolve()
    run_id = args.run_id or datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    batch_started = datetime.now().astimezone()
    batch_clock = time.perf_counter()
    rows = []

    for dataset in datasets:
        command = [
            sys.executable,
            str(SCRIPT_DIR / 'main.py'),
            '--config', str(config_path),
            '--dataset', dataset,
            '--output-root', str(output_root),
            '--run-id', run_id,
        ]
        if args.gpu is not None:
            command.extend(['--gpu', str(args.gpu)])
        if args.epochs is not None:
            command.extend(['--epochs', str(args.epochs)])
        if args.seed is not None:
            command.extend(['--seed', str(args.seed)])
        if args.missing_rate is not None:
            command.extend(['--missing-rate', str(args.missing_rate)])
        if args.missing_mode is not None:
            command.extend(['--missing-mode', str(args.missing_mode)])
        if args.missing_seed is not None:
            command.extend(['--missing-seed', str(args.missing_seed)])
        if args.superpixel_threshold is not None:
            command.extend(['--superpixel-threshold', str(args.superpixel_threshold)])
        if args.ablation_variant is not None:
            command.extend(['--ablation-variant', str(args.ablation_variant)])
        if args.graph_beta is not None:
            command.extend(['--graph-beta', str(args.graph_beta)])
        if args.sinkhorn_epsilon is not None:
            command.extend(['--sinkhorn-epsilon', str(args.sinkhorn_epsilon)])
        if args.bssat_dual is not None:
            command.append('--bssat-dual' if args.bssat_dual else '--no-bssat-dual')
        if args.bssat_balanced is not None:
            command.append(
                '--bssat-balanced' if args.bssat_balanced else '--no-bssat-balanced'
            )
        if args.lambda_spatial is not None:
            command.extend(['--lambda-spatial', str(args.lambda_spatial)])
        if args.lambda_semantic is not None:
            command.extend(['--lambda-semantic', str(args.lambda_semantic)])
        if args.mass_uniform_alpha is not None:
            command.extend(['--mass-uniform-alpha', str(args.mass_uniform_alpha)])
        if args.reliability_adaptive_graph is not None:
            command.append(
                '--reliability-adaptive-graph' if args.reliability_adaptive_graph
                else '--no-reliability-adaptive-graph'
            )
        if args.radg_mode is not None:
            command.extend(['--radg-mode', str(args.radg_mode)])
        if args.radg_base_beta is not None:
            command.extend(['--radg-base-beta', str(args.radg_base_beta)])
        if args.radg_gamma is not None:
            command.extend(['--radg-gamma', str(args.radg_gamma)])
        if args.radg_beta_floor is not None:
            command.extend(['--radg-beta-floor', str(args.radg_beta_floor)])
        if args.radg_beta_ceiling is not None:
            command.extend(['--radg-beta-ceiling', str(args.radg_beta_ceiling)])
        if args.radg_center_tolerance is not None:
            command.extend(['--radg-center-tolerance', str(args.radg_center_tolerance)])
        if args.radg_beta_min is not None:
            command.extend(['--radg-beta-min', str(args.radg_beta_min)])
        if args.radg_beta_max is not None:
            command.extend(['--radg-beta-max', str(args.radg_beta_max)])
        if args.radg_edge_reliability is not None:
            command.extend(['--radg-edge-reliability', str(args.radg_edge_reliability)])
        if args.deterministic_mode is not None:
            command.append(
                '--deterministic-mode' if args.deterministic_mode
                else '--no-deterministic-mode'
            )
        if args.drop_nonpositive_labels:
            command.append('--drop-nonpositive-labels')
        if args.validate_only:
            command.append('--validate-only')
        # ---- outline v9, module III --------------------------------------
        if args.completion_enabled is not None:
            command.append('--completion-enabled' if args.completion_enabled
                           else '--no-completion')
        if args.completion_stage is not None:
            command.extend(['--completion-stage', str(args.completion_stage)])
        if args.completion_lambdas is not None:
            command.extend(['--completion-lambdas'] +
                           [str(x) for x in args.completion_lambdas])
        if args.completion_arms is not None:
            command.extend(['--completion-arms'] + [str(x) for x in args.completion_arms])
        if args.adapter_seed is not None:
            command.extend(['--adapter-seed', str(args.adapter_seed)])
        if args.diagnostic_seed is not None:
            command.extend(['--diagnostic-seed', str(args.diagnostic_seed)])
        if args.diagnostic_epochs is not None:
            command.extend(['--diagnostic-epochs'] +
                           [str(x) for x in args.diagnostic_epochs])
        if args.completion_min_valid_mass is not None:
            command.extend(['--completion-min-valid-mass',
                            str(args.completion_min_valid_mass)])

        print(f"\n{'=' * 72}\nRunning dataset: {dataset}\n{'=' * 72}")
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=SCRIPT_DIR, check=False)
        row = {
            'dataset': dataset,
            'seed': args.seed if args.seed is not None else config.get('seed'),
            'gpu': args.gpu if args.gpu is not None else config.get('gpu_id'),
            'epochs': args.epochs if args.epochs is not None else config.get('n_epoches'),
            'missing_rate': args.missing_rate if args.missing_rate is not None else config.get('missing_rate'),
            'missing_mode': args.missing_mode if args.missing_mode is not None else config.get('missing_mode'),
            'missing_seed': args.missing_seed if args.missing_seed is not None else config.get('missing_seed'),
            'ablation_variant': args.ablation_variant if args.ablation_variant is not None else config.get('ablation_variant'),
            'graph_beta': args.graph_beta if args.graph_beta is not None else config.get('graph_beta'),
            'sinkhorn_epsilon': (
                args.sinkhorn_epsilon if args.sinkhorn_epsilon is not None
                else config.get('sinkhorn_epsilon')
            ),
            'csata_dual_signature': (
                args.bssat_dual if args.bssat_dual is not None
                else config.get('csata_dual_signature', True)
            ),
            'csata_balanced': (
                args.bssat_balanced if args.bssat_balanced is not None
                else config.get('csata_balanced', True)
            ),
            'mass_uniform_alpha': (
                args.mass_uniform_alpha if args.mass_uniform_alpha is not None
                else config.get('mass_uniform_alpha', 0.0)
            ),
            'reliability_adaptive_graph': (
                args.reliability_adaptive_graph
                if args.reliability_adaptive_graph is not None
                else config.get('reliability_adaptive_graph', False)
            ),
            'radg_beta_min': (
                args.radg_beta_min if args.radg_beta_min is not None
                else config.get('radg_beta_min', 0.2)
            ),
            'radg_beta_max': (
                args.radg_beta_max if args.radg_beta_max is not None
                else config.get('radg_beta_max', 0.8)
            ),
            'radg_mode': (
                args.radg_mode if args.radg_mode is not None
                else config.get('radg_mode', 'centered')
            ),
            'radg_base_beta': (
                args.radg_base_beta if args.radg_base_beta is not None
                else config.get('radg_base_beta', 0.5)
            ),
            'radg_gamma': (
                args.radg_gamma if args.radg_gamma is not None
                else config.get('radg_gamma', 0.5)
            ),
            'radg_beta_floor': (
                args.radg_beta_floor if args.radg_beta_floor is not None
                else config.get('radg_beta_floor', 0.2)
            ),
            'radg_beta_ceiling': (
                args.radg_beta_ceiling if args.radg_beta_ceiling is not None
                else config.get('radg_beta_ceiling', 0.8)
            ),
            'radg_center_tolerance': (
                args.radg_center_tolerance if args.radg_center_tolerance is not None
                else config.get('radg_center_tolerance', 0.005)
            ),
            'radg_edge_reliability': (
                args.radg_edge_reliability if args.radg_edge_reliability is not None
                else config.get('radg_edge_reliability', 'geometric_mean')
            ),
            'deterministic_mode': (
                args.deterministic_mode if args.deterministic_mode is not None
                else config.get('deterministic_mode', False)
            ),
            'drop_nonpositive_labels': bool(
                args.drop_nonpositive_labels or config.get('drop_nonpositive_labels', False)
            ),
            # outline v9, module III
            'completion_enabled': (
                args.completion_enabled if args.completion_enabled is not None
                else config.get('completion_enabled', False)
            ),
            'completion_arms': (
                ' '.join(args.completion_arms) if args.completion_arms is not None
                else ' '.join(config.get('completion_arms', []))
            ),
            'adapter_seed': (
                args.adapter_seed if args.adapter_seed is not None
                else config.get('adapter_seed', 20260922)
            ),
            'status': 'completed' if completed.returncode == 0 else 'failed',
            'exit_code': completed.returncode,
            'runtime_seconds': float(time.perf_counter() - started),
            'result_directory': str(output_root / dataset / run_id),
        }
        metrics_path = output_root / dataset / run_id / 'metrics.json'
        if metrics_path.exists():
            metrics = load_json(metrics_path)
            for key in ['ACC', 'Kappa', 'NMI', 'ARI', 'PURITY', 'best_round']:
                row[key] = metrics.get(key)
        rows.append(row)
        if completed.returncode != 0 and not args.continue_on_error:
            break

    if args.validate_only:
        return 0 if all(row['exit_code'] == 0 for row in rows) else 1

    batch_dir = output_root / '_batches' / run_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(batch_dir / 'summary.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    save_json(batch_dir / 'summary.json', {
        'run_id': run_id,
        'seed': args.seed if args.seed is not None else config.get('seed'),
        'gpu': args.gpu if args.gpu is not None else config.get('gpu_id'),
        'epochs': args.epochs if args.epochs is not None else config.get('n_epoches'),
        'missing_mode': args.missing_mode if args.missing_mode is not None else config.get('missing_mode'),
        'missing_rate': args.missing_rate if args.missing_rate is not None else config.get('missing_rate'),
        'missing_seed': args.missing_seed if args.missing_seed is not None else config.get('missing_seed'),
        'ablation_variant': args.ablation_variant if args.ablation_variant is not None else config.get('ablation_variant'),
        'graph_beta': args.graph_beta if args.graph_beta is not None else config.get('graph_beta'),
        'reliability_adaptive_graph': (
            args.reliability_adaptive_graph if args.reliability_adaptive_graph is not None
            else config.get('reliability_adaptive_graph', False)
        ),
        'radg_beta_min': (
            args.radg_beta_min if args.radg_beta_min is not None
            else config.get('radg_beta_min', 0.2)
        ),
        'radg_beta_max': (
            args.radg_beta_max if args.radg_beta_max is not None
            else config.get('radg_beta_max', 0.8)
        ),
        'start_time': batch_started.isoformat(),
        'end_time': datetime.now().astimezone().isoformat(),
        'runtime_seconds': float(time.perf_counter() - batch_clock),
        'datasets': datasets,
        'runs': rows,
    })
    print(f'\nBatch summary: {batch_dir}')
    return 0 if all(row['exit_code'] == 0 for row in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
