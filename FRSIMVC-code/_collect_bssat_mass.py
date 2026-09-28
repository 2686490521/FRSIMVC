"""Collect one BSSAT-v2 (alpha sweep) batch into a single JSON for analysis.

Stdlib only, so it can run directly inside the experiment server's
environment. Reads, for every run id listed in ``--run-ids-file``:

    <root>/<dataset>/<run_id>/{run_metadata.json, metrics.json, history.csv,
                               artifacts/diagnostics.json}

and emits per run: alpha / protocol fields, best-round metrics, the outline-v5
concentration diagnostics, the foreground-only metrics, the best round's
per-class recall, the raw vs tempered anchor mass with their concentrations,
transport diagnostics, and per-round statistics for all five headline metrics.

Usage (on the server):
  python _collect_bssat_mass.py --root results/bssat_mass \
      --run-ids-file results/bssat_mass/_launches/<ts>/run_ids.txt \
      --datasets Trento MUUFL --out /tmp/bssat_mass.json
"""

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

HEADLINE = ('ACC', 'Kappa', 'NMI', 'ARI', 'PURITY')
TAIL = 30


def read_history(path):
    if not path.exists():
        return []
    rows = []
    with open(path, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            parsed = {}
            for key, value in row.items():
                if key == 'observed_rates':
                    continue
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    continue
            if parsed:
                rows.append(parsed)
    return rows


def summarize(rounds, key):
    values = [r[key] for r in rounds if key in r]
    if not values:
        return None
    ordered = sorted(values)
    return {
        'n_rounds': len(values),
        'best': max(values),
        'best_round': values.index(max(values)) + 1,
        'median': statistics.median(values),
        'mean': statistics.fmean(values),
        'p25': ordered[max(0, int(0.25 * (len(ordered) - 1)))],
        'p10': ordered[max(0, int(0.10 * (len(ordered) - 1)))],
        'final': values[-1],
        'tail_mean': statistics.fmean(values[-TAIL:]),
        'std': statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def load_json(path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--run-ids-file', required=True, type=Path)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()

    run_ids = [
        line.strip() for line in args.run_ids_file.read_text(encoding='utf-8').splitlines()
        if line.strip() and not line.startswith('#')
    ]
    print('run ids: %d' % len(run_ids))

    runs = []
    problems = []
    for dataset in args.datasets:
        for run_id in run_ids:
            run_dir = args.root / dataset / run_id
            meta = load_json(run_dir / 'run_metadata.json')
            if meta is None:
                problems.append('%s/%s: missing run_metadata.json' % (dataset, run_id))
                continue
            metrics = load_json(run_dir / 'metrics.json', {}) or {}
            rounds = read_history(run_dir / 'history.csv')
            diag = load_json(run_dir / 'artifacts' / 'diagnostics.json', {}) or {}
            clustering = diag.get('clustering') or {}
            foreground = diag.get('foreground_metrics') or {}

            sizes = clustering.get('cluster_sizes_vector') or []
            record = {
                'dataset': dataset,
                'run_id': run_id,
                'arm': run_id.split('__')[1] if '__' in run_id else None,
                'status': meta.get('status'),
                'seed': meta.get('seed'),
                'missing_seed': meta.get('missing_seed'),
                'alpha': meta.get('mass_uniform_alpha', 0.0),
                'beta': meta.get('graph_beta'),
                'epsilon': meta.get('sinkhorn_epsilon'),
                'mode': meta.get('missing_mode'),
                'rate': meta.get('missing_rate'),
                'dual_signature': meta.get('csata_dual_signature'),
                'balanced': meta.get('csata_balanced'),
                'best_round': metrics.get('best_round'),
                'num_active_nodes': metrics.get('num_active_nodes'),
                'num_gt_classes': metrics.get('num_gt_classes'),
                'foreground_nodes': foreground.get('foreground_nodes'),
                'background_nodes': foreground.get('background_nodes'),
                'cluster_sizes': sizes,
                'cluster_max_nodes': max(sizes) if sizes else None,
                'cluster_min_nodes': min(sizes) if sizes else None,
                'transport': {
                    'C_T': diag.get('transport_confidence_C_T_mean'),
                    'H_T': diag.get('transport_entropy_H_T_mean'),
                    'D_inter': diag.get('signature_separation_D_inter'),
                },
                'mass': {
                    'raw': diag.get('global_mass_raw'),
                    'tempered': diag.get('global_mass_tempered'),
                    'conc_raw': diag.get('mass_concentration_raw'),
                    'conc_tempered': diag.get('mass_concentration_tempered'),
                },
                'per_gt_class': clustering.get('per_gt_class'),
                'per_cluster': clustering.get('per_cluster'),
                'acc_series': [r['ACC'] for r in rounds if 'ACC' in r],
            }
            for key in HEADLINE:
                record[key] = metrics.get(key)
                record['stats_%s' % key] = summarize(rounds, key)
            for key in ('ACC_foreground', 'NMI_foreground', 'ARI_foreground'):
                record[key] = metrics.get(key)
            for key in ('cluster_max_ratio', 'cluster_min_ratio',
                        'cluster_entropy', 'effective_num_clusters',
                        'num_predicted_clusters'):
                record[key] = metrics.get(key)
            runs.append(record)

    payload = {
        'root': str(args.root),
        'run_ids_file': str(args.run_ids_file),
        'datasets': args.datasets,
        'n_runs': len(runs),
        'status_counts': dict(Counter(r['status'] for r in runs)),
        'problems': problems,
        'runs': runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    print('collected %d runs -> %s' % (len(runs), args.out))
    print('status:', payload['status_counts'])
    if problems:
        print('problems (%d):' % len(problems))
        for item in problems[:20]:
            print('  ', item)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
