"""Aggregate one protocol batch into a mean +/- std table.

Reads ``<output_root>/<dataset>/<run_id>/{run_metadata.json,metrics.json}``
so it never has to guess what a run did: the effective missing protocol,
rate, seed and ablation arm are all taken from the recorded metadata.

The arm label is parsed out of the run id (``<batch>__<arm>__<mode>__mr...``)
because the ``complete`` arm reuses the A0 weights and would otherwise be
indistinguishable from A0.

Outline v5 adds two families of columns on top of the core metrics:

* ``ACC_foreground`` / ``NMI_foreground`` / ``ARI_foreground`` -- MUUFL scored
  on ``y_i != -1`` only (blank for the datasets whose background label is 0);
* ``cluster_max_ratio`` / ``cluster_min_ratio`` / ``cluster_entropy`` /
  ``effective_num_clusters`` -- the concentration diagnostics replacing the old
  "collapse count", read from the same best round as the headline ACC.

Both families are printed in a compact secondary table and always written to
the CSV; they are never folded into the ACC column.

Outline v6 (module I / RADG-v1) adds the two axes ``beta_min`` / ``beta_max``
and a third column family: the adaptive fusion's own modulation diagnostics
(``beta_mean/std/p10/p50/p90``, the *attained* ``beta_min_actual`` /
``beta_max_actual``, ``observed_ratio_mean``, the graph change
``graph_relative_frobenius_change`` and ``corr_r_beta_node``).  They are read
from ``metrics.json`` like every other column, so a sweep where the modulation
never moved is visible in the aggregation itself rather than three weeks later.
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

CORE_METRICS = ('ACC', 'Kappa', 'NMI', 'ARI', 'PURITY')
EXTRA_METRICS = (
    'ACC_foreground',
    'NMI_foreground',
    'ARI_foreground',
    'cluster_max_ratio',
    'cluster_min_ratio',
    'cluster_entropy',
    'effective_num_clusters',
)
# outline v6/v7, module I (RADG): was the adaptive fusion actually adaptive?
RADG_METRICS = (
    'beta_mean',
    'beta_std',
    'beta_p10',
    'beta_p25',
    'beta_p50',
    'beta_p75',
    'beta_p90',
    'beta_min_actual',
    'beta_max_actual',
    'beta_mean_error_from_base',
    'beta_mean_error_before_recentre',
    'recentre_iterations',
    'recentre_delta',
    'edge_uncertainty_mean',
    'edge_uncertainty_std',
    'observed_ratio_mean',
    'observed_ratio_std',
    'graph_relative_frobenius_change',
    'corr_r_beta_node',
    'corr_r_neighbour_missing',
    'corr_r_feature_edge_loss',
)
# Short console labels so the secondary table stays inside 108 columns.
EXTRA_LABELS = {
    'ACC_foreground': 'ACC_fg',
    'NMI_foreground': 'NMI_fg',
    'ARI_foreground': 'ARI_fg',
    'cluster_max_ratio': 'clu_p_max',
    'cluster_min_ratio': 'clu_p_min',
    'cluster_entropy': 'clu_H_norm',
    'effective_num_clusters': 'clu_K_eff',
}
RADG_LABELS = {
    'beta_mean': 'beta_mean',
    'beta_std': 'beta_std',
    'beta_p10': 'beta_p10',
    'beta_p50': 'beta_p50',
    'beta_p90': 'beta_p90',
    'beta_min_actual': 'b_min_act',
    'beta_max_actual': 'b_max_act',
    'beta_mean_error_from_base': 'err_from_base',
    'beta_mean_error_before_recentre': 'err_pre_recent',
    'recentre_iterations': 'recentre_it',
    'recentre_delta': 'recentre_d',
    'edge_uncertainty_mean': 'u_mean',
    'edge_uncertainty_std': 'u_std',
    'observed_ratio_mean': 'r_mean',
    'observed_ratio_std': 'r_std',
    'graph_relative_frobenius_change': 'delta_A',
    'corr_r_beta_node': 'corr(r,beta)',
    'corr_r_neighbour_missing': 'corr(r,nbr)',
    'corr_r_feature_edge_loss': 'corr(r,loss)',
}
CSV_FIELDS = [
    'dataset', 'arm', 'alpha', 'radg_mode', 'beta_min', 'beta_max', 'base_beta',
    'gamma', 'beta_floor', 'beta_ceiling', 'mode', 'rate', 'beta', 'epsilon',
    'bssat_dual', 'bssat_balanced', 'metric', 'mean', 'std', 'n',
]
# ``tag()`` in the launcher writes 0.2 as "0p2"; recover the RADG bounds from
# the run id when a legacy run has no radg_beta_min/max in its metadata.
_RADG_BOUNDS_RE = re.compile(r'__bm(\d+p?\d*)-(\d+p?\d*)__')


def _tagged_float(token):
    return float(str(token).replace('p', '.'))


def parse_radg_bounds(run_id):
    match = _RADG_BOUNDS_RE.search(str(run_id))
    if not match:
        return None, None
    try:
        return _tagged_float(match.group(1)), _tagged_float(match.group(2))
    except ValueError:
        return None, None


def parse_arm(run_id, meta):
    parts = str(run_id).split('__')
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return meta.get('ablation_variant') or 'FULL'


def load_runs(output_root, prefix=None, run_ids=None):
    keep = None if run_ids is None else set(run_ids)
    rows = []
    for meta_path in sorted(Path(output_root).glob('*/*/run_metadata.json')):
        run_dir = meta_path.parent
        if prefix and not run_dir.name.startswith(prefix):
            continue
        if keep is not None and run_dir.name not in keep:
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        metrics_path = run_dir / 'metrics.json'
        metrics = {}
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                metrics = {}
        rate = meta.get('missing_rate')
        beta_min = meta.get('radg_beta_min')
        beta_max = meta.get('radg_beta_max')
        if beta_min is None or beta_max is None:
            parsed_min, parsed_max = parse_radg_bounds(run_dir.name)
            beta_min = parsed_min if beta_min is None else beta_min
            beta_max = parsed_max if beta_max is None else beta_max
        row = {
            'dataset': meta.get('dataset'),
            'run_id': run_dir.name,
            'arm': parse_arm(run_dir.name, meta),
            'alpha': meta.get('mass_uniform_alpha'),
            'beta_min': None if beta_min is None else float(beta_min),
            'beta_max': None if beta_max is None else float(beta_max),
            'radg_mode': (
                meta.get('radg_mode') if meta.get('reliability_adaptive_graph')
                else None
            ),
            'base_beta': (
                float(meta['radg_base_beta']) if meta.get('radg_base_beta') is not None
                else None
            ),
            'gamma': (
                float(meta['radg_gamma']) if meta.get('radg_gamma') is not None
                else None
            ),
            'beta_floor': (
                float(meta['radg_beta_floor']) if meta.get('radg_beta_floor') is not None
                else None
            ),
            'beta_ceiling': (
                float(meta['radg_beta_ceiling']) if meta.get('radg_beta_ceiling') is not None
                else None
            ),
            'mode': meta.get('missing_mode'),
            'rate': None if rate is None else float(rate),
            'beta': meta.get('graph_beta'),
            'epsilon': meta.get('sinkhorn_epsilon'),
            'bssat_dual': meta.get('csata_dual_signature'),
            'bssat_balanced': meta.get('csata_balanced'),
            'missing_seed': meta.get('missing_seed'),
            'seed': meta.get('seed'),
            'status': meta.get('status'),
            'best_round': metrics.get('best_round'),
            'result_directory': str(run_dir),
        }
        for key in CORE_METRICS + EXTRA_METRICS + RADG_METRICS:
            row[key] = metrics.get(key)
        rows.append(row)
    return rows


def mean_std(values):
    clean = [float(value) for value in values
             if value is not None and math.isfinite(float(value))]
    if not clean:
        return None, None, 0
    mean = sum(clean) / len(clean)
    if len(clean) == 1:
        return mean, 0.0, 1
    variance = sum((value - mean) ** 2 for value in clean) / (len(clean) - 1)
    return mean, math.sqrt(max(variance, 0.0)), len(clean)


def format_cell(mean, std, count):
    if count == 0 or mean is None:
        return '--'
    return '%.4f+-%.4f' % (mean, std)


def collect_metric_cells(csv_rows, dataset, arm, axis_values, arm_rows, metrics, width):
    """Append one CSV row per metric and return the formatted console cells."""
    cells = []
    for metric in metrics:
        mean, std, count = mean_std([
            member[metric] for member in arm_rows
            if member['status'] == 'completed'
        ])
        cells.append(format_cell(mean, std, count))
        csv_rows.append({
            'dataset': dataset,
            'arm': arm,
            'alpha': axis_values.get('alpha'),
            'beta_min': axis_values.get('beta_min'),
            'beta_max': axis_values.get('beta_max'),
            'radg_mode': axis_values.get('radg_mode'),
            'base_beta': axis_values.get('base_beta'),
            'gamma': axis_values.get('gamma'),
            'beta_floor': axis_values.get('beta_floor'),
            'beta_ceiling': axis_values.get('beta_ceiling'),
            'mode': axis_values.get('mode'),
            'rate': axis_values.get('rate'),
            'beta': axis_values.get('beta'),
            'epsilon': axis_values.get('epsilon'),
            'bssat_dual': axis_values.get('bssat_dual'),
            'bssat_balanced': axis_values.get('bssat_balanced'),
            'metric': metric,
            'mean': None if mean is None else round(mean, 6),
            'std': None if std is None else round(std, 6),
            'n': count,
        })
    return ''.join('%-*s' % (width, cell) for cell in cells)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--run-prefix', default=None,
                        help='Only aggregate run ids starting with this prefix. '
                             'NOTE: a batch id (timestamp) is NOT a run-id prefix.')
    parser.add_argument('--run-ids-file', type=Path, default=None,
                        help='Only aggregate the run ids listed in this file '
                             '(one per line; blank lines and #comments ignored).')
    parser.add_argument('--group-by', default='dataset,arm,mode,rate',
                        help='Comma separated grouping axes.')
    parser.add_argument('--csv-out', type=Path, default=None)
    args = parser.parse_args()

    run_ids = None
    if args.run_ids_file is not None:
        if not args.run_ids_file.is_file():
            print('ERROR: --run-ids-file not found: %s' % args.run_ids_file)
            return 2
        run_ids = []
        for line in args.run_ids_file.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line and not line.startswith('#'):
                run_ids.append(line)
        print('restricting to %d run id(s) from %s' % (len(run_ids), args.run_ids_file))
    rows = load_runs(args.output_root, args.run_prefix, run_ids)
    if not rows:
        print('No runs found under %s (prefix=%s, run_ids_file=%s)' % (
            args.output_root, args.run_prefix, args.run_ids_file))
        return 1

    group_axes = [axis.strip() for axis in args.group_by.split(',') if axis.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(axis) for axis in group_axes)].append(row)

    datasets = sorted({row['dataset'] for row in rows if row['dataset'] is not None})
    print('Aggregated %d run(s) from %s' % (len(rows), args.output_root))
    print('Grouped by: %s' % ', '.join(group_axes))

    csv_rows = []
    for dataset in datasets:
        print()
        print('=' * 108)
        print('  %s' % dataset)
        print('=' * 108)
        keys = [key for key in groups if key[0] == dataset]
        # Sort protocol axes first and the arm last, so the table reads
        # "(mode, rate) -> arms" instead of being arm-major.
        order = [group_axes.index(axis) for axis in reversed(group_axes) if axis != 'dataset']
        keys.sort(key=lambda key: tuple(str(key[index]) for index in order))
        for key in keys:
            members = groups[key]
            axis_values = dict(zip(group_axes, key))
            status_counts = defaultdict(int)
            for member in members:
                status_counts[member['status']] += 1
            label = '  '.join(
                '%s=%s' % (axis, key[index]) for index, axis in enumerate(group_axes) if axis != 'dataset'
            )
            arm_labels = sorted({member['arm'] for member in members})
            print()
            print('[%s]  %s' % (label, ', '.join(
                '%s:%d' % (status, count) for status, count in sorted(status_counts.items())
            )))
            header = '%-10s' % 'arm' + ''.join('%-18s' % metric for metric in CORE_METRICS) + '%6s' % 'n'
            print(header)
            print('-' * len(header))
            for arm in arm_labels:
                arm_rows = [member for member in members if member['arm'] == arm]
                completed = sum(1 for member in arm_rows if member['status'] == 'completed')
                print(
                    '%-10s' % arm
                    + collect_metric_cells(
                        csv_rows, dataset, arm, axis_values, arm_rows, CORE_METRICS, 18
                    )
                    + '%6d' % completed
                )
            # outline v5: foreground + concentration diagnostics (secondary table)
            available_extras = [
                metric for metric in EXTRA_METRICS
                if any(member[metric] is not None for member in members)
            ]
            if available_extras:
                extra_header = (
                    '%-10s' % 'arm'
                    + ''.join(
                        '%-18s' % EXTRA_LABELS.get(metric, metric)
                        for metric in available_extras
                    )
                    + '%6s' % 'n'
                )
                print('  [outline-v5 diagnostics]')
                print(extra_header)
                print('-' * len(extra_header))
                for arm in arm_labels:
                    arm_rows = [member for member in members if member['arm'] == arm]
                    completed = sum(
                        1 for member in arm_rows if member['status'] == 'completed'
                    )
                    print(
                        '%-10s' % arm
                        + collect_metric_cells(
                            csv_rows, dataset, arm, axis_values, arm_rows,
                            available_extras, 18,
                        )
                        + '%6d' % completed
                    )
            # outline v6: RADG modulation / graph-change diagnostics
            available_radg = [
                metric for metric in RADG_METRICS
                if any(member[metric] is not None for member in members)
            ]
            if available_radg:
                radg_header = (
                    '%-10s' % 'arm'
                    + ''.join(
                        '%-18s' % RADG_LABELS.get(metric, metric)
                        for metric in available_radg
                    )
                    + '%6s' % 'n'
                )
                print('  [outline-v6 RADG diagnostics]')
                print(radg_header)
                print('-' * len(radg_header))
                for arm in arm_labels:
                    arm_rows = [member for member in members if member['arm'] == arm]
                    completed = sum(
                        1 for member in arm_rows if member['status'] == 'completed'
                    )
                    print(
                        '%-10s' % arm
                        + collect_metric_cells(
                            csv_rows, dataset, arm, axis_values, arm_rows,
                            available_radg, 18,
                        )
                        + '%6d' % completed
                    )
            failed = [member['run_id'] for member in members if member['status'] != 'completed']
            if failed:
                print('  failed/missing metrics: %s' % ', '.join(failed))

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv_out, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(csv_rows)
        print()
        print('CSV: %s' % args.csv_out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
