"""Aggregate and analyse one outline-v9 Completion batch.

Usage:
    python report_completion.py --batch-dir results/completion_v1/batch_<id>_<tag>

Outputs everything into ``<batch-dir>/_analysis/``:

* ``completion_runs.json``  every arm record (dataset, seed, arm, metrics)
* ``main_table.md``         five metrics per arm: best (same round), final, tail30
* ``paired.md``             per-seed paired differences, Wilcoxon + Holm
* ``noninferiority.md``     bootstrap lower bound vs delta = 0.005
* ``completion_summary.csv`` flat table for further work
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None

METRICS = ['ACC', 'NMI', 'ARI', 'Kappa', 'PURITY']
PRIMARY_ARM = 'C050'
PRIMARY_LAMBDA = 0.5
DELTA = 0.005
N_BOOTSTRAP = 10000


def load_batch(batch_dir: Path):
    runs = []
    protocol_versions = set()
    for summary in sorted(batch_dir.glob('*/seed_*/completion_summary.json')):
        dataset = summary.parent.parent.name
        seed = int(summary.parent.name.split('_')[1])
        payload = json.loads(summary.read_text())
        for arm, entry in payload.items():
            runs.append({'dataset': dataset, 'seed': seed, 'arm': arm, **entry})
    return runs


def stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ''
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def holm(pvalues):
    """Holm step-down adjusted p-values (same order as the input)."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * pvalues[idx]
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return adjusted


def paired_diffs(runs, left, right, metric='ACC', stat='best'):
    index = defaultdict(dict)
    for run in runs:
        index[(run['dataset'], run['seed'])][run['arm']] = run
    out = defaultdict(list)
    for (dataset, seed), arms in sorted(index.items()):
        if left in arms and right in arms:
            key = f'{metric}_{stat}'
            out[dataset].append(arms[left][key] - arms[right][key])
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def wilcoxon(values):
    if stats is None or values.size < 5 or np.allclose(values - np.mean(values), 0):
        return np.nan
    try:
        _, p = stats.wilcoxon(values, zero_method='wilcox', alternative='two-sided')
    except Exception:
        return np.nan
    return float(p)


def bootstrap_lower(values, delta=DELTA, n_bootstrap=N_BOOTSTRAP, alpha=0.05, seed=20260922):
    """One-sided lower confidence bound of the paired mean difference."""
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_bootstrap, values.size), replace=True).mean(axis=1)
    lower = float(np.percentile(draws, 100.0 * alpha))
    return lower, float(draws.mean())


def equal_weight_mean(diffs_by_dataset):
    per_dataset = [float(v.mean()) for v in diffs_by_dataset.values()]
    return float(np.mean(per_dataset)) if per_dataset else np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-dir', required=True)
    parser.add_argument('--primary-arm', default=PRIMARY_ARM)
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir).resolve()
    analysis = batch_dir / '_analysis'
    analysis.mkdir(parents=True, exist_ok=True)
    runs = load_batch(batch_dir)
    if not runs:
        print(f'No completion_summary.json found under {batch_dir}')
        return 1

    with open(analysis / 'completion_runs.json', 'w', encoding='utf-8') as handle:
        json.dump(runs, handle, indent=2)

    datasets = sorted({run['dataset'] for run in runs})
    arms = sorted({run['arm'] for run in runs})
    seeds = sorted({run['seed'] for run in runs})
    print(f'batch      : {batch_dir}')
    print(f'datasets   : {len(datasets)}  seeds: {len(seeds)}  arms: {len(arms)}')
    print(f'records    : {len(runs)}')

    # ---- integrity: every dataset must have the same seed set -------------
    lines = ['# outline v9 -- Completion batch report', '']
    lines.append(f'- batch: `{batch_dir}`')
    lines.append(f'- datasets: {len(datasets)}, seeds: {len(seeds)}, arms: {len(arms)}')
    lines.append(f'- arm records: {len(runs)} (shared training trajectories, NOT independent runs)')
    lines.append('')
    missing = []
    for dataset in datasets:
        have = {run['seed'] for run in runs if run['dataset'] == dataset}
        if have != set(seeds):
            missing.append((dataset, sorted(set(seeds) - have)))
    if missing:
        lines.append('## ⚠️ integrity')
        lines.append('')
        for dataset, gone in missing:
            lines.append(f'- {dataset}: missing seeds {gone}')
        lines.append('')

    # ---- main table -------------------------------------------------------
    lines.append('## main table (best round, same-round companion metrics)')
    lines.append('')
    header = '| dataset | arm | ' + ' | '.join(METRICS) + ' | final ACC | tail30 ACC | median ACC | round std |'
    lines.append(header)
    lines.append('|' + '---|' * (len(METRICS) + 6))
    for dataset in datasets:
        for arm in arms:
            rows = [r for r in runs if r['dataset'] == dataset and r['arm'] == arm]
            if not rows:
                continue
            def mean(key):
                return float(np.mean([r[key] for r in rows]))
            cells = [f'{mean(m + "_best"):.4f}' for m in METRICS]
            lines.append(
                f'| {dataset} | {arm} | ' + ' | '.join(cells) +
                f' | {mean("ACC_final"):.4f} | {mean("ACC_tail30"):.4f} | '
                f'{mean("ACC_median"):.4f} | {mean("ACC_std"):.4f} |'
            )
        lines.append('')

    # ---- paired comparisons ----------------------------------------------
    contrasts = [(args.primary_arm, 'B0'), (args.primary_arm, 'G0'),
                 (args.primary_arm, 'D0'), ('E0', 'B0'), ('G0', 'E0'), ('D0', 'E0')]
    lines.append('## paired differences (per seed, best ACC)')
    lines.append('')
    lines.append('| contrast | dataset | mean Δ | wins | p | Holm p |')
    lines.append('|---|---|---|---|---|---|')
    for left, right in contrasts:
        diffs = paired_diffs(runs, left, right)
        if not diffs:
            continue
        raw = []
        rows = []
        for dataset in datasets:
            values = diffs.get(dataset)
            if values is None:
                continue
            p = wilcoxon(values)
            raw.append((dataset, values, p))
        pvalues = [np.nan if np.isnan(p) else p for _, _, p in raw]
        adjusted = holm(np.array([1.0 if np.isnan(v) else v for v in pvalues]))
        for (dataset, values, p), adj in zip(raw, adjusted):
            wins = int((values > 0).sum())
            lines.append(
                f'| {left} − {right} | {dataset} | {values.mean():+.4f} | {wins}/{values.size} | '
                f'{p:.4f} | {adj:.4f} {stars(adj)} |'
            )
        pooled = np.concatenate([v for _, v, _ in raw])
        lines.append(
            f'| {left} − {right} | **six-dataset mean** | {equal_weight_mean(diffs):+.4f} | '
            f'{int((pooled > 0).sum())}/{pooled.size} | - | - |'
        )
        lines.append('')

    # ---- non-inferiority --------------------------------------------------
    lines.append(f'## non-inferiority of {args.primary_arm} vs B0 (delta = {DELTA})')
    lines.append('')
    lines.append('| dataset | mean Δ | bootstrap lower bound (one-sided 95%) | passes |')
    lines.append('|---|---|---|---|')
    diffs = paired_diffs(runs, args.primary_arm, 'B0')
    verdicts = []
    for dataset in datasets:
        values = diffs.get(dataset)
        if values is None:
            continue
        lower, _ = bootstrap_lower(values, alpha=0.05 / max(1, len(datasets)))
        passes = lower > -DELTA
        verdicts.append(passes)
        lines.append(f'| {dataset} | {values.mean():+.4f} | {lower:+.4f} | '
                     f'{"yes" if passes else "no"} |')
    lines.append('')
    lines.append(f'- six-dataset equal-weight mean Δ: **{equal_weight_mean(diffs):+.4f}**')
    lines.append(f'- non-inferiority satisfied on {int(np.sum(verdicts))}/{len(verdicts)} datasets')
    lines.append(f'- engineering gate (mean must not decrease): '
                 f'{sum(1 for d in diffs.values() if d.mean() < 0)}/{len(diffs)} datasets decreased')
    lines.append('')

    # ---- flat csv ---------------------------------------------------------
    import csv
    keys = ['dataset', 'seed', 'arm', 'rounds'] + [
        f'{m}_{s}' for m in METRICS for s in ('best', 'best_round', 'final', 'tail30', 'median', 'std')
    ]
    with open(analysis / 'completion_summary.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        for run in sorted(runs, key=lambda r: (r['dataset'], r['arm'], r['seed'])):
            writer.writerow(run)

    (analysis / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:40]))
    print('...')
    print(f'written: {analysis / "report.md"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
