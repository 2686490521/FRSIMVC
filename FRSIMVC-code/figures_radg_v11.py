"""Figures for the RADG-v1.1 four-arm report."""
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = '/home/ubuntu/Tangzl/FRSIMVC/FRSIMVC/FRSIMVC-code/results/radg_v11'
OUTDIR = os.path.join(ROOT, '_analysis')
JSON_IN = os.path.join(OUTDIR, 'radg_v11_runs.json')

DATASETS = ['Augsburg', 'MDAS', 'MUUFL', 'Salinas', 'Trento', 'XuZhou']
ARMS = ['F05', 'F03', 'U_RADG', 'C_RADG']
COLORS = {'F05': '#4C72B0', 'F03': '#DD8452', 'U_RADG': '#C44E52', 'C_RADG': '#55A868'}


def load():
    with open(JSON_IN) as fh:
        recs = json.load(fh)
    by = {}
    for r in recs:
        by.setdefault((r['dataset'], r['arm']), {})[r['seed']] = r
    return by


def mv(rec, key):
    if key in rec and rec[key] is not None:
        return float(rec[key])
    v = rec['metrics'].get(key)
    return float(v) if v is not None else np.nan


def main():
    by = load()
    # fig 1: grouped bars of best ACC
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(DATASETS))
    w = 0.2
    for i, arm in enumerate(ARMS):
        means, errs = [], []
        for ds in DATASETS:
            v = np.array([mv(by[(ds, arm)][s], 'ACC') for s in range(10)], dtype=float)
            means.append(np.nanmean(v))
            errs.append(np.nanstd(v, ddof=1) / np.sqrt(10))
        ax.bar(x + (i - 1.5) * w, means, w, yerr=errs, label=arm, color=COLORS[arm],
               capsize=2, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylabel('best ACC (mean ± SEM, 10 seeds)')
    ax.set_title('RADG-v1.1 four arms — best ACC by dataset (random@0.3, 300 ep)')
    ax.legend(ncol=4, frameon=False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(OUTDIR, 'fig1_acc_by_arm.png')
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    # fig 2: paired deltas
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    contrasts = [('C_RADG', 'F05'), ('U_RADG', 'F05')]
    for ax, (a, b) in zip(axes, contrasts):
        data, labels = [], []
        for ds in DATASETS:
            d = np.array([mv(by[(ds, a)][s], 'ACC') - mv(by[(ds, b)][s], 'ACC') for s in range(10)], dtype=float)
            data.append(d)
            labels.append(ds)
        bp = ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#dfe6f2')
        ax.axhline(0, color='#C44E52', ls='--', lw=1)
        ax.set_title('%s − %s (paired, per seed)' % (a, b))
        ax.set_ylabel('Δ best ACC')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    p2 = os.path.join(OUTDIR, 'fig2_paired_delta.png')
    fig.savefig(p2, dpi=160)
    plt.close(fig)

    # fig 3: dose response
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for ds in DATASETS:
        xs = np.array([mv(by[(ds, 'C_RADG')][s], 'beta_std') for s in range(10)], dtype=float)
        ys = np.array([mv(by[(ds, 'C_RADG')][s], 'ACC') - mv(by[(ds, 'F05')][s], 'ACC') for s in range(10)], dtype=float)
        ax.scatter(xs, ys, s=26, alpha=0.8, label=ds)
    xs = np.array([mv(by[(ds, 'C_RADG')][s], 'beta_std') for ds in DATASETS for s in range(10)], dtype=float)
    ys = np.array([mv(by[(ds, 'C_RADG')][s], 'ACC') - mv(by[(ds, 'F05')][s], 'ACC') for ds in DATASETS for s in range(10)], dtype=float)
    ok = ~(np.isnan(xs) | np.isnan(ys))
    lr = __import__('scipy.stats', fromlist=['linregress']).linregress(xs[ok], ys[ok])
    xx = np.linspace(xs[ok].min(), xs[ok].max(), 50)
    ax.plot(xx, lr.intercept + lr.slope * xx, color='k', lw=1.2,
            label='fit slope=%+.2f (p=%.2f)' % (lr.slope, lr.pvalue))
    ax.axhline(0, color='#C44E52', ls='--', lw=1)
    ax.set_xlabel('realised β std (C_RADG)')
    ax.set_ylabel('Δ best ACC (C_RADG − F05)')
    ax.set_title('dose–response (60 runs)')
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p3 = os.path.join(OUTDIR, 'fig3_dose_response.png')
    fig.savefig(p3, dpi=160)
    plt.close(fig)

    print('written:\n', p1, '\n', p2, '\n', p3)


if __name__ == '__main__':
    main()
