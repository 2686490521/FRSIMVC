"""Collect all RADG-v1.1 (outline v7) runs into a single JSON for offline analysis.

Usage (server):
    /home/ubuntu/miniconda3/envs/ICMVC/bin/python analysis_radg_v11_collect.py

Outputs:
    <ROOT>/_analysis/radg_v11_runs.json   one record per completed run
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = '/home/ubuntu/Tangzl/FRSIMVC/FRSIMVC/FRSIMVC-code/results/radg_v11'
OUTDIR = os.path.join(ROOT, '_analysis')
OUT = os.path.join(OUTDIR, 'radg_v11_runs.json')

ARM_RE = re.compile(r'__(F05|F03|U_RADG|C_RADG)__')
SEED_RE = re.compile(r'seed(\d+)$')
SKIP_DIRS = {'_batches', '_launches', '_analysis'}

HIST_COLS = ['ACC', 'NMI', 'ARI', 'Kappa', 'PURITY']


def hist_stats(path):
    out = {}
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover
        return {'history_error': str(exc)}
    n = len(df)
    tail = max(1, min(30, n // 10 if n // 10 else 30))
    for c in HIST_COLS:
        if c not in df.columns:
            continue
        v = df[c].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        if v.size == 0:
            continue
        out['%s_median' % c] = float(np.median(v))
        out['%s_p25' % c] = float(np.percentile(v, 25))
        out['%s_tail_mean' % c] = float(v[-tail:].mean())
        out['%s_final' % c] = float(v[-1])
        out['%s_std' % c] = float(v.std(ddof=1)) if v.size > 1 else 0.0
        out['%s_nrounds' % c] = int(v.size)
        out['%s_argmax_round' % c] = int(np.argmax(v) + 1)
    return out


def per_class_recall(art_dir):
    """Best-matched per-GT-class recall from the stored confusion matrix (GT x pred)."""
    p = os.path.join(art_dir, 'confusion_matrix.npy')
    if not os.path.exists(p):
        return None, None, None
    cm = np.load(p)
    if cm.ndim != 2:
        return None, None, None
    out = {}
    for i in range(cm.shape[0]):
        tot = float(cm[i].sum())
        out[str(i)] = float(cm[i].max() / tot) if tot > 0 else None
    sizes = [float(cm[i].sum()) for i in range(cm.shape[0])]
    labels = None
    lp = os.path.join(art_dir, 'evaluation_labels.npy')
    if os.path.exists(lp):
        y = np.load(lp).ravel()
        vals, cnts = np.unique(y, return_counts=True)
        if vals.size == cm.shape[0]:
            labels = [int(v) for v in vals]
            sizes = [float(c) for c in cnts]
    return out, sizes, labels


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    records = []
    for ds in sorted(os.listdir(ROOT)):
        dpath = os.path.join(ROOT, ds)
        if ds in SKIP_DIRS or not os.path.isdir(dpath):
            continue
        for run in sorted(os.listdir(dpath)):
            rpath = os.path.join(dpath, run)
            mj = os.path.join(rpath, 'metrics.json')
            if not os.path.isdir(rpath) or not os.path.exists(mj):
                continue
            marm = ARM_RE.search(run)
            mseed = SEED_RE.search(run)
            if not marm or not mseed:
                print('SKIP (unparsed name):', run, file=sys.stderr)
                continue
            with open(mj) as fh:
                m = json.load(fh)
            rec = {
                'dataset': ds,
                'arm': marm.group(1),
                'seed': int(mseed.group(1)),
                'run_dir': rpath,
                'run_name': run,
                'metrics': m,
            }
            rec.update(hist_stats(os.path.join(rpath, 'history.csv')))
            pcr, sizes, labels = per_class_recall(os.path.join(rpath, 'artifacts'))
            rec['per_class_recall'] = pcr
            rec['class_sizes'] = sizes
            rec['class_labels'] = labels
            records.append(rec)
    with open(OUT, 'w') as fh:
        json.dump(records, fh)
    n_by = {}
    for r in records:
        n_by[(r['dataset'], r['arm'])] = n_by.get((r['dataset'], r['arm']), 0) + 1
    print('records:', len(records))
    for k in sorted(n_by):
        print('  %-10s %-7s %d' % (k[0], k[1], n_by[k]))
    print('written:', OUT)


if __name__ == '__main__':
    main()
