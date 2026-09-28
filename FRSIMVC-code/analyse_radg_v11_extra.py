"""Supplementary dose-response analysis for the RADG-v1.1 four-arm batch.

Key question: is there ANY intervention strength (gamma) at which the local
re-allocation helps?  We regress the paired gain Delta_ACC = ACC_variant - ACC_F05
on the realised intervention magnitude (beta_std / Delta_A):
  * within-dataset slope over the 10 seeds (does more adaptation help within a dataset?)
  * across-dataset correlation (6 points)
If slopes are <= 0 everywhere, gamma = 0 (i.e. plain fixed fusion) is optimal and
Module I has no operating point.

Usage:  /home/ubuntu/miniconda3/envs/ICMVC/bin/python analyse_radg_v11_extra.py
"""
import json
import os

import numpy as np
from scipy import stats

ROOT = '/home/ubuntu/Tangzl/FRSIMVC/FRSIMVC/FRSIMVC-code/results/radg_v11'
OUTDIR = os.path.join(ROOT, '_analysis')
JSON_IN = os.path.join(OUTDIR, 'radg_v11_runs.json')
TXT_OUT = os.path.join(OUTDIR, 'dose_response.txt')

DATASETS = ['Augsburg', 'MDAS', 'MUUFL', 'Salinas', 'Trento', 'XuZhou']
ARMS = ['F05', 'F03', 'U_RADG', 'C_RADG']

_lines = []


def say(s=''):
    print(s)
    _lines.append(s)


def load():
    with open(JSON_IN) as fh:
        recs = json.load(fh)
    by = {}
    for r in recs:
        by.setdefault((r['dataset'], r['arm']), {})[r['seed']] = r
    return by


def mval(rec, key):
    if key in rec and rec[key] is not None:
        return float(rec[key])
    v = rec['metrics'].get(key)
    return float(v) if v is not None else None


def main():
    by = load()
    say('# RADG-v1.1 剂量—反应补析（是否存在可行的 γ 工作点）')
    say()

    # ---- dataset-level summary ----
    say('## D1 数据集层面：干预幅度 vs 配对增益')
    say()
    say('| dataset | GT cls | βstd(C) | βstd(U) | ΔA(C) | ΔA(U) | ΔACC(C−F05) | ΔACC(U−F05) | ΔACC(C−U) |')
    say('|---|---|---|---|---|---|---|---|---|')
    rows = []
    for ds in DATASETS:
        def mean(arm, key):
            vs = [mval(by[(ds, arm)][s], key) for s in range(10)]
            vs = [v for v in vs if v is not None]
            return float(np.mean(vs)) if vs else np.nan
        dc = np.array([mval(by[(ds, 'C_RADG')][s], 'ACC') - mval(by[(ds, 'F05')][s], 'ACC') for s in range(10)])
        du = np.array([mval(by[(ds, 'U_RADG')][s], 'ACC') - mval(by[(ds, 'F05')][s], 'ACC') for s in range(10)])
        dcu = np.array([mval(by[(ds, 'C_RADG')][s], 'ACC') - mval(by[(ds, 'U_RADG')][s], 'ACC') for s in range(10)])
        r = dict(ds=ds, gt=mean('F05', 'num_gt_classes'),
                 bsc=mean('C_RADG', 'beta_std'), bsu=mean('U_RADG', 'beta_std'),
                 dac=mean('C_RADG', 'graph_relative_frobenius_change'),
                 dau=mean('U_RADG', 'graph_relative_frobenius_change'),
                 dc=dc.mean(), du=du.mean(), dcu=dcu.mean())
        rows.append(r)
        say('| %s | %d | %.4f | %.4f | %.3f | %.3f | %+.4f | %+.4f | %+.4f |' % (
            ds, r['gt'], r['bsc'], r['bsu'], r['dac'], r['dau'], r['dc'], r['du'], r['dcu']))
    say()

    # across-dataset correlations
    def corr(xs, ys, nm):
        x = np.array(xs, dtype=float)
        y = np.array(ys, dtype=float)
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < 3:
            return
        r, p = stats.pearsonr(x[ok], y[ok])
        say('  - r(ΔACC(C−F05), %s) = %+.3f (p=%.3f, %s)' % (nm, r, p, '**' if p < 0.01 else ('*' if p < 0.05 else 'ns')))

    say('跨数据集（n=6）相关：')
    corr([r['bsc'] for r in rows], [r['dc'] for r in rows], 'β std(C)')
    corr([r['dac'] for r in rows], [r['dc'] for r in rows], 'Δ_A(C)')
    corr([r['gt'] for r in rows], [r['dc'] for r in rows], 'GT 类数')
    corr([r['bsu'] for r in rows], [r['du'] for r in rows], 'β std(U) → ΔACC(U−F05)')
    say()

    # ---- within-dataset dose-response slopes ----
    say('## D2 数据集内剂量—反应斜率（10 seeds，ΔACC ~ β_std / Δ_A）')
    say()
    say('| dataset | slope(ΔACC~βstd) | p | slope(ΔACC~ΔA) | p | 判读 |')
    say('|---|---|---|---|---|---|')
    slopes = []
    for ds in DATASETS:
        x1 = np.array([mval(by[(ds, 'C_RADG')][s], 'beta_std') for s in range(10)])
        x2 = np.array([mval(by[(ds, 'C_RADG')][s], 'graph_relative_frobenius_change') for s in range(10)])
        y = np.array([mval(by[(ds, 'C_RADG')][s], 'ACC') - mval(by[(ds, 'F05')][s], 'ACC') for s in range(10)])
        out = []
        for x in (x1, x2):
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() >= 4 and np.std(x[ok]) > 1e-9:
                lr = stats.linregress(x[ok], y[ok])
                out.append((lr.slope, lr.pvalue))
            else:
                out.append((np.nan, np.nan))
        slopes.append(out[0][0])
        verdict = 'γ=0 最优' if (out[0][0] is np.nan or out[0][0] <= 0) else '存在正向剂量区'
        say('| %s | %+.3f | %s | %+.3f | %s | %s |' % (
            ds, out[0][0], ('%.3f' % out[0][1]) if out[0][1] == out[0][1] else 'nan',
            out[1][0], ('%.3f' % out[1][1]) if out[1][1] == out[1][1] else 'nan', verdict))
    slopes = np.array([s for s in slopes if s == s])
    say()
    say('负斜率数据集 %d/%d；斜率均值 %+.3f' % (int((slopes <= 0).sum()), slopes.size, slopes.mean()))
    if slopes.size:
        t, p = stats.ttest_1samp(slopes, 0)
        say('  单样本 t 检验（斜率是否为 0）：t=%+.3f, p=%.3f %s' % (t, p, '**' if p < 0.01 else ('*' if p < 0.05 else 'ns')))
    say()

    # ---- best operating point check: per-seed win rate of C over F05 ----
    say('## D3 C_RADG 对 F05 的逐 seed 胜率（best ACC）')
    say()
    allw = 0
    alln = 0
    for ds in DATASETS:
        d = np.array([mval(by[(ds, 'C_RADG')][s], 'ACC') - mval(by[(ds, 'F05')][s], 'ACC') for s in range(10)])
        w = int((d > 0).sum())
        allw += w
        alln += d.size
        say('  - %-9s 胜 %d/10   mean %+.4f   min %+.4f   max %+.4f' % (ds, w, d.mean(), d.min(), d.max()))
    say()
    say('  合计胜率 %d/%d = %.1f%%（随机期望 50%%）' % (allw, alln, 100.0 * allw / alln))
    bt = stats.binomtest(allw, alln, 0.5) if hasattr(stats, 'binomtest') else stats.binom_test(allw, alln, 0.5)
    say('  二项检验 p = %.4g' % bt.pvalue)
    say()

    with open(TXT_OUT, 'w') as fh:
        fh.write('\n'.join(_lines) + '\n')
    print('written:', TXT_OUT)


if __name__ == '__main__':
    main()
