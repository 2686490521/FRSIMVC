"""Analyse the 240-run RADG-v1.1 (outline v7) four-arm batch.

Usage (server):
    /home/ubuntu/miniconda3/envs/ICMVC/bin/python analyse_radg_v11.py

Inputs : results/radg_v11/_analysis/radg_v11_runs.json  (from collect_radg_v11.py)
Outputs: results/radg_v11/_analysis/report_radg_v11.md
         results/radg_v11/_analysis/fig_acc_by_arm.png
         results/radg_v11/_analysis/fig_paired_delta.png
         stdout = same tables in text form
"""
import json
import os
from collections import OrderedDict, defaultdict

import numpy as np
from scipy import stats

ROOT = '/home/ubuntu/Tangzl/FRSIMVC/FRSIMVC/FRSIMVC-code/results/radg_v11'
OUTDIR = os.path.join(ROOT, '_analysis')
JSON_IN = os.path.join(OUTDIR, 'radg_v11_runs.json')
MD_OUT = os.path.join(OUTDIR, 'report_radg_v11.md')

ARMS = ['F05', 'F03', 'U_RADG', 'C_RADG']
DATASETS = ['Augsburg', 'MDAS', 'MUUFL', 'Salinas', 'Trento', 'XuZhou']
METRICS = ['ACC', 'NMI', 'ARI', 'Kappa', 'PURITY']
ACC_STATS = ['ACC', 'ACC_median', 'ACC_tail_mean', 'ACC_final', 'ACC_p25', 'ACC_std']
CONTRASTS = [('C_RADG', 'F05'), ('F05', 'F03'), ('C_RADG', 'U_RADG'), ('U_RADG', 'F05')]

_lines = []


def say(s=''):
    print(s)
    _lines.append(s)


def fmt(v, nd=4):
    if v is None:
        return '  n/a '
    if isinstance(v, (int, np.integer)):
        return '%d' % v
    return ('%.' + str(nd) + 'f') % v


def load():
    with open(JSON_IN) as fh:
        recs = json.load(fh)
    by = defaultdict(dict)  # (ds, arm) -> seed -> rec
    for r in recs:
        by[(r['dataset'], r['arm'])][r['seed']] = r
    return recs, by


def mval(rec, key):
    if key in rec and rec[key] is not None:
        return float(rec[key])
    return rec['metrics'].get(key)


def paired(a_recs, b_recs, key):
    """Return arrays of paired values (a, b) over shared seeds."""
    seeds = sorted(set(a_recs) & set(b_recs))
    a = []
    b = []
    for s in seeds:
        va, vb = mval(a_recs[s], key), mval(b_recs[s], key)
        if va is None or vb is None:
            continue
        a.append(va)
        b.append(vb)
    return np.array(a), np.array(b), seeds


def wilcox(a, b):
    d = a - b
    d = d[~np.isnan(d)]
    if d.size < 5:
        return np.nan, np.mean(d) if d.size else np.nan, d.size, int((d > 0).sum())
    if np.allclose(d, 0):
        return 1.0, 0.0, d.size, 0
    try:
        _, p = stats.wilcoxon(d, zero_method='wilcox', alternative='two-sided')
    except Exception:
        p = np.nan
    return p, float(np.mean(d)), d.size, int((d > 0).sum())


def stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return '  '
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '* '
    return 'ns'


def main():
    recs, by = load()
    say('# RADG-v1.1 (centered) four-arm batch — 分析结果')
    say()
    say('- runs: %d   (datasets %d × arms %d × seeds %d)' % (len(recs), len(DATASETS), len(ARMS), 10))
    say('- 协议: random@0.3, ε=0.1, α=0, A3+dual+balanced, 300 ep, ACC=best-round')
    say()

    # ---------- 0. pairing sanity ----------
    say('## T0 配对一致性检查（节点集必须逐 seed 相同）')
    say()
    say('| dataset | seed | F05 nodes | F03 | U_RADG | C_RADG | GT classes |')
    say('|---|---|---|---|---|---|---|')
    bad = 0
    for ds in DATASETS:
        for s in range(10):
            vals = []
            for arm in ARMS:
                r = by[(ds, arm)].get(s)
                vals.append(mval(r, 'num_active_nodes') if r else None)
            gt = mval(by[(ds, 'F05')][s], 'num_gt_classes')
            ok = all(v == vals[0] for v in vals if v is not None)
            if not ok:
                bad += 1
            say('| %s | %d | %s | %s | %s | %s | %s |' % (
                ds, s, fmt(vals[0], 0), fmt(vals[1], 0), fmt(vals[2], 0), fmt(vals[3], 0), fmt(gt, 0)))
    say()
    say('**不一致的 (dataset,seed) 单元数 = %d** → %s' % (bad, '逐 seed 配对有效' if bad == 0 else '⚠️ 配对失效'))
    say()

    # ---------- T1 main table ----------
    say('## T1 主表：Best ACC（mean±std, 10 seeds）+ 稳定性统计量')
    say()
    hdr = '| dataset | arm | best ACC | std | median | tail30 | final | p25 | round std | NMI | ARI | Kappa | PURITY |'
    say(hdr)
    say('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    t1 = {}
    for ds in DATASETS:
        for arm in ARMS:
            rs = [by[(ds, arm)][s] for s in range(10)]
            acc = np.array([mval(r, 'ACC') for r in rs], dtype=float)
            row = {
                'best': acc.mean(), 'std': acc.std(ddof=1),
                'median': np.mean([mval(r, 'ACC_median') for r in rs]),
                'tail': np.mean([mval(r, 'ACC_tail_mean') for r in rs]),
                'final': np.mean([mval(r, 'ACC_final') for r in rs]),
                'p25': np.mean([mval(r, 'ACC_p25') for r in rs]),
                'rstd': np.mean([mval(r, 'ACC_std') for r in rs]),
            }
            for m in ['NMI', 'ARI', 'Kappa', 'PURITY']:
                row[m] = np.mean([mval(r, m) for r in rs])
            t1[(ds, arm)] = row
            say('| %s | %s | **%s** | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
                ds, arm, fmt(row['best']), fmt(row['std']), fmt(row['median']), fmt(row['tail']),
                fmt(row['final']), fmt(row['p25']), fmt(row['rstd']),
                fmt(row['NMI']), fmt(row['ARI']), fmt(row['Kappa']), fmt(row['PURITY'])))
    say()

    # ---------- T2 paired contrasts ----------
    say('## T2 逐 seed 配对比较（Wilcoxon signed-rank, n=10/数据集；pooled n=60）')
    say()
    for a, b in CONTRASTS:
        say('### %s − %s' % (a, b))
        say()
        say('| dataset | stat | mean Δ | wins | p | sig |')
        say('|---|---|---|---|---|---|')
        pool_d = []
        for key in ACC_STATS:
            label = {'ACC': 'best ACC', 'ACC_median': 'median', 'ACC_tail_mean': 'tail30',
                     'ACC_final': 'final', 'ACC_p25': 'p25', 'ACC_std': 'round std'}[key]
            for ds in DATASETS:
                av, bv, _ = paired(by[(ds, a)], by[(ds, b)], key)
                p, md, n, w = wilcox(av, bv)
                if key == 'ACC':
                    pool_d.append(av - bv)
                say('| %s | %s | %+.4f | %d/%d | %s | %s |' % (
                    ds, label, md, w, n, ('%.4f' % p) if p == p else 'nan', stars(p)))
        if pool_d:
            d = np.concatenate(pool_d)
            try:
                _, pp = stats.wilcoxon(d, zero_method='wilcox', alternative='two-sided')
            except Exception:
                pp = np.nan
            say('| **pooled (6×10)** | best ACC | %+.4f | %d/%d | %s | %s |' % (
                d.mean(), int((d > 0).sum()), d.size, ('%.4g' % pp) if pp == pp else 'nan', stars(pp)))
        say()

    # ---------- T3 mechanism check ----------
    say('## T3 机制核对：β 分布 / 中心化误差 / Δ_A')
    say()
    say('| dataset | arm | E[β] | |E[β]−β0| | β std | p10 | p50 | p90 | min | max | recentre it | Δ_A | edges(radg/fixed) | λ2(radg/fixed) |')
    say('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for ds in DATASETS:
        for arm in ARMS:
            rs = [by[(ds, arm)][s] for s in range(10)]
            def avg(k, src='metrics'):
                vs = [mval(r, k) for r in rs]
                vs = [v for v in vs if v is not None]
                return np.mean(vs) if vs else None
            base = avg('radg_base_beta')
            bm = avg('beta_mean')
            err = avg('beta_mean_error_from_base')
            if err is None and bm is not None and base is not None:
                err = abs(bm - base)
            say('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
                ds, arm, fmt(bm), fmt(err), fmt(avg('beta_std')), fmt(avg('beta_p10')),
                fmt(avg('beta_p50')), fmt(avg('beta_p90')), fmt(avg('beta_min_actual')),
                fmt(avg('beta_max_actual')), fmt(avg('recentre_iterations'), 2),
                fmt(avg('graph_relative_frobenius_change'), 3),
                '%s/%s' % (fmt(avg('radg_undirected_edges'), 0), fmt(avg('fixed_undirected_edges'), 0)),
                '%s/%s' % (fmt(avg('radg_algebraic_connectivity'), 4), fmt(avg('fixed_algebraic_connectivity'), 4))))
    say()

    # ---------- T4 correlations ----------
    say('## T4 机制相关性')
    say()
    say('| dataset | corr(r_i, β_node) | corr(r_i, neigh_missing) | corr(r_i, feat_edge_loss) | r mean | r std | u mean |')
    say('|---|---|---|---|---|---|---|')
    for ds in DATASETS:
        rs = [by[(ds, 'C_RADG')][s] for s in range(10)]
        def avg(k):
            vs = [mval(r, k) for r in rs]
            vs = [v for v in vs if v is not None]
            return np.mean(vs) if vs else None
        say('| %s | %s | %s | %s | %s | %s | %s |' % (
            ds, fmt(avg('corr_r_beta_node'), 3), fmt(avg('corr_r_neighbour_missing'), 3),
            fmt(avg('corr_r_feature_edge_loss'), 3), fmt(avg('observed_ratio_mean'), 3),
            fmt(avg('observed_ratio_std'), 3), fmt(avg('edge_uncertainty_mean'), 3)))
    say()

    # intervention magnitude vs gain, across the 60 C_RADG runs
    xs, ys = [], []
    for ds in DATASETS:
        for s in range(10):
            c = by[(ds, 'C_RADG')][s]
            f = by[(ds, 'F05')][s]
            dacc = mval(c, 'ACC') - mval(f, 'ACC')
            xs.append((mval(c, 'beta_std'), mval(c, 'graph_relative_frobenius_change'),
                       mval(c, 'observed_ratio_mean')))
            ys.append(dacc)
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    say('干预幅度 vs 增益（60 个 C_RADG run，跨数据集）:')
    for i, nm in enumerate(['β std', 'Δ_A', 'observed_ratio_mean']):
        r, p = stats.pearsonr(xs[:, i], ys)
        say('  - r(ΔACC, %s) = %+.3f  (p=%.3f, %s)' % (nm, r, p, stars(p)))
    # within C_RADG: beta_std vs ACC level
    acc_c = np.array([mval(by[(ds, 'C_RADG')][s], 'ACC') for ds in DATASETS for s in range(10)])
    r, p = stats.pearsonr(xs[:, 0], acc_c)
    say('  - r(ACC_C, β std) = %+.3f (p=%.3f)' % (r, p))
    say()

    # ---------- T5 per-class ----------
    say('## T5 逐类 best-matched recall：C_RADG − F05（配对，n=10）')
    say()
    for ds in DATASETS:
        r0 = by[(ds, 'C_RADG')][0]
        labels = r0.get('class_labels')
        say('### %s (GT labels: %s)' % (ds, labels))
        say()
        say('| class | size | F05 | C_RADG | U_RADG | Δ(C−F05) | wins | p | sig |')
        say('|---|---|---|---|---|---|---|---|---|')
        n_cls = len(r0.get('per_class_recall') or {})
        for i in range(n_cls):
            key = str(i)
            a = []
            b = []
            for s in range(10):
                va = (by[(ds, 'C_RADG')][s].get('per_class_recall') or {}).get(key)
                vb = (by[(ds, 'F05')][s].get('per_class_recall') or {}).get(key)
                if va is None or vb is None:
                    continue
                a.append(va)
                b.append(vb)
            if not a:
                continue
            a = np.array(a)
            b = np.array(b)
            p, md, n, w = wilcox(a, b)
            fu = np.nanmean([(by[(ds, 'U_RADG')][s].get('per_class_recall') or {}).get(key)
                             for s in range(10)])
            size = (r0.get('class_sizes') or [None] * n_cls)[i]
            lbl = labels[i] if labels else i
            say('| %s | %s | %s | %s | %s | %+.4f | %d/%d | %s | %s |' % (
                lbl, fmt(size, 0), fmt(np.nanmean(b)), fmt(np.nanmean(a)), fmt(fu), md, w, n,
                ('%.4f' % p) if p == p else 'nan', stars(p)))
        say()

    # ---------- T6 cluster concentration ----------
    say('## T6 簇集中度（非 KPI）')
    say()
    say('| dataset | arm | p_max | H_clu | K_eff | pred clusters | GT classes |')
    say('|---|---|---|---|---|---|---|')
    for ds in DATASETS:
        for arm in ARMS:
            rs = [by[(ds, arm)][s] for s in range(10)]
            def avg(k):
                vs = [mval(r, k) for r in rs]
                vs = [v for v in vs if v is not None]
                return np.mean(vs) if vs else None
            say('| %s | %s | %s | %s | %s | %s | %s |' % (
                ds, arm, fmt(avg('cluster_max_ratio'), 3), fmt(avg('cluster_entropy'), 3),
                fmt(avg('effective_num_clusters'), 2), fmt(avg('num_predicted_clusters'), 1),
                fmt(avg('num_gt_classes'), 1)))
    say()

    # ---------- T7 verdict ----------
    say('## T7 验收（大纲 §24 / §32）')
    say()
    say('| dataset | C−F05 (best) | p | C−F05 (median) | C−F05 (tail30) | F05−F03 | C−U | 判定 |')
    say('|---|---|---|---|---|---|---|---|')
    verdict_rows = []
    for ds in DATASETS:
        av, bv, _ = paired(by[(ds, 'C_RADG')], by[(ds, 'F05')], 'ACC')
        p1, d1, _, w1 = wilcox(av, bv)
        _, d1m, _, _ = wilcox(*paired(by[(ds, 'C_RADG')], by[(ds, 'F05')], 'ACC_median')[:2])
        _, d1t, _, _ = wilcox(*paired(by[(ds, 'C_RADG')], by[(ds, 'F05')], 'ACC_tail_mean')[:2])
        _, d23, _, _ = wilcox(*paired(by[(ds, 'F05')], by[(ds, 'F03')], 'ACC')[:2])
        _, dcu, _, _ = wilcox(*paired(by[(ds, 'C_RADG')], by[(ds, 'U_RADG')], 'ACC')[:2])
        v = 'A(正向)' if (d1 > 0.005 and p1 < 0.05) else ('C(负向)' if d1 < -0.005 else 'B(持平)')
        verdict_rows.append((ds, d1, p1, v))
        say('| %s | %+.4f | %s | %+.4f | %+.4f | %+.4f | %+.4f | %s |' % (
            ds, d1, ('%.4f' % p1) if p1 == p1 else 'nan', d1m, d1t, d23, dcu, v))
    say()
    pos = [r for r in verdict_rows if r[1] > 0]
    sig_pos = [r for r in verdict_rows if r[1] > 0.005 and r[2] < 0.05]
    sig_neg = [r for r in verdict_rows if r[1] < -0.005 and r[2] < 0.05]
    say('汇总：正向 %d/6，显著正向(Δ>+0.005 & p<0.05) %d/6，显著负向 %d/6' % (
        len(pos), len(sig_pos), len(sig_neg)))
    say('显著正向的数据集：%s' % (', '.join('%s(%+.4f)' % (r[0], r[1]) for r in sig_pos) or '无'))
    say('显著负向的数据集：%s' % (', '.join('%s(%+.4f)' % (r[0], r[1]) for r in sig_neg) or '无'))
    say()

    with open(MD_OUT, 'w') as fh:
        fh.write('\n'.join(_lines) + '\n')
    print('written:', MD_OUT)


if __name__ == '__main__':
    main()
