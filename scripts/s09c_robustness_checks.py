# %% [9c] Robustness checks + comfort-efficiency gap tests --------------------
# Run AFTER stage 9 in the same session (run_all.py) (reuses HM, FULL, QREF, CFG, faults, normals,
# fit_full, fit_pooled, prior_from, window_dev, pick_test, metrics, ar1, MIN_* constants).
#
#   C1  Model upgrade: extended design (+ T_ss x load interaction, + load^2) vs base design,
#       compared on out-of-time RMSE (paired). The better one is used below.
#   C2  Low-load artefact check: residual bias by within-substation load decile on normal
#       hours, for both designs. A flat profile means low-load fault windows are not biased.
#   C3  Fault excess, revised: model fitted WITHOUT fault windows AND without normal-event
#       windows (so both are out-of-sample, a fair comparison), 7-day windows before each
#       report, and LOAD-/WEATHER-MATCHED baselines (same load decile x 5 K outdoor bin of
#       the same substation's normal hours). Tests: faults vs normal events, sign split.
#   C4  Comfort-efficiency gap: do substations with poor cooling (high reference return
#       temperature) report more faults per year?
#   C5  (optional) RHO_CAP sensitivity of the pooling result.
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu, spearmanr, binomtest

import os
RUN_RHO_SENS = os.environ.get('PREDIST_RHO_SENS', '1') == '1'   # autocorrelation-cap sensitivity (slow)
COMFORT = {'no heat', 'not enough heat', 'no dhw'}
rng9c = np.random.default_rng(23)
BLUE, ORANGE, AQUA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8a85'

def style(ax, grid='y'):
    ax.grid(axis=grid, color='#ecebe6', lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

# ------------------------------------------------------------------ designs
def design_base(d, tag):
    X = np.column_stack([np.ones(len(d)), d['T_ss'].values - 50, (d['T_ps'] - d['T_ss']).values - 30,
                         np.log(d['Q'].values / QREF[tag]) + 1.0, d['T_out'].values - 5])
    return X

def design_ext(d, tag):
    Tss = d['T_ss'].values - 50
    L = np.log(d['Q'].values / QREF[tag]) + 1.0
    return np.column_stack([np.ones(len(d)), Tss, (d['T_ps'] - d['T_ss']).values - 30, L,
                            d['T_out'].values - 5, Tss * L / 10.0, L ** 2])

DESIGNS = {'base': design_base, 'extended': design_ext}

# Normal-event windows (first 7 d) are held out of every fit in 9c, like fault windows.
def heldout_normal_mask(H, key):
    m = np.zeros(len(H), dtype=bool)
    ne = normals[(normals.manufacturer == key[0]) & (normals.substation_id == key[1])]
    for s0 in ne['event_start'].dropna():
        m |= (H.index >= s0) & (H.index < s0 + pd.Timedelta(days=7))
    return m

FIT_MASK = {k: (H['ok'] & H['normal'] & ~heldout_normal_mask(H, k)) for k, H in HM.items()}

def build_full(dfn):
    """Full-history fits for every substation with a given design (sets global `design`)."""
    global design
    design = dfn
    F = {}
    for key, H in HM.items():
        d = H[FIT_MASK[key]]
        if len(d) < MIN_FULL_H or d['T_out'].std() < 1.0:
            continue
        f = fit_full(design(d, key[0]), d['T_pr'].values)
        devs, Vw = window_dev(d, key[0], f['b'])
        F[key] = {**f, 'cfg': CFG.get(key, 'unknown'), 'n_hours': len(d), 'devs': devs, 'Vw': Vw}
    return F

def pooled_full(F, key):
    d = HM[key][FIT_MASK[key]]
    mu, T, s0 = prior_from([v for k, v in F.items() if k != key], F[key]['cfg'])
    return fit_pooled(design(d, key[0]), d['T_pr'].values, mu, T, s0)

def eval_rmse(F, days=(7, None)):
    rows = []
    for key in F:
        d = HM[key][FIT_MASK[key]]
        tb = pick_test(d)
        if tb is None:
            continue
        ts, te = tb
        test = d[(d.index > ts) & (d.index <= te)]
        cutoff = ts - pd.Timedelta(days=GAP_DAYS)
        mu, T, s0 = prior_from([v for k, v in F.items() if k != key], F[key]['cfg'])
        for n in days:
            tr = d[d.index < cutoff] if n is None else d[(d.index < cutoff) & (d.index >= cutoff - pd.Timedelta(days=n))]
            if len(tr) < MIN_TRAIN_H:
                continue
            Xtr, ytr = design(tr, key[0]), tr['T_pr'].values
            Xte, yte = design(test, key[0]), test['T_pr'].values
            lab = 'full' if n is None else f'{n} d'
            for name, b in [('own', fit_own(Xtr, ytr)), ('pooled', fit_pooled(Xtr, ytr, mu, T, s0))]:
                rows.append(dict(manufacturer=key[0], substation_id=key[1], train=lab, method=name,
                                 **metrics(yte, Xte @ b)))
    return pd.DataFrame(rows)

def paired(ev, a, b, train):
    w = ev[ev.train == train].pivot_table(index=['manufacturer', 'substation_id'], columns='method', values='rmse')
    return w

# ------------------------------------------------------------ C1 model choice
FULLS, EVALS = {}, {}
for name, dfn in DESIGNS.items():
    FULLS[name] = build_full(dfn)
    EVALS[name] = eval_rmse(FULLS[name])
    print(f'{name}: {len(FULLS[name])} substations fitted')

c1 = []
for tr in ['7 d', 'full']:
    wb = paired(EVALS['base'], 'own', 'pooled', tr)['pooled']
    we = paired(EVALS['extended'], 'own', 'pooled', tr)['pooled']
    j = pd.concat([wb.rename('base'), we.rename('extended')], axis=1).dropna()
    p = wilcoxon(j['extended'], j['base']).pvalue if len(j) >= 10 else np.nan
    c1.append(dict(train=tr, n=len(j), base_median_rmse=j['base'].median(),
                   extended_median_rmse=j['extended'].median(),
                   median_diff_ext_minus_base=(j['extended'] - j['base']).median(),
                   extended_better_share=float((j['extended'] < j['base']).mean()), wilcoxon_p=p))
c1 = pd.DataFrame(c1)
c1.to_csv(OUT_DIR / 'hm9c_model_choice.csv', index=False)
print('\nC1 pooled out-of-time RMSE, base vs extended design [K]:')
print(c1.round(3).to_string(index=False))
full_row = c1[c1.train == 'full'].iloc[0]
CHOSEN = 'extended' if (full_row['median_diff_ext_minus_base'] < 0 and full_row['wilcoxon_p'] < 0.05) else 'base'
print(f'-> using the {CHOSEN} design for C2-C4')

# ------------------------------------------------------------ C2 low-load check
def residual_profile(F, dfn):
    global design
    design = dfn
    prof = []
    for key in F:
        H = HM[key]; d = H[FIT_MASK[key]]
        b = pooled_full(F, key)
        r = d['T_pr'].values - design(d, key[0]) @ b
        dec = pd.qcut(d['Q'].rank(method='first'), 10, labels=False) + 1
        prof.append(pd.Series(r, index=d.index).groupby(dec.values).median())
    P = pd.DataFrame(prof)                          # rows: substations, cols: deciles 1..10
    return P

fig, ax = plt.subplots(figsize=(6.2, 3.8))
c2 = {}
for (name, dfn), col, mk, off in [(('base', design_base), ORANGE, 's', -0.12),
                                  (('extended', design_ext), BLUE, 'o', 0.12)]:
    P = residual_profile(FULLS[name], dfn)
    med = P.median(); q1, q3 = P.quantile(0.25), P.quantile(0.75)
    c2[name] = med
    x = np.array(med.index, dtype=float)
    ax.errorbar(x + off, med.values, yerr=[med.values - q1.values, q3.values - med.values], fmt=mk, ms=6,
                lw=1.3, capsize=0, color=col, mec='white', mew=1.0, label=f'{name} design')
ax.axhline(0, color=GREY, lw=1)
ax.set_xticks(range(1, 11)); ax.set_xlabel('Heat-load decile within substation (1 = lowest load)')
ax.set_ylabel('Median residual $T_{p,return}$ [K]\n(across substations, IQR)')
ax.legend(frameon=False, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
style(ax); plt.tight_layout()
plt.savefig(FIG_DIR / 'hm9c_residual_by_load.png', dpi=300, bbox_inches='tight'); plt.show()
c2 = pd.DataFrame(c2); c2.index.name = 'load_decile'
c2.to_csv(OUT_DIR / 'hm9c_residual_by_load_decile.csv')
print('\nC2 median residual by load decile [K] (flat ~0 = no low-load bias):')
print(c2.round(2).T.to_string())

# ------------------------------------------------- C3 matched fault excess (chosen design)
design = DESIGNS[CHOSEN]
F = FULLS[CHOSEN]
TBINS = np.arange(-25, 41, 5)

def matched_excess(key, b, t0, t1):
    H = HM[key]
    valid = H[H['ok']]
    w = valid[(valid.index >= t0) & (valid.index < t1)]
    if len(w) < 12:
        return np.nan, np.nan, len(w)
    rw = w['T_pr'].values - design(w, key[0]) @ b
    ref = H[FIT_MASK[key]]
    rr = ref['T_pr'].values - design(ref, key[0]) @ b
    edges = np.unique(np.percentile(ref['Q'], np.linspace(0, 100, 11)))
    ld_ref = np.clip(np.searchsorted(edges, ref['Q'].values, side='right') - 1, 0, len(edges) - 2)
    ld_w = np.clip(np.searchsorted(edges, w['Q'].values, side='right') - 1, 0, len(edges) - 2)
    tb_ref, tb_w = np.digitize(ref['T_out'].values, TBINS), np.digitize(w['T_out'].values, TBINS)
    base = np.empty(len(w))
    for i in range(len(w)):
        m = (ld_ref == ld_w[i]) & (tb_ref == tb_w[i])
        if m.sum() < 10:
            m = ld_ref == ld_w[i]
        base[i] = np.median(rr[m]) if m.sum() else 0.0
    return float(np.mean(rw)), float(np.mean(rw - base)), len(w)

ex = []
for key in F:
    tag, sid = key
    b = pooled_full(F, key)
    for _, r in faults[(faults.manufacturer == tag) & (faults.substation_id == sid)].iterrows():
        rep = r['report_date']
        if pd.isna(rep):
            continue
        raw, mat, nh = matched_excess(key, b, rep - pd.Timedelta(days=7), rep)
        ex.append(dict(manufacturer=tag, substation_id=sid, kind='fault', label=r.get('fault_label'),
                       problem=str(r.get('problem_en')).strip().lower(), mp_class=r.get('mp_class'),
                       excess_raw_K=raw, excess_matched_K=mat, hours=nh))
    for s0 in normals[(normals.manufacturer == tag) & (normals.substation_id == sid)]['event_start'].dropna():
        raw, mat, nh = matched_excess(key, b, s0, s0 + pd.Timedelta(days=7))
        ex.append(dict(manufacturer=tag, substation_id=sid, kind='normal', label='normal event', problem='normal',
                       mp_class='n/a', excess_raw_K=raw, excess_matched_K=mat, hours=nh))
ex = pd.DataFrame(ex).dropna(subset=['excess_matched_K'])
ex['group'] = np.where(ex.kind == 'normal', 'normal event',
              np.where(ex.problem.isin(COMFORT), 'comfort fault (no heat / not enough heat / no DHW)',
                       'other fault (leakage / noise / other)'))
nrm = ex.loc[ex.kind == 'normal', 'excess_matched_K']
lo, hi = np.percentile(nrm, [5, 95])
ex['vs_normal_band'] = np.where(ex.excess_matched_K < lo, 'below', np.where(ex.excess_matched_K > hi, 'above', 'within'))
ex.to_csv(OUT_DIR / 'hm9c_fault_excess_matched.csv', index=False)

print(f'\nC3 normal events (held out): n={len(nrm)}, median {nrm.median():+.2f} K, 5-95 % band [{lo:+.2f}, {hi:+.2f}] K')
tab = ex.groupby('group').agg(n=('excess_matched_K', 'size'), median_K=('excess_matched_K', 'median'),
                              mean_K=('excess_matched_K', 'mean'))
tab = tab.join(pd.crosstab(ex.group, ex.vs_normal_band))
print(tab.round(2).to_string())
for g in tab.index:
    if g == 'normal event':
        continue
    v = ex.loc[ex.group == g, 'excess_matched_K']
    p_mw = mannwhitneyu(v, nrm, alternative='two-sided').pvalue
    out = ex[(ex.group == g) & (ex.vs_normal_band != 'within')]
    k_below = int((out.vs_normal_band == 'below').sum())
    p_bin = binomtest(k_below, len(out), 0.5).pvalue if len(out) else np.nan
    print(f'  {g}: vs normal Mann-Whitney p={p_mw:.4f}; outside band {len(out)} '
          f'(below {k_below}, above {len(out) - k_below}), sign test p={p_bin:.4f}')
lab_tab = (ex[ex.kind == 'fault'].groupby('label')['excess_matched_K']
           .agg(['count', 'median', 'mean']).sort_values('median'))
lab_tab.to_csv(OUT_DIR / 'hm9c_fault_excess_by_label.csv')

fig, ax = plt.subplots(figsize=(7.2, 3.4))
groups = ['normal event', 'comfort fault (no heat / not enough heat / no DHW)', 'other fault (leakage / noise / other)']
cols = [GREY, BLUE, AQUA]
ax.axvspan(lo, hi, color='#d4d3cd', alpha=0.5, lw=0, label='Normal events, 5-95 %')
ax.axvline(0, color=GREY, lw=1)
for i, (g, c) in enumerate(zip(groups, cols)):
    v = ex.loc[ex.group == g, 'excess_matched_K'].values
    if len(v) == 0:
        continue
    ax.scatter(v, np.full(len(v), i) + rng9c.uniform(-0.15, 0.15, len(v)), s=24, color=c,
               edgecolor='white', linewidth=0.8, zorder=3)
    ax.plot([np.median(v)] * 2, [i - 0.32, i + 0.32], color='#184f95', lw=2.2, zorder=4)
ax.set_yticks(range(len(groups))); ax.set_yticklabels(['Normal events', 'Comfort faults', 'Other faults'])
ax.set_xlabel('Load- and weather-matched excess $T_{p,return}$ [K] (7 d before report)')
ax.legend(frameon=False, fontsize=8, loc='lower right')
style(ax, 'x'); plt.tight_layout()
plt.savefig(FIG_DIR / 'hm9c_fault_excess_matched.png', dpi=300, bbox_inches='tight'); plt.show()

# ------------------------------------------------- C4 cooling quality vs complaint rate
REF_X = np.zeros(F[next(iter(F))]['b'].shape[0]); REF_X[0] = 1.0   # all regressors at reference
c4 = []
for key in F:
    H = HM[key]
    years = (H.index.max() - H.index.min()).days / 365.25
    if years < 0.5:
        continue
    n_rep = int(((faults.manufacturer == key[0]) & (faults.substation_id == key[1])).sum())
    c4.append(dict(manufacturer=key[0], substation_id=key[1], configuration=F[key]['cfg'], years=years,
                   ref_return_C=float(REF_X @ pooled_full(F, key)), reports=n_rep, reports_per_year=n_rep / years))
c4 = pd.DataFrame(c4)
c4['ref_return_rank'] = c4.groupby('manufacturer')['ref_return_C'].rank(pct=True)   # within-manufacturer
c4['report_rank'] = c4.groupby('manufacturer')['reports_per_year'].rank(pct=True)
c4.to_csv(OUT_DIR / 'hm9c_cooling_vs_complaints.csv', index=False)

print('\nC4 reference return temperature vs complaint rate (Spearman):')
for m, d in list(c4.groupby('manufacturer')) + [('pooled (within-manufacturer ranks)', c4)]:
    a, bcol = ('ref_return_rank', 'report_rank') if m.startswith('pooled') else ('ref_return_C', 'reports_per_year')
    rho, p = spearmanr(d[a], d[bcol])
    boots = []
    for _ in range(2000):
        i = rng9c.integers(0, len(d), len(d))
        boots.append(spearmanr(d[a].values[i], d[bcol].values[i])[0])
    print(f'  {m}: n={len(d)}, rho={rho:+.2f} [{np.nanpercentile(boots, 2.5):+.2f}, '
          f'{np.nanpercentile(boots, 97.5):+.2f}], p={p:.3f}')
q75 = c4.groupby('manufacturer')['ref_return_C'].transform(lambda s: s.quantile(0.75))
poor, rest = c4[c4.ref_return_C >= q75], c4[c4.ref_return_C < q75]
print(f'  poorest-cooling quartile: median {poor.reports_per_year.median():.2f} reports/yr (n={len(poor)}) vs '
      f'others {rest.reports_per_year.median():.2f} (n={len(rest)}), Mann-Whitney p='
      f'{mannwhitneyu(poor.reports_per_year, rest.reports_per_year).pvalue:.3f}')
print(f'  spread of reference return temperature: {c4.ref_return_C.min():.1f} - {c4.ref_return_C.max():.1f} degC '
      f'(IQR {c4.ref_return_C.quantile(.25):.1f} - {c4.ref_return_C.quantile(.75):.1f})')

fig, ax = plt.subplots(figsize=(6.0, 4.0))
for m, c, mk in zip(sorted(c4.manufacturer.unique()), [BLUE, ORANGE], ['o', 's']):
    d = c4[c4.manufacturer == m]
    ax.scatter(d.ref_return_C, d.reports_per_year, s=34, color=c, marker=mk, edgecolor='white',
               linewidth=0.8, label=m, zorder=3)
ax.set_xlabel('Reference primary return temperature [°C]\n(model prediction at common operating point)')
ax.set_ylabel('Customer fault reports per year')
ax.legend(frameon=False, fontsize=8, loc='upper left')
style(ax, 'both'); plt.tight_layout()
plt.savefig(FIG_DIR / 'hm9c_cooling_vs_complaints.png', dpi=300, bbox_inches='tight'); plt.show()

# ------------------------------------------------------------ C5 RHO_CAP sensitivity
if RUN_RHO_SENS:
    base_cap = RHO_CAP
    sens = []
    for cap in [0.5, 0.8, 0.9]:
        RHO_CAP = cap
        Fc = build_full(DESIGNS[CHOSEN])
        evc = eval_rmse(Fc)
        for tr in ['7 d', 'full']:
            w = paired(evc, 'own', 'pooled', tr).dropna()
            diff = w['pooled'] - w['own']
            sens.append(dict(rho_cap=cap, train=tr, n=len(w), median_pooled_minus_own_K=diff.median(),
                             pooled_better_share=float((diff < 0).mean()),
                             wilcoxon_p=wilcoxon(w['pooled'], w['own']).pvalue if len(w) >= 10 else np.nan))
    RHO_CAP = base_cap
    design = DESIGNS[CHOSEN]
    sens = pd.DataFrame(sens)
    sens.to_csv(OUT_DIR / 'hm9c_rho_cap_sensitivity.csv', index=False)
    print('\nC5 RHO_CAP sensitivity (pooled - own RMSE):')
    print(sens.round(3).to_string(index=False))

print('\nSaved hm9c_*.csv and figures hm9c_*.png in', OUT_DIR)
