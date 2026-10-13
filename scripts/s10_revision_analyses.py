# %% [10] Revision analyses requested in the pre-submission assessment --------------
# Run AFTER stages 9, 9c, 9d and 9e in the same session (run_all.py). Reuses:
#   MANUF, HM, FIT_MASK, FULLS, CHOSEN, DESIGNS, CFG, QREF, faults, normals, ex, c4, en, OPS,
#   pick_test, prior_from, fit_own, fit_pooled, fit_full, pooled_full, metrics, ar1,
#   GAP_DAYS, MIN_TRAIN_H, MIN_FULL_H, OUT_DIR, FIG_DIR, rho_water, cp_water
# Every block writes one CSV named hm10_<block>.csv into OUT_DIR. Upload them all afterwards.
#
#   R1  Substations excluded from modelling (93 -> 81) and why
#   R2  Hours retained per configuration, DHW-charging detection route
#   R3  Collinearity: VIF and condition number of the regressors
#   R4  Data efficiency extended: random-intercept baseline, means / trimmed means,
#       BCa CIs, rank-biserial effect sizes, recency control (oldest 7 d of the 90-d window)
#   R5  Prior hyper-parameters: NU0, KAPPA_CFG, number of iterations (convergence)
#   R6  Residual diagnostics: AR(1) rho, Durbin-Watson, Ljung-Box, QQ/ACF figure
#   R7  Per-manufacturer priors vs pooled priors (7 d)
#   R8  Fault signature statistics: CIs for proportions, one-sided sign test, BH-FDR by label
#   R9  Complaints: partial Spearman and negative-binomial GLM with exposure and size
#   R10 Excess volume: Gini / top-k shares, load-proportional gap, energy-balance check,
#       annualisation coverage, indicative cost range
import warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
warnings.filterwarnings('ignore')
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.proportion import confint_proportions_2indep, proportion_confint
except ImportError:
    raise ImportError('statsmodels is required: pip install -r requirements.txt')

missing = [v for v in ('DESIGNS', 'FULLS', 'CHOSEN', 'HM', 'FIT_MASK', 'ex', 'c4', 'en', 'OPS') if v not in globals()]
if missing:
    raise RuntimeError(f'Stage 10 needs the variables from stages 9, 9c, 9d and 9e (missing: {missing}). '
                       'Run the pipeline with run_all.py so that all stages share one session.')
design = DESIGNS[CHOSEN]
F = FULLS[CHOSEN]
rng10 = np.random.default_rng(101)
KEYS = sorted(F)
NU0_BASE, KAPPA_BASE = NU0, KAPPA_CFG
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 9, 'axes.grid': False,
                     'axes.spines.top': False, 'axes.spines.right': False})

def save(df, name):
    df.to_csv(OUT_DIR / f'hm10_{name}.csv', index=False)
    print(f'\n=== {name} ===')
    print(df.round(3).to_string(index=False) if len(df) < 40 else df.head(40).round(3).to_string(index=False))

# ------------------------------------------------------------------ R1 exclusions
r1 = []
for tag, m in MANUF.items():
    for sid in sorted(m['ops']):
        key = (tag, sid)
        if key in F:
            reason = 'modelled'
        elif key not in HM:
            reason = 'missing model signal (T_ps, T_pr, Q, T_ss or T_o)'
        else:
            n = int(FIT_MASK[key].sum())
            reason = (f'insufficient clean hours ({n} < {MIN_FULL_H})' if n < MIN_FULL_H
                      else 'outdoor temperature range too narrow')
        H = HM.get(key)
        r1.append(dict(manufacturer=tag, substation_id=sid, configuration=CFG.get(key, 'unknown'),
                       record_years=((H.index.max() - H.index.min()).days / 365.25) if H is not None else np.nan,
                       fault_reports=int(((faults.manufacturer == tag) & (faults.substation_id == sid)).sum()),
                       status=reason))
r1 = pd.DataFrame(r1)
save(r1[r1.status != 'modelled'], 'R1_excluded_substations')
print(r1.groupby(['manufacturer', r1.status == 'modelled']).size())

# ------------------------------------------------------------ R2 retained hours / DHW
r2 = []
for key in KEYS:
    H = HM[key]
    valid_sig = H[['T_ps', 'T_pr', 'Q', 'T_ss', 'T_out']].notna().all(axis=1)
    charging = ((H['T_up'].diff() > 1.5) | (H['T_lo'].diff() > 1.5) | (H['v3'] > 0.5)).fillna(False)
    r2.append(dict(manufacturer=key[0], substation_id=key[1], configuration=F[key]['cfg'],
                   hours_with_signals=int(valid_sig.sum()), hours_fit=int(FIT_MASK[key].sum()),
                   share_retained=FIT_MASK[key].sum() / max(valid_sig.sum(), 1),
                   share_dhw_charging=float(charging[valid_sig].mean()),
                   dhw_route=('valve status' if H['v3'].notna().any() else
                              'storage-temperature proxy' if (H['T_up'].notna().any() or H['T_lo'].notna().any())
                              else 'none (no DHW signals)')))
r2 = pd.DataFrame(r2)
r2.to_csv(OUT_DIR / 'hm10_R2_hours_per_substation.csv', index=False)
save(r2.groupby('configuration').agg(n=('substation_id', 'size'), hours_fit_median=('hours_fit', 'median'),
                                     retained_median=('share_retained', 'median'),
                                     dhw_charging_median=('share_dhw_charging', 'median')).reset_index(),
     'R2_hours_by_configuration')
save(r2.groupby(['configuration', 'dhw_route']).size().rename('n').reset_index(), 'R2_dhw_route')

# ------------------------------------------------------------------- R3 collinearity
r3 = []
for key in KEYS:
    d = HM[key][FIT_MASK[key]]
    X = DESIGNS['base'](d, key[0])[:, 1:]                       # main-effect regressors
    Z = (X - X.mean(0)) / X.std(0)
    try:
        vif = np.diag(np.linalg.inv(np.corrcoef(Z.T)))
    except np.linalg.LinAlgError:
        vif = np.full(Z.shape[1], np.nan)
    Xe = design(d, key[0]); Xs = Xe / np.linalg.norm(Xe, axis=0)
    r3.append(dict(manufacturer=key[0], substation_id=key[1], vif_T_ss=vif[0], vif_approach=vif[1],
                   vif_load=vif[2], vif_T_out=vif[3], cond_number_ext=np.linalg.cond(Xs),
                   corr_Tss_Tout=np.corrcoef(X[:, 0], X[:, 3])[0, 1]))
r3 = pd.DataFrame(r3)
r3.to_csv(OUT_DIR / 'hm10_R3_collinearity_per_substation.csv', index=False)
save(r3.drop(columns=['manufacturer', 'substation_id']).describe(percentiles=[.25, .5, .75, .9]).T
     .reset_index().rename(columns={'index': 'quantity'}), 'R3_collinearity_summary')

# ------------------------------------------------------- R4 data efficiency extended
def random_intercept(Xtr, ytr, mu):
    """Fleet slopes (configuration prior mean), intercept estimated from the target's own data."""
    b = mu.copy()
    b[0] = float(np.mean(ytr - Xtr[:, 1:] @ mu[1:]))
    return b

def eval_histories(F, prior_fn=None, days=(7, 14, 30, 90, None), extra=True, iters=4):
    rows = []
    for key in F:
        d = HM[key][FIT_MASK[key]]
        tb = pick_test(d)
        if tb is None:
            continue
        ts, te = tb
        test = d[(d.index > ts) & (d.index <= te)]
        cutoff = ts - pd.Timedelta(days=GAP_DAYS)
        mu, T, s0 = (prior_fn or (lambda k: prior_from([v for kk, v in F.items() if kk != k], F[k]['cfg'])))(key)
        Xte, yte = design(test, key[0]), test['T_pr'].values
        wins = [(('full' if n is None else f'{n} d'),
                 d[d.index < cutoff] if n is None else d[(d.index < cutoff) & (d.index >= cutoff - pd.Timedelta(days=n))])
                for n in days]
        if extra:   # recency control: the OLDEST 7 d of the 90-d window (same amount of data as '7 d')
            wins.append(('7 d, 83 d earlier', d[(d.index < cutoff - pd.Timedelta(days=83)) &
                                                 (d.index >= cutoff - pd.Timedelta(days=90))]))
        for lab, tr in wins:
            if len(tr) < MIN_TRAIN_H:
                continue
            Xtr, ytr = design(tr, key[0]), tr['T_pr'].values
            ests = [('own', fit_own(Xtr, ytr)), ('fleet', mu),
                    ('pooled', fit_pooled(Xtr, ytr, mu, T, s0, iters=iters))]
            if extra:
                ests.append(('random_intercept', random_intercept(Xtr, ytr, mu)))
            for name, b in ests:
                rows.append(dict(manufacturer=key[0], substation_id=key[1], train=lab, method=name,
                                 train_hours=len(tr), **metrics(yte, Xte @ b)))
    return pd.DataFrame(rows)

def bca(x, stat=np.median):
    x = np.asarray(x)[np.isfinite(x)]
    if len(x) < 5 or np.ptp(x) == 0:
        return np.nan, np.nan
    r = stats.bootstrap((x,), stat, n_resamples=2000, method='BCa', random_state=11)
    return r.confidence_interval.low, r.confidence_interval.high

def rank_biserial(diff):
    diff = np.asarray(diff); diff = diff[diff != 0]
    r = stats.rankdata(np.abs(diff))
    return (r[diff > 0].sum() - r[diff < 0].sum()) / r.sum()

ev10 = eval_histories(F)
ev10.to_csv(OUT_DIR / 'hm10_R4_data_efficiency_long.csv', index=False)
r4 = []
for lab in ['7 d', '14 d', '30 d', '90 d', 'full', '7 d, 83 d earlier']:
    w = ev10[ev10.train == lab].pivot_table(index=['manufacturer', 'substation_id'], columns='method',
                                            values='rmse').dropna()
    if len(w) < 5:
        continue
    for m in ['own', 'fleet', 'random_intercept', 'pooled']:
        lo, hi = bca(w[m])
        r4.append(dict(train=lab, method=m, n=len(w), median=w[m].median(), bca_lo=lo, bca_hi=hi,
                       mean=w[m].mean(), trimmed_mean_10=stats.trim_mean(w[m], 0.1)))
    for a, b in [('pooled', 'own'), ('pooled', 'random_intercept'), ('random_intercept', 'own')]:
        diff = w[a] - w[b]
        lo, hi = bca(diff)
        r4.append(dict(train=lab, method=f'{a} - {b}', n=len(w), median=diff.median(), bca_lo=lo, bca_hi=hi,
                       mean=diff.mean(), trimmed_mean_10=stats.trim_mean(diff, 0.1),
                       better_share=float((diff < 0).mean()), wilcoxon_p=stats.wilcoxon(w[a], w[b]).pvalue,
                       rank_biserial=rank_biserial(diff)))
save(pd.DataFrame(r4), 'R4_data_efficiency_extended')

# ------------------------------------------------------- R5 hyper-parameter sensitivity
r5 = []
for nu0 in [5, 10, 20, 50]:
    for kap in [1, 3, 10]:
        NU0, KAPPA_CFG = nu0, kap
        e = eval_histories(F, days=(7,), extra=False)
        w = e.pivot_table(index=['manufacturer', 'substation_id'], columns='method', values='rmse').dropna()
        diff = w['pooled'] - w['own']
        r5.append(dict(NU0=nu0, KAPPA_CFG=kap, n=len(w), pooled_median=w['pooled'].median(),
                       pooled_minus_own_median=diff.median(), better_share=float((diff < 0).mean()),
                       wilcoxon_p=stats.wilcoxon(w['pooled'], w['own']).pvalue))
NU0, KAPPA_CFG = NU0_BASE, KAPPA_BASE
save(pd.DataFrame(r5), 'R5_prior_sensitivity_7d')

conv = []                                           # convergence of the iteration in eq. (5)
for key in KEYS:
    d = HM[key][FIT_MASK[key]]
    tb = pick_test(d)
    if tb is None:
        continue
    cutoff = tb[0] - pd.Timedelta(days=GAP_DAYS)
    tr = d[(d.index < cutoff) & (d.index >= cutoff - pd.Timedelta(days=7))]
    if len(tr) < MIN_TRAIN_H:
        continue
    mu, T, s0 = prior_from([v for k, v in F.items() if k != key], F[key]['cfg'])
    Xtr, ytr = design(tr, key[0]), tr['T_pr'].values
    b4, b20 = fit_pooled(Xtr, ytr, mu, T, s0, iters=4), fit_pooled(Xtr, ytr, mu, T, s0, iters=20)
    x0 = np.zeros_like(b4); x0[0] = 1.0
    conv.append(dict(manufacturer=key[0], substation_id=key[1], max_abs_coef_change=np.max(np.abs(b20 - b4)),
                     ref_return_change_K=abs(x0 @ (b20 - b4))))
conv = pd.DataFrame(conv)
save(conv[['max_abs_coef_change', 'ref_return_change_K']].describe(percentiles=[.5, .9, .99]).T.reset_index(),
     'R5_iteration_convergence_4_vs_20')

# ------------------------------------------------------------ R6 residual diagnostics
r6, acfs, qq = [], [], []
for key in KEYS:
    d = HM[key][FIT_MASK[key]]
    b = pooled_full(F, key)
    r = pd.Series(d['T_pr'].values - design(d, key[0]) @ b, index=d.index)
    rh = r.asfreq('1h')                                          # gaps stay NaN
    rc = rh.dropna().values
    dw = np.sum(np.diff(rc) ** 2) / np.sum(rc ** 2)
    lb = acorr_ljungbox(rc, lags=[24], return_df=True)
    acfs.append([rh.autocorr(l) for l in range(1, 49)])
    qq.append(rc[rng10.choice(len(rc), min(len(rc), 500), replace=False)] / rc.std())
    r6.append(dict(manufacturer=key[0], substation_id=key[1], rho_lag1=rh.autocorr(1),
                   rho_lag24=rh.autocorr(24), durbin_watson=dw, ljungbox_24_p=float(lb['lb_pvalue'].iloc[0]),
                   skew=stats.skew(rc), excess_kurtosis=stats.kurtosis(rc),
                   resid_sd_low_load=r[d['Q'] < d['Q'].quantile(.3)].std(),
                   resid_sd_high_load=r[d['Q'] > d['Q'].quantile(.7)].std()))
r6 = pd.DataFrame(r6)
r6.to_csv(OUT_DIR / 'hm10_R6_residual_diagnostics_per_substation.csv', index=False)
save(r6.drop(columns=['manufacturer', 'substation_id']).describe(percentiles=[.25, .5, .75]).T.reset_index(),
     'R6_residual_diagnostics_summary')

fig, ax = plt.subplots(1, 2, figsize=(174 / 25.4, 2.6))
A = np.array(acfs); lags = np.arange(1, 49)
ax[0].fill_between(lags, np.nanpercentile(A, 25, 0), np.nanpercentile(A, 75, 0), color='#c9d6ea', lw=0)
ax[0].plot(lags, np.nanmedian(A, 0), color='#1f4e9c', lw=1.2)
ax[0].axhline(0, color='0.4', lw=0.6); ax[0].set_xticks([1, 12, 24, 36, 48])
ax[0].set_xlabel('Lag (h)'); ax[0].set_ylabel('Residual autocorrelation'); ax[0].text(0.02, 0.95, '(a)', transform=ax[0].transAxes, va='top')
z = np.sort(np.concatenate(qq)); th = stats.norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
ax[1].plot(th, z, '.', ms=1.5, color='#1f4e9c'); lim = [min(th[0], z[0]), max(th[-1], z[-1])]
ax[1].plot(lim, lim, color='0.4', lw=0.6, ls='--')
ax[1].set_xlabel('Normal quantile'); ax[1].set_ylabel('Standardised residual'); ax[1].text(0.02, 0.95, '(b)', transform=ax[1].transAxes, va='top')
plt.tight_layout(); plt.savefig(FIG_DIR / 'FigS2_residual_diagnostics.pdf', bbox_inches='tight')
plt.savefig(FIG_DIR / 'FigS2_residual_diagnostics.png', dpi=300, bbox_inches='tight'); plt.show()

# ------------------------------------------------------ R7 per-manufacturer priors
def prior_same_manufacturer(key):
    return prior_from([v for k, v in F.items() if k != key and k[0] == key[0]], F[key]['cfg'])
e_pm = eval_histories(F, prior_fn=prior_same_manufacturer, days=(7, None), extra=False)
e_all = ev10[ev10.train.isin(['7 d', 'full']) & (ev10.method == 'pooled')]
r7 = []
for lab in ['7 d', 'full']:
    a = e_all[e_all.train == lab].set_index(['manufacturer', 'substation_id'])['rmse']
    b = e_pm[(e_pm.train == lab) & (e_pm.method == 'pooled')].set_index(['manufacturer', 'substation_id'])['rmse']
    j = pd.concat([a.rename('fleet_prior'), b.rename('manufacturer_prior')], axis=1).dropna()
    r7.append(dict(train=lab, n=len(j), fleet_prior_median=j.fleet_prior.median(),
                   manufacturer_prior_median=j.manufacturer_prior.median(),
                   diff_median=(j.manufacturer_prior - j.fleet_prior).median(),
                   wilcoxon_p=stats.wilcoxon(j.manufacturer_prior, j.fleet_prior).pvalue))
save(pd.DataFrame(r7), 'R7_manufacturer_priors')

# ----------------------------------------------------------- R8 fault signature stats
grp = {'normal': 'normal event', 'comfort': 'comfort fault (no heat / not enough heat / no DHW)',
       'other': 'other fault (leakage / noise / other)'}
nrm = ex[ex.group == grp['normal']]
r8 = []
for g in ['comfort', 'other']:
    v = ex[ex.group == grp[g]]
    a, n1 = int((v.vs_normal_band != 'within').sum()), len(v)
    b, n2 = int((nrm.vs_normal_band != 'within').sum()), len(nrm)
    if n1 == 0 or n2 == 0:
        continue
    dlo, dhi = confint_proportions_2indep(a, n1, b, n2, method='newcomb')
    rlo, rhi = (confint_proportions_2indep(a, n1, b, n2, compare='ratio', method='log')
                if a > 0 and b > 0 else (np.nan, np.nan))
    below = int((v.vs_normal_band == 'below').sum())
    r8.append(dict(group=g, outside=a, n=n1, outside_normal=b, n_normal=n2, share=a / n1, share_normal=b / n2,
                   diff=a / n1 - b / n2, diff_lo=dlo, diff_hi=dhi, ratio=(a / n1) / (b / n2) if b else np.nan, ratio_lo=rlo,
                   ratio_hi=rhi, fisher_p=stats.fisher_exact([[a, n1 - a], [b, n2 - b]])[1],
                   below=below, sign_p_two=stats.binomtest(below, a, 0.5).pvalue if a else np.nan,
                   sign_p_one=stats.binomtest(below, a, 0.5, alternative='greater').pvalue if a else np.nan,
                   mw_p=stats.mannwhitneyu(v.excess_matched_K, nrm.excess_matched_K).pvalue))
save(pd.DataFrame(r8), 'R8_fault_signature_stats')
lab = []
for l, v in ex[ex.kind == 'fault'].groupby('label'):
    if len(v) >= 3:
        lab.append(dict(label=l, n=len(v), median=v.excess_matched_K.median(),
                        mw_p=stats.mannwhitneyu(v.excess_matched_K, nrm.excess_matched_K).pvalue))
lab = pd.DataFrame(lab)
if len(lab):
    lab['bh_q'] = multipletests(lab.mw_p, method='fdr_bh')[1]
    lab['bonferroni_p'] = multipletests(lab.mw_p, method='bonferroni')[1]
save(lab, 'R8_labels_multiple_testing')

# ------------------------------------------------------------------ R9 complaints
cc = c4.merge(en[['manufacturer', 'substation_id', 'heat_MWh_a', 'gap_K_config_median']],
              on=['manufacturer', 'substation_id'], how='left')
cc['log_heat'] = np.log(cc.heat_MWh_a)
cc = cc.dropna(subset=['log_heat'])
def partial_spearman(df, x, y, z):
    R = df[[x, y] + z].rank()
    rx = sm.OLS(R[x], sm.add_constant(R[z])).fit().resid
    ry = sm.OLS(R[y], sm.add_constant(R[z])).fit().resid
    return stats.pearsonr(rx, ry)
r9 = []
for lab_, d in [('M1', cc[cc.manufacturer == 'M1']), ('M2', cc[cc.manufacturer == 'M2']), ('all', cc)]:
    z = ['log_heat']
    dd = d.copy()
    if lab_ == 'all':
        dd['m2'] = (dd.manufacturer == 'M2').astype(float); z = ['log_heat', 'm2']
    rho, p = partial_spearman(dd, 'ref_return_C', 'reports_per_year', z)
    r9.append(dict(sample=lab_, n=len(d), partial_spearman_given=('+'.join(z)), rho=rho, p=p,
                   rho_heat_reports=stats.spearmanr(d.log_heat, d.reports_per_year)[0],
                   rho_heat_ref=stats.spearmanr(d.log_heat, d.ref_return_C)[0],
                   zero_report_substations=int((d.reports == 0).sum()),
                   years_median=d.years.median(), years_min=d.years.min()))
save(pd.DataFrame(r9), 'R9_complaints_partial')
nb = smf.glm('reports ~ ref_return_C + log_heat + C(manufacturer)', data=cc,
             family=sm.families.NegativeBinomial(alpha=1.0), offset=np.log(cc.years)).fit()
po = smf.glm('reports ~ ref_return_C + log_heat + C(manufacturer)', data=cc,
             family=sm.families.Poisson(), offset=np.log(cc.years)).fit()
glm = []
for name, mdl in [('negative binomial', nb), ('Poisson', po)]:
    ci = mdl.conf_int()
    for t in ['ref_return_C', 'log_heat']:
        glm.append(dict(model=name, term=t, rate_ratio=np.exp(mdl.params[t]), rr_lo=np.exp(ci.loc[t, 0]),
                        rr_hi=np.exp(ci.loc[t, 1]), p=mdl.pvalues[t], pearson_chi2_df=mdl.pearson_chi2 / mdl.df_resid))
save(pd.DataFrame(glm), 'R9_complaints_glm')
dfc = cc.assign(deficit=cc.gap_K_config_median > 0, never=cc.reports == 0)
a, n1 = int(dfc[dfc.deficit].never.sum()), int(dfc.deficit.sum())
b, n2 = int(dfc[~dfc.deficit].never.sum()), int((~dfc.deficit).sum())
lo, hi = (confint_proportions_2indep(a, n1, b, n2, method='newcomb') if n1 and n2 else (np.nan, np.nan))
save(pd.DataFrame([dict(never_deficit=a, n_deficit=n1, never_no_deficit=b, n_no_deficit=n2,
                        share_deficit=a / max(n1, 1), share_no_deficit=b / max(n2, 1), diff=a / max(n1, 1) - b / max(n2, 1), diff_lo=lo, diff_hi=hi,
                        fisher_p=stats.fisher_exact([[a, n1 - a], [b, n2 - b]])[1])]), 'R9_never_complained')
z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
print('Minimum detectable |rho| at 80 % power: n=62 ->', round(np.tanh(z / np.sqrt(59)), 2),
      '; n=81 ->', round(np.tanh(z / np.sqrt(78)), 2))

# ------------------------------------------------------------- R10 excess volume
x = np.sort(en['excess_volume_m3_a_config_median'].values)[::-1]
share = np.cumsum(x) / x.sum()
n = len(x)
xs = np.sort(x); gini = (2 * np.sum(np.arange(1, n + 1) * xs) / (n * xs.sum())) - (n + 1) / n
r10 = dict(n=n, gini=gini, top2=share[min(2, n) - 1], top5=share[min(5, n) - 1], top10=share[min(10, n) - 1],
           top10pct=np.interp(0.10 * n, np.arange(1, n + 1), share), top20pct=np.interp(0.20 * n, np.arange(1, n + 1), share),
           excess_without_top2_pct=100 * x[2:].sum() / (en.volume_m3_a.sum() - en.sort_values(
               'excess_volume_m3_a_config_median', ascending=False).volume_m3_a.iloc[:2].sum()),
           own_excess_pct_median_deficit=en.loc[en.gap_K_config_median > 0, 'excess_volume_pct_config_median'].median(),
           own_excess_pct_mean_deficit=en.loc[en.gap_K_config_median > 0, 'excess_volume_pct_config_median'].mean())
# load-proportional gap: g_t = g * Q_t / mean(Q), same mean gap
lp, cost = [], []
for _, r in en.iterrows():
    key = (r.manufacturer, r.substation_id); H = OPS.get(key); g = r.gap_K_config_median
    if H is None or g <= 0:
        continue
    scale = 8760.0 / len(H); dT = (H['T_ps'] - H['T_pr']).values
    gt = g * H['Q'].values / H['Q'].values.mean()
    Ts = np.maximum(H['T_pr'].values - gt, 20.0)
    lp.append((H['V'].values - H['V'].values * dT / (H['T_ps'].values - Ts)).sum() * scale)
    cost.append(r.heat_MWh_a * r.fw_excess_return_K_config_median)
r10['excess_m3_a_constant_gap'] = en.excess_volume_m3_a_config_median.sum()
r10['excess_m3_a_load_proportional_gap'] = float(np.sum(lp))
r10['heat_x_K_MWhK_a'] = float(np.sum(cost))
r10['cost_EUR_a_low_0.08'] = 0.08 * r10['heat_x_K_MWhK_a']
r10['cost_EUR_a_high_7.91'] = 7.91 * r10['heat_x_K_MWhK_a']
r10['median_data_coverage'] = float(np.median([len(OPS[(a, b)]) / (8760 * max((OPS[(a, b)].index.max() - OPS[(a, b)].index.min()).days / 365.25, 1e-9))
                                               for a, b in zip(en.manufacturer, en.substation_id) if OPS.get((a, b)) is not None]))
save(pd.DataFrame([r10]), 'R10_excess_volume_extras')
eb = []
for key, H in OPS.items():
    if H is None or len(H) < 100:
        continue
    dT = H['T_ps'] - H['T_pr']
    qc = H['V'] / 3600 * rho_water(H['T_pr']) * cp_water(0.5 * (H['T_ps'] + H['T_pr'])) * dT
    eb.append(dict(manufacturer=key[0], substation_id=key[1], ratio=float(np.nanmedian(H['Q'] / qc))))
eb = pd.DataFrame(eb)
save(eb.groupby('manufacturer')['ratio'].describe(percentiles=[.25, .5, .75]).reset_index(), 'R10_energy_balance')
print('\nStage 10 finished. Upload all hm10_*.csv files and FigS2_residual_diagnostics.png.')
