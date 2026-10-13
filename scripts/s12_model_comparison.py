# %% [12] Model comparison: input ablation and machine-learning benchmark ------------
# Run AFTER stages 9, 9c, 9d, 9e (and optionally 10) in the same session (run_all.py).
# Reuses HM, FIT_MASK, FULLS, CHOSEN, DESIGNS, CFG, QREF, build_full, pick_test, prior_from,
# fit_own, fit_pooled, metrics, GAP_DAYS, MIN_TRAIN_H, OUT_DIR.
#
# Same substations, test blocks (30-d heating-season block, 14-d gap) and training windows
# (last 7 d / full history) as the data-efficiency test (Table 5), so every comparison is paired.
#
#   A  Input ablation of the extended model, hierarchical and own estimation:
#        - each regressor removed in turn (terms that contain it are removed with it)
#        - extended model + diurnal term (sin/cos of hour of day)
#   B  Gradient-boosting benchmark (sklearn HistGradientBoostingRegressor, fixed settings,
#      no tuning on the test blocks), same four physical inputs:
#        - GB own      : fitted on the target's own training window only
#        - GB fleet    : fitted on the OTHER substations (5 substation-grouped folds, so the
#                        target never contributes), inputs + configuration + manufacturer
#        - GB fleet+own offset : GB fleet corrected by the target's mean training residual
#      (the ML analogue of the random-intercept baseline)
#   C  Paired summary against the hierarchical extended model: median RMSE [BCa CI],
#      median difference, share of substations improved, Wilcoxon signed-rank p.
#
# Outputs: hm12_model_comparison_long.csv, hm12_model_comparison_summary.csv,
#          paper_tables/T10_model_comparison.tex
import time
import warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings('ignore')

missing = [v for v in ('DESIGNS', 'FULLS', 'CHOSEN', 'HM', 'FIT_MASK', 'build_full') if v not in globals()]
if missing:
    raise RuntimeError(f'Stage 12 needs the state of stages 9-9e (missing: {missing}); use run_all.py.')

HIST = (7, None)                      # training histories: last 7 d, full history
GB_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
                 l2_regularization=1.0, random_state=0)
POOL_MAX_H = 3000                     # hours sampled per substation for the fleet GB model
N_FOLDS = 5
rng12 = np.random.default_rng(12)
T12 = time.time()

# ------------------------------------------------------------------ design variants
def make_design(drop=(), diurnal=False):
    """Extended design of equation (1) with optional terms removed / diurnal term added.
    Column 0 is always the intercept (= reference return temperature at the centring point)."""
    def dfn(d, tag):
        cols = [np.ones(len(d))]
        Tss = d['T_ss'].values - 50.0
        A = (d['T_ps'] - d['T_ss']).values - 30.0
        L = np.log(d['Q'].values / QREF[tag]) + 1.0
        To = d['T_out'].values - 5.0
        if 'T_ss' not in drop:
            cols.append(Tss)
        if 'approach' not in drop:
            cols.append(A)
        if 'load' not in drop:
            cols.append(L)
        if 'T_out' not in drop:
            cols.append(To)
        if 'T_ss' not in drop and 'load' not in drop:
            cols.append(Tss * L / 10.0)
        if 'load' not in drop:
            cols.append(L ** 2)
        if diurnal:
            h = 2 * np.pi * d.index.hour.values / 24.0
            cols += [np.sin(h), np.cos(h)]
        return np.column_stack(cols)
    return dfn

VARIANTS = {
    'extended': make_design(),
    'without T_ss': make_design(drop=('T_ss',)),
    'without approach': make_design(drop=('approach',)),
    'without load': make_design(drop=('load',)),
    'without T_o': make_design(drop=('T_out',)),
    'extended + diurnal': make_design(diurnal=True),
}

def windows(key):
    """Test block and training windows, identical to stage 9e."""
    d = HM[key][FIT_MASK[key]]
    tb = pick_test(d)
    if tb is None:
        return None
    ts, te = tb
    test = d[(d.index > ts) & (d.index <= te)]
    cutoff = ts - pd.Timedelta(days=GAP_DAYS)
    tr = {('full' if n is None else f'{n} d'):
          (d[d.index < cutoff] if n is None else d[(d.index < cutoff) & (d.index >= cutoff - pd.Timedelta(days=n))])
          for n in HIST}
    return test, tr

# check: the 'extended' variant reproduces the design used in stages 9c-9e
_k = next(iter(FULLS[CHOSEN]))
_d = HM[_k][FIT_MASK[_k]].iloc[:50]
assert np.allclose(VARIANTS['extended'](_d, _k[0]), DESIGNS['extended'](_d, _k[0])), 'design mismatch'

# ------------------------------------------------------------------ A  ablation
rows = []
for vname, dfn in VARIANTS.items():
    t0 = time.time()
    F = FULLS['extended'] if vname == 'extended' else build_full(dfn)     # sets global `design`
    design = dfn
    for key in F:
        w = windows(key)
        if w is None:
            continue
        test, trs = w
        mu, T, s0 = prior_from([v for k, v in F.items() if k != key], F[key]['cfg'])
        Xte, yte = dfn(test, key[0]), test['T_pr'].values
        for lab, tr in trs.items():
            if len(tr) < MIN_TRAIN_H:
                continue
            Xtr, ytr = dfn(tr, key[0]), tr['T_pr'].values
            for est, b in [('own', fit_own(Xtr, ytr)), ('hierarchical', fit_pooled(Xtr, ytr, mu, T, s0))]:
                rows.append(dict(manufacturer=key[0], substation_id=key[1], train=lab,
                                 model=f'{vname} ({est})', family='physics-structured',
                                 **metrics(yte, Xte @ b)))
    print(f'A  {vname:<22s} done ({(time.time() - t0) / 60:.1f} min)')
design = DESIGNS[CHOSEN]

# ------------------------------------------------------------------ B  gradient boosting
FEATS = ['x_Tss', 'x_approach', 'x_load', 'x_Tout']

def gb_frame(d, key):
    X = pd.DataFrame({'x_Tss': d['T_ss'].values, 'x_approach': (d['T_ps'] - d['T_ss']).values,
                      'x_load': np.log(d['Q'].values / QREF[key[0]]), 'x_Tout': d['T_out'].values})
    return X

KEYS = sorted(FULLS['extended'])
WIN = {k: windows(k) for k in KEYS}
CFG_LIST = sorted({str(CFG.get(k, 'unknown')) for k in KEYS})
MAN_LIST = sorted({k[0] for k in KEYS})

def gb_fleet_frame(d, key):
    X = gb_frame(d, key)
    for c in CFG_LIST:
        X[f'cfg_{c}'] = float(str(CFG.get(key, 'unknown')) == c)
    for m in MAN_LIST:
        X[f'man_{m}'] = float(key[0] == m)
    return X

# substation-grouped folds: the fleet model for a target never sees the target's data
perm = rng12.permutation(len(KEYS))
FOLD = {KEYS[i]: j % N_FOLDS for j, i in enumerate(perm)}
t0 = time.time()
fleet_models = {}
for f in range(N_FOLDS):
    Xs, ys = [], []
    for k in KEYS:
        if FOLD[k] == f:
            continue
        d = HM[k][FIT_MASK[k]]
        if len(d) > POOL_MAX_H:
            d = d.iloc[np.sort(rng12.choice(len(d), POOL_MAX_H, replace=False))]
        Xs.append(gb_fleet_frame(d, k)); ys.append(d['T_pr'].values)
    fleet_models[f] = HistGradientBoostingRegressor(**GB_PARAMS).fit(pd.concat(Xs), np.concatenate(ys))
print(f'B  fleet GB models fitted ({(time.time() - t0) / 60:.1f} min)')

for key in KEYS:
    if WIN[key] is None:
        continue
    test, trs = WIN[key]
    yte = test['T_pr'].values
    fm = fleet_models[FOLD[key]]
    p_fleet = fm.predict(gb_fleet_frame(test, key))
    for lab, tr in trs.items():
        if len(tr) < MIN_TRAIN_H:
            continue
        ytr = tr['T_pr'].values
        own = HistGradientBoostingRegressor(**GB_PARAMS).fit(gb_frame(tr, key), ytr)
        offset = float(np.mean(ytr - fm.predict(gb_fleet_frame(tr, key))))
        for name, p in [('GB own', own.predict(gb_frame(test, key))), ('GB fleet', p_fleet),
                        ('GB fleet + own offset', p_fleet + offset)]:
            rows.append(dict(manufacturer=key[0], substation_id=key[1], train=lab, model=name,
                             family='gradient boosting', **metrics(yte, p)))
print(f'B  gradient-boosting benchmark done ({(time.time() - T12) / 60:.1f} min in stage 12)')

ev12 = pd.DataFrame(rows)
ev12.to_csv(OUT_DIR / 'hm12_model_comparison_long.csv', index=False)

# ------------------------------------------------------------------ C  paired summary
REF = 'extended (hierarchical)'

def bca(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 5 or np.ptp(x) == 0:
        return np.nan, np.nan
    r = stats.bootstrap((x,), np.median, n_resamples=2000, method='BCa', random_state=12)
    return r.confidence_interval.low, r.confidence_interval.high

summ = []
for lab in ['7 d', 'full']:
    w = ev12[ev12.train == lab].pivot_table(index=['manufacturer', 'substation_id'], columns='model',
                                            values='rmse').dropna()
    for m in w.columns:
        lo, hi = bca(w[m])
        row = dict(train=lab, model=m, n=len(w), median_rmse=w[m].median(), ci_lo=lo, ci_hi=hi,
                   mean_rmse=w[m].mean())
        if m != REF:
            diff = w[m] - w[REF]
            row.update(diff_vs_hier=diff.median(), ref_better_share=float((diff > 0).mean()),
                       wilcoxon_p=stats.wilcoxon(w[m], w[REF]).pvalue if np.any(diff != 0) else np.nan)
        summ.append(row)
summ = pd.DataFrame(summ)
order = list(VARIANTS)
summ['_o'] = summ.model.map({**{f'{v} (hierarchical)': i for i, v in enumerate(order)},
                             **{f'{v} (own)': 10 + i for i, v in enumerate(order)},
                             'GB own': 20, 'GB fleet': 21, 'GB fleet + own offset': 22})
summ = summ.sort_values(['train', '_o']).drop(columns='_o')
summ.to_csv(OUT_DIR / 'hm12_model_comparison_summary.csv', index=False)
pd.set_option('display.width', 200)
print(summ.round(3).to_string(index=False))

# LaTeX table (booktabs), same conventions as the other paper tables
TAB = OUT_DIR / 'paper_tables'
TAB.mkdir(parents=True, exist_ok=True)
def fmt_p(p):
    return '--' if not np.isfinite(p) else ('$<$0.001' if p < 0.001 else f'{p:.3f}')
lines = [r'\begin{table*}[!t]', r'\centering',
         r'\caption{Model comparison on the data-efficiency test: out-of-time RMSE of $T_{\mathrm{p,r}}$ (K), '
         r'median [95\% BCa CI]; $\Delta$: median paired difference to the hierarchical extended model; '
         r'worse: share of substations for which the model is worse than the hierarchical extended model; '
         r'$p$: Wilcoxon signed-rank test.}',
         r'\label{tab:models}', r'\begin{tabular}{@{}llcrrr@{}}', r'\toprule',
         r'History & Model & RMSE (K) & $\Delta$ (K) & Worse (\%) & $p$ \\', r'\midrule']
for lab in ['7 d', 'full']:
    for _, r in summ[summ.train == lab].iterrows():
        d_ = '--' if r.model == REF else f'{r.diff_vs_hier:+.3f}'.replace('-', '$-$')
        wr = '--' if r.model == REF else f'{100 * r.ref_better_share:.0f}'
        label = r.model.replace('_', '\\_')
        lines.append(f'{lab} & {label} & {r.median_rmse:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}] & {d_} & {wr} & '
                     f'{fmt_p(r.get("wilcoxon_p", np.nan))} \\\\')
    lines.append(r'\midrule' if lab == '7 d' else r'\bottomrule')
lines += [r'\end{tabular}', r'\end{table*}']
(TAB / 'T10_model_comparison.tex').write_text('\n'.join(lines))
print(f'\nStage 12 finished in {(time.time() - T12) / 60:.1f} min. Upload hm12_model_comparison_summary.csv.')
