# %% [9e] Data-efficiency test with the model selected in 9c (all histories) ----
# Run AFTER stage 9c in the same session (run_all.py). Uses FULLS, CHOSEN, DESIGNS, HM, FIT_MASK,
# pick_test, prior_from, fit_own, fit_pooled, metrics, boot_median, GAP_DAYS, MIN_TRAIN_H.
# Writes hm9e_data_efficiency(.csv/_summary.csv) in the same schema as stage 9, so the
# paper figure/table script picks it up automatically (one model used throughout).
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

design = DESIGNS[CHOSEN]
F = FULLS[CHOSEN]
DAYS = [7, 14, 30, 90, None]
print(f'Data efficiency with the {CHOSEN} design, {len(F)} substations')

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
    Xte, yte = design(test, key[0]), test['T_pr'].values
    for n in DAYS:
        tr = d[d.index < cutoff] if n is None else d[(d.index < cutoff) & (d.index >= cutoff - pd.Timedelta(days=n))]
        if len(tr) < MIN_TRAIN_H:
            continue
        Xtr, ytr = design(tr, key[0]), tr['T_pr'].values
        lab = 'full' if n is None else f'{n} d'
        for name, b in [('own', fit_own(Xtr, ytr)), ('fleet', mu), ('pooled', fit_pooled(Xtr, ytr, mu, T, s0))]:
            rows.append(dict(manufacturer=key[0], substation_id=key[1], configuration=F[key]['cfg'],
                             train=lab, method=name, train_hours=len(tr), test_hours=len(test),
                             **metrics(yte, Xte @ b)))
ev9e = pd.DataFrame(rows)
ev9e.to_csv(OUT_DIR / 'hm9e_data_efficiency.csv', index=False)

summ = []
for lab in ['7 d', '14 d', '30 d', '90 d', 'full']:
    w = ev9e[ev9e.train == lab].pivot_table(index=['manufacturer', 'substation_id'],
                                            columns='method', values='rmse').dropna()
    if len(w) < 5:
        continue
    for m in ['own', 'fleet', 'pooled']:
        lo, hi = boot_median(w[m])
        summ.append(dict(train=lab, method=m, n_substations=len(w), median_rmse_K=w[m].median(), ci_lo=lo, ci_hi=hi))
    for ref in ['own', 'fleet']:
        diff = w['pooled'] - w[ref]
        p = wilcoxon(w['pooled'], w[ref]).pvalue if len(w) >= 10 else np.nan
        summ.append(dict(train=lab, method=f'pooled_vs_{ref}', n_substations=len(w), median_rmse_K=diff.median(),
                         ci_lo=np.nan, ci_hi=np.nan, pooled_better_share=float((diff < 0).mean()), wilcoxon_p=p))
summ = pd.DataFrame(summ)
summ['design'] = CHOSEN
summ.to_csv(OUT_DIR / 'hm9e_data_efficiency_summary.csv', index=False)
print(summ[summ.method.isin(['own', 'fleet', 'pooled'])]
      .pivot(index='train', columns='method', values='median_rmse_K').round(2))
print(summ[summ.method == 'pooled_vs_own'][['train', 'n_substations', 'median_rmse_K',
                                             'pooled_better_share', 'wilcoxon_p']].round(3).to_string(index=False))
