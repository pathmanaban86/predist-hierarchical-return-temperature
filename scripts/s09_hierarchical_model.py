# %% [9] Hierarchical (partially pooled) physics-structured return-temperature model
# Run AFTER stages 0-6 (s00_data_and_screening.py) in the same session (run_all.py)
# (uses MANUF, faults, normals, get_ops, find_col, NAMES, train_mask_for,
#  config_lookup, OUT_DIR, FIG_DIR). Stage 8 is NOT required.
#
# Model (hourly, per substation i):
#   T_pr = b0 + b1*(T_ss-50) + b2*((T_ps-T_ss)-30) + b3*ln(Q/Q_ref,m) + b4*(T_out-5) + e
#     T_pr  primary return temperature      T_ss  secondary HC1 supply temperature
#     T_ps  primary supply temperature      Q     heat power; Q_ref,m = one FIXED reference
#     T_out outdoor temperature                   per manufacturer (fleet median of substation
#                                                 Q95). With a log, building size and the M1/M2
#                                                 unit difference only shift the intercept b0
#                                                 (substation-specific anyway), so slopes are
#                                                 comparable and the scale never depends on how
#                                                 much data the target has.
#   Fixed centring constants make b0 the return temperature at a typical operating point,
#   so coefficients are comparable and poolable across substations.
#
# Hierarchy: fleet -> configuration type -> substation -> 30-day window (empirical Bayes):
#   1. Per-substation full-history OLS fits b_j with covariance V_j (inflated for AR(1)
#      residual autocorrelation, since hourly residuals are not independent).
#   2. Fleet mean mu and between-substation covariance T = cov(b_j) - mean(V_j)
#      (method of moments, projected to positive definite).
#   3. Configuration mean mu_c = shrinkage of the configuration average toward mu.
#   3b. Within-substation (seasonal) covariance T_w of 30-day-window coefficients around
#      each substation's full fit: the linear model is a local linearisation, so its
#      coefficients drift with season. Prior covariance for a target window = T + T_w.
#   4. Target substation posterior: b = (P + T^-1)^-1 (P b_own + T^-1 mu_c), with
#      P = X'X / (s2 * inflation), s2 itself shrunk toward the fleet median.
#   The prior for every target is built LEAVE-ONE-SUBSTATION-OUT (no leakage).
#
# Experiment 9a (data efficiency): for each substation, hold out a 30-day heating-season
#   test block; train on the last 7/14/30/90 days or the full history before it
#   (14-day gap, as in the Energy paper). Compare three estimators of the SAME model:
#     own    : OLS on the target's own data only
#     fleet  : configuration prior mean only (no target data at all)
#     pooled : empirical-Bayes posterior (own data + configuration prior)
#   Metric: out-of-time RMSE of T_pr [K].
# Experiment 9b (preliminary efficiency cost): excess return temperature
#   (observed - fault-free model prediction) in labelled fault windows vs normal events.
import re
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

TRAIN_DAYS = [7, 14, 30, 90, None]          # None = full history
GAP_DAYS, TEST_BLOCK_DAYS = 14, 30
MIN_TEST_H, MIN_TRAIN_H, MIN_FULL_H = 240, 24, 500
CENT = np.array([0.0, 50.0, 30.0, -1.0, 5.0])
TERMS = ['b0_return_at_ref', 'b1_T_ss', 'b2_approach', 'b3_load', 'b4_T_out']
NU0 = 20                                    # prior weight (pseudo-hours) for s2 shrinkage
KAPPA_CFG = 3                               # pseudo-substations shrinking config mean to fleet
COMMISSIONING_DAYS = 30
CHARGE_KPH = 1.5                            # storage warming faster than this = DHW charging
RHO_CAP = 0.8                               # cap on AR(1) rho for effective-sample inflation
                                            # (sensitivity: rerun with 0.5 / 0.9 and report)
rng = np.random.default_rng(11)

# ---------------------------------------------------------------- data frames
def find_extra(df, pats):
    for p in pats:
        h = [c for c in df.columns if re.search(p, c) and 'setpoint' not in c]
        if h:
            return sorted(h, key=len)[0]
    return None

def normal_mask(df, tag, sid):
    """Same definition as stage 8: after commissioning, away from disturbances/faults."""
    m = train_mask_for(df, tag, sid, df.index.min() + pd.Timedelta(days=COMMISSIONING_DAYS),
                       df.index.max())
    f = faults[(faults.manufacturer == tag) & (faults.substation_id == sid)]
    for _, r in f.iterrows():
        rep = r['report_date']
        if pd.isna(rep):
            continue
        t0 = r.get('possible_anomaly_start')
        t0 = rep - pd.Timedelta(days=14) if pd.isna(t0) else min(t0, rep - pd.Timedelta(days=14))
        t1 = r.get('possible_anomaly_end')
        t1 = rep + pd.Timedelta(days=2) if pd.isna(t1) else t1
        m &= ~((df.index >= t0) & (df.index <= t1))
    return m

def build_hm(tag, sid):
    df = get_ops(tag, sid)
    c = {k: find_col(df, k) for k in NAMES}
    if not all(c[k] for k in ('T_ps', 'T_pr', 'Q', 'T_ss', 'T_out')):
        return None
    H = pd.DataFrame({'T_ps': df[c['T_ps']], 'T_pr': df[c['T_pr']], 'Q': df[c['Q']],
                      'T_ss': df[c['T_ss']], 'T_out': df[c['T_out']]})
    H['T_up'] = df[c['T_up']] if c['T_up'] else np.nan
    H['T_lo'] = df[c['T_lo']] if c['T_lo'] else np.nan
    v3 = find_extra(df, [r'3.?way.*status'])
    H['v3'] = df[v3] if v3 else np.nan
    nm = normal_mask(df, tag, sid)
    Hh = H.resample('1h').mean()
    Hh['normal'] = nm.resample('1h').min().reindex(Hh.index).fillna(False).astype(bool)
    charging = ((Hh['T_up'].diff() > CHARGE_KPH) | (Hh['T_lo'].diff() > CHARGE_KPH) |
                (Hh['v3'] > 0.5)).fillna(False)
    sane = (Hh[['T_ps', 'T_pr', 'Q', 'T_ss', 'T_out']].notna().all(axis=1)
            & (Hh['T_ps'] > 50) & (Hh['T_ps'] - Hh['T_pr']).between(0.5, 80)
            & (Hh['T_ss'] > 25) & (Hh['Q'] > 0) & Hh['T_out'].between(-30, 40))
    Hh['ok'] = sane & ~charging              # physically valid space-heating hours
    return Hh

CFG = {}
for tag in MANUF:
    for sid, v in config_lookup(tag).items():
        CFG[(tag, sid)] = str(v)

HM = {}
for tag, m in MANUF.items():
    for sid in sorted(m['ops']):
        h = build_hm(tag, sid)
        if h is not None:
            HM[(tag, sid)] = h
print(f'substations with all model signals: {len(HM)} / {sum(len(m["ops"]) for m in MANUF.values())}')

# ---------------------------------------------------------------- estimation
def design(d, tag):
    X = np.column_stack([np.ones(len(d)), d['T_ss'].values, (d['T_ps'] - d['T_ss']).values,
                         np.log(d['Q'].values / QREF[tag]), d['T_out'].values]) - CENT
    X[:, 0] = 1.0
    return X

def ar1(r):
    if len(r) < 12:
        return 0.5
    r = r - r.mean()
    den = np.dot(r, r)
    return float(np.clip(np.dot(r[:-1], r[1:]) / den, 0.0, RHO_CAP)) if den > 0 else 0.5

def fit_own(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)      # minimum-norm if rank deficient
    return b

def fit_full(X, y):
    b = fit_own(X, y); r = y - X @ b
    n, p = X.shape
    rho = ar1(r); infl = (1 + rho) / (1 - rho)
    s2 = r @ r / (n - p)
    return dict(b=b, V=s2 * infl * np.linalg.pinv(X.T @ X), s2=s2, rho=rho)

def prior_from(fits, target_cfg):
    B = np.array([f['b'] for f in fits]); Vm = np.mean([f['V'] for f in fits], axis=0)
    mu = B.mean(axis=0)
    T = np.cov(B.T) - Vm
    T = (T + T.T) / 2
    w, U = np.linalg.eigh(T)
    w = np.maximum(w, 1e-3 * max(w.max(), 1e-6))
    T = U @ np.diag(w) @ U.T
    same = np.array([f['b'] for f in fits if f['cfg'] == target_cfg])
    n_c = len(same)
    mu_c = (n_c * same.mean(axis=0) + KAPPA_CFG * mu) / (n_c + KAPPA_CFG) if n_c else mu
    s0sq = float(np.median([f['s2'] for f in fits]))
    # within-substation (window-to-window) covariance, pooled over the fleet
    D = [dv for f in fits for dv in f['devs']]
    Vw = [v for f in fits for v in f['Vw']]
    if len(D) >= 10:
        D = np.array(D)
        lo, hi = np.percentile(D, [2.5, 97.5], axis=0)       # trim ill-conditioned windows
        D = D[np.all((D >= lo) & (D <= hi), axis=1)]
        Tw = psd(D.T @ D / len(D) - np.median(Vw, axis=0), floor=1e-6)
        T = psd(T + Tw)
    return mu_c, T, s0sq

def psd(M, floor=1e-6):
    M = (M + M.T) / 2
    w, U = np.linalg.eigh(M)
    return U @ np.diag(np.maximum(w, floor)) @ U.T

WIN_DAYS, MIN_WIN_H = 30, 200

def window_dev(d, tag, b_full):
    """Deviations of non-overlapping 30-day-window fits from the full fit (+ mean V)."""
    devs, Vs = [], []
    t = d.index.min()
    while t < d.index.max():
        w = d[(d.index >= t) & (d.index < t + pd.Timedelta(days=WIN_DAYS))]
        t += pd.Timedelta(days=WIN_DAYS)
        if len(w) < MIN_WIN_H or w['T_out'].std() < 0.5:
            continue
        X = design(w, tag)
        if np.linalg.cond(X.T @ X) > 1e8:
            continue
        f = fit_full(X, w['T_pr'].values)
        devs.append(f['b'] - b_full); Vs.append(f['V'])
    return devs, Vs

def fit_pooled(X, y, mu, T, s0sq, iters=4):
    if len(y) == 0:
        return mu.copy()
    Ti = np.linalg.inv(T)
    XtX, Xty, n = X.T @ X, X.T @ y, len(y)
    s2, infl, b = s0sq, 1.0, mu.copy()
    for _ in range(iters):
        P = XtX / (s2 * infl)
        b = np.linalg.solve(P + Ti, Xty / (s2 * infl) + Ti @ mu)
        r = y - X @ b
        rho = ar1(r); infl = (1 + rho) / (1 - rho)
        s2 = (r @ r + NU0 * s0sq) / (n + NU0)
    return b

# Fixed load reference per manufacturer (median of substation Q95 over normal valid hours)
QREF = {}
for tag in MANUF:
    q = [np.percentile(H.loc[H['ok'] & H['normal'], 'Q'], 95) for (t, s), H in HM.items()
         if t == tag and (H['ok'] & H['normal']).sum() > 100]
    QREF[tag] = float(np.median(q)) if q else 1.0
for key, H in HM.items():                         # drop near-zero load hours (log leverage)
    H['ok'] &= H['Q'] > 0.02 * QREF[key[0]]
print('Load reference per manufacturer (native units):', {k: round(v, 3) for k, v in QREF.items()})

# Full-history fits used to build the (leave-one-out) priors
FULL = {}
for key, H in HM.items():
    d = H[H['ok'] & H['normal']]
    if len(d) < MIN_FULL_H or d['T_out'].std() < 1.0:
        continue
    f = fit_full(design(d, key[0]), d['T_pr'].values)
    devs, Vw = window_dev(d, key[0], f['b'])
    FULL[key] = {**f, 'cfg': CFG.get(key, 'unknown'), 'n_hours': len(d), 'devs': devs, 'Vw': Vw}
print(f'substations with enough clean history for the fleet prior: {len(FULL)}')

coef = pd.DataFrame([{'manufacturer': k[0], 'substation_id': k[1], 'configuration': v['cfg'],
                      'n_hours': v['n_hours'], 'ar1_rho': v['rho'], 'resid_sd_K': np.sqrt(v['s2']),
                      **dict(zip(TERMS, v['b']))} for k, v in FULL.items()])
coef.to_csv(OUT_DIR / 'hm_full_coefficients.csv', index=False)
print('\nCoefficient summary across substations (full-history own fits):')
print(coef[TERMS + ['resid_sd_K']].describe().T[['mean', '50%', 'std', 'min', 'max']].round(3))
print('\nMedian coefficients by configuration:')
print(coef.groupby('configuration')[TERMS].median().round(3))

# ------------------------------------------------------- 9a data-efficiency test
def pick_test(d):
    heat = d[d['T_out'] < 12]
    if len(heat) < MIN_TEST_H:
        return None
    end = heat.index.max()
    for _ in range(40):
        start = end - pd.Timedelta(days=TEST_BLOCK_DAYS)
        blk = d[(d.index > start) & (d.index <= end)]
        hist = d[d.index < start - pd.Timedelta(days=GAP_DAYS)]
        if len(blk) >= MIN_TEST_H and len(hist) >= MIN_TRAIN_H:
            return start, end
        end -= pd.Timedelta(days=14)
    return None

def metrics(y, p):
    e = y - p
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))), bias=float(np.mean(e)))

rows = []
for key in FULL:
    d = HM[key][HM[key]['ok'] & HM[key]['normal']]
    tb = pick_test(d)
    if tb is None:
        continue
    ts, te = tb
    test = d[(d.index > ts) & (d.index <= te)]
    cutoff = ts - pd.Timedelta(days=GAP_DAYS)
    others = [{**v} for k, v in FULL.items() if k != key]
    mu_c, T, s0sq = prior_from(others, FULL[key]['cfg'])
    for n in TRAIN_DAYS:
        tr = d[d.index < cutoff] if n is None else d[(d.index < cutoff) & (d.index >= cutoff - pd.Timedelta(days=n))]
        label = 'full' if n is None else f'{n} d'
        base = dict(manufacturer=key[0], substation_id=key[1], configuration=FULL[key]['cfg'],
                    train=label, train_hours=len(tr), test_hours=len(test))
        if len(tr) < MIN_TRAIN_H:
            continue
        Xtr, ytr = design(tr, key[0]), tr['T_pr'].values
        Xte, yte = design(test, key[0]), test['T_pr'].values
        for name, b in [('own', fit_own(Xtr, ytr)), ('fleet', mu_c),
                        ('pooled', fit_pooled(Xtr, ytr, mu_c, T, s0sq))]:
            rows.append({**base, 'method': name, **metrics(yte, Xte @ b)})

ev = pd.DataFrame(rows)
ev.to_csv(OUT_DIR / 'hm_data_efficiency.csv', index=False)
ORDER = ['7 d', '14 d', '30 d', '90 d', 'full']

def boot_median(x, B=2000):
    x = np.asarray(x); x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan
    meds = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return np.percentile(meds, 2.5), np.percentile(meds, 97.5)

summ = []
for tr_lab in ORDER:
    w = ev[ev.train == tr_lab].pivot_table(index=['manufacturer', 'substation_id'],
                                           columns='method', values='rmse')
    w = w.dropna()                                   # paired: same substations, all 3 methods
    if len(w) < 5:
        continue
    for mth in ['own', 'fleet', 'pooled']:
        lo, hi = boot_median(w[mth])
        summ.append(dict(train=tr_lab, method=mth, n_substations=len(w),
                         median_rmse_K=w[mth].median(), ci_lo=lo, ci_hi=hi))
    for ref in ['own', 'fleet']:
        diff = w['pooled'] - w[ref]
        try:
            p = wilcoxon(w['pooled'], w[ref]).pvalue
        except ValueError:
            p = np.nan
        summ.append(dict(train=tr_lab, method=f'pooled_vs_{ref}', n_substations=len(w),
                         median_rmse_K=diff.median(), ci_lo=np.nan, ci_hi=np.nan,
                         pooled_better_share=float((diff < 0).mean()), wilcoxon_p=p))
summ = pd.DataFrame(summ)
summ.to_csv(OUT_DIR / 'hm_data_efficiency_summary.csv', index=False)
print('\nOut-of-time RMSE of primary return temperature [K] (median across substations):')
print(summ[summ.method.isin(['own', 'fleet', 'pooled'])]
      .pivot(index='train', columns='method', values='median_rmse_K').reindex(ORDER).round(2))
print('\nPaired comparisons (negative = pooled better):')
print(summ[summ.method.str.startswith('pooled_vs')][['train', 'method', 'n_substations', 'median_rmse_K',
      'pooled_better_share', 'wilcoxon_p']].round(3).to_string(index=False))

# Figure 9a: median RMSE vs training history (95 % bootstrap CI of the median).
COL = {'own': '#eb6834', 'fleet': '#1baf7a', 'pooled': '#2a78d6'}
MRK = {'own': 's', 'fleet': '^', 'pooled': 'o'}
LBL = {'own': 'Own data only', 'fleet': 'Fleet / configuration prior only',
       'pooled': 'Hierarchical (partially pooled)'}
fig, ax = plt.subplots(figsize=(6.4, 4.2))
x = np.arange(len(ORDER))
for k, mth in enumerate(['own', 'fleet', 'pooled']):
    q = summ[summ.method == mth].set_index('train').reindex(ORDER)
    off = (k - 1) * 0.16
    ax.errorbar(x + off, q['median_rmse_K'], yerr=[q['median_rmse_K'] - q['ci_lo'], q['ci_hi'] - q['median_rmse_K']],
                fmt=MRK[mth], ms=7, lw=1.5, capsize=0, color=COL[mth], mec='white', mew=1.2, label=LBL[mth])
ax.set_xticks(x); ax.set_xticklabels(['7 days', '14 days', '30 days', '90 days', 'Full'])
ax.set_xlabel('Own training history before the test block')
ax.set_ylabel('Out-of-time RMSE of $T_{p,return}$ [K]\n(median across substations, 95 % CI)')
ax.set_ylim(bottom=0)
ax.grid(axis='y', color='#ecebe6', lw=0.8); ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3,
          handletextpad=0.3, columnspacing=1.0)
plt.tight_layout(); plt.savefig(FIG_DIR / 'hm_data_efficiency.png', dpi=300, bbox_inches='tight'); plt.show()

# ------------------------------------------- 9b excess return temperature (preliminary)
# Fault-free model per substation = pooled posterior on ALL its normal hours (fault windows
# were excluded from 'normal'). Excess = mean(observed - predicted) over the fault window.
# Purpose: magnitude of the efficiency penalty, not detection (the model sees post-fault data).
exc = []
for key in FULL:
    tag, sid = key
    H = HM[key]
    d = H[H['ok'] & H['normal']]
    others = [v for k, v in FULL.items() if k != key]
    mu_c, T, s0sq = prior_from(others, FULL[key]['cfg'])
    b = fit_pooled(design(d, tag), d['T_pr'].values, mu_c, T, s0sq)
    valid = H[H['ok']]
    def window_excess(t0, t1):
        w = valid[(valid.index >= t0) & (valid.index < t1)]
        if len(w) < 12:
            return np.nan, len(w)
        return float(np.mean(w['T_pr'].values - design(w, tag) @ b)), len(w)
    for _, r in faults[(faults.manufacturer == tag) & (faults.substation_id == sid)].iterrows():
        rep = r['report_date']
        if pd.isna(rep):
            continue
        t0 = r.get('possible_anomaly_start')
        t0 = rep - pd.Timedelta(days=7) if pd.isna(t0) else t0
        e, nh = window_excess(t0, rep)
        exc.append(dict(manufacturer=tag, substation_id=sid, kind='fault', label=r.get('fault_label'),
                        problem=str(r.get('problem_en')).lower(), mp_class=r.get('mp_class'),
                        window='anomaly start->report' if pd.notna(r.get('possible_anomaly_start')) else '7 d->report',
                        excess_K=e, hours=nh))
    for _, r in normals[(normals.manufacturer == tag) & (normals.substation_id == sid)].iterrows():
        s0 = r.get('event_start')
        if pd.isna(s0):
            continue
        e, nh = window_excess(s0, s0 + pd.Timedelta(days=7))
        exc.append(dict(manufacturer=tag, substation_id=sid, kind='normal', label='normal event',
                        problem='normal', mp_class='n/a', window='first 7 d', excess_K=e, hours=nh))

exc = pd.DataFrame(exc)
exc.to_csv(OUT_DIR / 'hm_excess_return_temperature.csv', index=False)
ok_exc = exc.dropna(subset=['excess_K'])
norm = ok_exc[ok_exc.kind == 'normal']['excess_K']
print(f'\nNormal events: mean excess {norm.mean():+.2f} K, sd {norm.std():.2f} K, n={len(norm)}')
by_label = (ok_exc[ok_exc.kind == 'fault'].groupby('label')['excess_K']
            .agg(['count', 'mean', 'median', 'min', 'max']).sort_values('mean', ascending=False))
print('Fault windows, excess primary return temperature [K] by fault label:')
print(by_label.round(2))

# Figure 9b: per-event excess by fault label against the normal-event spread.
fl = ok_exc[ok_exc.kind == 'fault']
order = fl.groupby('label')['excess_K'].mean().sort_values().index.tolist()
if len(order):
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(order) + 1.6))
    lo, hi = np.percentile(norm, [5, 95]) if len(norm) >= 5 else (np.nan, np.nan)
    if np.isfinite(lo):
        ax.axvspan(lo, hi, color='#d4d3cd', alpha=0.5, lw=0, label='Normal events, 5-95 %')
    ax.axvline(0, color='#8a8a85', lw=1)
    for i, lab in enumerate(order):
        v = fl.loc[fl.label == lab, 'excess_K'].values
        ax.scatter(v, np.full(len(v), i) + rng.uniform(-0.12, 0.12, len(v)), s=26, color='#2a78d6',
                   edgecolor='white', linewidth=0.8, zorder=3)
        ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color='#184f95', lw=2, zorder=4)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([str(l)[:55] for l in order], fontsize=8)
    ax.set_xlabel('Excess primary return temperature vs fault-free model [K]')
    ax.grid(axis='x', color='#ecebe6', lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    plt.tight_layout(); plt.savefig(FIG_DIR / 'hm_excess_by_fault.png', dpi=300); plt.show()

print('\nSaved: hm_full_coefficients.csv, hm_data_efficiency(.csv, _summary.csv), '
      'hm_excess_return_temperature.csv and figures hm_*.png in', OUT_DIR)
