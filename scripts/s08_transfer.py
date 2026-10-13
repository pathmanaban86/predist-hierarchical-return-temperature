# %% [8] Limited-training-data and transfer experiment -------------------------
# Run AFTER stages 0-6 (s00_data_and_screening.py) in the same session (run_all.py)
# (uses MANUF, faults, events, get_ops, find_col, NAMES, train_mask_for, OUT_DIR, FIG_DIR).
#
# Question: do thermodynamic features (temperature differences) keep their detection
# ability better than the raw temperatures they are built from when the target
# substation has little or no history of its own?
#
# Conditions (the reference distribution each event is compared against):
#   own_full      target's own recommended training window (baseline, as in stage 6)
#   own_90/30/14/7  only the LAST N days of that window
#   fleet_same    no target history: pooled normal data of the OTHER substations,
#                 same manufacturer (leave-one-substation-out)
#   fleet_cross   no target history: pooled normal data of the OTHER manufacturer
#
# All conditions use outdoor-temperature-conditioned reference bands (5 K bins,
# global fallback), so short windows and pooled fleets are not penalised merely for
# season. Score = mean exceedance of the reference 1-99 % band over the 7-day test
# window, averaged over the feature set (same scheme as stage 6c).
#
# Feature sets (equal information, same for every substation):
#   PHYS : dT_primary, TTD_hot, track_err_hc1   (no target-fitted parameters)
#   RAW_T: p_net_supply_T, p_net_return_T, s_hc1_supply_T  (the inputs of PHYS)
#   RAW_C: RAW_T + meter flow + meter heat power (units / building size dependent)
# meter_balance_z and heating_curve_res are left out: both need target-fitted
# parameters, which do not exist in the zero-history (fleet) conditions.
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

RAW_T = ['p_net_supply_temperature', 'p_net_return_temperature', 's_hc1_supply_temperature']
RAW_C = RAW_T + ['p_net_meter_flow', 'p_net_meter_heat_power']
PHYS = ['dT_primary', 'TTD_hot', 'track_err_hc1']
SETS = {'PHYS': PHYS, 'RAW_T': RAW_T, 'RAW_C': RAW_C}
OWN_DAYS = [None, 90, 30, 14, 7]                 # None = full window
BINS = np.arange(-25, 41, 5)                     # outdoor-temperature bins [degC]
MIN_REF = 48                                     # min reference hours per bin
POOL_HOURS_PER_SUB = 3000                        # balance the fleet pool
COMMISSIONING_DAYS = 30                          # README: drop first month
rng = np.random.default_rng(7)

# ---------------------------------------------------------------- hourly frames
def hourly_frame(tag, sid):
    """Hourly core signals + PHYS, and a mask of 'normal' hours for pooling."""
    df = get_ops(tag, sid)
    c = {k: find_col(df, k) for k in NAMES}
    H = pd.DataFrame(index=df.index)
    for n in RAW_T:
        H[n] = df[n] if n in df else np.nan
    H['p_net_meter_flow'] = df[c['V']] if c['V'] else np.nan          # canonical names,
    H['p_net_meter_heat_power'] = df[c['Q']] if c['Q'] else np.nan    # resolved like stage 5
    H['T_out'] = df[c['T_out']] if c['T_out'] else np.nan
    H['dT_primary'] = H['p_net_supply_temperature'] - H['p_net_return_temperature']
    H['TTD_hot'] = H['p_net_supply_temperature'] - H['s_hc1_supply_temperature']
    sp = df[c['T_ss_sp']] if c['T_ss_sp'] else np.nan
    H['track_err_hc1'] = H['s_hc1_supply_temperature'] - sp

    # normal hours: after commissioning, away from disturbances and fault windows
    normal = train_mask_for(df, tag, sid, df.index.min() + pd.Timedelta(days=COMMISSIONING_DAYS),
                            df.index.max())
    f = faults[(faults.manufacturer == tag) & (faults.substation_id == sid)]
    for _, r in f.iterrows():
        t0 = r.get('possible_anomaly_start')
        t0 = (r['report_date'] - pd.Timedelta(days=14)) if pd.isna(t0) else min(t0, r['report_date'] - pd.Timedelta(days=14))
        t1 = r.get('possible_anomaly_end')
        t1 = (r['report_date'] + pd.Timedelta(days=2)) if pd.isna(t1) else t1
        if pd.notna(t0) and pd.notna(t1):
            normal &= ~((df.index >= t0) & (df.index <= t1))
    Hh = H.resample('1h').mean()
    nh = normal.resample('1h').min().reindex(Hh.index).fillna(False).astype(bool)
    return Hh, nh

HOURLY, NORMAL = {}, {}
for tag, m in MANUF.items():
    for sid in sorted(m['ops']):
        HOURLY[(tag, sid)], NORMAL[(tag, sid)] = hourly_frame(tag, sid)
print('hourly frames built:', len(HOURLY))

# Fleet pool per manufacturer (balanced sample of normal hours per substation)
POOL = {}
for tag in MANUF:
    parts = []
    for (t, sid), H in HOURLY.items():
        if t != tag:
            continue
        Hn = H[NORMAL[(t, sid)]].dropna(subset=['dT_primary'])
        if len(Hn) > POOL_HOURS_PER_SUB:
            Hn = Hn.iloc[np.sort(rng.choice(len(Hn), POOL_HOURS_PER_SUB, replace=False))]
        parts.append(Hn.assign(_sid=sid))
    POOL[tag] = pd.concat(parts)
    print(f'pool {tag}: {POOL[tag]["_sid"].nunique()} substations, {len(POOL[tag])} hours')

# ------------------------------------------------------- conditioned exceedance
def cond_exceed(ref, test, col):
    r = ref[[col, 'T_out']].dropna(subset=[col])
    t = test[[col, 'T_out']].dropna(subset=[col])
    if len(r) < MIN_REF or len(t) < 12 or r[col].nunique() <= 3:
        return np.nan
    glo, ghi = np.percentile(r[col], [1, 99])
    lo, hi = np.full(len(t), glo), np.full(len(t), ghi)
    rb = np.digitize(r['T_out'].fillna(-999).values, BINS)
    tb = np.digitize(t['T_out'].fillna(-999).values, BINS)
    rv = r[col].values
    for b in np.unique(tb):
        if b == 0 or b == len(BINS):          # NaN / out-of-range outdoor temp -> global band
            continue
        vals = rv[rb == b]
        if len(vals) >= MIN_REF:
            m = tb == b
            lo[m], hi[m] = np.percentile(vals, [1, 99])
    v = t[col].values
    return float(np.mean((v < lo) | (v > hi)))

def set_score(ref, test, cols):
    vals = [cond_exceed(ref, test, c) for c in cols]
    vals = [v for v in vals if np.isfinite(v)]
    return np.mean(vals) if len(vals) == len(cols) else np.nan   # require complete set

# ------------------------------------------------------------------ main loop
other = {t: [u for u in MANUF if u != t] for t in MANUF}
rows = []
for _, e in events.iterrows():
    key = (e['manufacturer'], e['substation_id'])
    if key not in HOURLY:
        continue
    H = HOURLY[key]
    test = H[(H.index >= e['t_start']) & (H.index < e['t_end'])]
    if len(test.dropna(subset=['dT_primary'])) < 24:
        continue
    refs = {}
    own = H[(H.index >= e['train_start']) & (H.index <= e['train_end'])]
    df10 = get_ops(*key)
    tm = train_mask_for(df10, *key, e['train_start'], e['train_end'])
    own = own[tm.resample('1h').min().reindex(own.index).fillna(False).astype(bool).values]
    for d in OWN_DAYS:
        name = 'own_full' if d is None else f'own_{d}'
        refs[name] = own if d is None else own[own.index > e['train_end'] - pd.Timedelta(days=d)]
    P = POOL[e['manufacturer']]
    refs['fleet_same'] = P[P['_sid'] != e['substation_id']]
    if other[e['manufacturer']]:
        refs['fleet_cross'] = pd.concat([POOL[u] for u in other[e['manufacturer']]])
    base = dict(manufacturer=e['manufacturer'], substation_id=e['substation_id'], kind=e['kind'],
                mp_class=e['mp_class'], label=e['label'])
    for cname, ref in refs.items():
        row = {**base, 'condition': cname, 'ref_hours': len(ref)}
        for sname, cols in SETS.items():
            row[sname] = set_score(ref, test, cols)
        rows.append(row)

tl = pd.DataFrame(rows)
tl['label'] = tl['label'].fillna('normal').astype(str)      # NaN keys would be dropped by pivot
tl.to_csv(OUT_DIR / 'transfer_limited_scores.csv', index=False)
COND_ORDER = ['own_full', 'own_90', 'own_30', 'own_14', 'own_7', 'fleet_same', 'fleet_cross']

# ------------------------------------------------------------ paired bootstrap
def fast_auc(y, s):
    n1 = y.sum(); n0 = len(y) - n1
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def boot(d, a, b, B=2000):
    d = d.dropna(subset=[a, b])
    y = (d['kind'] == 'fault').astype(int).values
    if len(d) < 10 or len(np.unique(y)) < 2:
        return {}
    sa, sb = d[a].values, d[b].values
    out = dict(n=len(d), n_fault=int(y.sum()), AUC_a=fast_auc(y, sa), AUC_b=fast_auc(y, sb))
    A, Bv, D = [], [], []
    for _ in range(B):
        i = rng.integers(0, len(d), len(d)); yi = y[i]
        if 0 < yi.sum() < len(yi):
            aa, bb = fast_auc(yi, sa[i]), fast_auc(yi, sb[i])
            A.append(aa); Bv.append(bb); D.append(aa - bb)
    D = np.array(D)
    out.update(a_lo=np.percentile(A, 2.5), a_hi=np.percentile(A, 97.5),
               b_lo=np.percentile(Bv, 2.5), b_hi=np.percentile(Bv, 97.5),
               diff=out['AUC_a'] - out['AUC_b'], d_lo=np.percentile(D, 2.5), d_hi=np.percentile(D, 97.5),
               p_boot=min(1.0, 2 * min((D <= 0).mean(), (D >= 0).mean())))
    return out

# Paired evaluation: every condition is scored on the SAME events (those valid in all
# conditions), otherwise AUROCs across conditions are not comparable.
EXCL = tl['label'].astype(str).str.contains('Incorrect parameterisation', case=False, na=False)
SUBSETS = {'all': tl, 'excl. commissioning faults': tl[~EXCL]}
res = []
for sub, d in SUBSETS.items():
    wide = d.pivot_table(index=['manufacturer', 'substation_id', 'kind', 'label'],
                         columns='condition', values=list(SETS), aggfunc='first')
    common = wide.dropna().index
    dd = d.set_index(['manufacturer', 'substation_id', 'kind', 'label']).loc[common].reset_index()
    for scope, ds in [('pooled', dd)] + [(t, dd[dd.manufacturer == t]) for t in MANUF]:
        for cond in COND_ORDER:
            dc = ds[ds.condition == cond]
            for raw in ['RAW_T', 'RAW_C']:
                r = boot(dc, 'PHYS', raw)
                if r:
                    res.append(dict(subset=sub, scope=scope, condition=cond, phys_vs=raw, **r))
res = pd.DataFrame(res)
res.to_csv(OUT_DIR / 'transfer_limited_auroc.csv', index=False)
show = res[(res.subset == 'excl. commissioning faults') & (res.phys_vs == 'RAW_T')]
print(show[['scope', 'condition', 'n', 'AUC_a', 'AUC_b', 'diff', 'd_lo', 'd_hi', 'p_boot']]
      .rename(columns={'AUC_a': 'AUC_PHYS', 'AUC_b': 'AUC_RAW_T'}).round(3).to_string(index=False))
print('\nRead: physics is more robust where AUC_PHYS holds up as history shrinks and d_lo > 0.')

# ------------------------------------------------------------------- figure
# Point-range plot: AUROC (95 % bootstrap CI) per reference condition.
# Categorical slots 1-3 of the reference palette + distinct marker shapes, so series
# identity never relies on colour alone.
COL = {'PHYS': '#2a78d6', 'RAW_T': '#eb6834', 'RAW_C': '#1baf7a'}
MRK = {'PHYS': 'o', 'RAW_T': 's', 'RAW_C': '^'}
LBL = {'PHYS': 'Physics (ΔT, approach, tracking)', 'RAW_T': 'Raw temperatures (same inputs)',
       'RAW_C': 'Raw temperatures + flow + power'}
XL = {'own_full': 'Own\n(full)', 'own_90': 'Own\n90 d', 'own_30': 'Own\n30 d', 'own_14': 'Own\n14 d',
      'own_7': 'Own\n7 d', 'fleet_same': 'Fleet\nsame mfr', 'fleet_cross': 'Fleet\nother mfr'}

def plot_scope(scope, ax, subset='excl. commissioning faults'):
    r = res[(res.subset == subset) & (res.scope == scope)]
    x = np.arange(len(COND_ORDER))
    for k, (sname, off) in enumerate([('PHYS', -0.18), ('RAW_T', 0.0), ('RAW_C', 0.18)]):
        if sname == 'PHYS':
            q = r[r.phys_vs == 'RAW_T'].set_index('condition').reindex(COND_ORDER)
            y, lo, hi = q['AUC_a'], q['a_lo'], q['a_hi']
        else:
            q = r[r.phys_vs == sname].set_index('condition').reindex(COND_ORDER)
            y, lo, hi = q['AUC_b'], q['b_lo'], q['b_hi']
        ax.errorbar(x + off, y, yerr=[y - lo, hi - y], fmt=MRK[sname], ms=7, lw=1.5, capsize=0,
                    color=COL[sname], mec='white', mew=1.2, label=LBL[sname])
    ax.axhline(0.5, color='#8a8a85', lw=1, ls=(0, (4, 3)))
    ax.text(len(COND_ORDER) - 0.5, 0.505, 'chance', ha='right', va='bottom', fontsize=8, color='#5d5d58')
    ax.axvline(4.5, color='#d4d3cd', lw=1)
    ax.set_xticks(x); ax.set_xticklabels([XL[c] for c in COND_ORDER], fontsize=8)
    ax.set_ylim(0.3, 1.0); ax.set_title(scope if scope != 'pooled' else 'Both manufacturers', fontsize=10)
    ax.grid(axis='y', color='#ecebe6', lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

scopes = ['pooled'] + list(MANUF)
fig, axes = plt.subplots(1, len(scopes), figsize=(5.2 * len(scopes), 4.2), sharey=True)
for ax, sc in zip(np.atleast_1d(axes), scopes):
    plot_scope(sc, ax)
np.atleast_1d(axes)[0].set_ylabel('AUROC, fault vs normal week (95 % CI)')
h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
plt.tight_layout(rect=(0, 0, 1, 0.93))
plt.savefig(FIG_DIR / 'transfer_limited_auroc.png', dpi=300, bbox_inches='tight'); plt.show()
print('Saved:', OUT_DIR / 'transfer_limited_auroc.csv', '|', FIG_DIR / 'transfer_limited_auroc.png')
