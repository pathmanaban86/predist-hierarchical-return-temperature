# =============================================================================
# PreDist (district heating substations) — exploratory + physics-feature analysis
# For IMechE Part A paper. Run via run_all.py (or cell by cell; "# %%" = new cell).
#
# Stages
#   0  Setup: unzip, discover folder layout (paths from config.py)
#   1  Load metadata (faults, normal events, disturbances, features, configurations)
#   2  Substation inventory: span, completeness, features, configuration  -> CSV
#   3  Fault / event summary tables and plots                              -> CSV + PNG
#   4  Feature-availability matrix (which signals each substation has)     -> CSV + PNG
#   5  Physics feature layer (thermodynamic residuals)
#   6  Event screening: fault vs normal windows, physics vs raw signals    -> CSV
#   7  Per-fault event plots (14 d before report)                          -> PNG
#
# Evaluation conventions follow Roelofs et al., Energy 355 (2026) 141178:
#   fault test window  = 7 days ending at the incident report
#   normal test window = first 7 days of each pre-defined normal event
#   training window    = 'Training start'/'Training end' given in the CSVs
# Stage 6 is a SCREENING analysis (distribution shift), not the final detector.
# =============================================================================

# %% [0] Setup -----------------------------------------------------------------
import re, zipfile, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 50)

from config import ZIP_PATH, EXTRACT_DIR, PARQ_DIR, OUT_DIR   # paths: see config.py
FIG_DIR     = OUT_DIR / 'figures'
for d in (OUT_DIR, FIG_DIR, FIG_DIR / 'events', PARQ_DIR):
    d.mkdir(parents=True, exist_ok=True)

if not EXTRACT_DIR.exists():
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(EXTRACT_DIR)
# unpack any nested zips (Zenodo bundles sometimes contain one zip per manufacturer)
for nz in list(EXTRACT_DIR.rglob('*.zip')):
    target = nz.with_suffix('')
    if not target.exists():
        with zipfile.ZipFile(nz) as z:
            z.extractall(target)

print('Top-level contents:')
for p in sorted(EXTRACT_DIR.rglob('*'))[:40]:
    if p.is_dir() or p.suffix != '.csv' or 'operational_data' not in str(p):
        print('  ', p.relative_to(EXTRACT_DIR))


# %% [1] Helpers + metadata ----------------------------------------------------
def norm(c: str) -> str:
    """'Possible anomaly start' -> 'possible_anomaly_start'."""
    return re.sub(r'[^a-z0-9]+', '_', str(c).strip().lower()).strip('_')

def sniff_sep(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        head = f.readline()
    return ';' if head.count(';') > head.count(',') else ','

def read_table(path):
    sep = sniff_sep(path)
    df = pd.read_csv(path, sep=sep, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df, sep

def to_bool(s):
    return s.astype(str).str.strip().str.lower().isin(['true', '1', 'yes', 'y'])

def load_meta(path):
    df, _ = read_table(path)
    df.columns = [norm(c) for c in df.columns]
    id_col = next((c for c in df.columns if 'substation' in c), None)
    if id_col and id_col != 'substation_id':
        df = df.rename(columns={id_col: 'substation_id'})
    if 'substation_id' in df:
        df['substation_id'] = pd.to_numeric(
            df['substation_id'].astype(str).str.extract(r'(\d+)')[0], errors='coerce').astype('Int64')
        df = df.dropna(subset=['substation_id']).reset_index(drop=True)
    for c in df.columns:
        if re.search(r'(date|start|end)$', c):
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df

# Discover manufacturer folders = any folder that holds faults.csv
MANUF = {}
for fp in sorted(EXTRACT_DIR.rglob('faults.csv')):
    d = fp.parent
    mnum = re.search(r'(?:manufacturer|^m)[\s_\-]*(\d)', d.name, re.I) or re.search(r'(\d)\s*$', d.name)
    tag = f'M{mnum.group(1)}' if mnum else d.name
    if tag in MANUF:                      # unexpected layout: keep both, never overwrite
        tag = f'{tag}_{d.name}'
    ops = sorted((d / 'operational_data').glob('*.csv')) or sorted(d.rglob('*.csv'))
    ops = [p for p in ops if 'operational_data' in str(p)]
    MANUF[tag] = {
        'dir': d,
        'ops': {int(re.findall(r'\d+', p.stem)[-1]): p for p in ops},
        'faults': load_meta(d / 'faults.csv'),
        'normal': load_meta(d / 'normal_events.csv'),
        'dist': load_meta(d / 'disturbances.csv'),
        'feat': pd.read_csv(d / 'feature_descriptions.csv', sep=sniff_sep(d / 'feature_descriptions.csv')),
        'config': load_meta(d / 'configuration_types.csv') if (d / 'configuration_types.csv').exists() else None,
    }
    print(f"{tag}: folder='{d.name}', substations={len(MANUF[tag]['ops'])}, "
          f"faults={len(MANUF[tag]['faults'])}, normal events={len(MANUF[tag]['normal'])}")

for tag, m in MANUF.items():
    print(f'\n{tag} faults columns   :', list(m['faults'].columns))
    print(f'{tag} normal columns   :', list(m['normal'].columns))
    print(f'{tag} config columns   :', None if m['config'] is None else list(m['config'].columns))
    print(f'{tag} feature desc head:\n', m['feat'].head(8))

# Combined feature dictionary (for the paper's nomenclature table)
feat_all = pd.concat([m['feat'].assign(manufacturer=t) for t, m in MANUF.items()], ignore_index=True)
feat_all.to_csv(OUT_DIR / 'feature_descriptions_combined.csv', index=False)


# %% [1b] Operational data loader with parquet cache ---------------------------
_CACHE = {}

def load_ops_raw(path):
    df, sep = read_table(path)
    tcol = next((c for c in df.columns if re.search(r'time|date', c, re.I)), df.columns[0])
    df[tcol] = pd.to_datetime(df[tcol], errors='coerce')
    df = df.dropna(subset=[tcol]).set_index(tcol).sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df.index.name = 'timestamp'
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].astype(str).str.replace(',', '.', regex=False) if sep == ';' else df[c]
            df[c] = pd.to_numeric(s, errors='coerce')
    return df

def get_ops(tag, sid):
    key = (tag, sid)
    if key in _CACHE:
        return _CACHE[key]
    pq = PARQ_DIR / tag / f'{sid}.parquet'
    pk = pq.with_suffix('.pkl')
    if pq.exists():
        df = pd.read_parquet(pq)
    elif pk.exists():
        df = pd.read_pickle(pk)
    else:
        pq.parent.mkdir(parents=True, exist_ok=True)
        df = load_ops_raw(MANUF[tag]['ops'][sid])
        try:
            df.to_parquet(pq)
        except ImportError:                  # no pyarrow/fastparquet -> pickle cache
            df.to_pickle(pk)
    if len(_CACHE) > 12:          # keep RAM bounded
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = df
    return df


# %% [2] Substation inventory --------------------------------------------------
def config_lookup(tag):
    c = MANUF[tag]['config']
    if c is None or 'substation_id' not in c:
        return {}
    val_col = next((x for x in c.columns if 'config' in x),
                   [x for x in c.columns if x != 'substation_id'][0])
    return dict(zip(c['substation_id'].astype(int), c[val_col]))

rows = []
for tag, m in MANUF.items():
    cfg = config_lookup(tag)
    for sid in sorted(m['ops']):
        df = get_ops(tag, sid)
        if df.empty:
            continue
        span = df.index.max() - df.index.min()
        expected = span / pd.Timedelta('10min') + 1
        num = df.select_dtypes('number')
        rows.append({
            'manufacturer': tag, 'substation_id': sid,
            'configuration': cfg.get(sid, np.nan),
            'start': df.index.min(), 'end': df.index.max(),
            'span_years': span.days / 365.25,
            'rows': len(df),
            'timestamp_completeness_%': 100 * len(df) / expected,
            'mean_feature_nonnull_%': 100 * num.notna().mean().mean(),
            'n_features': num.shape[1],
            'n_faults': int((m['faults']['substation_id'] == sid).sum()),
            'n_normal_events': int((m['normal']['substation_id'] == sid).sum()),
            'n_disturbances': int((m['dist']['substation_id'] == sid).sum()),
        })
inv = pd.DataFrame(rows)
inv.to_csv(OUT_DIR / 'substation_inventory.csv', index=False)
print(inv.groupby('manufacturer').agg(
    substations=('substation_id', 'count'),
    mean_years=('span_years', 'mean'), total_years=('span_years', 'sum'),
    completeness=('timestamp_completeness_%', 'mean'),
    feat_min=('n_features', 'min'), feat_max=('n_features', 'max')).round(2))
print('\nConfiguration counts:\n', inv.groupby(['manufacturer', 'configuration']).size())
# Sanity check vs. Table 1 of the Energy paper: M1 35/4.1 y, M2 58/1.6 y, 93 total.


# %% [3] Fault and event summary ----------------------------------------------
faults = pd.concat([m['faults'].assign(manufacturer=t) for t, m in MANUF.items()], ignore_index=True)
normals = pd.concat([m['normal'].assign(manufacturer=t) for t, m in MANUF.items()], ignore_index=True)

if 'efd_possible' in faults:
    faults['efd_possible'] = to_bool(faults['efd_possible'])
else:
    faults['efd_possible'] = True
# Monitoring potential may be numeric (1-5) or text ('high'/'low'/'unknown'); handle both.
mp_raw = faults['monitoring_potential'] if 'monitoring_potential' in faults else pd.Series(np.nan, index=faults.index)
mp = pd.to_numeric(mp_raw, errors='coerce')
mp_txt = mp_raw.astype(str).str.strip().str.lower()
faults['mp_numeric'] = mp
faults['mp_class'] = np.where(mp >= 2.5, 'high',
                     np.where(mp < 2.5, 'low',
                     np.where(mp_txt.str.startswith('high'), 'high',
                     np.where(mp_txt.str.startswith('low'), 'low', 'unknown'))))

def count_table(col):
    if col not in faults:
        return None
    t = faults.groupby([col, 'manufacturer']).size().unstack(fill_value=0)
    t['total'] = t.sum(axis=1)
    return t.sort_values('total', ascending=False)

summ = {
    'problem_category': count_table('problem_en'),
    'fault_label': count_table('fault_label'),
    'monitoring_potential': count_table('mp_class'),
    'efd_possible': count_table('efd_possible'),
}
with pd.ExcelWriter(OUT_DIR / 'fault_summary.xlsx') as xw:
    for name, t in summ.items():
        if t is not None:
            t.to_excel(xw, sheet_name=name[:31])
            print(f'\n=== {name} ===\n', t)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, key in zip(axes, ['problem_category', 'fault_label']):
    t = summ[key]
    if t is not None:
        t.drop(columns='total').head(15).iloc[::-1].plot.barh(ax=ax)
        ax.set_title(key.replace('_', ' ')); ax.set_xlabel('# faults')
plt.tight_layout(); plt.savefig(FIG_DIR / 'fault_counts.png', dpi=300); plt.show()


# %% [4] Feature availability matrix ------------------------------------------
avail = {}
for tag, m in MANUF.items():
    for sid in sorted(m['ops']):
        cols = get_ops(tag, sid).select_dtypes('number').columns
        avail[f'{tag}-{sid}'] = pd.Series(True, index=cols)
avail = pd.DataFrame(avail).fillna(False).astype(bool)
avail = avail.loc[avail.sum(axis=1).sort_values(ascending=False).index]
avail.to_csv(OUT_DIR / 'feature_availability.csv')
print('Signals present in >=90% of substations:\n',
      avail.index[avail.mean(axis=1) >= 0.9].tolist())

fig, ax = plt.subplots(figsize=(18, max(6, 0.22 * len(avail))))
ax.imshow(avail.values, aspect='auto', cmap='Greys', interpolation='nearest')
ax.set_yticks(range(len(avail))); ax.set_yticklabels(avail.index, fontsize=7)
ax.set_xticks(range(avail.shape[1])); ax.set_xticklabels(avail.columns, rotation=90, fontsize=5)
ax.set_title('Feature availability per substation (black = present)')
plt.tight_layout(); plt.savefig(FIG_DIR / 'feature_availability.png', dpi=300); plt.show()


# %% [5] Physics feature layer -------------------------------------------------
# Standardised names (README): side_circuit_place_component_dimension_metric.
# Exact names first, regex fallbacks after. Setpoint/mode/status columns are excluded
# unless the key ends with '_sp'.
NAMES = {
    'T_ps':    [r'^p_net_supply_temperature$'],
    'T_pr':    [r'^p_net_return_temperature$'],
    'V':       [r'^p_net_meter_flow$', r'^p_net.*flow$'],
    'Q':       [r'^p_net_meter_(heat_)?power$', r'^p_net.*power$'],
    'T_ss':    [r'^s_hc1_supply_temperature$'],
    'T_ss_sp': [r'^s_hc1_supply_temperature_setpoint$'],
    'T_out':   [r'^outdoor_temperature$', r'outdoor.*temperature$'],
    'T_up':    [r'^s_.*upper.*temperature$'],
    'T_lo':    [r'^s_.*lower.*temperature$'],
}

def find_col(df, key):
    for pat in NAMES[key]:
        hits = [c for c in df.columns if re.search(pat, c)]
        if not key.endswith('_sp'):
            hits = [c for c in hits if not re.search(r'setpoint|mode|status', c)]
        if hits:
            return sorted(hits, key=len)[0]
    return None

def rho_water(T):
    """Liquid water density [kg/m3], T in degC (Kell-type fit, ~0.1 % for 0-100 degC)."""
    T = np.clip(T, 1, 120)
    return 1000 * (1 - (T + 288.9414) / (508929.2 * (T + 68.12963)) * (T - 3.9863) ** 2)

def cp_water(T):
    """Liquid water isobaric heat capacity [kJ/(kg K)], T in degC."""
    T = np.clip(T, 1, 120)
    return 4.2174 - 0.0056181 * T + 0.0012992 * T ** 1.52 - 0.00011535 * T ** 2 + 4.14964e-6 * T ** 2.5

def physics_features(df, train_mask):
    """
    Returns (P, info). Features needing a fitted reference are fitted on train_mask only.
      dT_primary        T_p,supply - T_p,return              cooling / return-temperature quality
      TTD_hot           T_p,supply - T_s,supply(HC1)         heat-exchanger hot-end approach
      track_err_hc1     T_s,supply - setpoint                valve / actuator / dp-regulator faults
      storage_strat     T_upper - T_lower                    charging pump / storage faults
      meter_balance_z   (Q_meter - k*V*rho*cp*dT) / std(Q_train)
                        k fitted on training data, so unknown flow/power units cancel
      heating_curve_res setpoint - f(T_outdoor), f = quadratic fit on training data
    """
    c = {k: find_col(df, k) for k in NAMES}
    g = lambda k: df[c[k]]
    P = pd.DataFrame(index=df.index)
    info = {'cols': c}

    if c['T_ps'] and c['T_pr']:
        P['dT_primary'] = g('T_ps') - g('T_pr')
    if c['T_ps'] and c['T_ss']:
        P['TTD_hot'] = g('T_ps') - g('T_ss')
    if c['T_ss'] and c['T_ss_sp']:
        P['track_err_hc1'] = g('T_ss') - g('T_ss_sp')
    if c['T_up'] and c['T_lo']:
        P['storage_strat'] = g('T_up') - g('T_lo')

    if all(c[k] for k in ('V', 'Q', 'T_ps', 'T_pr')):
        dT = g('T_ps') - g('T_pr')
        Tm = 0.5 * (g('T_ps') + g('T_pr'))
        qc = g('V') * rho_water(g('T_pr')) * cp_water(Tm) * dT   # unit-free proxy; meter assumed in return line
        tr = train_mask & (qc > 0) & (g('Q') > 0) & (dT > 2)
        if tr.sum() > 200:
            k = np.nanmedian(g('Q')[tr] / qc[tr])
            s = np.nanstd(g('Q')[train_mask])
            if np.isfinite(k) and s > 0:
                P['meter_balance_z'] = (g('Q') - k * qc) / s
                info['k_meter'] = k

    if c['T_ss_sp'] and c['T_out']:
        tr = train_mask & g('T_out').between(-20, 18) & (g('T_ss_sp') > 20)
        if tr.sum() > 200:
            coef = np.polyfit(g('T_out')[tr], g('T_ss_sp')[tr], 2)
            P['heating_curve_res'] = g('T_ss_sp') - np.polyval(coef, g('T_out'))
            info['heating_curve_coef'] = coef
    return P, info

# Quick check on one substation
_tag = next(iter(MANUF)); _sid = sorted(MANUF[_tag]['ops'])[0]
_df = get_ops(_tag, _sid)
_P, _info = physics_features(_df, pd.Series(True, index=_df.index))
print(f'{_tag}-{_sid} matched columns:', _info['cols'])
print(_P.describe().T.round(2))


# %% [6] Event screening: fault vs normal, physics vs raw ----------------------
TEST_DAYS = 7
PRE_DIST_H, POST_DIST_H = 48, 24          # exclusion around disturbances in training data

def build_events():
    ev = []
    for _, r in faults.iterrows():
        if 'efd_possible' in faults and not r['efd_possible']:
            continue
        ev.append(dict(manufacturer=r['manufacturer'], substation_id=int(r['substation_id']),
                       kind='fault', t_end=r.get('report_date'),
                       t_start=r.get('report_date') - pd.Timedelta(days=TEST_DAYS),
                       train_start=r.get('training_start'), train_end=r.get('training_end'),
                       mp_class=r['mp_class'], problem=r.get('problem_en'), label=r.get('fault_label'),
                       anomaly_start=r.get('possible_anomaly_start')))
    for _, r in normals.iterrows():
        s, e_end = r.get('event_start'), r.get('event_end')
        if pd.isna(s):
            continue
        e = s + pd.Timedelta(days=TEST_DAYS)
        if pd.notna(e_end) and e_end < e:
            e = e_end
        ev.append(dict(manufacturer=r['manufacturer'], substation_id=int(r['substation_id']),
                       kind='normal', t_start=s, t_end=e,
                       train_start=r.get('training_start'), train_end=r.get('training_end'),
                       mp_class='n/a', problem=None, label=None, anomaly_start=pd.NaT))
    ev = pd.DataFrame(ev)
    # paper default if training window missing: last 2 years up to 2 weeks before the test window
    miss = ev['train_end'].isna()
    ev.loc[miss, 'train_end'] = ev.loc[miss, 't_start'] - pd.Timedelta(days=14)
    miss = ev['train_start'].isna()
    ev.loc[miss, 'train_start'] = ev.loc[miss, 'train_end'] - pd.Timedelta(days=730)
    return ev.dropna(subset=['t_start', 't_end'])

def train_mask_for(df, tag, sid, t0, t1):
    m = (df.index >= t0) & (df.index <= t1)
    d = MANUF[tag]['dist']
    col = next((c for c in d.columns if c.startswith('event_start')), None)
    if col:
        for t in d.loc[d['substation_id'] == sid, col].dropna():
            m &= ~((df.index >= t - pd.Timedelta(hours=PRE_DIST_H)) &
                   (df.index <= t + pd.Timedelta(hours=POST_DIST_H)))
    return pd.Series(m, index=df.index)

def shift_metrics(train, test):
    train, test = train.dropna().values, test.dropna().values
    if len(train) < 48 or len(test) < 12 or np.unique(train).size <= 3:
        return np.nan, np.nan
    q01, q99 = np.percentile(train, [1, 99])
    med = np.median(train)
    iqr = np.subtract(*np.percentile(train, [75, 25]))
    scale = iqr / 1.349 if iqr > 1e-9 else np.std(train)
    z = (np.median(test) - med) / scale if scale > 0 else np.nan
    exceed = np.mean((test < q01) | (test > q99))
    return z, exceed

events = build_events()
print(events.groupby(['manufacturer', 'kind']).size())

recs = []
for (tag, sid), grp in events.groupby(['manufacturer', 'substation_id']):
    if sid not in MANUF[tag]['ops']:
        continue
    df = get_ops(tag, sid)
    raw_cols = [c for c in df.select_dtypes('number').columns]
    for _, e in grp.iterrows():
        tmask = train_mask_for(df, tag, sid, e['train_start'], e['train_end'])
        if tmask.sum() < 14 * 144 * 0.5:          # < ~7 days of usable training data
            recs.append({**e.to_dict(), 'status': 'insufficient_training'}); continue
        P, info = physics_features(df, tmask)
        H = pd.concat([df[raw_cols], P], axis=1).resample('1h').mean()
        tr_h = H[(H.index >= e['train_start']) & (H.index <= e['train_end'])]
        tr_h = tr_h.loc[tmask.resample('1h').max().reindex(tr_h.index).fillna(False).astype(bool)]
        te_h = H[(H.index >= e['t_start']) & (H.index < e['t_end'])]
        if len(te_h.dropna(how='all')) < 24:
            recs.append({**e.to_dict(), 'status': 'no_test_data'}); continue
        row = {**e.to_dict(), 'status': 'ok', 'k_meter': info.get('k_meter', np.nan)}
        for col in H.columns:
            z, ex = shift_metrics(tr_h[col], te_h[col])
            pref = 'phys' if col in P.columns else 'raw'
            row[f'{pref}__{col}__z'] = z
            row[f'{pref}__{col}__exc'] = ex
        phys_exc = [v for k, v in row.items() if k.startswith('phys__') and k.endswith('__exc')]
        raw_exc = [v for k, v in row.items() if k.startswith('raw__') and k.endswith('__exc')]
        row['score_phys_max'] = np.nanmax(phys_exc) if np.isfinite(phys_exc).any() else np.nan
        row['score_raw_max'] = np.nanmax(raw_exc) if np.isfinite(raw_exc).any() else np.nan
        row['score_phys_mean'] = np.nanmean(phys_exc) if np.isfinite(phys_exc).any() else np.nan
        row['score_raw_mean'] = np.nanmean(raw_exc) if np.isfinite(raw_exc).any() else np.nan
        recs.append(row)

screen = pd.DataFrame(recs)
screen.to_csv(OUT_DIR / 'event_screening.csv', index=False)
ok = screen[screen['status'] == 'ok'].copy()
print(screen['status'].value_counts())

def auroc(df, score):
    d = df.dropna(subset=[score])
    y = (d['kind'] == 'fault').astype(int)
    return (roc_auc_score(y, d[score]) if y.nunique() == 2 else np.nan), len(d)

res = []
for subset, d in [('all', ok), ('M1', ok[ok.manufacturer == 'M1']), ('M2', ok[ok.manufacturer == 'M2']),
                  ('high-MP faults only', ok[(ok.kind == 'normal') | (ok.mp_class == 'high')])]:
    for s in ['score_raw_max', 'score_phys_max', 'score_raw_mean', 'score_phys_mean']:
        a, n = auroc(d, s)
        res.append(dict(subset=subset, score=s, AUROC=a, n_events=n))
    for c in [c for c in ok.columns if c.startswith('phys__') and c.endswith('__exc')]:
        a, n = auroc(d, c)
        res.append(dict(subset=subset, score=c, AUROC=a, n_events=n))
res = pd.DataFrame(res)
res.to_csv(OUT_DIR / 'screening_auroc.csv', index=False)
print(res.pivot_table(index='score', columns='subset', values='AUROC').round(3))
# Interpretation: AUROC of how well a 7-day distribution shift separates pre-report fault
# windows from normal windows. Small n -> bootstrap CIs before using any number in the paper.


# %% [6b] Bootstrap confidence intervals for the key comparison ----------------
rng = np.random.default_rng(42)
def boot_auc(d, score, B=2000):
    d = d.dropna(subset=[score]); y = (d['kind'] == 'fault').astype(int).values; s = d[score].values
    vals = []
    if len(d) < 10 or len(np.unique(y)) < 2:
        return np.array([np.nan] * 3)
    for _ in range(B):
        i = rng.integers(0, len(d), len(d))
        if len(np.unique(y[i])) == 2:
            vals.append(roc_auc_score(y[i], s[i]))
    return np.percentile(vals, [2.5, 50, 97.5])

for s in ['score_raw_max', 'score_phys_max', 'score_raw_mean', 'score_phys_mean']:
    lo, med, hi = boot_auc(ok, s)
    print(f'{s:18s} AUROC median {med:.3f}  95% CI [{lo:.3f}, {hi:.3f}]')


# %% [7] Per-fault event plots -------------------------------------------------
PLOT_PRE_DAYS, PLOT_POST_DAYS = 14, 3

def plot_event(e):
    tag, sid = e['manufacturer'], e['substation_id']
    df = get_ops(tag, sid)
    tmask = train_mask_for(df, tag, sid, e['train_start'], e['train_end'])
    P, info = physics_features(df, tmask)
    t0 = e['t_end'] - pd.Timedelta(days=PLOT_PRE_DAYS)
    t1 = e['t_end'] + pd.Timedelta(days=PLOT_POST_DAYS)
    W = P[(P.index >= t0) & (P.index <= t1)]
    if W.dropna(how='all').empty:
        return
    trP = P[tmask]
    fig, axes = plt.subplots(len(W.columns), 1, figsize=(12, 2.0 * len(W.columns)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, W.columns):
        ax.plot(W.index, W[col], lw=0.7, color='k')
        lo, hi = np.nanpercentile(trP[col], [1, 99]) if trP[col].notna().sum() > 50 else (np.nan, np.nan)
        if np.isfinite(lo):
            ax.axhspan(lo, hi, color='tab:green', alpha=0.12, label='train 1-99 %')
        ax.axvline(e['t_end'], color='r', lw=1.2, label='incident report')
        ax.axvspan(e['t_start'], e['t_end'], color='gold', alpha=0.15, label='7-d test window')
        if pd.notna(e.get('anomaly_start')):
            ax.axvline(e['anomaly_start'], color='tab:purple', ls='--', lw=1, label='possible anomaly start')
        ax.set_ylabel(col, fontsize=8)
    axes[0].legend(fontsize=7, ncol=4, loc='upper left')
    axes[0].set_title(f"{tag}-{sid} | {e['problem']} | {e['label']} | MP={e['mp_class']}", fontsize=9)
    plt.tight_layout()
    fname = f"{tag}_sub{sid}_{pd.Timestamp(e['t_end']):%Y%m%d}.png"
    plt.savefig(FIG_DIR / 'events' / fname, dpi=200); plt.close(fig)

for _, e in events[events.kind == 'fault'].iterrows():
    try:
        plot_event(e)
    except Exception as ex:
        print('plot failed', e['manufacturer'], e['substation_id'], ex)
print('Event plots saved to', FIG_DIR / 'events')
print('All outputs in', OUT_DIR)
