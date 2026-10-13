# %% [6c] Fair physics-vs-raw comparison (paired bootstrap) --------------------
# Add after stage 6b. Reads event_screening.csv, so stage 6 need not be re-run.
#
# Fixes three biases in the first screening table:
#   1. Equal information: raw score uses ONLY the core signals the physics features are
#      built from (same columns for every substation), and both scores are means.
#   2. Paired test: bootstrap the AUROC *difference* on the same resampled events.
#   3. Detectability: drop faults present from commissioning ('Control unit: Incorrect
#      parameterisation'), as in Roelofs et al. (2026), Sec. 5.2.
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

from config import OUT_DIR
scr = pd.read_csv(OUT_DIR / 'event_screening.csv')
scr = scr[scr['status'] == 'ok'].copy()

CORE_RAW = ['p_net_supply_temperature', 'p_net_return_temperature', 'p_net_meter_flow',
            'p_net_meter_heat_power', 'p_hc1_return_temperature', 's_hc1_supply_temperature',
            's_hc1_supply_temperature_setpoint', 'outdoor_temperature']
CORE_PHYS = ['dT_primary', 'TTD_hot', 'track_err_hc1', 'meter_balance_z', 'heating_curve_res']
BEST_PHYS = ['dT_primary', 'TTD_hot']                 # the two features carrying the signal
BEST_RAW = ['p_net_supply_temperature', 'p_net_return_temperature', 's_hc1_supply_temperature']
# BEST_RAW = the raw signals that dT_primary and TTD_hot are computed from

def mean_exc(df, prefix, names):
    cols = [f'{prefix}__{n}__exc' for n in names if f'{prefix}__{n}__exc' in df]
    return df[cols].mean(axis=1, skipna=True)

scr['raw_core_mean'] = mean_exc(scr, 'raw', CORE_RAW)
scr['phys_core_mean'] = mean_exc(scr, 'phys', CORE_PHYS)
scr['raw_best_mean'] = mean_exc(scr, 'raw', BEST_RAW)
scr['phys_best_mean'] = mean_exc(scr, 'phys', BEST_PHYS)

# Normalise labels (e.g. 'noise' vs 'Noise')
for c in ('problem', 'label'):
    if c in scr:
        scr[c] = scr[c].astype(str).str.strip()
scr['problem'] = scr['problem'].str.lower()

COMMISSIONING = scr['label'].str.contains('Incorrect parameterisation', case=False, na=False)
subsets = {
    'all': scr,
    'M1': scr[scr.manufacturer == 'M1'],
    'M2': scr[scr.manufacturer == 'M2'],
    'excl. commissioning faults': scr[~COMMISSIONING],
    'detectable high-MP': scr[(scr.kind == 'normal') | ((scr.mp_class == 'high') & ~COMMISSIONING)],
}

from scipy.stats import rankdata
rng = np.random.default_rng(42)

def fast_auc(y, s):
    """AUROC via Mann-Whitney U (identical to roc_auc_score, ties averaged)."""
    n1 = y.sum(); n0 = len(y) - n1
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def paired_boot(d, a, b, B=5000):
    d = d.dropna(subset=[a, b])
    y = (d['kind'] == 'fault').astype(int).values
    sa, sb = d[a].values, d[b].values
    if len(np.unique(y)) < 2:
        return dict(n=len(d))
    auc_a, auc_b = fast_auc(y, sa), fast_auc(y, sb)
    assert abs(auc_a - roc_auc_score(y, sa)) < 1e-9
    diffs = []
    for _ in range(B):
        i = rng.integers(0, len(d), len(d))
        yi = y[i]
        if 0 < yi.sum() < len(yi):
            diffs.append(fast_auc(yi, sa[i]) - fast_auc(yi, sb[i]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))   # two-sided bootstrap p
    return dict(n=len(d), n_fault=int(y.sum()), AUROC_phys=auc_a, AUROC_raw=auc_b,
                diff=auc_a - auc_b, CI_lo=lo, CI_hi=hi, p_boot=max(p, 1 / len(diffs)))

rows = []
for name, d in subsets.items():
    for pa, pb, tag in [('phys_core_mean', 'raw_core_mean', 'core (5 phys vs 8 raw)'),
                        ('phys_best_mean', 'raw_best_mean', 'heat-exchanger (2 phys vs 3 raw)')]:
        rows.append(dict(subset=name, comparison=tag, **paired_boot(d, pa, pb)))
fair = pd.DataFrame(rows)
fair.to_csv(OUT_DIR / 'screening_fair_comparison.csv', index=False)
print(fair.round(3).to_string(index=False))
# Read: a physics advantage is supported when CI_lo > 0 for that subset.
