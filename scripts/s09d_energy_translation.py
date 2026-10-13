# %% [9d] Energy translation of the comfort-efficiency gap ---------------------
# Run AFTER stage 9c in the same session (run_all.py) (uses F, design, pooled_full, c4, MANUF,
# get_ops, find_col, NAMES, rho_water, cp_water, OUT_DIR, FIG_DIR).
#
# For every substation the hierarchical model gives a REFERENCE return temperature
# (predicted primary return at one common operating point). Its gap to a benchmark is
# the substation's persistent cooling deficit at equal conditions. Counterfactual:
#   T_pr* = max(T_pr - gap, T_FLOOR)            same supply temperature, same heat
#   V*    = V * (T_ps - T_pr) / (T_ps - T_pr*)  flow needed to deliver the same heat
# Outputs per substation (annualised over its operating hours):
#   heat delivered [MWh/a], circulated volume [m3/a], excess volume [m3/a, %],
#   flow-weighted excess return temperature [K], pumping electricity and CO2 ranges.
# Assumption (state it in the paper): the cooling deficit is a constant offset across
# operating conditions, i.e. the intercept gap of the hierarchical model.
#
# !!! E_PUMP and GRID_CO2 below are ILLUSTRATIVE PLACEHOLDERS. Replace them with values
# !!! from the district-heating literature / the national grid factor for the study
# !!! period, and cite the sources. Results scale linearly with both.
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

E_PUMP = [0.1, 0.3, 0.5]          # kWh_el per m3 circulated (low / mid / high) -- PLACEHOLDER
GRID_CO2 = 0.38                   # kg CO2 per kWh_el -- PLACEHOLDER
T_FLOOR = 20.0                    # degC, physical floor for counterfactual return temperature
BENCHMARKS = ['config_median', 'fleet_median', 'best_quartile']
MAIN_BENCH = 'config_median'      # fairest: compares a substation with its own configuration type
AUTO_FIX_FLOW_UNITS = False       # keep False: a ratio far from 1 can come from flow OR power units
UNIT_OVERRIDE = {}                # e.g. {'M2': {'flow': 1e-6, 'power': 1e-3}} to force factors
COMMISSIONING_DAYS = 30

FLOW_TO_M3H = {'m3/h': 1.0, 'm³/h': 1.0, 'l/h': 1e-3, 'ml/h': 1e-6, 'l/s': 3.6, 'm3/s': 3600.0}
POWER_TO_KW = {'kw': 1.0, 'w': 1e-3, 'mw': 1e3}

# --------------------------------------------------------------- unit handling
def unit_of(tag, colname):
    fd = MANUF[tag]['feat'].copy()
    fd.columns = [str(c).strip().lower() for c in fd.columns]
    ccol = next((c for c in fd.columns if c in ('column', 'feature', 'name', 'feature_name')), fd.columns[0])
    ucol = next((c for c in fd.columns if 'unit' in c), None)
    row = fd[fd[ccol].astype(str).str.strip() == colname]
    return str(row[ucol].iloc[0]).strip() if (ucol and len(row)) else ''

FV, FQ = {}, {}
for tag in MANUF:
    sid0 = sorted(MANUF[tag]['ops'])[0]
    df0 = get_ops(tag, sid0)
    cv, cq = find_col(df0, 'V'), find_col(df0, 'Q')
    uv, uq = unit_of(tag, cv), unit_of(tag, cq)
    FV[tag] = FLOW_TO_M3H.get(uv.replace(' ', '').lower().replace('³', '3'), np.nan)
    FQ[tag] = POWER_TO_KW.get(uq.replace(' ', '').lower(), np.nan)
    if np.isnan(FV[tag]):
        FV[tag] = 1.0
    if np.isnan(FQ[tag]):
        FQ[tag] = 1.0
    FV[tag] = UNIT_OVERRIDE.get(tag, {}).get('flow', FV[tag])
    FQ[tag] = UNIT_OVERRIDE.get(tag, {}).get('power', FQ[tag])
    print(f'{tag}: flow column {cv!r} unit {uv!r} -> x{FV[tag]:g} to m3/h | '
          f'power column {cq!r} unit {uq!r} -> x{FQ[tag]:g} to kW')

def op_frame(key):
    """Hourly operating data in SI-consistent units (m3/h, kW, degC)."""
    df = get_ops(*key)
    c = {k: find_col(df, k) for k in NAMES}
    if not all(c[k] for k in ('T_ps', 'T_pr', 'V', 'Q')):
        return None
    H = pd.DataFrame({'T_ps': df[c['T_ps']], 'T_pr': df[c['T_pr']],
                      'V': df[c['V']] * FV[key[0]], 'Q': df[c['Q']] * FQ[key[0]]}).resample('1h').mean()
    H = H[H.index >= H.index.min() + pd.Timedelta(days=COMMISSIONING_DAYS)].dropna()
    return H[(H['T_ps'] - H['T_pr']).between(0.5, 80) & (H['V'] > 0) & (H['Q'] > 0)]

# Energy-balance unit check: Q [kW] should equal V/3600 * rho * cp * dT
OPS = {key: op_frame(key) for key in F}
for tag in MANUF:
    ratios = []
    for key, H in OPS.items():
        if key[0] != tag or H is None or len(H) < 100:
            continue
        dT = H['T_ps'] - H['T_pr']
        q_calc = H['V'] / 3600 * rho_water(H['T_pr']) * cp_water(0.5 * (H['T_ps'] + H['T_pr'])) * dT
        ratios.append(np.nanmedian(H['Q'] / q_calc))
    r = float(np.nanmedian(ratios)) if ratios else np.nan
    print(f'{tag}: energy-balance ratio Q_meter / Q_calc = {r:.3g} (should be ~1)')
    if np.isfinite(r) and not (0.5 < r < 2.0) and not AUTO_FIX_FLOW_UNITS:
        raise ValueError(f'{tag}: energy balance off by ~{r:.3g}. Check the flow and power units in '
                         f'feature_descriptions.csv and set UNIT_OVERRIDE, e.g. '
                         f"UNIT_OVERRIDE = {{'{tag}': {{'flow': 1e-6, 'power': 1e-3}}}}, then rerun this cell.")
    if AUTO_FIX_FLOW_UNITS and np.isfinite(r) and not (0.5 < r < 2.0):
        fix = 10 ** np.round(np.log10(r))
        FV[tag] *= fix
        print(f'   -> flow factor rescaled by {fix:g}; check units before publishing')
        for key in [k for k in OPS if k[0] == tag]:
            OPS[key] = op_frame(key)

# ---------------------------------------------------------- reference temperatures
REF_X = np.zeros(F[next(iter(F))]['b'].shape[0]); REF_X[0] = 1.0
ref = pd.DataFrame([dict(manufacturer=k[0], substation_id=k[1], configuration=F[k]['cfg'],
                         ref_return_C=float(REF_X @ pooled_full(F, k))) for k in F])
ref['bench_fleet_median'] = ref['ref_return_C'].median()
ref['bench_config_median'] = ref.groupby('configuration')['ref_return_C'].transform('median')
ref['bench_best_quartile'] = ref['ref_return_C'].quantile(0.25)

# ----------------------------------------------------------------- counterfactuals
rows = []
for _, r in ref.iterrows():
    key = (r['manufacturer'], r['substation_id'])
    H = OPS.get(key)
    if H is None or len(H) < 500:
        continue
    scale = 8760.0 / len(H)                            # annualise over operating hours
    dT = (H['T_ps'] - H['T_pr']).values
    out = dict(manufacturer=key[0], substation_id=key[1], configuration=r['configuration'],
               ref_return_C=r['ref_return_C'], operating_hours=len(H),
               heat_MWh_a=H['Q'].sum() / 1000 * scale, volume_m3_a=H['V'].sum() * scale,
               mean_dT_K=float(np.average(dT, weights=H['V'])))
    for bench in BENCHMARKS:
        gap = max(r['ref_return_C'] - r[f'bench_{bench}'], 0.0)
        Tpr_star = np.maximum(H['T_pr'].values - gap, T_FLOOR)
        dT_star = H['T_ps'].values - Tpr_star
        V_star = H['V'].values * dT / dT_star
        exc_V = (H['V'].values - V_star).sum() * scale
        out[f'gap_K_{bench}'] = gap
        out[f'excess_volume_m3_a_{bench}'] = exc_V
        out[f'excess_volume_pct_{bench}'] = 100 * exc_V / out['volume_m3_a']
        out[f'fw_excess_return_K_{bench}'] = float(np.average(H['T_pr'].values - Tpr_star, weights=H['V']))
        for e in E_PUMP:
            out[f'pump_kWh_a_{bench}_e{e}'] = exc_V * e
        out[f'co2_kg_a_{bench}_mid'] = exc_V * E_PUMP[1] * GRID_CO2
    rows.append(out)

en = pd.DataFrame(rows).merge(c4[['manufacturer', 'substation_id', 'reports_per_year', 'reports']],
                              on=['manufacturer', 'substation_id'], how='left')
en.to_csv(OUT_DIR / 'hm9d_energy_per_substation.csv', index=False)

# ------------------------------------------------------------------- fleet summary
summ = []
for bench in BENCHMARKS:
    exc = en[f'excess_volume_m3_a_{bench}']
    summ.append(dict(benchmark=bench, substations=len(en),
                     with_deficit=int((en[f'gap_K_{bench}'] > 0).sum()),
                     fleet_heat_MWh_a=en['heat_MWh_a'].sum(), fleet_volume_m3_a=en['volume_m3_a'].sum(),
                     fleet_excess_volume_m3_a=exc.sum(),
                     fleet_excess_volume_pct=100 * exc.sum() / en['volume_m3_a'].sum(),
                     median_gap_of_deficit_K=en.loc[en[f'gap_K_{bench}'] > 0, f'gap_K_{bench}'].median(),
                     **{f'pump_MWh_a_e{e}': en[f'pump_kWh_a_{bench}_e{e}'].sum() / 1000 for e in E_PUMP},
                     co2_t_a_mid=en[f'co2_kg_a_{bench}_mid'].sum() / 1000))
summ = pd.DataFrame(summ)
summ.to_csv(OUT_DIR / 'hm9d_energy_fleet_summary.csv', index=False)
print('\nFleet summary (annualised; pumping/CO2 use PLACEHOLDER factors):')
print(summ.round(2).T.to_string())

# -------------------------------------------- concentration + link to complaints
col = f'excess_volume_m3_a_{MAIN_BENCH}'
p = en[en[col] > 0].sort_values(col, ascending=False).reset_index(drop=True)
tot = p[col].sum()
p['cum_share'] = p[col].cumsum() / tot
p['sub_share'] = (np.arange(len(p)) + 1) / len(en)
xc = np.concatenate([[0], p['sub_share'].values]); yc = np.concatenate([[0], p['cum_share'].values])
share_top20 = float(np.interp(0.2, xc, yc)) if tot > 0 else np.nan     # exactly 20 % of the fleet
no_rep = p['reports'].fillna(0) == 0
low_rep = p['reports_per_year'] <= p.groupby('manufacturer')['reports_per_year'].transform('median')
print(f'\nConcentration ({MAIN_BENCH}): top 20 % of substations (of {len(en)}) cause '
      f'{100 * share_top20:.0f} % of the fleet excess circulated volume')
print(f'Share of fleet excess volume from substations with NO fault report: '
      f'{100 * p.loc[no_rep, col].sum() / tot:.0f} %  ({int(no_rep.sum())} of {len(p)} deficit substations)')
print(f'Share from substations at or below their manufacturer median complaint rate: '
      f'{100 * p.loc[low_rep, col].sum() / tot:.0f} %')
print('\nTen largest contributors:')
print(p[['manufacturer', 'substation_id', 'configuration', 'ref_return_C', f'gap_K_{MAIN_BENCH}',
         f'excess_volume_pct_{MAIN_BENCH}', col, 'reports_per_year']].head(10).round(2).to_string(index=False))

# Figure: Lorenz-type concentration curve; marker shape/fill = has the substation ever complained
fig, ax = plt.subplots(figsize=(6.0, 4.2))
ax.plot(xc * 100, yc * 100, color='#2a78d6', lw=2, zorder=2)
ax.scatter(p.loc[no_rep, 'sub_share'] * 100, p.loc[no_rep, 'cum_share'] * 100, s=30, marker='o',
           facecolor='white', edgecolor='#2a78d6', linewidth=1.4, zorder=3, label='No fault report ever')
ax.scatter(p.loc[~no_rep, 'sub_share'] * 100, p.loc[~no_rep, 'cum_share'] * 100, s=30, marker='s',
           color='#eb6834', edgecolor='white', linewidth=0.8, zorder=3, label='At least one report')
ax.plot([0, 100], [0, 100], color='#8a8a85', lw=1, ls=(0, (4, 3)))
ax.text(97, 92, 'equal contribution', ha='right', fontsize=8, color='#5d5d58')
ax.axvline(20, color='#d4d3cd', lw=1)
ax.scatter([20], [100 * share_top20], s=18, color='#2b2b28', zorder=4)
ax.annotate(f'top 20 % of substations:\n{100 * share_top20:.0f} % of excess flow', xy=(20, 100 * share_top20),
            xytext=(34, 100 * share_top20 - 22), fontsize=8, color='#2b2b28', va='top',
            arrowprops=dict(arrowstyle='-', color='#8a8a85', lw=0.8))
ax.set_xlim(0, 100); ax.set_ylim(0, 102)
ax.set_xlabel('Substations ranked by excess circulated volume [% of fleet]')
ax.set_ylabel('Cumulative share of fleet excess volume [%]')
ax.legend(frameon=False, fontsize=8, loc='lower right')
ax.grid(color='#ecebe6', lw=0.8); ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(FIG_DIR / 'hm9d_excess_flow_concentration.png', dpi=300, bbox_inches='tight')
plt.show()
print('\nSaved hm9d_energy_per_substation.csv, hm9d_energy_fleet_summary.csv and '
      'figures/hm9d_excess_flow_concentration.png in', OUT_DIR)
