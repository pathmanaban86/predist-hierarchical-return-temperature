# =============================================================================
# PreDist / IMechE Part A -- publication figures and tables
# Reads the CSV outputs of predist_analysis.py stages 2-9d from OUT_DIR, so it can be
# re-run on its own (no recomputation): python scripts/s11_paper_figures_tables.py
#
# Style: white background, no grids, top/right spines off, minimal text, STIX serif
# fonts matching LaTeX, one symbol set (SYM) used identically in figures and tables,
# fixed colour/marker per entity across all figures.
# Output: OUT_DIR/paper_figures/*.pdf (vector) + *.png (600 dpi)
#         OUT_DIR/paper_tables/*.tex (booktabs) + *.csv
# Widths follow SAGE/IMechE: single column 84 mm, double column 174 mm.
# =============================================================================
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon, binomtest, fisher_exact

from config import OUT_DIR
FIG = OUT_DIR / 'paper_figures'
TAB = OUT_DIR / 'paper_tables'
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1)

# ----------------------------------------------------------------------- style
MM = 1 / 25.4
W1, W2 = 84 * MM, 174 * MM
mpl.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'axes.grid': False, 'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'legend.frameon': False, 'legend.handletextpad': 0.3,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'lines.linewidth': 1.0,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

# Colours (validated categorical palette) and fixed entity encodings
BLUE, ORANGE, AQUA, GREY, INK = '#2a78d6', '#eb6834', '#1baf7a', '#8a8a85', '#1a1a19'
MFR = {'M1': dict(marker='o', color=BLUE), 'M2': dict(marker='s', color=ORANGE)}
METHOD = {'own': dict(marker='s', color=ORANGE, label='Own'),
          'fleet': dict(marker='^', color=AQUA, label='Fleet'),
          'pooled': dict(marker='o', color=BLUE, label='Hierarchical')}
GROUP = {'normal': dict(color=GREY, label='Normal'),
         'comfort': dict(color=BLUE, label='Comfort fault'),
         'other': dict(color=AQUA, label='Other fault')}
CFG_ABBR = {'SH': 'SH', 'SH + DHW': 'SH+DHW', 'SH + DHW with sub-circuits': 'SH+DHW+sub',
            'SH with buffer tank': 'SH+buffer', 'SH with sub-circuits': 'SH+sub'}
CFG_ORDER = ['SH', 'SH+DHW', 'SH+DHW+sub', 'SH+buffer', 'SH+sub']

# One symbol set for figures AND tables (LaTeX math)
SYM = {
    'Tps': r'$T_{\mathrm{p,s}}$', 'Tpr': r'$T_{\mathrm{p,r}}$', 'Tss': r'$T_{\mathrm{s,s}}$',
    'To': r'$T_{\mathrm{o}}$', 'Q': r'$\dot{Q}$', 'V': r'$\dot{V}$',
    'dTp': r'$\Delta T_{\mathrm{p}}$', 'Tref': r'$T_{\mathrm{p,r}}^{\mathrm{ref}}$',
    'ex': r'$\Delta T_{\mathrm{p,r}}^{\mathrm{ex}}$', 'Vex': r'$\dot{V}^{\mathrm{ex}}$',
    'b0': r'$\beta_0$', 'b1': r'$\beta_1$', 'b2': r'$\beta_2$', 'b3': r'$\beta_3$', 'b4': r'$\beta_4$',
}
COEF = {'b0_return_at_ref': 'b0', 'b1_T_ss': 'b1', 'b2_approach': 'b2', 'b3_load': 'b3', 'b4_T_out': 'b4'}
COEF_UNIT = {'b0': r'$^{\circ}$C', 'b1': '--', 'b2': '--', 'b3': 'K', 'b4': '--'}

COMFORT = {'no heat', 'not enough heat', 'no dhw'}
MIN_CFG_N = 3                      # configurations with fewer substations: 'All' row only

def load(name, **kw):
    p = OUT_DIR / name
    if not p.exists():
        print(f'  [skip] missing {name}')
        return None
    return pd.read_excel(p, **kw) if p.suffix == '.xlsx' else pd.read_csv(p)

def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf'); fig.savefig(FIG / f'{name}.png')
    plt.close(fig); print(f'  figure  {name}')

def tag(ax, letter, x=-0.16, y=1.02):
    ax.text(x, y, f'({letter})', transform=ax.transAxes, fontsize=8, va='bottom', ha='left')

def cfg(s):
    return pd.Series(s).astype(str).map(lambda v: CFG_ABBR.get(v, v)).values

def tex_escape(v):
    s = str(v)
    if s.startswith('$') and s.endswith('$'):
        return s
    parts = re.split(r'(\$[^$]*\$)', s)       # keep inline math untouched
    return ''.join(p if p.startswith('$') else
                   p.replace('&', r'\&').replace('%', r'\%').replace('#', r'\#').replace('_', r'\_')
                   for p in parts)

def table(df, name, caption, label, align=None, note=None):
    df.to_csv(TAB / f'{name}.csv', index=False)
    align = align or ('l' + 'r' * (df.shape[1] - 1))
    lines = [r'\begin{table}[!t]', r'\centering', rf'\caption{{{caption}}}', rf'\label{{{label}}}',
             rf'\begin{{tabular}}{{{align}}}', r'\toprule',
             ' & '.join(tex_escape(c) for c in df.columns) + r' \\', r'\midrule']
    for _, r in df.iterrows():
        lines.append(' & '.join(tex_escape(v) for v in r.values) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    if note:
        lines.append(rf'\par\smallskip\footnotesize{{{tex_escape(note)}}}')
    lines.append(r'\end{table}')
    (TAB / f'{name}.tex').write_text('\n'.join(lines))
    print(f'  table   {name}')

def fmt(x, d=2):
    return '--' if pd.isna(x) else f'{x:.{d}f}'

def fmt_p(p):
    return '--' if pd.isna(p) else ('<0.001' if p < 0.001 else f'{p:.3f}')

# ------------------------------------------------------------------------ inputs
print('Loading results from', OUT_DIR)
inv = load('substation_inventory.csv')
fprob = load('fault_summary.xlsx', sheet_name='problem_category')
coef = load('hm_full_coefficients.csv')
eff = load('hm9e_data_efficiency_summary.csv')          # selected (extended) design, preferred
if eff is None:
    eff = load('hm_data_efficiency_summary.csv')
    print('  [note] Fig03/T3 use the stage-9 base design; run stage 9e for consistency')
choice = load('hm9c_model_choice.csv')
resid = load('hm9c_residual_by_load_decile.csv')
exc = load('hm9c_fault_excess_matched.csv')
c4 = load('hm9c_cooling_vs_complaints.csv')
rho_s = load('hm9c_rho_cap_sensitivity.csv')
en = load('hm9d_energy_per_substation.csv')
ensum = load('hm9d_energy_fleet_summary.csv')
fair = load('screening_fair_comparison.csv')
trans = load('transfer_limited_auroc.csv')

# =============================================================== TABLE 0: symbols
nom = pd.DataFrame([
    (SYM['Tps'], 'Primary supply temperature', r'$^{\circ}$C'),
    (SYM['Tpr'], 'Primary return temperature', r'$^{\circ}$C'),
    (SYM['Tss'], 'Secondary supply temperature (space heating)', r'$^{\circ}$C'),
    (SYM['To'], 'Outdoor temperature', r'$^{\circ}$C'),
    (SYM['dTp'], r'Primary cooling, $T_{\mathrm{p,s}}-T_{\mathrm{p,r}}$', 'K'),
    (SYM['Q'], 'Heat rate (energy meter)', 'kW'),
    (SYM['V'], 'Primary volume flow rate', r'm$^3$ h$^{-1}$'),
    (SYM['Tref'], 'Reference return temperature (model, common operating point)', r'$^{\circ}$C'),
    (SYM['ex'], 'Excess return temperature vs fault-free model', 'K'),
    (SYM['Vex'], 'Excess circulated volume vs benchmark', r'm$^3$ a$^{-1}$'),
    (r'$\beta_0$--$\beta_4$', 'Model coefficients (Eq. 1)', '--'),
], columns=['Symbol', 'Description', 'Unit'])
table(nom, 'T0_nomenclature', 'Nomenclature.', 'tab:nomenclature', align='lll')

# ============================================================ FIGURE 1: dataset
if inv is not None:
    inv['cfg'] = cfg(inv['configuration'])
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.1), gridspec_kw=dict(wspace=0.55))
    ax = axs[0]
    ct = inv.groupby(['cfg', 'manufacturer']).size().unstack(fill_value=0)
    ct = ct.reindex([c for c in CFG_ORDER if c in ct.index])[::-1]
    y = np.arange(len(ct)); h = 0.38
    for k, m in enumerate(['M1', 'M2']):
        if m in ct:
            ax.barh(y + (0.5 - k) * h, ct[m], height=h, color=MFR[m]['color'], label=m)
    ax.set_yticks(y); ax.set_yticklabels(ct.index); ax.set_xlabel('Substations'); tag(ax, 'a', -0.32)
    ax.legend(loc='lower right')
    if fprob is not None:
        ax = axs[1]
        fp = fprob.rename(columns={fprob.columns[0]: 'problem'})
        fp['problem'] = fp['problem'].astype(str).str.strip().str.lower()
        fp = fp.groupby('problem')[[c for c in ['M1', 'M2'] if c in fp]].sum()
        fp = fp.loc[fp.sum(axis=1).sort_values().index]
        y = np.arange(len(fp))
        for k, m in enumerate(['M1', 'M2']):
            if m in fp:
                ax.barh(y + (0.5 - k) * h, fp[m], height=h, color=MFR[m]['color'])
        ax.set_yticks(y); ax.set_yticklabels([p.replace('dhw', 'DHW') for p in fp.index])
        ax.set_xlabel('Fault reports'); tag(ax, 'b', -0.38)
    save(fig, 'Fig01_dataset')

    # TABLE 1: dataset summary
    rows = []
    for m, d in inv.groupby('manufacturer'):
        rows.append([m, len(d), fmt(d['span_years'].mean(), 1), fmt(d['span_years'].sum(), 0),
                     fmt(d['timestamp_completeness_%'].mean(), 1),
                     f"{int(d['n_features'].min())}--{int(d['n_features'].max())}",
                     int(d['n_faults'].sum()), int(d['n_normal_events'].sum())])
    rows.append(['Total', len(inv), fmt(inv['span_years'].mean(), 1), fmt(inv['span_years'].sum(), 0),
                 fmt(inv['timestamp_completeness_%'].mean(), 1),
                 f"{int(inv['n_features'].min())}--{int(inv['n_features'].max())}",
                 int(inv['n_faults'].sum()), int(inv['n_normal_events'].sum())])
    table(pd.DataFrame(rows, columns=['Manufacturer', 'Substations', 'Mean span (a)', 'Total span (a)',
                                      'Completeness (%)', 'Signals', 'Fault reports', 'Normal events']),
          'T1_dataset', 'PreDist dataset used in this study.', 'tab:dataset')

# ====================================================== FIGURE 2: coefficients
if coef is not None:
    coef = coef.rename(columns=COEF)
    coef['cfg'] = cfg(coef['configuration'])
    n_cfg = coef['cfg'].value_counts()
    present = [c for c in CFG_ORDER if n_cfg.get(c, 0) >= MIN_CFG_N]
    omitted = [f'{c} ($n={n_cfg[c]}$)' for c in CFG_ORDER if 0 < n_cfg.get(c, 0) < MIN_CFG_N]
    fig, axs = plt.subplots(1, 5, figsize=(W2, 1.9), sharey=True, gridspec_kw=dict(wspace=0.12))
    for i, (ax, b) in enumerate(zip(axs, ['b0', 'b1', 'b2', 'b3', 'b4'])):
        for m in ['M1', 'M2']:
            d = coef[(coef.manufacturer == m) & coef['cfg'].isin(present)]
            yy = np.array([present.index(c) if c in present else np.nan for c in d['cfg']], dtype=float)
            ax.scatter(d[b], yy + rng.uniform(-0.18, 0.18, len(d)), s=9, marker=MFR[m]['marker'],
                       color=MFR[m]['color'], edgecolor='white', linewidth=0.3, label=m)
        if b != 'b0' and coef[b].min() < 0 < coef[b].max():
            ax.axvline(0, color=GREY, lw=0.5, zorder=0)
        ax.set_xlabel(SYM[b] + (f' ({COEF_UNIT[b]})' if COEF_UNIT[b] != '--' else ''))
        if i > 0:
            ax.tick_params(left=False)
    axs[0].set_yticks(range(len(present))); axs[0].set_yticklabels(present)
    axs[0].invert_yaxis()
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h[:2], l[:2], ncol=2, loc='lower center', bbox_to_anchor=(0.5, 0.98))
    save(fig, 'Fig02_coefficients')

    # TABLE 2: median [IQR] per configuration
    rows = []
    for c in present:
        d = coef[coef.cfg == c]
        rows.append([c, len(d)] + [f"{d[b].median():.2f} [{d[b].quantile(.25):.2f}, {d[b].quantile(.75):.2f}]"
                                   for b in ['b0', 'b1', 'b2', 'b3', 'b4']]
                    + [fmt(d['resid_sd_K'].median(), 2)])
    rows.append(['All', len(coef)] + [f"{coef[b].median():.2f} [{coef[b].quantile(.25):.2f}, {coef[b].quantile(.75):.2f}]"
                                      for b in ['b0', 'b1', 'b2', 'b3', 'b4']] + [fmt(coef['resid_sd_K'].median(), 2)])
    table(pd.DataFrame(rows, columns=['Configuration', '$n$'] + [SYM[b] for b in ['b0', 'b1', 'b2', 'b3', 'b4']]
                       + [r'$\sigma_{\varepsilon}$ (K)']),
          'T2_coefficients', 'Substation-level model coefficients, median [interquartile range].',
          'tab:coefficients', align='lr' + 'c' * 5 + 'r',
          note=('Shown in All only: ' + ', '.join(omitted) + '.') if omitted else None)

# ===================================================== FIGURE 3: data efficiency
ORDER = ['7 d', '14 d', '30 d', '90 d', 'full']
XLAB = ['7', '14', '30', '90', 'Full']
if eff is not None:
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    x = np.arange(len(ORDER))
    for k, m in enumerate(['own', 'fleet', 'pooled']):
        q = eff[eff.method == m].set_index('train').reindex(ORDER)
        ax.errorbar(x + (k - 1) * 0.17, q['median_rmse_K'],
                    yerr=[q['median_rmse_K'] - q['ci_lo'], q['ci_hi'] - q['median_rmse_K']],
                    fmt=METHOD[m]['marker'], ms=4, lw=0.9, capsize=0, color=METHOD[m]['color'],
                    mec='white', mew=0.5, label=METHOD[m]['label'])
    ax.set_xticks(x); ax.set_xticklabels(XLAB)
    ax.set_xlabel('Own training history (d)'); ax.set_ylabel(f"RMSE of {SYM['Tpr']} (K)")
    ax.set_ylim(bottom=0)
    ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0), columnspacing=0.8)
    save(fig, 'Fig03_data_efficiency')

    # TABLE 3
    rows = []
    for t, lab in zip(ORDER, XLAB):
        g = eff[eff.train == t].set_index('method')
        if g.empty:
            continue
        cell = lambda m: f"{g.loc[m, 'median_rmse_K']:.2f} [{g.loc[m, 'ci_lo']:.2f}, {g.loc[m, 'ci_hi']:.2f}]" if m in g.index else '--'
        po = g.loc['pooled_vs_own'] if 'pooled_vs_own' in g.index else None
        rows.append([lab, int(g['n_substations'].iloc[0]), cell('own'), cell('fleet'), cell('pooled'),
                     fmt(po['median_rmse_K'], 3) if po is not None else '--',
                     fmt(100 * po['pooled_better_share'], 0) if po is not None else '--',
                     fmt_p(po['wilcoxon_p']) if po is not None else '--'])
    table(pd.DataFrame(rows, columns=['History (d)', '$n$', 'Own', 'Fleet', 'Hierarchical',
                                      r'$\Delta$ (K)', 'Better (%)', '$p$']),
          'T3_data_efficiency',
          rf"Out-of-time RMSE of {SYM['Tpr']} (K), median [95\% CI] across substations; "
          r"$\Delta$ = hierarchical $-$ own (median), Wilcoxon signed-rank $p$.",
          'tab:efficiency', align='lrcccrrr')

# ===================================================== FIGURE 4: residual vs load
if resid is not None:
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    x = resid['load_decile'].values
    styles = {'base': dict(color=GREY, marker='s', label='Base'),
              'extended': dict(color=BLUE, marker='o', label='Extended')}
    for k, col in enumerate([c for c in ['base', 'extended'] if c in resid]):
        ax.plot(x + (k - 0.5) * 0.15, resid[col], lw=0.9, ms=3.5, mec='white', mew=0.4, **styles[col])
    ax.axhline(0, color=INK, lw=0.5)
    ax.set_xticks(x); ax.set_xlabel(f"{SYM['Q']} decile"); ax.set_ylabel('Median residual (K)')
    ax.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    save(fig, 'Fig04_residual_load')

# ================================================= FIGURE 5: matched fault excess
if exc is not None:
    exc['g'] = np.where(exc['kind'] == 'normal', 'normal',
                        np.where(exc['problem'].astype(str).str.lower().isin(COMFORT), 'comfort', 'other'))
    nrm = exc.loc[exc.g == 'normal', 'excess_matched_K']
    lo, hi = np.percentile(nrm, [5, 95])
    fig, ax = plt.subplots(figsize=(W1, 1.7))
    ax.axvspan(lo, hi, color='#ecebe6', lw=0, zorder=0)
    ax.axvline(0, color=INK, lw=0.5, zorder=1)
    order = ['normal', 'comfort', 'other']
    for i, g in enumerate(order):
        v = exc.loc[exc.g == g, 'excess_matched_K'].values
        if len(v) == 0:
            continue
        ax.scatter(v, np.full(len(v), i) + rng.uniform(-0.17, 0.17, len(v)), s=9,
                   color=GROUP[g]['color'], edgecolor='white', linewidth=0.3, zorder=3)
        ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color=INK, lw=1.4, zorder=4)
    ax.set_yticks(range(3)); ax.set_yticklabels([GROUP[g]['label'] for g in order])
    ax.set_xlabel(f"{SYM['ex']} (K)"); ax.invert_yaxis()
    save(fig, 'Fig05_fault_excess')

    # TABLE 4: group statistics
    exc['side'] = np.where(exc.excess_matched_K < lo, 'below', np.where(exc.excess_matched_K > hi, 'above', 'within'))
    rows = []
    dn = exc[exc.g == 'normal']
    n_out_norm = int((dn.side != 'within').sum())
    for g in order:
        d = exc[exc.g == g]
        if d.empty:
            continue
        p_mw = mannwhitneyu(d.excess_matched_K, nrm).pvalue if g != 'normal' else np.nan
        out = d[d.side != 'within']
        kb = int((out.side == 'below').sum())
        p_s = binomtest(kb, len(out), 0.5).pvalue if (g != 'normal' and len(out)) else np.nan
        p_f = (fisher_exact([[len(out), len(d) - len(out)], [n_out_norm, len(dn) - n_out_norm]])[1]
               if g != 'normal' else np.nan)
        rows.append([GROUP[g]['label'], len(d), fmt(d.excess_matched_K.median()), fmt(d.excess_matched_K.mean()),
                     kb, int((d.side == 'within').sum()), int((d.side == 'above').sum()),
                     fmt(100 * len(out) / len(d), 0), fmt_p(p_mw), fmt_p(p_s), fmt_p(p_f)])
    table(pd.DataFrame(rows, columns=['Group', '$n$', 'Median (K)', 'Mean (K)', 'Below', 'Within', 'Above',
                                      'Outside (%)', r'$p_{\mathrm{MW}}$', r'$p_{\mathrm{sign}}$',
                                      r'$p_{\mathrm{F}}$']),
          'T4_fault_excess',
          rf"Load- and weather-matched {SYM['ex']} in the 7 d before each report. Below/within/above: "
          rf"relative to the 5--95\% range of normal events [{lo:.1f}, {hi:.1f}] K. "
          r"$p_{\mathrm{MW}}$: Mann--Whitney vs normal events; $p_{\mathrm{sign}}$: sign test below vs above; "
          r"$p_{\mathrm{F}}$: Fisher exact test, share outside the band vs normal events.",
          'tab:excess', align='lrrrrrrrrrr')

    # TABLE 5: by fault label
    lab = (exc[exc.kind == 'fault'].groupby('label')['excess_matched_K']
           .agg(['count', 'median', 'mean']).sort_values('median').reset_index())
    lab.columns = ['Fault label', '$n$', 'Median (K)', 'Mean (K)']
    for c_ in ['Median (K)', 'Mean (K)']:
        lab[c_] = lab[c_].map(fmt)
    table(lab, 'T5_fault_excess_by_label', rf"{SYM['ex']} by fault label (matched, 7 d before report).",
          'tab:excess_label', align='p{0.55\\linewidth}rrr')

# ====================================== FIGURE 6: validation + complaints (2 panels)
if en is not None or c4 is not None:
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.2), gridspec_kw=dict(wspace=0.32))
    if en is not None:
        ax = axs[0]
        for m in ['M1', 'M2']:
            d = en[en.manufacturer == m]
            ax.scatter(d['ref_return_C'], d['mean_dT_K'], s=12, marker=MFR[m]['marker'],
                       color=MFR[m]['color'], edgecolor='white', linewidth=0.4, label=m)
        r, p = spearmanr(en['ref_return_C'], en['mean_dT_K'])
        ax.set_xlabel(f"{SYM['Tref']} ($^{{\\circ}}$C)"); ax.set_ylabel(f"Flow-weighted {SYM['dTp']} (K)")
        ax.text(0.98, 0.97, rf'$\rho_s={r:.2f}$', transform=ax.transAxes, ha='right', va='top')
        tag(ax, 'a')
        h, l = ax.get_legend_handles_labels()
        fig.legend(h, l, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 0.98))
    if c4 is not None:
        ax = axs[1]
        for m in ['M1', 'M2']:
            d = c4[c4.manufacturer == m]
            ax.scatter(d['ref_return_C'], d['reports_per_year'], s=12, marker=MFR[m]['marker'],
                       color=MFR[m]['color'], edgecolor='white', linewidth=0.4)
        r, p = spearmanr(c4['ref_return_rank'], c4['report_rank'])
        ax.set_xlabel(f"{SYM['Tref']} ($^{{\\circ}}$C)"); ax.set_ylabel(r'Fault reports (a$^{-1}$)')
        ax.text(0.98, 0.97, rf'$\rho_s={r:.2f}$', transform=ax.transAxes, ha='right', va='top')
        tag(ax, 'b')
    save(fig, 'Fig06_validation_complaints')

    # TABLE 6: correlations
    rows = []
    if en is not None:
        r, p = spearmanr(en['ref_return_C'], en['mean_dT_K'])
        rows.append([f"{SYM['Tref']} vs flow-weighted {SYM['dTp']}", 'All', len(en), fmt(r), fmt_p(p)])
    if c4 is not None:
        for m, d in c4.groupby('manufacturer'):
            r, p = spearmanr(d['ref_return_C'], d['reports_per_year'])
            rows.append([f"{SYM['Tref']} vs fault reports", m, len(d), fmt(r), fmt_p(p)])
        r, p = spearmanr(c4['ref_return_rank'], c4['report_rank'])
        rows.append([f"{SYM['Tref']} vs fault reports", 'All (within-mfr. ranks)', len(c4), fmt(r), fmt_p(p)])
    table(pd.DataFrame(rows, columns=['Pair', 'Scope', '$n$', r'$\rho_s$', '$p$']),
          'T6_correlations', 'Spearman rank correlations at substation level.', 'tab:correlations',
          align='llrrr')

# ===================================================== FIGURE 7: concentration
if en is not None:
    col = 'excess_volume_m3_a_config_median'
    p = en[en[col] > 0].sort_values(col, ascending=False).reset_index(drop=True)
    tot = p[col].sum()
    xs = (np.arange(len(p)) + 1) / len(en) * 100
    ys = p[col].cumsum().values / tot * 100
    xc, yc = np.concatenate([[0], xs]), np.concatenate([[0], ys])
    s20 = float(np.interp(20, xc, yc))
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    ax.plot([0, 100], [0, 100], color=GREY, lw=0.6, ls=(0, (3, 2)))
    ax.plot(xc, yc, color=INK, lw=0.9, zorder=2)
    rep = p['reports']
    m0, m1, mu = rep.fillna(-1) == 0, rep.fillna(-1) > 0, rep.isna()
    ax.scatter(xs[m0], ys[m0], s=12, marker='o', facecolor='white', edgecolor=BLUE, linewidth=0.8, zorder=3,
               label='No report')
    ax.scatter(xs[m1], ys[m1], s=12, marker='s', color=ORANGE, edgecolor='white', linewidth=0.3, zorder=3,
               label=r'$\geq$1 report')
    if mu.any():
        ax.scatter(xs[mu], ys[mu], s=6, marker='o', color=GREY, zorder=3, label='Short record')
    ax.plot([20, 20], [0, s20], color=GREY, lw=0.5); ax.plot([0, 20], [s20, s20], color=GREY, lw=0.5)
    ax.text(21, s20 - 3, f'{s20:.0f}%', va='top', fontsize=7)
    ax.set_xlim(0, 100); ax.set_ylim(0, 101)
    ax.set_xlabel('Substations, ranked (%)'); ax.set_ylabel(f"Cumulative {SYM['Vex']} (%)")
    ax.legend(loc='lower right')
    save(fig, 'Fig07_concentration')

    # TABLE 7: largest contributors
    top = p.head(10)
    t7 = pd.DataFrame({'Substation': top['manufacturer'] + '-' + top['substation_id'].astype(str),
                       'Configuration': cfg(top['configuration']),
                       f"{SYM['Q']} (MWh a$^{{-1}}$)": top['heat_MWh_a'].map(lambda v: fmt(v, 0)),
                       f"{SYM['Tref']} ($^{{\\circ}}$C)": top['ref_return_C'].map(lambda v: fmt(v, 1)),
                       'Gap (K)': top['gap_K_config_median'].map(lambda v: fmt(v, 1)),
                       f"{SYM['dTp']} (K)": top['mean_dT_K'].map(lambda v: fmt(v, 1)),
                       f"{SYM['Vex']} (m$^3$ a$^{{-1}}$)": top[col].map(lambda v: fmt(v, 0)),
                       f"{SYM['Vex']} (%)": top['excess_volume_pct_config_median'].map(lambda v: fmt(v, 1)),
                       'Share (%)': (100 * top[col] / tot).map(lambda v: fmt(v, 1)),
                       'Reports': top['reports'].map(lambda v: '--' if pd.isna(v) else str(int(v)))})
    table(t7, 'T7_top_contributors', 'Ten largest contributors to excess circulated volume '
          '(benchmark: configuration median).', 'tab:top', align='ll' + 'r' * 8)

# ====================================================== TABLE 8: energy summary
if ensum is not None:
    BN = {'config_median': 'Configuration median', 'fleet_median': 'Fleet median', 'best_quartile': 'Best quartile'}
    pump_cols = sorted([c for c in ensum.columns if c.startswith('pump_MWh_a_e')])
    rows = []
    for _, r in ensum.iterrows():
        pumps = [r[c] for c in pump_cols]
        rows.append([BN.get(r['benchmark'], r['benchmark']), f"{int(r['with_deficit'])}/{int(r['substations'])}",
                     fmt(r['median_gap_of_deficit_K'], 1), fmt(r['fleet_excess_volume_m3_a'] / 1000, 1),
                     fmt(r['fleet_excess_volume_pct'], 1),
                     f"{min(pumps):.0f}--{max(pumps):.0f}" if pumps else '--'])
    table(pd.DataFrame(rows, columns=['Benchmark', 'Deficit', 'Median gap (K)',
                                      f"{SYM['Vex']} ($10^3$ m$^3$ a$^{{-1}}$)", f"{SYM['Vex']} (%)",
                                      r'Pumping (MWh$_{\mathrm{el}}$ a$^{-1}$)']),
          'T8_energy',
          (f"Fleet excess circulated volume for {int(ensum['substations'].iloc[0])} substations "
           f"({ensum['fleet_heat_MWh_a'].iloc[0] / 1000:.1f} GWh a$^{{-1}}$ heat, "
           f"{ensum['fleet_volume_m3_a'].iloc[0] / 1000:.0f}$\\times10^3$ m$^3$ a$^{{-1}}$ circulated)."),
          'tab:energy', align='lrrrrr',
          note='Pumping range uses the specific pumping energies set in stage 9d (to be replaced by cited values).')

# ===================================================== TABLE 9: robustness checks
rows = []
if choice is not None:
    for _, r in choice.iterrows():
        rows.append(['Design (extended $-$ base)', r['train'], fmt(r['median_diff_ext_minus_base'], 3),
                     fmt(100 * r['extended_better_share'], 0), fmt_p(r['wilcoxon_p'])])
if rho_s is not None:
    for _, r in rho_s.iterrows():
        rows.append([rf"$\rho_{{\max}}={r['rho_cap']:.1f}$ (hier. $-$ own)", r['train'],
                     fmt(r['median_pooled_minus_own_K'], 3), fmt(100 * r['pooled_better_share'], 0),
                     fmt_p(r['wilcoxon_p'])])
if rows:
    table(pd.DataFrame(rows, columns=['Comparison', 'History', r'$\Delta$RMSE (K)', 'Better (%)', '$p$']),
          'T9_robustness', 'Robustness checks (paired, Wilcoxon signed-rank).', 'tab:robustness',
          align='llrrr')

# ======================================== SUPPLEMENTARY S1: physics vs raw (screening)
if trans is not None:
    sub = trans[(trans.subset == 'excl. commissioning faults') & (trans.scope == 'pooled')]
    COND = ['own_full', 'own_90', 'own_30', 'own_14', 'own_7', 'fleet_same', 'fleet_cross']
    CL = ['Full', '90', '30', '14', '7', 'Same', 'Other']
    FS = {'PHYS': dict(marker='o', color=BLUE, label='Physics'),
          'RAW_T': dict(marker='s', color=ORANGE, label='Raw temperatures'),
          'RAW_C': dict(marker='^', color=AQUA, label='Raw + flow, power')}
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    x = np.arange(len(COND))
    for k, s in enumerate(['PHYS', 'RAW_T', 'RAW_C']):
        if s == 'PHYS':
            q = sub[sub.phys_vs == 'RAW_T'].set_index('condition').reindex(COND)
            y, lo_, hi_ = q['AUC_a'], q['a_lo'], q['a_hi']
        else:
            q = sub[sub.phys_vs == s].set_index('condition').reindex(COND)
            y, lo_, hi_ = q['AUC_b'], q['b_lo'], q['b_hi']
        ax.errorbar(x + (k - 1) * 0.17, y, yerr=[y - lo_, hi_ - y], fmt=FS[s]['marker'], ms=3.5, lw=0.8,
                    capsize=0, color=FS[s]['color'], mec='white', mew=0.4, label=FS[s]['label'])
    ax.axhline(0.5, color=GREY, lw=0.6, ls=(0, (3, 2)))
    ax.axvline(4.5, color=GREY, lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(CL)
    ax.set_xlabel('Own history (d)  |  fleet reference'); ax.set_ylabel('AUROC')
    ax.set_ylim(0.3, 1.0)
    ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0), columnspacing=0.6)
    save(fig, 'FigS1_screening_transfer')

if fair is not None:
    f2 = fair.copy()
    f2['Physics'] = f2['AUROC_phys'].map(fmt); f2['Raw'] = f2['AUROC_raw'].map(fmt)
    f2[r'$\Delta$ [95\% CI]'] = [f"{d:.3f} [{a:.3f}, {b:.3f}]" for d, a, b in zip(f2['diff'], f2['CI_lo'], f2['CI_hi'])]
    f2['$p$'] = f2['p_boot'].map(fmt_p)
    table(f2[['subset', 'comparison', 'n', 'Physics', 'Raw', r'$\Delta$ [95\% CI]', '$p$']]
          .rename(columns={'subset': 'Subset', 'comparison': 'Feature sets', 'n': '$n$'}),
          'TS1_screening_fair', 'Fault-vs-normal screening AUROC, physics vs raw features (paired bootstrap).',
          'tab:screening', align='llrrrcr')

print('\nDone. Figures ->', FIG, '\n      Tables  ->', TAB)
