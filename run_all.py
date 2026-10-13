"""Run the complete analysis of the paper in one Python session.

The stages share their in-memory state (data frames, fitted models), exactly as when
the scripts are run as consecutive notebook cells, so they are executed one after
another in a common namespace.

    python run_all.py                  # full pipeline (several hours on the full dataset)
    python run_all.py --skip-figures   # without the publication figures/tables
    python run_all.py --figures-only   # only re-draw figures/tables from existing outputs
    python run_all.py --model-comparison-only   # stages needed for the model comparison (s12) only
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / 'scripts'
STAGES = [
    ('s00_data_and_screening.py', 'Stages 0-7: data, inventory, fault summary, physics features, screening'),
    ('s06c_fair_screening_test.py', 'Stage 6c: paired physics-vs-raw screening comparison (Table S1)'),
    ('s08_transfer.py', 'Stage 8: limited-history and transfer screening (Figure S1)'),
    ('s09_hierarchical_model.py', 'Stage 9: hierarchical physics-structured model'),
    ('s09c_robustness_checks.py', 'Stage 9c: model choice, residual profile, fault signatures, complaints'),
    ('s09d_energy_translation.py', 'Stage 9d: reference return temperature and excess circulated volume'),
    ('s09e_data_efficiency.py', 'Stage 9e: data efficiency with the selected model'),
    ('s10_revision_analyses.py', 'Stage 10: baselines, sensitivity, diagnostics, complaint models'),
    ('s12_model_comparison.py', 'Stage 12: input ablation and gradient-boosting benchmark'),
]
FIGURES = ('s11_paper_figures_tables.py', 'Stage 11: publication figures and tables')


def run(script, label, ns):
    print(f'\n{"=" * 78}\n{label}\n{"=" * 78}', flush=True)
    t0 = time.time()
    path = SCRIPTS / script
    code = compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    exec(code, ns)
    print(f'-- {script} finished in {(time.time() - t0) / 60:.1f} min', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--skip-figures', action='store_true')
    ap.add_argument('--figures-only', action='store_true')
    ap.add_argument('--model-comparison-only', action='store_true',
                    help='run only s00, s09, s09c (without the slow sensitivity rerun) and s12')
    args = ap.parse_args()

    os.environ.setdefault('MPLBACKEND', 'Agg')       # no GUI needed; figures are saved to files
    sys.path.insert(0, str(ROOT))                     # makes `import config` work in every stage
    ns = {'__name__': '__main__', '__file__': str(ROOT / 'run_all.py')}

    if args.model_comparison_only:
        os.environ['PREDIST_RHO_SENS'] = '0'
        keep = ('s00_', 's09_', 's09c_', 's12_')
        stages = [s for s in STAGES if s[0].startswith(keep)]
    else:
        stages = [] if args.figures_only else list(STAGES)
        if not args.skip_figures:
            stages.append(FIGURES)
    for script, label in stages:
        run(script, label, ns)
    from config import OUT_DIR
    print(f'\nAll done. Results are in {OUT_DIR}')


if __name__ == '__main__':
    main()
