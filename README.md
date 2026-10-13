# Hierarchical physics-structured modelling of district heating substation return temperatures

Code for the paper

> **Pugazhendi P.** Hierarchical physics-structured modelling of district heating substation return temperatures: data efficiency, fault signatures and the comfort–efficiency gap. *Proceedings of the Institution of Mechanical Engineers, Part A: Journal of Power and Energy* (submitted, 2026).

The pipeline fits a physics-structured model of hourly primary return temperature to every substation of the public **PreDist** dataset (93 substations, 2 controller manufacturers, 239 substation-years). It pools the coefficients across configuration types by empirical Bayes and uses the model to

1. quantify how little own data a substation needs (data efficiency against own-only, fleet-only and random-intercept estimation),
2. characterise the load- and weather-matched return-temperature signature of customer-reported faults,
3. derive a condition-normalised *reference return temperature* and relate it to complaint frequency, and
4. translate cooling deficits into excess circulated water volume.

## Repository structure

```
.
├── run_all.py                  # runs the complete pipeline in one session
├── config.py                   # data and output paths (overridable by environment variables)
├── requirements.txt
├── scripts/
│   ├── s00_data_and_screening.py     # stages 0-7: data loading, inventory, physics features, screening
│   ├── s06c_fair_screening_test.py   # paired physics-vs-raw screening comparison
│   ├── s08_transfer.py               # limited-history and transfer screening
│   ├── s09_hierarchical_model.py     # hierarchical empirical-Bayes return-temperature model
│   ├── s09c_robustness_checks.py     # model choice, residual profile, fault signatures, complaints
│   ├── s09d_energy_translation.py    # reference return temperature, excess circulated volume
│   ├── s09e_data_efficiency.py       # data efficiency with the selected (extended) model
│   ├── s10_revision_analyses.py      # baselines, prior sensitivity, diagnostics, complaint models
│   ├── s12_model_comparison.py       # input ablation and gradient-boosting benchmark
│   └── s11_paper_figures_tables.py   # publication figures (PDF/PNG) and tables (LaTeX/CSV)
├── notebooks/
│   └── run_in_colab.ipynb      # one-click run on Google Colab
├── data/README.md              # how to obtain the dataset
└── outputs/                    # created by the pipeline (not tracked)
```

The stages share their in-memory state, like consecutive notebook cells; `run_all.py` therefore executes them one after another in a common namespace. Each script can also be pasted into a notebook cell in the order listed above.

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/pathmanaban86/predist-hierarchical-return-temperature.git
cd predist-hierarchical-return-temperature
pip install -r requirements.txt
```

## Data

Download PreDist version 2 from Zenodo (https://doi.org/10.5281/zenodo.17522254, CC BY 4.0) and save it as `data/predist_dataset.zip`; see [data/README.md](data/README.md). The dataset is not redistributed here.

## Running

```bash
python run_all.py                 # full pipeline
python run_all.py --figures-only  # re-draw figures and tables from existing outputs
python run_all.py --model-comparison-only   # only the stages needed for the model comparison (s12)
```

Paths can be changed without editing code:

| Variable | Default | Content |
|---|---|---|
| `PREDIST_ZIP` | `data/predist_dataset.zip` | dataset archive |
| `PREDIST_EXTRACT` | `work/predist` | unpacked data |
| `PREDIST_CACHE` | `work/cache` | hourly operational-data cache |
| `PREDIST_OUT` | `outputs` | result CSVs, figures and tables |

On the full dataset the pipeline takes hours rather than minutes; most of the time is spent in the robustness and sensitivity stages (9c, 10). The first run also builds the hourly cache.

**Google Colab:** open `notebooks/run_in_colab.ipynb`, set the Drive paths in the first cell and run all cells.

**Meter units.** Stage 9d checks the energy balance $\dot Q \approx \dot V \rho c_p \Delta T$ for each manufacturer and stops if the flow or heat-rate units are inconsistent. The units are read from the dataset's feature descriptions; if the check fails, set `UNIT_OVERRIDE` at the top of `s09d_energy_translation.py` as indicated in the error message.

## Outputs and the paper

All result files are written to `outputs/`; figures and tables for the paper to `outputs/paper_figures/` and `outputs/paper_tables/`.

| Paper item | Source |
|---|---|
| Figure 1 | `paper_figures/Fig01_dataset` |
| Figure 2 | `paper_figures/Fig02_coefficients` |
| Figure 3 (residual by load decile) | `paper_figures/Fig04_residual_load` |
| Figure 4 (data efficiency) | `paper_figures/Fig03_data_efficiency` |
| Figures 5–7 | `paper_figures/Fig05_fault_excess`, `Fig06_validation_complaints`, `Fig07_concentration` |
| Table 2 | `paper_tables/T1_dataset` |
| Table 3 | `paper_tables/T2_coefficients` |
| Table 4 | `paper_tables/T9_robustness`, `hm10_R5_prior_sensitivity_7d.csv`, `hm10_R7_manufacturer_priors.csv` |
| Table 5 | `hm10_R4_data_efficiency_extended.csv` |
| Table 6 | `hm12_model_comparison_summary.csv`, `paper_tables/T10_model_comparison` |
| Table 7 | `paper_tables/T4_fault_excess`, `hm10_R8_fault_signature_stats.csv` |
| Tables 8–9 | `paper_tables/T8_energy`, `paper_tables/T7_top_contributors` |
| Complaint statistics | `paper_tables/T6_correlations`, `hm10_R9_*.csv` |
| Excess-volume statistics | `hm9d_energy_per_substation.csv`, `hm10_R10_*.csv` |
| Figure S1, Table S1 | `paper_figures/FigS1_screening_transfer`, `paper_tables/TS1_screening_fair` |
| Figure S2 | `figures/FigS2_residual_diagnostics` |
| Table S2 | `paper_tables/T5_fault_excess_by_label`, `hm10_R8_labels_multiple_testing.csv` |
| Tables S3–S6 | `hm10_R2_*`, `hm10_R1_*`, `hm10_R3_*` + `hm10_R6_*`, `hm10_R5_*` |

Random seeds are fixed in every stage, so repeated runs give identical results for the same data and package versions.

## Citation

If you use this code, please cite the paper above and the dataset:

> Roelofs CMA, Guevara Bastidas E, Hugo T, Faulstich S, Cadenbach A. PreDist dataset – Operational data of district heating substations labelled with faults and maintenance information, version 2. Zenodo, 2025. https://doi.org/10.5281/zenodo.17522254

Citation metadata for this repository is in [CITATION.cff](CITATION.cff).

## Licence

Code: MIT licence (see [LICENSE](LICENSE)). The PreDist dataset is licensed separately under CC BY 4.0 by its authors.

## Contact

Pathmanaban Pugazhendi, Department of Automobile Engineering, Easwari Engineering College (SRM Group), Chennai, India — princepathu@gmail.com
