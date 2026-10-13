"""Paths used by all pipeline stages.

Override any of them with environment variables, e.g.

    export PREDIST_ZIP=/data/predist_dataset.zip
    export PREDIST_OUT=/results/predist_analysis

Defaults assume the repository layout described in README.md.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ZIP_PATH = Path(os.environ.get('PREDIST_ZIP', ROOT / 'data' / 'predist_dataset.zip'))
EXTRACT_DIR = Path(os.environ.get('PREDIST_EXTRACT', ROOT / 'work' / 'predist'))      # unzipped data
PARQ_DIR = Path(os.environ.get('PREDIST_CACHE', ROOT / 'work' / 'cache'))             # hourly cache
OUT_DIR = Path(os.environ.get('PREDIST_OUT', ROOT / 'outputs'))                       # CSV, figures, tables

for _d in (EXTRACT_DIR.parent, PARQ_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
