"""Run the whole pipeline in dependency order.

Usage: ``.venv/bin/python scripts/run_all.py``
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

STEPS = [
    ("build the cleaned dataset", ["-m", "src.build_dataset"]),
    ("run the data-quality audit", ["-m", "src.audit"]),
    ("generate exploratory figures", ["-m", "src.eda"]),
    ("run the hypothesis suite", ["-m", "src.analysis"]),
    ("train and evaluate the price model", ["-m", "src.modelling"]),
    ("write the model report", ["-m", "src.reporting"]),
    ("build and execute notebooks", ["scripts/make_notebooks.py", "--execute"]),
]


def main() -> None:
    for i, (label, args) in enumerate(STEPS, 1):
        print(f"\n[{i}/{len(STEPS)}] {label}", flush=True)
        subprocess.run([PY, *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
