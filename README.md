# wesad-stress

Minimal and clean structure for a wrist-only WESAD project.

## Structure

```text
wesad-stress/
│
├── README.md
├── requirements.txt
├── .gitignore
├── config.yaml
│
├── data/
│   ├── 01_raw/         # original local data
│   ├── 02_intermediate/ # intermediate data
│   └── 03_processed/   # model-ready data
│
├── documents/
│   ├── journal-club-checklist.pdf
│   ├── project-brief.md
│   └── notes.txt
│
├── notebooks/
│   ├── 01-eda.ipynb
│   ├── 02-preprocessing.ipynb
│   ├── 03-features.ipynb
│   └── 04-results.ipynb
│
├── src/
│   ├── __init__.py
│   ├── loader.py
│   ├── preprocessing.py
│   ├── features.py
│   └── evaluation.py
│
├── scripts/
│   ├── build_dataset.py
│   └── train_evaluate.py
│
└── experiments/
    └── run_001/
        ├── config.yaml
        ├── metrics.json
        └── figures/
```

## Folder Meaning

- `data/01_raw/`: original WESAD files stored locally.
- `data/02_intermediate/`: temporary or partially processed data.
- `data/03_processed/`: final data ready for training and evaluation.
- `documents/`: project notes, checklist, and academic brief.
- `notebooks/`: notebooks for analysis, checks, and results.
- `src/`: reusable Python modules.
- `scripts/`: command-line entry points.
- `experiments/`: saved outputs from each run.

## Dataset

- WESAD: https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html

## Reproducible Workflow

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Place the raw WESAD subject folders under `data/01_raw/WESAD`, then build
transition-safe wrist-only features:

```bash
python scripts/build_dataset.py
```

By default, the builder discards any 60 second physiological window that spans
more than one protocol label. This reduces transition leakage caused by assigning
a window label from only the final time point. To reproduce the original,
less strict windowing, pass `--allow-mixed-label-windows`.

Train and evaluate both tasks with leave-one-subject-out validation on CUDA:

```bash
python scripts/train_evaluate.py
```

The final script intentionally runs a small compliant model comparison:
`Logistic Regression`, `Linear SVM`, `Torch MLP`, and `Torch MLP Balanced`.
The Torch models are forced onto CUDA. The script saves metrics, reports,
confusion matrices, per-subject tables, ROC/AUC outputs where probabilities are
available, and report-ready figures under `experiments/final_gpu`.

For a CUDA run on Windows, install PyTorch CUDA first:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Then run the final GPU experiment:

```bash
python scripts/train_evaluate.py
```

The script fails early if CUDA is not available for the Torch models.
For a faster smoke test, add for example `--torch-epochs 5`.

## Evaluation Notes

- Primary binary task: train a real stress vs. non-stress classifier.
- Multi-class task: baseline vs. stress vs. amusement, reported as secondary.
- Main metrics: Macro-F1 and balanced accuracy, with weighted metrics retained
  for comparison.
- ROC/AUC outputs are saved as CSV summaries and PNG figures.
- `metrics_binary_from_multiclass.csv` is diagnostic only; it collapses
  multi-class predictions and should not be reported as the primary binary
  result.
- The code keeps only the required final model comparison; old tuning branches,
  LSTM code, cuML branches, and cache-heavy explainability helpers were removed
  from `src/`.
