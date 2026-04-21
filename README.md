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
│   ├── evaluation.py
│   └── explainability.py
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
