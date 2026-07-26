<h1 align="center">Stroke Risk Prediction</h1>

<p align="center">
Created by <b>Suluh Arif Wibowo</b>
</p>

<p align="center">
  <em>An end-to-end machine-learning project — from raw data to a deployed, interactive web app.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-1.3-F7931E.svg?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Flask-3.0-000000.svg?logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/tests-passing-1B9C7D.svg" alt="Tests">
  <img src="https://img.shields.io/badge/license-MIT-E4572E.svg" alt="License">
</p>

A logistic-regression model estimates a patient's stroke risk from routine health
indicators. The repository is organised the way a small production ML project would
be: a reusable package under `src/`, a documented, reproducible pipeline under
`pipeline/`, a thin Flask serving layer under `app/`, and tests.

> **Disclaimer.** This is an educational / portfolio project. It is **not** a medical
> device and must not be used for real clinical decisions.

---

## Highlights

- **Clean package** (`stroke_prediction`) with typed functions, docstrings, logging, and input validation.
- **Reproducible pipeline** — the original notebooks are rewritten as five documented, runnable stages.
- **Class-imbalance study** — BorderlineSMOTE vs. RandomUnderSampler vs. SMOTE-ENN (dataset is ~4.9% positive).
- **Interactive web app** — a modern single-page UI with a live risk gauge and real-time BMI/glucose category feedback.
- **Tested & containerised** — `pytest` suite and a `Dockerfile`.

---

## Project structure

```
stroke-prediction/
├── src/stroke_prediction/     # the reusable package (all ML logic)
│   ├── config.py              # paths, feature order, category rules, hyper-params
│   ├── data.py                # dataset loading
│   ├── preprocessing.py       # cleaning + encoding -> 23-feature matrix
│   ├── features.py            # WHO weight & glucose categories
│   ├── training.py            # scaler + resampling variants
│   ├── evaluation.py          # metrics on the held-out test set
│   ├── inference.py           # StrokePredictor used by the app
│   ├── schema.py              # PatientRecord validation
│   ├── io.py · logger.py      # artifact I/O and logging
│   └── __init__.py            # public API
├── pipeline/                  # notebooks rewritten as documented stages
│   ├── stage_01_exploratory_analysis.py
│   ├── stage_02_preprocessing.py
│   ├── stage_03_feature_engineering.py
│   ├── stage_04_model_training.py
│   └── stage_05_evaluation.py
├── app/                       # Flask serving layer
│   ├── app.py
│   ├── templates/index.html
│   └── static/ (css, js, img)
├── models/                    # scaler + trained models + metrics.json
├── data/{raw,processed}/      # dataset in, generated features out
├── reports/figures/           # EDA & confusion-matrix plots
├── tests/                     # pytest suite
└── docs/                      # design prototype + assets
```

---

## Quickstart

```bash
# 1. Install (editable, with dev + notebook extras)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,notebooks]"

# 2. (Optional) reproduce everything from raw data
make pipeline        # eda -> preprocess -> features -> train -> evaluate

# 3. Run the web app  ->  http://127.0.0.1:5001
make serve
```

Pre-trained artifacts ship in `models/`, so you can run the app without retraining.
Common tasks are wrapped in the `Makefile` (`make help`-style targets): `install`,
`test`, `lint`, `train`, `evaluate`, `serve`, `docker`.

### Docker

```bash
make docker          # build + run on :5001
```

---

## The pipeline

Each stage is an independently runnable script that documents its step and delegates
the heavy lifting to the package.

| Stage | Script | What it does |
|------:|--------|--------------|
| 01 | `stage_01_exploratory_analysis.py` | Structure, missing values, class balance, distribution figures |
| 02 | `stage_02_preprocessing.py` | Impute BMI, drop `id`/`Residence_type`, fix `Unknown` smoking, cap outliers |
| 03 | `stage_03_feature_engineering.py` | WHO weight & glucose categories, encoding → 23-feature matrix |
| 04 | `stage_04_model_training.py` | Scale, stratified 70/30 split, train 3 resampling variants |
| 05 | `stage_05_evaluation.py` | Metrics + confusion matrices on the held-out test set |

---

## Results

Evaluated on the held-out test set (stratified 30%). Because the data is heavily
imbalanced, **recall** and **ROC-AUC** are the metrics that matter — not accuracy.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|:--------:|:---------:|:------:|:--:|:-------:|:------:|
| **BorderlineSMOTE** *(served)* | 0.766 | 0.143 | 0.760 | 0.241 | **0.844** | 0.245 |
| RandomUnderSampler | 0.710 | 0.124 | **0.813** | 0.216 | 0.838 | 0.243 |
| SMOTE-ENN | 0.716 | 0.126 | 0.813 | 0.219 | 0.841 | 0.239 |

Low precision is expected for a rare, high-cost outcome: the model catches most true
strokes (high recall) at the cost of more false positives — a reasonable trade-off for
a screening tool. Full numbers, including confusion matrices, are in
[`models/metrics.json`](models/metrics.json).

---

## Using the package

```python
from stroke_prediction import StrokePredictor, PatientRecord

predictor = StrokePredictor()
result = predictor.predict(PatientRecord(
    gender="Male", age=67, hypertension=0, heart_disease=1,
    ever_married="Yes", work_type="Private",
    avg_glucose_level=228.0, bmi=36.6, smoking_status="formerly smoked",
))
print(result.as_dict())   # {'label': 1, 'probability': 0.93}
```

### Web API

`POST /api/predict` with a JSON body of raw values returns the label, probability,
risk band, and tailored advice:

```bash
curl -X POST http://127.0.0.1:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"age":67,"gender":"Male","ever_married":"Yes","work_type":"Private",
       "avg_glucose_level":228,"bmi":36.6,"hypertension":0,"heart_disease":1,
       "smoking_status":"formerly smoked"}'
```

---

## Dataset

Kaggle **Stroke Prediction Dataset** — 5,110 patient records, 11 features + target.
See [`data/README.md`](data/README.md) for the schema. The raw CSV lives in
`data/raw/`; the CSV is included for convenience but is not the project's own work.

---

## Testing & linting

```bash
make test     # pytest
make lint     # ruff
```

---

## License

Released under the [MIT License](LICENSE).

---

## Author

**Suluh Arif Wibowo**

- Machine Learning Engineer
- Palembang, Indonesia
- Email: suluharif.w@gmail.com
- LinkedIn: https://linkedin/suluharifw
- GitHub: https://github.com/suluharifw
