# Pipeline

The original Colab notebooks (`STROKE_PREDICTION.ipynb`, `TRAINING_MODEL.ipynb`)
have been rewritten as five documented, runnable `.py` stages. Each stage imports
the reusable `stroke_prediction` package, so the logic has a single source of truth
and the notebook flow becomes reproducible and testable.

Run a single stage:

```bash
python pipeline/stage_02_preprocessing.py
```

Run everything in order:

```bash
make pipeline
```

| Stage | Purpose | Output |
|------:|---------|--------|
| 01 | Exploratory analysis | figures in `reports/figures/` |
| 02 | Data cleaning | `data/processed/cleaned.csv` |
| 03 | Feature engineering | `data/processed/features.parquet` |
| 04 | Model training | `models/*.joblib` |
| 05 | Evaluation | `models/metrics.json`, confusion-matrix figures |
