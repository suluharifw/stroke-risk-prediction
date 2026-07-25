.PHONY: install test lint eda preprocess features train evaluate pipeline serve docker

install:        ## Install the package + dev extras (editable)
	pip install -e ".[dev,notebooks]"

test:           ## Run the test suite
	pytest

lint:           ## Lint with ruff
	ruff check src tests app pipeline

eda:            ## Stage 01 — exploratory analysis
	python pipeline/stage_01_exploratory_analysis.py

preprocess:     ## Stage 02 — cleaning
	python pipeline/stage_02_preprocessing.py

features:       ## Stage 03 — feature engineering
	python pipeline/stage_03_feature_engineering.py

train:          ## Stage 04 — model training
	python pipeline/stage_04_model_training.py

evaluate:       ## Stage 05 — evaluation
	python pipeline/stage_05_evaluation.py

pipeline: eda preprocess features train evaluate  ## Run all stages end to end

serve:          ## Run the Flask app on :5001
	python app/app.py

docker:         ## Build and run the container
	docker build -t stroke-prediction . && docker run -p 5001:5001 stroke-prediction
