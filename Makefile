.PHONY: doctor build-dataset train export drift quality validate android-build status test clean help kd-train kd-train-fast predict

PYTHON ?= python
CLI := $(PYTHON) tools/spam_cli.py

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

doctor: ## Run environment checks
	$(CLI) doctor

build-dataset: ## Build training dataset
	$(CLI) build-dataset

train: ## Train all models (no export)
	$(CLI) train

train-export: ## Train + export TFLite
	$(CLI) train --export-tflite --plots

train-optuna: ## Train with Optuna search
	$(CLI) train --export-tflite --optuna-trials 30 --plots

kd-train: ## Knowledge Distillation: CatBoost teacher → Keras MLP student → TFLite
	$(CLI) kd-train

kd-train-fast: ## KD without Optuna (stage1 grid only, ~9 runs)
	$(CLI) kd-train --optuna-trials 0

predict: ## Predict ALLOW/WARN/BLOCK for one or more numbers (NUMBER=+79991234567)
	$(CLI) predict $(NUMBER)

export: ## Export TFLite model
	$(CLI) export --plots

drift: ## Drift detection (set DRIFT_REF=path.csv)
	$(CLI) drift --reference $(or $(DRIFT_REF),datasets/ru/processed/ru_tflite_features.csv)

quality: ## Data quality check
	$(CLI) quality

validate: ## Validate feature schema + data
	$(CLI) validate --strict

android-build: ## Build Android APK
	$(CLI) android-build

status: ## Git + model status
	$(CLI) status

test: ## Run Python tests
	$(PYTHON) -m pytest tests/ -v --tb=short

clean: ## Remove generated reports and __pycache__
	rm -rf datasets/ru/reports/*.json datasets/ru/reports/*.html datasets/ru/reports/*.md datasets/ru/reports/*.png
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
