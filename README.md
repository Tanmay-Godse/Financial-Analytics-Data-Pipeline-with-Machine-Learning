# Financial Analytics Data Pipeline with Scratch ML

An end-to-end academic project that combines data cleaning, SQL storage, feature engineering, machine learning from scratch, and interactive visualization on a financial fraud dataset.

This project uses the credit card fraud detection dataset to show the full workflow from raw data to model evaluation. 

## Overview

The goal of this project is to study how financial transaction data can be:
- collected and cleaned
- stored in a relational database
- transformed into model-ready features
- used to train machine learning algorithms implemented from scratch
- visualized through simple dashboards and plots

The project is built around fraud detection, which makes it a good example of a real financial analytics problem with class imbalance and evaluation challenges.

## Highlights

- End-to-end Python + SQL pipeline
- Uses the Kaggle credit card fraud dataset
- Stores raw, cleaned, and engineered data in SQLite
- Implements core ML algorithms from scratch
- Generates interactive HTML visualizations
- Includes tests and runnable scripts

## Dataset

- Dataset: Credit Card Fraud Detection
- Kaggle source: `mlg-ulb/creditcardfraud`
- Fallback source: OpenML mirror of the same dataset
- Rows: `284,807`
- Fraud cases: `492`
- Raw dataset size in this project: about `146 MB`

The pipeline first looks for a local `creditcard.csv`. If it is not available and Kaggle credentials are missing, it automatically uses the OpenML mirror so the project can still run.

## Project Workflow

### 1. Data pipeline

- load the raw dataset
- check for missing values and duplicates
- normalize important numeric fields
- engineer time based and amount based features
- save raw, cleaned, and feature ready outputs

### 2. SQL integration

The project stores data in SQLite tables for:
- raw transactions
- cleaned transactions
- feature store
- pipeline run history
- model run history

It also creates SQL summary views for:
- class distribution
- hourly fraud summary

### 3. Machine learning from scratch

Implemented in this project:

**Supervised**
- Linear Regression
- Logistic Regression
- Decision Tree Classifier
- Decision Tree Regressor
- Random Forest Classifier
- Random Forest Regressor
- Linear SVM
- K-Nearest Neighbors

**Ensemble**
- AdaBoost
- Gradient Boosting Regressor

**Unsupervised**
- PCA
- K-Means
- Hierarchical Clustering

### 4. Visualization

The pipeline generates interactive visual outputs in `outputs/`:
- amount distribution before and after preprocessing
- classifier comparison dashboard
- confusion matrix
- feature importance plot
- regression comparison plot

## Results Snapshot

From the latest run in this repository:

- Best classifier: `RandomForestClassifierScratch`
- Accuracy: `0.9994`
- Precision: `0.7778`
- Recall: `0.7467`
- F1-score: `0.7619`
- ROC-AUC: `0.9853`

Top feature signals from the best classifier included:
- `V14`
- `V17`
- `V10`
- `V9`
- `V7`

This is a heavily imbalanced dataset, so recall, precision, and F1-score are more informative than accuracy alone.

## Tech Stack

- Python
- Pandas
- NumPy
- Plotly
- Matplotlib
- Seaborn
- SQLite
- Pytest

Note:
Scikit-learn is used only for dataset access and helper utilities where needed. The main learning algorithms in this project are implemented manually in the codebase.

## Project Structure

```text
.
├── README.md
├── environment.yml
├── pyproject.toml
├── data/
├── artifacts/
├── outputs/
├── scripts/
│   ├── run_pipeline.py
│   ├── train_models.py
│   └── refresh_dashboard.py
├── sql/
│   └── schema.sql
├── src/financial_analytics_pipeline/
│   ├── data_pipeline.py
│   ├── database.py
│   ├── experiments.py
│   ├── orchestrator.py
│   ├── visualization.py
│   └── models/
└── tests/
```

## Where the Scratch ML Code Lives

- `src/financial_analytics_pipeline/models/supervised.py`
- `src/financial_analytics_pipeline/models/ensemble.py`
- `src/financial_analytics_pipeline/models/unsupervised.py`
- `src/financial_analytics_pipeline/models/metrics.py`
- `src/financial_analytics_pipeline/models/utils.py`

The training logic that uses these models is in:
- `src/financial_analytics_pipeline/experiments.py`

## How to Run

Install the project in the `project_1` micromamba environment:

```bash
micromamba run -n project_1 python -m pip install -e .
```

Run only the data pipeline:

```bash
micromamba run -n project_1 python scripts/run_pipeline.py
```

Run the full project:

```bash
micromamba run -n project_1 python scripts/train_models.py
```

Refresh dashboard outputs one time:

```bash
micromamba run -n project_1 python scripts/refresh_dashboard.py --once
```

Run tests:

```bash
micromamba run -n project_1 pytest
```

## Generated Outputs

After running the project, useful outputs are saved here:

- `data/raw/creditcard_snapshot.csv`
- `data/processed/creditcard_cleaned.csv`
- `data/processed/creditcard_features.csv`
- `data/processed/financial_pipeline.sqlite`
- `artifacts/validation_report.json`
- `artifacts/model_results.json`
- `artifacts/cluster_assignments.csv`
- `outputs/dashboard.html`
- `outputs/amount_distribution.html`
- `outputs/confusion_matrix.html`
- `outputs/feature_importance.html`
- `outputs/regression_comparison.html`

## Limitations

- SQLite is used instead of MySQL for simplicity
- The dashboard is refresh based, not a deployed live web application
- From scratch models are slower than optimized library implementations.


## Future Improvements

- add MySQL support
- build a live dashboard using Streamlit or Dash
- add screenshots of outputs directly to this README
- improve training logs and progress messages
- include a short final report or presentation summary

