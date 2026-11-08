# PaySim Fraud Risk Scoring

A Python-based application that estimates the probability of fraud in mobile-money transactions using a saved XGBoost classification model and an interactive Streamlit interface.

**Author:** Madhav Kumar

## Overview

Fraud detection is an imbalanced classification problem: fraudulent transactions make up a small proportion of the available records. This project explores transaction-level fraud indicators in the PaySim synthetic mobile-money dataset and provides an interface for scoring an individual transaction.

The repository contains three main parts:

- A Streamlit application for entering transaction details and viewing a predicted fraud probability.
- A Jupyter notebook for initial dataset exploration, feature preparation and creation of a stratified train/test split.
- A serialized XGBoost model artifact used by the application for inference.

The notebook currently does **not** contain the complete model-training and evaluation workflow. The saved model is available for inference, but the steps that produced it are not fully reproducible from the notebook as currently committed.

## Features

- **Transaction scoring:** Load the saved XGBoost model and obtain a fraud probability for an entered transaction.
- **Interactive Streamlit UI:** Enter transaction type, amount, and sender/receiver balances before and after the transaction.
- **Balance-based feature engineering:** Calculate sender and receiver balance-discrepancy features before prediction.
- **Adjustable decision threshold:** Select the probability threshold used to flag a transaction.
- **Dataset exploration:** Inspect the PaySim data, column types, descriptive statistics, missing values and fraud-class distribution in the notebook.
- **Class-imbalance awareness:** Explore the highly imbalanced target distribution rather than relying on accuracy alone.

## Technology Stack

| Area | Technologies |
|---|---|
| Language | Python |
| Data processing | Pandas |
| Machine learning | XGBoost, scikit-learn |
| Model loading | Joblib |
| User interface | Streamlit |
| Experiment environment | Jupyter Notebook |

## Repository Structure

```text
.
├── app.py
├── paysim-fraud.ipynb
├── requirements.txt
├── models/
│   └── xgb_fraud.joblib
└── images/
    ├── feature_importance.png
    ├── pr_curves.png
    └── shap_summary.png
```

The raw PaySim CSV is not included in the repository. The notebook expects the dataset at `Data/paysim-fraud.csv`. Create that directory and place the dataset there before running the notebook.

## How the Application Works

1. The user selects a transaction type: `TRANSFER` or `CASH_OUT`.
2. The user enters the transaction amount and sender/receiver balances before and after the transaction.
3. The application converts the transaction type into the numeric representation expected by the saved model.
4. It derives two additional features:

   ```python
   errorOrig = oldbalanceOrg - amount - newbalanceOrig
   errorDest = oldbalanceDest + amount - newbalanceDest
   ```

5. The application arranges the eight model input features in the expected order and calls `predict_proba()`.
6. The resulting probability is compared with the user-selected threshold. Transactions above the threshold are flagged as likely fraud; the remaining transactions are shown as not flagged.

A lower threshold generally flags more transactions and can increase false positives. A higher threshold generally produces fewer alerts but can miss more fraudulent transactions. The current default threshold is 0.50; it has not been selected through a documented business-cost optimization process in the checked-in notebook.

## Dataset and Preparation

The project uses the synthetic PaySim mobile-money transaction dataset. The raw data file is intentionally not included because of its size.

The notebook currently performs these preparation steps:

1. Loads the CSV and inspects its structure, descriptive statistics and missing-value counts.
2. Examines the distribution of the fraud target.
3. Filters the data to `TRANSFER` and `CASH_OUT` transaction types.
4. Engineers `errorOrig` and `errorDest` from the transaction balance fields.
5. Encodes the transaction type as a numeric feature.
6. Removes the target, identifier columns, the rule-based `isFlaggedFraud` field and the `step` column from the feature matrix.
7. Creates a stratified 80/20 train/test split.

**Reproducibility note:** The committed notebook currently ends after the train/test split. It does not currently show the complete model fitting, model comparison, threshold-selection or evaluation process. The visual files in `images/` are included as project artifacts, but their generation is not reproduced by the notebook as currently committed.

## Getting Started

### Prerequisites

- Python installed
- pip
- Git
- The PaySim CSV file if you intend to run the notebook

### 1. Clone the repository

Use the HTTPS clone address shown in the repository's **Code** menu:

```bash
git clone <repository-clone-address>
```

Move into the directory that contains `app.py`, `paysim-fraud.ipynb` and `requirements.txt`. In the downloaded repository layout, this is `paysim-fraud-main/paysim-fraud-main/`.

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

From the directory containing `requirements.txt`, run:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit application

Make sure `models/xgb_fraud.joblib` is present, then run:

```bash
streamlit run app.py
```

Streamlit prints a local address in the terminal. Open that address in your browser to use the application.

### 5. Run the notebook

Create a directory named `Data` alongside the notebook and place the CSV file at:

```text
Data/paysim-fraud.csv
```

Open `paysim-fraud.ipynb` in Jupyter and execute its cells in order. The notebook currently covers exploratory inspection, filtering, feature construction and train/test splitting; it does not reproduce the saved model's training process.

## Model Input Features

The Streamlit app supplies these eight features to the classifier:

| Feature | Description |
|---|---|
| `type` | Encoded transaction type (`TRANSFER` or `CASH_OUT`) |
| `amount` | Transaction amount |
| `oldbalanceOrg` | Sender balance before the transaction |
| `newbalanceOrig` | Sender balance after the transaction |
| `oldbalanceDest` | Receiver balance before the transaction |
| `newbalanceDest` | Receiver balance after the transaction |
| `errorOrig` | Sender balance discrepancy |
| `errorDest` | Receiver balance discrepancy |

The feature names, transformations and ordering must remain consistent with the model artifact. If the model is retrained, update the application and artifact together.

## Limitations

- PaySim is synthetic data. Results on this dataset do not establish real-world fraud-detection performance.
- The training process for the committed model is not fully documented or reproducible from the current notebook.
- The default decision threshold is 0.50 and has not been tuned against a documented validation objective.
- The application scores one manually entered transaction at a time; it is not a streaming or production transaction-monitoring system.
- The included SHAP and other plot images are static artifacts. The current Streamlit application does not generate per-transaction SHAP explanations.
- Model probabilities should be interpreted as outputs of this particular saved model, not as certainty that a transaction is fraudulent.

## Future Improvements

- Complete a reproducible training pipeline that saves the model artifact from documented code.
- Compare baseline and tree-based classifiers using precision, recall, F1-score and PR-AUC.
- Select a classification threshold using a validation set and an explicit false-positive/false-negative cost trade-off.
- Add automated tests for feature engineering, feature order, model loading and prediction behavior.
- Add per-transaction explanations and integrate them into the Streamlit interface.
- Record model version, training configuration and feature metadata alongside the model artifact.

## Responsible Use

This project is intended for educational and demonstration purposes. It uses a synthetic dataset and should not be used as the basis for real financial decisions or production fraud controls.

## Author

**Madhav Kumar**
