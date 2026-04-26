 # ANN-Classification-Churn

A machine learning project that predicts **customer churn** using an Artificial Neural Network (ANN), with a **Streamlit** interface for interactive inference.

## Project Overview

This repository contains:

- A full churn **classification** workflow (data preparation, ANN training, artifact persistence, and inference).
- A separate ANN **regression** experiment for predicting `EstimatedSalary`.
- Notebook-based experimentation plus an app entrypoint for real-time predictions.

## Repository Structure

```text
ANN-Classification-Churn/
├── app.py
├── data/
│   └── Churn_Modelling.csv
├── models/
│   ├── model.h5
│   ├── label_encoder_gender.pkl
│   ├── OneHotEncoder_Geography.pkl
│   ├── Scaler.pkl
│   ├── regression_model.h5
│   ├── Reg_gender.pkl
│   ├── Reg_geo.pkl
│   └── Reg_scaler.pkl
├── Notebook/
│   ├── ANN.ipynb
│   ├── hyperparametertuningann.ipynb
│   ├── prediction.ipynb
│   └── regression.ipynb
├── log/
├── reg/
└── requirements.txt
```

## Dataset

The dataset used is `data/Churn_Modelling.csv`.

Key columns include:

- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`
- `Exited` (classification target)

Rows in file: 10,000 records (+ header).

## Churn Classification Workflow

Implemented primarily in `Notebook/ANN.ipynb` and served via `app.py`.

### 1) Preprocessing

- Drop identifier columns: `RowNumber`, `CustomerId`, `Surname`.
- Encode `Gender` with `LabelEncoder`.
- One-hot encode `Geography`.
- Split into train/test sets.
- Scale features with `StandardScaler`.

### 2) ANN Model

- Dense(64, relu)
- Dense(32, relu)
- Dense(1, sigmoid)
- Loss: `binary_crossentropy`
- Metric: `accuracy`
- Uses `EarlyStopping` and `TensorBoard` callbacks.

### 3) Saved Artifacts

The workflow saves reusable artifacts for inference:

- `models/model.h5`
- `models/label_encoder_gender.pkl`
- `models/OneHotEncoder_Geography.pkl`
- `models/Scaler.pkl`

## Hyperparameter Tuning

`Notebook/hyperparametertuningann.ipynb` explores ANN architecture with `GridSearchCV` + `scikeras`.

Search space includes:

- `neurons`: `[16, 32, 64, 128]`
- `layers`: `[1, 2]`
- `epochs`: `[50, 100]`

Recorded best setting in notebook output:

- `{'epochs': 100, 'layers': 1, 'neurons': 64}`

## Inference Notebook

`Notebook/prediction.ipynb` demonstrates manual prediction using saved artifacts:

1. Load model + encoders + scaler.
2. Build a single-row input payload.
3. Apply encoding/scaling.
4. Run `model.predict(...)`.

## Streamlit App

`app.py` provides an interactive churn prediction interface.

### Inputs collected

- Geography
- Gender
- Age
- Balance
- Credit Score
- Estimated Salary
- Tenure
- Number of Products
- Has Credit Card (0/1)
- Is Active Member (0/1)

### Output

- Churn probability
- Decision rule:
  - `< 0.5` → not likely to churn
  - `>= 0.5` → likely to churn

## Regression Experiment

`Notebook/regression.ipynb` trains a separate ANN to predict `EstimatedSalary`.

Saved regression artifacts:

- `models/regression_model.h5`
- `models/Reg_gender.pkl`
- `models/Reg_geo.pkl`
- `models/Reg_scaler.pkl`

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Run the Streamlit App

From the project root:

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (commonly `http://localhost:8501`).

## Notes and Caveats

- The repo includes notebook outputs and training logs (`log/`, `reg/`) that can be large.
- Models are saved as `.h5` files (legacy Keras format). You may migrate to `.keras` format in future updates.
- Ensure runtime paths in `app.py` match artifact filenames under `models/`.

## Requirements

Current dependencies are listed in `requirements.txt`:

- pandas
- scikit-learn
- numpy
- streamlit
- tensorflow
- tensorboard
- keras
- matplotlib
- scikeras

## Future Improvements

- Convert notebooks into a reusable Python package (`src/` layout).
- Add reproducible training scripts and CLI entrypoints.
- Add model evaluation metrics report (ROC-AUC, confusion matrix, precision/recall).
- Add automated tests for preprocessing and inference schema checks.
- Expand README with sample screenshots from Streamlit UI.
