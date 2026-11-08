# Healthcare Machine Learning – Diabetes Prediction

A beginner-friendly machine-learning project that trains a Random Forest classifier on the Pima Indians Diabetes dataset and presents predictions through a Streamlit dashboard.

> **Important:** This is an educational demonstration only. It is not a medical device, diagnosis, or clinical risk calculator. Do not use it to make healthcare decisions.

## Features

- Data loading and basic validation
- Missing-value handling with median imputation
- Random Forest classification
- Train/test split with stratification
- Accuracy, ROC-AUC, confusion matrix, and classification report
- Streamlit interface for entering sample measurements

## Project structure

```text
Healthcare-ML/
├── app.py
├── train_model.py
├── download_dataset.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── (diabetes.csv is downloaded locally)
└── models/
    └── (trained model is created locally)
```

## 1. Create and activate a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

## 2. Install packages

```bash
pip install -r requirements.txt
```

## 3. Download the dataset

```bash
python download_dataset.py
```

The downloader retrieves a public mirror of the Pima Indians Diabetes dataset into `data/diabetes.csv`.

## 4. Train the model

```bash
python train_model.py
```

This creates `models/diabetes_model.joblib` and `models/metrics.json`.

## 5. Run the Streamlit app

```bash
streamlit run app.py
```

Open the local address shown in the terminal (usually `http://localhost:8501`).

## Input features

Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, and Age.

## Dataset note

The Pima Indians Diabetes dataset is a historical, limited dataset. It does not represent all populations. Its outputs should not be interpreted as an individual's actual medical risk.

## Technologies

Python, Pandas, Scikit-learn, Joblib, Streamlit.
