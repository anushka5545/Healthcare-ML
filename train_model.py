from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = Path("data/diabetes.csv")
MODEL_PATH = Path("models/diabetes_model.joblib")
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET = "Outcome"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run 'python download_dataset.py' first."
        )

    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df[required].copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    # In this dataset, zero is not a realistic measurement for these fields;
    # treat it as missing so the imputer can handle it.
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df.loc[df[col] == 0, col] = float("nan")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ))
    ])
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        )
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": pipeline, "metrics": metrics, "features": FEATURES}, MODEL_PATH)
    Path("models/metrics.json").write_text(json.dumps(metrics, indent=2))

    print("Training complete.")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"Saved model: {MODEL_PATH}")

if __name__ == "__main__":
    main()
