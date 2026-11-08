import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Healthcare ML | Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
)

MODEL_PATH = Path("models/diabetes_model.joblib")

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# ---------------------------------------------------------
# Custom styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.35rem;
        font-weight: 750;
        margin-bottom: 0.15rem;
    }
    .sub-header {
        color: #64748b;
        margin-bottom: 1.4rem;
    }
    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 0.6rem;
    }
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 14px;
        border-radius: 12px;
    }
    .result-note {
        padding: 14px 16px;
        border-radius: 10px;
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------
if not MODEL_PATH.exists():
    st.error(
        "The trained model was not found. From your project folder, "
        "run: python train_model.py"
    )
    st.stop()

try:
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    metrics = artifact.get("metrics", {})
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

# ---------------------------------------------------------
# Header and sidebar
# ---------------------------------------------------------
st.markdown(
    '<div class="main-header">🩺 Diabetes Risk Prediction</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Healthcare Machine Learning project | '
    'Random Forest classification demonstration</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("About this project")
    st.write(
        "This educational application demonstrates a machine-learning "
        "classification workflow using eight health-related input features."
    )
    st.warning(
        "Educational use only. This result is not a diagnosis or medical advice, "
        "and must not be used to make healthcare decisions."
    )
    st.divider()
    st.subheader("Model performance")
    if "accuracy" in metrics:
        st.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
    if "roc_auc" in metrics:
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
st.markdown('<div class="section-title">Enter the health measurements</div>',
            unsafe_allow_html=True)
st.caption("Enter values, then select Run prediction to view the model output.")

with st.form("prediction_form"):
    left, right = st.columns(2)

    with left:
        pregnancies = st.number_input(
            "Pregnancies", min_value=0, max_value=20, value=1, step=1
        )
        glucose = st.number_input(
            "Glucose", min_value=0.0, max_value=300.0, value=120.0, step=1.0
        )
        blood_pressure = st.number_input(
            "Blood pressure (mm Hg)", min_value=0.0, max_value=200.0,
            value=70.0, step=1.0
        )
        skin_thickness = st.number_input(
            "Skin thickness (mm)", min_value=0.0, max_value=100.0,
            value=20.0, step=1.0
        )

    with right:
        insulin = st.number_input(
            "Insulin (mu U/ml)", min_value=0.0, max_value=900.0,
            value=80.0, step=1.0
        )
        bmi = st.number_input(
            "BMI", min_value=0.0, max_value=90.0, value=25.0, step=0.1
        )
        pedigree = st.number_input(
            "Diabetes pedigree function", min_value=0.0, max_value=3.0,
            value=0.5, step=0.01
        )
        age = st.number_input(
            "Age (years)", min_value=1, max_value=120, value=30, step=1
        )

    submitted = st.form_submit_button(
        "Run prediction", type="primary", use_container_width=True
    )

# ---------------------------------------------------------
# Prediction result
# ---------------------------------------------------------
if submitted:
    input_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            pedigree,
            age,
        ]],
        columns=FEATURES,
    )

    try:
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        class_labels = list(model.classes_)
        positive_index = class_labels.index(1) if 1 in class_labels else -1
        probability = (
            float(probabilities[positive_index])
            if positive_index >= 0
            else float(probabilities[-1])
        )
    except Exception as error:
        st.error(f"Prediction could not be completed: {error}")
        st.stop()

    st.divider()
    st.markdown('<div class="section-title">Prediction result</div>',
                unsafe_allow_html=True)

    result_col, probability_col = st.columns([1, 1])

    with result_col:
        if prediction == 1:
            st.error("Model output: Class 1 (diabetes-positive label)")
        else:
            st.success("Model output: Class 0 (diabetes-negative label)")

    with probability_col:
        st.metric("Model probability for class 1", f"{probability:.1%}")
        st.progress(max(0.0, min(probability, 1.0)))

    st.info(
        "This is a prediction from a model trained on a historical dataset. "
        "It is not an individual's clinical risk estimate or a diagnosis. "
        "Consult a qualified healthcare professional for medical questions."
    )

    with st.expander("View submitted input values"):
        st.dataframe(input_data, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------
st.divider()
st.markdown('<div class="section-title">Model performance overview</div>',
            unsafe_allow_html=True)

if metrics:
    metric_cols = st.columns(4)
    metric_items = [
        ("Accuracy", "accuracy", ".1%"),
        ("ROC-AUC", "roc_auc", ".3f"),
    ]

    # Show only values actually saved in the model artifact.
    for index, (label, key, number_format) in enumerate(metric_items):
        if key in metrics:
            metric_cols[index].metric(label, format(metrics[key], number_format))

    report = metrics.get("classification_report", {})
    if isinstance(report, dict):
        class_rows = []
        for class_key, class_name in [("0", "Class 0"), ("1", "Class 1")]:
            class_metrics = report.get(class_key)
            if isinstance(class_metrics, dict):
                class_rows.append({
                    "Class": class_name,
                    "Precision": class_metrics.get("precision"),
                    "Recall": class_metrics.get("recall"),
                    "F1-score": class_metrics.get("f1-score"),
                    "Support": class_metrics.get("support"),
                })

        if class_rows:
            st.subheader("Classification report")
            report_df = pd.DataFrame(class_rows).set_index("Class")
            st.dataframe(
                report_df.style.format({
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "F1-score": "{:.3f}",
                    "Support": "{:.0f}",
                }),
                use_container_width=True,
            )

    confusion = metrics.get("confusion_matrix")
    if (
        isinstance(confusion, list)
        and len(confusion) == 2
        and all(isinstance(row, list) and len(row) == 2 for row in confusion)
    ):
        st.subheader("Confusion matrix")
        confusion_df = pd.DataFrame(
            confusion,
            index=["Actual class 0", "Actual class 1"],
            columns=["Predicted class 0", "Predicted class 1"],
        )
        st.dataframe(confusion_df, use_container_width=True)

        st.caption(
            "Rows represent actual classes; columns represent model predictions."
        )
else:
    st.info("Evaluation metrics were not found in the saved model artifact.")

# ---------------------------------------------------------
# Model details
# ---------------------------------------------------------
with st.expander("Model information and input features"):
    st.write("**Algorithm:** Random Forest Classifier")
    st.write("**Input features:**")
    st.write(", ".join(FEATURES))
    st.write(
        "**Dataset:** Pima Indians Diabetes dataset (as configured in the "
        "training script)."
    )
    st.caption(
        "The displayed metrics describe this model's test-set evaluation. "
        "Performance on other populations or data may differ."
    )

st.divider()
st.caption(
    "Healthcare ML | Student project | For educational and demonstration purposes only"
)
