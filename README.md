# 🩺 Healthcare Machine Learning – Diabetes Prediction System

A machine learning project that demonstrates diabetes classification using a Random Forest Classifier and an interactive Streamlit web application.

## 📌 Project Overview

The Diabetes Prediction System is an educational healthcare machine learning application. It accepts eight health-related input measurements and uses a trained machine learning model to predict a diabetes class label.

The project demonstrates the complete machine learning workflow, including data preparation, model training, evaluation and interactive prediction.

## 🎯 Objectives

- Build a diabetes classification model using Python.
- Apply the Random Forest algorithm.
- Evaluate model performance using classification metrics.
- Develop an interactive Streamlit dashboard.
- Demonstrate a practical machine learning workflow.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## 📊 Dataset

**Dataset:** Pima Indians Diabetes Dataset

The model uses eight input features:

1. Pregnancies
2. Glucose
3. Blood Pressure
4. Skin Thickness
5. Insulin
6. BMI
7. Diabetes Pedigree Function
8. Age

## 🤖 Machine Learning Algorithm

**Random Forest Classifier**

The model performs binary classification and returns a predicted class label and estimated probability for class 1.

## 📈 Model Performance

The current test results are:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 74.03% |
| ROC-AUC | 0.829 |
| Class 0 F1-score | 0.794 |
| Class 1 F1-score | 0.649 |
| Weighted F1-score | 0.743 |

### Confusion Matrix

| Actual / Predicted | Class 0 | Class 1 |
|---|---:|---:|
| Class 0 | 77 | 23 |
| Class 1 | 17 | 37 |

*These results describe the current test-set evaluation and are not a guarantee of performance on other data.*

## ✨ Project Features

- Interactive Streamlit dashboard
- Eight health measurement inputs
- Diabetes class prediction
- Estimated class-1 probability
- Model accuracy and ROC-AUC display
- Classification report
- Confusion matrix
- Model information section

## 📂 Project Structure

```text
Healthcare-ML/
│
├── app.py
├── train_model.py
├── download_dataset.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── diabetes.csv
│
└── models/
    ├── diabetes_model.joblib
    └── metrics.json
```

*The data and model filenames above describe the expected project files; check your repository for the exact files.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/anushka5545/Healthcare-ML.git
```

### 2. Open the project folder

```bash
cd Healthcare-ML
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

For Windows Git Bash:

```bash
source .venv/Scripts/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Download the dataset

```bash
python download_dataset.py
```

### 7. Train the model

```bash
python train_model.py
```

### 8. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open at:

```text
http://localhost:8501
```

## 👩‍💻 Developer

**Anushka Santosh Powar**

- GitHub: [anushka5545](https://github.com/anushka5545)
- LinkedIn: [Anushka Powar](https://www.linkedin.com/in/anushka-powar-26276733b)

## ⚠️ Disclaimer

This project is created for educational and demonstration purposes only. Its predictions are not medical diagnoses, personal clinical risk assessments or medical advice. Always consult a qualified healthcare professional for medical decisions.

---

⭐ If you find this project interesting, feel free to explore the repository!
