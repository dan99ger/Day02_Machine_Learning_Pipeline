# 📊 Day 02: Machine Learning Pipeline & Hyperparameter Tuning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green)
![Optuna](https://img.shields.io/badge/Optuna-Automated%20Tuning-blueviolet)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Evaluation-F7931E)

An automated machine learning modeling pipeline that compares classical algorithms, tunes hyperparameters using Bayesian search with Optuna, and evaluates performance with classification metrics and saved model artifacts.

---

## 🛠️ Key Features

* **Model Comparison:** Cross-validation across Logistic Regression, Random Forest, and XGBoost.
* **Bayesian Optimization:** Automated hyperparameter tuning using **Optuna** to maximize F1-Score.
* **Evaluation Artifacts:** Generates Confusion Matrix visual reports and computes ROC-AUC metrics.
* **Model Serialization:** Exports the trained model to `best_xgboost_model.joblib` for production serving.

---

## 📂 Repository Structure

```text
Day02_Machine_Learning_Pipeline/
├── artifacts/
│   ├── best_xgboost_model.joblib   # Serialized model
│   └── confusion_matrix.png        # Evaluation plot
├── ml_pipeline.py                  # Core ML training & tuning script
├── .gitignore
└── README.md

🚀 How to Run
1. Setup Environment
Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install pandas numpy scikit-learn xgboost optuna matplotlib seaborn joblib
2. Execute Training & Tuning
Bash
python ml_pipeline.py
Part of the 7-Day Machine Learning Engineering Challenge.
