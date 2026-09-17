import os 
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import optuna

def run_ml_pipeline(data_path):
    print("---تحميل البيانات وتقسيمها----")
    df = pd.read_csv(data_path)
    
    x = df.drop(columns=['Target'])
    y = df['Target']
    
     # تقسيم البيانات إلى تدريب واختبار مع الحفاظ على التوازن (Stratify)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    print(f"حجم بيانات التدريب: {x_train.shape}, حجم بيانات الاختبار: {x_test.shape}\n")
    
    # --- 2. مقارنة سريعة بين النماذج الأساسية ---
    print("--- تقييم اداء الخوارزميات الاساسية عبر استخدام Cross Validation ---")
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
    }
    for name, model in models.items():
        scores = cross_val_score(model, x_train, y_train, cv=3, scoring='f1')
        print(f"معدل F1-Score لنموذج {name}: {scores.mean():.4f}")
        
    # --- 3. ضبط المعاملات الفائقة (Hyperparameter Tuning) لـ XGBoost باستخدام Optuna ---
    print("\n---ضبط المعاملات الفائقة ل XGBoost باستخدام Optuna---") 
    
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'eval_metric': 'logloss',
            'random_state': 42
        }
        model = XGBClassifier(**param)
        score = cross_val_score(model, x_train, y_train, cv=3, scoring='f1').mean()
        return score  
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=15)
    
    print(f"أفضل المعاملات الفائقة: {study.best_params}")
    
    # --- 4. تدريب النموذج الأفضل والتقييم النهائي ---
    print("\n--- التقييم النهائي على بيانات الاختبار (Test Set) ---")
    best_model = XGBClassifier(**study.best_params)
    best_model.fit(x_train, y_train)
    
    y_pred = best_model.predict(x_test)
    y_proba = best_model.predict_proba(x_test)[:, 1]
    
    print("\n[تقرير التصنيف Classification Report]:")
    print(classification_report(y_test, y_pred))
    
    auc_score = roc_auc_score(y_test, y_proba)
    print(f" مقياس ROC AUC Score: {auc_score:.4f}")
    
    # --- 5. حفظ النموذج النهائي واستخراج رسم مصفوفة الارتباك ---
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(best_model, 'artifacts/best_xgboost_model.joblib')
    print("\nتم حفظ النموذج في artifacts/best_xgboost_model.joblib بنجاح!")
    # رسم وحفظ مصفوفة الارتباك Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.xlabel(' التوقع(Predicted)')
    plt.ylabel('الحقيقة(Actual)')
    plt.title('مصفوفة الارتباك Confusion Matrix')
    plt.tight_layout()
    plt.savefig('artifacts/confusion_matrix.png')
    print("تم حفظ مصفوفة الارتباك في artifacts/confusion_matrix.png بنجاح!")
    
if __name__ == "__main__":
    data_path = '../Day01_Data_Preprocessing/processed_data.csv'  # ضع هنا مسار ملف البيانات الخاص بك
    run_ml_pipeline(data_path)