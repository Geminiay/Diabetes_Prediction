import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.preprocessing import clean_and_impute
from src.feature_engineering import create_features
from src.train_model import train_final_model, report_cv_stability
from src.explainability import generate_shap_report

os.makedirs("models", exist_ok=True)
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, "data", "diabetes.csv")

data = pd.read_csv(data_path)
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_imp, X_test_imp, imputer = clean_and_impute(X_train, X_test)
X_train_final = create_features(X_train_imp)
X_test_final  = create_features(X_test_imp)

print("[START] Training with BEST_PARAMS + SMOTE...")
model = train_final_model(X_train_final, y_train)

THRESHOLD = 0.35
y_probs  = model.predict_proba(X_test_final)[:, 1]
y_pred   = (y_probs >= THRESHOLD).astype(int)

print(f"\n--- FINAL MODEL RESULTS (Threshold: {THRESHOLD}) ---")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")
print(f"Missed Diabetic Cases (False Negatives): {cm[1][0]}")

report_cv_stability(model, X_train_final, y_train)
generate_shap_report(model, X_test_final)

joblib.dump(model, 'models/final_model.pkl')
joblib.dump(imputer, 'models/knn_imputer.pkl')
print("\n[SUCCESS] Pipeline complete. Model and Imputer saved.")