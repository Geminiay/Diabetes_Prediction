from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
import numpy as np

BEST_PARAMS = {
    'n_estimators': 438,
    'max_depth': 13,
    'min_samples_split': 4,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'random_state': 42
}

def train_final_model(X_train, y_train):
    X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
    model = RandomForestClassifier(**BEST_PARAMS)
    model.fit(X_res, y_res)
    return model

def report_cv_stability(model, X, y):
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
    print("\n--- Cross-Validation Stability Report ---")
    print(f"Scores per Fold: {np.round(cv_scores, 3)}")
    print(f"Mean F1: {cv_scores.mean():.3f}")
    print(f"Standard Deviation: {cv_scores.std():.3f}")