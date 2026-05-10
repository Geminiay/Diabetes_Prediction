import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def clean_and_impute(X_train, X_test):
 
    cols_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    
    X_train[cols_to_fix] = X_train[cols_to_fix].replace(0, np.nan)
    X_test[cols_to_fix] = X_test[cols_to_fix].replace(0, np.nan)

    imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    
    return X_train_imputed, X_test_imputed, imputer