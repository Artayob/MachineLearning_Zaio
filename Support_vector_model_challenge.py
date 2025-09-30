# =============================
# IMPORTS
# =============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    make_scorer
)

# =============================
# STEP 1: Load Data
# =============================
def load_cancer_data_svm():
    """
    Loads the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn.
    """
    data = load_breast_cancer(as_frame=True)
    return data

# =============================
# STEP 2: Separate Features/Target
# =============================
def separate_cancer_features_target_svm(cancer_bunch):
    """
    Separates the features (X) and target (y).
    """
    X = cancer_bunch.data
    y = cancer_bunch.target
    return X, y

# =============================
# STEP 3: Scale Features
# =============================
def scale_features(X):
    """
    Scales features using StandardScaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return scaler, X_scaled

# =============================
# STEP 5: Hyperparameter Tuning
# =============================
def tune_svm(X_scaled, y):
    """
    Performs hyperparameter tuning for SVC using GridSearchCV
    with Stratified K-Fold CV on scaled data.
    """
    svc = SVC(probability=True, random_state=42)

    param_grid = [
        {"kernel": ["linear"], "C": [0.1, 1, 10, 100]},
        {"kernel": ["rbf"], "C": [0.1, 1, 10, 100],
         "gamma": [1, 0.1, 0.01, 0.001, "scale", "auto"]}
    ]

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=svc,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring="accuracy",
        n_jobs=1
    )
    grid_search.fit(X_scaled, y)
    return grid_search

# =============================
# STEP 6: Get Best Model
# =============================
def get_best_svm_model(grid_search):
    """
    Extracts the best parameters, score, and model from fitted GridSearchCV.
    """
    return {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "best_model": grid_search.best_estimator_
    }

# =============================
# STEP 7: Report Performance
# =============================
def report_svm_cv_performance(best_svm_results):
    """
    Reports the best CV performance.
    """
    best_score = best_svm_results["best_score"]
    best_params = best_svm_results["best_params"]

    print("Best Mean Cross-validated Accuracy: {:.4f}".format(best_score))
    print("Best Parameters:", best_params)
    return best_score

# =============================
# RUN PIPELINE (example)
# =============================
if __name__ == "__main__":
    cancer_data = load_cancer_data_svm()
    X, y = separate_cancer_features_target_svm(cancer_data)
    scaler, X_scaled = scale_features(X)

    grid_search = tune_svm(X_scaled, y)
    best_results = get_best_svm_model(grid_search)
    report_svm_cv_performance(best_results)
