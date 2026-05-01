import os
import mlflow
import subprocess
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier
import joblib


# =============================
# MLflow setup
# =============================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("credit_risk_model")


# =============================
# Git version tracking (Feast/data version proxy)
# =============================
def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode().strip()
    except:
        return "unknown"


git_commit = get_git_commit()


# =============================
# LOAD DATA
# =============================
base_df = pd.read_parquet(
    os.path.join(BASE_DIR, "data/data/processed/cleaned_data.parquet")
)


# =============================
# TRAIN / VAL / TEST SPLIT
# =============================
train_df, temp_df = train_test_split(
    base_df,
    test_size=0.3,
    random_state=42,
    stratify=base_df["target"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["target"]
)


# =============================
# FEATURES
# =============================
feature_cols = [
    "annual_inc", "loan_amnt", "dti", "loan_to_income",
    "dti_ratio", "log_income", "log_loan",
    "income_bucket", "loan_bucket",
    "income_x_dti", "loan_x_dti",
    "high_dti_flag", "low_income_flag", "high_loan_flag"
]


X_train, y_train = train_df[feature_cols], train_df["target"]
X_val, y_val = val_df[feature_cols], val_df["target"]
X_test, y_test = test_df[feature_cols], test_df["target"]


# =============================
# MLflow RUN
# =============================
with mlflow.start_run():

    mlflow.set_tag("project", "credit-risk-platform")
    mlflow.set_tag("model", "xgboost")
    mlflow.set_tag("author", "kaarvin")


    # -------------------------
    # VERSIONING + DATA INFO
    # -------------------------
    mlflow.log_param("git_commit", git_commit)
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("val_size", len(X_val))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("feature_count", len(feature_cols))

    mlflow.log_metric("train_pos_rate", y_train.mean())
    mlflow.log_metric("val_pos_rate", y_val.mean())
    mlflow.log_metric("test_pos_rate", y_test.mean())


    # =============================
    # MODEL
    # =============================
    model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)


    # =============================
    # PREDICTIONS
    # =============================
    train_preds = model.predict_proba(X_train)[:, 1]
    val_preds = model.predict_proba(X_val)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]


    # =============================
    # METRICS
    # =============================
    train_auc = roc_auc_score(y_train, train_preds)
    val_auc = roc_auc_score(y_val, val_preds)
    test_auc = roc_auc_score(y_test, test_preds)

    mlflow.log_metric("train_auc", train_auc)
    mlflow.log_metric("val_auc", val_auc)
    mlflow.log_metric("test_auc", test_auc)
    

    print("Train AUC:", train_auc)
    print("Val AUC:", val_auc)
    print("Test AUC:", test_auc)

    #Gini
    train_gini = 2 * train_auc - 1
    val_gini = 2 * val_auc - 1
    test_gini = 2 * test_auc - 1

    mlflow.log_metric("train_gini", train_gini)
    mlflow.log_metric("val_gini", val_gini)
    mlflow.log_metric("test_gini", test_gini)


    # =============================
    # THRESHOLD (VALIDATION-BASED)
    # =============================
    fpr, tpr, thresholds = roc_curve(y_val, val_preds)

    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    print("Best Threshold:", best_threshold)

    mlflow.log_param("best_threshold", float(best_threshold))
    mlflow.set_tag("decision_threshold", float(best_threshold))
    with open("threshold.json", "w") as f:
        json.dump({
            "best_threshold": float(best_threshold),
            "method": "youden_j",
            "source": "validation"
        }, f)

    mlflow.log_artifact("threshold.json")


    # =============================
    # SAVE MODEL
    # =============================
    # joblib.dump(model, "model.pkl")
    # mlflow.log_artifact("model.pkl")

    # Register model in MLflow Model Registry
    import mlflow.xgboost

    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path="model",
        registered_model_name="credit_risk_model"
    )
#     mlflow.sklearn.log_model(
#     sk_model=model,
#     name="model",
#     registered_model_name="credit_risk_model"
# )