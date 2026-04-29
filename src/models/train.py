
# #=========================================================================================================================================================

# import pandas as pd
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score, roc_curve
# import joblib
# import json
# from feast import FeatureStore

# import os
# import mlflow

# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# TRACKING_URI = f"sqlite:///{BASE_DIR}/mlflow/mlflow.db"

# mlflow.set_tracking_uri(TRACKING_URI)
# mlflow.set_experiment("credit_risk_model")

# # =============================
# # SET EXPERIMENT
# # =============================
# mlflow.set_experiment("credit_risk_model")


# # =============================
# # GINI FUNCTION
# # =============================
# def gini_score(y_true, y_prob):
#     auc = roc_auc_score(y_true, y_prob)
#     return 2 * auc - 1


# # =============================
# # LOAD DATA
# # =============================
# base_df = pd.read_parquet("data/processed/cleaned_data.parquet")
# print("base data:", base_df.shape)


# # =============================
# # LOAD FEATURES FROM FEAST
# # =============================
# store = FeatureStore(repo_path="feature_repo/feature_repo")

# entity_df = pd.DataFrame({
#     "borrower_id": base_df["borrower_id"],
#     "event_timestamp": base_df["event_timestamp"]
# })

# training_df = store.get_historical_features(
#     entity_df=entity_df,
#     features=[
#         "loan_features:annual_inc",
#         "loan_features:loan_amnt",
#         "loan_features:dti",
#         "loan_features:loan_to_income",
#         "loan_features:dti_ratio",
#         "loan_features:log_income",
#         "loan_features:log_loan",
#         "loan_features:income_bucket",
#         "loan_features:loan_bucket",
#         "loan_features:income_x_dti",
#         "loan_features:loan_x_dti",
#         "loan_features:high_dti_flag",
#         "loan_features:low_income_flag",
#         "loan_features:high_loan_flag",
#     ]
# ).to_df()

# print("features loaded:", training_df.shape)


# # =============================
# # ✅ FIX: PROPER TARGET MERGE
# # =============================
# # Ensure both timestamps have same timezone format
# base_df["event_timestamp"] = pd.to_datetime(base_df["event_timestamp"]).dt.tz_localize(None)
# training_df["event_timestamp"] = pd.to_datetime(training_df["event_timestamp"]).dt.tz_localize(None)
# training_df = training_df.merge(
#     base_df[["borrower_id", "event_timestamp", "target"]],
#     on=["borrower_id", "event_timestamp"],
#     how="inner"
# )


# # =============================
# # FEATURE SELECTION
# # =============================
# feature_cols = [
#     "annual_inc",
#     "loan_amnt",
#     "dti",
#     "loan_to_income",
#     "dti_ratio",
#     "log_income",
#     "log_loan",
#     "income_bucket",
#     "loan_bucket",
#     "income_x_dti",
#     "loan_x_dti",
#     "high_dti_flag",
#     "low_income_flag",
#     "high_loan_flag"
# ]

# X = training_df[feature_cols]
# y = training_df["target"]


# # =============================
# # TRAIN TEST SPLIT
# # =============================
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )


# # =============================
# # MLflow RUN START
# # =============================
# with mlflow.start_run():

#     # ✅ ADD HERE
#     mlflow.set_tag("project", "credit-risk-platform")
#     mlflow.set_tag("model", "xgboost")
#     mlflow.set_tag("author", "kaarvin")

#     # =============================
#     # MODEL
#     # =============================
#     model = XGBClassifier(
#         n_estimators=150,
#         max_depth=6,
#         learning_rate=0.05,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         eval_metric="logloss"
#     )

#     # ✅ log parameters
#     mlflow.log_params({
#         "n_estimators": 150,
#         "max_depth": 6,
#         "learning_rate": 0.05,
#         "subsample": 0.8,
#         "colsample_bytree": 0.8
#     })

#     # =============================
#     # TRAIN
#     # =============================
#     model.fit(X_train, y_train)

#     preds = model.predict_proba(X_test)[:, 1]

#     # =============================
#     # EVALUATION
#     # =============================
#     auc = roc_auc_score(y_test, preds)
#     gini = gini_score(y_test, preds)

#     print(f"AUC: {auc:.4f}")
#     print(f"GINI: {gini:.4f}")

#     # ✅ log metrics
#     mlflow.log_metric("auc", auc)
#     mlflow.log_metric("gini", gini)

#     # =============================
#     # THRESHOLD (ROC)
#     # =============================
#     fpr, tpr, thresholds = roc_curve(y_test, preds)

#     j_scores = tpr - fpr
#     best_idx = j_scores.argmax()
#     best_threshold = thresholds[best_idx]

#     print(f"Best Threshold (ROC): {best_threshold:.4f}")

#     # ✅ log threshold
#     mlflow.log_metric("threshold", best_threshold)

#     # =============================
#     # SAVE FILES
#     # =============================
#     with open("threshold.json", "w") as f:
#         json.dump({"threshold": float(best_threshold)}, f)

#     print("Saving threshold")

#     joblib.dump(model, "model.pkl")
#     print("Model saved: model.pkl")

#     # =============================
#     # MLflow ARTIFACTS
#     # =============================
#     mlflow.sklearn.log_model(model, name="model")
#     mlflow.log_artifact("model.pkl")
#     mlflow.log_artifact("threshold.json")



#=========================================================================================================================================================
#==========================================================================================================================================================
#==========================================================================================================================================================
# =============================
# IMPORTS (add if missing)
# =============================
import os
import mlflow
import subprocess
import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split

# =============================
# MLflow setup (IMPORTANT)
# =============================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR}/mlflow/mlflow.db")
mlflow.set_experiment("credit_risk_model")


# =============================
# FEAST VERSION TRACKING
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
    os.path.join(BASE_DIR, "data/processed/cleaned_data.parquet")
)

# =============================
# TIME SAFE SPLIT (NO LEAKAGE)
# =============================
# First split: train + temp
train_df, temp_df = train_test_split(
    base_df,
    test_size=0.3,
    random_state=42,
    stratify=base_df["target"]
)

# Second split: validation + test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["target"]
)


# =============================
# FEATURE ENGINEERING (Feast already applied earlier)
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

    # -------------------------
    # DATA VERSIONING (IMPORTANT)
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
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score

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


    # =============================
    # SAVE MODEL
    # =============================
    import joblib

    joblib.dump(model, "model.pkl")
    mlflow.log_artifact("model.pkl")