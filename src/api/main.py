from fastapi import FastAPI
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from src.api.schema import LoanRequest
from src.features.engineering import create_features
from src.decision.engine import decide
from src.decision.scoring import probability_to_score

from src.models.explainer import (
    get_shap_values,
    get_top_features,
    format_explanations
)

from src.config.mlflow_config import init_mlflow

# ---------------------------
# INIT
# ---------------------------
init_mlflow()

app = FastAPI(title="Credit Risk API")

MODEL_NAME = "credit_risk_model"

client = MlflowClient()

# Load latest model ONCE at startup
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = max(versions, key=lambda v: int(v.version))

model_uri = f"models:/{MODEL_NAME}/{latest_version.version}"
import mlflow.xgboost

model = mlflow.xgboost.load_model(model_uri)
# Get threshold ONCE at startup
run = client.get_run(latest_version.run_id)
threshold = float(run.data.params["best_threshold"])


# ---------------------------
# FEATURE COLUMNS
# ---------------------------
feature_cols = [
    "annual_inc",
    "loan_amnt",
    "dti",
    "loan_to_income",
    "dti_ratio",
    "log_income",
    "log_loan",
    "income_bucket",
    "loan_bucket",
    "income_x_dti",
    "loan_x_dti",
    "high_dti_flag",
    "low_income_flag",
    "high_loan_flag"
]


# ---------------------------
# ENDPOINT
# ---------------------------
@app.post("/score")
def score(request: LoanRequest):

    # Convert input to DataFrame
    df = pd.DataFrame([request.dict()])

    # Feature engineering
    df = create_features(df)

    X = df[feature_cols]

    # ---------------------------
    # PREDICTION
    # ---------------------------
    prob = model.predict_proba(X)[:, 1][0]
    print("MODEL OUTPUT:", model.predict(X)[0])
    # ---------------------------
    # SHAP EXPLANATIONS
    # ---------------------------
    shap_vals, explainer = get_shap_values(X)
    top_feats = get_top_features(X, shap_vals)

    explanations = format_explanations(
        X,
        shap_vals,
        top_feats,
        explainer
    )[0]

    # ---------------------------
    # DECISION LOGIC
    # ---------------------------
    decision = decide(
        prob,
        income=request.annual_inc,
        loan_amnt=request.loan_amnt,
        dti=request.dti,
        threshold=threshold
    )

    # ---------------------------
    # FINAL SCORE
    # ---------------------------
    score = probability_to_score(prob)

    return {
        "borrower_id": request.borrower_id,
        "probability_of_default": float(prob),
        "credit_score": score,
        "decision": decision,
        "explanations": explanations
    }