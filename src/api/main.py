from fastapi import FastAPI
import pandas as pd
import joblib
import json
from src.decision.scoring import probability_to_score

from src.api.schema import LoanRequest
from src.features.engineering import create_features
from src.decision.engine import decide
from src.models.explainer import (
    get_shap_values,
    get_top_features,
    format_explanations
)

app = FastAPI(title="Credit Risk API")

# Load model
model = joblib.load("model.pkl")


@app.post("/score")
def score(request: LoanRequest):

    df = pd.DataFrame([request.dict()])

    df = create_features(df)
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

    X = df[feature_cols]

    prob = model.predict_proba(X)[0][1]


    shap_vals = get_shap_values(X)
    top_feats = get_top_features(X, shap_vals)
    explanations = format_explanations(top_feats)[0]

    with open("threshold.json") as f:
        threshold = json.load(f)["threshold"]
 
    decision = decide(
    prob,
    income=request.annual_inc,
    loan_amnt=request.loan_amnt,
    dti=request.dti,
    threshold=threshold
)

    score = probability_to_score(prob)

    return {
    "probability_of_default": float(prob),
    "credit_score": score,
    "decision": decision,
    "explanations": explanations
}