import shap
import joblib
import pandas as pd
import numpy as np
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import mlflow.xgboost



#from ml flow
MODEL_NAME = "credit_risk_model"
client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
# get all versions
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
# pick latest version number
latest_version = max(versions, key=lambda v: int(v.version))
MODEL_URI = f"models:/{MODEL_NAME}/{latest_version.version}"
mlflow.set_tracking_uri(MLFLOW_URI)
model = mlflow.xgboost.load_model(MODEL_URI)
explainer = shap.TreeExplainer(model)
# TreeExplainer works well with XGBoost
explainer = shap.TreeExplainer(model)

# we convert log-odds into proablities for easy interpertation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_shap_values(X: pd.DataFrame):
    shap_values = explainer.shap_values(X)
    return shap_values, explainer
# we get matrix of (n_samples, n_features)


def get_top_features(X: pd.DataFrame, shap_values, top_n=3): 
    results = []

# we get feature,shap values and take the abs of them and sort to 
#get the major contributor irrespective of direction.
    for i in range(len(X)):
        vals = shap_values[i]
        features = X.columns

        pairs = list(zip(features, vals))
        pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)

        top = pairs[:top_n]

        results.append([
            {"feature": f, "impact": float(v)}
            for f, v in top
        ])

    return results


#easy  human interpertation
def format_explanations(X, shap_values, top_features, explainer):
    explanations = []

    #avg prediction value
    base_value = explainer.expected_value

    for i, row in enumerate(top_features):
        reasons = []

        # for each top feature, we get the log-odd value(shap) and convert to proablity
        shap_row = shap_values[i]
        total_log_odds = base_value + shap_row.sum()
        final_prob = sigmoid(total_log_odds)

        for item in row:
            feature = item["feature"]
            shap_val = item["impact"]

            # index of feature
            idx = list(X.columns).index(feature)
            log_odds_without = total_log_odds - shap_val
            prob_without = sigmoid(log_odds_without)

            prob_impact = final_prob - prob_without

            direction = "↑ risk" if shap_val > 0 else "↓ risk"

            reasons.append(
                f"{feature} {direction} by {abs(prob_impact)*100:.1f}% "
                f"(SHAP: {shap_val:+.2f} log-odds)"
            )

        explanations.append(reasons)

    return explanations