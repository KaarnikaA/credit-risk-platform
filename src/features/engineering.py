import pandas as pd
import numpy as np

#convert df to parquet
def safe_qcut(series, q=5):
    try:
        return pd.qcut(series, q, labels=False, duplicates="drop")
    except Exception:
        return pd.Series([0] * len(series))

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()


    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"]

    # dti scaling to deci,al
    df["dti_ratio"] = df["dti"] / 100  

    #to compress skewness
    df["log_income"] = np.log1p(df["annual_inc"])
    df["log_loan"] = np.log1p(df["loan_amnt"])

    df["income_bucket"] = safe_qcut(df["annual_inc"])
    df["loan_bucket"] = safe_qcut(df["loan_amnt"])

    df["income_x_dti"] = df["annual_inc"] * df["dti"]
    df["loan_x_dti"] = df["loan_amnt"] * df["dti"]

    df["high_dti_flag"] = (df["dti"] > 20).astype(int)
    df["low_income_flag"] = (df["annual_inc"] < 30000).astype(int)
    df["high_loan_flag"] = (df["loan_amnt"] > 30000).astype(int)

    df["dti"] = df["dti"].clip(0, 50)

    return df

