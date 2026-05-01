# must be in airflow registed dag folder

from datetime import datetime, timedelta
import os
import json
import logging
import subprocess

from airflow import DAG
from airflow.operators.python import PythonOperator

import mlflow
from mlflow.tracking import MlflowClient
import mlflow.xgboost

from feast import FeatureStore
import redis

# =============================
# CONFIG
# =============================
BASE_DIR = "/home/kaarvin/projects/credit-risk-platform"
FEAST_REPO = f"{BASE_DIR}/feature_repo"

mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_NAME = "credit_risk_model"

logger = logging.getLogger("airflow.task")


# =============================
# DEFAULT ARGS
# =============================
default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
}


# =============================
# TRAIN MODEL (FIXED)
# =============================
def train_model():
    logger.info("Starting training pipeline...")

    result = subprocess.run(
        ["python", f"{BASE_DIR}/src/models/train.py"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        logger.error(result.stderr)
        raise Exception("Training failed")

    logger.info("Training completed successfully")


# =============================
# VALIDATE MODEL (FIXED - MLflow API)
# =============================
def validate_model():
    client = MlflowClient()

    exp = client.get_experiment_by_name(MODEL_NAME)
    if not exp:
        raise Exception("Experiment not found")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.test_auc DESC"],
        max_results=1
    )

    best_run = runs[0]
    auc = best_run.data.metrics.get("test_auc")

    if auc is None:
        raise Exception("AUC missing in run")

    logger.info(f"Test AUC: {auc}")

    if auc < 0.60:
        raise Exception("Model rejected due to low AUC")

    return best_run.info.run_id


# =============================
# REGISTER MODEL (FIXED XGBOOST)
# =============================
def register_model(ti):
   
    run_id = ti.xcom_pull(task_ids="validate_model")

    logger.info(f"Registering model from run: {run_id}")

    mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=MODEL_NAME
    )

    logger.info("Model registered successfully")
# =============================
# FETCH THRESHOLD
# =============================
def fetch_threshold():
    client = MlflowClient()

    exp = client.get_experiment_by_name(MODEL_NAME)
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.test_auc DESC"],
        max_results=1
    )

    best_run = runs[0]

    threshold = best_run.data.params.get("best_threshold")
    if threshold is None:
        raise Exception("Threshold missing")

    logger.info(f"Threshold: {threshold}")
    return float(threshold)


# =============================
# FEAST MATERIALIZATION
# =============================
def materialize_features():
    store = FeatureStore(repo_path=FEAST_REPO)

    logger.info("Materializing Feast features...")
    store.materialize_incremental(end_date=datetime.now())
    logger.info("Feast materialization complete")


# =============================
# REDIS PUSH (VERSION SAFE)
# =============================
def push_to_redis():
    store = FeatureStore(repo_path=FEAST_REPO)
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)

    MODEL_VERSION = "v1"  # can later be dynamic from MLflow

    borrower_ids = [1, 2, 3]

    features_list = [
        "loan_features:annual_inc",
        "loan_features:loan_amnt",
        "loan_features:dti",
        "loan_features:loan_to_income",
    ]

    for bid in borrower_ids:

        resp = store.get_online_features(
            features=features_list,
            entity_rows=[{"borrower_id": bid}],
        ).to_dict()

        cleaned = {
            k: v[0]
            for k, v in resp.items()
            if v and v[0] is not None
        }

        if not cleaned:
            logger.warning(f"No features for borrower {bid}")
            continue

        r.set(
            f"loan_features:{MODEL_VERSION}:{bid}",
            json.dumps(cleaned)
        )

        logger.info(f"Redis updated for borrower {bid}")


# =============================
# DAG
# =============================
with DAG(
    dag_id="credit_risk_mlflow_pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["ml", "mlflow", "feast", "redis"],
) as dag:

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    validate = PythonOperator(
        task_id="validate_model",
        python_callable=validate_model
    )

    register = PythonOperator(
        task_id="register_model",
        python_callable=register_model
    )

    fetch_threshold = PythonOperator(
        task_id="fetch_threshold",
        python_callable=fetch_threshold
    )

    materialize = PythonOperator(
        task_id="materialize_features",
        python_callable=materialize_features
    )

    redis_push = PythonOperator(
        task_id="push_to_redis",
        python_callable=push_to_redis
    )


    # =============================
    # PIPELINE FLOW (FIXED)
    # =============================
    train >> validate >> register >> fetch_threshold

    materialize >> redis_push