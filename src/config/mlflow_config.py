import mlflow

MLFLOW_URI = "http://127.0.0.1:5000"

def init_mlflow():
    mlflow.set_tracking_uri(MLFLOW_URI)