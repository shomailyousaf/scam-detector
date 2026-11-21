"""MLflow setup shared by every training script.

If DagsHub credentials are set, runs go to DagsHub.
If not, runs fall back to a local ./mlruns folder so graders can still run everything.
"""
import os
import mlflow

DAGSHUB_URI = "https://dagshub.com/shomailusuf/scam-detector.mlflow"
EXPERIMENT = "scam-detector"


def setup_mlflow():
    user = os.environ.get("MLFLOW_TRACKING_USERNAME")
    token = os.environ.get("MLFLOW_TRACKING_PASSWORD")
    if user and token:
        mlflow.set_tracking_uri(DAGSHUB_URI)
        mode = "DagsHub"
    else:
        mlflow.set_tracking_uri("file:./mlruns")
        mode = "local ./mlruns (DagsHub credentials not set)"
    mlflow.set_experiment(EXPERIMENT)
    print(f"[mlflow] tracking -> {mode}")
    return mode


if __name__ == "__main__":
    setup_mlflow()
    with mlflow.start_run(run_name="connection_test"):
        mlflow.log_param("hello", "world")
        mlflow.log_metric("test_metric", 1.0)
    print("[mlflow] test run logged OK")