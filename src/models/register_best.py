"""Register the best run's model in the MLflow Model Registry as version 1.

Attempt 1: register straight from the logged model's storage location.
Attempt 2: re-attach the model files to the run as plain artifacts and register those.
"""
import sys
import tempfile

import mlflow
from mlflow import MlflowClient

from src.models.mlflow_utils import EXPERIMENT, setup_mlflow

RUN_NAME = "xlmr_base"
MODEL_NAME = "scam-detector-xlmr"

setup_mlflow()
client = MlflowClient()
exp = client.get_experiment_by_name(EXPERIMENT)
runs = client.search_runs(
    [exp.experiment_id],
    filter_string=f"tags.mlflow.runName = '{RUN_NAME}'",
    order_by=["attributes.start_time DESC"],
    max_results=1,
)
if not runs:
    sys.exit(f"No run named {RUN_NAME} found. Are your DagsHub credentials set?")
run = runs[0]
run_id = run.info.run_id
f1 = round(run.data.metrics["test_scam_f1"], 4)
print("run:", run_id, "test_scam_f1:", f1)

# find the MLflow 3 logged model for this run
found = mlflow.search_logged_models(
    experiment_ids=[exp.experiment_id],
    filter_string=f"source_run_id = '{run_id}'",
    output_format="list",
)
if not found:
    sys.exit("Could not find the logged model for this run.")
lm = found[0]
print("logged model id:", lm.model_id)
print("artifact location:", lm.artifact_location)

# make sure the registry entry exists
try:
    client.create_registered_model(MODEL_NAME)
except Exception:
    pass  # already exists


def done(mv):
    client.set_model_version_tag(MODEL_NAME, mv.version, "test_scam_f1", str(f1))
    print(f"Registered {MODEL_NAME} version {mv.version}")
    sys.exit(0)


# ---- Attempt 1
try:
    mv = client.create_model_version(
        MODEL_NAME, source=lm.artifact_location, run_id=run_id)
    done(mv)
except SystemExit:
    raise
except Exception as e:
    print("Attempt 1 failed:", e)

# ---- Attempt 2 (downloads about 1 GB, then uploads it again)
try:
    local = mlflow.artifacts.download_artifacts(
        artifact_uri=lm.artifact_location, dst_path=tempfile.mkdtemp())
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifacts(local, artifact_path="model_files")
    mv = client.create_model_version(
        MODEL_NAME, source=f"runs:/{run_id}/model_files", run_id=run_id)
    done(mv)
except SystemExit:
    raise
except Exception as e:
    print("Attempt 2 failed:", e)
    sys.exit("Registration failed on both routes. Paste the two errors above.")