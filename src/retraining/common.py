# shared pieces for training, scoring and registering the production model
import json
import subprocess
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import Pipeline

from src.models.mlflow_utils import setup_mlflow


def read(path):
    return pd.read_csv(path, keep_default_na=False)


def build_pipeline(cfg):
    r = cfg["retraining"]
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(r["ngram_min"], r["ngram_max"]),
                                  sublinear_tf=True, min_df=r["min_df"],
                                  max_features=r["max_features"])),
        ("clf", LogisticRegression(C=r["C"], class_weight="balanced", max_iter=1000,
                                   solver="liblinear", random_state=r["seed"])),
    ])


def scores(y, prob):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "scam_f1": round(f1_score(y, pred, zero_division=0), 6),
        "scam_precision": round(precision_score(y, pred, zero_division=0), 6),
        "scam_recall": round(recall_score(y, pred, zero_division=0), 6),
        "pr_auc": round(average_precision_score(y, prob), 6),
        "false_alarm_rate": round(fp / (fp + tn) if (fp + tn) else 0.0, 6),
    }


def evaluate(model, cfg):
    # always the same fixed sets so every model is judged on the same messages
    d = Path(cfg["data"]["processed_dir"])
    out = {}
    for name in ("test", "challenge_v2"):
        df = read(d / f"{name}.csv")
        prob = model.predict_proba(df["text"])[:, 1]
        out.update({f"{name}_{k}": v for k, v in scores(df["label_id"], prob).items()})
    return out


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def train_log_register(train_df, cfg, run_name, tags):
    # trains on train_df, scores it, logs to mlflow and adds a new registry version
    setup_mlflow()
    name = cfg["retraining"]["model_name"]
    model = build_pipeline(cfg).fit(train_df["text"], train_df["label_id"])
    metrics = evaluate(model, cfg)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({**cfg["retraining"], "train_rows": len(train_df)})
        mlflow.set_tags({**tags, "git_commit": git_commit()})
        mlflow.log_metrics(metrics)
        # saved as plain files so the registry works the same on dagshub and locally
        with tempfile.TemporaryDirectory() as tmp:
            # cloudpickle is much faster than the default format for a big tfidf vocabulary
            mlflow.sklearn.save_model(model, Path(tmp) / "model",
                                      serialization_format="cloudpickle",
                                      pip_requirements=["scikit-learn==1.5.2", "pandas==2.2.3"])
            mlflow.log_artifacts(str(Path(tmp) / "model"), artifact_path="model")
        run_id = run.info.run_id

    client = MlflowClient()
    try:
        client.create_registered_model(name)
    except Exception:
        pass
    mv = client.create_model_version(name, source=f"runs:/{run_id}/model", run_id=run_id)
    for k, v in metrics.items():
        client.set_model_version_tag(name, mv.version, k, str(v))
    for k, v in tags.items():
        client.set_model_version_tag(name, mv.version, k, str(v))
    return model, mv.version, metrics, run_id


def set_role(cfg, role, version):
    # roles like champion and challenger are saved twice
    # as an mlflow alias and as a version tag, because dagshub cant read aliases back
    client, name = MlflowClient(), cfg["retraining"]["model_name"]
    try:
        client.set_registered_model_alias(name, role, str(version))
    except Exception:
        pass
    for mv in client.search_model_versions(f"name='{name}'"):
        if mv.tags.get("role") == role and str(mv.version) != str(version):
            client.delete_model_version_tag(name, mv.version, "role")
    client.set_model_version_tag(name, str(version), "role", role)


def get_role(cfg, role):
    # returns the version number holding a role, or none
    client, name = MlflowClient(), cfg["retraining"]["model_name"]
    try:
        return int(client.get_model_version_by_alias(name, role).version)
    except Exception:
        pass
    try:
        for mv in client.search_model_versions(f"name='{name}'"):
            if mv.tags.get("role") == role:
                return int(mv.version)
    except Exception:
        pass
    return None


def find_version(cfg, tag, value):
    # finds a version by one of its tags, used to recover an earlier bootstrap
    client, name = MlflowClient(), cfg["retraining"]["model_name"]
    try:
        for mv in client.search_model_versions(f"name='{name}'"):
            if mv.tags.get(tag) == value:
                return int(mv.version)
    except Exception:
        pass
    return None


def version_metrics(cfg, version):
    mv = MlflowClient().get_model_version(cfg["retraining"]["model_name"], str(version))
    return {k: float(v) for k, v in mv.tags.items() if k.startswith(("test_", "challenge_v2_"))}


def load_role(cfg, role):
    v = get_role(cfg, role)
    if v is None:
        return None, None
    return mlflow.sklearn.load_model(f"models:/{cfg['retraining']['model_name']}/{v}"), v


def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
