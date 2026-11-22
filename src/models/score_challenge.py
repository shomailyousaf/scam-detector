"""Score the registered XLM-R (scam-detector-xlmr v1) on the v2 challenge set.

Runs on CPU. Logs to MLflow (DagsHub, or local fallback).
"""
import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow import MlflowClient
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.models.mlflow_utils import setup_mlflow

MODEL_NAME = "scam-detector-xlmr"
VERSION = "1"
CHALLENGE = Path("data/processed/challenge_v2.csv")
REPORTS = Path("reports")

setup_mlflow()
client = MlflowClient()
mv = client.get_model_version(MODEL_NAME, VERSION)
print("model source:", mv.source)

local = mlflow.artifacts.download_artifacts(artifact_uri=mv.source, dst_path=tempfile.mkdtemp())
# find the folder that holds config.json
cfg = next(Path(local).rglob("config.json"))
model_dir = cfg.parent
print("model dir:", model_dir)

tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()

df = pd.read_csv(CHALLENGE).fillna({"text": ""})
probs = []
with torch.no_grad():
    for i in range(0, len(df), 16):
        enc = tok(df["text"].iloc[i:i + 16].tolist(), truncation=True,
                  max_length=128, padding=True, return_tensors="pt")
        p = torch.softmax(model(**enc).logits, dim=-1)[:, 1].numpy()
        probs.append(p)
prob = np.concatenate(probs)
pred = (prob >= 0.5).astype(int)
y = df["label_id"].to_numpy()

tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
m = {
    "challenge_scam_f1": f1_score(y, pred, zero_division=0),
    "challenge_scam_precision": precision_score(y, pred, zero_division=0),
    "challenge_scam_recall": recall_score(y, pred, zero_division=0),
    "challenge_false_alarm_rate": fp / (fp + tn),
}
print(m)
print(f"TN={tn} FP={fp} FN={fn} TP={tp}")

# wrong predictions, for error analysis
wrong = df.assign(pred=pred, prob=prob.round(3))
wrong = wrong[wrong["pred"] != wrong["label_id"]][["text", "label", "pred", "prob", "language", "scam_type"]]
wrong.to_csv(REPORTS / "challenge_xlmr_errors.csv", index=False, encoding="utf-8", lineterminator="\n")

with mlflow.start_run(run_name="xlmr_challenge_v2"):
    mlflow.log_params({"model": f"{MODEL_NAME} v{VERSION}", "eval_set": "challenge_v2", "n": len(df)})
    for k, v in m.items():
        mlflow.log_metric(k, v)

(REPORTS / "metrics_xlmr_challenge_v2.json").write_text(
    json.dumps({k: round(v, 6) for k, v in m.items()}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8", newline="\n")
print("done")