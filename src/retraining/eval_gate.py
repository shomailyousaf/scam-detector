# evaluation gate for pull requests
# builds the model the way this branch would, using the same data as the production champion
# scores it on the fixed test and challenge sets, and fails if it is worse than the champion
# usage
#   python -m src.retraining.eval_gate
import os
import sys
from pathlib import Path

import pandas as pd
from mlflow import MlflowClient

from src.common.io import load_params
from src.models.mlflow_utils import setup_mlflow
from src.retraining.common import (build_pipeline, evaluate, get_role, read, version_metrics,
                                   write_json)

KEYS = ("test_scam_f1", "challenge_v2_scam_f1")


def champion_training_data(cfg, version):
    # the champion may have been trained on train plus a new batch, so we use the same data
    tags = MlflowClient().get_model_version(cfg["retraining"]["model_name"], str(version)).tags
    parts, used = [read(Path(cfg["data"]["processed_dir"]) / "train.csv")], ["train"]
    path = tags.get("batch_path")
    if not path and tags.get("batch"):
        found = sorted(Path(cfg["monitoring"]["batches_dir"]).glob(f"*_{tags['batch']}.csv"))
        path = found[0].as_posix() if found else None
    if path:
        parts.append(read(path))
        used.append(Path(path).stem)
    df = pd.concat([p[["text", "label_id"]] for p in parts], ignore_index=True)
    return df, "+".join(used)


def gate(candidate, champion, tolerance):
    return {k: bool(candidate[k] >= champion[k] - tolerance) for k in KEYS}


def summary(res):
    lines = ["## Model evaluation gate", "",
             f"Production champion: version {res['champion_version']} (trained on {res['trained_on']})", "",
             "| metric | champion | this branch | result |", "|---|---|---|---|"]
    for k in KEYS:
        ok = res["checks"][k]
        lines.append(f"| {k} | {res['champion'][k]:.4f} | {res['candidate'][k]:.4f} | {'pass' if ok else 'FAIL'} |")
    lines += ["", f"**{'PASSED, safe to merge' if res['passed'] else 'BLOCKED, this branch makes the model worse'}**"]
    return "\n".join(lines)


def main():
    cfg = load_params()
    setup_mlflow()
    champ_v = get_role(cfg, "champion")
    if champ_v is None:
        sys.exit("[gate] no production champion found in the registry")

    train, trained_on = champion_training_data(cfg, champ_v)
    model = build_pipeline(cfg).fit(train["text"], train["label_id"])
    candidate, champion = evaluate(model, cfg), version_metrics(cfg, champ_v)
    checks = gate(candidate, champion, cfg["retraining"]["gate_tolerance"])
    res = {"champion_version": champ_v, "trained_on": trained_on, "candidate": candidate,
           "champion": champion, "checks": checks, "passed": all(checks.values())}
    write_json(res, "reports/eval_gate.json")

    text = summary(res)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(text + "\n")
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
