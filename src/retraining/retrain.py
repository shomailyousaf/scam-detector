# the retraining trigger
# checks a new batch for input drift and prediction drift
# if drift is found it trains a challenger on train plus the batch and registers it
# usage
#   python -m src.retraining.retrain --batch data/batches/batch_02_everyday.csv --name everyday
import argparse
import sys
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.common.io import load_params
from src.models.mlflow_utils import setup_mlflow
from src.monitoring.drift import check_drift, load_reference
from src.retraining.common import (load_role, read, set_role, train_log_register,
                                   version_metrics, write_json)


def prediction_drift(model, reference, batch, threshold):
    # does the champion score the new batch very differently from the training data
    ref = pd.DataFrame({"scam_probability": model.predict_proba(reference["text"])[:, 1]})
    cur = pd.DataFrame({"scam_probability": model.predict_proba(batch["text"])[:, 1]})
    definition = DataDefinition(numerical_columns=["scam_probability"])
    snap = Report([DataDriftPreset(num_method="wasserstein", threshold=threshold)]).run(
        current_data=Dataset.from_pandas(cur, data_definition=definition),
        reference_data=Dataset.from_pandas(ref, data_definition=definition))
    score = next(float(m["value"]) for m in snap.dict()["metrics"]
                 if m["config"]["type"].endswith("ValueDrift"))
    return {
        "score": round(score, 4),
        "drifted": score >= threshold,
        "mean_probability_reference": round(float(ref["scam_probability"].mean()), 4),
        "mean_probability_batch": round(float(cur["scam_probability"].mean()), 4),
    }


def run_retrain(cfg, batch_path, name, force=False):
    m = cfg["monitoring"]
    batch = read(batch_path)
    reference = load_reference(cfg)
    report = {"batch": name, "batch_rows": len(batch)}

    inp = check_drift(batch, reference, m["drift_share"], name, threshold=m["feature_threshold"])
    report["input_drift"] = {k: inp[k] for k in ("drift_detected", "drifted_share", "drifted_features")}

    champion, champ_v = load_role(cfg, "champion")
    if champion is None:
        sys.exit("no champion yet, run python -m src.retraining.bootstrap_champion first")
    report["champion_version"] = champ_v
    report["prediction_drift"] = prediction_drift(champion, reference, batch, m["feature_threshold"])

    print(f"[retrain] input drift: {inp['drift_detected']}  ({inp['drifted_share']:.0%} of features)")
    pd_ = report["prediction_drift"]
    print(f"[retrain] prediction drift: {pd_['drifted']}  (score {pd_['score']}, mean scam probability "
          f"{pd_['mean_probability_reference']} on training data vs {pd_['mean_probability_batch']} on batch)")

    if not (inp["drift_detected"] or force):
        report["retrained"] = False
        write_json(report, f"reports/retraining/{name}.json")
        print("[retrain] no drift, champion stays, nothing to retrain")
        return report

    print("[retrain] drift alert, training a challenger on train plus the new batch")
    train = read(Path(cfg["data"]["processed_dir"]) / "train.csv")
    new_train = pd.concat([train[["text", "label_id"]], batch[["text", "label_id"]]], ignore_index=True)
    _, version, metrics, run_id = train_log_register(
        new_train, cfg, f"challenger_{name}",
        {"trigger": "drift_alert", "trained_on": f"train+{name}", "batch": name,
         "batch_path": Path(batch_path).as_posix()})
    set_role(cfg, "challenger", version)

    report.update({"retrained": True, "challenger_version": version, "challenger_run_id": run_id,
                   "challenger": metrics, "champion": version_metrics(cfg, champ_v)})
    write_json(report, f"reports/retraining/{name}.json")
    print(f"[retrain] registered version {version} as challenger")
    for k in ("test_scam_f1", "challenge_v2_scam_f1", "challenge_v2_false_alarm_rate"):
        print(f"   {k:32s} champion {report['champion'][k]:.4f}   challenger {metrics[k]:.4f}")
    return report


def main():
    cfg = load_params()
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--force", action="store_true", help="retrain even without drift")
    a = ap.parse_args()
    setup_mlflow()
    run_retrain(cfg, a.batch, a.name, a.force)


if __name__ == "__main__":
    main()
