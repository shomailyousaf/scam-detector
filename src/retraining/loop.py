# the whole closed loop in one command
# drift check, retrain if needed, compare with the champion, promote or reject
# usage
#   python -m src.retraining.loop --batch data/batches/batch_02_everyday.csv
#   python -m src.retraining.loop --latest      picks the newest file in data/incoming
import argparse
import os
import sys
from pathlib import Path

from src.common.io import load_params
from src.models.mlflow_utils import setup_mlflow
from src.retraining.promote import promote
from src.retraining.retrain import run_retrain
from src.retraining.common import write_json


def latest_incoming(folder="data/incoming"):
    files = sorted(Path(folder).glob("*.csv"))
    return files[-1] if files else None


def summary(r, p):
    d = r["input_drift"]
    lines = ["## Retraining loop", "",
             f"Batch: `{r['batch']}` ({r['batch_rows']} messages)", "",
             f"- Input drift: **{'DRIFT' if d['drift_detected'] else 'no drift'}** "
             f"({d['drifted_share']:.0%} of features)",
             f"- Prediction drift: **{'DRIFT' if r['prediction_drift']['drifted'] else 'no drift'}** "
             f"(score {r['prediction_drift']['score']})"]
    if not r["retrained"]:
        lines.append(f"- No retraining needed, champion stays version {r['champion_version']}")
        return "\n".join(lines)
    c, h = p["champion_metrics"], p["challenger_metrics"]
    lines += [f"- Retrained, challenger is version {r['challenger_version']}", "",
              "| metric | champion | challenger |", "|---|---|---|"]
    for k in ("test_scam_f1", "challenge_v2_scam_f1", "challenge_v2_false_alarm_rate"):
        lines.append(f"| {k} | {c[k]:.4f} | {h[k]:.4f} |")
    lines += ["", f"**Decision: {p['decision'].upper()}, champion is now version {p['champion_after']}**"]
    return "\n".join(lines)


def main():
    cfg = load_params()
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch")
    ap.add_argument("--name")
    ap.add_argument("--latest", action="store_true")
    a = ap.parse_args()

    batch = Path(a.batch) if a.batch else (latest_incoming() if a.latest else None)
    if batch is None or not batch.exists():
        sys.exit("[loop] no batch to check")
    name = a.name or batch.stem

    setup_mlflow()
    report = run_retrain(cfg, batch.as_posix(), name)
    decision = promote(cfg, report["challenger_version"]) if report["retrained"] else None
    write_json({"retraining": report, "decision": decision}, f"reports/retraining/loop_{name}.json")

    text = summary(report, decision)
    print("\n" + text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
