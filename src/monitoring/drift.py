import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=SyntaxWarning)

from evidently import DataDefinition, Dataset, Report  # noqa: E402
from evidently.presets import DataDriftPreset  # noqa: E402

from src.common.io import load_params  # noqa: E402

OUT_DIR = Path("reports/drift")
NUMERIC = ["n_chars", "n_words", "digit_ratio", "upper_ratio", "urdu_ratio",
           "has_url", "has_money", "has_code"]
CATEGORICAL = ["language"]

_URL = re.compile(r"https?://|www\.|\.(?:com|pk|xyz|top|click|ly)\b|bit\.ly|tinyurl", re.I)
_MONEY = re.compile(r"\b(?:rs|pkr|rupay|rupees?|paisay|paise|paisa|amount|cash|fee)\b|روپ|پیس", re.I)
_CODE = re.compile(r"\b(?:otp|code|pin|mpin|cvv|password|verification)\b|کوڈ|پن", re.I)
_URDU = re.compile(r"[؀-ۿ]")


def features(df):
    # raw text cant be compared directly so we turn each message into a few simple numbers
    text = df["text"].astype(str)
    n = text.str.len().clip(lower=1)
    out = pd.DataFrame({
        "n_chars": text.str.len(),
        "n_words": text.str.split().str.len(),
        "digit_ratio": text.str.count(r"\d") / n,
        "upper_ratio": text.str.count(r"[A-Z]") / n,
        "urdu_ratio": text.str.count(_URDU) / n,
        "has_url": text.str.contains(_URL).astype(int),
        "has_money": text.str.contains(_MONEY).astype(int),
        "has_code": text.str.contains(_CODE).astype(int),
    })
    out["language"] = df["language"].astype(str) if "language" in df else "unknown"
    return out


def check_drift(current, reference, drift_share, name, out_dir=OUT_DIR, threshold=0.25):
    definition = DataDefinition(numerical_columns=NUMERIC, categorical_columns=CATEGORICAL)
    # same distance tests for every batch size so results dont flip with sample size
    preset = DataDriftPreset(drift_share=drift_share, num_method="wasserstein",
                             cat_method="jensenshannon", threshold=threshold)
    snapshot = Report([preset]).run(
        current_data=Dataset.from_pandas(features(current), data_definition=definition),
        reference_data=Dataset.from_pandas(features(reference), data_definition=definition),
    )

    columns, share = {}, 0.0
    for m in snapshot.dict()["metrics"]:
        cfg = m["config"]
        if cfg["type"].endswith("DriftedColumnsCount"):
            share = float(m["value"]["share"])
        elif cfg["type"].endswith("ValueDrift"):
            score, threshold, method = float(m["value"]), float(cfg["threshold"]), cfg["method"]
            # a p value is drift when low, a distance is drift when high
            drifted = score < threshold if "p_value" in method else score >= threshold
            columns[cfg["column"]] = {
                "score": round(score, 4),
                "threshold": threshold,
                "method": method,
                "drifted": bool(drifted),
            }

    result = {
        "batch": name,
        "rows": int(len(current)),
        "drift_detected": share >= drift_share,
        "drifted_share": round(share, 4),
        "drift_share_threshold": drift_share,
        "drifted_features": sorted(c for c, v in columns.items() if v["drifted"]),
        "features": columns,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out_dir / f"{name}.html"))
    (out_dir / f"{name}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    return result


def load_reference(cfg):
    m = cfg["monitoring"]
    ref = pd.read_csv(m["reference_path"], keep_default_na=False)
    return ref.sample(min(m["reference_sample"], len(ref)), random_state=m["seed"])


def main():
    cfg = load_params()
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--fail-on-drift", action="store_true")
    a = ap.parse_args()

    current = pd.read_csv(a.current, keep_default_na=False)
    m = cfg["monitoring"]
    res = check_drift(current, load_reference(cfg), m["drift_share"], a.name,
                      threshold=m["feature_threshold"])

    status = "DRIFT DETECTED" if res["drift_detected"] else "no drift"
    print(f"[drift] {a.name}: {status}  ({res['drifted_share']:.0%} of features drifted, "
          f"alert at {res['drift_share_threshold']:.0%})")
    for c, v in res["features"].items():
        print(f"   {'DRIFT' if v['drifted'] else 'ok   '}  {c:12s} score={v['score']:.3f} (threshold {v['threshold']})")
    print(f"[drift] report: {OUT_DIR / (a.name + '.html')}")
    sys.exit(2 if (a.fail_on_drift and res["drift_detected"]) else 0)


if __name__ == "__main__":
    main()
