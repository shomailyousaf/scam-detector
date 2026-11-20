"""Stage 1 - validate the RAW data before anything else runs.

If the CSV is corrupted (missing column, empty text, unknown label, wrong
types, too few rows, broken class balance) this script exits with code 1,
so `dvc repro` stops and training never starts on bad data.

Usage:  python -m src.data.validate [--input path] [--report path]
"""
import argparse
import sys

import pandas as pd
import pandera as pa

from src.common.io import load_params, write_json
from src.data.schema import raw_schema


def quality_summary(df: pd.DataFrame, cfg: dict) -> dict:
    """Non-fatal quality facts, written to the report for the README/viva."""
    lens = df["text"].astype(str).str.len()
    return {
        "rows": int(len(df)),
        "label_counts": df["label"].value_counts().to_dict(),
        "language_counts": df["language"].value_counts().to_dict(),
        "scam_ratio": round(float((df["label"] == "SCAM").mean()), 4),
        "synthetic_rows": int(df["synthetic"].sum()),
        "exact_duplicate_texts": int(df["text"].duplicated().sum()),
        "texts_longer_than_max": int((lens > cfg["prepare"]["max_text_chars"]).sum()),
        "missing_scam_type_in_scam_rows": int(
            df.loc[df["label"] == "SCAM", "scam_type"].isna().sum()),
    }


def validate(input_path: str, report_path: str, cfg: dict) -> bool:
    report = {"input": input_path, "passed": False, "errors": []}
    try:
        df = pd.read_csv(input_path)
    except Exception as e:  # unreadable / corrupted file
        report["errors"].append(f"cannot read CSV: {type(e).__name__}: {e}")
        write_json(report, report_path)
        return False

    try:
        raw_schema(cfg).validate(df, lazy=True)   # lazy = collect ALL errors
        report["passed"] = True
    except pa.errors.SchemaErrors as err:
        fc = err.failure_cases
        report["errors"] = (
            fc[["column", "check", "failure_case"]].astype(str).head(50).to_dict("records"))
        report["n_failures"] = int(len(fc))
    except pa.errors.SchemaError as err:
        report["errors"].append(str(err)[:500])

    if report["passed"] or all(c in df.columns for c in ("text", "label", "language", "synthetic", "scam_type")):
        try:
            report["quality"] = quality_summary(df, cfg)
        except Exception:
            pass
    write_json(report, report_path)
    return report["passed"]


def main() -> None:
    cfg = load_params()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=cfg["data"]["raw_path"])
    ap.add_argument("--report", default="reports/validation_raw.json")
    a = ap.parse_args()

    ok = validate(a.input, a.report, cfg)
    print(f"[validate] {a.input}: {'PASSED' if ok else 'FAILED'}  (details: {a.report})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
