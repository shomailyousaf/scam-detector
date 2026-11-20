import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
import pandera as pa

from src.common.io import load_params, write_json
from src.data.schema import processed_schema


def check(splits: dict, cfg: dict) -> dict:
    v = cfg["validation"]
    rep = {"checks": {}, "sizes": {}, "scam_ratio": {}, "language_counts": {}}

    schema = processed_schema(cfg)
    schema_ok = True
    for name, d in splits.items():
        try:
            schema.validate(d, lazy=True)
        except (pa.errors.SchemaErrors, pa.errors.SchemaError) as e:
            schema_ok = False
            rep.setdefault("schema_errors", {})[name] = str(e)[:500]
    rep["checks"]["schema_valid"] = schema_ok

    for col, label in (("split_group", "no_group_overlap"),
                       ("dedup_key", "no_near_duplicate_overlap"),
                       ("id", "no_id_overlap")):
        overlaps = {f"{a}-{b}": len(set(splits[a][col]) & set(splits[b][col]))
                    for a, b in combinations(splits, 2)}
        rep[f"{label}_counts"] = overlaps
        rep["checks"][label] = all(n == 0 for n in overlaps.values())

    langs = set(v["allowed_languages"])
    rep["checks"]["all_languages_in_every_split"] = all(
        langs <= set(d["language"]) for d in splits.values())

    for name, d in splits.items():
        rep["sizes"][name] = int(len(d))
        rep["scam_ratio"][name] = round(float(d["label_id"].mean()), 4)
        rep["language_counts"][name] = d["language"].value_counts().to_dict()
    drift = max(abs(rep["scam_ratio"][s] - rep["scam_ratio"]["train"]) for s in ("val", "test"))
    rep["max_scam_ratio_drift"] = round(drift, 4)
    rep["checks"]["class_balance_consistent"] = drift <= v["max_split_ratio_drift"]

    rep["passed"] = all(rep["checks"].values())
    return rep


def main() -> None:
    cfg = load_params()
    d = Path(cfg["data"]["processed_dir"])
    splits = {n: pd.read_csv(d / f"{n}.csv", keep_default_na=False) for n in ("train", "val", "test")}
    rep = check(splits, cfg)
    write_json(rep, "reports/split_report.json")
    for k, ok in rep["checks"].items():
        print(f"[check_splits] {'PASS' if ok else 'FAIL'}  {k}")
    sys.exit(0 if rep["passed"] else 1)


if __name__ == "__main__":
    main()
