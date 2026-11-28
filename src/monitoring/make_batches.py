from pathlib import Path

import pandas as pd

from src.common.io import load_params


def make_batches(cfg):
    d, m = cfg["data"], cfg["monitoring"]
    clean = pd.read_csv(d["clean_path"], keep_default_na=False)
    val = pd.read_csv(Path(d["processed_dir"]) / "val.csv", keep_default_na=False)
    challenge = pd.read_csv(Path(d["processed_dir"]) / "challenge_v2.csv", keep_default_na=False)

    normal = val.sample(m["normal_size"], random_state=m["seed"])

   
    v2 = clean[clean["source"] == d["additions_source"]]
    everyday = v2[~v2["split_group"].isin(set(challenge["split_group"]))]

    # same messages so drift is still detected, but the labels came back wrong
    bad = everyday.copy()
    bad["label_id"] = 1 - bad["label_id"]
    bad["label"] = bad["label_id"].map({1: "SCAM", 0: "NORMAL"})

    return {"batch_01_normal": normal, "batch_02_everyday": everyday, "batch_03_bad_labels": bad}


def main():
    cfg = load_params()
    out = Path(cfg["monitoring"]["batches_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for name, df in make_batches(cfg).items():
        df.to_csv(out / f"{name}.csv", index=False, lineterminator="\n")
        print(f"[batches] {name:20s} {len(df):5d} rows  scam={df['label_id'].mean():.1%}")


if __name__ == "__main__":
    main()
