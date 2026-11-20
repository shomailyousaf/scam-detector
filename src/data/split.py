from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io import load_params

SPLITS = ("train", "val", "test")


def make_split(df: pd.DataFrame, cfg: dict) -> pd.Series:
    s = cfg["split"]
    ratios = {"test": s["test_ratio"], "val": s["val_ratio"]}
    ratios["train"] = 1.0 - ratios["test"] - ratios["val"]

    stratum = df["label"] + "_" + df["language"]
    comp = pd.crosstab(df["split_group"], stratum)          # groups x strata counts
    counts = comp.to_numpy(dtype=float)
    totals = counts.sum(axis=0)
    target = np.array([totals * ratios[sp] for sp in SPLITS])  # splits x strata
    current = np.zeros_like(target)

    # biggest groups first; equal sizes shuffled with the fixed seed
    rng = np.random.default_rng(s["seed"])
    order = np.lexsort((rng.random(len(comp)), -counts.sum(axis=1)))

    assign = np.empty(len(comp), dtype=object)
    for i in order:
        c = counts[i]
        nz = c > 0
        # relative deficit of each split for this group's label x language mix
        need = (((target[:, nz] - current[:, nz]) / target[:, nz]) * c[nz]).sum(axis=1)
        best = int(np.argmax(need))
        assign[i] = SPLITS[best]
        current[best] += c

    return df["split_group"].map(dict(zip(comp.index, assign)))


def main() -> None:
    cfg = load_params()
    df = pd.read_csv(cfg["data"]["clean_path"], keep_default_na=False)
    split = make_split(df, cfg)
    out_dir = Path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        part = df[split == name]
        part.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"[split] {name:5s} {len(part):6d} rows ({len(part)/len(df):.1%})  scam={part.label_id.mean():.1%}")


if __name__ == "__main__":
    main()
