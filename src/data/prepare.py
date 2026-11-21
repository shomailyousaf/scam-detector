import pandas as pd

from src.common.io import load_params, write_json
from src.common.textnorm import clean_text, dedup_key


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_split_groups(df: pd.DataFrame) -> pd.Series:
    """Connect rows that share a group_id OR a dedup_key."""
    uf = UnionFind()
    for gid, key in zip(df["group_id"], df["dedup_key"]):
        uf.union("g:" + str(gid), "k:" + key)
    roots = [uf.find("g:" + str(g)) for g in df["group_id"]]
    return pd.Series(pd.factorize(pd.Series(roots))[0], index=df.index)


def prepare(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    rep = {"rows_in": int(len(df))}

    df = df.copy()
    df["text"] = df["text"].map(clean_text)

    too_long = df["text"].str.len() > cfg["prepare"]["max_text_chars"]
    rep["dropped_too_long"] = int(too_long.sum())
    df = df[~too_long]

    df["scam_type"] = df["scam_type"].where(
        df["scam_type"].notna(), df["label"].map({"SCAM": "unspecified", "NORMAL": "none"}))

    df["dedup_key"] = df["text"].map(dedup_key)
    conflicting = df.groupby("dedup_key")["label"].transform("nunique") > 1
    rep["dropped_conflicting_label_duplicates"] = int(conflicting.sum())
    df = df[~conflicting]

    rep["near_duplicate_rows_kept_grouped"] = int(df["dedup_key"].duplicated(keep=False).sum())

    df["split_group"] = build_split_groups(df)
    rep["split_groups"] = int(df["split_group"].nunique())
    rep["largest_group_size"] = int(df["split_group"].value_counts().max())

    df["label_id"] = (df["label"] == "SCAM").astype(int)
    cols = ["id", "text", "label", "label_id", "language", "scam_type",
            "source", "synthetic", "split_group", "dedup_key"]
    df = df[cols].reset_index(drop=True)
    rep["rows_out"] = int(len(df))
    return df, rep


def main() -> None:
    cfg = load_params()
    df = pd.read_csv(cfg["data"]["raw_path"])
    out, rep = prepare(df, cfg)
    from pathlib import Path
    Path(cfg["data"]["clean_path"]).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg["data"]["clean_path"], index=False, lineterminator="\n")
    write_json(rep, "reports/prepare_report.json")
    print("[prepare]", rep)


if __name__ == "__main__":
    main()
