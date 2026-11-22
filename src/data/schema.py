
import pandera as pa
from pandera import Check, Column, DataFrameSchema


def raw_schema(cfg: dict, min_rows: int | None = None) -> DataFrameSchema:
    v = dict(cfg["validation"])
    if min_rows is not None:
        v["min_rows"] = min_rows
    return DataFrameSchema(
        {
            "id": Column(int, unique=True, nullable=False),
            "group_id": Column(str, nullable=False),
            "text": Column(
                str,
                nullable=False,
                checks=Check(lambda s: s.str.strip().str.len() >= v["min_text_chars"],
                             error="text is empty or too short"),
            ),
            "label": Column(str, Check.isin(v["allowed_labels"]), nullable=False),
            "language": Column(str, Check.isin(v["allowed_languages"]), nullable=False),
            "scam_type": Column(str, nullable=True),
            "source": Column(str, nullable=False),
            "synthetic": Column(bool, nullable=False),
        },
        checks=[
            Check(lambda df: len(df) >= v["min_rows"],
                  error=f"fewer than {v['min_rows']} rows"),
            Check(lambda df: v["scam_ratio_min"] <= (df["label"] == "SCAM").mean() <= v["scam_ratio_max"],
                  error="SCAM ratio outside allowed range"),
        ],
        strict=False,  
        coerce=False,   
    )


def processed_schema(cfg: dict) -> DataFrameSchema:
    v = cfg["validation"]
    return DataFrameSchema(
        {
            "id": Column(int, unique=True, nullable=False),
            "text": Column(str, Check(lambda s: s.str.len().between(
                v["min_text_chars"], cfg["prepare"]["max_text_chars"])), nullable=False),
            "label": Column(str, Check.isin(v["allowed_labels"])),
            "label_id": Column(int, Check.isin([0, 1])),
            "language": Column(str, Check.isin(v["allowed_languages"])),
            "scam_type": Column(str, nullable=False),
            "source": Column(str),
            "synthetic": Column(bool),
            "split_group": Column(int, nullable=False),
            "dedup_key": Column(str, nullable=False),
        },
        strict=True,   
        coerce=False,
    )
