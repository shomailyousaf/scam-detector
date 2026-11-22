"""Tests for data v2 (generic everyday messages added on top of v1)."""
from pathlib import Path

import pandas as pd

from src.common.io import load_params
from src.data.make_v2_additions import build
from src.data.prepare import prepare, prepare_additions
from src.data.schema import raw_schema
from src.data.split import make_split

ROOT = Path(__file__).resolve().parents[1]
CFG = load_params(ROOT / "params.yaml")


def test_v2_generator_is_reproducible():
    assert build().equals(build())          


def test_v2_additions_pass_the_raw_schema():
    raw_schema(CFG, CFG["validation"]["min_rows_additions"]).validate(build(), lazy=True)


def test_variants_of_one_template_share_a_group():
    df = build()
    
    assert df.groupby("group_id")["label"].nunique().max() == 1
    assert df.groupby("group_id").size().max() <= 3


def test_template_variants_never_cross_splits():
    clean, _ = prepare(build(), CFG)
    split = make_split(clean, CFG)
    per_group = pd.DataFrame({"g": clean["group_id"] if "group_id" in clean else clean["split_group"],
                              "s": split}).groupby("g")["s"].nunique()
    assert per_group.max() == 1


def test_additions_overlapping_v1_are_dropped():
    add = build()
    v1_raw = add.head(5).copy()             
    v1_raw["id"] = range(1, 6)
    v1_raw["group_id"] = [f"old{i}" for i in range(5)]
    v1_clean, _ = prepare(v1_raw, CFG)
    out, rep = prepare_additions(add, v1_clean, CFG)
    assert rep["dropped_overlap_with_v1"] >= 5
    assert not set(out["dedup_key"]) & set(v1_clean["dedup_key"])
    assert out["split_group"].min() > v1_clean["split_group"].max()   
