from pathlib import Path

import pandas as pd
import pytest

from src.common.io import load_params
from src.common.textnorm import dedup_key
from src.data.check_splits import check
from src.data.prepare import prepare
from src.data.split import make_split
from src.data.validate import validate

ROOT = Path(__file__).resolve().parents[1]
CFG = load_params(ROOT / "params.yaml")


def good_df(n: int = 1200) -> pd.DataFrame:
    """Small synthetic dataset that satisfies the raw schema."""
    langs = ["English", "Roman-Urdu", "Urdu", "Code-Mixed"]
    rows = []
    for i in range(n):
        scam = i % 2 == 0
        rows.append({
            "id": i,
            "group_id": f"g{i // 2}",
            "text": (f"Send your OTP now or account {i} will be blocked" if scam
                     else f"Meeting moved to room {i}, see you there"),
            "label": "SCAM" if scam else "NORMAL",
            "language": langs[i % 4],
            "scam_type": "otp_request" if scam else None,
            "source": "test",
            "synthetic": True,
        })
    return pd.DataFrame(rows)


def run_validate(df: pd.DataFrame, tmp_path) -> bool:
    p = tmp_path / "raw.csv"
    df.to_csv(p, index=False)
    return validate(str(p), str(tmp_path / "rep.json"), CFG)


# ---------- validation catches corrupted / missing inputs ----------

def test_good_data_passes(tmp_path):
    assert run_validate(good_df(), tmp_path)


@pytest.mark.parametrize("corrupt", [
    "missing_column", "empty_text", "unknown_label", "unknown_language",
    "duplicate_id", "too_few_rows", "only_one_class", "wrong_type",
])
def test_corrupted_data_fails(tmp_path, corrupt):
    df = good_df()
    if corrupt == "missing_column":
        df = df.drop(columns=["label"])
    elif corrupt == "empty_text":
        df.loc[5, "text"] = "   "
    elif corrupt == "unknown_label":
        df.loc[5, "label"] = "MAYBE"
    elif corrupt == "unknown_language":
        df.loc[5, "language"] = "French"
    elif corrupt == "duplicate_id":
        df.loc[5, "id"] = df.loc[6, "id"]
    elif corrupt == "too_few_rows":
        df = df.head(50)
    elif corrupt == "only_one_class":
        df["label"] = "SCAM"
    elif corrupt == "wrong_type":
        df["synthetic"] = "yes"
    assert not run_validate(df, tmp_path)


def test_unreadable_file_fails(tmp_path):
    p = tmp_path / "raw.csv"
    p.write_bytes(b"\x00\x01garbage\x00")
    assert not validate(str(p), str(tmp_path / "rep.json"), CFG)


# ---------- near-duplicate key ----------

def test_dedup_key_groups_templates():
    a = "Mubarak! Aap ne Rs 5000 jeete, visit http://win.pk/a1"
    b = "mubarak!  aap ne rs 75000 jeete, visit https://claim.xyz/zz"
    assert dedup_key(a) == dedup_key(b)


def test_dedup_key_keeps_different_messages_apart():
    assert dedup_key("Apna OTP abhi bhejo") != dedup_key("Apna OTP kisi ko mat dena")


# ---------- split is leakage-free and reproducible ----------

def splits_from(df):
    clean, _ = prepare(df, CFG)
    s = make_split(clean, CFG)
    return clean, s, {n: clean[s == n].reset_index(drop=True) for n in ("train", "val", "test")}


def test_split_has_no_leakage():
    _, _, parts = splits_from(good_df())
    rep = check(parts, CFG)
    assert rep["checks"]["no_group_overlap"]
    assert rep["checks"]["no_near_duplicate_overlap"]
    assert rep["checks"]["no_id_overlap"]


def test_split_is_reproducible():
    _, s1, _ = splits_from(good_df())
    _, s2, _ = splits_from(good_df())
    assert (s1 == s2).all()


def test_leakage_check_detects_overlap():
    _, _, parts = splits_from(good_df())
    parts["test"] = pd.concat([parts["test"], parts["train"].head(3)], ignore_index=True)
    rep = check(parts, CFG)
    assert not rep["checks"]["no_group_overlap"]
    assert not rep["passed"]


def test_contradicting_duplicates_are_dropped():
    df = good_df()
    df.loc[1, "text"] = df.loc[0, "text"]  # same text, opposite label
    clean, rep = prepare(df, CFG)
    assert rep["dropped_conflicting_label_duplicates"] >= 2
    assert clean["dedup_key"].groupby(clean["dedup_key"]).size().index.is_unique


# ---------- real processed data (only if pipeline has run) ----------

PROCESSED = ROOT / CFG["data"]["processed_dir"]


@pytest.mark.skipif(not (PROCESSED / "test.csv").exists(), reason="run `dvc repro` first")
def test_real_splits_are_clean():
    parts = {n: pd.read_csv(PROCESSED / f"{n}.csv", keep_default_na=False)
             for n in ("train", "val", "test")}
    assert check(parts, CFG)["passed"]
