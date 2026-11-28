# drift tests, skipped if evidently is not installed
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("evidently")

from src.common.io import load_params  # noqa: E402
from src.monitoring.drift import check_drift, features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = load_params(ROOT / "params.yaml")


def batch(texts, language, n):
    rows = [{"text": texts[i % len(texts)] + f" {i}", "language": language} for i in range(n)]
    return pd.DataFrame(rows)


ENGLISH = ["Your parcel is out for delivery today", "Meeting moved to 3pm, see you there",
           "Happy birthday! Have a great day", "Can you send the notes from class"]
URDU = ["امی یہ میرا نیا نمبر ہے، فوراً 25,000 روپے بھیج دیں",
        "آپ کا اکاؤنٹ بند ہونے والا ہے، فوراً اس لنک پر تصدیق کریں",
        "اپنا پن کوڈ کسی کو نہ بتائیں"]


def test_features_are_computed():
    f = features(pd.DataFrame({"text": ["Send OTP to bit.ly/x now Rs 500", "سلام"],
                               "language": ["English", "Urdu"]}))
    assert f.loc[0, "has_url"] == 1 and f.loc[0, "has_code"] == 1 and f.loc[0, "has_money"] == 1
    assert f.loc[1, "urdu_ratio"] > 0.5


def test_same_kind_of_data_is_not_drift(tmp_path):
    ref = batch(ENGLISH, "English", 400)
    res = check_drift(batch(ENGLISH, "English", 200), ref, CFG["monitoring"]["drift_share"], "same", tmp_path)
    assert not res["drift_detected"]


def test_shifted_data_is_drift(tmp_path):
    ref = batch(ENGLISH, "English", 400)
    res = check_drift(batch(URDU, "Urdu", 200), ref, CFG["monitoring"]["drift_share"], "shifted", tmp_path)
    assert res["drift_detected"]
    assert "urdu_ratio" in res["drifted_features"]
    assert (tmp_path / "shifted.json").exists()
