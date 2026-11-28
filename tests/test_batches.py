import os
from pathlib import Path

import pandas as pd
import pytest

from src.common.io import load_params
from src.monitoring.make_batches import make_batches

ROOT = Path(__file__).resolve().parents[1]
CFG = load_params(ROOT / "params.yaml")
PROCESSED = ROOT / CFG["data"]["processed_dir"]

pytestmark = pytest.mark.skipif(not (PROCESSED / "challenge_v2.csv").exists(),
                                reason="run dvc repro first")


@pytest.fixture(scope="module")
def batches():
    os.chdir(ROOT)
    return make_batches(CFG)


def test_batches_never_touch_test_or_challenge(batches):
    test = pd.read_csv(PROCESSED / "test.csv", keep_default_na=False)
    challenge = pd.read_csv(PROCESSED / "challenge_v2.csv", keep_default_na=False)
    for df in batches.values():
        assert not set(df["split_group"]) & set(test["split_group"])
        assert not set(df["split_group"]) & set(challenge["split_group"])


def test_bad_batch_has_every_label_flipped(batches):
    good, bad = batches["batch_02_everyday"], batches["batch_03_bad_labels"]
    assert (good["text"].values == bad["text"].values).all()
    assert (good["label_id"].values != bad["label_id"].values).all()


def test_batches_are_reproducible(batches):
    again = make_batches(CFG)
    for name, df in batches.items():
        assert df.equals(again[name])