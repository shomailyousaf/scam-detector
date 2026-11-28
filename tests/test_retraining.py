# tests for the retraining helpers using a throwaway local mlflow
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

import src.retraining.common as common
from src.common.io import load_params

ROOT = Path(__file__).resolve().parents[1]
CFG = load_params(ROOT / "params.yaml")
pytestmark = pytest.mark.skipif(not (ROOT / CFG["data"]["processed_dir"] / "challenge_v2.csv").exists(),
                                reason="run dvc repro first")


@pytest.fixture
def local_mlflow(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"

    def setup():
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("test")
    monkeypatch.setattr(common, "setup_mlflow", setup)
    setup()
    cfg = {**CFG, "retraining": {**CFG["retraining"], "model_name": "test-model", "max_features": 2000}}
    return cfg


def toy(n=200):
    rows = [("send your otp now or account blocked", 1), ("see you at dinner tonight", 0)]
    return pd.DataFrame([{"text": f"{t} {i}", "label_id": y} for i in range(n) for t, y in rows])


def test_scores_are_correct():
    s = common.scores(pd.Series([1, 0, 1, 0]), np.array([0.9, 0.2, 0.4, 0.7]))
    assert s["scam_recall"] == 0.5 and s["false_alarm_rate"] == 0.5


def test_register_and_move_roles(local_mlflow):
    cfg = local_mlflow
    _, v1, m1, _ = common.train_log_register(toy(), cfg, "a", {"trigger": "test"})
    _, v2, _, _ = common.train_log_register(toy(), cfg, "b", {"trigger": "test"})
    common.set_role(cfg, "champion", v1)
    common.set_role(cfg, "challenger", v2)
    assert common.get_role(cfg, "champion") == v1
    assert common.get_role(cfg, "challenger") == v2
    common.set_role(cfg, "champion", v2)
    assert common.get_role(cfg, "champion") == v2
    assert "test_scam_f1" in common.version_metrics(cfg, v1)
    model, v = common.load_role(cfg, "champion")
    assert v == v2 and model.predict_proba(["send otp now"]).shape == (1, 2)


def test_roles_work_when_server_has_no_aliases(local_mlflow, monkeypatch):
    # dagshub style server where aliases cant be read back
    cfg = local_mlflow
    _, v1, _, _ = common.train_log_register(toy(), cfg, "a", {"trigger": "bootstrap"})

    def no_alias(*a, **k):
        raise Exception("aliases not supported")
    monkeypatch.setattr(common.MlflowClient, "get_model_version_by_alias", no_alias)
    assert common.get_role(cfg, "champion") is None
    assert common.find_version(cfg, "trigger", "bootstrap") == v1
    common.set_role(cfg, "champion", v1)
    assert common.get_role(cfg, "champion") == v1
