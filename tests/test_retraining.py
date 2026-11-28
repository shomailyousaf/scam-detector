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


def test_promotion_rule():
    from src.retraining.promote import decide
    champ = {"test_scam_f1": 0.957, "challenge_v2_scam_f1": 0.60}
    better = {"test_scam_f1": 0.959, "challenge_v2_scam_f1": 0.73}
    worse = {"test_scam_f1": 0.957, "challenge_v2_scam_f1": 0.46}
    broken_test = {"test_scam_f1": 0.90, "challenge_v2_scam_f1": 0.80}
    assert decide(champ, better, CFG["retraining"])[0]
    assert not decide(champ, worse, CFG["retraining"])[0]
    assert not decide(champ, broken_test, CFG["retraining"])[0]


def test_promote_reject_and_rollback(local_mlflow):
    from mlflow import MlflowClient
    from src.retraining.promote import promote
    from src.retraining.rollback import rollback
    cfg = local_mlflow
    name = cfg["retraining"]["model_name"]
    versions = [common.train_log_register(toy(), cfg, n, {"trigger": "test"})[1] for n in "abc"]
    client = MlflowClient()
    for v, f1 in zip(versions, (0.60, 0.73, 0.46)):
        client.set_model_version_tag(name, v, "challenge_v2_scam_f1", str(f1))
        client.set_model_version_tag(name, v, "test_scam_f1", "0.957")
    v1, v2, v3 = versions
    common.set_role(cfg, "champion", v1)

    assert promote(cfg, v3)["decision"] == "rejected"
    assert common.get_role(cfg, "champion") == v1
    assert promote(cfg, v2)["decision"] == "promoted"
    assert common.get_role(cfg, "champion") == v2
    rollback(cfg)
    assert common.get_role(cfg, "champion") == v1


def test_gate_blocks_worse_branch():
    from src.retraining.eval_gate import gate
    champ = {"test_scam_f1": 0.96, "challenge_v2_scam_f1": 0.73}
    assert all(gate({"test_scam_f1": 0.958, "challenge_v2_scam_f1": 0.73}, champ, 0.005).values())
    assert not all(gate({"test_scam_f1": 0.90, "challenge_v2_scam_f1": 0.73}, champ, 0.005).values())
    assert not all(gate({"test_scam_f1": 0.96, "challenge_v2_scam_f1": 0.60}, champ, 0.005).values())
