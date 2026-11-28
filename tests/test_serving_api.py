# tests for the fastapi service, they are skipped when the web packages are not installed
# (CI installs requirements-api.txt so they run there)
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("prometheus_client")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from src.serving.app import create_app  # noqa: E402
from src.serving.model_store import ModelStore  # noqa: E402

@pytest.fixture
def client(pipe, monkeypatch):
    monkeypatch.setenv("MODEL_REFRESH_SECONDS", "0")
    monkeypatch.delenv("RELOAD_TOKEN", raising=False)
    store = ModelStore(name="scam-detector-tfidf")
    store.pipeline, store.version = pipe, "9"
    with TestClient(create_app(store, load_on_start=False)) as c:
        yield c


def test_predict_returns_label_risk_and_explanation(client):
    r = client.post("/predict", json={"text": "Apna OTP abhi bhejo warna account band ho jayega"})
    assert r.status_code == 200
    d = r.json()
    assert d["label"] in ("SCAM", "NORMAL") and d["risk_level"] in ("low", "medium", "high")
    assert 0 <= d["scam_probability"] <= 1
    assert d["model"] == {"name": "scam-detector-tfidf", "version": "9"}
    assert d["explanation"]["tokens"]
    assert (d["label"] == "SCAM") == (d["scam_probability"] >= 0.5)


def test_predict_without_explanation(client):
    d = client.post("/predict", json={"text": "kal class hai", "explain": False}).json()
    assert "explanation" not in d


def test_predict_works_for_urdu_script(client):
    r = client.post("/predict", json={"text": "اپنا او ٹی پی بھیجیں"})
    assert r.status_code == 200 and r.json()["label"] == "SCAM"


@pytest.mark.parametrize("body", [{}, {"text": ""}, {"text": "x" * 5001}])
def test_predict_rejects_bad_input(client, body):
    assert client.post("/predict", json=body).status_code == 422


def test_predict_returns_503_when_no_model_is_loaded():
    store = ModelStore(name="scam-detector-tfidf")
    with TestClient(create_app(store, load_on_start=False)) as c:
        assert c.post("/predict", json={"text": "hello"}).status_code == 503
        h = c.get("/health").json()
        assert h["status"] == "no_model" and h["model_loaded"] is False


def test_health_and_index_page(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_version"] == "9"
    page = client.get("/")
    assert page.status_code == 200 and "Scam Message Detector" in page.text


def test_reload_needs_the_token_when_one_is_set(client, monkeypatch):
    monkeypatch.setenv("RELOAD_TOKEN", "s3cret")
    assert client.post("/reload").status_code == 403
    assert client.post("/reload", headers={"X-Reload-Token": "wrong"}).status_code == 403


def test_metrics_count_requests_and_predictions_per_class(client):
    from src.serving import metrics as m

    def total(counter, **labels):
        return counter.labels(**labels)._value.get()

    before = {l: total(m.PREDICTIONS, label=l) for l in ("SCAM", "NORMAL")}
    r = client.post("/predict", json={"text": "Apna OTP abhi bhejo warna account band ho jayega"})
    label = r.json()["label"]
    assert total(m.PREDICTIONS, label=label) == before[label] + 1

    text = client.get("/metrics").text
    for name in ("scam_api_requests_total", "scam_api_request_latency_seconds_bucket", "scam_predictions_total",
                 "scam_predictions_by_risk_total", "scam_model_version", "scam_api_errors_total"):
        assert name in text
    assert 'path="/predict"' in text


def test_server_errors_are_counted(pipe, monkeypatch):
    from src.serving import metrics as m
    monkeypatch.setenv("MODEL_REFRESH_SECONDS", "0")

    class Broken:
        named_steps = pipe.named_steps
        def predict_proba(self, X):
            raise RuntimeError("boom")

    store = ModelStore(name="scam-detector-tfidf")
    store.pipeline, store.version = Broken(), "1"
    with TestClient(create_app(store, load_on_start=False), raise_server_exceptions=False) as c:
        before = m.REQUESTS.labels("POST", "/predict", "500")._value.get()
        assert c.post("/predict", json={"text": "hello there"}).status_code == 500
        assert m.REQUESTS.labels("POST", "/predict", "500")._value.get() == before + 1
