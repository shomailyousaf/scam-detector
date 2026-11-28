# tests for the web service: risk levels, explanation maths, model loading fallback, the api and metrics
# the model is a tiny pipeline trained here, so no dagshub, no data files and no network are needed
import pytest

from src.serving.explain import explain, risk_level

# ---------- risk level ----------
@pytest.mark.parametrize("p,level", [(0.0, "low"), (0.29, "low"), (0.30, "medium"), (0.69, "medium"),
                                     (0.70, "high"), (1.0, "high")])
def test_risk_level_default_thresholds(p, level):
    assert risk_level(p) == level


def test_risk_level_custom_thresholds():
    assert risk_level(0.5, {"medium": 0.6, "high": 0.9}) == "low"
    assert risk_level(0.95, {"medium": 0.6, "high": 0.9}) == "high"


# ---------- explanation ----------
@pytest.mark.parametrize("text", ["Send your OTP now or account band", "kal meeting hai",
                                  "اپنا OTP بھیجیں", "win money http://x.com"])
def test_explanation_adds_up_to_the_model_score(pipe, text):
    e = explain(pipe, text)
    logit = float(pipe.decision_function([text])[0])
    assert e["logit"] == pytest.approx(logit, abs=1e-3)
    # the words add up to the score too, small slack because each word is rounded
    assert sum(t["score"] for t in e["tokens"]) + e["intercept"] == pytest.approx(logit, abs=1e-2)


def test_explanation_points_scam_words_up_and_normal_words_down(pipe):
    e = explain(pipe, "Apna OTP abhi bhejo warna account band, kal meeting hai")
    assert all(w["score"] > 0 for w in e["scam_words"])
    assert all(w["score"] < 0 for w in e["normal_words"])
    scam_list = [w["word"] for w in e["scam_words"]]
    assert scam_list == sorted(scam_list, key=lambda w: -next(x["score"] for x in e["scam_words"] if x["word"] == w))
    assert "otp" in scam_list


def test_explanation_keeps_every_word_in_order_including_urdu(pipe):
    text = "اپنا OTP بھیجیں jaldi"
    e = explain(pipe, text)
    assert [t["text"] for t in e["tokens"]] == text.split()
    assert any(any("؀" <= ch <= "ۿ" for ch in p["pattern"]) for p in e["scam_patterns"] + e["normal_patterns"])


def test_explanation_handles_text_the_vocabulary_has_never_seen(pipe):
    e = explain(pipe, "zzzzqqqq")
    assert e["scam_words"] == [] and e["scam_patterns"] == []
    assert e["logit"] == pytest.approx(float(pipe.decision_function(["zzzzqqqq"])[0]), abs=1e-3)


# ---------- model loading ----------
class FakeVersion:
    def __init__(self, version, tags=None):
        self.version, self.tags = str(version), tags or {}


class FakeClient:
    def __init__(self, alias_ok, versions):
        self.alias_ok, self.versions = alias_ok, versions

    def get_model_version_by_alias(self, name, alias):
        if not self.alias_ok:
            raise RuntimeError("dagshub cannot read aliases")
        return FakeVersion(7)

    def search_model_versions(self, query):
        return self.versions


def make_store(monkeypatch, alias_ok, versions):
    from src.serving.model_store import ModelStore
    s = ModelStore(name="scam-detector-tfidf", role="champion")
    monkeypatch.setattr(s, "_client", lambda: FakeClient(alias_ok, versions))
    return s


def test_champion_resolved_from_alias(monkeypatch):
    s = make_store(monkeypatch, True, [])
    assert s.resolve_version() == ("7", "alias @champion")


def test_champion_falls_back_to_role_tag_when_alias_fails(monkeypatch):
    versions = [FakeVersion(3, {"role": "previous_champion"}), FakeVersion(5, {"role": "champion"}),
                FakeVersion(6, {"role": "challenger"})]
    s = make_store(monkeypatch, False, versions)
    assert s.resolve_version() == ("5", "tag role=champion")


def test_no_champion_anywhere_is_an_error(monkeypatch):
    s = make_store(monkeypatch, False, [FakeVersion(1, {"role": "challenger"})])
    with pytest.raises(RuntimeError):
        s.resolve_version()


def test_failed_load_keeps_the_old_model_and_records_the_error(monkeypatch, pipe):
    s = make_store(monkeypatch, False, [])
    s.pipeline, s.version = pipe, "4"
    assert s.load() is False
    assert s.ready and s.version == "4" and s.last_error


def test_refresh_only_reloads_when_the_champion_moved(monkeypatch, pipe):
    s = make_store(monkeypatch, True, [])         # registry says champion is v7
    s.pipeline, s.version = pipe, "7"
    monkeypatch.setattr(s, "load", lambda: pytest.fail("should not reload the same version"))
    assert s.refresh_if_changed() is False
    s.version = "6"
    calls = []
    monkeypatch.setattr(s, "load", lambda: calls.append(1) or True)
    assert s.refresh_if_changed() is True and calls == [1]
