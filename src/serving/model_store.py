# loads the production model from the mlflow registry on dagshub
#
# Always the champion, so whatever the retraining loop promotes is what gets served.
# Order: the alias  models:/<name>@champion  first, and if the registry cannot resolve it
# (dagshub cant always read aliases back) the version tagged role=champion.
# Credentials only come from the environment, nothing is written in the code:
#   MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD   dagshub user and token
#   MLFLOW_TRACKING_URI                                   optional, defaults to our dagshub repo
#   MODEL_NAME                                            optional, defaults to scam-detector-tfidf
#   MODEL_ROLE                                            optional, defaults to champion
#   LOCAL_MODEL_PATH                                      optional joblib file for offline demos and tests
import logging
import os
import threading
import time

log = logging.getLogger("serving")

DAGSHUB_URI = "https://dagshub.com/shomailusuf/scam-detector.mlflow"


class ModelStore:
    def __init__(self, name=None, role=None):
        self.name = name or os.environ.get("MODEL_NAME", "scam-detector-tfidf")
        self.role = role or os.environ.get("MODEL_ROLE", "champion")
        self.pipeline = None
        self.version = None
        self.source = None
        self.loaded_at = None
        self.last_error = None
        self._lock = threading.Lock()

    # ----- registry helpers -----
    def _client(self):
        import mlflow
        from mlflow import MlflowClient
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DAGSHUB_URI))
        return MlflowClient()

    def resolve_version(self):
        # returns (version, how it was found)
        client = self._client()
        try:
            mv = client.get_model_version_by_alias(self.name, self.role)
            return str(mv.version), f"alias @{self.role}"
        except Exception as e:
            log.warning("alias @%s did not resolve (%s), trying the version tag role=%s", self.role, e, self.role)
        for mv in client.search_model_versions(f"name='{self.name}'"):
            if mv.tags.get("role") == self.role:
                return str(mv.version), f"tag role={self.role}"
        raise RuntimeError(f"no version of {self.name} has alias or tag '{self.role}'")

    # ----- loading -----
    def load(self):
        # loads the current champion and swaps it in, the old model keeps serving if this fails
        try:
            path = os.environ.get("LOCAL_MODEL_PATH")
            if path:
                import joblib
                pipe, version, source = joblib.load(path), "local", f"file {path}"
            else:
                import mlflow.sklearn
                version, source = self.resolve_version()
                pipe = mlflow.sklearn.load_model(f"models:/{self.name}/{version}")
            with self._lock:
                self.pipeline, self.version, self.source = pipe, version, source
                self.loaded_at, self.last_error = time.time(), None
            log.info("serving %s v%s (found by %s)", self.name, version, source)
            return True
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            log.error("model load failed: %s", self.last_error)
            return False

    def refresh_if_changed(self):
        # cheap check, only downloads again when the retraining loop moved the champion
        if os.environ.get("LOCAL_MODEL_PATH"):
            return False
        try:
            version, _ = self.resolve_version()
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return False
        if version != self.version:
            log.info("champion changed from v%s to v%s, reloading", self.version, version)
            return self.load()
        return False

    def get(self):
        with self._lock:
            return self.pipeline, self.version

    @property
    def ready(self):
        return self.pipeline is not None
