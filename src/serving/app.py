# FastAPI service: /predict, /health, /metrics and the web page at /
#   uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.common.textnorm import clean_text
from src.serving import metrics as m
from src.serving.explain import DEFAULT_THRESHOLDS, explain, risk_level
from src.serving.model_store import ModelStore

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("serving")

STATIC = Path(__file__).parent / "static"
MAX_CHARS = 5000


def load_thresholds():
    # defaults in params.yaml (serving:), can be overridden per deployment with env vars
    t = dict(DEFAULT_THRESHOLDS)
    try:
        import yaml
        cfg = yaml.safe_load(open("params.yaml", encoding="utf-8")) or {}
        t.update(cfg.get("serving", {}).get("risk_thresholds", {}))
    except Exception:
        pass
    if os.environ.get("RISK_MEDIUM"):
        t["medium"] = float(os.environ["RISK_MEDIUM"])
    if os.environ.get("RISK_HIGH"):
        t["high"] = float(os.environ["RISK_HIGH"])
    return t


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_CHARS)
    explain: bool = True


def update_model_gauges(store):
    m.MODEL_LOADED.set(1 if store.ready else 0)
    try:
        m.MODEL_VERSION.set(float(store.version))
    except (TypeError, ValueError):
        m.MODEL_VERSION.set(0)


async def refresh_loop(store, seconds):
    # picks up a model the retraining loop promoted without restarting the service
    while True:
        await asyncio.sleep(seconds)
        before = store.version
        changed = await asyncio.to_thread(store.refresh_if_changed)
        if changed and store.version != before:
            m.MODEL_RELOADS.inc()
        update_model_gauges(store)


def create_app(store=None, load_on_start=True):
    store = store or ModelStore()
    thresholds = load_thresholds()
    refresh_seconds = int(os.environ.get("MODEL_REFRESH_SECONDS", "300"))

    @asynccontextmanager
    async def lifespan(app):
        task = None
        if load_on_start:
            await asyncio.to_thread(store.load)
            update_model_gauges(store)
        if refresh_seconds > 0:
            task = asyncio.create_task(refresh_loop(store, refresh_seconds))
        yield
        if task:
            task.cancel()

    app = FastAPI(title="Scam message detector", version="1.0", lifespan=lifespan)
    app.state.store = store

    @app.middleware("http")
    async def track(request: Request, call_next):
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        except Exception:
            m.ERRORS.labels(path=request.url.path).inc()
            raise
        finally:
            route = request.scope.get("route")
            path = route.path if route else "unmatched"   # route template keeps the labels few
            if path != "/metrics":
                m.REQUESTS.labels(request.method, path, str(status)).inc()
                m.LATENCY.labels(path).observe(time.perf_counter() - start)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok" if store.ready else "no_model", "model_loaded": store.ready,
                "model_name": store.name, "model_version": store.version, "found_by": store.source,
                "last_error": store.last_error}

    @app.post("/predict")
    def predict(req: PredictRequest):
        pipe, version = store.get()
        if pipe is None:
            raise HTTPException(503, f"model not loaded: {store.last_error or 'still starting'}")
        text = clean_text(req.text)
        if not text:
            raise HTTPException(422, "message is empty")
        prob = float(pipe.predict_proba([text])[0, 1])
        label = "SCAM" if prob >= 0.5 else "NORMAL"
        risk = risk_level(prob, thresholds)
        m.PREDICTIONS.labels(label).inc()
        m.RISK.labels(risk).inc()
        m.PROBABILITY.observe(prob)
        out = {"label": label, "scam_probability": round(prob, 4), "risk_level": risk,
               "model": {"name": store.name, "version": version}}
        if req.explain:
            out["explanation"] = explain(pipe, text)
        return out

    @app.post("/reload")
    def reload_model(request: Request):
        # lets the retraining workflow (or a person) pull the new champion straight away
        # if RELOAD_TOKEN is set the caller has to send it as the X-Reload-Token header
        token = os.environ.get("RELOAD_TOKEN")
        if token and request.headers.get("x-reload-token") != token:
            raise HTTPException(403, "bad reload token")
        before = store.version
        ok = store.refresh_if_changed() if store.ready else store.load()
        if ok and store.version != before:
            m.MODEL_RELOADS.inc()
        update_model_gauges(store)
        return {"reloaded": bool(ok), "model_version": store.version, "last_error": store.last_error}

    @app.get("/metrics", include_in_schema=False)
    def prometheus():
        return Response(generate_latest(m.REGISTRY), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
