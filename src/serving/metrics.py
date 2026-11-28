# prometheus metrics for the api
# error rate is not its own number, grafana divides 5xx requests by all requests
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()

REQUESTS = Counter("scam_api_requests_total", "HTTP requests", ["method", "path", "status"], registry=REGISTRY)
LATENCY = Histogram("scam_api_request_latency_seconds", "HTTP request latency", ["path"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5), registry=REGISTRY)
ERRORS = Counter("scam_api_errors_total", "Unhandled errors while serving a request", ["path"], registry=REGISTRY)
PREDICTIONS = Counter("scam_predictions_total", "Predictions by class", ["label"], registry=REGISTRY)
RISK = Counter("scam_predictions_by_risk_total", "Predictions by risk level", ["risk"], registry=REGISTRY)
PROBABILITY = Histogram("scam_probability", "Scam probability given by the model",
                        buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0), registry=REGISTRY)
MODEL_VERSION = Gauge("scam_model_version", "Registry version of the champion being served", registry=REGISTRY)
MODEL_LOADED = Gauge("scam_model_loaded", "1 when a model is loaded and ready", registry=REGISTRY)
MODEL_RELOADS = Counter("scam_model_reloads_total", "Times a new champion was loaded", registry=REGISTRY)
