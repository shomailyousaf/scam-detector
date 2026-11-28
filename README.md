# Scam Message Detection (English + Roman Urdu) - MLOps Project

This project detects scam/fraud messages written in English, Roman Urdu, Urdu script, and mixed English-Urdu. This repo is Deliverable 1: a data and training pipeline that is versioned, validated, leak-free, tracked in MLflow, and reproducible with one command.

## Team

| Member | Work |
|---|---|
| Member A | Data versioning with DVC, schema and quality validation, cleaning, leakage-free split and leakage checks, data tests, one-command reproduction (`reproduce.py`) and Docker, data v2 and the v2 experiment |
| Member B | MLflow tracking on DagsHub, rule-based and TF-IDF baselines, XLM-RoBERTa fine-tuning, model registry, challenge-set scoring of XLM-R |

## How to run it

```
git clone https://github.com/shomailyousaf/scam-detector.git
cd scam-detector
python -m venv .venv
.venv\Scripts\activate          (on Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
dvc pull
python reproduce.py
```

Or with Docker:

```
docker build -t scam-detector .
docker run --rm -v "${PWD}:/app" scam-detector
```

`reproduce.py` downloads the raw data from DVC, reruns the whole pipeline from scratch, runs the 22 tests, and then checks that the MD5 hash of every input and output matches `dvc.lock`. If everything matches it prints `REPRODUCED`. On our machines it gives `compared 27 files, 0 different`.

MLflow logs go to DagsHub when `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` are set, and to a local `mlflow.db` file otherwise, so the pipeline also runs without an account.

XLM-RoBERTa needs a GPU (we used Kaggle/Colab), so it is not part of `reproduce.py`:

```
pip install -r requirements-gpu.txt
dvc pull
python -m src.models.train_xlmr --epochs 2 --run_name xlmr_base
```

Experiment tracking: https://dagshub.com/shomailusuf/scam-detector.mlflow

One thing we ran into: the first Docker run said the outputs didn't match. The data was the same, but files written on Windows use `\r\n` line endings and on Linux `\n`, so the hashes were different. We fixed it by always writing `\n` and adding a `.gitattributes` file. Docker on Windows also marks files as executable, so the check compares content hashes only.

## The pipeline

The stages are in `dvc.yaml` and run with `dvc repro`:

```
raw data -> validate_raw -> prepare -> split -> check_splits -> train_baselines
```

1. **validate_raw** - checks both raw files with Pandera: required columns, no empty messages, label only SCAM or NORMAL, one of the four languages, unique ids, enough rows, scam share between 20% and 80%. If anything fails the pipeline stops.
2. **prepare** - removes extra spaces, drops texts longer than 2000 characters (long emails, not messages), drops near-duplicates with opposite labels, and groups near-duplicate messages together (explained below).
3. **split** - 70% train, 15% validation, 15% test. A whole group always goes into one split, and every split keeps the same mix of scam/normal and languages. Seed 42.
4. **check_splits** - 7 automatic checks: schema, no shared groups, no shared near-duplicates, no shared ids, all languages in every split, same scam share, and the challenge set never in train or validation. Training cannot start if any check fails.
5. **train_baselines** - trains the rules-only and TF-IDF baselines and logs them to MLflow.

All settings are in `params.yaml`.

## The data

We merged two public datasets and our own synthetic messages into one raw file, then built this pipeline on it.

| Source | Type | Rows | Share |
|---|---|---|---|
| `mshenoda/spam-messages` (Hugging Face) | Real, English | 54,391 | 88.2% |
| Pak-Spam-Guard (GitHub) | Real, Pakistani SMS (English / Roman Urdu / Urdu) | 5,971 | 9.7% |
| Synthetic business-chat messages | Synthetic | 1,272 | 2.1% |
| Hand-written Roman-Urdu scams | Hand-written | 24 | 0.04% |
| **Data v2 additions** | Synthetic | 334 | added in v2 |

About 98% of the v1 data is real. The synthetic messages were added because the real Roman-Urdu data had 1,964 scam messages but only 37 legitimate ones. The synthetic set includes 199 hard negatives: normal messages that contain scam-like words, such as "never share your OTP".

| English | Roman Urdu | Urdu | Mixed |
|---|---|---|---|
| 56,996 | 3,587 | 846 | 229 |

## Leakage problem we found

Many public messages are templates where only a number or link changes, like "Aap ne Rs 5000 jeete..." and "Aap ne Rs 7500 jeete...". An earlier split only kept translations of the same message together, so these near-copies ended up in both train and test.

In that split, 1,060 of the 9,242 test messages (11.5%) had a near-copy in training. For Roman Urdu it was 295 of 545 (54%), so test scores were too high.

To fix it, every message gets a simple key (lowercase, numbers replaced with 0, links replaced with `<url>`, punctuation removed). Messages with the same key, plus translations of the same message, form one group, and a group goes to only one split.

After prepare: 61,658 rows in, 56,062 out. 5,582 removed for being too long, 14 removed as duplicates with opposite labels, 12,304 near-duplicate rows grouped, 46,370 groups.

| Split | Rows | Scam % | English | Roman Urdu | Urdu | Mixed |
|---|---|---|---|---|---|---|
| train | 39,073 | 41.7% | 36,359 | 1,917 | 638 | 159 |
| val | 8,505 | 42.6% | 7,793 | 573 | 104 | 35 |
| test | 8,484 | 42.5% | 7,792 | 553 | 104 | 35 |

There are 0 shared groups, near-duplicates or ids between any two splits.

## Model results

All models were trained on `train.csv`, selected on `val.csv`, and reported on `test.csv`. Seed 42.

| Model | Test F1 | Precision | Recall | PR-AUC | False-alarm rate |
|---|---|---|---|---|---|
| Rules only | 0.6230 | 0.7140 | 0.5526 | 0.6763 | 0.1636 |
| Char TF-IDF + Logistic Regression | 0.9574 | 0.9632 | 0.9517 | 0.9931 | 0.0268 |
| **XLM-RoBERTa base (registered)** | **0.9799** | **0.9865** | **0.9734** | **0.9980** | **0.0098** |

XLM-RoBERTa is registered in the MLflow Model Registry as `scam-detector-xlmr`, version 1.

Test F1 by language (XLM-RoBERTa): English 0.980, Roman-Urdu 0.993, Code-Mixed 0.941, Urdu 0.895.

Limitations:
- Urdu is the weakest language (F1 0.895). It has only 638 training rows, against 36,359 for English.
- Validation scores (F1 0.929) are lower than test scores (0.980). The splits hold different messages with a different language mix.

## Data v2 and the challenge set

v1 has very few everyday legitimate messages in Roman Urdu and almost no WhatsApp-takeover scams. For v2 we generated 334 synthetic messages with a seeded script (`src/data/make_v2_additions.py`): bank alerts (HBL, Meezan, UBL, MCB, Allied, Alfalah), JazzCash/Easypaisa, parcel delivery (TCS, Leopards, M&P, Trax, PostEx), WhatsApp takeover, "new number" family scams, job fraud, and everyday chats. Each topic has scam, legitimate, and hard-negative messages.

The v2 messages are split separately from v1, so the v1 split stays exactly the same. Their test part (55 messages) is a fixed **challenge set** that no model is trained on, so models can be compared fairly.

v1 is the default. v2 is run as a DVC experiment:

```
dvc exp run -n data-v2 -S data.version=v2
dvc metrics diff main data-v2 --targets reports/metrics_tfidf_logreg.json
```

Results on the 55-message challenge set:

| Model | Trained on | Scam F1 | Recall | False alarms |
|---|---|---|---|---|
| TF-IDF + LogReg | v1 | 0.607 | 68% | 47% |
| TF-IDF + LogReg | v2 | 0.733 | 88% | 43% |
| XLM-RoBERTa (registered) | v1 | 0.745 | 76% | 23% |

On the same v1 training data, XLM-R scores F1 0.745 with about 23% false alarms, against TF-IDF's F1 0.607 and about 47% false alarms. Adding v2 data helps TF-IDF catch more scams, but its false alarms stay high.

What this shows: models that score 0.96-0.98 on the benchmark test set still struggle on everyday messages. XLM-R missed Urdu-script "new number" scams and flagged real bank debit alerts and genuine delivery OTP messages (`reports/challenge_xlmr_errors.csv`). The challenge set is small and synthetic, so these numbers are indicative.

## Tests

`pytest -q` runs 22 tests. They check that good data passes and 8 kinds of broken data are rejected, that the near-duplicate key groups templates but keeps different messages apart, that the split has no leakage and is the same every run, that the leakage check catches overlap when we add some on purpose, and that the v2 generator is reproducible and never overlaps v1.

## Folder structure

```
params.yaml              settings (including data.version)
dvc.yaml / dvc.lock      pipeline stages and hashes of every input and output
data/raw/                raw data v1 and v2 additions (tracked by DVC, stored on DagsHub)
src/common/              shared helpers (text cleaning, reading params)
src/data/                validation, cleaning, split, checks, v2 generator
src/models/              baselines, XLM-R training, registry, challenge scoring, MLflow setup
reports/                 metrics and check results written by the pipeline
tests/                   tests
reproduce.py             runs everything with one command
Dockerfile               same thing inside a container
```

## Web service (API, UI, monitoring)

A FastAPI service that always serves the current **champion** from the MLflow registry on DagsHub, so whatever the retraining loop promotes is what gets served (the service checks every `MODEL_REFRESH_SECONDS`, default 300, and `POST /reload` forces it).

| Endpoint | What it does |
|---|---|
| `GET /` | web page: paste a message, see label, risk level and which words pushed it towards scam |
| `POST /predict` | `{"text": "...", "explain": true}` -> label, `scam_probability`, `risk_level`, explanation, model version |
| `GET /health` | model loaded or not, version, how the champion was found |
| `GET /metrics` | Prometheus metrics |
| `POST /reload` | load the champion again now (needs header `X-Reload-Token` if `RELOAD_TOKEN` is set) |

Model loading: `models:/scam-detector-tfidf@champion`. If DagsHub does not resolve the alias, the version tagged `role=champion` is used.

Risk level from the scam probability (thresholds in `params.yaml` under `serving`): below 0.30 low, 0.30 to 0.70 medium, 0.70 and above high. Label is SCAM at 0.5 and above.

Explanation: the model is char TF-IDF x logistic regression, so score = intercept + sum(tfidf x coefficient). Each character n-gram sits inside one word, so we add them up per word. The word scores plus the intercept equal the model's logit exactly. Positive pushes towards scam, negative towards normal. The top character patterns are shown too.

Run it locally (credentials only as environment variables, see `.env.example`):

```
cp .env.example .env            # fill in your DagsHub username and token
docker compose up --build       # app :8000, Prometheus :9090, Grafana :3000
```

Grafana opens with the "Scam detector API" dashboard already loaded: request rate, error rate, p50/p95/p99 latency, predictions per class and per risk level, mean probability, and the champion version being served.

Deploy: `render.yaml` is a Render blueprint (Docker, free plan). Set `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` in the Render dashboard, never in git.
