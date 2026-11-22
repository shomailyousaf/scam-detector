


import json
import re
import subprocess
import threading
import itertools
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.models.mlflow_utils import setup_mlflow

SEED = 42
DATA = Path("data/processed")
REPORTS = Path("reports")




def run_with_progress(label, function, *args, **kwargs):
    stop_event = threading.Event()
    start_time = time.perf_counter()

    def animate():
        spinner = itertools.cycle(["|", "/", "-", "\\"])
        while not stop_event.is_set():
            elapsed = time.perf_counter() - start_time
            print(
                f"\r{label} {next(spinner)} "
                f"{elapsed:.0f}s elapsed",
                end="",
                flush=True,
            )
            stop_event.wait(0.5)

    thread = threading.Thread(target=animate, daemon=True)
    print(f"\n[START] {label}", flush=True)
    thread.start()

    try:
        result = function(*args, **kwargs)
    except Exception:
        stop_event.set()
        thread.join()
        print(f"\n[FAILED] {label}", flush=True)
        raise

    stop_event.set()
    thread.join()

    elapsed = time.perf_counter() - start_time
    print(
        f"\r[DONE] {label} completed in {elapsed:.1f} seconds",
        flush=True,
    )
    return result


def checkpoint(message):
    print(f"\n{'=' * 55}\n{message}\n{'=' * 55}", flush=True)




KEYWORDS = [
    "winner", "won", "prize", "lottery", "claim", "free", "urgent", "verify",
    "suspended", "blocked", "otp", "pin", "cnic", "password", "click",
    "congratulations", "reward", "gift", "limited time", "act now",
    "bank account", "send money", "easypaisa", "jazzcash", "loan",
    "investment", "bitcoin",
    "inaam", "inam", "jeeta", "kamyab", "foran", "abhi", "register",
    "kamayein", "ghar baithe", "rupay", "paisay", "account band",
    "tasdeeq", "shanakhti card", "mubarak", "bhejein", "bhej dein",
    "انعام", "مبارک", "فوری", "اکاؤنٹ", "پن", "شناختی کارڈ",
    "رجسٹر", "روپے", "لاٹری", "کلک", "پاس ورڈ", "بلاک",
]

URL_RE = re.compile(
    r"(https?://|www\.|bit\.ly|tinyurl|t\.co/|\.xyz|\.top|\.click|\.info|\.cc\b)",
    re.IGNORECASE,
)


def rule_score(text: str) -> float:
    t = str(text).lower()
    hits = sum(1 for k in KEYWORDS if k in t)
    url = 1 if URL_RE.search(t) else 0
    return float(hits + 2 * url)




def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def dataset_versions() -> dict:
    out = {}
    try:
        params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
        out["data_version"] = params["data"]["version"]
    except Exception as e:
        out["data_version_error"] = str(e)
    try:
        lock = yaml.safe_load(
            Path("dvc.lock").read_text(encoding="utf-8")
        )
        for o in lock["stages"]["split"]["outs"]:
            out["data_md5_" + Path(o["path"]).name] = o["md5"]
    except Exception as e:
        out["data_md5_error"] = str(e)
    return out


def metrics_for(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "scam_precision": precision_score(y, pred, zero_division=0),
        "scam_recall": recall_score(y, pred, zero_division=0),
        "scam_f1": f1_score(y, pred, zero_division=0),
        "pr_auc": average_precision_score(y, score),
        "false_alarm_rate": fp / (fp + tn) if (fp + tn) else 0.0,
    }


def f1_by_language(df, pred):
    res = {}
    for lang, g in df.assign(pred=pred).groupby("language"):
        res[lang] = f1_score(
            g["label_id"], g["pred"], zero_division=0
        )
    return res


def save_confusion(y, pred, title, path):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["NORMAL", "SCAM"])
    ax.set_yticklabels(["NORMAL", "SCAM"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def log_run(
    run_name, params, val_m, test_m, test_lang,
    test_y, test_pred, model=None, challenge_m=None
):
    checkpoint(f"LOGGING {run_name} TO DAGSHUB")

    versions = dataset_versions()

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**params, "seed": SEED, **versions})
        mlflow.set_tag("git_commit", git_commit())

        for k, v in val_m.items():
            mlflow.log_metric(f"val_{k}", v)

        for k, v in test_m.items():
            mlflow.log_metric(f"test_{k}", v)

        for lang, v in test_lang.items():
            mlflow.log_metric(f"test_f1_{lang}", v)

        for k, v in (challenge_m or {}).items():
            mlflow.log_metric(f"challenge_v2_{k}", v)

        png = REPORTS / f"confusion_{run_name}.png"

        run_with_progress(
            "Creating confusion matrix",
            save_confusion,
            test_y, test_pred, f"{run_name} (test)", png,
        )

        run_with_progress(
            "Uploading confusion matrix to DagsHub",
            mlflow.log_artifact,
            str(png),
        )

        if model is not None:
            run_with_progress(
                "Saving TF-IDF model to MLflow using CloudPickle",
                mlflow.sklearn.log_model,
                model,
                name="model",
                serialization_format="cloudpickle",
            )

    report = {
        "model": run_name,
        "params": params,
        "val": {k: round(v, 6) for k, v in val_m.items()},
        "test": {k: round(v, 6) for k, v in test_m.items()},
        "test_f1_by_language": {
            k: round(v, 6) for k, v in sorted(test_lang.items())
        },
    }
    if challenge_m:
        # same fixed messages in every data version -> fair v1 vs v2 comparison
        report["challenge_v2"] = {k: round(v, 6) for k, v in challenge_m.items()}

    (REPORTS / f"metrics_{run_name}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        f"\n[{run_name}] test F1={test_m['scam_f1']:.4f} "
        f"precision={test_m['scam_precision']:.4f} "
        f"recall={test_m['scam_recall']:.4f} "
        f"PR-AUC={test_m['pr_auc']:.4f} "
        f"false_alarm={test_m['false_alarm_rate']:.4f}",
        flush=True,
    )




def main():
    REPORTS.mkdir(exist_ok=True)

    checkpoint("INITIALIZING MLFLOW")
    setup_mlflow()

    checkpoint("LOADING DATASETS")

    train = run_with_progress(
        "Loading training dataset",
        pd.read_csv,
        DATA / "train.csv",
    ).fillna({"text": ""})

    val = run_with_progress(
        "Loading validation dataset",
        pd.read_csv,
        DATA / "val.csv",
    ).fillna({"text": ""})

    test = run_with_progress(
        "Loading test dataset",
        pd.read_csv,
        DATA / "test.csv",
    ).fillna({"text": ""})

    challenge = pd.read_csv(DATA / "challenge_v2.csv").fillna({"text": ""})

    print(
        f"\nChallenge rows (v2, never trained on): {len(challenge)}"
        f"\nTraining rows: {len(train)}"
        f"\nValidation rows: {len(val)}"
        f"\nTest rows: {len(test)}",
        flush=True,
    )

   

    checkpoint("BASELINE 1: RULES-ONLY MODEL")

    val_scores = run_with_progress(
        "Scoring validation messages",
        lambda: val["text"].map(rule_score).to_numpy(),
    )

    test_scores = run_with_progress(
        "Scoring test messages",
        lambda: test["text"].map(rule_score).to_numpy(),
    )

    best_t, best_f1 = 1, -1

    for t in [1, 2, 3, 4]:
        f = f1_score(
            val["label_id"],
            (val_scores >= t).astype(int),
            zero_division=0,
        )
        if f > best_f1:
            best_t, best_f1 = t, f

    val_pred = (val_scores >= best_t).astype(int)
    test_pred = (test_scores >= best_t).astype(int)
    ch_scores = challenge["text"].map(rule_score).to_numpy()
    ch_pred = (ch_scores >= best_t).astype(int)

    log_run(
        "rules_only",
        {
            "model": "rules_only",
            "min_score": best_t,
            "n_keywords": len(KEYWORDS),
        },
        metrics_for(val["label_id"], val_pred, val_scores),
        metrics_for(test["label_id"], test_pred, test_scores),
        f1_by_language(test, test_pred),
        test["label_id"],
        test_pred,
        challenge_m=metrics_for(challenge["label_id"], ch_pred, ch_scores),
    )

   

    checkpoint("BASELINE 2: TF-IDF + LOGISTIC REGRESSION")

    ngram = (2, 5)
    C = 4.0

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram,
        sublinear_tf=True,
        min_df=2,
        max_features=300000,
    )

    classifier = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
        random_state=SEED,
    )

    checkpoint("BUILDING TF-IDF FEATURES")

    X_train = run_with_progress(
        "Building training TF-IDF features",
        vectorizer.fit_transform,
        train["text"],
    )

    X_val = run_with_progress(
        "Building validation TF-IDF features",
        vectorizer.transform,
        val["text"],
    )

    X_test = run_with_progress(
        "Building test TF-IDF features",
        vectorizer.transform,
        test["text"],
    )

    print(
        f"\nTF-IDF feature count: {X_train.shape[1]}",
        flush=True,
    )

    checkpoint("TRAINING LOGISTIC REGRESSION")

    run_with_progress(
        "Fitting Logistic Regression classifier",
        classifier.fit,
        X_train,
        train["label_id"],
    )

    pipe = Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])

    checkpoint("EVALUATING TF-IDF MODEL")

    val_prob = run_with_progress(
        "Predicting validation probabilities",
        lambda: classifier.predict_proba(X_val)[:, 1],
    )

    test_prob = run_with_progress(
        "Predicting test probabilities",
        lambda: classifier.predict_proba(X_test)[:, 1],
    )

    val_pred = (val_prob >= 0.5).astype(int)
    test_pred = (test_prob >= 0.5).astype(int)
    ch_prob = classifier.predict_proba(vectorizer.transform(challenge["text"]))[:, 1]
    ch_pred = (ch_prob >= 0.5).astype(int)

    log_run(
        "tfidf_logreg",
        {
            "model": "char_tfidf+logreg",
            "analyzer": "char_wb",
            "ngram_range": f"{ngram[0]}-{ngram[1]}",
            "C": C,
            "class_weight": "balanced",
            "max_features": 300000,
        },
        metrics_for(val["label_id"], val_pred, val_prob),
        metrics_for(test["label_id"], test_pred, test_prob),
        f1_by_language(test, test_pred),
        test["label_id"],
        test_pred,
        model=pipe,
        challenge_m=metrics_for(challenge["label_id"], ch_pred, ch_prob),
    )

    checkpoint("ALL BASELINES COMPLETED SUCCESSFULLY")

    print("\nReports saved in:", REPORTS.resolve(), flush=True)


if __name__ == "__main__":
    main()
