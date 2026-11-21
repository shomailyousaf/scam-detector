"""Fine-tune XLM-RoBERTa on train.csv, select epoch on val.csv, report on test.csv.

Run on a GPU (Kaggle/Colab). Logs to MLflow (DagsHub, or local ./mlruns fallback).
Example:  python -m src.models.train_xlmr --epochs 2
Smoke test: python -m src.models.train_xlmr --subset 300 --epochs 1
"""
import argparse
import json
import random
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from src.models.mlflow_utils import setup_mlflow
from src.models.train_baselines import (DATA, REPORTS, dataset_versions,
                                        f1_by_language, git_commit,
                                        metrics_for, save_confusion)

SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(df, tok, max_len, batch, shuffle):
    texts = df["text"].tolist()
    labels = df["label_id"].tolist()
    items = list(zip(texts, labels))

    def collate(b):
        enc = tok([x[0] for x in b], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([x[1] for x in b])
        return enc

    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(items, batch_size=batch, shuffle=shuffle,
                      collate_fn=collate, generator=g)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**batch).logits
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="xlm-roberta-base")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--subset", type=int, default=0, help="rows per split, for smoke tests")
    ap.add_argument("--run_name", default="xlmr_base")
    args = ap.parse_args()

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    print(f"[xlmr] device={device} model={args.model_name}")

    train = pd.read_csv(DATA / "train.csv").fillna({"text": ""})
    val = pd.read_csv(DATA / "val.csv").fillna({"text": ""})
    test = pd.read_csv(DATA / "test.csv").fillna({"text": ""})
    if args.subset:
        train = train.sample(args.subset, random_state=SEED)
        val = val.sample(args.subset, random_state=SEED)
        test = test.sample(args.subset, random_state=SEED)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2).to(device)

    train_dl = make_loader(train, tok, args.max_len, args.batch, True)
    val_dl = make_loader(val, tok, args.max_len, args.batch * 2, False)
    test_dl = make_loader(test, tok, args.max_len, args.batch * 2, False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    setup_mlflow()
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "model": args.model_name, "epochs": args.epochs, "lr": args.lr,
            "batch": args.batch, "max_len": args.max_len, "seed": SEED,
            "subset": args.subset, **dataset_versions()})
        mlflow.set_tag("git_commit", git_commit())

        best_f1, best_state, best_epoch = -1.0, None, 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            for step, batch in enumerate(train_dl, 1):
                batch = {k: v.to(device) for k, v in batch.items()}
                opt.zero_grad()
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                    loss = model(**batch).loss
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                running += loss.item()
                if step % 100 == 0:
                    print(f"[xlmr] epoch {epoch} step {step}/{len(train_dl)} "
                          f"loss={running / step:.4f}", flush=True)

            val_prob = predict(model, val_dl, device)
            val_m = metrics_for(val["label_id"], (val_prob >= 0.5).astype(int), val_prob)
            mlflow.log_metric("train_loss", running / len(train_dl), step=epoch)
            mlflow.log_metric("val_scam_f1_by_epoch", val_m["scam_f1"], step=epoch)
            print(f"[xlmr] epoch {epoch} val F1={val_m['scam_f1']:.4f}", flush=True)
            if val_m["scam_f1"] > best_f1:
                best_f1, best_epoch = val_m["scam_f1"], epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.to(device)
        val_prob = predict(model, val_dl, device)
        test_prob = predict(model, test_dl, device)
        val_pred = (val_prob >= 0.5).astype(int)
        test_pred = (test_prob >= 0.5).astype(int)
        val_m = metrics_for(val["label_id"], val_pred, val_prob)
        test_m = metrics_for(test["label_id"], test_pred, test_prob)
        test_lang = f1_by_language(test, test_pred)

        mlflow.log_param("best_epoch", best_epoch)
        for k, v in val_m.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_m.items():
            mlflow.log_metric(f"test_{k}", v)
        for lang, v in test_lang.items():
            mlflow.log_metric(f"test_f1_{lang}", v)

        REPORTS.mkdir(exist_ok=True)
        png = REPORTS / f"confusion_{args.run_name}.png"
        save_confusion(test["label_id"], test_pred, f"{args.run_name} (test)", png)
        mlflow.log_artifact(str(png))

        try:
            import mlflow.transformers
            mlflow.transformers.log_model(
                transformers_model={"model": model.cpu(), "tokenizer": tok},
                name="model", task="text-classification")
            print("[xlmr] model logged with mlflow.transformers")
        except Exception as e:
            print(f"[xlmr] mlflow.transformers failed ({e}); saving raw files instead")
            out = Path("xlmr_model")
            model.cpu().save_pretrained(out)
            tok.save_pretrained(out)
            mlflow.log_artifacts(str(out), artifact_path="model")

    report = {
        "model": args.run_name,
        "params": {"model": args.model_name, "epochs": args.epochs, "lr": args.lr,
                   "batch": args.batch, "max_len": args.max_len, "seed": SEED,
                   "best_epoch": best_epoch},
        "val": {k: round(v, 6) for k, v in val_m.items()},
        "test": {k: round(v, 6) for k, v in test_m.items()},
        "test_f1_by_language": {k: round(v, 6) for k, v in sorted(test_lang.items())},
    }
    (REPORTS / f"metrics_{args.run_name}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"[xlmr] test F1={test_m['scam_f1']:.4f} precision={test_m['scam_precision']:.4f} "
          f"recall={test_m['scam_recall']:.4f} PR-AUC={test_m['pr_auc']:.4f} "
          f"false_alarm={test_m['false_alarm_rate']:.4f}")


if __name__ == "__main__":
    main()