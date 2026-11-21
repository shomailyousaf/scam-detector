## Model results (Member B)

All models were trained on `train.csv`, selected on `val.csv`, and reported on `test.csv`. Seed 42.

| Model | Test F1 | Precision | Recall | PR-AUC | False-alarm rate |
|---|---|---|---|---|---|
| Rules only | 0.6230 | 0.7140 | 0.5526 | 0.6763 | 0.1636 |
| Char TF-IDF + Logistic Regression | 0.9574 | 0.9632 | 0.9517 | 0.9931 | 0.0268 |
| **XLM-RoBERTa base (registered)** | **0.9799** | **0.9865** | **0.9734** | **0.9980** | **0.0098** |

XLM-RoBERTa is registered in the MLflow Model Registry as `scam-detector-xlmr`, version 1.

### Test F1 by language (XLM-RoBERTa)
English 0.980, Roman-Urdu 0.993, Code-Mixed 0.941, Urdu 0.895.

### Limitations
- Urdu is the weakest language (F1 0.895). It has only 638 training rows, against 36,359 for English.
- Validation scores (F1 0.929) are lower than test scores (0.980). The splits hold different messages with a different language mix.

### How to reproduce
- Data and CPU baselines: `python reproduce.py`
- XLM-RoBERTa needs a GPU (Kaggle or Colab). Install `requirements-gpu.txt`, run `dvc pull`, then
  `python -m src.models.train_xlmr --epochs 2 --run_name xlmr_base`
- Experiment tracking: https://dagshub.com/shomailusuf/scam-detector.mlflow