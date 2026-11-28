# one time setup, trains the current tfidf model on the training data
# and registers it as the first champion
from pathlib import Path

from src.common.io import load_params
from src.models.mlflow_utils import setup_mlflow
from src.retraining.common import (find_version, get_role, read, set_role, train_log_register,
                                   write_json)


def main():
    cfg = load_params()
    setup_mlflow()
    if get_role(cfg, "champion") is not None:
        print(f"[bootstrap] version {get_role(cfg, 'champion')} is already the champion, nothing to do")
        return
    # an earlier bootstrap may have registered the model without the role sticking
    earlier = find_version(cfg, "trigger", "bootstrap")
    if earlier is not None:
        set_role(cfg, "champion", earlier)
        print(f"[bootstrap] marked existing version {earlier} as champion")
        return
    train = read(Path(cfg["data"]["processed_dir"]) / "train.csv")
    _, version, metrics, _ = train_log_register(
        train, cfg, "champion_v1", {"trigger": "bootstrap", "trained_on": "train"})
    set_role(cfg, "champion", version)
    write_json({"champion_version": version, **metrics}, "reports/retraining/bootstrap.json")
    print(f"[bootstrap] registered version {version} as champion")
    print(f"[bootstrap] test f1 {metrics['test_scam_f1']:.4f}  challenge f1 {metrics['challenge_v2_scam_f1']:.4f}")


if __name__ == "__main__":
    main()
