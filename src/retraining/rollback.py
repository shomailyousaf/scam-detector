
import sys

from mlflow import MlflowClient

from src.common.io import load_params
from src.models.mlflow_utils import setup_mlflow
from src.retraining.common import get_role, set_role, write_json


def rollback(cfg):
    previous, current = get_role(cfg, "previous_champion"), get_role(cfg, "champion")
    if previous is None:
        sys.exit("no previous champion to go back to")
    set_role(cfg, "champion", previous)
    set_role(cfg, "previous_champion", current)
    MlflowClient().set_model_version_tag(cfg["retraining"]["model_name"], str(current),
                                         "status", "rolled_back")
    res = {"decision": "rolled_back", "champion_before": current, "champion_after": previous}
    write_json(res, "reports/retraining/rollback.json")
    return res


def main():
    cfg = load_params()
    setup_mlflow()
    res = rollback(cfg)
    print(f"[rollback] champion v{res['champion_before']} replaced, champion is now v{res['champion_after']}")


if __name__ == "__main__":
    main()
