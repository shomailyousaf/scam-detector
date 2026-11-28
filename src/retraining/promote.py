import argparse
import sys

from mlflow import MlflowClient

from src.common.io import load_params
from src.models.mlflow_utils import setup_mlflow
from src.retraining.common import get_role, load_role, set_role, version_metrics, write_json

SMOKE = ["Apna OTP abhi bhejo warna account band ho jayega",
         "Kal class 10 baje hai, notes le aana"]


def decide(champ, chall, r):
    test_drop = champ["test_scam_f1"] - chall["test_scam_f1"]
    challenge_gain = chall["challenge_v2_scam_f1"] - champ["challenge_v2_scam_f1"]
    checks = {
        "test_f1_not_worse": test_drop <= r["max_test_f1_drop"],
        "challenge_f1_not_worse": challenge_gain >= r["min_challenge_f1_gain"],
    }
    return all(checks.values()), checks, round(test_drop, 4), round(challenge_gain, 4)


def smoke_test(cfg):
    # the new champion has to load and give sensible probabilities before we trust it
    model, _ = load_role(cfg, "champion")
    prob = model.predict_proba(SMOKE)[:, 1]
    return bool(((prob >= 0) & (prob <= 1)).all())


def promote(cfg, challenger):
    client, name = MlflowClient(), cfg["retraining"]["model_name"]
    champion = get_role(cfg, "champion")
    if challenger is None or champion is None:
        sys.exit("need both a champion and a challenger")
    if challenger == champion:
        sys.exit(f"version {challenger} is already the champion")

    champ, chall = version_metrics(cfg, champion), version_metrics(cfg, challenger)
    ok, checks, test_drop, challenge_gain = decide(champ, chall, cfg["retraining"])
    result = {"champion_before": champion, "challenger": challenger, "checks": checks,
              "test_f1_drop": test_drop, "challenge_f1_gain": challenge_gain,
              "champion_metrics": champ, "challenger_metrics": chall}

    if not ok:
        client.set_model_version_tag(name, str(challenger), "status", "rejected")
        result.update(decision="rejected", champion_after=champion)
    else:
        set_role(cfg, "previous_champion", champion)
        set_role(cfg, "champion", challenger)
        client.set_model_version_tag(name, str(challenger), "status", "promoted")
        if smoke_test(cfg):
            result.update(decision="promoted", champion_after=challenger)
        else:
            # safe promotion, if the new model is broken we go straight back
            set_role(cfg, "champion", champion)
            client.set_model_version_tag(name, str(challenger), "status", "rolled_back")
            result.update(decision="rolled_back_after_smoke_test", champion_after=champion)

    write_json(result, f"reports/retraining/decision_v{challenger}.json")
    return result


def main():
    cfg = load_params()
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenger", type=int, default=None)
    a = ap.parse_args()
    setup_mlflow()
    challenger = a.challenger or get_role(cfg, "challenger")
    res = promote(cfg, challenger)

    c, h = res["champion_metrics"], res["challenger_metrics"]
    print(f"[promote] champion v{res['champion_before']} vs challenger v{res['challenger']}")
    for k in ("test_scam_f1", "challenge_v2_scam_f1", "challenge_v2_false_alarm_rate"):
        print(f"   {k:32s} champion {c[k]:.4f}   challenger {h[k]:.4f}")
    for k, v in res["checks"].items():
        print(f"   {'PASS' if v else 'FAIL'}  {k}")
    print(f"[promote] decision: {res['decision'].upper()}, champion is now v{res['champion_after']}")


if __name__ == "__main__":
    main()
