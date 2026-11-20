"""One-command reproduction (works on Windows, Mac, Linux and in Docker).

    python reproduce.py

1. downloads the versioned raw data from the DVC remote (if not present)
2. reruns EVERY pipeline stage from scratch (ignores cached results)
3. runs the automated tests
4. verifies the rebuilt outputs are identical to the committed dvc.lock
"""
import subprocess
import sys
from pathlib import Path

RAW = Path("data/raw/scam_messages_v1.csv")


def run(step: str, cmd: list[str], must_pass: bool = True) -> bool:
    print(f"\n=== {step} ===\n$ {' '.join(cmd)}", flush=True)
    ok = subprocess.run(cmd).returncode == 0
    if not ok and must_pass:
        print(f"\nFAILED at: {step}")
        sys.exit(1)
    return ok


def main() -> None:
    if not RAW.exists():
        run("1/4 Pull raw data from DVC remote", ["dvc", "pull", f"{RAW}.dvc"])
    else:
        print("\n=== 1/4 Raw data already present, skipping pull ===")

    run("2/4 Rebuild the whole pipeline from scratch", ["dvc", "repro", "--force"])
    run("3/4 Run automated tests", [sys.executable, "-m", "pytest", "-q"])

    same = run("4/4 Compare with committed results",
               ["git", "diff", "--exit-code", "--stat", "dvc.lock", "reports"], must_pass=False)
    if same:
        print("\nREPRODUCED: every output is identical to the committed dvc.lock")
    else:
        print("\nMISMATCH: outputs differ from the committed version (see diff above)")
        sys.exit(1)


if __name__ == "__main__":
    main()