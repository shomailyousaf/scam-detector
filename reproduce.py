"""One-command reproduction (works on Windows, Mac, Linux and in Docker).

    python reproduce.py

1. downloads the versioned raw data from the DVC remote (if not present)
2. reruns EVERY pipeline stage from scratch (ignores cached results)
3. runs the automated tests
4. verifies every rebuilt file has the same MD5 fingerprint as in the
   committed dvc.lock (only content hashes are compared, so file-permission
   flags that differ between Windows, Linux and Docker do not matter)
"""
import subprocess
import sys
from pathlib import Path

import yaml

RAW = Path("data/raw/scam_messages_v1.csv")


def run(step: str, cmd: list[str]) -> None:
    print(f"\n=== {step} ===\n$ {' '.join(cmd)}", flush=True)
    if subprocess.run(cmd).returncode != 0:
        print(f"\nFAILED at: {step}")
        sys.exit(1)


def hashes(lock_text: str) -> dict:
    """{(stage, 'deps'/'outs', path): md5} from a dvc.lock file."""
    lock = yaml.safe_load(lock_text)
    out = {}
    for stage, info in lock.get("stages", {}).items():
        for kind in ("deps", "outs"):
            for item in info.get(kind, []):
                out[(stage, kind, item["path"])] = item["md5"]
    return out


def main() -> None:
    if not RAW.exists():
        run("1/4 Pull raw data from DVC remote", ["dvc", "pull", f"{RAW}.dvc"])
    else:
        print("\n=== 1/4 Raw data already present, skipping pull ===")

    run("2/4 Rebuild the whole pipeline from scratch", ["dvc", "repro", "--force"])
    run("3/4 Run automated tests", [sys.executable, "-m", "pytest", "-q"])

    print("\n=== 4/4 Compare MD5 of every input/output with the committed dvc.lock ===")
    committed = subprocess.run(["git", "show", "HEAD:dvc.lock"],
                               capture_output=True, text=True, check=True).stdout
    before, after = hashes(committed), hashes(Path("dvc.lock").read_text(encoding="utf-8"))
    diffs = [k for k in before.keys() | after.keys() if before.get(k) != after.get(k)]
    for stage, kind, path in sorted(diffs):
        print(f"  DIFFERENT  {stage:13s} {kind}  {path}")
    print(f"  compared {len(before)} files, {len(diffs)} different")

    if diffs:
        print("\nMISMATCH: some outputs differ from the committed version")
        sys.exit(1)
    print("\nREPRODUCED: every input and output is identical to the committed dvc.lock")


if __name__ == "__main__":
    main()