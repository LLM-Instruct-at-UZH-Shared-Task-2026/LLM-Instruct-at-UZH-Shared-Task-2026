import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ZIP_PATH = BASE.parent / "argmining_uzh_shared_task_baseline.zip"

def main():
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for p in BASE.rglob("*"):
            if p.is_file() and ".venv" not in str(p):
                z.write(p, p.relative_to(BASE))
    print("Wrote:", ZIP_PATH)

if __name__ == "__main__":
    main()
