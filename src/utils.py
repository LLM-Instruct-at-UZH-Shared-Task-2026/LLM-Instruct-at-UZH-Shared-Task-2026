from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def safe_first_token(s: str) -> str:
    s = normalize_ws(s)
    return s.split(" ", 1)[0].strip().strip(",;:()[]{}\"'").lower() if s else ""
