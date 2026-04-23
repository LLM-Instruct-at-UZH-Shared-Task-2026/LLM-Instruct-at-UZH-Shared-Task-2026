from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import yaml
from pathlib import Path

@dataclass
class Cfg:
    raw: Dict[str, Any]

    @staticmethod
    def load(path: str | Path) -> "Cfg":
        p = Path(path)
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("config.yaml must be a mapping")
        return Cfg(raw=data)

    def get(self, *keys, default=None):
        cur: Any = self.raw
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur
