"""Quick smoke test for Task 2 judge_relation.

Usage:
  python eval/_test_task2.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import google.genai as genai
import yaml
from dotenv import load_dotenv


def _get_paragraphs(doc: dict):
    body = doc.get("body") or {}
    return body.get("paragraphs") or body.get("paras") or []


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    sys.path.insert(0, str(repo_root))

    from eval.judge import judge_relation  # local import after path setup

    with (repo_root / "eval" / "judge_settings.yaml").open("r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    api_key = os.environ["GOOGLE_API_KEY"]
    model_name = os.environ["GEMINI_MODEL"]
    client = genai.Client(api_key=api_key)
    model = (client, model_name)

    with (repo_root / "outputs" / "submission.json").open("r", encoding="utf-8") as f:
        sub = json.load(f)

    doc = sub[0]
    paras = _get_paragraphs(doc)

    rels = []
    para_by_num = {p["para_number"]: p for p in paras if "para_number" in p}
    for p in paras:
        for b_str, rel in (p.get("matched_paras") or p.get("matched_pars") or {}).items():
            b_num = int(b_str)
            if b_num in para_by_num:
                rels.append((p, para_by_num[b_num], rel if isinstance(rel, str) else rel[0]))

    print(f"Doc {doc.get('TEXT_ID', '')} - {len(rels)} relations")
    if not rels:
        raise RuntimeError("No relations found in outputs/submission.json")

    para_a, para_b, pred = rels[0]
    print(f"Testing para {para_a['para_number']} -> {para_b['para_number']}, relation={pred}")
    result = judge_relation(model, para_a, para_b, pred, settings, lang="en")
    print("SUCCESS:", result)


if __name__ == "__main__":
    main()
