#!/usr/bin/env python3
"""Fast pseudo-label builder using heuristic type + embedding-based tags (no LLM).

Runs on all 2694 training docs in ~5 minutes.
Usage:
    python -m scripts.build_pseudo_labels_fast
"""
from __future__ import annotations
import argparse
import glob
import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Cfg
from src.tags import load_tag_metadata
from src.pipeline.task1 import classify_type, predict_tags


def _load_train_file(path: str, idx: int) -> dict:
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    paragraphs = []
    para_num = 1
    for item in items:
        if item.get("type") != "paragraph":
            continue
        paragraphs.append({
            "para_number": para_num,
            "para":    (item.get("text_fr") or "").strip(),
            "para_en": (item.get("text_en") or "").strip(),
            "tags": [], "type": None, "think": "", "matched_pars": {},
        })
        para_num += 1
    return {
        "TEXT_ID": f"TRAIN_{idx:04d}_{Path(path).stem}",
        "body": {"paragraphs": paragraphs},
        "METADATA": {"structure": {"preambular_para": [], "operative_para": [], "think": ""}},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  default="config.yaml")
    ap.add_argument("--train-dir", default="dataset/train-data")
    ap.add_argument("--out-dir",   default="outputs/pseudo_labels")
    ap.add_argument("--limit",   type=int, default=0)
    ap.add_argument("--lang",    default="en")
    args = ap.parse_args()

    cfg = Cfg.load(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embed_name = cfg.get("models", "embedding_name")
    device_cfg = cfg.get("models", "device", default="auto")
    k_tag   = int(cfg.get("task1", "k_tag_candidates",  default=40))
    thr     = float(cfg.get("task1", "tag_conf_threshold", default=0.30))
    max_t   = int(cfg.get("task1", "max_tags_per_para", default=8))
    lang    = args.lang

    train_files = sorted(glob.glob(str(Path(args.train_dir) / "*.json")))
    if args.limit > 0:
        train_files = train_files[:args.limit]
    print(f"[fast] {len(train_files)} train docs | embed={embed_name}")

    print("[fast] Loading embedder …")
    from src.embed import Embedder
    dev = "cuda" if device_cfg in ("auto", "cuda") else "cpu"
    embedder = Embedder(embed_name, device=dev)

    tag_rows = load_tag_metadata(
        cfg.get("data", "tags_csv_path", default="dataset/education_dimensions_updated.csv"),
    )
    print(f"[fast] {len(tag_rows)} tags loaded. Pre-computing embeddings …")
    tag_embs = embedder.encode([r["_text"] for r in tag_rows])
    print(f"[fast] Tag embeddings ready ({tag_embs.shape}).")

    out_path = out_dir / "pseudo_labels.jsonl"
    written = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, fpath in enumerate(tqdm(train_files, desc="train docs")):
            doc = _load_train_file(fpath, i)
            paras = doc["body"]["paragraphs"]
            out_paras = []
            for p in paras:
                text = (p["para_en"] if lang == "en" else p["para"]).strip()
                if not text:
                    continue
                # Heuristic type
                t, conf, _ = classify_type(text, lang=lang, mode="heuristic", llm=None)
                # Embedding tags
                tags, _ = predict_tags(
                    text, tag_rows=tag_rows, mode="heuristic",
                    embedder=embedder, llm=None,
                    k_candidates=k_tag, conf_threshold=thr, max_tags=max_t,
                    tag_embs=tag_embs,
                )
                if conf >= 0.70:
                    out_paras.append({
                        "para_number": p["para_number"],
                        "text_fr": p["para"],
                        "text_en": p["para_en"],
                        "type": t,
                        "type_conf": conf,
                        "tags": tags,
                        "source": str(fpath),
                    })
            if out_paras:
                record = {"source_file": str(fpath), "paragraphs": out_paras}
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                written += 1

    print(f"[fast] Done. {written} docs → {out_path}")


if __name__ == "__main__":
    main()
