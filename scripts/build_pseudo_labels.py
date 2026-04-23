#!/usr/bin/env python3
"""Build pseudo-labels from the 2,694 training documents.

The training data contains documents with paragraphs (type, text_fr, text_en)
but NO gold preambular/operative or tag labels.  This script runs the pipeline
(Task 1a + 1b) on all training docs and saves high-confidence predictions as
pseudo-labels that can later be used for RAG retrieval.

Usage::

    # From repo root (activate venv first):
    python -m scripts.build_pseudo_labels \\
        --config config.yaml \\
        --train-dir dataset/train-data \\
        --out-dir outputs/pseudo_labels \\
        --conf-threshold 0.70 \\
        --limit 0          # 0 = all docs
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from pathlib import Path

# Allow running as a top-level script from `python -m scripts.build_pseudo_labels`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Cfg
from src.tags import load_tag_metadata


# ─────────────────────────────────────────────────────────────────────────────
# Train-doc → pipeline-compatible format
# ─────────────────────────────────────────────────────────────────────────────

def _convert_train_doc(path: str, idx: int) -> dict:
    """Convert a training JSON file to the pipeline's input format.

    Training format: list of {type, level, text_fr, text_en}
    Pipeline format: {TEXT_ID, body: {paragraphs: [...]}}
    """
    with open(path, encoding="utf-8") as f:
        items = json.load(f)

    paragraphs = []
    para_num = 1
    for item in items:
        if item.get("type") != "paragraph":
            continue  # skip headings
        paragraphs.append({
            "para_number": para_num,
            "para":    (item.get("text_fr") or "").strip(),
            "para_en": (item.get("text_en") or "").strip(),
            "tags":       [],
            "type":       None,
            "think":      "",
            "matched_pars": {},
        })
        para_num += 1

    stem = Path(path).stem
    return {
        "TEXT_ID": f"TRAIN_{idx:04d}_{stem}",
        "body": {"paragraphs": paragraphs},
        "METADATA": {"structure": {
            "preambular_para": [],
            "operative_para": [],
            "think": "",
        }},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Build pseudo-labels from training data.")
    ap.add_argument("--config",         default="config.yaml")
    ap.add_argument("--train-dir",      default="dataset/train-data")
    ap.add_argument("--out-dir",        default="outputs/pseudo_labels")
    ap.add_argument("--conf-threshold", type=float, default=0.70,
                    help="Minimum confidence to keep a pseudo-label.")
    ap.add_argument("--limit",          type=int, default=0,
                    help="Process at most this many docs (0 = all).")
    ap.add_argument("--batch-size",     type=int, default=8,
                    help="Docs per checkpoint flush.")
    args = ap.parse_args()

    cfg = Cfg.load(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect training files ─────────────────────────────────────────────
    train_files = sorted(glob.glob(os.path.join(args.train_dir, "*.json")))
    if not train_files:
        print(f"[ERROR] No JSON files in {args.train_dir!r}"); sys.exit(1)
    if args.limit > 0:
        train_files = train_files[:args.limit]
    print(f"[pseudo] {len(train_files)} training docs to process.")

    # ── Load models ────────────────────────────────────────────────────────
    generator    = cfg.get("models", "generator_name")
    embed_name   = cfg.get("models", "embedding_name")
    device       = cfg.get("models", "device", default="auto")
    load_in_4bit = bool(cfg.get("models", "load_in_4bit", default=True))
    enable_thinking = bool(cfg.get("models", "enable_thinking", default=False))
    task1_mode   = cfg.get("task1", "mode", default="llm")
    k_tag        = int(cfg.get("task1", "k_tag_candidates", default=40))
    tag_thr      = float(cfg.get("task1", "tag_conf_threshold", default=0.30))
    max_tags     = int(cfg.get("task1", "max_tags_per_para", default=8))

    print("[pseudo] Loading LLM …")
    from src.llm import LocalLLM
    llm = LocalLLM(generator, device=device, load_in_4bit=load_in_4bit,
                   enable_thinking=enable_thinking)

    print("[pseudo] Loading embedder …")
    from src.embed import Embedder, topk_by_cosine
    dev_emb = "cuda" if device in ("auto", "cuda") else "cpu"
    embedder = Embedder(embed_name, device=dev_emb)

    tag_rows = load_tag_metadata(cfg.get("data", "tags_csv_path"))
    tag_embs = embedder.encode([r["_text"] for r in tag_rows]) if tag_rows else None

    # ── Pipeline imports ───────────────────────────────────────────────────
    from src.pipeline.task1 import classify_type, predict_tags

    # ── Process ────────────────────────────────────────────────────────────
    results_path = out_dir / "pseudo_labels.jsonl"
    stats = {"total_para": 0, "kept_type": 0, "kept_tag": 0, "docs": 0}

    with open(results_path, "w", encoding="utf-8") as fout:
        for file_idx, fpath in enumerate(train_files):
            doc = _convert_train_doc(fpath, file_idx)
            paras = doc["body"]["paragraphs"]
            if not paras:
                continue

            pseudo_paras = []
            for p in paras:
                text_fr = p["para"]
                text_en = p["para_en"]
                # Use English text as primary (better for English tags)
                text = text_en if text_en else text_fr

                # Task 1a: type classification
                t, t_conf, t_think = classify_type(
                    text, lang="en", mode=task1_mode, llm=llm
                )
                p["type"] = t
                p["think"] = t_think
                stats["total_para"] += 1
                if t_conf >= args.conf_threshold:
                    stats["kept_type"] += 1

                # Task 1b: tag prediction
                tags, tag_think = predict_tags(
                    text, tag_rows=tag_rows, mode=task1_mode,
                    embedder=embedder, llm=llm,
                    k_candidates=k_tag, conf_threshold=tag_thr,
                    max_tags=max_tags, tag_embs=tag_embs,
                )
                p["tags"] = tags
                if tags:
                    stats["kept_tag"] += 1

                # Keep pseudo-label only if type confidence is high enough
                if t_conf >= args.conf_threshold:
                    pseudo_paras.append({
                        "para_number": p["para_number"],
                        "text_fr": text_fr,
                        "text_en": text_en,
                        "type": t,
                        "type_conf": t_conf,
                        "tags": tags,
                        "source_file": Path(fpath).name,
                    })

            if pseudo_paras:
                record = {
                    "TEXT_ID": doc["TEXT_ID"],
                    "source_file": Path(fpath).name,
                    "paragraphs": pseudo_paras,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            stats["docs"] += 1
            if (file_idx + 1) % args.batch_size == 0:
                print(f"  [{file_idx+1}/{len(train_files)}] docs processed  "
                      f"paras={stats['total_para']}  "
                      f"kept_type={stats['kept_type']}  kept_tag={stats['kept_tag']}",
                      flush=True)

    print(f"\n[pseudo] Done.  Docs={stats['docs']}  Total paras={stats['total_para']}")
    print(f"         High-conf type: {stats['kept_type']}  "
          f"Tagged: {stats['kept_tag']}")
    print(f"[pseudo] Output → {results_path}")


if __name__ == "__main__":
    main()
