#!/usr/bin/env python3
"""Build FAISS RAG index from fast pseudo-labels.

This is a simplified version that works directly with the fast pseudo-labels
format produced by build_pseudo_labels_fast.py.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Cfg
from src.tags import load_tag_metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudo-labels", default="outputs/pseudo_labels/pseudo_labels.jsonl")
    ap.add_argument("--index-path",    default="outputs/rag_index.faiss")
    ap.add_argument("--meta-path",     default="outputs/rag_meta.json")
    ap.add_argument("--config",        default="config.yaml")
    ap.add_argument("--lang",          default="en")
    ap.add_argument("--min-conf",      type=float, default=0.70)
    args = ap.parse_args()

    try:
        import faiss
        import numpy as np
    except ImportError:
        print("[ERROR] faiss-cpu not installed. Run: pip install faiss-cpu")
        sys.exit(1)

    pl_path = Path(args.pseudo_labels)
    if not pl_path.exists():
        print(f"[ERROR] Pseudo-label file not found: {pl_path}")
        sys.exit(1)

    print(f"[rag] Loading pseudo-labels from {pl_path} ...")
    records = []
    with open(pl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[rag] Loaded {len(records)} docs from pseudo-labels.")

    # Flatten to individual paragraphs
    paras = []
    for rec in records:
        for p in rec.get("paragraphs", []):
            text_en = (p.get("text_en") or "").strip()
            text_fr = (p.get("text_fr") or "").strip()
            text = text_en if args.lang == "en" else text_fr
            if not text:
                continue
            conf = float(p.get("type_conf", 0.0))
            if conf < args.min_conf and p.get("tags"):
                continue  # skip low-confidence without tags
            paras.append({
                "text":      text,
                "text_fr":   text_fr,
                "text_en":   text_en,
                "type":      p.get("type"),
                "type_conf": conf,
                "tags":      p.get("tags", []),
                "source":    rec.get("source_file", ""),
                "para_number": p.get("para_number"),
            })

    print(f"[rag] {len(paras)} paragraphs to index.")
    if not paras:
        print("[ERROR] No paragraphs found. Aborting.")
        sys.exit(1)

    cfg = Cfg.load(args.config)
    embed_name = cfg.get("models", "embedding_name")
    device_cfg = cfg.get("models", "device", default="auto")

    print(f"[rag] Loading embedder: {embed_name} ...")
    from src.embed import Embedder
    dev = "cuda" if device_cfg in ("auto", "cuda") else "cpu"
    embedder = Embedder(embed_name, device=dev)

    texts = [p["text"] for p in paras]
    print(f"[rag] Encoding {len(texts)} paragraphs ...")
    import numpy as np
    embs = embedder.encode(texts).astype("float32")
    # Normalize for cosine similarity
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1e-9, norms)
    embs_norm = embs / norms

    dim = embs_norm.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product on normalized = cosine
    index.add(embs_norm)
    print(f"[rag] Added {index.ntotal} vectors to FAISS index (dim={dim}).")

    index_path = Path(args.index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    print(f"[rag] Saved index → {index_path}")

    meta_path = Path(args.meta_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(paras, f, ensure_ascii=False)
    print(f"[rag] Saved metadata → {meta_path} ({len(paras)} entries)")
    print("[rag] Done. Set rag.enabled: true in config.yaml to activate.")


if __name__ == "__main__":
    main()
