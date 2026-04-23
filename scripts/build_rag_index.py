#!/usr/bin/env python3
"""Build a FAISS RAG index from pseudo-labeled training paragraphs.

Run after build_pseudo_labels.py has produced outputs/pseudo_labels/pseudo_labels.jsonl.

Usage::

    python -m scripts.build_rag_index \\
        --pseudo-labels outputs/pseudo_labels/pseudo_labels.jsonl \\
        --index-path    outputs/rag_index.faiss \\
        --meta-path     outputs/rag_meta.json \\
        --config        config.yaml
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Cfg


def main():
    ap = argparse.ArgumentParser(description="Build FAISS RAG index from pseudo-labels.")
    ap.add_argument("--pseudo-labels", default="outputs/pseudo_labels/pseudo_labels.jsonl")
    ap.add_argument("--index-path",    default="outputs/rag_index.faiss")
    ap.add_argument("--meta-path",     default="outputs/rag_meta.json")
    ap.add_argument("--config",        default="config.yaml")
    ap.add_argument("--lang",          default="en", choices=["en", "fr"],
                    help="Which language's text to embed for the index.")
    args = ap.parse_args()

    cfg = Cfg.load(args.config)
    try:
        import faiss
        import numpy as np
    except ImportError:
        print("[ERROR] faiss-cpu is not installed.  Run: pip install faiss-cpu")
        sys.exit(1)

    # ── Load pseudo-labels ─────────────────────────────────────────────────
    pl_path = Path(args.pseudo_labels)
    if not pl_path.exists():
        print(f"[ERROR] Pseudo-label file not found: {pl_path}\n"
              "Run build_pseudo_labels.py first.")
        sys.exit(1)

    records = []
    with open(pl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Flatten to individual paragraphs
    paras = []
    for rec in records:
        for p in rec.get("paragraphs", []):
            text = (p.get("text_en") if args.lang == "en" else p.get("text_fr") or "").strip()
            if text:
                paras.append({
                    "text": text,
                    "text_fr": p.get("text_fr", ""),
                    "text_en": p.get("text_en", ""),
                    "type": p.get("type"),
                    "type_conf": p.get("type_conf", 0.0),
                    "tags": p.get("tags", []),
                    "source": rec.get("source_file", ""),
                    "para_number": p.get("para_number"),
                })

    print(f"[rag] {len(paras)} paragraphs to index.")
    if not paras:
        print("[ERROR] No paragraphs found — aborting.")
        sys.exit(1)

    # ── Embed ──────────────────────────────────────────────────────────────
    embed_name = cfg.get("models", "embedding_name")
    device = cfg.get("models", "device", default="auto")
    dev_emb = "cuda" if device in ("auto", "cuda") else "cpu"

    print(f"[rag] Loading embedder {embed_name!r} …")
    from src.embed import Embedder
    embedder = Embedder(embed_name, device=dev_emb)

    texts = [p["text"] for p in paras]
    print("[rag] Encoding paragraphs …")
    embs = embedder.encode(texts)  # (N, D) numpy float32
    embs = embs.astype("float32")
    if embs.ndim != 2:
        raise ValueError(f"Expected 2-D embeddings, got shape {embs.shape}")

    # L2-normalize for cosine similarity via inner product
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    embs_norm = embs / norms

    # ── Build FAISS index ──────────────────────────────────────────────────
    dim = embs_norm.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product ≡ cosine on normalised vecs
    index.add(embs_norm)
    print(f"[rag] Index built: {index.ntotal} vectors, dim={dim}.")

    # Save index
    index_path = Path(args.index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    print(f"[rag] FAISS index → {index_path}")

    # Save metadata (parallel list to index vectors)
    meta_path = Path(args.meta_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(paras, f, ensure_ascii=False, indent=2)
    print(f"[rag] Metadata    → {meta_path}")
    print(f"[rag] Done!  {len(paras)} entries indexed.")


if __name__ == "__main__":
    main()
