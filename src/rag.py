"""RAG (Retrieval-Augmented Generation) index wrapper.

Provides a simple interface for retrieving the k most similar paragraphs
from the pseudo-labeled training corpus, keyed by embedding similarity.

Usage::

    rag = RagIndex.load(
        index_path="outputs/rag_index.faiss",
        meta_path="outputs/rag_meta.json",
        embedder=embedder,
    )
    hits = rag.retrieve(text_en, k=3)
    # hits: list of dict {text, text_fr, text_en, type, tags, score, ...}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


class RagIndex:
    """Thin wrapper around a FAISS index + parallel metadata list."""

    def __init__(self, index, meta: List[Dict], embedder):
        self._index = index
        self._meta = meta
        self._embedder = embedder

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        index_path: str,
        meta_path: str,
        embedder,
    ) -> "RagIndex":
        """Load a FAISS index and its metadata from disk.

        Args:
            index_path:  Path to the .faiss file.
            meta_path:   Path to the parallel JSON metadata file.
            embedder:    Embedder instance used to encode query paragraphs.

        Raises:
            ImportError: If faiss-cpu is not installed.
            FileNotFoundError: If either path does not exist.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for RAG retrieval.  "
                "Install with: pip install faiss-cpu"
            )

        index_path = Path(index_path)
        meta_path = Path(meta_path)

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"RAG metadata not found: {meta_path}")

        index = faiss.read_index(str(index_path))

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        if index.ntotal != len(meta):
            raise ValueError(
                f"Index/metadata size mismatch: {index.ntotal} vs {len(meta)}"
            )

        return cls(index=index, meta=meta, embedder=embedder)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = 3,
        min_score: float = 0.0,
    ) -> List[Dict]:
        """Retrieve the top-k most similar paragraphs for a query text.

        Args:
            query:      Query text to embed and search.
            k:          Number of nearest neighbours to return.
            min_score:  Minimum cosine similarity to include a result.

        Returns:
            List of metadata dicts (from the pseudo-label JSONL), each
            enriched with a ``"score"`` field (cosine similarity 0-1).
        """
        import numpy as np

        if not query.strip():
            return []

        emb = self._embedder.encode([query])  # (1, D)
        emb = emb.astype("float32")
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        norm = max(float(norm[0, 0]), 1e-9)
        emb_norm = emb / norm

        k_actual = min(k, self._index.ntotal)
        scores, indices = self._index.search(emb_norm, k_actual)  # (1, k)

        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if float(score) < min_score:
                continue
            hit = dict(self._meta[idx])
            hit["score"] = float(score)
            hits.append(hit)

        return hits

    def retrieve_batch(
        self,
        queries: List[str],
        k: int = 3,
        min_score: float = 0.0,
    ) -> List[List[Dict]]:
        """Batch-encode all queries, then FAISS-search once.

        Returns a list-of-lists with the same length as ``queries``.
        Empty queries get an empty result list.
        """
        import numpy as np

        results: List[List[Dict]] = [[] for _ in queries]
        active = [(i, q) for i, q in enumerate(queries) if q.strip()]
        if not active:
            return results

        texts = [q for _, q in active]
        embs = self._embedder.encode(texts)  # (N, D) — single batch call
        embs = embs.astype("float32")
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        embs_norm = embs / norms

        k_actual = min(k, self._index.ntotal)
        all_scores, all_indices = self._index.search(embs_norm, k_actual)  # (N, k)

        for row, (orig_i, _) in enumerate(active):
            hits = []
            for score, idx in zip(all_scores[row], all_indices[row]):
                if idx < 0:
                    continue
                if float(score) < min_score:
                    continue
                hit = dict(self._meta[idx])
                hit["score"] = float(score)
                hits.append(hit)
            results[orig_i] = hits

        return results

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of indexed paragraphs."""
        return self._index.ntotal

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"RagIndex(size={self.size})"
