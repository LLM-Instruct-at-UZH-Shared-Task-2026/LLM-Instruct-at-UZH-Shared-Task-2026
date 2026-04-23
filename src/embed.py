from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


def _doc_paras(doc: Dict) -> List[Dict]:
    body = doc.get("body") or {}
    return body.get("paras") or body.get("paragraphs") or []


class Embedder:
    """SentenceTransformer wrapper with normalised cosine similarity helpers."""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], normalize: bool = True, batch_size: int = 64) -> np.ndarray:
        """Encode a list of texts, return (N, D) normalised float32 array."""
        embs = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=batch_size,
        )
        if normalize:
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
            embs = embs / norms
        return embs.astype(np.float32)


def topk_by_cosine(query_emb: np.ndarray, cand_embs: np.ndarray, k: int) -> List[int]:
    """Return indices of top-k candidates by cosine similarity (highest first)."""
    sims = cand_embs @ query_emb
    k = min(k, sims.shape[0])
    idx = np.argpartition(-sims, kth=k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx.tolist()


# ──────────────────────────────────────────────────────────────────────────────
# RAG index over training documents
# ──────────────────────────────────────────────────────────────────────────────

class RAGIndex:
    """Lightweight document-level retrieval index for in-context learning.

    Indexes training documents by their full text (French or English) and
    retrieves the top-k most similar documents for a query document.
    Used to provide few-shot examples from the training set.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._embs: Optional[np.ndarray] = None
        self._docs: List[Dict] = []

    def build(self, docs: List[Dict], lang: str = "en") -> None:
        """Build the index from a list of normalised training documents."""
        texts = []
        for doc in docs:
            paras = _doc_paras(doc)
            all_text = " ".join(
                (p.get("para_en") if lang == "en" else p.get("para")) or p.get("para") or ""
                for p in paras
            )[:2000]  # truncate per doc
            texts.append(all_text)
        self._embs = self.embedder.encode(texts)
        self._docs = docs

    def retrieve(self, query_doc: Dict, k: int = 3, lang: str = "en") -> List[Dict]:
        """Return top-k most similar training documents for a query document."""
        if self._embs is None or not self._docs:
            return []
        paras = _doc_paras(query_doc)
        q_text = " ".join(
            (p.get("para_en") if lang == "en" else p.get("para")) or p.get("para") or ""
            for p in paras
        )[:2000]
        q_emb = self.embedder.encode([q_text])[0]
        idx = topk_by_cosine(q_emb, self._embs, k=k)
        return [self._docs[i] for i in idx]

    def format_icl_examples(
        self,
        similar_docs: List[Dict],
        lang: str = "en",
        max_paras_per_doc: int = 4,
    ) -> str:
        """Format retrieved docs as in-context learning examples for prompts."""
        lines = []
        for doc in similar_docs:
            title = doc.get("TITLE") or doc.get("TEXT_ID", "")
            paras = _doc_paras(doc)
            lines.append(f"=== Example from: {title} ===")
            for p in paras[:max_paras_per_doc]:
                text = (p.get("para_en") if lang == "en" else p.get("para")) or p.get("para") or ""
                ptype = p.get("type") or "unknown"
                tags = ", ".join(p.get("tags") or []) or "none"
                lines.append(f"  Paragraph {p['para_number']}: [{ptype}] tags=[{tags}] | {text[:120]}…")
        return "\n".join(lines)
