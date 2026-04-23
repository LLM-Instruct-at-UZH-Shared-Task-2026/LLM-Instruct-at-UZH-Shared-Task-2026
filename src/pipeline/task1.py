from __future__ import annotations
from typing import Any, Dict, List, Tuple
import json
import re

from ..heuristics import classify_type_heuristic
from ..prompts import (
    SYSTEM_MSG,
    TYPE_PROMPT, TAG_DIM_PROMPT, TAGS_PROMPT,
    TYPE_PROMPT_BILINGUAL, TAG_DIM_PROMPT_BILINGUAL, TAGS_PROMPT_BILINGUAL,
    TYPE_REFEREE_PROMPT, TAGS_REFEREE_PROMPT,
)
from ..embed import topk_by_cosine
from .repair import parse_with_repair, parse_without_repair


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _parse_json(s: str) -> Dict[str, Any]:
    """Extract and parse the first JSON object found in a string."""
    s = s.strip()
    # Strip optional markdown fences
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        s = m.group(0)
    return json.loads(s)


_JSON_REPAIR_RETRIES = 3  # override via global config if needed


def _call_llm(llm, user_msg: str, max_new_tokens: int = 512, temperature: float = 0.2) -> str:
    """Use chat() if available, fall back to generate()."""
    if hasattr(llm, "chat"):
        return llm.chat(user_msg, system_msg=SYSTEM_MSG, max_new_tokens=max_new_tokens, temperature=temperature)
    return llm.generate(user_msg, max_new_tokens=max_new_tokens, temperature=temperature)


# ──────────────────────────────────────────────────────────────────────────────
# Task 1a: Type classification
# ──────────────────────────────────────────────────────────────────────────────

def classify_type(text: str, lang: str, mode: str, llm=None) -> Tuple[str, float, str]:
    """Classify paragraph as 'preambular' or 'operative'.

    Returns (type_str, confidence, think).
    """
    if mode == "heuristic" or llm is None:
        return classify_type_heuristic(text, lang)

    user_msg = TYPE_PROMPT.format(para=text[:2000])  # truncate very long paras
    out = _call_llm(llm, user_msg, max_new_tokens=700, temperature=0.1)
    try:
        j = parse_with_repair(out, user_msg, llm, max_retries=_JSON_REPAIR_RETRIES)
        t = str(j.get("type", "")).lower().strip()
        conf = float(j.get("confidence", 0.0))
        think = str(j.get("think", ""))
        if t not in ("preambular", "operative"):
            raise ValueError(f"Unexpected type value: {t!r}")
        return t, max(0.0, min(1.0, conf)), think
    except Exception as exc:
        t, conf, think = classify_type_heuristic(text, lang)
        return t, conf * 0.9, f"{think} | [LLM fallback: {exc}]"


# ──────────────────────────────────────────────────────────────────────────────
# Batched Task 1a  (single GPU pass for all paragraphs in a document)
# ──────────────────────────────────────────────────────────────────────────────

def classify_type_batch(
    texts: List[str],
    lang: str,
    llm,
) -> List[Tuple[str, float, str]]:
    """Classify all paragraph types in a single batch LLM call."""
    prompts = [TYPE_PROMPT.format(para=t[:2000]) for t in texts]
    outputs = llm.chat_batch(prompts, system_msg=SYSTEM_MSG, max_new_tokens=700, temperature=0.1)
    results = []
    for text, out in zip(texts, outputs):
        try:
            j = parse_without_repair(out)
            t = str(j.get("type", "")).lower().strip()
            conf = float(j.get("confidence", 0.0))
            think = str(j.get("think", ""))
            if t not in ("preambular", "operative"):
                raise ValueError(f"Unexpected type: {t!r}")
            results.append((t, max(0.0, min(1.0, conf)), think))
        except Exception as exc:
            t_h, conf_h, think_h = classify_type_heuristic(text, lang)
            results.append((t_h, conf_h * 0.9, f"{think_h} | [LLM fallback: {exc}]"))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Task 1b: Multi-label tag prediction (two-stage)
# ──────────────────────────────────────────────────────────────────────────────

def _group_by_dimension(tag_rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """Group tag rows by dimension string."""
    groups: Dict[str, List[Dict[str, str]]] = {}
    for r in tag_rows:
        dim = r.get("_dim", "Unknown")
        groups.setdefault(dim, []).append(r)
    return groups


def _apply_tag_caps(
    tags_scored: List[Tuple[str, float]],
    tag_to_dim: Dict[str, str],
    max_tags: int,
    max_per_dim: int,
) -> List[str]:
    """Select top tags while preventing one dimension from dominating output."""
    selected: List[str] = []
    dim_counts: Dict[str, int] = {}

    for tag, _conf in tags_scored:
        dim = tag_to_dim.get(tag, "")
        if dim and dim_counts.get(dim, 0) >= max_per_dim:
            continue
        selected.append(tag)
        if dim:
            dim_counts[dim] = dim_counts.get(dim, 0) + 1
        if len(selected) >= max_tags:
            break
    return selected


def predict_tags(
    text: str,
    tag_rows: List[Dict[str, str]],
    mode: str,
    embedder=None,
    llm=None,
    k_candidates: int = 24,
    conf_threshold: float = 0.35,
    max_tags: int = 8,
    max_tags_per_dimension: int = 2,
    tag_embs=None,  # pre-computed embeddings for speed
    rag_examples: List[Dict] = None,  # few-shot examples from RAG
) -> Tuple[List[str], str]:
    """Predict multi-label tags via two-stage retrieval + LLM selection.

    Stage 1 (embedding): retrieve top-k candidate tags by cosine similarity.
    Stage 2 (LLM, optional): select from candidates with per-tag confidence.
    Falls back gracefully if embedder or LLM is absent.

    Returns (list_of_tag_codes, think_str).
    """
    if not tag_rows:
        return [], "No tag metadata available; tags left empty."

    # ── Stage 1: Embedding retrieval ─────────────────────────────────────────
    if embedder is None:
        if mode == "heuristic" or llm is None:
            return [], "No embedder; tags left empty."
        # Pure LLM without embedding — skip to stage 2 with all tags
        candidates: List[str] = [r["_tag"] for r in tag_rows[:k_candidates]]
        candidate_rows = tag_rows[:k_candidates]
    else:
        cand_texts = [r["_text"] for r in tag_rows]
        para_emb = embedder.encode([text])[0]
        if tag_embs is None:
            tag_embs_local = embedder.encode(cand_texts)
        else:
            tag_embs_local = tag_embs
        idx = topk_by_cosine(para_emb, tag_embs_local, k=k_candidates)
        candidates = [tag_rows[i]["_tag"] for i in idx]
        candidate_rows = [tag_rows[i] for i in idx]

    if mode == "heuristic" or llm is None:
        # Return top-5 candidates directly
        return candidates[: min(max_tags, 5)], "Heuristic: top embedding candidates."

    # ── Stage 1b (optional): Narrow by dimension first ───────────────────────
    # Ask LLM which dimensions are relevant, then restrict candidate rows
    all_dims = sorted({r.get("_dim", "") for r in tag_rows if r.get("_dim")})
    dim_prompt = TAG_DIM_PROMPT.format(
        dimensions="\n".join(f"- {d}" for d in all_dims),
        para=text[:1500],
    )
    try:
        dim_out = _call_llm(llm, dim_prompt, max_new_tokens=700, temperature=0.1)
        dim_j = parse_with_repair(dim_out, dim_prompt, llm, max_retries=2)
        selected_dims = set(str(d).strip() for d in (dim_j.get("dimensions") or []))
        if selected_dims:
            filtered = [r for r in candidate_rows if r.get("_dim", "") in selected_dims]
            if filtered:  # only restrict if we got meaningful results
                candidate_rows = filtered
                candidates = [r["_tag"] for r in candidate_rows]
    except Exception:
        pass  # keep original candidates

    # ── Stage 2: LLM selection from candidates ────────────────────────────────
    tag_to_dim = {r["_tag"]: r.get("_dim", "") for r in tag_rows}
    cand_lines = "\n".join(
        f"- {r['_tag']} | {r.get('_cat', '')} | {r.get('_dim', '')}"
        for r in candidate_rows
    )
    # Format RAG few-shot examples if provided
    examples_block = ""
    if rag_examples:
        ex_parts = []
        for ex in rag_examples[:3]:
            ex_text = (ex.get("text_en") or ex.get("text", ""))[:400]
            ex_tags = ex.get("tags", [])
            if ex_text and ex_tags:
                ex_parts.append(f'Example paragraph: "{ex_text}"\nExample tags: {", ".join(ex_tags)}')
        if ex_parts:
            examples_block = "\n\nSIMILAR PARAGRAPHS FOR REFERENCE:\n" + "\n---\n".join(ex_parts) + "\n"
    tags_prompt = TAGS_PROMPT.format(para=text[:1500], candidates=cand_lines, examples_block=examples_block)
    out = _call_llm(llm, tags_prompt, max_new_tokens=1000, temperature=0.2)
    try:
        j = parse_with_repair(out, tags_prompt, llm, max_retries=_JSON_REPAIR_RETRIES)
        tags_scored: List[Tuple[str, float]] = []
        for item in (j.get("tags") or []):
            tag = str(item.get("tag", "")).strip()
            conf = float(item.get("confidence", 0.0))
            # Validate tag is in candidates (no hallucinated tags)
            if tag in candidates and conf >= conf_threshold:
                tags_scored.append((tag, conf))
        tags_scored.sort(key=lambda x: -x[1])
        tags = _apply_tag_caps(
            tags_scored,
            tag_to_dim=tag_to_dim,
            max_tags=max_tags,
            max_per_dim=max_tags_per_dimension,
        )
        think = str(j.get("think", ""))
        return tags, think
    except Exception as exc:
        return candidates[: min(max_tags, 5)], f"LLM parse failed ({exc}) → embedding fallback."


# ──────────────────────────────────────────────────────────────────────────────
# Batched Task 1b  (all paragraphs processed together)
# ──────────────────────────────────────────────────────────────────────────────

def predict_tags_batch(
    texts: List[str],
    tag_rows: List[Dict[str, str]],
    llm,
    embedder=None,
    k_candidates: int = 20,
    conf_threshold: float = 0.50,
    max_tags: int = 4,
    max_tags_per_dimension: int = 2,
    tag_embs=None,
    rag_examples_per_para: List[List[Dict]] | None = None,
) -> List[Tuple[List[str], str]]:
    """Batch-predict tags for all paragraphs in a document.

    Stage 1: embedding retrieval done in a single encode() call.
    Stage 2: single chat_batch() call for all tag-selection prompts.
    """
    if not tag_rows:
        return [([], "No tag metadata.")] * len(texts)

    # ── Stage 1: bulk embedding retrieval ─────────────────────────────────
    if embedder is not None:
        para_embs = embedder.encode(texts)
        if tag_embs is None:
            tag_embs = embedder.encode([r["_text"] for r in tag_rows])
        all_cand_rows = []
        for pe in para_embs:
            idx = topk_by_cosine(pe, tag_embs, k=k_candidates)
            all_cand_rows.append([tag_rows[i] for i in idx])
    else:
        all_cand_rows = [tag_rows[:k_candidates]] * len(texts)

    # ── Stage 1b: batch dimension narrowing ───────────────────────────────
    all_dims = sorted({r.get("_dim", "") for r in tag_rows if r.get("_dim")})
    dim_prompts = [
        TAG_DIM_PROMPT.format(
            dimensions="\n".join(f"- {d}" for d in all_dims),
            para=text[:1500],
        )
        for text in texts
    ]
    dim_outputs = llm.chat_batch(dim_prompts, system_msg=SYSTEM_MSG, max_new_tokens=700, temperature=0.1)

    filtered_cand_rows = []
    for cand_rows, dim_out in zip(all_cand_rows, dim_outputs):
        try:
            dim_j = parse_without_repair(dim_out)
            selected_dims = set(str(d).strip() for d in (dim_j.get("dimensions") or []))
            if selected_dims:
                narrowed = [r for r in cand_rows if r.get("_dim", "") in selected_dims]
                filtered_cand_rows.append(narrowed if narrowed else cand_rows)
            else:
                filtered_cand_rows.append(cand_rows)
        except Exception:
            filtered_cand_rows.append(cand_rows)

    # ── Stage 2: build prompts → single batch LLM call ───────────────────
    prompts = []
    for i, (text, cand_rows) in enumerate(zip(texts, filtered_cand_rows)):
        cand_lines = "\n".join(
            f"- {r['_tag']} | {r.get('_cat', '')} | {r.get('_dim', '')}"
            for r in cand_rows
        )
        examples_block = ""
        if rag_examples_per_para and i < len(rag_examples_per_para):
            rag_examples = rag_examples_per_para[i] or []
            ex_parts = []
            for ex in rag_examples[:3]:
                ex_text = (ex.get("text_en") or ex.get("text", ""))[:400]
                ex_tags = ex.get("tags", [])
                if ex_text and ex_tags:
                    ex_parts.append(f'Example paragraph: "{ex_text}"\nExample tags: {", ".join(ex_tags)}')
            if ex_parts:
                examples_block = "\n\nSIMILAR PARAGRAPHS FOR REFERENCE:\n" + "\n---\n".join(ex_parts) + "\n"
        prompts.append(TAGS_PROMPT.format(para=text[:1500], candidates=cand_lines, examples_block=examples_block))

    outputs = llm.chat_batch(prompts, system_msg=SYSTEM_MSG, max_new_tokens=1000, temperature=0.2)

    # ── Parse ─────────────────────────────────────────────────────────────
    results = []
    tag_to_dim = {r["_tag"]: r.get("_dim", "") for r in tag_rows}
    for out, cand_rows in zip(outputs, filtered_cand_rows):
        cand_set = {r["_tag"] for r in cand_rows}
        try:
            j = parse_without_repair(out)
            tags_scored: List[Tuple[str, float]] = []
            for item in (j.get("tags") or []):
                tag = str(item.get("tag", "")).strip()
                conf = float(item.get("confidence", 0.0))
                if tag in cand_set and conf >= conf_threshold:
                    tags_scored.append((tag, conf))
            tags_scored.sort(key=lambda x: -x[1])
            tags = _apply_tag_caps(
                tags_scored,
                tag_to_dim=tag_to_dim,
                max_tags=max_tags,
                max_per_dim=max_tags_per_dimension,
            )
            think = str(j.get("think", ""))
            results.append((tags, think))
        except Exception as exc:
            fallback = [r["_tag"] for r in cand_rows[:min(max_tags, 3)]]
            results.append((fallback, f"parse failed ({exc}) → embedding fallback"))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Cross-lingual Deliberative Alignment — Task 1a (type)
# ──────────────────────────────────────────────────────────────────────────────

def classify_type_debate(
    text_fr: str,
    text_en: str,
    llm,
) -> Tuple[str, float, str]:
    """Bilingual debate for type classification.

    Agent FR analyses the French paragraph; Agent EN analyses the English one.
    If they agree, return with boosted confidence.  If they disagree, a
    Referee (bilingual) arbitrates.

    Returns (type_str, confidence, think).
    """
    import json as _json

    lang_map = [("French", text_fr), ("English", text_en)]
    proposals = []

    for lang_name, para_text in lang_map:
        if not para_text.strip():
            continue
        prompt = TYPE_PROMPT_BILINGUAL.format(language=lang_name, para=para_text[:2000])
        out = _call_llm(llm, prompt, max_new_tokens=700, temperature=0.2)
        try:
            j = parse_with_repair(out, prompt, llm, max_retries=_JSON_REPAIR_RETRIES)
            t = str(j.get("type", "")).lower().strip()
            conf = float(j.get("confidence", 0.0))
            think = str(j.get("think", ""))
            if t not in ("preambular", "operative"):
                raise ValueError(f"Invalid type: {t!r}")
            proposals.append({"lang": lang_name, "type": t, "confidence": conf, "think": think})
        except Exception as exc:
            proposals.append({
                "lang": lang_name,
                "type": None,
                "confidence": 0.0,
                "think": f"Agent parse error: {exc}",
            })

    valid = [p for p in proposals if p["type"] is not None]
    if not valid:
        return classify_type_heuristic(text_fr or text_en, "fr")

    types = [p["type"] for p in valid]

    # Both agreed → high confidence
    if len(set(types)) == 1 and len(valid) == 2:
        avg_conf = sum(p["confidence"] for p in valid) / len(valid)
        combined_think = " | ".join(f"[{p['lang']}] {p['think']}" for p in valid)
        return types[0], min(1.0, avg_conf * 1.05), combined_think  # slight boost

    if len(valid) == 1:
        p = valid[0]
        return p["type"], p["confidence"], p["think"]

    # Disagreement → Referee
    ref_msg = TYPE_REFEREE_PROMPT.format(
        para_fr=text_fr[:1500],
        para_en=text_en[:1500],
        proposals=_json.dumps(proposals, ensure_ascii=False, indent=2),
    )
    out = _call_llm(llm, ref_msg, max_new_tokens=700, temperature=0.1)
    try:
        j = parse_with_repair(out, ref_msg, llm, max_retries=_JSON_REPAIR_RETRIES)
        t = str(j.get("type", "")).lower().strip()
        conf = float(j.get("confidence", 0.0))
        think = str(j.get("think", ""))
        if t not in ("preambular", "operative"):
            raise ValueError(f"Referee invalid type: {t!r}")
        return t, max(0.0, min(1.0, conf)), f"[Referee] {think}"
    except Exception as exc:
        majority = max(set(types), key=types.count)
        avg_conf = (sum(p["confidence"] for p in valid if p["type"] == majority)
                    / types.count(majority))
        return majority, avg_conf * 0.85, f"[Debate majority vote | referee failed: {exc}]"


# ──────────────────────────────────────────────────────────────────────────────
# Cross-lingual Deliberative Alignment — Task 1b (tags)
# ──────────────────────────────────────────────────────────────────────────────

def predict_tags_debate(
    text_fr: str,
    text_en: str,
    tag_rows: List[Dict[str, str]],
    llm,
    embedder=None,
    k_candidates: int = 40,
    conf_threshold: float = 0.30,
    max_tags: int = 8,
    tag_embs=None,
) -> Tuple[List[str], str]:
    """Bilingual debate for tag prediction.

    Each language-agent retrieves candidates independently and predicts tags.
    A referee merges and validates with the full valid candidate set.

    Returns (list_of_tag_codes, think_str).
    """
    import json as _json

    if not tag_rows:
        return [], "No tag metadata."

    all_tag_codes: set = {r["_tag"] for r in tag_rows}
    all_dims = sorted({r.get("_dim", "") for r in tag_rows if r.get("_dim")})
    proposals = []

    for lang_name, para_text in [("French", text_fr), ("English", text_en)]:
        if not para_text.strip():
            continue

        # Embedding retrieval per language
        if embedder is not None:
            cand_texts = [r["_text"] for r in tag_rows]
            para_emb = embedder.encode([para_text])[0]
            tag_embs_local = tag_embs if tag_embs is not None else embedder.encode(cand_texts)
            idx = topk_by_cosine(para_emb, tag_embs_local, k=k_candidates)
            cand_rows = [tag_rows[i] for i in idx]
        else:
            cand_rows = tag_rows[:k_candidates]

        cand_set = {r["_tag"] for r in cand_rows}

        # Dimension filter
        dim_prompt = TAG_DIM_PROMPT_BILINGUAL.format(
            language=lang_name,
            dimensions="\n".join(f"- {d}" for d in all_dims),
            para=para_text[:1500],
        )
        try:
            dim_out = _call_llm(llm, dim_prompt, max_new_tokens=700, temperature=0.1)
            dim_j = parse_with_repair(dim_out, dim_prompt, llm, max_retries=2)
            sel_dims = set(str(d).strip() for d in (dim_j.get("dimensions") or []))
            if sel_dims:
                filtered = [r for r in cand_rows if r.get("_dim", "") in sel_dims]
                if filtered:
                    cand_rows = filtered
                    cand_set = {r["_tag"] for r in cand_rows}
        except Exception:
            pass

        cand_lines = "\n".join(
            f"- {r['_tag']} | {r.get('_cat', '')} | {r.get('_dim', '')}"
            for r in cand_rows
        )
        tag_prompt = TAGS_PROMPT_BILINGUAL.format(
            language=lang_name,
            para=para_text[:1500],
            candidates=cand_lines,
        )
        out = _call_llm(llm, tag_prompt, max_new_tokens=1000, temperature=0.2)
        try:
            j = parse_with_repair(out, tag_prompt, llm, max_retries=_JSON_REPAIR_RETRIES)
            agent_tags = []
            for item in (j.get("tags") or []):
                tag = str(item.get("tag", "")).strip()
                conf = float(item.get("confidence", 0.0))
                if tag in cand_set and conf >= conf_threshold:
                    agent_tags.append({"tag": tag, "confidence": conf,
                                       "why": str(item.get("why", ""))})
            proposals.append({
                "lang": lang_name,
                "tags": agent_tags,
                "think": str(j.get("think", "")),
            })
        except Exception as exc:
            proposals.append({"lang": lang_name, "tags": [], "think": f"parse error: {exc}"})

    # ── Merge via Referee ──────────────────────────────────────────────────
    ref_msg = TAGS_REFEREE_PROMPT.format(
        para_fr=text_fr[:1200],
        para_en=text_en[:1200],
        candidates=", ".join(sorted(all_tag_codes)),
        proposals=_json.dumps(proposals, ensure_ascii=False, indent=2),
    )
    out = _call_llm(llm, ref_msg, max_new_tokens=1000, temperature=0.1)
    try:
        j = parse_with_repair(out, ref_msg, llm, max_retries=_JSON_REPAIR_RETRIES)
        tags_scored: List[Tuple[str, float]] = []
        for item in (j.get("tags") or []):
            tag = str(item.get("tag", "")).strip()
            conf = float(item.get("confidence", 0.0))
            if tag in all_tag_codes and conf >= conf_threshold:
                tags_scored.append((tag, conf))
        tags_scored.sort(key=lambda x: -x[1])
        tags = [t for t, _ in tags_scored][:max_tags]
        think = f"[Debate] {j.get('think', '')}"
        return tags, think
    except Exception as exc:
        # Fallback: union of both agents above conf threshold
        union: Dict[str, float] = {}
        for prop in proposals:
            for item in prop.get("tags", []):
                tag = item["tag"]
                if tag in all_tag_codes:
                    union[tag] = max(union.get(tag, 0.0), item["confidence"])
        tags = sorted(union, key=lambda t: -union[t])[:max_tags]
        return tags, f"[Debate union fallback | referee failed: {exc}]"