from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import json
import re

from ..prompts import (
    SYSTEM_MSG,
    REL_PROMPT, DEBATE_AGENT_PROMPT, DEBATE_REFEREE_PROMPT,
)
from ..embed import topk_by_cosine
from .repair import parse_with_repair, parse_without_repair

REL_SET = {"contradictive", "supporting", "complemental", "modifying"}
_JSON_REPAIR_RETRIES = 3


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _parse_json(s: str) -> Dict[str, Any]:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        s = m.group(0)
    return json.loads(s)


def _call_llm(llm, user_msg: str, max_new_tokens: int = 512, temperature: float = 0.2) -> str:
    if hasattr(llm, "chat"):
        return llm.chat(user_msg, system_msg=SYSTEM_MSG, max_new_tokens=max_new_tokens, temperature=temperature)
    return llm.generate(user_msg, max_new_tokens=max_new_tokens, temperature=temperature)


def build_doc_outline(para_texts: List[str], para_numbers: List[int], max_paras: int = 30) -> str:
    """Build a brief document outline: index + first ~60 chars of each paragraph.

    Used as document context so the LLM understands the overall structure.
    Limits to max_paras to stay within token budget.
    """
    lines = []
    step = max(1, len(para_texts) // max_paras)
    for i in range(0, len(para_texts), step):
        sn = para_numbers[i] if i < len(para_numbers) else i + 1
        preview = para_texts[i][:70].replace("\n", " ")
        lines.append(f"[{sn}] {preview}…")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Candidate pair generation
# ──────────────────────────────────────────────────────────────────────────────

def candidate_pairs(
    paras: List[str],
    embedder,
    k: int = 8,
    window: int = 1,
) -> Dict[int, List[int]]:
    """Generate candidate (i, j) pairs for relation prediction.

    Strategy:
    - Local window ±window (discourse continuity)
    - Top-k by embedding similarity (topic/entity overlap)

    Reduced defaults (k=8, window=1) to avoid O(n²) explosion and
    reduce low-quality predictions from distant pairs.
    """
    n = len(paras)
    embs = embedder.encode(paras) if embedder else None
    pairs: Dict[int, List[int]] = {i: [] for i in range(n)}

    for i in range(n):
        cands: set = set()
        # Local window
        for d in range(-window, window + 1):
            j = i + d
            if 0 <= j < n and j != i:
                cands.add(j)
        # Embedding top-k
        if embs is not None:
            idx = topk_by_cosine(embs[i], embs, k=min(k + 1, n))
            for j in idx:
                if j != i:
                    cands.add(j)
        pairs[i] = sorted(cands)
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Relation prediction
# ──────────────────────────────────────────────────────────────────────────────

def predict_relation(
    a: str,
    b: str,
    mode: str,
    llm=None,
    a_idx: int = 0,
    b_idx: int = 1,
    doc_outline: str = "",
) -> Tuple[List[str], float, str]:
    """Predict the argumentative relation from paragraph A to paragraph B.

    Returns (relation_list, confidence, think).
    Returns [] (empty list) when the model predicts 'none'
    so the pair is omitted from matched_paras.
    """
    if mode == "heuristic" or llm is None:
        low_b = b.lower()
        if any(x in low_b for x in ["furthermore", "moreover", "in addition", "also ", "additionally"]):
            return ["complemental"], 0.55, "Heuristic: additive discourse marker."
        if any(x in low_b for x in ["provided that", "unless", "subject to", "except", "notwithstanding"]):
            return ["modifying"], 0.55, "Heuristic: conditional/exception marker."
        # Consecutive operative paragraphs in the same document tend to be complemental
        return ["complemental"], 0.45, "Heuristic default (adjacent paragraphs)."

    user_msg = REL_PROMPT.format(
        doc_outline=doc_outline or "(not available)",
        a=a[:800],
        b=b[:800],
        a_idx=a_idx,
        b_idx=b_idx,
    )
    out = _call_llm(llm, user_msg, max_new_tokens=800, temperature=0.2)
    try:
        j = parse_with_repair(out, user_msg, llm, max_retries=_JSON_REPAIR_RETRIES)
        rel = j.get("relation") or []
        if isinstance(rel, str):
            rel = [rel]
        # Filter to valid labels; "none" means no relation → return empty
        rel_clean = []
        for r in rel:
            r = str(r).lower().strip()
            if r == "none":
                return [], float(j.get("confidence", 0.5)), str(j.get("think", "No relation predicted."))
            if r in REL_SET:
                rel_clean.append(r)
        conf = max(0.0, min(1.0, float(j.get("confidence", 0.0))))
        think = str(j.get("think", ""))
        if not rel_clean:
            return [], conf, "Empty relation → omitting pair."
        return rel_clean, conf, think
    except Exception as exc:
        return ["complemental"], 0.4, f"LLM parse failed ({exc}) → default complemental."


# ──────────────────────────────────────────────────────────────────────────────
# Debate mode
# ──────────────────────────────────────────────────────────────────────────────

def debate_relation(
    a: str,
    b: str,
    llm,
    agent_names: Tuple[str, ...] = ("Analyst", "Critic"),
    language: str = "English",
) -> Tuple[List[str], float, str]:
    """Two-agent monolingual debate + referee for relation labelling.

    Use debate_relation_bilingual for the CDA bilingual version.
    """
    proposals = []
    for name in agent_names:
        user_msg = DEBATE_AGENT_PROMPT.format(
            agent_name=name, language=language, a=a[:700], b=b[:700]
        )
        out = _call_llm(llm, user_msg, max_new_tokens=800, temperature=0.5)
        try:
            j = parse_with_repair(out, user_msg, llm, max_retries=_JSON_REPAIR_RETRIES)
            prop = j.get("proposed") or []
            if isinstance(prop, str):
                prop = [prop]
            valid = [p.lower() for p in prop if p.lower() in REL_SET or p.lower() == "none"]
            proposals.append({
                "agent": name,
                "proposed": valid,
                "confidence": float(j.get("confidence", 0.0)),
                "argument": str(j.get("argument", "")),
            })
        except Exception:
            proposals.append({"agent": name, "proposed": [], "confidence": 0.0, "argument": "parse failed"})

    ref_msg = DEBATE_REFEREE_PROMPT.format(
        a_fr=a[:700], a_en=a[:700],
        b_fr=b[:700], b_en=b[:700],
        proposals=json.dumps(proposals, ensure_ascii=False, indent=2)
    )
    out = _call_llm(llm, ref_msg, max_new_tokens=800, temperature=0.2)
    try:
        j = parse_with_repair(out, ref_msg, llm, max_retries=_JSON_REPAIR_RETRIES)
        rel = j.get("relation") or []
        if isinstance(rel, str):
            rel = [rel]
        if any(str(r).lower() == "none" for r in rel):
            return [], float(j.get("confidence", 0.5)), "Debate consensus: no relation."
        rel_clean = [r.lower() for r in rel if r.lower() in REL_SET]
        conf = max(0.0, min(1.0, float(j.get("confidence", 0.0))))
        think = str(j.get("think", ""))
        return rel_clean or ["complemental"], conf, think
    except Exception:
        flat = [p for proposal in proposals for p in proposal.get("proposed", []) if p in REL_SET]
        rel = [max(set(flat), key=flat.count)] if flat else ["complemental"]
        return rel, 0.40, "Debate referee parse failed → majority vote fallback."


# ──────────────────────────────────────────────────────────────────────────────
# Batched relation prediction  (all candidate pairs in one GPU pass)
# ──────────────────────────────────────────────────────────────────────────────

def predict_relation_batch(
    pairs: List[Tuple[str, str, int, int]],   # (a_text, b_text, a_num, b_num)
    mode: str,
    llm=None,
    doc_outline: str = "",
) -> List[Tuple[List[str], float, str]]:
    """Predict relations for multiple (A, B) pairs in a single batch call.

    Heuristic mode falls back to per-pair sequential (fast enough).
    """
    if not pairs:
        return []
    if mode == "heuristic" or llm is None:
        return [
            predict_relation(a, b, mode=mode, llm=None, a_idx=ai, b_idx=bi, doc_outline=doc_outline)
            for a, b, ai, bi in pairs
        ]

    prompts = []
    for a, b, ai, bi in pairs:
        prompts.append(REL_PROMPT.format(
            doc_outline=doc_outline or "(not available)",
            a=a[:800],
            b=b[:800],
            a_idx=ai,
            b_idx=bi,
        ))

    outputs = llm.chat_batch(prompts, system_msg=SYSTEM_MSG, max_new_tokens=800, temperature=0.2)

    results = []
    for out in outputs:
        try:
            j = parse_without_repair(out)
            rel = j.get("relation") or []
            if isinstance(rel, str):
                rel = [rel]
            rel_clean = []
            for r in rel:
                r = str(r).lower().strip()
                if r == "none":
                    results.append(([], float(j.get("confidence", 0.5)), "No relation."))
                    break
                if r in REL_SET:
                    rel_clean.append(r)
            else:
                conf = max(0.0, min(1.0, float(j.get("confidence", 0.0))))
                think = str(j.get("think", ""))
                results.append((rel_clean or [], conf, think))
        except Exception as exc:
            results.append((["complemental"], 0.40, f"parse failed ({exc})"))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Cross-lingual Deliberative Alignment — Task 2 (bilingual debate)
# ──────────────────────────────────────────────────────────────────────────────

def debate_relation_bilingual(
    a_fr: str,
    a_en: str,
    b_fr: str,
    b_en: str,
    llm,
    doc_outline: str = "",
    a_idx: int = 0,
    b_idx: int = 1,
) -> Tuple[List[str], float, str]:
    """Cross-lingual Deliberative Alignment for relation prediction.

    Agent FR uses (a_fr, b_fr), Agent EN uses (a_en, b_en).
    Referee sees all four texts and both proposals to make a final decision.

    Returns (relation_list, confidence, think).
    """
    lang_map = [("French", a_fr, b_fr), ("English", a_en, b_en)]
    proposals = []

    for lang_name, a_text, b_text in lang_map:
        if not a_text.strip() or not b_text.strip():
            continue
        agent_prompt = DEBATE_AGENT_PROMPT.format(
            agent_name=f"Agent_{lang_name}",
            language=lang_name,
            a=a_text[:700],
            b=b_text[:700],
        )
        out = _call_llm(llm, agent_prompt, max_new_tokens=800, temperature=0.4)
        try:
            j = parse_with_repair(out, agent_prompt, llm, max_retries=_JSON_REPAIR_RETRIES)
            prop = j.get("proposed") or []
            if isinstance(prop, str):
                prop = [prop]
            valid = [p.lower() for p in prop if p.lower() in REL_SET or p.lower() == "none"]
            proposals.append({
                "agent": f"Agent_{lang_name}",
                "proposed": valid,
                "confidence": float(j.get("confidence", 0.0)),
                "argument": str(j.get("argument", "")),
            })
        except Exception as exc:
            proposals.append({
                "agent": f"Agent_{lang_name}",
                "proposed": [],
                "confidence": 0.0,
                "argument": f"parse failed: {exc}",
            })

    # Referee with all four texts
    ref_msg = DEBATE_REFEREE_PROMPT.format(
        a_fr=a_fr[:700],
        a_en=a_en[:700],
        b_fr=b_fr[:700],
        b_en=b_en[:700],
        proposals=json.dumps(proposals, ensure_ascii=False, indent=2),
    )
    out = _call_llm(llm, ref_msg, max_new_tokens=800, temperature=0.1)
    try:
        j = parse_with_repair(out, ref_msg, llm, max_retries=_JSON_REPAIR_RETRIES)
        rel = j.get("relation") or []
        if isinstance(rel, str):
            rel = [rel]
        if any(str(r).lower() == "none" for r in rel):
            return [], float(j.get("confidence", 0.5)), "[CDA] No relation (referee)."
        rel_clean = [r.lower() for r in rel if r.lower() in REL_SET]
        conf = max(0.0, min(1.0, float(j.get("confidence", 0.0))))
        think = f"[CDA] {j.get('think', '')}"
        return rel_clean or ["complemental"], conf, think
    except Exception as exc:
        # Fallback: majority vote among agent proposals
        flat = [p for pr in proposals for p in pr.get("proposed", []) if p in REL_SET]
        rel = [max(set(flat), key=flat.count)] if flat else ["complemental"]
        return rel, 0.40, f"[CDA majority vote | referee failed: {exc}]"
