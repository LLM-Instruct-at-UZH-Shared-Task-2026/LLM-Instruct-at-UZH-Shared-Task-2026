from __future__ import annotations
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from .build_submission import build_doc_skeleton
from .task1 import (
    classify_type, classify_type_batch, classify_type_debate,
    predict_tags, predict_tags_batch, predict_tags_debate,
)
from .task2 import (
    candidate_pairs, predict_relation, predict_relation_batch,
    debate_relation_bilingual, build_doc_outline,
)
from ..embed import topk_by_cosine
from ..tags import load_tag_metadata


def _get_text(para: Dict, lang: str) -> str:
    """Get the best-available text from a paragraph dict."""
    if lang == "en":
        return (para.get("para_en") or para.get("para") or "").strip()
    return (para.get("para") or para.get("para_en") or "").strip()


def _get_both(para: Dict) -> tuple:
    """Return (text_fr, text_en) for bilingual/debate mode."""
    return (
        (para.get("para") or "").strip(),
        (para.get("para_en") or "").strip(),
    )


def _text_field_used(para: Dict, lang: str) -> str:
    """Return which field was effectively used by _get_text()."""
    has_en = bool((para.get("para_en") or "").strip())
    has_fr = bool((para.get("para") or "").strip())
    if lang == "en":
        if has_en:
            return "para_en"
        if has_fr:
            return "para"
    else:
        if has_fr:
            return "para"
        if has_en:
            return "para_en"
    return ""


def run_pipeline(
    docs: List[Dict[str, Any]],
    cfg,
    llm=None,
    embedder=None,
    rag_index=None,
    on_doc_done=None,   # optional callable(doc_result) called after each doc
    trace_enabled: bool = False,
    trace_path: Optional[str] = None,
    trace_summary_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    # ── Config ────────────────────────────────────────────────────────────────
    task1_mode = cfg.get("task1", "mode", default="llm")
    task1_type_mode = cfg.get("task1", "type_mode", default=task1_mode)
    task1_tag_mode = cfg.get("task1", "tag_mode", default=task1_mode)
    task1_lang = cfg.get("task1", "language", default="en")
    k_tag = int(cfg.get("task1", "k_tag_candidates", default=40))
    tag_thr = float(cfg.get("task1", "tag_conf_threshold", default=0.30))
    max_tags = int(cfg.get("task1", "max_tags_per_para", default=8))
    max_tags_per_dimension = int(cfg.get("task1", "max_tags_per_dimension", default=2))

    task2_mode = cfg.get("task2", "mode", default="llm")
    task2_lang = cfg.get("task2", "language", default="en")
    k_cand = int(cfg.get("task2", "k_candidates", default=8))
    window = int(cfg.get("task2", "window", default=2))
    max_edges = int(cfg.get("task2", "max_edges_per_para", default=5))
    rel_conf_threshold = float(cfg.get("task2", "rel_conf_threshold", default=0.40))

    debate_enabled = cfg.get("debate", "enabled", default=False)
    if debate_enabled and isinstance(debate_enabled, str):
        debate_enabled = debate_enabled.lower() == "true"
    debate_apply_to_relations = cfg.get("debate", "apply_to_relations", default=False)
    if debate_apply_to_relations and isinstance(debate_apply_to_relations, str):
        debate_apply_to_relations = debate_apply_to_relations.lower() == "true"
    debate_margin_thr = float(cfg.get("debate", "margin_threshold", default=0.05))
    debate_overlap_thr = float(cfg.get("debate", "overlap_threshold", default=0.70))

    use_llm = (llm is not None) and (
        task1_type_mode == "llm" or task1_tag_mode == "llm" or task2_mode == "llm" or debate_enabled
    )
    use_batch = use_llm and hasattr(llm, "chat_batch") and not debate_enabled

    # ── Tag metadata + embeddings (pre-computed once) ─────────────────────────
    tag_rows = load_tag_metadata(cfg.get("data", "tags_csv_path", default=None))
    tag_embs = None
    if tag_rows and embedder is not None:
        tag_embs = embedder.encode([r["_text"] for r in tag_rows])

    # ── Optional trace logging (JSONL + summary) ─────────────────────────────
    trace_fh = None
    trace_summary: Dict[str, Any] = {
        "task1_lang": task1_lang,
        "task2_lang": task2_lang,
        "debate_enabled": bool(debate_enabled),
        "debate_apply_to_relations": bool(debate_apply_to_relations),
        "documents": 0,
        "paragraphs": 0,
        "language_path": {
            "english_available": 0,
            "english_missing": 0,
            "used_en_task1": 0,
            "used_fr_task1": 0,
            "used_en_task2": 0,
            "used_fr_task2": 0,
        },
        "debate": {
            "paragraphs_debated": 0,
            "paragraphs_not_debated": 0,
            "trigger_counts": Counter(),
            "total_runtime_seconds": 0.0,
            "mean_runtime_seconds": 0.0,
        },
        "relation": {
            "candidate_edges_total": 0,
            "kept_after_threshold_total": 0,
            "kept_after_cap_total": 0,
            "pruned_by_cap_total": 0,
        },
    }

    if trace_enabled:
        if trace_path:
            trace_file = Path(trace_path)
        else:
            trace_file = Path("outputs") / "trace_events.jsonl"
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        trace_fh = trace_file.open("w", encoding="utf-8")

    def _emit(event: str, **payload: Any) -> None:
        if trace_fh is None:
            return
        rec = {"event": event, "ts": time.time(), **payload}
        trace_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    outputs: List[Dict[str, Any]] = []

    for raw_doc in tqdm(docs, desc="docs"):
        doc = build_doc_skeleton(raw_doc)
        doc_id = doc.get("TEXT_ID") or raw_doc.get("TEXT_ID") or ""
        paras = doc["body"]["paragraphs"]
        n = len(paras)
        trace_summary["documents"] += 1
        trace_summary["paragraphs"] += n

        texts_t1 = [_get_text(p, task1_lang) for p in paras]
        texts_t2 = [_get_text(p, task2_lang) for p in paras]
        para_numbers = [p["para_number"] for p in paras]

        if trace_fh is not None:
            _emit(
                "doc_start",
                doc_id=doc_id,
                n_paragraphs=n,
                debate_enabled=bool(debate_enabled),
                debate_apply_to_relations=bool(debate_apply_to_relations),
                task1_lang=task1_lang,
                task2_lang=task2_lang,
            )
            for p in paras:
                has_en = bool((p.get("para_en") or "").strip())
                field_t1 = _text_field_used(p, task1_lang)
                field_t2 = _text_field_used(p, task2_lang)
                if has_en:
                    trace_summary["language_path"]["english_available"] += 1
                else:
                    trace_summary["language_path"]["english_missing"] += 1
                if field_t1 == "para_en":
                    trace_summary["language_path"]["used_en_task1"] += 1
                elif field_t1 == "para":
                    trace_summary["language_path"]["used_fr_task1"] += 1
                if field_t2 == "para_en":
                    trace_summary["language_path"]["used_en_task2"] += 1
                elif field_t2 == "para":
                    trace_summary["language_path"]["used_fr_task2"] += 1
                _emit(
                    "language_path",
                    doc_id=doc_id,
                    para_number=p.get("para_number"),
                    english_available=has_en,
                    task1_field=field_t1,
                    task2_field=field_t2,
                )

        para_top1_margin: List[float] = [0.0] * n
        para_overlap_signal: List[bool] = [False] * n
        para_retrieval_trace: List[Dict[str, Any]] = [{} for _ in range(n)]
        if (
            trace_fh is not None
            and n > 0
            and tag_rows
            and embedder is not None
            and tag_embs is not None
        ):
            try:
                para_embs_for_trace = embedder.encode(texts_t1)
                for i, pe in enumerate(para_embs_for_trace):
                    idx = topk_by_cosine(pe, tag_embs, k=k_tag)
                    cand_tags = [tag_rows[j]["_tag"] for j in idx]
                    sims = [round(float(tag_embs[j] @ pe), 6) for j in idx]
                    dims = [tag_rows[j].get("_dim", "") for j in idx]
                    margin = (float(sims[0] - sims[1]) if len(sims) >= 2 else 1.0)
                    para_top1_margin[i] = margin

                    dim_counts = Counter(dims)
                    overlap_ratio = 0.0
                    if len(dim_counts) >= 2:
                        top2 = dim_counts.most_common(2)
                        overlap_ratio = float(top2[1][1]) / float(max(1, top2[0][1]))
                    overlap_flag = overlap_ratio >= debate_overlap_thr
                    para_overlap_signal[i] = overlap_flag
                    para_retrieval_trace[i] = {
                        "candidates": cand_tags,
                        "scores": sims,
                        "dimensions": dims,
                    }
                    _emit(
                        "tag_retrieval",
                        doc_id=doc_id,
                        para_number=para_numbers[i],
                        retrieved_candidates=cand_tags,
                        similarity_scores=sims,
                        dimensions=dims,
                        top1_margin=round(margin, 6),
                        overlap_ratio=round(overlap_ratio, 6),
                        overlap_signal=overlap_flag,
                    )
            except Exception as exc:
                _emit("trace_warning", doc_id=doc_id, stage="tag_retrieval", error=str(exc))

        # Bilingual texts for debate mode
        if debate_enabled:
            bilingual = [_get_both(p) for p in paras]
            texts_fr = [b[0] for b in bilingual]
            texts_en = [b[1] for b in bilingual]

        # RAG context (per-paragraph retrieval - injected into tag prompts)
        rag_note = ""
        para_rag_hits: List[List[Dict]] = [[] for _ in range(n)]
        if rag_index is not None:
            try:
                # Batch-encode all texts at once (single BERT forward pass)
                batch_hits = rag_index.retrieve_batch(texts_t1, k=3, min_score=0.70)
                para_rag_hits = batch_hits
                if any(para_rag_hits):
                    rag_note = f"[RAG] examples loaded for {sum(1 for h in para_rag_hits if h)} paras"
            except Exception:
                pass

        # ── Task 1: Type (batch or debate or sequential) ──────────────────────
        if debate_enabled and llm is not None:
            # CDA: run bilingual agents + referee per paragraph
            type_results = []
            type_runtimes = [0.0] * n
            for i in range(n):
                t0 = time.perf_counter()
                type_results.append(classify_type_debate(texts_fr[i], texts_en[i], llm))
                type_runtimes[i] = time.perf_counter() - t0
        elif use_batch and task1_type_mode == "llm":
            type_results = classify_type_batch(texts_t1, task1_lang, llm)
            type_runtimes = [0.0] * n
        else:
            type_results = [
                classify_type(t, lang=task1_lang, mode=task1_type_mode, llm=llm)
                for t in texts_t1
            ]
            type_runtimes = [0.0] * n

        pre_idx, op_idx, doc_think_lines = [], [], []
        for i, p in enumerate(paras):
            t, conf, thk = type_results[i]
            p["type"] = t
            p["think"] = thk
            (pre_idx if t == "preambular" else op_idx).append(p["para_number"])
            doc_think_lines.append(f"[{p['para_number']}] {t} (conf={conf:.2f})")

        doc["METADATA"]["structure"]["preambular_para"] = pre_idx
        doc["METADATA"]["structure"]["operative_para"] = op_idx

        # ── Task 1: Tags (batch or debate or sequential) ──────────────────────
        if debate_enabled and llm is not None and tag_rows:
            tag_results = []
            tag_runtimes = [0.0] * n
            for i in range(n):
                t0 = time.perf_counter()
                tag_results.append(
                    predict_tags_debate(
                        texts_fr[i], texts_en[i], tag_rows, llm,
                        embedder=embedder,
                        k_candidates=k_tag,
                        conf_threshold=tag_thr,
                        max_tags=max_tags,
                        tag_embs=tag_embs,
                    )
                )
                tag_runtimes[i] = time.perf_counter() - t0
        elif use_batch and task1_tag_mode == "llm" and tag_rows:
            tag_results = predict_tags_batch(
                texts_t1, tag_rows, llm,
                embedder=embedder,
                k_candidates=k_tag,
                conf_threshold=tag_thr,
                max_tags=max_tags,
                max_tags_per_dimension=max_tags_per_dimension,
                tag_embs=tag_embs,
                rag_examples_per_para=para_rag_hits,
            )
            tag_runtimes = [0.0] * n
        else:
            tag_results = [
                predict_tags(
                    t, tag_rows=tag_rows, mode=task1_tag_mode,
                    embedder=embedder, llm=llm,
                    k_candidates=k_tag, conf_threshold=tag_thr,
                    max_tags=max_tags, max_tags_per_dimension=max_tags_per_dimension,
                    tag_embs=tag_embs,
                    rag_examples=para_rag_hits[i],
                )
                for i, t in enumerate(texts_t1)
            ]
            tag_runtimes = [0.0] * n

        for i, p in enumerate(paras):
            tags, tthk = tag_results[i]
            p["tags"] = tags
            if tthk:
                p["think"] = (p["think"] + " | tags: " + tthk).strip(" |")

            if trace_fh is not None:
                heuristic_t, _, _ = classify_type(
                    texts_t1[i],
                    lang=task1_lang,
                    mode="heuristic",
                    llm=None,
                )
                type_disagree = heuristic_t != p.get("type")
                debate_used = bool(debate_enabled and llm is not None)
                trigger_reasons: List[str] = []
                if para_top1_margin[i] < debate_margin_thr:
                    trigger_reasons.append("low_top1_margin")
                if type_disagree:
                    trigger_reasons.append("type_disagreement")
                if para_overlap_signal[i]:
                    trigger_reasons.append("dimension_overlap")
                if debate_used:
                    trigger_reasons.append("debate_enabled_global")
                    trace_summary["debate"]["paragraphs_debated"] += 1
                else:
                    trace_summary["debate"]["paragraphs_not_debated"] += 1
                runtime_sec = float(type_runtimes[i] + tag_runtimes[i]) if debate_used else 0.0
                trace_summary["debate"]["total_runtime_seconds"] += runtime_sec
                for reason in trigger_reasons:
                    trace_summary["debate"]["trigger_counts"][reason] += 1

                _emit(
                    "tag_decision",
                    doc_id=doc_id,
                    para_number=para_numbers[i],
                    retrieved_candidates=para_retrieval_trace[i].get("candidates", []),
                    similarity_scores=para_retrieval_trace[i].get("scores", []),
                    raw_selected_tags=None,
                    tag_scores=None,
                    post_cap_tags=tags,
                    post_closed_set_tags=tags,
                )
                _emit(
                    "debate_paragraph",
                    doc_id=doc_id,
                    para_number=para_numbers[i],
                    trigger_reasons=trigger_reasons,
                    top1_margin=round(float(para_top1_margin[i]), 6),
                    type_disagreement=type_disagree,
                    overlap_signal=bool(para_overlap_signal[i]),
                    debate_used=debate_used,
                    runtime_seconds=round(runtime_sec, 6),
                    pre_debate_prediction=None,
                    post_debate_prediction={"type": p.get("type"), "tags": tags},
                )

        doc["METADATA"]["structure"]["think"] = (
            "; ".join(doc_think_lines) + (f" | {rag_note}" if rag_note else "")
        )[:4000]

        # ── Task 2: Relations (batch or debate or sequential) ─────────────────
        if n == 0:
            outputs.append(doc)
            if on_doc_done:
                on_doc_done(doc)
            continue

        doc_outline = build_doc_outline(texts_t2, para_numbers)
        pair_map = candidate_pairs(texts_t2, embedder=embedder, k=k_cand, window=window)

        if trace_fh is not None:
            for i, cands in pair_map.items():
                cand_nums = [para_numbers[j] for j in cands]
                trace_summary["relation"]["candidate_edges_total"] += len(cand_nums)
                _emit(
                    "relation_candidates",
                    doc_id=doc_id,
                    source_para=para_numbers[i],
                    candidate_list_before_pruning=cand_nums,
                )

        if debate_enabled and llm is not None and debate_apply_to_relations:
            # CDA: bilingual debate per pair
            from collections import defaultdict
            edges_by_i: Dict[int, List] = defaultdict(list)
            for i, cands in pair_map.items():
                for j in cands:
                    t0 = time.perf_counter()
                    rel, conf, rthk = debate_relation_bilingual(
                        texts_fr[i], texts_en[i],
                        texts_fr[j], texts_en[j],
                        llm,
                        doc_outline=doc_outline,
                        a_idx=para_numbers[i],
                        b_idx=para_numbers[j],
                    )
                    runtime_sec = time.perf_counter() - t0
                    if trace_fh is not None:
                        _emit(
                            "relation_score",
                            doc_id=doc_id,
                            source_para=para_numbers[i],
                            target_para=para_numbers[j],
                            predicted_relation=rel,
                            confidence=round(float(conf), 6),
                            debate_used=True,
                            runtime_seconds=round(runtime_sec, 6),
                        )
                    if rel and conf >= rel_conf_threshold:
                        edges_by_i[i].append((j, para_numbers[j], rel, conf, rthk))
            for i, edges in edges_by_i.items():
                kept_threshold = len(edges)
                trace_summary["relation"]["kept_after_threshold_total"] += kept_threshold
                edges.sort(key=lambda x: -x[3])
                kept = edges[:max_edges]
                pruned = max(0, len(edges) - len(kept))
                trace_summary["relation"]["kept_after_cap_total"] += len(kept)
                trace_summary["relation"]["pruned_by_cap_total"] += pruned
                if trace_fh is not None:
                    _emit(
                        "relation_keep",
                        doc_id=doc_id,
                        source_para=para_numbers[i],
                        kept_edges_after_threshold=[
                            {"target_para": b_num, "relation": rel if isinstance(rel, str) else (rel[0] if rel else ""), "confidence": round(float(conf), 6)}
                            for _, b_num, rel, conf, _ in edges
                        ],
                        kept_edges_after_cap=[
                            {"target_para": b_num, "relation": rel if isinstance(rel, str) else (rel[0] if rel else ""), "confidence": round(float(conf), 6)}
                            for _, b_num, rel, conf, _ in kept
                        ],
                        pruned_edges_by_cap=pruned,
                    )
                for j, b_num, rel, conf, rthk in kept:
                    doc["body"]["paragraphs"][i]["matched_pars"][str(b_num)] = (
                        rel if len(rel) > 1 else rel[0]
                    )
                    if rthk:
                        existing = doc["body"]["paragraphs"][i]["think"]
                        doc["body"]["paragraphs"][i]["think"] = (
                            existing + f" | rel→{b_num}:{rthk}"
                        ).strip(" |")[:4000]
        elif use_batch and task2_mode == "llm":
            # Flatten all pairs → one batch call
            flat_pairs = []
            flat_idx = []   # (i, j) so we can put results back
            for i, cands in pair_map.items():
                for j in cands:
                    flat_pairs.append((texts_t2[i], texts_t2[j], para_numbers[i], para_numbers[j]))
                    flat_idx.append((i, j))

            rel_results = predict_relation_batch(
                flat_pairs, mode=task2_mode, llm=llm, doc_outline=doc_outline
            )

            # Re-assemble: group by source para i, keep top max_edges
            from collections import defaultdict
            edges_by_i: Dict[int, List] = defaultdict(list)
            for (i, j), (rel, conf, rthk) in zip(flat_idx, rel_results):
                if trace_fh is not None:
                    _emit(
                        "relation_score",
                        doc_id=doc_id,
                        source_para=para_numbers[i],
                        target_para=para_numbers[j],
                        predicted_relation=rel,
                        confidence=round(float(conf), 6),
                        debate_used=False,
                        runtime_seconds=0.0,
                    )
                if rel and conf >= rel_conf_threshold:
                    edges_by_i[i].append((j, para_numbers[j], rel, conf, rthk))

            for i, edges in edges_by_i.items():
                kept_threshold = len(edges)
                trace_summary["relation"]["kept_after_threshold_total"] += kept_threshold
                edges.sort(key=lambda x: -x[3])
                kept = edges[:max_edges]
                pruned = max(0, len(edges) - len(kept))
                trace_summary["relation"]["kept_after_cap_total"] += len(kept)
                trace_summary["relation"]["pruned_by_cap_total"] += pruned
                if trace_fh is not None:
                    _emit(
                        "relation_keep",
                        doc_id=doc_id,
                        source_para=para_numbers[i],
                        kept_edges_after_threshold=[
                            {"target_para": b_num, "relation": rel if isinstance(rel, str) else (rel[0] if rel else ""), "confidence": round(float(conf), 6)}
                            for _, b_num, rel, conf, _ in edges
                        ],
                        kept_edges_after_cap=[
                            {"target_para": b_num, "relation": rel if isinstance(rel, str) else (rel[0] if rel else ""), "confidence": round(float(conf), 6)}
                            for _, b_num, rel, conf, _ in kept
                        ],
                        pruned_edges_by_cap=pruned,
                    )
                for j, b_num, rel, conf, rthk in kept:
                    doc["body"]["paragraphs"][i]["matched_pars"][str(b_num)] = (
                        rel if len(rel) > 1 else rel[0]
                    )
                    if rthk:
                        existing = doc["body"]["paragraphs"][i]["think"]
                        doc["body"]["paragraphs"][i]["think"] = (
                            existing + f" | rel→{b_num}:{rthk}"
                        ).strip(" |")[:4000]
        else:
            # Sequential heuristic / debate path
            for i, cands in pair_map.items():
                if not cands:
                    continue
                a = texts_t2[i]
                a_num = para_numbers[i]
                edges = []
                for j in cands:
                    b = texts_t2[j]
                    b_num = para_numbers[j]
                    rel, conf, rthk = predict_relation(
                        a, b, mode=task2_mode, llm=llm,
                        a_idx=a_num, b_idx=b_num, doc_outline=doc_outline,
                    )
                    if trace_fh is not None:
                        _emit(
                            "relation_score",
                            doc_id=doc_id,
                            source_para=a_num,
                            target_para=b_num,
                            predicted_relation=rel,
                            confidence=round(float(conf), 6),
                            debate_used=False,
                            runtime_seconds=0.0,
                        )
                    if rel and conf >= rel_conf_threshold:
                        edges.append((j, b_num, rel, conf, rthk))
                trace_summary["relation"]["kept_after_threshold_total"] += len(edges)
                edges.sort(key=lambda x: -x[3])
                kept = edges[:max_edges]
                pruned = max(0, len(edges) - len(kept))
                trace_summary["relation"]["kept_after_cap_total"] += len(kept)
                trace_summary["relation"]["pruned_by_cap_total"] += pruned
                if trace_fh is not None:
                    _emit(
                        "relation_keep",
                        doc_id=doc_id,
                        source_para=a_num,
                        kept_edges_after_threshold=[
                            {"target_para": b_num, "relation": rel if isinstance(rel, str) else (rel[0] if rel else ""), "confidence": round(float(conf), 6)}
                            for _, b_num, rel, conf, _ in edges
                        ],
                        kept_edges_after_cap=[
                            {"target_para": b_num, "relation": rel if isinstance(rel, str) else (rel[0] if rel else ""), "confidence": round(float(conf), 6)}
                            for _, b_num, rel, conf, _ in kept
                        ],
                        pruned_edges_by_cap=pruned,
                    )
                for j, b_num, rel, conf, rthk in kept:
                    doc["body"]["paragraphs"][i]["matched_pars"][str(b_num)] = (
                        rel if len(rel) > 1 else rel[0]
                    )
                    if rthk:
                        existing = doc["body"]["paragraphs"][i]["think"]
                        doc["body"]["paragraphs"][i]["think"] = (
                            existing + f" | rel→{b_num}:{rthk}"
                        ).strip(" |")[:4000]

        outputs.append(doc)
        if trace_fh is not None:
            _emit("doc_done", doc_id=doc_id, n_paragraphs=n)
        if on_doc_done:
            on_doc_done(doc)

    if trace_fh is not None:
        debated = trace_summary["debate"]["paragraphs_debated"]
        if debated > 0:
            trace_summary["debate"]["mean_runtime_seconds"] = round(
                trace_summary["debate"]["total_runtime_seconds"] / debated,
                6,
            )
        trace_summary["debate"]["total_runtime_seconds"] = round(
            trace_summary["debate"]["total_runtime_seconds"],
            6,
        )
        trace_summary["debate"]["trigger_counts"] = dict(
            sorted(trace_summary["debate"]["trigger_counts"].items())
        )
        trace_fh.close()

    if trace_enabled and trace_summary_path:
        p = Path(trace_summary_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(trace_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return outputs

