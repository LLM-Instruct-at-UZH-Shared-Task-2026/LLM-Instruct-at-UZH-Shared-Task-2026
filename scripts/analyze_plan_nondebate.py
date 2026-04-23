#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import yaml


VALID_RELS = {"supporting", "contradictive", "complemental", "modifying"}
TYPE_LABELS = ("preambular", "operative")


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[Dict], headers: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def normalize_rel(v) -> Optional[str]:
    if isinstance(v, list):
        if not v:
            return None
        v = v[0]
    if not isinstance(v, str):
        return None
    vv = v.strip().lower()
    return vv if vv in VALID_RELS else None


def parse_key(k: str) -> Tuple[str, int]:
    tid, pnum = k.rsplit("|||", 1)
    return tid, int(pnum)


def get_paragraphs(doc: Dict) -> List[Dict]:
    body = doc.get("body") or {}
    paras = body.get("paragraphs") or body.get("paras") or []
    return paras if isinstance(paras, list) else []


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tag2dim(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            code = (row.get("CODE") or "").strip()
            dim = (row.get("Dimensions") or "").strip()
            if not code or code.upper() == "NA":
                continue
            out[code] = dim
    return out


def type_metrics(keys: Iterable[Tuple[str, int]], pred_type: Dict[Tuple[str, int], Optional[str]], ref_type: Dict[Tuple[str, int], Optional[str]]) -> Dict[str, float]:
    keys = list(keys)
    total = 0
    correct = 0
    per_tp = {l: 0 for l in TYPE_LABELS}
    per_fp = {l: 0 for l in TYPE_LABELS}
    per_fn = {l: 0 for l in TYPE_LABELS}

    for k in keys:
        p = pred_type.get(k)
        r = ref_type.get(k)
        if p in TYPE_LABELS and r in TYPE_LABELS:
            total += 1
            if p == r:
                correct += 1
                per_tp[p] += 1
            else:
                per_fp[p] += 1
                per_fn[r] += 1

    per_f1 = []
    for l in TYPE_LABELS:
        prec = safe_div(per_tp[l], per_tp[l] + per_fp[l])
        rec = safe_div(per_tp[l], per_tp[l] + per_fn[l])
        per_f1.append(f1(prec, rec))

    return {
        "n": total,
        "accuracy": safe_div(correct, total),
        "macro_f1": sum(per_f1) / len(per_f1) if per_f1 else 0.0,
    }


def tags_micro_metrics(keys: Iterable[Tuple[str, int]], pred_tags: Dict[Tuple[str, int], Set[str]], ref_tags: Dict[Tuple[str, int], Set[str]], tag_filter=None) -> Dict[str, float]:
    tp = fp = fn = 0
    n = 0
    for k in keys:
        pset = set(pred_tags.get(k, set()))
        rset = set(ref_tags.get(k, set()))
        if tag_filter is not None:
            pset = {t for t in pset if tag_filter(t)}
            rset = {t for t in rset if tag_filter(t)}
        tp += len(pset & rset)
        fp += len(pset - rset)
        fn += len(rset - pset)
        n += 1

    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1(prec, rec),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Non-debate PLAN analysis pack generator")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = Path(args.root).resolve()

    submission_path = root / "final_submission" / "LLM-Instruct_predictions.json"
    phase3_submission_path = root / "outputs" / "runs" / "phase3_tagboost_v1" / "submission.json"
    cfg_path = root / "config.yaml"
    tags_csv_path = root / "dataset" / "education_dimensions_updated.csv"
    gt_tags_path = root / "eval" / "runs" / "phase0_llm_v2_eval" / "ground_truth_tags.json"
    gt_types_path = root / "eval" / "runs" / "phase0_llm_v2_eval" / "ground_truth_types.json"
    phase_reports = {
        "phase0_llm_v2_eval": root / "eval" / "runs" / "phase0_llm_v2_eval" / "report.json",
        "phase1_rag_v1_eval": root / "eval" / "runs" / "phase1_rag_v1_eval" / "report.json",
        "phase2_precision_v1_eval": root / "eval" / "runs" / "phase2_precision_v1_eval" / "report.json",
        "phase3_tagboost_v1": root / "eval" / "runs" / "phase3_tagboost_v1" / "report.json",
    }
    l1_trace_summary = root / "outputs" / "runs" / "nondebate_l1_faststats" / "trace_summary.json"
    l1_repair_stats = root / "outputs" / "runs" / "nondebate_l1_faststats" / "repair_stats.json"

    results_abl = root / "results" / "ablations"
    results_rel = root / "results" / "relation_eval"
    results_lang = root / "results" / "language_split"
    results_rep = root / "results" / "repair_stats"

    submission = read_json(submission_path)
    tag2dim = load_tag2dim(tags_csv_path)
    gt_tags_raw = read_json(gt_tags_path)
    gt_types_raw = read_json(gt_types_path)

    gt_tags: Dict[Tuple[str, int], Set[str]] = {parse_key(k): set(v or []) for k, v in gt_tags_raw.items()}
    gt_types: Dict[Tuple[str, int], Optional[str]] = {parse_key(k): (v if v in TYPE_LABELS else None) for k, v in gt_types_raw.items()}

    pred_tags: Dict[Tuple[str, int], Set[str]] = {}
    pred_types: Dict[Tuple[str, int], Optional[str]] = {}
    english_available: Dict[Tuple[str, int], bool] = {}
    source_edge_counts: Dict[Tuple[str, int], int] = {}

    rel_label_counts = Counter()
    distance_bucket_counts = Counter()
    total_pred_edges = 0
    total_possible_edges = 0
    total_paragraphs = 0
    doc_rows = []

    for doc in submission:
        tid = doc.get("TEXT_ID", "")
        paras = get_paragraphs(doc)
        total_paragraphs += len(paras)
        total_possible_edges += len(paras) * max(0, len(paras) - 1)
        per_doc_edges = 0

        for p in paras:
            pnum = p.get("para_number")
            if not isinstance(pnum, int):
                continue
            key = (tid, pnum)
            ptype = (p.get("type") or "").strip().lower()
            pred_types[key] = ptype if ptype in TYPE_LABELS else None
            pred_tags[key] = set(t for t in (p.get("tags") or []) if isinstance(t, str) and t)
            english_available[key] = bool((p.get("para_en") or "").strip())

            rels = p.get("matched_pars") or p.get("matched_paras") or {}
            if not isinstance(rels, dict):
                rels = {}
            source_edge_counts[key] = len(rels)
            per_doc_edges += len(rels)

            for b_str, rel_v in rels.items():
                try:
                    bnum = int(b_str)
                except Exception:
                    continue
                rel = normalize_rel(rel_v)
                if rel is None:
                    continue
                total_pred_edges += 1
                rel_label_counts[rel] += 1
                dist = abs(bnum - pnum)
                if dist <= 1:
                    distance_bucket_counts["adjacent_1"] += 1
                elif dist <= 3:
                    distance_bucket_counts["near_2_3"] += 1
                else:
                    distance_bucket_counts["long_4_plus"] += 1

        doc_rows.append({
            "TEXT_ID": tid,
            "n_paragraphs": len(paras),
            "pred_edges": per_doc_edges,
            "edge_density_pct": round(100.0 * safe_div(per_doc_edges, len(paras) * max(0, len(paras) - 1)), 4),
        })

    write_csv(
        results_rel / "relation_doc_stats.csv",
        doc_rows,
        ["TEXT_ID", "n_paragraphs", "pred_edges", "edge_density_pct"],
    )

    rel_summary = {
        "documents": len(submission),
        "paragraphs": total_paragraphs,
        "pred_edges": total_pred_edges,
        "possible_edges": total_possible_edges,
        "edge_density_pct": round(100.0 * safe_div(total_pred_edges, total_possible_edges), 4),
        "long_range_edges": int(distance_bucket_counts["near_2_3"] + distance_bucket_counts["long_4_plus"]),
        "long_range_pct": round(100.0 * safe_div(distance_bucket_counts["near_2_3"] + distance_bucket_counts["long_4_plus"], total_pred_edges), 4),
        "label_counts": dict(rel_label_counts),
        "distance_buckets": dict(distance_bucket_counts),
    }
    write_json(results_rel / "relation_output_stats.json", rel_summary)

    rel_rows = []
    for rel in sorted(VALID_RELS):
        rel_rows.append({
            "relation": rel,
            "count": rel_label_counts.get(rel, 0),
            "pct": round(100.0 * safe_div(rel_label_counts.get(rel, 0), total_pred_edges), 4),
        })
    write_csv(results_rel / "relation_label_distribution.csv", rel_rows, ["relation", "count", "pct"])

    dist_rows = []
    for b in ["adjacent_1", "near_2_3", "long_4_plus"]:
        dist_rows.append({
            "distance_bucket": b,
            "count": distance_bucket_counts.get(b, 0),
            "pct": round(100.0 * safe_div(distance_bucket_counts.get(b, 0), total_pred_edges), 4),
        })
    write_csv(results_rel / "relation_distance_buckets.csv", dist_rows, ["distance_bucket", "count", "pct"])

    common_keys = sorted(set(gt_tags) & set(pred_tags))

    dim_rows = []
    for dim in sorted(set(tag2dim.values())):
        m = tags_micro_metrics(common_keys, pred_tags, gt_tags, tag_filter=lambda t, d=dim: tag2dim.get(t) == d)
        dim_rows.append({
            "dimension": dim,
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
        })
    write_csv(results_abl / "dimension_breakdown.csv", dim_rows, ["dimension", "tp", "fp", "fn", "precision", "recall", "f1"])

    ref_tag_freq = Counter()
    for tags in gt_tags.values():
        ref_tag_freq.update(tags)

    def freq_bucket(tag: str) -> str:
        c = ref_tag_freq.get(tag, 0)
        if c <= 5:
            return "rare_le_5"
        if c <= 20:
            return "medium_6_20"
        return "frequent_gt_20"

    bucket_tp = Counter()
    bucket_fp = Counter()
    bucket_fn = Counter()

    fp_tag_counts = Counter()
    cross_dim_pairs = Counter()

    for k in common_keys:
        pset = pred_tags.get(k, set())
        rset = gt_tags.get(k, set())

        for t in pset & rset:
            bucket_tp[freq_bucket(t)] += 1
        for t in pset - rset:
            b = freq_bucket(t)
            bucket_fp[b] += 1
            fp_tag_counts[t] += 1
            fp_dim = tag2dim.get(t, "UNKNOWN")
            ref_dims = {tag2dim.get(rt, "UNKNOWN") for rt in rset} if rset else {"NONE"}
            for rd in ref_dims:
                cross_dim_pairs[(rd, fp_dim)] += 1
        for t in rset - pset:
            bucket_fn[freq_bucket(t)] += 1

    bucket_rows = []
    for b in ["frequent_gt_20", "medium_6_20", "rare_le_5"]:
        tp = bucket_tp[b]
        fp = bucket_fp[b]
        fn = bucket_fn[b]
        p = safe_div(tp, tp + fp)
        r = safe_div(tp, tp + fn)
        bucket_rows.append({
            "bucket": b,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1(p, r), 4),
        })
    write_csv(results_abl / "rare_tag_breakdown.csv", bucket_rows, ["bucket", "tp", "fp", "fn", "precision", "recall", "f1"])

    over_rows = []
    for tag, c in fp_tag_counts.most_common(10):
        over_rows.append({
            "tag": tag,
            "dimension": tag2dim.get(tag, "UNKNOWN"),
            "fp_count": c,
            "ref_frequency": ref_tag_freq.get(tag, 0),
        })
    write_csv(results_abl / "top_overpredicted_tags.csv", over_rows, ["tag", "dimension", "fp_count", "ref_frequency"])

    cross_rows = []
    for (rd, fd), c in cross_dim_pairs.most_common(20):
        cross_rows.append({
            "ref_dimension": rd,
            "fp_dimension": fd,
            "count": c,
        })
    write_csv(results_abl / "cross_dimension_fp_pairs.csv", cross_rows, ["ref_dimension", "fp_dimension", "count"])

    lang_groups = {
        "english_available": [k for k in common_keys if english_available.get(k, False)],
        "french_only": [k for k in common_keys if not english_available.get(k, False)],
    }

    lang_rows = []
    for g, keys in lang_groups.items():
        tm = type_metrics(keys, pred_types, gt_types)
        tagm = tags_micro_metrics(keys, pred_tags, gt_tags)
        mean_tags = sum(len(pred_tags.get(k, set())) for k in keys)
        mean_edges = sum(source_edge_counts.get(k, 0) for k in keys)
        n = len(keys)
        lang_rows.append({
            "group": g,
            "n_paragraphs": n,
            "type_accuracy": round(tm["accuracy"], 4),
            "type_macro_f1": round(tm["macro_f1"], 4),
            "tag_micro_precision": round(tagm["precision"], 4),
            "tag_micro_recall": round(tagm["recall"], 4),
            "tag_micro_f1": round(tagm["f1"], 4),
            "mean_pred_tags_per_para": round(safe_div(mean_tags, n), 4),
            "mean_pred_edges_per_source": round(safe_div(mean_edges, n), 4),
        })
    write_csv(
        results_lang / "language_split_task1_metrics.csv",
        lang_rows,
        [
            "group",
            "n_paragraphs",
            "type_accuracy",
            "type_macro_f1",
            "tag_micro_precision",
            "tag_micro_recall",
            "tag_micro_f1",
            "mean_pred_tags_per_para",
            "mean_pred_edges_per_source",
        ],
    )

    lang_count_rows = []
    all_keys = list(pred_tags.keys())
    n_en = sum(1 for k in all_keys if english_available.get(k, False))
    n_fr = len(all_keys) - n_en
    lang_count_rows.append({"group": "english_available", "count": n_en, "pct": round(100.0 * safe_div(n_en, len(all_keys)), 4)})
    lang_count_rows.append({"group": "french_only", "count": n_fr, "pct": round(100.0 * safe_div(n_fr, len(all_keys)), 4)})
    write_csv(results_lang / "language_split_counts.csv", lang_count_rows, ["group", "count", "pct"])

    traj_rows = []
    for run_id, rp in phase_reports.items():
        if not rp.exists():
            continue
        rep = read_json(rp)
        t1 = rep.get("task1_type", {})
        t1b = rep.get("task1_tags", {})
        t2 = rep.get("task2_relations", {})
        traj_rows.append({
            "run_id": run_id,
            "task1a_accuracy": t1.get("accuracy"),
            "task1a_macro_f1": t1.get("macro_f1"),
            "task1b_micro_precision": t1b.get("micro_precision"),
            "task1b_micro_recall": t1b.get("micro_recall"),
            "task1b_micro_f1": t1b.get("micro_f1"),
            "task1b_macro_f1": t1b.get("macro_f1"),
            "task2_judge_avg_weighted": t2.get("avg_weighted_score"),
            "task2_label_match_rate": t2.get("label_match_rate"),
            "task2_judged": t2.get("total_judged"),
        })
    write_csv(
        results_abl / "phase_trajectory_nondebate.csv",
        traj_rows,
        [
            "run_id",
            "task1a_accuracy",
            "task1a_macro_f1",
            "task1b_micro_precision",
            "task1b_micro_recall",
            "task1b_micro_f1",
            "task1b_macro_f1",
            "task2_judge_avg_weighted",
            "task2_label_match_rate",
            "task2_judged",
        ],
    )

    phase3_rep = read_json(phase_reports["phase3_tagboost_v1"])
    t1 = phase3_rep["task1_type"]
    t1b = phase3_rep["task1_tags"]
    t2 = phase3_rep["task2_relations"]
    write_csv(
        results_abl / "task1_absolute_metrics_phase3_gemini_ref.csv",
        [{
            "task1a_accuracy": t1.get("accuracy"),
            "task1a_macro_f1": t1.get("macro_f1"),
            "task1b_micro_precision": t1b.get("micro_precision"),
            "task1b_micro_recall": t1b.get("micro_recall"),
            "task1b_micro_f1": t1b.get("micro_f1"),
            "task1b_macro_f1": t1b.get("macro_f1"),
            "task2_judge_avg_weighted": t2.get("avg_weighted_score"),
            "task2_label_match_rate": t2.get("label_match_rate"),
        }],
        [
            "task1a_accuracy",
            "task1a_macro_f1",
            "task1b_micro_precision",
            "task1b_micro_recall",
            "task1b_micro_f1",
            "task1b_macro_f1",
            "task2_judge_avg_weighted",
            "task2_label_match_rate",
        ],
    )

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    final_hash = sha256_file(submission_path)
    phase_hash = sha256_file(phase3_submission_path)

    manifest_rows = [
        {"parameter": "generator_name", "value": cfg["models"]["generator_name"], "source": "config.yaml"},
        {"parameter": "embedding_name", "value": cfg["models"]["embedding_name"], "source": "config.yaml"},
        {"parameter": "thinking_budget", "value": cfg["models"].get("thinking_budget"), "source": "config.yaml"},
        {"parameter": "task1_type_mode", "value": cfg["task1"].get("type_mode"), "source": "config.yaml"},
        {"parameter": "task1_tag_mode", "value": cfg["task1"].get("tag_mode"), "source": "config.yaml"},
        {"parameter": "task1_language", "value": cfg["task1"].get("language"), "source": "config.yaml"},
        {"parameter": "k_tag_candidates", "value": cfg["task1"].get("k_tag_candidates"), "source": "config.yaml"},
        {"parameter": "tag_conf_threshold", "value": cfg["task1"].get("tag_conf_threshold"), "source": "config.yaml"},
        {"parameter": "max_tags_per_para", "value": cfg["task1"].get("max_tags_per_para"), "source": "config.yaml"},
        {"parameter": "max_tags_per_dimension", "value": cfg["task1"].get("max_tags_per_dimension"), "source": "config.yaml"},
        {"parameter": "task2_mode", "value": cfg["task2"].get("mode"), "source": "config.yaml"},
        {"parameter": "task2_language", "value": cfg["task2"].get("language"), "source": "config.yaml"},
        {"parameter": "relation_window", "value": cfg["task2"].get("window"), "source": "config.yaml"},
        {"parameter": "relation_k_candidates", "value": cfg["task2"].get("k_candidates"), "source": "config.yaml"},
        {"parameter": "relation_conf_threshold", "value": cfg["task2"].get("rel_conf_threshold"), "source": "config.yaml"},
        {"parameter": "max_edges_per_para", "value": cfg["task2"].get("max_edges_per_para"), "source": "config.yaml"},
        {"parameter": "json_repair_max_retries", "value": cfg["json_repair"].get("max_retries"), "source": "config.yaml"},
        {"parameter": "debate_enabled", "value": cfg.get("debate", {}).get("enabled"), "source": "config.yaml"},
        {"parameter": "language_path", "value": "English-if-available-else-French", "source": "src/pipeline/pipeline.py"},
        {"parameter": "final_submission_sha256", "value": final_hash, "source": "final_submission/LLM-Instruct_predictions.json"},
        {"parameter": "phase3_submission_sha256", "value": phase_hash, "source": "outputs/runs/phase3_tagboost_v1/submission.json"},
        {"parameter": "artifact_hash_match", "value": str(final_hash == phase_hash), "source": "sha256"},
    ]
    write_csv(results_abl / "baseline_manifest.csv", manifest_rows, ["parameter", "value", "source"])

    if l1_repair_stats.exists():
        rs = read_json(l1_repair_stats)
        write_json(results_rep / "nondebate_l1_faststats_repair_stats.json", rs)
        write_csv(
            results_rep / "nondebate_l1_faststats_repair_summary.csv",
            [{
                "pre_repair_invalid_json": rs.get("pre_repair_invalid_json"),
                "post_repair_valid_json": rs.get("post_repair_valid_json"),
                "failed_after_max_retries": rs.get("failed_after_max_retries"),
                "retry_count_histogram": json.dumps(rs.get("retry_count_histogram", {}), ensure_ascii=False),
            }],
            ["pre_repair_invalid_json", "post_repair_valid_json", "failed_after_max_retries", "retry_count_histogram"],
        )

    if l1_trace_summary.exists():
        ts = read_json(l1_trace_summary)
        write_json(results_rep / "nondebate_l1_faststats_trace_summary.json", ts)

    # Check whether held-out test split has gold labels (for blocker transparency).
    test_dir = root / "dataset" / "test-data"
    gold_counts = {
        "docs": 0,
        "paragraphs": 0,
        "non_null_type": 0,
        "non_empty_tags": 0,
        "non_empty_matched_pars": 0,
    }
    for fp in sorted(test_dir.glob("*.json")):
        gold_counts["docs"] += 1
        d = read_json(fp)
        for p in get_paragraphs(d):
            gold_counts["paragraphs"] += 1
            if isinstance(p.get("type"), str) and p.get("type").strip():
                gold_counts["non_null_type"] += 1
            if isinstance(p.get("tags"), list) and len(p.get("tags")) > 0:
                gold_counts["non_empty_tags"] += 1
            mp = p.get("matched_pars") or {}
            if isinstance(mp, dict) and len(mp) > 0:
                gold_counts["non_empty_matched_pars"] += 1

    blockers = {
        "test_split_gold_check": gold_counts,
        "has_task1_gold_on_test": bool(gold_counts["non_null_type"] > 0 or gold_counts["non_empty_tags"] > 0),
        "has_task2_gold_on_test": bool(gold_counts["non_empty_matched_pars"] > 0),
        "note": "If false, absolute dev/test gold P/R/F1 for tags/relations is not directly computable from organizer test files.",
    }
    write_json(results_rel / "gold_availability_blocker.json", blockers)

    summary = {
        "created": {
            "baseline_manifest": str(results_abl / "baseline_manifest.csv"),
            "phase_trajectory": str(results_abl / "phase_trajectory_nondebate.csv"),
            "task1_absolute": str(results_abl / "task1_absolute_metrics_phase3_gemini_ref.csv"),
            "dimension_breakdown": str(results_abl / "dimension_breakdown.csv"),
            "rare_tag_breakdown": str(results_abl / "rare_tag_breakdown.csv"),
            "language_split": str(results_lang / "language_split_task1_metrics.csv"),
            "relation_stats": str(results_rel / "relation_output_stats.json"),
            "repair_stats": str(results_rep / "nondebate_l1_faststats_repair_stats.json"),
            "blocker_check": str(results_rel / "gold_availability_blocker.json"),
        },
        "final_vs_phase3_hash_match": final_hash == phase_hash,
    }
    write_json(root / "analysis" / "task_A1" / "nondebate_analysis_summary.json", summary)

    print("[done] Non-debate analysis artifacts created.")
    for k, v in summary["created"].items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
