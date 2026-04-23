#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

VALID_TYPES = ("preambular", "operative")
VALID_RELS = ("supporting", "contradictive", "complemental", "modifying")


@dataclass
class ParaData:
    ptype: Optional[str]
    tags: Set[str]
    rels: Dict[int, str]  # target_para -> relation


def _normalize_type(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v if v in VALID_TYPES else None


def _normalize_rel(value) -> Optional[str]:
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v if v in VALID_RELS else None


def _get_paragraphs(doc: dict) -> List[dict]:
    body = doc.get("body") or {}
    paras = body.get("paragraphs") or body.get("paras") or []
    return paras if isinstance(paras, list) else []


def _doc_to_para_map(doc: dict) -> Dict[int, ParaData]:
    out: Dict[int, ParaData] = {}
    for p in _get_paragraphs(doc):
        num = p.get("para_number")
        if not isinstance(num, int):
            continue
        ptype = _normalize_type(p.get("type"))

        tags_raw = p.get("tags") or []
        tags: Set[str] = set()
        if isinstance(tags_raw, list):
            for t in tags_raw:
                if isinstance(t, str) and t.strip():
                    tags.add(t.strip())

        rels_raw = p.get("matched_pars") or p.get("matched_paras") or {}
        rels: Dict[int, str] = {}
        if isinstance(rels_raw, dict):
            for k, v in rels_raw.items():
                try:
                    b = int(k)
                except Exception:
                    continue
                rel = _normalize_rel(v)
                if rel is not None:
                    rels[b] = rel

        out[num] = ParaData(ptype=ptype, tags=tags, rels=rels)
    return out


def load_gpt_pseudolabels(pseudo_dir: Path) -> Dict[str, Dict[int, ParaData]]:
    refs: Dict[str, Dict[int, ParaData]] = {}
    files = sorted(pseudo_dir.glob("*.json"))
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            doc = json.load(f)
        text_id = doc.get("TEXT_ID")
        if not isinstance(text_id, str) or not text_id:
            continue
        refs[text_id] = _doc_to_para_map(doc)
    return refs


def load_submission(path: Path) -> Dict[str, Dict[int, ParaData]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    docs = data if isinstance(data, list) else [data]

    out: Dict[str, Dict[int, ParaData]] = {}
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        text_id = doc.get("TEXT_ID")
        if not isinstance(text_id, str) or not text_id:
            continue
        out[text_id] = _doc_to_para_map(doc)
    return out


def evaluate_type(pred_docs: Dict[str, Dict[int, ParaData]], ref_docs: Dict[str, Dict[int, ParaData]]) -> Dict:
    tp = {"preambular": 0, "operative": 0}
    fp = {"preambular": 0, "operative": 0}
    fn = {"preambular": 0, "operative": 0}

    all_units = 0
    correct = 0

    common_docs = sorted(set(pred_docs) & set(ref_docs))
    for d in common_docs:
        pmap = pred_docs[d]
        rmap = ref_docs[d]
        for para in sorted(set(pmap) | set(rmap)):
            p = pmap.get(para).ptype if para in pmap else None
            r = rmap.get(para).ptype if para in rmap else None
            if p == r and p is not None:
                tp[p] += 1
                correct += 1
            else:
                if p is not None:
                    fp[p] += 1
                if r is not None:
                    fn[r] += 1
            all_units += 1

    per = {}
    for lbl in VALID_TYPES:
        p_denom = tp[lbl] + fp[lbl]
        r_denom = tp[lbl] + fn[lbl]
        prec = tp[lbl] / p_denom if p_denom else 0.0
        rec = tp[lbl] / r_denom if r_denom else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        per[lbl] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support_pred": p_denom,
            "support_ref": r_denom,
        }

    macro_f1 = sum(per[l]["f1"] for l in VALID_TYPES) / len(VALID_TYPES)
    acc = correct / all_units if all_units else 0.0
    return {
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(acc, 4),
        "total_paras": all_units,
        "per_class": per,
    }


def evaluate_tags(pred_docs: Dict[str, Dict[int, ParaData]], ref_docs: Dict[str, Dict[int, ParaData]]) -> Dict:
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    per_para_f1: List[float] = []

    common_docs = sorted(set(pred_docs) & set(ref_docs))
    for d in common_docs:
        pmap = pred_docs[d]
        rmap = ref_docs[d]
        for para in sorted(set(pmap) | set(rmap)):
            pset = pmap.get(para).tags if para in pmap else set()
            rset = rmap.get(para).tags if para in rmap else set()
            tp = len(pset & rset)
            fp = len(pset - rset)
            fn = len(rset - pset)

            micro_tp += tp
            micro_fp += fp
            micro_fn += fn

            prec = tp / (tp + fp) if (tp + fp) else 1.0
            rec = tp / (tp + fn) if (tp + fn) else 1.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            per_para_f1.append(f1)

    micro_prec = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_rec = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) else 0.0
    macro_f1 = sum(per_para_f1) / len(per_para_f1) if per_para_f1 else 0.0

    return {
        "micro_precision": round(micro_prec, 4),
        "micro_recall": round(micro_rec, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "total_paras": len(per_para_f1),
    }


def evaluate_relations(pred_docs: Dict[str, Dict[int, ParaData]], ref_docs: Dict[str, Dict[int, ParaData]]) -> Dict:
    pred_label_map: Dict[Tuple[str, int, int], str] = {}
    ref_label_map: Dict[Tuple[str, int, int], str] = {}

    common_docs = sorted(set(pred_docs) & set(ref_docs))
    for d in common_docs:
        for a, pdata in pred_docs[d].items():
            for b, rel in pdata.rels.items():
                pred_label_map[(d, a, b)] = rel
        for a, rdata in ref_docs[d].items():
            for b, rel in rdata.rels.items():
                ref_label_map[(d, a, b)] = rel

    union_keys = set(pred_label_map) | set(ref_label_map)

    # No reference relation edges in pseudo-labels -> Task2 cannot be evaluated.
    if len(ref_label_map) == 0:
        return {
            "available": False,
            "reason": "No reference matched_pars in GPT pseudo-labels",
            "label_accuracy_on_union_edges": None,
            "micro_precision": None,
            "micro_recall": None,
            "micro_f1": None,
            "macro_f1": None,
            "pred_edges": len(pred_label_map),
            "ref_edges": 0,
            "union_edges": len(union_keys),
            "per_relation": {},
        }
    correct = 0
    for k in union_keys:
        if pred_label_map.get(k) == ref_label_map.get(k):
            correct += 1
    acc = correct / len(union_keys) if union_keys else 0.0

    pred_triples = {(k[0], k[1], k[2], v) for k, v in pred_label_map.items()}
    ref_triples = {(k[0], k[1], k[2], v) for k, v in ref_label_map.items()}
    tp = len(pred_triples & ref_triples)
    fp = len(pred_triples - ref_triples)
    fn = len(ref_triples - pred_triples)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    per_rel = {}
    for rel in VALID_RELS:
        pset = {x for x in pred_triples if x[3] == rel}
        rset = {x for x in ref_triples if x[3] == rel}
        rtp = len(pset & rset)
        rfp = len(pset - rset)
        rfn = len(rset - pset)
        rprec = rtp / (rtp + rfp) if (rtp + rfp) else 0.0
        rrec = rtp / (rtp + rfn) if (rtp + rfn) else 0.0
        rf1 = (2 * rprec * rrec / (rprec + rrec)) if (rprec + rrec) else 0.0
        per_rel[rel] = {
            "precision": round(rprec, 4),
            "recall": round(rrec, 4),
            "f1": round(rf1, 4),
            "support_pred": len(pset),
            "support_ref": len(rset),
        }

    macro_f1 = sum(per_rel[r]["f1"] for r in VALID_RELS) / len(VALID_RELS)

    return {
        "available": True,
        "label_accuracy_on_union_edges": round(acc, 4),
        "micro_precision": round(prec, 4),
        "micro_recall": round(rec, 4),
        "micro_f1": round(f1, 4),
        "macro_f1": round(macro_f1, 4),
        "pred_edges": len(pred_label_map),
        "ref_edges": len(ref_label_map),
        "union_edges": len(union_keys),
        "per_relation": per_rel,
    }


def evaluate_submission(sub_path: Path, ref_docs: Dict[str, Dict[int, ParaData]]) -> Dict:
    pred_docs = load_submission(sub_path)
    common_docs = sorted(set(pred_docs) & set(ref_docs))

    pred_only = sorted(set(pred_docs) - set(ref_docs))
    ref_only = sorted(set(ref_docs) - set(pred_docs))

    # Evaluate only intersection for fair comparison.
    pred_common = {k: pred_docs[k] for k in common_docs}
    ref_common = {k: ref_docs[k] for k in common_docs}

    t1a = evaluate_type(pred_common, ref_common)
    t1b = evaluate_tags(pred_common, ref_common)
    t2 = evaluate_relations(pred_common, ref_common)

    return {
        "submission": str(sub_path),
        "n_docs_pred": len(pred_docs),
        "n_docs_ref": len(ref_docs),
        "n_docs_common": len(common_docs),
        "n_docs_pred_only": len(pred_only),
        "n_docs_ref_only": len(ref_only),
        "task1_type": t1a,
        "task1_tags": t1b,
        "task2_relations": t2,
    }


def discover_submissions(root: Path) -> List[Path]:
    files = sorted(Path(p) for p in glob.glob(str(root / "outputs" / "runs" / "*" / "submission.json")))
    main_sub = root / "outputs" / "submission.json"
    if main_sub.exists():
        files.append(main_sub)
    dedup = []
    seen = set()
    for f in files:
        if f not in seen:
            seen.add(f)
            dedup.append(f)
    return dedup


def write_outputs(results: List[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "gpt_pseudolabel_eval.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    csv_path = out_dir / "gpt_pseudolabel_eval_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "submission",
            "docs_common",
            "task1a_accuracy",
            "task1a_macro_f1",
            "task1b_micro_f1",
            "task1b_macro_f1",
            "task2_micro_f1",
            "task2_macro_f1",
            "task2_label_acc_union",
            "task2_pred_edges",
            "task2_ref_edges",
        ])
        for r in results:
            t2 = r["task2_relations"]
            w.writerow([
                r["submission"],
                r["n_docs_common"],
                r["task1_type"]["accuracy"],
                r["task1_type"]["macro_f1"],
                r["task1_tags"]["micro_f1"],
                r["task1_tags"]["macro_f1"],
                t2["micro_f1"],
                t2["macro_f1"],
                t2["label_accuracy_on_union_edges"],
                t2["pred_edges"],
                t2["ref_edges"],
            ])

    md_path = out_dir / "gpt_pseudolabel_eval_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# GPT Pseudo-Label Evaluation (All Submission Versions)\n\n")
        f.write("| Submission | Docs(common) | T1a Acc | T1a Macro-F1 | T1b Micro-F1 | T1b Macro-F1 | T2 Micro-F1 | T2 Macro-F1 | T2 Label-Acc |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in results:
            t2 = r["task2_relations"]
            t2_micro = "N/A" if t2["micro_f1"] is None else f"{t2['micro_f1']:.4f}"
            t2_macro = "N/A" if t2["macro_f1"] is None else f"{t2['macro_f1']:.4f}"
            t2_acc = "N/A" if t2["label_accuracy_on_union_edges"] is None else f"{t2['label_accuracy_on_union_edges']:.4f}"
            f.write(
                f"| `{r['submission']}` | {r['n_docs_common']} | "
                f"{r['task1_type']['accuracy']:.4f} | {r['task1_type']['macro_f1']:.4f} | "
                f"{r['task1_tags']['micro_f1']:.4f} | {r['task1_tags']['macro_f1']:.4f} | "
                f"{t2_micro} | {t2_macro} | "
                f"{t2_acc} |\n"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate submissions against GPT pseudo-label folder.")
    ap.add_argument("--root", default=".", help="Project root")
    ap.add_argument("--pseudo-dir", default="pseudo-label", help="Directory containing GPT pseudo-label JSON files")
    ap.add_argument("--out-dir", default="eval/gpt_pseudolabel", help="Output directory")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    pseudo_dir = (root / args.pseudo_dir).resolve()
    out_dir = (root / args.out_dir).resolve()

    ref_docs = load_gpt_pseudolabels(pseudo_dir)
    submissions = discover_submissions(root)

    if not submissions:
        raise SystemExit("No submission.json found under outputs/runs/*/submission.json or outputs/submission.json")

    results = []
    for sp in submissions:
        results.append(evaluate_submission(sp, ref_docs))

    # Sort by Task1b micro-F1 descending, then Task2 micro-F1.
    results.sort(key=lambda x: (x["task1_tags"]["micro_f1"], x["task2_relations"]["micro_f1"]), reverse=True)

    write_outputs(results, out_dir)

    print(f"[done] Evaluated {len(results)} submissions against {len(ref_docs)} GPT pseudo-label docs")
    print(f"[done] Outputs:")
    print(f"  - {out_dir / 'gpt_pseudolabel_eval.json'}")
    print(f"  - {out_dir / 'gpt_pseudolabel_eval_summary.csv'}")
    print(f"  - {out_dir / 'gpt_pseudolabel_eval_summary.md'}")


if __name__ == "__main__":
    main()
