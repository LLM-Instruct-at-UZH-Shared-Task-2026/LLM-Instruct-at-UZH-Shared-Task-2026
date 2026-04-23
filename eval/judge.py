#!/usr/bin/env python3
"""
Gemini Judge — Task 1 F1 + Task 2 LLM-as-a-judge
Usage:
    python eval/judge.py [--settings eval/judge_settings.yaml] [--submission outputs/submission.json]
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
import random
import re
import sys
import time
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv

try:
    from google import genai
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'google-genai'. Install with: pip install -r requirements.txt"
    ) from exc

# ─── Helpers ─────────────────────────────────────────────────────────────────

class TeeLogger:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "w", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout

    def write(self, msg: str) -> None:
        self._stdout.write(msg)
        self._file.write(msg)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def fileno(self):
        return self._stdout.fileno()


def load_settings(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_tags(csv_path: str) -> List[Dict]:
    tags = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            tags.append({
                "code": row["CODE"].strip(),
                "category": row["Categories"].strip(),
                "dimension": row["Dimensions"].strip(),
            })
    return tags


def para_text(para: Dict, lang: str = "en") -> str:
    if lang == "en":
        return (para.get("para_en") or para.get("para") or "").strip()
    return (para.get("para") or para.get("para_en") or "").strip()


def extract_json(text: str) -> Any:
    """Extract JSON object/array from LLM response, stripping markdown fences."""
    text = text.strip()
    # Remove markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    # Find first [ or { — try array first, then object
    for start_char, end_char in [('[', ']'), ('{', '}')]:
        idx = text.find(start_char)
        if idx >= 0:
            # Find matching close bracket
            depth = 0
            for i, c in enumerate(text[idx:], idx):
                if c == start_char:
                    depth += 1
                elif c == end_char:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[idx:i+1])
                        except json.JSONDecodeError:
                            break
    # Fallback: try whole string
    try:
        return json.loads(text)
    except Exception:
        return None


def gemini_call(model, prompt: str, retries: int = 3, delay: float = 5.0) -> str:
    """model is a tuple (client, model_name)."""
    client, model_name = model
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            return (resp.text or "").strip()
        except Exception as e:
            if attempt < retries - 1:
                print(f"  [retry {attempt+1}] {e}", file=sys.stderr)
                time.sleep(delay * (attempt + 1))
            else:
                raise


# ─── Task 1a: Type F1 ────────────────────────────────────────────────────────

TYPE_SYSTEM = """You are an expert in UN/UNESCO resolution structure.
Classify each paragraph as exactly one of: "preambular" or "operative".

Preambular paragraphs: introduce context, recall previous acts, express concerns,
  recognise facts. Typical verbs: Recalling, Noting, Recognizing, Considering,
  Bearing in mind, Reaffirming, Welcoming, Aware that, etc.

Operative paragraphs: make decisions, requests, invitations, encourage actions.
  Typical verbs: Decides, Requests, Invites, Urges, Calls upon, Recommends,
  Encourages, Invites, Notes (as a decision), etc.

Return STRICT JSON array — one object per paragraph, in input order:
[{"para_number": <int>, "type": "preambular"|"operative"}, ...]
"""


def judge_type_batch(model, paras: List[Dict], lang: str = "en") -> Dict[int, str]:
    """Call Gemini to classify types. Returns {para_number: type}."""
    items = [{"para_number": p["para_number"],
              "text": para_text(p, lang)[:600]} for p in paras]
    prompt = TYPE_SYSTEM + "\n\nParagraphs:\n" + json.dumps(items, ensure_ascii=False)
    raw = gemini_call(model, prompt)
    parsed = extract_json(raw)
    result = {}
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                num = item.get("para_number")
                t = (item.get("type") or "").strip().lower()
                if num is not None and t in ("preambular", "operative"):
                    result[int(num)] = t
    return result


def compute_type_metrics(pred_map: Dict[int, str], ref_map: Dict[int, str],
                         labels=("preambular", "operative")) -> Dict:
    from collections import Counter
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for num in set(pred_map) | set(ref_map):
        p = pred_map.get(num)
        r = ref_map.get(num)
        if p == r and p is not None:
            tp[p] += 1
        else:
            if p is not None:
                fp[p] += 1
            if r is not None:
                fn[r] += 1

    per_class = {}
    for lbl in labels:
        prec = tp[lbl] / (tp[lbl] + fp[lbl]) if (tp[lbl] + fp[lbl]) else 0.0
        rec  = tp[lbl] / (tp[lbl] + fn[lbl]) if (tp[lbl] + fn[lbl]) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[lbl] = {"precision": round(prec,4), "recall": round(rec,4), "f1": round(f1,4),
                          "support_pred": tp[lbl]+fp[lbl], "support_ref": tp[lbl]+fn[lbl]}

    total_tp = sum(tp.values())
    total = len(set(pred_map) | set(ref_map))
    accuracy = total_tp / total if total else 0.0

    # Macro F1
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class)
    return {"per_class": per_class, "macro_f1": round(macro_f1, 4),
            "accuracy": round(accuracy, 4), "total_paras": total}


# ─── Task 1b: Tag F1 ─────────────────────────────────────────────────────────

TAG_SYSTEM_TMPL = """You are an expert annotator for UN/UNESCO education resolutions.
Assign 0-{max_tags} education dimension tags from the provided taxonomy to each paragraph.
Only assign tags that clearly apply. It is OK to assign 0 tags.

TAG TAXONOMY ({n_tags} tags):
{tag_list}

Return STRICT JSON array — one object per paragraph, in input order:
[{{"para_number": <int>, "tags": ["CODE1", "CODE2", ...]}}, ...]
"""


def build_tag_list(tags: List[Dict]) -> str:
    lines = []
    for t in tags:
        lines.append(f"  {t['code']}: {t['category']} [{t['dimension']}]")
    return "\n".join(lines)


def judge_tags_batch(model, paras: List[Dict], all_tags: List[Dict],
                     max_tags: int = 5, lang: str = "en") -> Dict[int, List[str]]:
    tag_codes = {t["code"] for t in all_tags}
    tag_list_str = build_tag_list(all_tags)
    system = TAG_SYSTEM_TMPL.format(
        max_tags=max_tags, n_tags=len(all_tags), tag_list=tag_list_str
    )
    items = [{"para_number": p["para_number"],
              "text": para_text(p, lang)[:500]} for p in paras]
    prompt = system + "\n\nParagraphs:\n" + json.dumps(items, ensure_ascii=False)
    raw = gemini_call(model, prompt)
    parsed = extract_json(raw)
    result = {}
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                num = item.get("para_number")
                tags = [t for t in (item.get("tags") or []) if t in tag_codes]
                if num is not None:
                    result[int(num)] = tags
    return result


def compute_tag_f1(pred_map: Dict[int, List[str]],
                   ref_map: Dict[int, List[str]]) -> Dict:
    """Micro + macro multi-label F1."""
    micro_tp = micro_fp = micro_fn = 0
    per_para_f1 = []

    for num in set(pred_map) | set(ref_map):
        pred_set = set(pred_map.get(num, []))
        ref_set  = set(ref_map.get(num, []))
        tp = len(pred_set & ref_set)
        fp = len(pred_set - ref_set)
        fn = len(ref_set - pred_set)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        prec = tp / (tp + fp) if (tp + fp) else 1.0  # if pred empty, prec=1
        rec  = tp / (tp + fn) if (tp + fn) else 1.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        per_para_f1.append(f1)

    micro_prec = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_rec  = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1   = 2*micro_prec*micro_rec/(micro_prec+micro_rec) if (micro_prec+micro_rec) else 0.0
    macro_f1   = sum(per_para_f1) / len(per_para_f1) if per_para_f1 else 0.0

    return {
        "micro_precision": round(micro_prec, 4),
        "micro_recall":    round(micro_rec,  4),
        "micro_f1":        round(micro_f1,   4),
        "macro_f1":        round(macro_f1,   4),
        "total_paras":     len(per_para_f1),
    }


# ─── Task 2: LLM-as-a-Judge ──────────────────────────────────────────────────

def build_rel_def_str(settings: Dict) -> str:
    defs = settings["task2_relations"]["relation_definitions"]
    lines = []
    for label, desc in defs.items():
        lines.append(f"- {label.upper()}: {desc.strip()}")
    return "\n".join(lines)


_REPAIR_SCHEMA = (
    '{"scores": {"correctness": INT, "plausibility": INT,'
    ' "specificity": INT, "coherence": INT},\n'
    ' "weighted_score": FLOAT,\n'
    ' "best_label": "supporting|contradictive|complemental|modifying|none",\n'
    ' "label_match": BOOL,\n'
    ' "reasoning": "one sentence"}'
)


def judge_relation(model, para_a: Dict, para_b: Dict, pred_rel: str,
                   settings: Dict, lang: str = "en") -> Optional[Dict]:
    """Score one relation, with retry + self-repair on parse failure."""
    tmpl = settings["task2_relations"]["judge_prompt_template"]
    rel_defs = build_rel_def_str(settings)
    # Use explicit string replacement — avoids KeyError on JSON {braces} in template
    substitutions = {
        "{relation_definitions}": rel_defs,
        "{para_a_num}":           str(para_a["para_number"]),
        "{para_a_text}":          para_text(para_a, lang)[:800],
        "{para_b_num}":           str(para_b["para_number"]),
        "{para_b_text}":          para_text(para_b, lang)[:800],
        "{predicted_relation}":   pred_rel,
    }
    prompt = tmpl
    for placeholder, value in substitutions.items():
        prompt = prompt.replace(placeholder, value)

    raw = gemini_call(model, prompt)
    parsed = extract_json(raw)

    # ── Self-repair loop (up to 2 retries) ────────────────────────────────────
    for _attempt in range(2):
        if isinstance(parsed, dict):
            break
        repair_prompt = (
            f"Your previous response could not be parsed as JSON.\n"
            f"Previous response (first 500 chars):\n{raw[:500]}\n\n"
            f"Output ONLY valid JSON matching EXACTLY this structure "
            f"(INT=1-5, FLOAT=1.0-5.0, BOOL=true/false):\n{_REPAIR_SCHEMA}\n"
            f"No markdown, no explanation, JSON only."
        )
        raw = gemini_call(model, repair_prompt)
        parsed = extract_json(raw)

    if not isinstance(parsed, dict):
        return None

    # Compute weighted score
    weights = settings["task2_relations"]["criteria_weights"]
    scores = parsed.get("scores", {})
    ws = sum(scores.get(k, 3) * v for k, v in weights.items())
    parsed["weighted_score"] = round(ws, 3)
    return parsed


def run_task2_judge(model, docs: List[Dict], settings: Dict,
                    rng: random.Random,
                    forced_cases: Optional[List[Dict]] = None) -> Dict:
    """Run Task 2 judge.

    If *forced_cases* is given (list of {doc_id, para_a_num, para_b_num, pred_rel}),
    only those cases are judged (used for --rerun-errors mode).
    """
    max_per_doc = settings["sampling"]["max_relations_per_doc"]
    lang = "en"
    all_scores = []
    label_match_count = 0
    total_judged = 0
    per_rel_type: Dict[str, List[float]] = {}
    best_label_dist: Dict[str, int] = {}
    failed_cases: List[Dict] = []

    # Build doc lookup
    doc_map: Dict[str, Dict] = {d.get("TEXT_ID", ""): d for d in docs}

    # Decide which cases to run
    if forced_cases is not None:
        work: List[Tuple] = []
        for fc in forced_cases:
            doc = doc_map.get(fc["doc_id"])
            if doc is None:
                print(f"  [rerun] WARNING: doc {fc['doc_id']} not found, skipping")
                continue
            para_by_num = {p["para_number"]: p for p in doc.get("body", {}).get("paragraphs", [])}
            pa = para_by_num.get(fc["para_a_num"])
            pb = para_by_num.get(fc["para_b_num"])
            if pa and pb:
                work.append((doc, pa, pb, fc["pred_rel"]))
            else:
                print(f"  [rerun] WARNING: para {fc['para_a_num']}→{fc['para_b_num']} not found in {fc['doc_id']}")
    else:
        work = []
        for doc in docs:
            paras = doc.get("body", {}).get("paragraphs", [])
            para_by_num = {p["para_number"]: p for p in paras}
            relations: List[Tuple] = []
            for p in paras:
                for b_num_str, rel in (p.get("matched_pars") or {}).items():
                    rel_str = rel if isinstance(rel, str) else (rel[0] if rel else "")
                    b_num = int(b_num_str)
                    if b_num in para_by_num and rel_str:
                        relations.append((doc, p, para_by_num[b_num], rel_str))
            if relations:
                sample = relations if max_per_doc <= 0 else rng.sample(
                    relations, min(max_per_doc, len(relations)))
                work.extend(sample)

    for doc, para_a, para_b, pred_rel in work:
        print(f"  [rel judge] doc={doc.get('TEXT_ID','')} "
              f"para {para_a['para_number']}→{para_b['para_number']} [{pred_rel}]",
              end=" ", flush=True)
        result = judge_relation(model, para_a, para_b, pred_rel, settings, lang)
        if result is None:
            print("PARSE ERROR")
            failed_cases.append({
                "doc_id":     doc.get("TEXT_ID", ""),
                "para_a_num": para_a["para_number"],
                "para_b_num": para_b["para_number"],
                "pred_rel":   pred_rel,
            })
            continue
        ws = result.get("weighted_score", 3.0)
        all_scores.append(ws)
        match = result.get("label_match", False)
        if match:
            label_match_count += 1
        total_judged += 1
        per_rel_type.setdefault(pred_rel, []).append(ws)
        best = result.get("best_label", "")
        best_label_dist[best] = best_label_dist.get(best, 0) + 1
        print(f"score={ws:.2f} match={match}")

    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    label_acc = label_match_count / total_judged if total_judged else 0.0

    return {
        "total_judged": total_judged,
        "parse_errors": len(failed_cases),
        "avg_weighted_score": round(avg_score, 3),
        "avg_score_pct": round((avg_score - 1) / 4 * 100, 1),  # 1-5 → 0-100%
        "label_match_rate": round(label_acc, 4),
        "per_relation_type": {k: {"n": len(v), "avg": round(sum(v)/len(v), 3)}
                              for k, v in per_rel_type.items()},
        "best_label_distribution": best_label_dist,
        "failed_cases": failed_cases,   # persisted for --rerun-errors
        "all_scores": all_scores,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def _make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _merge_t2_results(base: Dict, patch: Dict) -> Dict:
    """Merge rerun-errors patch into base Task 2 results."""
    # If base has no raw scores, approximate from avg+total
    base_scores = list(base.get("all_scores", []))
    if not base_scores and base.get("total_judged", 0) > 0:
        base_scores = [base["avg_weighted_score"]] * base["total_judged"]

    merged_scores = base_scores + list(patch.get("all_scores", []))
    merged_per_type = dict(base.get("per_relation_type", {}))
    for rtype, info in patch.get("per_relation_type", {}).items():
        if rtype in merged_per_type:
            n_old = merged_per_type[rtype]["n"]
            avg_old = merged_per_type[rtype]["avg"]
            n_new = info["n"]
            avg_new = info["avg"]
            merged_per_type[rtype] = {
                "n": n_old + n_new,
                "avg": round((avg_old * n_old + avg_new * n_new) / (n_old + n_new), 3),
            }
        else:
            merged_per_type[rtype] = info
    merged_best = dict(base.get("best_label_distribution", {}))
    for k, v in patch.get("best_label_distribution", {}).items():
        merged_best[k] = merged_best.get(k, 0) + v
    total = len(merged_scores)
    avg = sum(merged_scores) / total if total else 0.0
    base_matched = round(base.get("label_match_rate", 0) * base.get("total_judged", 0))
    patch_matched = round(patch.get("label_match_rate", 0) * patch.get("total_judged", 0))
    merged_judged = base.get("total_judged", 0) + patch.get("total_judged", 0)
    still_failed = list(patch.get("failed_cases", []))
    return {
        "total_judged": merged_judged,
        "parse_errors": len(still_failed),
        "avg_weighted_score": round(avg, 3),
        "avg_score_pct": round((avg - 1) / 4 * 100, 1) if total else 0.0,
        "label_match_rate": round((base_matched + patch_matched) / merged_judged, 4) if merged_judged else 0.0,
        "per_relation_type": merged_per_type,
        "best_label_distribution": merged_best,
        "failed_cases": still_failed,
        "all_scores": merged_scores,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings",      default="eval/judge_settings.yaml")
    ap.add_argument("--submission",    default=None)
    ap.add_argument("--task1-only",    action="store_true")
    ap.add_argument("--task2-only",    action="store_true")
    ap.add_argument("--run-id",        default=None,
                    help="Explicit run ID (default: auto timestamp)")
    ap.add_argument("--rerun-errors",  default=None, metavar="RUN_ID",
                    help="Re-run only failed cases from a previous run")
    ap.add_argument("--save-ground-truth", action="store_true",
                    help="Save Gemini reference labels after Task 1 evaluation")
    ap.add_argument("--load-ground-truth", default=None, metavar="RUN_ID",
                    help="Load pre-saved Gemini reference labels from a previous run "
                         "(skips Gemini API calls for Task 1)")
    args = ap.parse_args()

    load_dotenv()
    api_key    = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL")
    if not api_key:
        sys.exit("ERROR: GOOGLE_API_KEY not set in .env")
    if not model_name:
        sys.exit("ERROR: GEMINI_MODEL not set in .env")

    cfg = load_settings(args.settings)
    runs_dir     = Path(cfg["paths"].get("runs_dir", "eval/runs"))
    submission_path = args.submission or cfg["paths"]["submission"]

    # ── Run ID & output directory ─────────────────────────────────────────────
    run_id = args.run_id or args.rerun_errors or _make_run_id()
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_dir / "report.json"
    table_path  = run_dir / "summary.txt"
    log_path    = run_dir / "run.log"

    # Tee all output to run.log
    tee = TeeLogger(log_path)
    sys.stdout = tee
    sys.stderr = tee

    print(f"[judge] Run ID: {run_id}")
    print(f"[judge] Run dir: {run_dir}")
    print(f"[judge] Model: {model_name}")
    print(f"[judge] Submission: {submission_path}")

    # Load submission
    with open(submission_path) as f:
        all_docs = json.load(f)
    if not isinstance(all_docs, list):
        all_docs = [all_docs]

    # Sampling
    rng = random.Random(cfg["sampling"]["seed"])
    n_eval = cfg["sampling"]["eval_docs"]
    if n_eval > 0 and n_eval < len(all_docs):
        docs = rng.sample(all_docs, n_eval)
        print(f"[judge] Sampling {n_eval}/{len(all_docs)} docs")
    else:
        docs = all_docs
        print(f"[judge] Evaluating all {len(docs)} docs")

    # Load tags
    tags = load_tags(cfg["paths"]["tags_csv"])
    print(f"[judge] Tag taxonomy: {len(tags)} tags")

    report = {
        "run_id":     run_id,
        "model":      model_name,
        "submission": submission_path,
        "n_docs":     len(docs),
    }

    def _save_report() -> None:
        """Write current report to disk (incremental checkpoint)."""
        out = dict(report)
        if "task2_relations" in out:
            t2 = dict(out["task2_relations"])
            t2.pop("all_scores", None)
            out["task2_relations"] = t2
        with open(report_path, "w", encoding="utf-8") as _f:
            json.dump(out, _f, indent=2, ensure_ascii=False)
        print(f"  [checkpoint] Report saved \u2192 {report_path}", flush=True)

    client = genai.Client(api_key=api_key)
    model  = (client, model_name)  # tuple passed to gemini_call()

    # ── Rerun-errors mode ────────────────────────────────────────────────
    if args.rerun_errors:
        prev_report_path = runs_dir / args.rerun_errors / "report.json"
        if not prev_report_path.exists():
            sys.exit(f"ERROR: No report at {prev_report_path}")
        with open(prev_report_path) as f:
            prev_report = json.load(f)
        failed_cases = prev_report.get("task2_relations", {}).get("failed_cases", [])
        if not failed_cases:
            print("[rerun] No failed cases in previous report. Nothing to do.")
            tee.close(); sys.stdout = tee._stdout; sys.stderr = tee._stdout
            return
        print(f"[rerun] Re-running {len(failed_cases)} failed cases from '{args.rerun_errors}'")
        patch  = run_task2_judge(model, docs, cfg, rng, forced_cases=failed_cases)
        merged = _merge_t2_results(prev_report["task2_relations"], patch)
        prev_report["task2_relations"] = merged
        prev_report["rerun_id"] = run_id
        # write to the ORIGINAL run_id directory (overwrite its report)
        orig_report = runs_dir / args.rerun_errors / "report.json"
        out = dict(prev_report)
        t2 = dict(out.get("task2_relations", {}))
        t2.pop("all_scores", None)
        out["task2_relations"] = t2
        with open(orig_report, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n[rerun] Updated report \u2192 {orig_report}")
        print(f"  Relations judged : {merged['total_judged']}")
        print(f"  Parse errors left: {merged['parse_errors']}")
        print(f"  Avg score        : {merged['avg_weighted_score']:.3f}/5.0  ({merged['avg_score_pct']:.1f}%)")
        tee.close(); sys.stdout = tee._stdout; sys.stderr = tee._stdout
        return

    # ── Task 1 ────────────────────────────────────────────────────────────────
    do_t1 = not args.task2_only
    if do_t1:
        print("\n" + "="*60)
        print("TASK 1a — Type Classification (Gemini as reference)")
        print("="*60)

        type_cfg = cfg["task1_type"]
        batch_sz = type_cfg["batch_size"]
        lang     = type_cfg["lang"]

        pred_type_map: Dict[tuple, str] = {}
        ref_type_map:  Dict[tuple, str] = {}

        # ── Optionally load pre-saved Gemini type labels ──────────────────────
        if args.load_ground_truth:
            gt_dir = runs_dir / args.load_ground_truth
            gt_types_path = gt_dir / "ground_truth_types.json"
            if not gt_types_path.exists():
                sys.exit(f"ERROR: {gt_types_path} not found. "
                         f"Run a previous judge with --save-ground-truth first.")
            with open(gt_types_path, encoding="utf-8") as _f:
                _raw = json.load(_f)
            for _k, _v in _raw.items():
                _tid, _pstr = _k.rsplit("|||", 1)
                ref_type_map[(_tid, int(_pstr))] = _v
            print(f"[judge] Loaded {len(ref_type_map)} ref type labels "
                  f"from {gt_types_path}", flush=True)
            # Still collect pred_type_map from submission
            for doc_i, doc in enumerate(docs):
                paras = doc.get("body", {}).get("paragraphs", [])
                tid = doc.get("TEXT_ID", str(doc_i))
                for p in paras:
                    t = (p.get("type") or "").strip().lower()
                    if t in type_cfg["valid_labels"]:
                        pred_type_map[(tid, p["para_number"])] = t
        else:
            for doc_i, doc in enumerate(docs):
                paras = doc.get("body", {}).get("paragraphs", [])
                if not paras:
                    continue
                tid = doc.get("TEXT_ID", str(doc_i))
                print(f"  [{doc_i+1}/{len(docs)}] {tid} "
                      f"({len(paras)} paras)", flush=True)

                # Collect pred types (keyed by (TEXT_ID, para_number))
                for p in paras:
                    key = (tid, p["para_number"])
                    t = (p.get("type") or "").strip().lower()
                    if t in type_cfg["valid_labels"]:
                        pred_type_map[key] = t

                # Get Gemini reference in batches
                for i in range(0, len(paras), batch_sz):
                    batch = paras[i:i+batch_sz]
                    ref_batch = judge_type_batch(model, batch, lang)
                    for pnum, t in ref_batch.items():
                        ref_type_map[(tid, pnum)] = t

        # ── Optionally save Gemini type labels ────────────────────────────────
        if args.save_ground_truth and not args.load_ground_truth:
            _gt_types = {f"{k[0]}|||{k[1]}": v for k, v in ref_type_map.items()}
            _gt_path = run_dir / "ground_truth_types.json"
            with open(_gt_path, "w", encoding="utf-8") as _f:
                json.dump(_gt_types, _f, indent=2, ensure_ascii=False)
            print(f"[judge] Saved {len(_gt_types)} ref type labels → {_gt_path}")

        type_metrics = compute_type_metrics(pred_type_map, ref_type_map)
        report["task1_type"] = type_metrics
        print(f"\n  Accuracy : {type_metrics['accuracy']:.4f}")
        print(f"  Macro F1 : {type_metrics['macro_f1']:.4f}")
        for lbl, m in type_metrics["per_class"].items():
            print(f"  {lbl:12s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
        _save_report()

        # ── Task 1b Tags ──────────────────────────────────────────────────────
        if cfg["task1_tags"]["enabled"]:
            print("\n" + "="*60)
            print("TASK 1b — Tag Assignment (Gemini as reference)")
            print("="*60)

            tag_cfg   = cfg["task1_tags"]
            tbatch_sz = tag_cfg["batch_size"]
            max_tags  = tag_cfg["max_tags_per_para"]

            pred_tag_map: Dict[tuple, List[str]] = {}
            ref_tag_map:  Dict[tuple, List[str]] = {}

            # ── Optionally load pre-saved Gemini tag labels ───────────────────
            if args.load_ground_truth:
                gt_dir = runs_dir / args.load_ground_truth
                gt_tags_path = gt_dir / "ground_truth_tags.json"
                if not gt_tags_path.exists():
                    sys.exit(f"ERROR: {gt_tags_path} not found. "
                             f"Run a previous judge with --save-ground-truth first.")
                with open(gt_tags_path, encoding="utf-8") as _f:
                    _raw = json.load(_f)
                for _k, _v in _raw.items():
                    _tid, _pstr = _k.rsplit("|||", 1)
                    ref_tag_map[(_tid, int(_pstr))] = _v
                print(f"[judge] Loaded {len(ref_tag_map)} ref tag labels "
                      f"from {gt_tags_path}", flush=True)
                # Still collect pred_tag_map from submission
                for doc_i, doc in enumerate(docs):
                    paras = doc.get("body", {}).get("paragraphs", [])
                    tid = doc.get("TEXT_ID", str(doc_i))
                    for p in paras:
                        pred_tag_map[(tid, p["para_number"])] = list(p.get("tags") or [])
            else:
                for doc_i, doc in enumerate(docs):
                    paras = doc.get("body", {}).get("paragraphs", [])
                    if not paras:
                        continue
                    tid = doc.get("TEXT_ID", str(doc_i))
                    print(f"  [{doc_i+1}/{len(docs)}] {tid} "
                          f"({len(paras)} paras)", flush=True)

                    for p in paras:
                        pred_tag_map[(tid, p["para_number"])] = list(p.get("tags") or [])

                    for i in range(0, len(paras), tbatch_sz):
                        batch = paras[i:i+tbatch_sz]
                        ref_batch = judge_tags_batch(model, batch, tags, max_tags, lang)
                        for pnum, tlist in ref_batch.items():
                            ref_tag_map[(tid, pnum)] = tlist

            # ── Optionally save Gemini tag labels ─────────────────────────────
            if args.save_ground_truth and not args.load_ground_truth:
                _gt_tags = {f"{k[0]}|||{k[1]}": v for k, v in ref_tag_map.items()}
                _gt_path = run_dir / "ground_truth_tags.json"
                with open(_gt_path, "w", encoding="utf-8") as _f:
                    json.dump(_gt_tags, _f, indent=2, ensure_ascii=False)
                print(f"[judge] Saved {len(_gt_tags)} ref tag labels → {_gt_path}")

            tag_metrics = compute_tag_f1(pred_tag_map, ref_tag_map)
            report["task1_tags"] = tag_metrics
            warn = tag_cfg["f1_warn_threshold"]
            flag = "⚠️" if tag_metrics["micro_f1"] < warn else "✅"
            print(f"\n  Micro F1  : {tag_metrics['micro_f1']:.4f} {flag}")
            print(f"  Micro P   : {tag_metrics['micro_precision']:.4f}")
            print(f"  Micro R   : {tag_metrics['micro_recall']:.4f}")
            print(f"  Macro F1  : {tag_metrics['macro_f1']:.4f}")
            _save_report()

    # ── Task 2 ────────────────────────────────────────────────────────────────
    do_t2 = not args.task1_only
    if do_t2 and cfg["task2_relations"]["enabled"]:
        print("\n" + "="*60)
        print("TASK 2 — Relation Judge (LLM-as-a-judge)")
        print("="*60)
        t2_results = run_task2_judge(model, docs, cfg, rng)
        report["task2_relations"] = t2_results
        print(f"\n  Relations judged      : {t2_results['total_judged']}")
        print(f"  Avg weighted score    : {t2_results['avg_weighted_score']:.3f} / 5.0")
        print(f"  Score (%)             : {t2_results['avg_score_pct']:.1f}%")
        print(f"  Label match rate      : {t2_results['label_match_rate']:.4f}")
        print(f"  Per-type scores:")
        for rtype, info in t2_results["per_relation_type"].items():
            print(f"    {rtype:14s}  n={info['n']:4d}  avg={info['avg']:.3f}")

    # ── Final save ────────────────────────────────────────────────────────────
    _save_report()
    print(f"\n[judge] Report saved → {report_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    lines = [
        "=" * 60,
        f"GEMINI JUDGE SUMMARY  (run_id: {run_id})",
        f"Model: {model_name}",
        f"Submission: {submission_path}   Docs: {len(docs)}",
        "=" * 60,
    ]
    if "task1_type" in report:
        m = report["task1_type"]
        lines += [
            "TASK 1a — Type Classification",
            f"  Accuracy  : {m['accuracy']:.4f}",
            f"  Macro F1  : {m['macro_f1']:.4f}",
        ]
        for lbl, v in m["per_class"].items():
            lines.append(f"  {lbl:12s}  P={v['precision']:.3f}  R={v['recall']:.3f}  F1={v['f1']:.3f}")
    if "task1_tags" in report:
        m = report["task1_tags"]
        lines += [
            "",
            "TASK 1b — Tag Assignment",
            f"  Micro F1  : {m['micro_f1']:.4f}",
            f"  Micro P   : {m['micro_precision']:.4f}",
            f"  Micro R   : {m['micro_recall']:.4f}",
            f"  Macro F1  : {m['macro_f1']:.4f}",
        ]
    if "task2_relations" in report:
        m = report["task2_relations"]
        lines += [
            "",
            "TASK 2 — Relations (LLM-as-a-judge)",
            f"  Judged    : {m['total_judged']}  (parse errors: {m.get('parse_errors', 0)})",
            f"  Score     : {m['avg_weighted_score']:.3f}/5.0  ({m['avg_score_pct']:.1f}%)",
            f"  Label acc : {m['label_match_rate']:.4f}",
        ]
        for rtype, info in m["per_relation_type"].items():
            lines.append(f"  {rtype:14s}  n={info['n']:4d}  avg={info['avg']:.3f}")
    lines.append("=" * 60)

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"[judge] Table saved → {table_path}")

    tee.close()
    sys.stdout = tee._stdout
    sys.stderr = tee._stdout


if __name__ == "__main__":
    main()
